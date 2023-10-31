#define PRINT_C

#include "Manifold.slang"
#include "Light.slang"

Texture2D<uint4> gVertices;
RWTexture2D<float4> gOutput;

struct SpecularChain {
    PackedVertex start;
    float3 startNg;
    PackedVertex firstVertex;
    float3 startLocalDirIn;
    uint vertexCount;
    float3 startThroughput;
	uint refractMask;
};

float3 ConnectToLight(const PathVertex vertex, const float3 localDirIn, const PathVertex lightVertex, const float lightW) {
    if (gMinManifoldVertices > 0)
        return 0;

	float3 dirOut = lightVertex.mPosition;
	float dist = POS_INFINITY;
	float G = 1;
	if (lightVertex.mIsSurface) {
		dirOut -= vertex.mPosition;
		dist = length(dirOut);
		dirOut /= dist;
		G = -dot(dirOut, lightVertex.mShadingNormal) / sqr(dist);
        if (G < 0)
            return 0;
	}

    if (Occluded(OffsetRayOrigin(vertex, dirOut), dirOut, dist))
        return 0;

    const float3 localDirOut = vertex.ToLocal(dirOut);
    return (lightVertex.mMaterial.Emission() * G * lightW) *
		EvalBrdf(vertex.mMaterial, localDirIn, localDirOut).mReflectance *
		vertex.GetShadingNormalCorrection(localDirIn, localDirOut);

}

float3 ConnectToLightManifold(const SpecularChain chain, const PathVertex lightVertex, const uint refractMask, const float lightW) {
    if (chain.vertexCount < gMinManifoldVertices || chain.vertexCount > gMaxManifoldVertices)
        return 0;

    PathVertex specularVertex = UnpackVertex(chain.firstVertex);
	if (!ManifoldSolve(chain.start.mPosition, chain.startNg, specularVertex, chain.vertexCount, refractMask, lightVertex.mPosition, lightVertex.mIsSurface))
		return 0;

    float3 throughput = chain.startThroughput;
    float3 dirIn;
	// connect start vertex to specularVertex
    {
        float3 dirOut = specularVertex.mPosition - chain.start.mPosition;
        const float dist = length(dirOut);
        dirOut /= dist;
		if (Occluded(OffsetRayOrigin(chain.start.mPosition, chain.startNg, dirOut), dirOut, dist))
            return 0;

        const PathVertex startVertex = UnpackVertex(chain.start);
        const float3 localDirOut = startVertex.ToLocal(dirOut);
        throughput *= EvalBrdf(startVertex.mMaterial, chain.startLocalDirIn, localDirOut).mReflectance;
        throughput *= startVertex.GetShadingNormalCorrection(chain.startLocalDirIn, localDirOut);
        if (!any(throughput > 0))
            return 0;
        dirIn = -dirOut;
	}

    // connect specular chain, update specularVertex
    if (gMaxManifoldVertices > 1) {
		for (uint i = 0; i < chain.vertexCount - 1; i++) {
			const float3 localDirIn  = specularVertex.ToLocal(dirIn);

			float3 dirOut;
			if ((chain.refractMask & (1 << i)) == 0)
				dirOut = normalize(-dirIn + 2 * localDirIn.z * specularVertex.mShadingNormal);
			else {
				const float eta = specularVertex.mMaterial.Eta();
				const float etaRcp = localDirIn.z < 0 ? eta : 1/eta;
				const float refractedCosOut2 = 1 - (1 - (localDirIn.z * localDirIn.z)) * (etaRcp * etaRcp);
				dirOut = normalize(-dirIn * etaRcp + (abs(localDirIn.z) * etaRcp - sqrt(refractedCosOut2)) * specularVertex.mShadingNormal * sign(localDirIn.z));
			}
			const float3 localDirOut = specularVertex.ToLocal(dirOut);

			DisneyBSDF bsdf = DisneyBSDF(specularVertex.mMaterial, localDirOut);
			throughput *= bsdf.Eval(specularVertex.mMaterial, DisneyBSDF::EvalFlags::eFresnelOnly).mReflectance;
			throughput *= specularVertex.GetShadingNormalCorrection(localDirIn, localDirOut);

			specularVertex = TraceRay(MakeRay(OffsetRayOrigin(specularVertex, dirOut), dirOut));
            dirIn = -dirOut;
        }
    }

    // connect specularVertex to lightVertex
	{
		float3 dirOut = lightVertex.mPosition;
		float dist = POS_INFINITY;
		float G = 1;
        if (lightVertex.mIsSurface) {
            dirOut -= specularVertex.mPosition;
			dist = length(dirOut);
			dirOut /= dist;
			G = -dot(dirOut, lightVertex.mShadingNormal) / sqr(dist);
			if (G < 0)
				return 0;
		}

        if (Occluded(OffsetRayOrigin(specularVertex, dirOut), dirOut, dist))
            return 0;

		const float3 localDirIn  = specularVertex.ToLocal(dirIn);
        const float3 localDirOut = specularVertex.ToLocal(dirOut);
        DisneyBSDF bsdf = DisneyBSDF(specularVertex.mMaterial, localDirIn, localDirOut);
        throughput *= bsdf.Eval(specularVertex.mMaterial, DisneyBSDF::EvalFlags::eFresnelOnly).mReflectance;
        throughput *= specularVertex.GetShadingNormalCorrection(localDirIn, localDirOut);
        return throughput * (lightVertex.mMaterial.Emission() * G * lightW);
	}
}

float LightTargetPdf(const PathVertex vertex, const float3 localDirIn, const PathVertex lightVertex) {
    float3 toLight = lightVertex.mPosition;
    float f = Luminance(lightVertex.mMaterial.Emission());
    if (lightVertex.mIsSurface) {
        toLight -= vertex.mPosition;
        const float d2 = dot(toLight, toLight);
        toLight /= sqrt(d2);
        f *= max(0, -dot(toLight, lightVertex.mShadingNormal)) / d2;
    }
    if (!(f > 0))
        return 0;

    const float3 localDirOut = vertex.ToLocal(toLight);
    const bool reflection = localDirIn.z * localDirOut.z > 0;
    if (!reflection)
        f *= vertex.mMaterial.Transmission();

    if (!(f > 0))
        return 0;

	if (IsDiffuse(vertex))
        f *= abs(localDirOut.z);
    else {
		#ifdef gConstraintTargetPdf
        if (reflection) {
			// reflection constraint
            f *= dot(localDirOut, normalize(-localDirIn + 2 * float3(0, 0, localDirIn.z)));
        } else {
			// refraction constraint
			float eta = vertex.mMaterial.Eta();
			float etaRcp = 1 / eta;
			if (localDirIn.z < 0) {
				const float tmp = eta;
				eta = etaRcp;
				etaRcp = tmp;
			}
			const float refractedCosOut2 = 1 - (1 - (localDirIn.z * localDirIn.z)) * (etaRcp * etaRcp);
			const float refractedCosIn2 = 1 - (1 - (localDirOut.z * localDirOut.z)) * (etaRcp * etaRcp);
			if (refractedCosOut2 > 0)
				f *= max(0, dot(localDirOut, normalize(-localDirIn * etaRcp + (abs(localDirIn.z) * etaRcp - sqrt(refractedCosOut2)) * float3(0, 0,  sign(localDirIn.z)))));
			else if (refractedCosIn2 > 0)
				f *= max(0, dot(localDirIn, normalize(-localDirOut * eta    + (abs(localDirOut.z) * eta   - sqrt(refractedCosIn2))  * float3(0, 0, -sign(localDirOut.z)))));
			else
				return 0;
		}
		#endif
    }
    return f;
}

PathVertex SampleLightVertex(const PathVertex vertex, const float3 localDirIn, inout RandomSampler rng, out float W) {
    float lightPdfA;
    PathVertex lightVertex = SampleEmission(rng.NextFloat(), lightPdfA);
    if (lightPdfA > 0)
        W = LightTargetPdf(vertex, localDirIn, lightVertex) / lightPdfA;
    else
        W = 0;

    for (uint i = 0; i < gLightSampleCandidates; i++) {
        const PathVertex c = SampleEmission(rng.NextFloat(), lightPdfA);
        if (!(lightPdfA > 0))
            continue;
        const float wi = LightTargetPdf(vertex, localDirIn, c) / lightPdfA;
		W += wi;
        if (W * rng.NextFloat().x <= wi)
            lightVertex = c;
	}

    if (W > 0)
        W /= (1+gLightSampleCandidates) * LightTargetPdf(vertex, localDirIn, lightVertex);
    else
        W = 0;
    return lightVertex;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleCameraPaths(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    const PackedVertex v = reinterpret<PackedVertex>(gVertices[id]);
    if (!v.mIsSurface)
        return;

    if (IsDebugPixel)
        gOutput[id].rgb += float3(1, 0, 1);

    PathVertex vertex = UnpackVertex(v);
	float3 localDirIn  = vertex.ToLocal(normalize(gCameraPosition - vertex.mPosition));
    float3 throughput = 1;
    SpecularChain chain;
    chain.vertexCount = 0;
    chain.startThroughput = 0;

    RandomSampler rng = RandomSampler(gRandomSeed, id);

	for (uint i = 0; i < gMaxBounces; i++) {
		if (GetContinuationProbability(vertex) == 0)
			return;

        const bool diffuse = IsDiffuse(vertex);
        if (gMaxManifoldVertices > 0) {
            if (diffuse) {
                chain.start = PackVertex(vertex);
                chain.startNg = vertex.mGeometryNormal;
				chain.startLocalDirIn = localDirIn;
				chain.startThroughput = throughput;
				chain.vertexCount = 0;
				chain.refractMask = 0;
			} else if (any(chain.startThroughput > 0)) {
				chain.vertexCount++;
				if (chain.vertexCount == 1)
					chain.firstVertex = PackVertex(vertex);
			}
        }

		// sample and connect to lightVertex
        float lightW;
        const PathVertex lightVertex = SampleLightVertex(vertex, localDirIn, rng, lightW);
        if (lightW > 0 && any(lightVertex.mMaterial.Emission() > 0)) {
			if (diffuse) {
                // connect vertex to lightVertex
                gOutput[id].rgb += throughput * ConnectToLight(vertex, localDirIn, lightVertex, lightW);
            } else if (gMaxManifoldVertices > 0 && chain.vertexCount > 0) {
                // connect specular chain to lightVertex
                uint refractMask = chain.refractMask;
                if (localDirIn.z * dot(lightVertex.mPosition - vertex.mPosition, vertex.mShadingNormal) < 0)
                    refractMask |= (1 << (chain.vertexCount - 1));
                gOutput[id].rgb += ConnectToLightManifold(chain, lightVertex, refractMask, lightW);
			}
        }

		// sample next vertex
		DirectionSampleRecord r = SampleBrdf(vertex.mMaterial, rng.NextFloat(), localDirIn);
		if (r.mFwdPdfW <= 0 || !any(r.mReflectance > 0))
            return;
        if (r.mEta > 0 && chain.vertexCount > 0)
            chain.refractMask |= (1 << (chain.vertexCount - 1));
        throughput *= r.mReflectance * vertex.GetShadingNormalCorrection(localDirIn, r.mDirection);
        throughput /= r.mFwdPdfW;
		r.mDirection = vertex.ToWorld(r.mDirection);
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, r.mDirection), r.mDirection));
        if (!vertex.mIsSurface)
            return;
        localDirIn = vertex.ToLocal(-r.mDirection);
	}
}

[shader("compute")]
[numthreads(8, 8, 1)]
void CopyDebugImage(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gOutputSize))
        return;

    float4 c = gOutput[index.xy];
    float t = (float)gDebugImage[index.y * gOutputSize.x + index.x];
    if (t > 0)
        c.rgb = ViridisQuintic((1 + t) / float(gManifoldSolverIterations));
    else
        c.rgb *= 0.5;

    gOutput[index.xy] = c;
}
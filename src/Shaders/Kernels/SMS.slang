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

float3 ConnectToLight(const PathVertex vertex, const float3 localDirIn, const PathVertex lightVertex, const float lightPdfA) {
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
	return (lightVertex.mMaterial.Emission() * G / lightPdfA) *
		EvalBrdf(vertex.mMaterial, localDirIn, localDirOut).mReflectance *
		vertex.GetShadingNormalCorrection(localDirIn, localDirOut);

}

float3 ConnectToLightManifold(const SpecularChain chain, const PathVertex lightVertex, const float lightPdfA) {
    if (chain.vertexCount < gMinManifoldVertices || chain.vertexCount > gMaxManifoldVertices)
        return 0;

    PathVertex specularVertex = UnpackVertex(chain.firstVertex);

    if (!ManifoldSolve(chain.start.mPosition, chain.startNg, specularVertex, lightVertex.mPosition, lightVertex.mIsSurface))
        return 0;

    float3 throughput = chain.startThroughput;
	// connect start vertex to first specular vertex
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
	}

    float3 dirIn = normalize(chain.start.mPosition - specularVertex.mPosition);
	float3 localDirIn = specularVertex.ToLocal(dirIn);
	// evaluate throughput along chain
    /*
    for (uint i = 0; i < chain.vertexCount - 1; i++) {
        const bool reflecting = (chain.refractMask & (1 << i)) == 0;

        float3 dirOut = normalize(reflecting ? reflect() : refract());
        const float3 localDirOut = specularVertex.ToLocal(dirOut);

        DisneyBSDF bsdf = DisneyBSDF(specularVertex.mMaterial, localDirOut);
        throughput *= bsdf.Eval(specularVertex.mMaterial, DisneyBSDF::EvalFlags::eFresnelOnly).mReflectance;
        throughput *= specularVertex.GetShadingNormalCorrection(localDirIn, localDirOut);

        specularVertex = TraceRay(MakeRay(OffsetRayOrigin(specularVertex, dirOut), dirOut));
        dirIn = -dirOut;
	}
	*/

	// connect last specular vertex to lightVertex
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

        const float3 localDirOut = specularVertex.ToLocal(dirOut);
        DisneyBSDF bsdf = DisneyBSDF(specularVertex.mMaterial, localDirIn, localDirOut);
        throughput *= bsdf.Eval(specularVertex.mMaterial, DisneyBSDF::EvalFlags::eFresnelOnly).mReflectance;
        throughput *= specularVertex.GetShadingNormalCorrection(localDirIn, localDirOut);
        return throughput * (lightVertex.mMaterial.Emission() * G / lightPdfA);
	}
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
    chain.startThroughput = 0;

    RandomSampler rng = RandomSampler(gRandomSeed, id);

    for (uint i = 0; i < gMaxBounces; i++) {
		if (GetContinuationProbability(vertex) == 0)
			return;

        const bool diffuse = IsDiffuse(vertex);
        if (gManifoldSampling) {
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
		float lightPdfA;
        const PathVertex lightVertex = SampleEmission(rng.NextFloat(), lightPdfA);
        if (lightPdfA > 0 && any(lightVertex.mMaterial.Emission() > 0)) {
			if (diffuse) {
				// connect vertex to lightVertex
                gOutput[id].rgb += throughput * ConnectToLight(vertex, localDirIn, lightVertex, lightPdfA);
            } else if (gManifoldSampling && chain.vertexCount > 0) {
				// connect specular chain to lightVertex
				gOutput[id].rgb += ConnectToLightManifold(chain, lightVertex, lightPdfA);
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
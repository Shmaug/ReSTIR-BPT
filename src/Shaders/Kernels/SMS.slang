#include "Manifold.slang"
#include "Light.slang"

Texture2D<uint4> gVertices;
RWTexture2D<float4> gOutput;

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleCameraPaths(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    const PackedVertex v = reinterpret<PackedVertex>(gVertices[id]);
    if (!v.mIsSurface)
        return;

	// sample initial vertices
	// vertices[0] is loaded from the visibility buffer (must be diffuse)
	// vertices[1] is sampled via BSDF sampling (can be diffuse or specular)
	// vertices[2] is sampled via light sampling

    float bsdfPdfW;
	float lightPdfA;
    PathVertex vertices[3];

    vertices[0] = UnpackVertex(v);

    if (GetContinuationProbability(vertices[0]) == 0)
        return;

	{
		RandomSampler rng = RandomSampler(gRandomSeed, id);

        const float3 localDirIn  = vertices[0].ToLocal(normalize(gCameraPosition - vertices[0].mPosition));

		{
			// sample vertices[1] with bsdf sampling
			DirectionSampleRecord r = SampleBrdf(vertices[0].mMaterial, rng.NextFloat(), localDirIn);
			if (r.mFwdPdfW <= 0 || !any(r.mReflectance > 0))
                return;
            bsdfPdfW = r.mFwdPdfW;
			r.mDirection = vertices[0].ToWorld(r.mDirection);
			vertices[1] = TraceRay(MakeRay(OffsetRayOrigin(vertices[0], r.mDirection), r.mDirection));

			// sample v2 with light sampling
			vertices[2] = SampleEmission(rng.NextFloat(), lightPdfA);
			if (lightPdfA <= 0 || !any(vertices[2].mMaterial.Emission() > 0))
				return;
		}

        // add v0 -> v2 path
        /*
		{
            float3 dirOut = vertices[2].mPosition;
            float dist = POS_INFINITY;
            float G = 1;
            if (vertices[2].mIsSurface) {
                dirOut -= vertices[0].mPosition;
                dist = length(dirOut);
                dirOut /= dist;
				G = -dot(dirOut, vertices[2].mShadingNormal) / sqr(dist);
            }
            if (G > 0) {
                const float3 localDirOut = vertices[0].ToLocal(dirOut);
                if (!Occluded(OffsetRayOrigin(vertices[0], dirOut), dirOut, dist))
                    gOutput[id].rgb += (vertices[2].mMaterial.Emission() * G / lightPdfA) * EvalBrdf(vertices[0].mMaterial, localDirIn, localDirOut).mReflectance * vertices[0].GetShadingNormalCorrection(localDirIn, localDirOut);
            }
        }
		*/
	}

    float G = 1;
    bool isManifoldPath = false;
    if (!IsDiffuse(vertices[1]) && gManifoldSampling) {
		if (!ManifoldSolve(vertices))
            return;
        G = ManifoldG(vertices);
        isManifoldPath = true;
	}

    // eval path contribution
	{
		float3 f[2];
		float3 le;

		float3 dirIn = normalize(vertices[0].mPosition - gCameraPosition);
		float G = 1;
		for (uint i = 0; i < 2; i++) {
			float3 dirOut = vertices[i + 1].mPosition;
            float dist = POS_INFINITY;
			if (vertices[i + 1].mIsSurface) {
				dirOut -= vertices[i].mPosition;
                dist = length(dirOut);
                dirOut /= dist;
                if (i == 1) {
                    const float cosLight = -dot(dirOut, vertices[i + 1].mShadingNormal);
                    if (cosLight < 0)
						return;
					if (!isManifoldPath)
						G = cosLight / sqr(dist);
                }
			}

			const float3 localDirIn  = vertices[i].ToLocal(-dirIn);
			const float3 localDirOut = vertices[i].ToLocal( dirOut);
            const DisneyBSDF bsdf = DisneyBSDF(vertices[i].mMaterial, localDirIn, localDirOut);
            f[i] = bsdf.Eval(vertices[i].mMaterial, (i == 1 && isManifoldPath) ? DisneyBSDF::EvalFlags::eFresnelOnly : DisneyBSDF::EvalFlags::eNone).mReflectance;
            f[i] *= vertices[i].GetShadingNormalCorrection(localDirIn, localDirOut);

			if (!any(f[i] > 0))
                return;

            if (Occluded(OffsetRayOrigin(vertices[i], dirOut), dirOut, dist)) {
                DebugPrintf("vertices[%u] -> vertices[%u] occluded\n", i, i+1);
				return;
			}
			dirIn = dirOut;
		}

		gOutput[id].rgb += f[0] * (f[1] / bsdfPdfW) * (vertices[2].mMaterial.Emission() * G / lightPdfA);
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
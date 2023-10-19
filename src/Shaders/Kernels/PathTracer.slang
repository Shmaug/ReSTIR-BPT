#include "Intersection.slang"
#include "Random.slang"
#include "Light.slang"
#include "BRDF.slang"

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
    uint gMaxBounces;
    uint gRandomSeed;
    float3 gCameraPosition;
};

float Mis(float a, float b) {
    a = a * a;
    return a / (a + b * b);
}

float3 SamplePath(PathVertex vertex, float3 dirIn, RandomSampler rngIn) {
    RandomSampler rng = rngIn;
    float3 color = 0;
    float3 throughput = 1;
    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const bool diffuse = IsDiffuse(vertex);
        const float contProb = diffuse ? GetContinuationProbability(vertex) : 1;
		const float3 localDirIn = vertex.ToLocal(-dirIn);
        if (gSampleLights && diffuse) {
            // sample direct light
            float lightPdf;
            const PathVertex l = SampleEmission(rng.NextFloat(), lightPdf);
            float3 le = l.mMaterial.Emission();
            float3 toLight;
            float dist, cosLight, G;
            if (l.mIsSurface) {
                toLight = l.mPosition - vertex.mPosition;
                dist = length(toLight);
                toLight /= dist;
                cosLight = max(0, -dot(toLight, l.mShadingNormal));
                G = cosLight / sqr(dist);
                lightPdf /= G;
                if (cosLight <= 0)
                    le = 0;
            } else {
                toLight = l.mPosition;
                dist = POS_INFINITY;
                cosLight = 1;
                G = 1;
            }
            if (any(le > 0) && lightPdf > 0) {
                const float3 localDirOut = vertex.ToLocal(toLight);
                ReflectanceEvalRecord r = EvalBrdf(vertex.mMaterial, localDirIn, localDirOut);
                le *= r.mReflectance * vertex.GetShadingNormalCorrection(localDirIn, localDirOut);
                if (any(le > 0)) {
					if (!Occluded(vertex, toLight, dist)) {
						color += throughput * le * Mis(lightPdf, contProb*r.mFwdPdfW) / lightPdf;
					}
                }
			}
		}
        if (contProb == 0 || rng.NextFloat().x > contProb)
            break;

        // sample brdf
        DirectionSampleRecord r = SampleBrdf(vertex.mMaterial, rng.NextFloat(), localDirIn);
        throughput *= r.mReflectance * vertex.GetShadingNormalCorrection(localDirIn, r.mDirection);
        r.mFwdPdfW *= contProb;
        throughput /= r.mFwdPdfW;
        if (all(throughput <= 0)) break;

        dirIn = vertex.ToWorld(r.mDirection);

        float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, lightPdf, dist);

        float G = 1;
        float3 le = vertex.mMaterial.Emission();
        if (vertex.mIsSurface) {
            const float cosIn = -dot(vertex.mShadingNormal, dirIn);
            lightPdf *= sqr(dist) / abs(cosIn);
            if (cosIn <= 0)
				le = 0;
        }

        if (any(le > 0)) {
            const float w = gSampleLights ? Mis(r.mFwdPdfW, lightPdf) : 1;
            color += throughput * le * w;
        }

    }
    return color;
}

Texture2D<uint4> gVertices;
RWTexture2D<float4> gOutput;

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleCameraPaths(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    const PackedVertex v = reinterpret<PackedVertex>(gVertices[id]);
    if (v.mIsSurface) {
        const PathVertex vertex = UnpackVertex(v);
        const float3 dir = normalize(v.mPosition - gCameraPosition);
        gOutput[id] += float4(SamplePath(vertex, dir, RandomSampler(gRandomSeed, id)), 0);
    }
}

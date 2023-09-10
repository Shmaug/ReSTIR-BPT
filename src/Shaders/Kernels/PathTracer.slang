#include "Intersection.slang"
#include "Random.slang"
#include "Light.slang"
#include "BRDF.slang"

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif

float Mis(float a, float b) {
    a = a * a;
    return a / (a + b * b);
}

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
    uint gMaxBounces;
    uint gRandomSeed;
    float3 gCameraPosition;
};

float3 SamplePath(PathVertex vertex, float3 dirIn, RandomSampler rngIn) {
    RandomSampler rng = rngIn;
    float3 color = 0;
    float3 throughput = 1;
    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const bool diffuse = IsDiffuse(vertex);
        const float contProb = diffuse ? GetContinuationProbability(vertex) : 1;
        if (gSampleLights && diffuse) {
            // sample direct light
            const DirectLightSampleRecord l = SampleDirectLight(vertex, rng.NextFloat());
            float3 le = l.mRadiance;
			if (any(le > 0)) {
                le *= EvalBrdf(vertex, -dirIn, l.mToLight);
                if (any(le > 0)) {
					if (!Occluded(vertex, l.mToLight, l.mDistance)) {
						color += throughput * le * Mis(l.mPdfW, contProb*EvalBrdfPdfW(vertex, -dirIn, l.mToLight)) / l.mPdfW;
					}
                }
			}
		}
        if (contProb == 0 || rng.NextFloat().x > contProb)
            break;

        // sample brdf
        DirectionSampleRecord r;
        SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection);
        const float brdfPdfW = contProb * EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
		throughput /= brdfPdfW;
        if (all(throughput <= 0)) break;

        dirIn = r.mDirection;

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
            const float w = gSampleLights ? Mis(brdfPdfW, lightPdf) : 1;
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
    sHeatmapPixel = id.y * gOutputSize.x + id.x;

    PathVertex vertex = UnpackVertex(reinterpret<PackedVertex>(gVertices[id]));

    if (vertex.mIsSurface) {
        gOutput[id] += float4(SamplePath(vertex, normalize(vertex.mPosition - gCameraPosition), RandomSampler(gRandomSeed, id)), 0);
    }
}

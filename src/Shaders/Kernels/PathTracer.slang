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
    float pdfW = 1;
    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const bool diffuse = IsDiffuse(vertex);
        if (gSampleLights && diffuse) {
            // sample direct light
			DirectLightSampleRecord l = SampleDirectLight(vertex, rng.NextFloat());
			const float3 le = l.mRadiance * EvalBrdf(vertex, -dirIn, l.mToLight);
			if (any(le > 0)) {
				if (!Occluded(vertex, l.mToLight, l.mDistance)) {
					color += throughput * le * Mis(l.mPdfW, EvalBrdfPdfW(vertex, -dirIn, l.mToLight)) / (pdfW * l.mPdfW);
				}
			}
		}

        // sample brdf
        DirectionSampleRecord r;
        SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection);
        const float brdfPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
        pdfW *= brdfPdfW;
        if (all(throughput <= 0)) break;

        float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, r.mDirection), r.mDirection), true, lightPdf, dist);
        dirIn = r.mDirection;

        float G = 1;
        float3 le = 0;
        if (vertex.mIsSurface) {
            const float cosIn = dot(vertex.mShadingNormal, dirIn);
            G = abs(cosIn) / sqr(dist);
            lightPdf /= G;
            le = cosIn < 0 ? vertex.mMaterial.Emission() : 0;
        } else {
            le = EvalBackground(dirIn, lightPdf);
        }

        if (any(le > 0)) {
            const float w = gSampleLights ? Mis(brdfPdfW, lightPdf) : 1;
            color += throughput * le * w / pdfW;
        }

        const float p = GetContinuationProbability(vertex);
        if (p == 0 || rng.NextFloat().x > p)
            break;
        pdfW *= p;
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
        gOutput[id] += float4(SamplePath(vertex, normalize(vertex.mPosition - gCameraPosition), RandomSampler(gRandomSeed, id)), 1);
    }
}

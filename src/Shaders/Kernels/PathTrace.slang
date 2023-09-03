#include "Intersection.slang"
#include "Light.slang"
#include "Random.slang"
#include "BRDF.slang"

RWTexture2D<float4> gOutput;
RWTexture2D<float4> gOutputPositions;
RWTexture2D<float4> gOutputAlbedo;

[[vk::push_constant]]
cbuffer PushConstants {
	uint2 gOutputSize;
    uint gRandomSeed;
	uint gMaxBounces;
};

uniform float4x4 gCameraToWorld;
uniform float4x4 gInverseProjection;
RayDesc GetCameraRay(uint2 id) {
    RayDesc ray;
    ray.TMin = 0;
    ray.TMax = POS_INFINITY;
    ray.Origin    = mul(gCameraToWorld, float4(0, 0, 0, 1)).xyz;
    ray.Direction = mul(gInverseProjection, float4((float2(id + 0.5) / float2(gOutputSize)) * 2 - 1, 0, 1)).xyz;
    ray.Direction = normalize(mul(gCameraToWorld, float4(ray.Direction, 0)).xyz);
    return ray;
}

float Mis(float a, float b) {
    float a2 = a * a;
	return a2 / (a2 + b*b);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void RenderMegakernel(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gOutputSize)) return;

    RayDesc ray = GetCameraRay(index.xy);

    IntersectionResult isect;
    ShadingData sd;
    float3 radiance = 0;
    if (!TraceRay(ray, true, sd, isect)) {
        gOutputPositions[index.xy] = float4(float3(POS_INFINITY), 0);
		gOutputAlbedo[index.xy] = 1;

		float tmp;
        radiance = EvalBackground(ray.Direction, tmp);
    } else {
		const bool frontFace = dot(ray.Direction, sd.GetShadingNormal()) < 0;
		gOutputPositions[index.xy] = float4(sd.mPosition, asfloat(frontFace ? sd.mPackedShadingNormal : PackNormal(-sd.GetShadingNormal())));

		GpuMaterial m = LoadMaterial(sd);
		gOutputAlbedo[index.xy] = float4(m.BaseColor(), 1);

        if (frontFace)
            radiance = m.Emission();

		RandomSampler rng = RandomSampler(gRandomSeed, index.xy);

		float3 throughput = 1;
		for (uint bounce = 0; bounce < gMaxBounces; bounce++) {
			#ifdef SAMPLE_LIGHTS
			// sample direct light
            DirectLightSampleRecord l = SampleDirectLight(sd, rng.NextFloat());
            l.mRadiance *= EvalBrdf(sd, m, -ray.Direction, l.mToLight);
            if (any(l.mRadiance > 0)) {
				if (!Occluded(sd, l.mToLight, l.mDistance)) {
					radiance += throughput * (l.mRadiance / l.mPdfW) * Mis(l.mPdfW, EvalBrdfPdfW(sd, m, -ray.Direction, l.mToLight));
				}
			}
			#endif

			// sample brdf
			DirectionSampleRecord r;
			SampleBrdf(sd, m, rng.NextFloat(), -ray.Direction, r);
			float brdfPdfW = EvalBrdfPdfW(sd, m, -ray.Direction, r.mDirection);
			throughput *= EvalBrdf(sd, m, -ray.Direction, r.mDirection) / brdfPdfW;
			if (all(throughput <= 0)) break;

			if (!TraceRay(MakeRay(OffsetRayOrigin(sd, r.mDirection), r.mDirection), true, sd, isect)) {
				const float3 le = EvalBackground(r.mDirection, isect.mLightPdf);
				#ifdef SAMPLE_LIGHTS
				radiance += throughput * le * Mis(brdfPdfW, isect.mLightPdf);
				#else
				radiance += throughput * le;
				#endif
				break;
			}

			m = LoadMaterial(sd);

			const float3 le = m.Emission();
			if (any(le > 0)) {
				isect.mLightPdf *= sqr(isect.mDistance) / abs(dot(ray.Direction, sd.GetGeometryNormal()));
				#ifdef SAMPLE_LIGHTS
				radiance += throughput * le * Mis(brdfPdfW, isect.mLightPdf);
				#else
				radiance += throughput * le;
				#endif
			}

			const float p = GetContinuationProbability(m);
			if (rng.NextFloat().x > p)
				break;
			throughput /= p;
		}
    }
    gOutput[index.xy] = float4(radiance, 1);
}

struct PathState {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;
    float3 mThroughput;
    uint mPackedDirIn;
    RandomSampler mRng;
    uint4 pad;
};
RWStructuredBuffer<PathState> gPathState;

void StorePathState(uint2 id, const uint instanceIndex, const uint primitiveIndex, const ShadingData sd, const float3 throughput, const float3 dirIn, const RandomSampler rng) {
    PathState s;
    s.mLocalPosition = TransformPoint(gScene.mInstanceInverseTransforms[instanceIndex], sd.mPosition);
    s.mInstancePrimitiveIndex = instanceIndex&0xFFFF | (primitiveIndex << 16);
    s.mThroughput = throughput;
    s.mPackedDirIn = PackNormal(dirIn);
    s.mRng = rng;
    gPathState[id.y * gOutputSize.x + id.x] = s;
}
bool LoadPathState(uint2 id, inout ShadingData sd, inout float3 throughput, inout float3 dirIn, inout RandomSampler rng) {
    const PathState s = gPathState[id.y * gOutputSize.x + id.x];
    if (any(s.mThroughput > 0)) {
    	sd = ShadingData::Load(s.mInstancePrimitiveIndex & 0xFFFF, s.mLocalPosition, s.mInstancePrimitiveIndex >> 16);
		throughput = s.mThroughput;
		dirIn = UnpackNormal(s.mPackedDirIn);
        rng = s.mRng;
        return true;
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void RenderVisibility(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gOutputSize)) return;

    RayDesc ray = GetCameraRay(index.xy);

    IntersectionResult isect;
    ShadingData sd;
    float3 radiance = 0;
    bool hit = TraceRay(ray, true, sd, isect);
    if (!hit) {
        gOutputPositions[index.xy] = float4(float3(POS_INFINITY), 0);
		gOutputAlbedo[index.xy] = 1;

		float tmp;
        radiance = EvalBackground(ray.Direction, tmp);
    } else {
		const bool frontFace = dot(ray.Direction, sd.GetShadingNormal()) < 0;
		gOutputPositions[index.xy] = float4(sd.mPosition, asfloat(frontFace ? sd.mPackedShadingNormal : PackNormal(-sd.GetShadingNormal())));

		GpuMaterial m = LoadMaterial(sd);
		gOutputAlbedo[index.xy] = float4(m.BaseColor(), 1);

        if (frontFace)
            radiance = m.Emission();
    }
    gOutput[index.xy] = float4(radiance, 1);
    StorePathState(index.xy, isect.mInstanceIndex, isect.mPrimitiveIndex, sd, hit?1:0, -ray.Direction, RandomSampler(gRandomSeed, index.xy));

}
[shader("compute")]
[numthreads(8, 8, 1)]
void RenderBounce(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gOutputSize)) return;

    float3 dirIn;
    RandomSampler rng;
    float3 throughput;
    ShadingData sd;
    if (!LoadPathState(index.xy, sd, throughput, dirIn, rng))
		return;

    GpuMaterial m = LoadMaterial(sd);

	#ifdef SAMPLE_LIGHTS
	// sample direct light
	DirectLightSampleRecord l = SampleDirectLight(sd, rng.NextFloat());
	l.mRadiance *= EvalBrdf(sd, m, dirIn, l.mToLight);
	if (any(l.mRadiance > 0)) {
        if (!Occluded(sd, l.mToLight, l.mDistance)) {
            gOutput[index.xy].rgb += throughput * (l.mRadiance / l.mPdfW) * Mis(l.mPdfW, EvalBrdfPdfW(sd, m, dirIn, l.mToLight));
		}
	}
	#endif

	// sample brdf
	DirectionSampleRecord r;
	SampleBrdf(sd, m, rng.NextFloat(), dirIn, r);
	float brdfPdfW = EvalBrdfPdfW(sd, m, dirIn, r.mDirection);
    throughput *= EvalBrdf(sd, m, dirIn, r.mDirection) / brdfPdfW;

	IntersectionResult isect;
    if (any(throughput > 0)) {
		if (!TraceRay(MakeRay(OffsetRayOrigin(sd, r.mDirection), r.mDirection), true, sd, isect)) {
			const float3 le = EvalBackground(r.mDirection, isect.mLightPdf);
			#ifdef SAMPLE_LIGHTS
			gOutput[index.xy].rgb += throughput * le * Mis(brdfPdfW, isect.mLightPdf);
			#else
			gOutput[index.xy].rgb += throughput * le;
			#endif
			throughput = 0;
		} else {
			m = LoadMaterial(sd);

			const float3 le = m.Emission();
			if (any(le > 0)) {
				isect.mLightPdf *= sqr(isect.mDistance) / abs(dot(dirIn, sd.GetGeometryNormal()));
				#ifdef SAMPLE_LIGHTS
				gOutput[index.xy].rgb += throughput * le * Mis(brdfPdfW, isect.mLightPdf);
				#else
				gOutput[index.xy].rgb += throughput * le;
				#endif
			}

			const float p = GetContinuationProbability(m);
			if (rng.NextFloat().x > p)
				throughput = 0;
			else
				throughput /= p;
		}
    }

    StorePathState(index.xy, isect.mInstanceIndex, isect.mPrimitiveIndex, sd, throughput, -r.mDirection, rng);
}
#include "Intersection.slang"
#include "Light.slang"
#include "Random.slang"

RWTexture2D<float4> gOutput;
RWTexture2D<float4> gOutputPositions;
RWTexture2D<float4> gOutputAlbedo;
uniform uint2 gOutputSize;

uniform uint gRandomSeed;

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

float3 SampleRadiance(const ShadingData sd, const GpuMaterial m, const float3 dirIn, inout RandomSampler rng) {
	EmissionSampleRecord l = SampleEmission(rng.NextFloat());

    float3 toLight;
    float dist;
    float3 le = 0;
    float cosAtLight;
    if (l.mInstanceIndex == INVALID_INSTANCE) {
        toLight = l.mShadingData.mPosition;
        dist = POS_INFINITY;
        float tmp;
        le = EvalBackground(toLight, tmp);
        cosAtLight = 1;
    } else {
		toLight = l.mShadingData.mPosition - sd.mPosition;
		dist = length(toLight);
		toLight /= dist;
		le = LoadMaterial(l.mShadingData).Emission();
		cosAtLight = max(0, -dot(toLight, l.mShadingData.GetShadingNormal()));
    }
    if (cosAtLight > 0) {
		const float3 ns = sd.GetShadingNormal();
		const float cosToLight = max(0, dot(toLight, ns) * sign(-dot(ns, dirIn)));
		const float3 brdf = (m.BaseColor() / M_PI) * cosToLight;
		if (cosToLight > 0) {
			if (!Occluded(sd, toLight, dist)) {
				const float G = cosAtLight / sqr(dist);
				return brdf * le * G / l.mPdf;
			}
		}
    }

    return 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gOutputSize)) return;

    const RayDesc ray = GetCameraRay(index.xy);

    IntersectionResult isect;
	float3 color = 0;
    if (TraceRay(ray, true, isect)) {
        const bool frontFace = dot(ray.Direction, isect.mShadingData.GetShadingNormal()) < 0;
        gOutputPositions[index.xy] = float4(isect.mShadingData.mPosition, asfloat(frontFace ? isect.mShadingData.mPackedShadingNormal : PackNormal(-isect.mShadingData.GetShadingNormal())));

        const GpuMaterial m = LoadMaterial(isect.mShadingData);
        gOutputAlbedo[index.xy] = float4(m.BaseColor(), 1);

        if (frontFace)
            color = m.Emission();

        RandomSampler rng = RandomSampler(gRandomSeed, index.xy);
        color += SampleRadiance(isect.mShadingData, m, ray.Direction, rng);
    } else {
        gOutputPositions[index.xy] = float4(float3(POS_INFINITY), 0);
		gOutputAlbedo[index.xy] = 1;

		float tmp;
        color = EvalBackground(ray.Direction, tmp);
    }

    gOutput[index.xy] = float4(color, 1);
}
#include "Intersection.slang"
#include "Light.slang"
#include "Random.slang"


RWTexture2D<float4> gOutput;
uniform uint2 gOutputSize;

uniform uint gRandomSeed;

uniform float4x4 gCameraToWorld;
uniform float4x4 gWorldToCamera;
uniform float4x4 gProjection;
uniform float4x4 gInverseProjection;
RayDesc GetCameraRay(uint2 id) {
    RayDesc ray;
    ray.TMin = 0;
    ray.TMax = POS_INFINITY;
    ray.Origin = mul(gCameraToWorld, float4(0, 0, 0, 1)).xyz;
    const float2 uv = (float2(id + 0.5) / float2(gOutputSize)) * 2 - 1;
    ray.Direction = mul(gInverseProjection, float4(uv.x, -uv.y, 0, 1)).xyz;
    ray.Direction = normalize(mul(gCameraToWorld, float4(ray.Direction, 0)).xyz);
    return ray;
}

float3 SampleRadiance(const ShadingData sd, const GpuMaterial m, const float3 dirIn, inout RandomSampler rng) {
	EmissionSampleRecord l = SampleEmission(rng.NextFloat());

	float3 toLight = l.mShadingData.mPosition - sd.mPosition;
	const float dist = length(toLight);
	toLight /= dist;

    const float3 ns = sd.GetShadingNormal();
    const float cosToLight = max(0, dot(toLight, ns) * sign(dot(ns, -dirIn)));
	const float cosAtLight = max(0, -dot(toLight, l.mShadingData.GetShadingNormal()));
    const float3 brdf = (m.BaseColor() / M_PI) * cosToLight;
	if (cosAtLight > 0 && cosToLight > 0) {
		if (!Occluded(sd, toLight, dist)) {
            const float3 le = LoadMaterial(l.mShadingData).Emission();
            const float G = cosAtLight / sqr(dist);
            return brdf * le * G / l.mPdf;
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
        RandomSampler rng = RandomSampler(gRandomSeed, index.xy);
        const GpuMaterial m = LoadMaterial(isect.mShadingData);
        color = SampleRadiance(isect.mShadingData, m, ray.Direction, rng);
    } else {
        float tmp;
        color = EvalBackground(ray.Direction, tmp);
    }

    gOutput[index.xy] = float4(color, 1);
}
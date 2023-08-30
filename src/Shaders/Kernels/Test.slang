#include "Intersection.slang"
#include "Random.slang"


RWTexture2D<float4> gOutput;
uniform uint2 gOutputSize;

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

[shader("compute")]
[numthreads(8, 8, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gOutputSize)) return;

    const RayDesc ray = GetCameraRay(index.xy);

    IntersectionResult isect;
	float3 color;
    if (TraceRay(ray, true, isect)) {
        const GpuMaterial m = LoadMaterial(isect.mShadingData);

		color = m.BaseColor();
    } else {
        float tmp;
        color = EvalBackground(ray.Direction, tmp);
    }

    gOutput[index.xy] = float4(color, 1);
}
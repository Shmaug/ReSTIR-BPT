#include "Common/Math.h"

#include "Intersection.slang"

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
};

uniform float4x4 gCameraToWorld;
uniform float4x4 gInverseProjection;

RayDesc GetCameraRay(uint2 id) {
    RayDesc ray;
    ray.TMin = 0;
    ray.TMax = POS_INFINITY;
    ray.Origin = mul(gCameraToWorld, float4(0, 0, 0, 1)).xyz;
    ray.Direction = mul(gInverseProjection, float4((float2(id + 0.5) / float2(gOutputSize)) * 2 - 1, 0, 1)).xyz;
    ray.Direction = normalize(mul(gCameraToWorld, float4(ray.Direction, 0)).xyz);
    return ray;
}

RWTexture2D<float4> gRadiance;
RWTexture2D<float4> gAlbedos;
RWTexture2D<float4> gDepthNormals;
RWTexture2D<uint4> gVertices;

[shader("compute")]
[numthreads(8,8,1)]
void RenderVisibility(uint3 index : SV_DispatchThreadID) {
	const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    sHeatmapPixel = id.y*gOutputSize.x + id.x;

    RayDesc ray = GetCameraRay(id);

    float lightPdf, dist;
    const PathVertex vertex = TraceRay(ray, true, lightPdf, dist);

    const bool frontFace = vertex.mIsSurface ? dot(vertex.mShadingNormal, ray.Direction) < 0 : true;

    gDepthNormals[id] = float4(vertex.mPosition, asfloat(PackNormal(vertex.mShadingNormal * (frontFace ? 1 : -1))));
    gVertices[id] = reinterpret<uint4>(PackVertex(vertex));
	gRadiance[id] = float4(frontFace ? vertex.mMaterial.Emission() : 0, 1);
	gAlbedos[id]  = float4(vertex.mIsSurface ? vertex.mMaterial.BaseColor() : 1, 1);

	#ifdef gRenderAlbedos
    gRadiance[id] = gAlbedos[id];
	#endif
	#ifdef gRenderNormals
    gRadiance[id] = float4(vertex.mShadingNormal*.5+.5, 1);
	#endif
}
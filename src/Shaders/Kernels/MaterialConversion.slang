#include "Common/Common.h"
#include "Common/Material.h"

Texture2D<float4> gDiffuse;
Texture2D<float4> gSpecular;
RWTexture2D<float4> gOutput;
RWStructuredBuffer<uint> gOutputMinAlpha;

RWTexture2D<float> gRoughnessRW;

[shader("compute")]
[numthreads(8,8,1)]
void ComputeMinAlpha(uint3 index : SV_DispatchThreadId) {
    uint2 size;
    gOutput.GetDimensions(size.x, size.y);
    if (any(index.xy >= size)) return;
    const float4 diffuse = gDiffuse[index.xy];
	if (diffuse.a < 1)
		InterlockedMin(gOutputMinAlpha[0], (uint)floor(saturate(diffuse.a)*255 + 0.5));
}

[shader("compute")]
[numthreads(8,8,1)]
void ConvertMetallicRoughness(uint3 index : SV_DispatchThreadId) {
    uint2 size;
    gOutput.GetDimensions(size.x, size.y);
	if (any(index.xy >= size)) return;

    const float4 metallic_roughness = gSpecular[index.xy];
    gOutput[index.xy] = float4(metallic_roughness.b, metallic_roughness.g, 1, 1);
}
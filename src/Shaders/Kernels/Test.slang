#include "Common/Math.h"
#include "Random.slang"

RWTexture2D<float4> gOutput;
RWStructuredBuffer<float4> gRadiance[2];
uniform float4 gColor0;
uniform float4 gColor1;
uniform float gPower;

[shader("compute")]
[numthreads(8, 8, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
    uint2 resolution;
    gOutput.GetDimensions(resolution.x, resolution.y);
    if (any(index.xy >= resolution)) return;

    gRadiance[0][index.y * resolution.x + index.x] = lerp(gColor0, gColor1, pow(UnormToFloat(pcg(index.y * resolution.x + index.x)), gPower));
}

uniform int gBlurAxis;
uniform int gBlurRadius;
uniform float gBlurScale;

[shader("compute")]
[numthreads(8, 8, 1)]
void Blur(uint3 index: SV_DispatchThreadID) {
    uint2 resolution;
    gOutput.GetDimensions(resolution.x, resolution.y);
    if (any(index.xy >= resolution)) return;

    float4 value = 0;
    float wsum = 0;
    for (int x = -gBlurRadius; x <= gBlurRadius; x++) {
        int2 p = int2(index.xy) + (gBlurAxis == 0 ? int2(x, 0) : int2(0, x));
        if (any(p < 0) || any(p >= resolution))
            continue;

        float w = exp(-gBlurScale*(x*x));
        wsum += w;
        value += gRadiance[gBlurAxis == 0 ? 0 : 1][p.y * resolution.x + p.x] * w;
	}

    value *= SaveDivide(1, wsum);

    gRadiance[gBlurAxis == 0 ? 1 : 0][index.y * resolution.x + index.x] = value;
}

uniform float gExposure;

[shader("compute")]
[numthreads(8, 8, 1)]
void PostProcess(uint3 index: SV_DispatchThreadID) {
    uint2 resolution;
    gOutput.GetDimensions(resolution.x, resolution.y);
    if (any(index.xy >= resolution)) return;

    gOutput[index.xy] = float4(gRadiance[0][index.y * resolution.x + index.x].rgb * gExposure, 1);
}
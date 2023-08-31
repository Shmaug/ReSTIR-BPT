#ifndef gReproject
#define gReproject false
#endif
#ifndef gDebugMode
#define gDebugMode (DenoiserDebugMode)0
#endif

#include "Common/Enums.h"
#include "Common/SceneTypes.h"

Texture2D<float4> gInput;
Texture2D<float4> gPositions;
RWTexture2D<float4> gAccumColor;
RWTexture2D<float2> gAccumMoments;

Texture2D<float4> gPrevAccumColor;
Texture2D<float2> gPrevAccumMoments;
Texture2D<float4> gPrevPositions;

uniform float gHistoryLimit;
uniform float gNormalReuseCutoff;
uniform float gDepthReuseCutoff;
uniform float4x4 gPrevWorldToClip;

[shader("compute")]
[numthreads(8, 8, 1)]
void Accumulate(uint3 index: SV_DispatchThreadId) {
    uint2 extent;
    gAccumColor.GetDimensions(extent.x, extent.y);
    if (any(index.xy >= extent)) return;

	const uint2 ipos = index.xy;

	float4 color_prev   = 0;
	float2 moments_prev = 0;
    float sum_w = 0;
    if (gReproject) {
        const float4 pos = gPositions[index.xy];
        if (all(isfinite(pos.xyz))) {
            const float4 prevClipPos = mul(gPrevWorldToClip, float4(pos.xyz, 1));
            const float2 prevUV = (prevClipPos.xy / prevClipPos.w) * .5 + .5;
            if (all(prevUV >= 0) && all(prevUV < 1)) {
                const float3 normal = UnpackNormal(asuint(pos.w));
                const float2 prevPixel = prevUV * extent - 0.5;
                const float2 w = frac(prevPixel);
				// bilinear interpolation, check each tap individually, renormalize afterwards
				for (int yy = 0; yy < 2; yy++) {
					for (int xx = 0; xx < 2; xx++) {
						const int2 tap = int2(prevPixel) + int2(xx, yy);
						if (any(tap < 0) || any(tap >= extent)) continue;

                        const float4 p = gPrevPositions[tap];
						if (!all(isfinite(p.xyz)))
							continue;
						const float3 n = UnpackNormal(asuint(p.w));
						if (dot(normal, n) < gNormalReuseCutoff)
							continue;
						if (abs(dot(p.xyz - pos.xyz, n)) > gDepthReuseCutoff)
							continue;

                        const float4 c = gPrevAccumColor[tap];

						if (c.a <= 0 || any(isnan(c)) || any(isinf(c)))
							continue;

                        float2 m = gPrevAccumMoments[tap];
                        if (any(isnan(m)) || any(isinf(m)))
                            m = 0;

                        color_prev.a = color_prev.a == 0 ? c.a : min(c.a, color_prev.a);

                        const float wc = (xx == 0 ? (1 - w.x) : w.x) * (yy == 0 ? (1 - w.y) : w.y);
                        color_prev.rgb += c.rgb * wc;
                        moments_prev += m * wc;
						sum_w        += wc;
					}
           		}
            }
        }
	} else {
		// static accumulation
		color_prev = gPrevAccumColor[ipos];
		if (any(isnan(color_prev.rgb)) || any(isinf(color_prev.rgb)))
            color_prev = 0;
        else {
            moments_prev = gPrevAccumMoments[ipos];
			if (any(isnan(moments_prev)) || any(isinf(moments_prev)))
                moments_prev = 0;
        }
		sum_w = 1;
	}

	float4 color_curr = gInput[ipos];

	// suppress NaNs from the current frame
	if (any(isinf(color_curr.rgb)) || any(isnan(color_curr.rgb))) color_curr = 0;

	const float l = Luminance(color_curr.rgb);

	// normalize reprojected sample

	if (sum_w > 0 && color_prev.a > 0) {
		const float invSum = 1/sum_w;
		color_prev.rgb *= invSum;
		moments_prev *= invSum;

        float n = color_prev.a + color_curr.a;
        if (gHistoryLimit > 0 && n > gHistoryLimit)
            n = gHistoryLimit;

        const float alpha = saturate(color_curr.a / n);

		gAccumColor[ipos] = float4(lerp(color_prev.rgb, color_curr.rgb, alpha), n);
		gAccumMoments[ipos] = lerp(moments_prev, float2(l, l*l), alpha);

        if (gDebugMode == DenoiserDebugMode::eSampleCount)
            gAccumColor[ipos] = float4(ViridisQuintic(saturate(n / (gHistoryLimit > 0 ? gHistoryLimit : 1024))), 1);
	} else {
		gAccumColor[ipos] = color_curr;
        gAccumMoments[ipos] = float2(l, l * l);

        if (gDebugMode == DenoiserDebugMode::eSampleCount)
			gAccumColor[ipos] = float4(ViridisQuintic(0), 1);
	}

    if (gDebugMode == DenoiserDebugMode::eWeightSum)
        gAccumColor[ipos] = float4(ViridisQuintic(sum_w), 1);
	else if (gDebugMode == DenoiserDebugMode::eVariance) {
        const float2 m = gAccumMoments[ipos];
        gAccumColor[ipos] = float4(ViridisQuintic(saturate(abs(m.y - sqr(m.x)))), 1);
	}
}
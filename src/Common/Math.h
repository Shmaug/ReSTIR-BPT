#pragma once

#include "Common.h"

PTVK_NAMESPACE_BEGIN

#ifndef M_PI
#define M_PI (3.1415926535897932)
#define M_1_PI (1/M_PI)
#endif

#define POS_INFINITY asfloat(0x7F800000)
#define NEG_INFINITY asfloat(0xFF800000)

#ifdef __cplusplus
inline float3 TransformPoint(float4x4 t, float3 p)  { return (float3)mul(t, float4(p,1)); }
inline float3 TransformVector(float4x4 t, float3 p) { return (float3)mul(t, float4(p,0)); }
#else
inline float3 TransformPoint(float4x4 t, float3 p)  { return mul(t, float4(p,1)).xyz; }
inline float3 TransformVector(float4x4 t, float3 p) { return mul(t, float4(p,0)).xyz; }
#endif

inline float min3(float3 v) { return min(min(v.r,v.g),v.b); }
inline float max3(float3 v) { return max(max(v.r,v.g),v.b); }

inline float sqr(float x) { return x*x; }
inline float2 sqr(float2 x) { return x*x; }
inline float3 sqr(float3 x) { return x*x; }
inline float4 sqr(float4 x) { return x*x; }
inline float pow5(float x) { return sqr(sqr(x))*x; }

inline float Luminance(const float3 color) { return dot(color, float3(0.2126, 0.7152, 0.0722)); }

inline float SaveDivide(const float numerator, const float denominator, const float value = 0) { return denominator == 0 ? value : numerator / denominator; }

inline float StableAtan2(const float y, const float x) {
	return x == 0.0 ? (y == 0 ? 0 : (y < 0 ? -M_PI/2 : M_PI/2)) : atan2(y, x);
}

inline float3x3 MakeOrthonormal(float3 N) {
    float3x3 r;
	if (N[0] != N[1] || N[0] != N[2])
		r[0] = float3(N[2] - N[1], N[0] - N[2], N[1] - N[0]);  // (1,1,1) x N
	else
		r[0] = float3(N[2] - N[1], N[0] + N[2], -N[1] - N[0]);  // (-1,1,1) x N
	r[0] = normalize(r[0]);
	r[1] = cross(N, r[0]);
    r[2] = N;
    return r;
}

inline float2 CartesianToSphericalUV(const float3 v) {
	const float theta = StableAtan2(v[2], v[0]);
	return float2(theta*M_1_PI*.5 + .5, acos(clamp(v[1], -1.f, 1.f))*M_1_PI);
}
inline float3 SphericalUVToCartesian(float2 uv) {
	uv[0] = uv[0]*2 - 1;
	uv *= M_PI;
	const float sinPhi = sin(uv[1]);
	return float3(sinPhi*cos(uv[0]), cos(uv[1]), sinPhi*sin(uv[0]));
}

inline float2 RayAabb(const float3 origin, const float3 invDir, const float3 mn, const float3 mx) {
	const float3 t0 = (mn - origin) * invDir;
	const float3 t1 = (mx - origin) * invDir;
	return float2(max3(min(t0, t1)), min3(max(t0, t1)));
}
inline float2 RaySphere(const float3 origin, const float3 dir, const float3 p, const float r) {
	const float3 f = origin - p;
	const float a = dot(dir, dir);
	const float b = dot(f, dir);
	const float3 l = a*f - dir*b;
	float det = sqr(a*r) - dot(l,l);
	if (det < 0) return float2(0,0);
	const float inv_a = 1/a;
	det = sqrt(det * inv_a) * inv_a;
	return -float2(1,1)*b*inv_a + float2(-det, det);
}

inline float3 XyzToRgb(const float3 xyz) {
	return float3(
			3.240479f * xyz[0] - 1.537150 * xyz[1] - 0.498535 * xyz[2],
		   -0.969256f * xyz[0] + 1.875991 * xyz[1] + 0.041556 * xyz[2],
			0.055648f * xyz[0] - 0.204043 * xyz[1] + 1.057311 * xyz[2]);
}

inline float3 SrgbToRgb(const float3 srgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_sRGB_to_CIE_XYZ
	float3 rgb;
	for (int i = 0; i < 3; i++)
		rgb[i] = srgb[i] <= 0.04045 ? srgb[i] / 12.92 : pow((srgb[i] + 0.055) / 1.055, 2.4);
	return rgb;
}
inline float3 RgbToSrgb(const float3 rgb) {
	// https://en.wikipedia.org/wiki/SRGB#From_CIE_XYZ_to_sRGB
	float3 srgb;
	for (int i = 0; i < 3; i++)
		srgb[i] = rgb[i] <= 0.0031308 ? rgb[i] * 12.92 : pow(rgb[i] * 1.055, 1/2.4) - 0.055;
	return srgb;
}

inline float3 ViridisQuintic(const float x) {
	// from https://www.shadertoy.com/view/XtGGzG
	float4 x1 = float4(1, x, x*x, x*x*x); // 1 x x2 x3
	float2 x2 = float2(x1[1], x1[2]) * x1[3]; // x4 x5
	return float3(
		dot(x1, float4( 0.280268003, -0.143510503,   2.225793877, -14.815088879)) + dot(x2, float2( 25.212752309, -11.772589584)),
		dot(x1, float4(-0.002117546,  1.617109353,  -1.909305070,   2.701152864)) + dot(x2, float2(-1.685288385 ,   0.178738871)),
		dot(x1, float4( 0.300805501,  2.614650302, -12.019139090,  28.933559110)) + dot(x2, float2(-33.491294770,  13.762053843)));
}

PTVK_NAMESPACE_END
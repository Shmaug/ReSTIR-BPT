#pragma once

#ifdef __cplusplus
#define PTVK_NAMESPACE ptvk
#define PTVK_NAMESPACE_BEGIN namespace PTVK_NAMESPACE {
#define PTVK_NAMESPACE_END }
#define CPP_CONST const
#else
#define CPP_CONST
#define PTVK_NAMESPACE_BEGIN
#define PTVK_NAMESPACE_END
#endif

#ifdef __SLANG_COMPILER__
#define SLANG_MUTATING [mutating]
#define SLANG_CTOR(TYPE) __init
#else
#define SLANG_MUTATING
#define SLANG_CTOR(TYPE) inline TYPE
#endif

// hlsl types

#ifdef __cplusplus

#include <glm/gtx/compatibility.hpp>
#include <concepts>
#include <bit>

PTVK_NAMESPACE_BEGIN

using uint = uint32_t;
using glm::int2;
using glm::int3;
using glm::int4;
using uint2 = glm::uvec2;
using uint3 = glm::uvec3;
using uint4 = glm::uvec4;
using glm::float2;
using glm::float3;
using glm::float3;
using glm::float4;
using glm::double2;
using glm::double3;
using glm::double3;
using glm::double4;
using glm::float2x2;
using glm::float3x3;
using glm::float4x4;
using glm::float4x3;
using glm::float3x4;
using glm::abs;
using glm::dot;
using glm::cross;
using glm::min;
using glm::max;
using glm::clamp;
using glm::floor;
using glm::ceil;
using glm::isnan;
using glm::trunc;
using glm::saturate;
using glm::lerp;
using glm::transpose;
using glm::inverse;

inline float asfloat(uint v) { return std::bit_cast<float>(v); }
inline float2 asfloat(uint2 v) { return float2( std::bit_cast<float>(v.x), std::bit_cast<float>(v.y)); }
inline float3 asfloat(uint3 v) { return float3( std::bit_cast<float>(v.x), std::bit_cast<float>(v.y), std::bit_cast<float>(v.z)); }
inline float4 asfloat(uint4 v) { return float4( std::bit_cast<float>(v.x), std::bit_cast<float>(v.y), std::bit_cast<float>(v.z), std::bit_cast<float>(v.w)); }
inline uint asuint(float v) { return std::bit_cast<uint>(v); }
inline uint2 asuint(float2 v) { return uint2( std::bit_cast<uint>(v.x), std::bit_cast<uint>(v.y)); }
inline uint3 asuint(float3 v) { return uint3( std::bit_cast<uint>(v.x), std::bit_cast<uint>(v.y), std::bit_cast<uint>(v.z)); }
inline uint4 asuint(float4 v) { return uint4( std::bit_cast<uint>(v.x), std::bit_cast<uint>(v.y), std::bit_cast<uint>(v.z), std::bit_cast<uint>(v.w)); }


// from https://gist.github.com/rygorous/2156668
union FP32 {
    uint u;
    float f;
    struct {
        uint Mantissa : 23;
        uint Exponent : 8;
        uint Sign : 1;
    };
};
union FP16 {
    unsigned short u;
    struct {
        uint Mantissa : 10;
        uint Exponent : 5;
        uint Sign : 1;
    };
};

inline uint f32tof16(float _f) {
	// Original ISPC reference version; this always rounds ties up.
	FP32 f;
	f.f = _f;

    FP16 o = { 0 };

    // Based on ISPC reference code (with minor modifications)
    if (f.Exponent == 0) // Signed zero/denormal (which will underflow)
        o.Exponent = 0;
    else if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
    {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
    }
    else // Normalized number
    {
        // Exponent unbias the single, then bias the halfp
        int newexp = f.Exponent - 127 + 15;
        if (newexp >= 31) // Overflow, return signed infinity
            o.Exponent = 31;
        else if (newexp <= 0) // Underflow
        {
            if ((14 - newexp) <= 24) // Mantissa might be non-zero
            {
                uint mant = f.Mantissa | 0x800000; // Hidden 1 bit
                o.Mantissa = mant >> (14 - newexp);
                if ((mant >> (13 - newexp)) & 1) // Check for rounding
                    o.u++; // Round, might overflow into exp bit, but this is OK
            }
        }
        else
        {
            o.Exponent = newexp;
            o.Mantissa = f.Mantissa >> 13;
            if (f.Mantissa & 0x1000) // Check for rounding
                o.u++; // Round, might overflow to inf, this is OK
        }
    }

    o.Sign = f.Sign;
    return o.u;
}
inline float f16tof32(uint h) {
    static const FP32 magic = { 113 << 23 };
    static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
    FP32 o;

    o.u = (h & 0x7fff) << 13;     // exponent/mantissa bits
    uint exp = shifted_exp & o.u;   // just the exponent
    o.u += (127 - 15) << 23;        // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
    else if (exp == 0) // Zero/Denormal?
    {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
    }

    o.u |= (h & 0x8000) << 16;    // sign bit
    return o.f;
}

PTVK_NAMESPACE_END

#endif
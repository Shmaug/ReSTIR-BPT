#pragma once

#include "Math.h"

PTVK_NAMESPACE_BEGIN

// bitfield macros from https://gist.github.com/Jeff-Russ/c9b471158fa7427280e6707d9b11d7d2

/* Bit Manipulation Macros
A good article: http://www.coranac.com/documents/working-with-bits-and-bitfields/
x    is a variable that will be modified.
y    will not.
pos  is a unsigned int (usually 0 through 7) representing a single bit position where the
     right-most bit is bit 0. So 00000010 is pos 1 since the second bit is high.
bm   (bit mask) is used to specify multiple bits by having each set ON.
bf   (bit field) is similar (it is in fact used as a bit mask) but it is used to specify a
     range of neighboring bit by having them set ON.
*/
/* shifts left the '1' over pos times to create a single HIGH bit at location pos. */
#define BIT(pos) ( 1u << (pos) )

/* Set single bit at pos to '1' by generating a mask
in the proper bit location and ORing x with the mask. */
#define SET_BIT(x, pos) ( (x) |= ((1 << (pos))) )
#define SET_BITS(x, bm) ( (x) |= (bm) ) // same but for multiple bits

/* Set single bit at pos to '0' by generating a mask
in the proper bit location and ORing x with the mask. */
#define UNSET_BIT(x, pos) ( (x) &= ~((1 << (pos))) )
#define UNSET_BITS(x, bm) ( (x) &= (~(bm)) ) // same but for multiple bits

/* Set single bit at pos to opposite of what is currently is by generating a mask
in the proper bit location and ORing x with the mask. */
#define FLIP_BIT(x, pos) ( (x) ^= ((1 << (pos))) )
#define FLIP_BITS(x, bm) ( (x) ^= (bm) ) // same but for multiple bits

/* Return '1' if the bit value at position pos within y is '1' and '0' if it's 0 by
ANDing x with a bit mask where the bit in pos's position is '1' and '0' elsewhere and
comparing it to all 0's.  Returns '1' in least significant bit position if the value
of the bit is '1', '0' if it was '0'. */
#define CHECK_BIT(y, pos) ( ( 0u == ( (y)&((1 << (pos))) ) ) ? 0u : 1u )
#define CHECK_BITS_ANY(y, bm) ( ( (y) & (bm) ) ? 0u : 1u )
// warning: evaluates bm twice:
#define CHECK_BITS_ALL(y, bm) ( ( (bm) == ((y)&(bm)) ) ? 0u : 1u )

// These are three preparatory macros used by the following two:
#define SET_LSBITS(len) ( (1u << (len)) - 1u ) // the first len bits are '1' and the rest are '0'
#define BF_MASK(start, len) ( SET_LSBITS(len) << (start) ) // same but with offset
#define BF_PREP(y, start, len) ( ((y)&SET_LSBITS(len)) << (start) ) // Prepare a bitmask

/* Extract a bitfield of length len starting at bit start from y. */
#define BF_GET(y, start, len) ( ((y) >> (start)) & SET_LSBITS(len) )

/* Insert a new bitfield value bf into x. */
#define BF_SET(x, bf, start, len) ( x = ((x) &~ BF_MASK(start, len)) | BF_PREP(bf, start, len) )

////////////////////////////////////////////////////////////////////////////////////////////////

#define BF_GET_UNORM(x, start, len) ( BF_GET(x, start, len) / float(SET_LSBITS(len)) )
#define BF_SET_UNORM(x, newValue, start, len) ( BF_SET(x, (uint)floor(saturate(newValue)*SET_LSBITS(len) + 0.5f), start, len) )

struct PackedUnorm4 {
	uint mValue;
    float Get(uint index) { return BF_GET_UNORM(mValue, index*8, 8); }
	SLANG_MUTATING void Set(uint index, float newValue) { BF_SET_UNORM(mValue, newValue, index*8, 8); }
};
struct PackedUnorm8 {
	uint2 mValue;
    float Get(uint index) { return BF_GET_UNORM(mValue[index/4], (index%4)*8, 8); }
	SLANG_MUTATING void Set(uint index, float newValue) { BF_SET_UNORM(mValue[index/4], newValue, (index%4)*8, 8); }
};
struct PackedUnorm16 {
	uint4 mValue;
    float Get(uint index) { return BF_GET_UNORM(mValue[index/4], (index%4)*8, 8); }
	SLANG_MUTATING void Set(uint index, float newValue) { BF_SET_UNORM(mValue[index/4], newValue, (index%4)*8, 8); }
};

// stores a 0-1 base color and a HDR emission color
struct PackedColors {
	PackedUnorm8 mPacked;

	float3 GetColor() {
		return float3( mPacked.Get(0), mPacked.Get(1), mPacked.Get(2) );
	}
	SLANG_MUTATING void SetColor(float3 newValue) {
		mPacked.Set(0, newValue.r);
		mPacked.Set(1, newValue.g);
		mPacked.Set(2, newValue.b);
	}

	float3 GetColorHDR() {
		float scale = f16tof32(BF_GET(mPacked.mValue[1], 16, 16));
		return scale * float3( mPacked.Get(3), mPacked.Get(4), mPacked.Get(5) );
	}
	SLANG_MUTATING void SetColorHDR(float3 newValue) {
		float m = max3(newValue);
		if (m > 0)
			newValue /= m;
		BF_SET(mPacked.mValue[1], f32tof16(m), 16, 16);
		mPacked.Set(3, newValue.r);
		mPacked.Set(4, newValue.g);
		mPacked.Set(5, newValue.b);
	}
};

////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef __SLANG_COMPILER__

#include "D3DX_DXGIFormatConvert.h"

float2 PackNormalF32(const float3 v) {
	// Project the sphere onto the octahedron, and then onto the xy plane
	const float2 p = v.xy * (1 / (abs(v.x) + abs(v.y) + abs(v.z)));
	// Reflect the folds of the lower hemisphere over the diagonals
	return (v.z <= 0) ? ((1 - abs(p.yx)) * lerp(-1, 1, float2(p >= 0))) : p;
}
float3 UnpackNormalF32(const float2 p) {
	float3 v = float3(p, 1 - dot(1, abs(p)));
	if (v.z < 0) v.xy = (1 - abs(v.yx)) * lerp(-1, 1, float2(v.xy >= 0));
	return normalize(v);
}

uint PackNormal(const float3 v)   { return D3DX_FLOAT2_to_R16G16_SNORM(PackNormalF32(v)); }
float3 UnpackNormal(const uint p) { return UnpackNormalF32(D3DX_R16G16_SNORM_to_FLOAT2(p)); }

#endif

PTVK_NAMESPACE_END
#pragma once

#include "Common/Enums.h"

struct DenoiserParameters {
	StructuredBuffer<ViewData> mViews;
	StructuredBuffer<uint> mInstanceIndexMap;
	Texture2D<uint2> mVisibility;
	Texture2D<uint2> mPrevVisibility;
	Texture2D<float4> mDepth;
	Texture2D<float4> mPrevDepth;
	Texture2D<float2> mPrevUVs;
	Texture2D<float4> mInput;
	Texture2D<float4> mAlbedo;
	RWTexture2D<float4> mAccumColor;
	RWTexture2D<float2> mAccumMoments;
	RWTexture2D<float4> mFilterImages[2];

	Texture2D<float4> mPrevAccumColor;
	Texture2D<float2> mPrevAccumMoments;
};
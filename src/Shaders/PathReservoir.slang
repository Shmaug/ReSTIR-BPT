#pragma once

#include "Random.slang"
#include "Common/PackedTypes.h"

struct ReconnectionVertex {
    PackedVertex mVertex;
    uint2 mPackedRadiance;
    float mG;
    float mPrefixPdfW;
    uint mPackedDirOut;

	__init() {
		mPackedRadiance = 0;
		mG = 0;
		mPrefixPdfW = 0;
	}

    property float3 mRadiance {
		get {
			return float3(
				f16tof32(BF_GET(mPackedRadiance[0],  0, 16)),
				f16tof32(BF_GET(mPackedRadiance[0], 16, 16)),
				f16tof32(BF_GET(mPackedRadiance[1],  0, 16)));
		}
		set {
			BF_SET(mPackedRadiance[0], f32tof16(newValue.r),  0, 16);
			BF_SET(mPackedRadiance[0], f32tof16(newValue.g), 16, 16);
			BF_SET(mPackedRadiance[1], f32tof16(newValue.b),  0, 16);
        }
    }

    property uint mPrefixBounces {
		get { return BF_GET(mPackedRadiance[1], 16, 8); }
        set { BF_SET(mPackedRadiance[1], newValue, 16, 8); }
    }

    property float3 mDirOut {
		get { return UnpackNormal(mPackedDirOut); }
        set { mPackedDirOut = PackNormal(newValue); }
    }
};
static const uint ReconnectionVertexSize = sizeof(ReconnectionVertex);

struct PathSample {
    float3 mRadiance;
    float mPdfW;
    uint mRngSeed;
    uint mPacked;
    #ifdef RECONNECTION
    ReconnectionVertex mReconnectionVertex;
	#else
    property ReconnectionVertex mReconnectionVertex { get { return ReconnectionVertex(); } set { } }
	#endif

    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPacked, 0, 12), BF_GET(mPacked, 12, 12)); }
        set { BF_SET(mPacked, newValue.x, 0, 12); BF_SET(mPacked, newValue.y, 12, 12); }
	}

    property uint mBounces {
        get { return BF_GET(mPacked, 24, 7); }
        set { BF_SET(mPacked, newValue, 24, 7); }
    }
    property bool mIsNEE {
        get { return (bool)BF_GET(mPacked, 31, 1); }
        set { BF_SET(mPacked, (uint)newValue, 31, 1); }
    }

	__init() {
        mRadiance = 0;
		mPdfW = 0;
        mRngSeed = 0;
        mPacked = 0;
		#ifdef RECONNECTION
        mReconnectionVertex = ReconnectionVertex();
		#endif
    }
    __init(float3 radiance, float pdfW, uint bounces, bool isNee, const uint rngSeed, const uint2 rngIdx, ReconnectionVertex rcv) {
		mRadiance = radiance;
        mPdfW = pdfW;
        mRngSeed = rngSeed;
        mRngIdx = rngIdx;
        mBounces = bounces;
        mIsNEE = isNee;
		#ifdef RECONNECTION
		mReconnectionVertex = rcv;
		#endif
	}
};
static const uint PathSampleSize = sizeof(PathSample);

struct PathReservoir {
    PathSample mSample;
    float mW;
    float mM;

	__init() {
		mW = 0;
        mM = 0;
        mSample = PathSample();
    }
    __init(const PathSample s) {
        mSample = s;
        mW = SafeDivide(1, s.mPdfW);
        mM = 1;
    }
    __init(const PathSample s, const float W, const float M) {
        mSample = s;
        mW = W;
        mM = M;
    }

	[mutating]
    void PrepareMerge(float misWeight = 1, float jacobian = 1) {
        mW *= Luminance(mSample.mRadiance) * misWeight * jacobian;
    }
    // note: PrepareMerge must be called on both reservoirs prior to calling Merge
	[mutating]
    bool Merge(float rnd, PathReservoir r) {
        mM += r.mM;

        if (r.mW <= 0 || isnan(r.mW))
            return false;

        mW += r.mW;
        if (rnd*mW < r.mW) {
            mSample = r.mSample;
            return true;
        }

        return false;
    }
	[mutating]
    void FinalizeMerge() {
        float p = Luminance(mSample.mRadiance);
        if (p > 0)
            mW /= p;
    }
};
static const uint PathReservoirSize = sizeof(PathReservoir);



#ifdef OUTPUT_RADIANCE_SHADER

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
}

RWTexture2D<float4> gRadiance;
StructuredBuffer<PathReservoir> gPathReservoirsIn;
uint FlattenIndex(uint2 id) { return id.y * gOutputSize.x + id.x; }
PathReservoir LoadReservoir(uint2 id) { return gPathReservoirsIn[FlattenIndex(id)]; }

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

    const PathReservoir r = LoadReservoir(id);
    if (r.mW > 0)
        gRadiance[id] += float4(r.mSample.mRadiance * r.mW, 0);
}

#endif
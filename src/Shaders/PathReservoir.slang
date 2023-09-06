#pragma once

#include "Random.slang"
#include "Common/PackedTypes.h"

struct PackedVertex {
    float3 mLocalPosition;
    uint mInstancePrimitiveIndex;

    uint GetInstanceIndex()  { return BF_GET(mInstancePrimitiveIndex,  0, 16); }
    uint GetPrimitiveIndex() { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
    [mutating] void SetInstanceIndex (uint newValue) { BF_SET(mInstancePrimitiveIndex, newValue,  0, 16); }
    [mutating] void SetPrimitiveIndex(uint newValue) { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
};

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

	float3 GetRadiance() {
        return float3(
            f16tof32(BF_GET(mPackedRadiance[0],  0, 16)),
            f16tof32(BF_GET(mPackedRadiance[0], 16, 16)),
            f16tof32(BF_GET(mPackedRadiance[1],  0, 16)));
    }
	[mutating]
	void SetRadiance(float3 newValue) {
        BF_SET(mPackedRadiance[0], f32tof16(newValue.r),  0, 16);
        BF_SET(mPackedRadiance[0], f32tof16(newValue.g), 16, 16);
        BF_SET(mPackedRadiance[1], f32tof16(newValue.b),  0, 16);
    }

	uint GetPrefixBounces() { return BF_GET(mPackedRadiance[1], 16, 8); }
	[mutating]
	void SetPrefixBounces(uint newValue) { BF_SET(mPackedRadiance[1], newValue, 16, 8); }

	bool IsLastVertex() { return BF_GET(mPackedRadiance[1], 31, 1); }
	[mutating]
	void SetIsLastVertex(bool newValue) { BF_SET(mPackedRadiance[1], newValue, 31, 1); }

	float3 GetDirOut() { return UnpackNormal(mPackedDirOut); }
	[mutating]
    void SetDirOut(float3 newValue) { mPackedDirOut = PackNormal(newValue); }
};
static const uint ReconnectionVertexSize = sizeof(ReconnectionVertex);

struct PathSample {
    float3 mRadiance;
    float mPdfW;
    RandomSampler mRngSeed;
    uint mPacked;
    #ifdef RECONNECTION
    ReconnectionVertex mReconnectionVertex;
	#else
    property ReconnectionVertex mReconnectionVertex { get { return ReconnectionVertex(); } }
	#endif

    property uint mBounces {
        get { return BF_GET(mPacked, 0, 16); }
        set { BF_SET(mPacked, newValue, 0, 16); }
    }
    property bool mIsNEE {
        get { return (bool)BF_GET(mPacked, 16, 1); }
        set { BF_SET(mPacked, (uint)newValue, 16, 1); }
    }

	__init() {
		mRadiance = 0;
		mPdfW = 0;
		mBounces = 0;
		#ifdef RECONNECTION
        mReconnectionVertex = ReconnectionVertex();
		#endif
	}
	__init(float3 radiance, float pdfW, uint bounces, bool isNee, RandomSampler seed, ReconnectionVertex rcv) {
		mRadiance = radiance;
		mPdfW = pdfW;
		mRngSeed = seed;
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
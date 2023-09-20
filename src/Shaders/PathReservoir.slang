#pragma once

#include "Common/PackedTypes.h"
#include "Random.slang"

struct ReconnectionVertex {
    PackedVertex mVertex;
    float3 mRadiance;
    uint mPackedDirOut;
    uint mPrefixBounces;
    float mDist;
    float mCos;
    float pad;

	__init() {
        mRadiance = 0;
        mPrefixBounces = 0;
        mDist = 0;
        mCos = 0;
	}

    property float3 mDirOut {
		get { return UnpackNormal(mPackedDirOut); }
        set { mPackedDirOut = PackNormal(newValue); }
    }
};
static const uint ReconnectionVertexSize = sizeof(ReconnectionVertex);

struct PathSample {
    #if defined(RECONNECTION) || defined(BIDIRECTIONAL)
    ReconnectionVertex mReconnectionVertex;
	#else
    property ReconnectionVertex mReconnectionVertex { get { return ReconnectionVertex(); } set { } }
	#endif
    float3 mRadiance;
    float mReplayPdfW;
    uint mRngSeed;
    uint mPackedRngIdx;
    uint mBounces;
    uint mLightVertices;

    property bool mHasRcv {
        get { return (bool)BF_GET(mPackedRngIdx, 30, 1); }
        set { BF_SET(mPackedRngIdx, (uint)newValue, 30, 1); }
    }
    property bool mIsNEE {
        get { return (bool)BF_GET(mPackedRngIdx, 31, 1); }
        set { BF_SET(mPackedRngIdx, (uint)newValue, 31, 1); }
    }
    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPackedRngIdx, 0, 15), BF_GET(mPackedRngIdx, 15, 15)); }
        set { BF_SET(mPackedRngIdx, newValue.x, 0, 15); BF_SET(mPackedRngIdx, newValue.y, 15, 15); }
	}

    __init() {
		#ifdef RECONNECTION
        mReconnectionVertex = ReconnectionVertex();
		#endif
        mRadiance = 0;
		mReplayPdfW = 0;
        mRngSeed = 0;
        mPackedRngIdx = 0;
        mBounces = 0;
        mLightVertices = 0;
    }
    __init(float3 radiance, float replayPdfW, uint bounces, uint lightVertices, const uint rngSeed, const uint2 rngIdx, ReconnectionVertex rcv, bool rcvValid) {
		mRadiance = radiance;
        mReplayPdfW = replayPdfW;
        mRngSeed = rngSeed;
        mRngIdx  = rngIdx;
        mBounces = bounces;
        mIsNEE = lightVertices == 1;
        mLightVertices = lightVertices;
		mReconnectionVertex = rcv;
		mHasRcv = rcvValid;
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
        else
            mW = 0;
    }
};
static const uint PathReservoirSize = sizeof(PathReservoir);

typedef ByteAddressBuffer ReservoirBuffer;
typedef RWByteAddressBuffer RWReservoirBuffer;

uint PreIncrement(inout uint address, const uint size, const uint index, const uint count) {
    uint val = address;
    address += count * size;
    return val + index * size;
}

PathReservoir LoadReservoir(ReservoirBuffer buf, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
    const uint count = extent.x * extent.y;
    if (index >= count) return PathReservoir();

    PathReservoir r = PathReservoir();

    uint address = 0;
    float2 data = buf.Load<float2>(PreIncrement(address, 8, index, count));
    r.mW = data[0];
    r.mM = data[1];

    if (r.mW > 0 && r.mM > 0)
        r.mSample = buf.Load<PathSample>(PreIncrement(address, PathSampleSize, index, count));

    return r;
}
void StoreReservoir(RWReservoirBuffer buf, const PathReservoir r, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
    const uint count = extent.x * extent.y;
    if (index >= count) return;

    uint address = 0;
    buf.Store<float2>(PreIncrement(address, 8, index, count), float2(r.mW, r.mM));

    if (r.mW > 0 && r.mM > 0)
        buf.Store<PathSample>(PreIncrement(address, PathSampleSize, index, count), r.mSample);
}
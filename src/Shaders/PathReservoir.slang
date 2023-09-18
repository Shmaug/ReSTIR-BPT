#pragma once

#include "Random.slang"
#include "Common/PackedTypes.h"

struct ReconnectionVertex {
    PackedVertex mVertex;
    uint2 mPackedRadiance;
    uint mPackedG;
    uint mPackedDirOut;

	__init() {
        mPackedRadiance = 0;
        mPackedG = 0;
	}

    property float mDist {
        get { return f16tof32(BF_GET(mPackedG, 0, 16)); }
        set { BF_SET(mPackedG, f32tof16(newValue), 0, 16); }
    }
    property float mCos {
        get { return f16tof32(BF_GET(mPackedG, 16, 16)); }
        set { BF_SET(mPackedG, f32tof16(newValue), 16, 16); }
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
		get { return BF_GET(mPackedRadiance[1], 16, 16); }
        set { BF_SET(mPackedRadiance[1], newValue, 16, 16); }
    }
    property float3 mDirOut {
		get { return UnpackNormal(mPackedDirOut); }
        set { mPackedDirOut = PackNormal(newValue); }
    }
};
static const uint ReconnectionVertexSize = sizeof(ReconnectionVertex);

struct PathSample {
    float3 mRadiance;
    float mReplayPdfW;
    uint mRngSeed;
    uint mPackedRngIdx;
	uint mLightVertices;
    #ifdef RECONNECTION
    ReconnectionVertex mReconnectionVertex;
	#else
    property ReconnectionVertex mReconnectionVertex { get { return ReconnectionVertex(); } set { } }
	#endif

    property uint mBounces {
        get { return BF_GET(mPackedRngIdx, 24, 6); }
        set { BF_SET(mPackedRngIdx, newValue, 24, 6); }
    }
    property bool mIsNEE {
        get { return (bool)BF_GET(mPackedRngIdx, 31, 1); }
        set { BF_SET(mPackedRngIdx, (uint)newValue, 31, 1); }
    }
    property bool mIsVC {
        get { return (bool)BF_GET(mPackedRngIdx, 30, 1); }
        set { BF_SET(mPackedRngIdx, (uint)newValue, 30, 1); }
    }
    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPackedRngIdx, 0, 12), BF_GET(mPackedRngIdx, 12, 12)); }
        set { BF_SET(mPackedRngIdx, newValue.x, 0, 12); BF_SET(mPackedRngIdx, newValue.y, 12, 12); }
	}

    __init() {
        mRadiance = 0;
		mReplayPdfW = 0;
        mRngSeed = 0;
        mPackedRngIdx = 0;
        mReconnectionVertex = ReconnectionVertex();
    }
    __init(float3 radiance, float replayPdfW, uint bounces, uint lightVertices, const uint rngSeed, const uint2 rngIdx, ReconnectionVertex rcv) {
		mRadiance = radiance;
        mReplayPdfW = replayPdfW;
        mRngSeed = rngSeed;
        mRngIdx  = rngIdx;
        mBounces = bounces;
        mIsNEE = lightVertices == 1;
        mIsVC  = lightVertices > 1;
        mLightVertices = lightVertices;
		mReconnectionVertex = rcv;
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
    uint4 data = buf.Load4(PreIncrement(address, 16, index, count));
    r.mW            = asfloat(data[0]);
    r.mM            = asfloat(data[1]);
	r.mSample.mRngSeed      = data[2];
	r.mSample.mPackedRngIdx = data[3];

    if (r.mW > 0 && r.mM > 0) {
        data = buf.Load4(PreIncrement(address, 16, index, count));
        r.mSample.mRadiance = asfloat(data.rgb);
		r.mSample.mReplayPdfW = asfloat(data.w);

		#ifdef RECONNECTION
        r.mSample.mReconnectionVertex.mVertex = reinterpret<PackedVertex>(buf.Load4(PreIncrement(address, 16, index, count)));

        data = buf.Load4(PreIncrement(address, 16, index, count));
        r.mSample.mReconnectionVertex.mPackedRadiance = data.xy;
        r.mSample.mReconnectionVertex.mPackedG        = data[2];
        r.mSample.mReconnectionVertex.mPackedDirOut   = data[3];
		#endif

		#ifdef BIDIRECTIONAL
        r.mSample.mLightVertices = buf.Load(PreIncrement(address, 4, index, count));
		#endif
    }

    return r;
}
void StoreReservoir(RWReservoirBuffer buf, const PathReservoir r, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
    const uint count = extent.x * extent.y;
    if (index >= count) return;

    uint address = 0;
    buf.Store4(PreIncrement(address, 16, index, count), uint4(asuint(r.mW), asuint(r.mM), r.mSample.mRngSeed, r.mSample.mPackedRngIdx));

    if (r.mW > 0 && r.mM > 0) {
		buf.Store4(PreIncrement(address, 16, index, count), asuint(float4(r.mSample.mRadiance, r.mSample.mReplayPdfW)));

		#ifdef RECONNECTION
        buf.Store4(PreIncrement(address, 16, index, count), reinterpret<uint4>(r.mSample.mReconnectionVertex.mVertex));
        buf.Store4(PreIncrement(address, 16, index, count), uint4(r.mSample.mReconnectionVertex.mPackedRadiance, r.mSample.mReconnectionVertex.mPackedG, r.mSample.mReconnectionVertex.mPackedDirOut));
		#endif

		#ifdef BIDIRECTIONAL
		buf.Store(PreIncrement(address, 4, index, count), r.mSample.mLightVertices);
		#endif
    }
}
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
	#ifdef BIDIRECTIONAL
	uint mLightVertices;
	#else
	property uint mLightVertices { get { return 0; } set {} }
	#endif
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
    }
};
static const uint PathReservoirSize = sizeof(PathReservoir);

#define ReservoirBuffer   ByteAddressBuffer
#define RWReservoirBuffer RWByteAddressBuffer

PathReservoir LoadReservoir(ReservoirBuffer buf, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
	const uint count = extent.x*extent.y;

    PathReservoir r = PathReservoir();

    const float4 data0 = buf.Load<float4>(index * sizeof(float4));
	uint offset = count * sizeof(float4);
    r.mW = data0[0];
    r.mM = data0[1];
	r.mSample.mRngSeed = asuint(data0[2]);
	r.mSample.mPackedRngIdx = asuint(data0[3]);

    if (r.mW > 0) {
		const float4 data1 = buf.Load<float4>(offset + index * sizeof(float4));
        offset += count * sizeof(float4);
		r.mSample.mRadiance = data1.rgb;
		r.mSample.mReplayPdfW = data1.w;

		#ifdef BIDIRECTIONAL
		r.mSample.mLightVertices = buf.Load<uint>(offset + index * sizeof(uint));
		offset += count * sizeof(uint);
		#endif

		#ifdef RECONNECTION
        const float4 data2 = buf.Load<float4>(offset + index * sizeof(float4));
		offset += count * sizeof(float4);
        r.mSample.mReconnectionVertex.mVertex = reinterpret<PackedVertex>(data2);

        const uint4 data3 = buf.Load<uint4>(offset + index * sizeof(uint4));
        offset += count * sizeof(uint4);
        r.mSample.mReconnectionVertex.mPackedRadiance = data3.xy;
        r.mSample.mReconnectionVertex.mPackedG = data3[2];
        r.mSample.mReconnectionVertex.mPackedDirOut = data3[3];
		#endif
    }

    return r;
}
void StoreReservoir(RWReservoirBuffer buf, const PathReservoir r, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
	const uint count = extent.x*extent.y;

    buf.Store<float4>(index * sizeof(float4), float4(r.mW, r.mM, asfloat(r.mSample.mRngSeed), asfloat(r.mSample.mPackedRngIdx)));
	uint offset = count * sizeof(float4);

    if (r.mW > 0) {
		buf.Store<float4>(offset + index * sizeof(float4), float4(r.mSample.mRadiance, r.mSample.mReplayPdfW));
		offset += count * sizeof(float4);

		#ifdef BIDIRECTIONAL
		buf.Store<uint>(offset + index * sizeof(uint), r.mSample.mLightVertices);
		offset += count * sizeof(uint);
		#endif

		#ifdef RECONNECTION
        buf.Store<float4>(offset + index * sizeof(float4), reinterpret<float4>(r.mSample.mReconnectionVertex.mVertex));
        offset += count * sizeof(float4);

        buf.Store<uint4>(offset + index * sizeof(uint4), uint4(r.mSample.mReconnectionVertex.mPackedRadiance, r.mSample.mReconnectionVertex.mPackedG, r.mSample.mReconnectionVertex.mPackedDirOut));
		offset += count * sizeof(float4);
		#endif
    }
}
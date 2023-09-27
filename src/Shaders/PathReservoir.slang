#pragma once

#include "Common/PackedTypes.h"
#include "Random.slang"
#include "Reservoir.slang"


struct PackedLightVertex {
    PackedVertex mVertex;
    float3 mThroughput;
    uint mPackedDirection;
    uint mPacked;
    float mPdfA;
    float dVC;
    float dVCM;

    property uint mRngIdx {
        get { return BF_GET(mPacked, 8, 24); }
        set { BF_SET(mPacked, newValue, 8, 24); }
    }
    property uint mPathLength {
        get { return BF_GET(mPacked, 24, 8); }
        set { BF_SET(mPacked, newValue, 24, 8); }
    }

    property float3 mDirection {
        get { return UnpackNormal(mPackedDirection); }
        set { mPackedDirection = PackNormal(newValue); }
    }
};
static const uint PackedLightVertexSize = sizeof(PackedLightVertex);

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


struct PathSample : ReservoirSample {
    uint4 mReconnectionData[3];
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
    property bool mIsLightPath { get { return mLightVertices == mBounces + 1; } }

    __init() {
        mReconnectionData = { uint4(0xFFFFFFFF), 0, 0 };
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
        mReconnectionData = reinterpret<uint4[3]>(rcv);
		mHasRcv = rcvValid;
	}

    float GetTargetPdf() { return Luminance(mRadiance); }
};
static const uint PathSampleSize = sizeof(PathSample);

typedef Reservoir<PathSample> PathReservoir;
static const uint PathReservoirSize = sizeof(PathReservoir);


typedef RWByteAddressBuffer ReservoirBuffer;
ReservoirBuffer gPathReservoirs[2];

[[vk::push_constant]]
cbuffer PushConstants {
    uint gSpatialReusePass;
    uint gSpatialReuseSamples;
    float gSpatialReuseRadius;
    uint gReservoirIndex;
    uint2 gOutputSize;
    uint gRandomSeed;
};

#define gPathReservoirsIn gPathReservoirs[gReservoirIndex]
#define gPathReservoirsOut gPathReservoirs[gReservoirIndex ^ 1]

uint FlattenIndex(const uint2 id) {
	return id.y * gOutputSize.x + id.x;
}

uint PreIncrement(inout uint address, const uint stride, const uint index, const uint count) {
    uint val = address;
    address += count * stride;
    return val + index * stride;
}
PathReservoir LoadReservoir(ReservoirBuffer buf, const uint2 id) {
    const uint index = FlattenIndex(id);
    const uint count = gOutputSize.x * gOutputSize.y;
    if (index >= count) return PathReservoir();

    PathReservoir r = PathReservoir();

    uint address = 0;
    float2 data = buf.Load<float2>(PreIncrement(address, 8, index, count));
    r.mW = data[0];
    r.mM = data[1];

    if (r.mW > 0 && r.mM > 0)
        r.mSample = buf.Load<PathSample>(PreIncrement(address, PathSampleSize, index, count));
    else
        r.mSample = PathSample();

    return r;
}
void StoreReservoir(ReservoirBuffer buf, const uint2 id, const PathReservoir r) {
    const uint index = FlattenIndex(id);
    const uint count = gOutputSize.x * gOutputSize.y;
    if (index >= count) return;

    uint address = 0;
    buf.Store<float2>(PreIncrement(address, 8, index, count), float2(r.mW, r.mM));

    if (r.mW > 0 && r.mM > 0)
        buf.Store<PathSample>(PreIncrement(address, PathSampleSize, index, count), r.mSample);
}
PathReservoir LoadReservoir(const uint2 id) {
	return LoadReservoir(gPathReservoirsIn, id);
}
void StoreReservoir(const uint2 id, const PathReservoir r) {
	StoreReservoir(gPathReservoirsOut, id, r);
}

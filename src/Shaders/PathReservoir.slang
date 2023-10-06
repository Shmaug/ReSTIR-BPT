#pragma once

#include "Common/PackedTypes.h"
#include "Random.slang"
#include "Reservoir.slang"

struct PackedLightVertex {
    PackedVertex mVertex;
    uint2 mPackedThroughput;
    uint mPackedDirection;
    uint mPackedRngLength;
    float mSubpathPdf;
    float dVCM;
    float dVC;
    float dVM;

    property float3 mThroughput {
        get {
			return float3(
				BF_GET_UNORM(mPackedThroughput[0],  0, 10),
				BF_GET_UNORM(mPackedThroughput[0], 10, 11),
				BF_GET_UNORM(mPackedThroughput[0], 21, 11)) * asfloat(mPackedThroughput[1]);
		}
        set {
            float m = max3(newValue);
			if (m <= 0) m = 1;
            BF_SET_UNORM(mPackedThroughput[0], newValue.x/m,  0, 10);
            BF_SET_UNORM(mPackedThroughput[0], newValue.y/m, 10, 11);
            BF_SET_UNORM(mPackedThroughput[0], newValue.z/m, 21, 11);
            mPackedThroughput[1] = asuint(m);
        }
    }

    property uint mRngIdx {
        get { return BF_GET(mPackedRngLength, 8, 24); }
        set { BF_SET(mPackedRngLength, newValue, 8, 24); }
    }
    property uint mBounces {
        get { return BF_GET(mPackedRngLength, 0, 8); }
        set { BF_SET(mPackedRngLength, newValue, 0, 8); }
    }
    property uint mSubpathLength {
        get { return BF_GET(mPackedRngLength, 0, 8); }
        set { BF_SET(mPackedRngLength, newValue, 0, 8); }
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
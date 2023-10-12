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
    float mDist;
    float mCos;
    uint pad1;
    uint pad2;

	__init() {
        mRadiance = 0;
        mDist = 0;
        mCos = 0;
	}

    property float3 mDirOut {
		get { return UnpackNormal(mPackedDirOut); }
        set { mPackedDirOut = PackNormal(newValue); }
    }
};
static const uint ReconnectionVertexSize = sizeof(ReconnectionVertex);

enum PathShiftType {
	eReplay,
	eReconnection,
	eLightPath,
};

struct PathSample : ReservoirSample {
    ReconnectionVertex mReconnectionVertex;
    float3 mRadiance;
    float mReplayPdfW;
    uint mRngSeed;
    uint mPackedRngIdx;
    uint mPackedBounces;
    uint mPacked;

    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPackedRngIdx, 0, 16), BF_GET(mPackedRngIdx, 16, 16)); }
        set { BF_SET(mPackedRngIdx, newValue.x, 0, 16); BF_SET(mPackedRngIdx, newValue.y, 16, 16); }
    }
    property uint mBounces {
        get { return BF_GET(mPackedBounces, 0, 12); }
        set { BF_SET(mPackedBounces, newValue, 0, 12); }
    }
    property uint mReplayBounces {
        get { return BF_GET(mPackedBounces, 12, 12); }
        set { BF_SET(mPackedBounces, newValue, 12, 12); }
    }
    property uint mLightVertices {
        get { return BF_GET(mPackedBounces, 24, 8); }
        set { BF_SET(mPackedBounces, newValue, 24, 8); }
    }
    property PathShiftType mShiftType {
        get { return (PathShiftType)BF_GET(mPacked, 0, 4); }
        set { BF_SET(mPacked, (uint)newValue, 0, 4); }
    }

    __init() {
        mReconnectionVertex = ReconnectionVertex();
        mRadiance = 0;
        mReplayPdfW = 0;
        mRngSeed = 0;
        mPackedRngIdx = 0;
        mPackedBounces = 0;
        mPacked = 0;
    }
    __init(const float3 radiance, const float replayPdfW, const uint bounces, const uint replayBounces, const uint rngSeed, const uint2 rngIdx, const ReconnectionVertex rcv, const PathShiftType shiftType, const uint lightVertices) {
        mRadiance = radiance;
        mReplayPdfW = replayPdfW;
        mBounces = bounces;
        mReplayBounces = replayBounces;
        mRngSeed = rngSeed;
        mRngIdx = rngIdx;
        mReconnectionVertex = rcv;
        mShiftType = shiftType;
        mLightVertices = lightVertices;
	}

    float GetTargetPdf() { return Luminance(mRadiance); }
};
static const uint PathSampleSize = sizeof(PathSample);

typedef Reservoir<PathSample> PathReservoir;
static const uint PathReservoirSize = sizeof(PathReservoir);
#pragma once

#include "Common/PackedTypes.h"
#include "Random.slang"
#include "Reservoir.slang"

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
	eMerge,
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
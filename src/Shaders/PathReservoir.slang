#pragma once

#include "Random.slang"
#include "Common/PackedTypes.h"

struct ReconnectionVertex {
    PackedVertex mVertex;
    uint2 mPackedRadiance;
    uint mPackedG;
    float mPrefixPdfW;
    uint mPackedDirOut;

	__init() {
        mPackedRadiance = 0;
        mPackedG = 0;
		mPrefixPdfW = 0;
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
    float mPdfW;
    uint mRngSeed;
    uint mPackedRngIdx;
    #ifdef RECONNECTION
    ReconnectionVertex mReconnectionVertex;
	#else
    property ReconnectionVertex mReconnectionVertex { get { return ReconnectionVertex(); } set { } }
	#endif

    property uint mBounces {
        get { return BF_GET(mPackedRngIdx, 24, 7); }
        set { BF_SET(mPackedRngIdx, newValue, 24, 7); }
    }
    property bool mIsNEE {
        get { return (bool)BF_GET(mPackedRngIdx, 31, 1); }
        set { BF_SET(mPackedRngIdx, (uint)newValue, 31, 1); }
    }
    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPackedRngIdx, 0, 12), BF_GET(mPackedRngIdx, 12, 12)); }
        set { BF_SET(mPackedRngIdx, newValue.x, 0, 12); BF_SET(mPackedRngIdx, newValue.y, 12, 12); }
	}

    __init() {
        mRadiance = 0;
		mPdfW = 0;
        mRngSeed = 0;
        mPackedRngIdx = 0;
        mReconnectionVertex = ReconnectionVertex();
    }
    __init(float3 radiance, float pdfW, uint bounces, bool isNee, const uint rngSeed, const uint2 rngIdx, ReconnectionVertex rcv) {
		mRadiance = radiance;
        mPdfW = pdfW;
        mRngSeed = rngSeed;
        mRngIdx = rngIdx;
        mBounces = bounces;
        mIsNEE = isNee;
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

#ifdef RESERVOIR_STRUCTURED_BUFFERS

#define ReservoirBuffer   StructuredBuffer<PathReservoir>
#define RWReservoirBuffer RWStructuredBuffer<PathReservoir>

PathReservoir LoadReservoir(ReservoirBuffer buf, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
    return buf[index];
}
void StoreReservoir(RWReservoirBuffer buf, const PathReservoir r, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
    buf[index] = r;
}

#else

#define ReservoirBuffer   ByteAddressBuffer
#define RWReservoirBuffer RWByteAddressBuffer

PathReservoir LoadReservoir(ReservoirBuffer buf, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
    PathReservoir r = PathReservoir();
    const float4 data0 = buf.Load<float4>(index * sizeof(float4));
    r.mW = data0[0];
    r.mM = data0[1];
	r.mSample.mRngSeed = asuint(data0[2]);
	r.mSample.mPackedRngIdx = asuint(data0[3]);

    if (r.mW > 0) {
		uint offset = extent.x * extent.y * sizeof(float4);
		const float4 data1 = buf.Load<float4>(offset + index * sizeof(float4));
        offset += extent.x * extent.y * sizeof(float4);
		r.mSample.mRadiance = data1.rgb;
		r.mSample.mPdfW = data1.w;

		#ifdef RECONNECTION
        const float4 data2 = buf.Load<float4>(offset + index * sizeof(float4));
		offset += extent.x * extent.y * sizeof(float4);
        r.mSample.mReconnectionVertex.mVertex = reinterpret<PackedVertex>(data2);

        const uint4 data3 = buf.Load<uint4>(offset + index * sizeof(uint4));
        offset += extent.x * extent.y * sizeof(uint4);
        r.mSample.mReconnectionVertex.mPackedRadiance = data3.xy;
        r.mSample.mReconnectionVertex.mPackedG = data3[2];
        r.mSample.mReconnectionVertex.mPrefixPdfW = asfloat(data3[3]);

        r.mSample.mReconnectionVertex.mPackedDirOut = buf.Load<uint>(offset + index * sizeof(uint));
		#endif
    }

    return r;
}
void StoreReservoir(RWReservoirBuffer buf, const PathReservoir r, const uint2 id, const uint2 extent) {
    const uint index = id.y * extent.x + id.x;
    buf.Store<float4>(index * sizeof(float4), float4(r.mW, r.mM, asfloat(r.mSample.mRngSeed), asfloat(r.mSample.mPackedRngIdx)));
    if (r.mW > 0) {
		uint offset = extent.x * extent.y * sizeof(float4);
		buf.Store<float4>(offset + index * sizeof(float4), float4(r.mSample.mRadiance, r.mSample.mPdfW));

		#ifdef RECONNECTION
        offset += extent.x * extent.y * sizeof(float4);
        buf.Store<float4>(offset + index * sizeof(float4), reinterpret<float4>(r.mSample.mReconnectionVertex.mVertex));

		offset += extent.x * extent.y * sizeof(float4);
        buf.Store<uint4>(offset + index * sizeof(uint4), uint4(r.mSample.mReconnectionVertex.mPackedRadiance, r.mSample.mReconnectionVertex.mPackedG, asuint(r.mSample.mReconnectionVertex.mPrefixPdfW)));

		offset += extent.x * extent.y * sizeof(float4);
        buf.Store<uint>(offset + index * sizeof(uint), r.mSample.mReconnectionVertex.mPackedDirOut);
		#endif
    }
}

#endif

#ifdef OUTPUT_RADIANCE_SHADER

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
};

RWTexture2D<float4> gRadiance;
ReservoirBuffer gPathReservoirsIn;

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    const PathReservoir r = LoadReservoir(gPathReservoirsIn, id, gOutputSize);
    if (r.mW > 0)
        gRadiance[id] += float4(r.mSample.mRadiance * r.mW, 0);
}

#endif
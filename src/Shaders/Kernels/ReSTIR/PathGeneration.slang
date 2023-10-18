#pragma once

#include "BRDF.slang"
#include "HashGrid.slang"
#include "Intersection.slang"
#include "Light.slang"
#include "PathReservoir.slang"

uint FlattenIndex(const uint2 id) {
    return id.y * gOutputSize.x + id.x;
}
uint PreIncrement(inout uint address, const uint stride, const uint index, const uint count) {
    uint val = address;
    address += count * stride;
    return val + index * stride;
}
PathReservoir LoadReservoir(RWByteAddressBuffer buf, const uint index, const uint count) {
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
void StoreReservoir(RWByteAddressBuffer buf, const uint index, const uint count, const PathReservoir r) {
    uint address = 0;
    buf.Store<float2>(PreIncrement(address, 8, index, count), float2(r.mW, r.mM));

    if (r.mW > 0 && r.mM > 0)
        buf.Store<PathSample>(PreIncrement(address, PathSampleSize, index, count), r.mSample);
}
PathReservoir LoadReservoir(RWByteAddressBuffer buf, const uint2 id) {
    const uint index = FlattenIndex(id);
    const uint count = gOutputSize.x * gOutputSize.y;
    if (index >= count) return PathReservoir();
    return LoadReservoir(buf, index, count);
}
void StoreReservoir(RWByteAddressBuffer buf, const uint2 id, const PathReservoir r) {
    const uint index = FlattenIndex(id);
    const uint count = gOutputSize.x * gOutputSize.y;
    if (index >= count) return;
    StoreReservoir(buf, index, count, r);
}


struct PackedLightVertex {
    PackedVertex mVertex;
    uint2 mPackedThroughput;
    uint mPackedDirection;
    uint mPacked;
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
    property float3 mDirection {
        get { return UnpackNormal(mPackedDirection); }
        set { mPackedDirection = PackNormal(newValue); }
    }

    property uint2 mRngIdx {
        get { return uint2(BF_GET(mPacked, 0, 12), BF_GET(mPacked, 12, 12)); }
        set { BF_SET(mPacked, newValue.x, 0, 12); BF_SET(mPacked, newValue.y, 12, 12); }
    }
    property uint mBounces {
        get { return BF_GET(mPacked, 24, 8); }
        set { BF_SET(mPacked, newValue, 24, 8); }
    }
};
static const uint PackedLightVertexSize = sizeof(PackedLightVertex);

[[vk::push_constant]]
cbuffer PushConstants {
    uint gSpatialReusePass;
    uint gSpatialReuseSamples;
    float gSpatialReuseRadius;
    uint gReservoirIndex;
    uint2 gOutputSize;
    uint gRandomSeed;
    float gDirectLightProb;
};

static const float4 gSceneSphere = float4(gScene.mSceneMax + gScene.mSceneMin, length(gScene.mSceneMax - gScene.mSceneMin)) / 2;

RWByteAddressBuffer gPathReservoirs[2];
#define gPathReservoirsIn gPathReservoirs[gReservoirIndex]
#define gPathReservoirsOut gPathReservoirs[gReservoirIndex ^ 1]

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif

#ifdef BIDIRECTIONAL
static const bool gBidirectional = true;
RWStructuredBuffer<PackedLightVertex> gLightVertices;
RWStructuredBuffer<uint> gLightVertexCount;
#ifdef LIGHT_TRACE_RESERVOIRS
RWByteAddressBuffer gLightTraceReservoirs;
ParameterBlock<HashGrid<uint>> gLightTraceReservoirGrid;
#endif
#ifdef VERTEX_MERGING
ParameterBlock<HashGrid<uint>> gLightVertexGrid;
#define gMergeRadius (gLightVertexGrid.mMinCellSize * 0.5)
#else
#define gMergeRadius 0
#endif // VERTEX_MERGING
#else // BIDIRECTIONAL
static const bool gBidirectional = false;
#define gMergeRadius 0
#endif

float Mis(float a) { return a * a; }
float Mis(float a, float b) {
    a = Mis(a);
    return SafeDivide(a, a + Mis(b));
}

uniform float3 gCameraPosition;
uniform float4x4 gWorldToCamera;
uniform float4x4 gProjection;

uniform float gReconnectionDistance;
uniform float gReconnectionRoughness;
uniform uint gMaxBounces;
uniform uint gLightSubpathCount;
uniform float gCameraImagePlaneDist;

uniform uint gDebugTotalVertices;
uniform uint gDebugLightVertices;

static const float gEtaVCM = sqr(gMergeRadius) * M_PI * gLightSubpathCount;
#if defined(BIDIRECTIONAL) && defined(VERTEX_MERGING)
static const float gMisVcWeightFactor = Mis(1 / gEtaVCM);
static const float gMisVmWeightFactor = Mis(gEtaVCM);
#else
static const float gMisVmWeightFactor = 0;
static const float gMisVcWeightFactor = 0;
#endif

ReflectanceEvalRecord EvaluateReflectance(const PathVertex vertex, const float3 localDirIn, const float3 localDirOut, const bool adjoint) {
    ReflectanceEvalRecord r = EvalBrdf(vertex.mMaterial, localDirIn, localDirOut, adjoint);
	r.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, adjoint);
    return r;
}
DirectionSampleRecord SampleDirection(const PathVertex vertex, const float4 rnd, const float3 localDirIn, const bool adjoint) {
    DirectionSampleRecord r = SampleBrdf(vertex.mMaterial, rnd, localDirIn, adjoint);
    r.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, r.mDirection, adjoint);
    return r;
}

struct PathState<let bShift : bool> {
	float3 throughput;
    float pdfW;

    float3 localDirIn;
    uint packedBounces;

    uint seed;
    uint packedId;
    float dVCM;
    float dVC;

    PackedVertex rcv;
    float3 throughputAtRcv;
    uint rcvPackedDirOut;
    float pdfAtRcv;
    float rcvCos;
    float rcvDist;

    float dVM;

    property float3 rcvDirOut {
        get { return UnpackNormal(rcvPackedDirOut); }
        set { rcvPackedDirOut = PackNormal(newValue); }
	}

    property uint bounces {
        get { return BF_GET(packedBounces, 0, 16); }
        set { BF_SET(packedBounces, newValue, 0, 16); }
    }
    property uint rcvBounces {
        get { return BF_GET(packedBounces, 16, 15); }
        set { BF_SET(packedBounces, newValue, 16, 15); }
    }
    property bool isLightPath {
        get { return (bool)BF_GET(packedBounces, 31, 1); }
        set { BF_SET(packedBounces, (uint)newValue, 31, 1); }
    }
    property bool rcvFound {
        get { return pdfAtRcv > 0; }
    }
    property uint2 id {
        get { return uint2(packedId & 0xFFFF, packedId >> 16); }
        set { packedId = (newValue.y << 16) | newValue.x; }
	}
};


enum PerVertexRandomNumbers {
	eDirectLight,
    eDirection,
    eBidirectionalConnection,
    eTerminate,
    eMerge,
    eNumRandomPerVertex
};
uint4 GetRandomUint<let bShift : bool>(const PathState<bShift> s, const PerVertexRandomNumbers type) {
	RandomSampler rng = RandomSampler(s.seed, s.id);
	rng.Skip(s.bounces * (uint)PerVertexRandomNumbers::eNumRandomPerVertex + (uint)type);
	return rng.Next();
}
float4 GetRandomFloat<let bShift : bool>(const PathState<bShift> s, const PerVertexRandomNumbers type) {
	return UnormToFloat(GetRandomUint<bShift>(s, type));
}

// ---------------------------------------------------------------------------------------------------------
// ----------------------------------------- Connection stategies ------------------------------------------
// ---------------------------------------------------------------------------------------------------------

PathSample SampleNextVertex<let bShift : bool>(inout PathState<bShift> s, inout PathVertex vertex, out float weight) {
	const float prevPdf = s.pdfW;
	const bool diffuse = IsDiffuse(vertex);

    weight = 0;

    const float contProb = diffuse ? GetContinuationProbability(vertex) : 1;
    if (!bShift && contProb < 1 && (contProb == 0 || GetRandomFloat(s, PerVertexRandomNumbers::eTerminate).x > contProb)) {
		s.throughput = 0;
		return PathSample();
	}

    DirectionSampleRecord r = SampleDirection(vertex, GetRandomFloat(s, PerVertexRandomNumbers::eDirection), s.localDirIn, s.isLightPath);
    r.mFwdPdfW *= contProb;
    r.mRevPdfW *= contProb;

    s.throughput *= r.mReflectance;
    s.pdfW *= r.mFwdPdfW;
	if (all(s.throughput <= 0))
		return PathSample();

    if (gBidirectional) {
		if (diffuse) {
            s.dVC = Mis(abs(r.mDirection.z) / r.mFwdPdfW) * (s.dVC * Mis(r.mRevPdfW) + s.dVCM + gMisVmWeightFactor);
            s.dVM = Mis(abs(r.mDirection.z) / r.mFwdPdfW) * (s.dVM * Mis(r.mRevPdfW) + s.dVCM * gMisVcWeightFactor + 1);
			s.dVCM = Mis(1.f / r.mFwdPdfW);
		} else {
            s.dVCM = 0.f;
            s.dVC *= Mis(abs(r.mDirection.z));
            s.dVM *= Mis(abs(r.mDirection.z));
        }
    }

    r.mDirection = vertex.ToWorld(r.mDirection);

    if (!bShift && !s.isLightPath && s.rcvFound && s.rcvBounces == s.bounces-1) {
		s.rcvDirOut = r.mDirection;
        s.throughputAtRcv = s.throughput;
	}

    const bool prevReconnectable = r.mRoughness >= gReconnectionRoughness;

    float directPdfA, dist;
    vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, r.mDirection), r.mDirection), true, directPdfA, dist);

	s.bounces++;

    float3 le = vertex.mMaterial.Emission();
    float lightPdfW = directPdfA;
    float emissionPdfW = directPdfA;
	if (vertex.mIsSurface) {
        s.localDirIn = vertex.ToLocal(-r.mDirection);

		const float cosIn = s.localDirIn.z;
        lightPdfW = directPdfA * sqr(dist) / abs(cosIn);
		if (cosIn <= 0) le = 0;

        emissionPdfW *= CosHemispherePdfW(cosIn);

        if (gBidirectional) {
			s.dVCM *= Mis(sqr(dist) / abs(cosIn));
			s.dVC  /= Mis(abs(cosIn));
			s.dVM  /= Mis(abs(cosIn));
		}

        if (!bShift && !s.isLightPath && !s.rcvFound) {
			// store reconnection vertex if one hasn't been found yet
			const bool reconnectable = dist >= gReconnectionDistance && (any(le > 0) || IsDiffuse(vertex) || vertex.mMaterial.Roughness() >= gReconnectionRoughness);
            if (prevReconnectable && reconnectable) {
				s.rcv = PackVertex(vertex);
				s.rcvCos = abs(cosIn);
				s.rcvDist = dist;
				s.rcvBounces = s.bounces-1;
                s.pdfAtRcv = prevPdf;
            }
        }
    } else {
        directPdfA = ConcentricDiscPdfA() / sqr(gSceneSphere.w);
    }

	if (any(le > 0) && !s.isLightPath)  {
		// mis weight
		weight = 1;
        if (diffuse) {
			if (gBidirectional) {
				const float wCamera = Mis(directPdfA) * s.dVCM + Mis(emissionPdfW) * s.dVC;
                weight = 1 / (1 + wCamera);
            } else if (gSampleLights)
				weight = Mis(r.mFwdPdfW, lightPdfW);
        }

		// divide by pdf
        weight /= s.pdfW;

        const bool hasRcv = !bShift && s.rcvFound;
        ReconnectionVertex rcv;
        if (hasRcv) {
			rcv.mVertex = s.rcv;
			rcv.mCos  = s.rcvCos;
			rcv.mDist = s.rcvDist;
			if (s.rcvBounces == s.bounces-1)
                rcv.mRadiance = le;
            else {
				rcv.mDirOut = s.rcvDirOut;
				rcv.mRadiance = le * float3(
					SafeDivide(s.throughput.r, s.throughputAtRcv.r),
					SafeDivide(s.throughput.g, s.throughputAtRcv.g),
					SafeDivide(s.throughput.b, s.throughputAtRcv.b));
				}
        }

        return PathSample(
            s.throughput * le,
            hasRcv ? s.pdfAtRcv : s.pdfW,
            s.bounces, hasRcv ? s.rcvBounces : s.bounces,
            s.seed, s.id,
            hasRcv ? rcv : ReconnectionVertex(),
            hasRcv ? PathShiftType::eReconnection : PathShiftType::eReplay,
			0);
    }
	return PathSample();
}

PathSample ConnectToLight<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, out float weight) {
    weight = 0;

    float lightPdf;
    const PathVertex l = SampleEmission(GetRandomFloat(s, PerVertexRandomNumbers::eDirectLight), lightPdf);

	float3 le = l.mMaterial.Emission();
	if (!any(le > 0))
		return PathSample();

    float3 toLight;
    float dist, cosLight, directPdfW;
    if (l.mIsSurface) {
        toLight = l.mPosition - vertex.mPosition;
        dist = length(toLight);
        toLight /= dist;
        cosLight = max(0, -dot(toLight, l.mShadingNormal));
        if (cosLight <= 0)
            return PathSample();
		directPdfW = lightPdf * sqr(dist) / cosLight;
    } else {
        toLight = l.mPosition;
        dist = POS_INFINITY;
        cosLight = 1;
		directPdfW = lightPdf;
    }

    const float3 localDirOut = vertex.ToLocal(toLight);
    ReflectanceEvalRecord f = EvaluateReflectance(vertex, s.localDirIn, localDirOut, false);
    const float contProb = GetContinuationProbability(vertex);
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    const float3 contrib = s.throughput * le * f.mReflectance;
    if (!any(contrib > 0))
		return PathSample();

	if (Occluded(vertex, toLight, dist))
		return PathSample();

	// mis weight
    if (gBidirectional) {
		const float emissionPdfW = l.mIsSurface ? lightPdf * CosHemispherePdfW(cosLight) : (lightPdf * ConcentricDiscPdfA() / sqr(gSceneSphere.w));
		const float wLight = Mis(f.mFwdPdfW / directPdfW);
        const float wCamera = Mis(emissionPdfW * abs(localDirOut.z) / (directPdfW * cosLight)) * (gMisVmWeightFactor + s.dVCM + s.dVC * Mis(f.mRevPdfW));
        weight = 1 / (wLight + 1 + wCamera);
    } else {
        weight = Mis(directPdfW, f.mFwdPdfW);
    }

	// divide by pdf
    weight /= s.pdfW * directPdfW;

    const bool hasRcv = !bShift && s.rcvFound;
    ReconnectionVertex rcv;
    if (hasRcv) {
        rcv.mVertex = s.rcv;
        rcv.mCos  = s.rcvCos;
        rcv.mDist = s.rcvDist;
		if (s.rcvBounces == s.bounces-1) {
			rcv.mDirOut = toLight;
            rcv.mRadiance = le;
        } else {
            rcv.mDirOut = s.rcvDirOut;
            rcv.mRadiance = float3(
                SafeDivide(contrib.r, s.throughputAtRcv.r),
                SafeDivide(contrib.g, s.throughputAtRcv.g),
                SafeDivide(contrib.b, s.throughputAtRcv.b));
        }
    } else {
        rcv.mVertex = PackVertex(l);
        rcv.mRadiance = le;
        rcv.mDirOut = 0;
        rcv.mCos = cosLight;
        rcv.mDist = dist;
	}

    return PathSample(
        contrib,
        hasRcv ? s.pdfAtRcv : s.pdfW,
        s.bounces + 1, hasRcv ? s.rcvBounces : s.bounces,
        s.seed, s.id,
        rcv,
        PathShiftType::eReconnection,
		1);
}
PathSample ConnectToLightSubpath<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, const PackedLightVertex v, out float weight) {
    weight = 0;

    const uint bounces = s.bounces + v.mBounces + 2;
    if (bounces > gMaxBounces)
        return PathSample();

    float cosLight, dist;
    float3 toLight;
    ReflectanceEvalRecord r;
    {
        const PathVertex lightVertex = UnpackVertex(v.mVertex);
        toLight = lightVertex.mPosition - vertex.mPosition;
        dist = length(toLight);
        toLight /= dist;

        const float3 localDirOut = lightVertex.ToLocal(-toLight);
        cosLight = abs(localDirOut.z);
        r = EvaluateReflectance(lightVertex, lightVertex.ToLocal(v.mDirection), localDirOut, true);
        const float contProb = GetContinuationProbability(lightVertex);
        r.mFwdPdfW *= contProb;
        r.mRevPdfW *= contProb;
    }
    if (all(r.mReflectance <= 0))
        return PathSample();

    float cosCam;
    ReflectanceEvalRecord f;
    {
        const float3 localDirOut = vertex.ToLocal(toLight);
        cosCam = abs(localDirOut.z);
        f = EvaluateReflectance(vertex, s.localDirIn, localDirOut, false);
        const float contProb = GetContinuationProbability(vertex);
        f.mFwdPdfW *= contProb;
        f.mRevPdfW *= contProb;
    }
    if (all(f.mReflectance <= 0))
        return PathSample();

    if (Occluded(vertex, toLight, dist))
        return PathSample();

    const float3 contrib = (s.throughput * v.mThroughput) * (f.mReflectance * (r.mReflectance / cosLight));

    // Convert pdfs to area pdf
    const float invDist2 = 1 / sqr(dist);
    const float cameraBsdfDirPdfA = f.mFwdPdfW * cosLight * invDist2;
    const float lightBsdfDirPdfA  = r.mFwdPdfW * cosCam   * invDist2;

	// mis weight
    const float wLight = Mis(cameraBsdfDirPdfA) * (gMisVmWeightFactor + v.dVCM + v.dVC * Mis(r.mRevPdfW));
    const float wCamera = Mis(lightBsdfDirPdfA) * (gMisVmWeightFactor + s.dVCM + s.dVC * Mis(f.mRevPdfW));
    weight = 1 / (wLight + 1.f + wCamera);

	// divide by pdf
    weight /= s.pdfW * (v.mSubpathPdf * sqr(dist) / cosLight);

    const bool hasRcv = !bShift && s.rcvFound;
    ReconnectionVertex rcv;
    if (hasRcv) {
        rcv.mVertex = s.rcv;
        rcv.mCos  = s.rcvCos;
        rcv.mDist = s.rcvDist;
        if (s.rcvBounces == s.bounces - 1) {
            rcv.mDirOut = toLight;
            rcv.mRadiance = v.mThroughput * (r.mReflectance / cosLight);
        } else {
            rcv.mDirOut = s.rcvDirOut;
            rcv.mRadiance = float3(
                SafeDivide(contrib.r, s.throughputAtRcv.r),
                SafeDivide(contrib.g, s.throughputAtRcv.g),
                SafeDivide(contrib.b, s.throughputAtRcv.b));
        }
    } else {
        rcv.mVertex = v.mVertex;
        rcv.mRadiance = v.mThroughput;
        rcv.mDirOut = v.mDirection;
        rcv.mCos = cosLight;
        rcv.mDist = dist;
    }

    return PathSample(
        contrib,
        hasRcv ? s.pdfAtRcv : s.pdfW,
        bounces, hasRcv ? s.rcvBounces : s.bounces,
        s.seed, s.id,
        rcv,
		PathShiftType::eReconnection,
		v.mBounces + 2);
}
PathSample MergeWithLightVertex<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, const PackedLightVertex v, out float weight) {
    weight = 0;

    const uint bounces = s.bounces + v.mBounces + 1;
    if (bounces > gMaxBounces)
        return PathSample();

    if (length(v.mVertex.mPosition - vertex.mPosition) >= gMergeRadius)
		return PathSample();

	ReflectanceEvalRecord f = EvaluateReflectance(vertex, s.localDirIn, vertex.ToLocal(v.mDirection), false);
	const float contProb = GetContinuationProbability(vertex);
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

	const float wLight = v.dVCM * gMisVcWeightFactor + v.dVM * Mis(f.mFwdPdfW);
	const float wCamera = s.dVCM * gMisVcWeightFactor + s.dVM * Mis(f.mRevPdfW);
    const float misWeight = 1.f / (wLight + 1.f + wCamera);

	#ifdef VERTEX_MERGING_ONLY
    weight = 1;
	#else
    weight = misWeight;
	#endif

	weight /= v.mSubpathPdf;

	return PathSample(
        s.throughput * v.mThroughput * f.mReflectance,
        s.pdfW,
        bounces, s.bounces,
        s.seed, s.id,
        reinterpret<ReconnectionVertex>(v),
        PathShiftType::eMerge,
        v.mBounces + 2);
}

PathSample ConnectToBasePath<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, const PathSample basePath) {
    if (!(IsDiffuse(vertex) || vertex.mMaterial.Roughness() >= gReconnectionRoughness))
        return PathSample();

    const PathVertex baseRcv = UnpackVertex(basePath.mReconnectionVertex.mVertex);

	float3 toRcv;
	float dist;
	if (baseRcv.mIsSurface) {
		toRcv = baseRcv.mPosition - vertex.mPosition;
		dist = length(toRcv);
		toRcv /= dist;
	} else {
		toRcv = baseRcv.mPosition;
		dist = POS_INFINITY;
	}

	if (dist < gReconnectionDistance)
		return PathSample();

    float3 le = s.throughput * basePath.mReconnectionVertex.mRadiance;
    if (!any(le > 0)) {
        DebugPrintf("Reconnection failed: zero le\n");
        return PathSample();
    }

    le *= EvaluateReflectance(vertex, s.localDirIn, vertex.ToLocal(toRcv), false).mReflectance;
    if (!any(le > 0)) {
        DebugPrintf("Reconnection failed: zero vertex reflectance\n");
        return PathSample();
    }

    if (basePath.mReplayBounces + 1 < basePath.mBounces)
        le *= EvaluateReflectance(baseRcv, baseRcv.ToLocal(-toRcv), baseRcv.ToLocal(basePath.mReconnectionVertex.mDirOut), false).mReflectance;
    else if (dot(toRcv, baseRcv.mShadingNormal) > 0)
		return PathSample();

    if (!any(le > 0)) {
        DebugPrintf("Reconnection failed: zero rcv reflectance (%f %f)\n", baseRcv.ToLocal(-toRcv).z, baseRcv.ToLocal(basePath.mReconnectionVertex.mDirOut).z);
        return PathSample();
    }
    if (Occluded(vertex, toRcv, dist)) {
        DebugPrintf("Reconnection failed: occluded\n");
        return PathSample();
    }

	PathSample ps = basePath;
	ps.mRadiance = le;
	ps.mReplayPdfW = s.pdfW;
	ps.mReconnectionVertex.mCos = abs(dot(baseRcv.mShadingNormal, toRcv));
    ps.mReconnectionVertex.mDist = dist;
	return ps;
}
PathSample ConnectToCamera<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, out int2 pixelCoord, out float weight) {
    weight = 0;
    pixelCoord = -1;

	#ifdef gDebugPathLengths
	if (gDebugTotalVertices != s.bounces + 3 || gDebugLightVertices != s.bounces + 2)
		return PathSample();
	#endif

    const float3 cameraPos = TransformPoint(gWorldToCamera, vertex.mPosition);
    const float4 clip = mul(gProjection, float4(cameraPos, 1));
    pixelCoord = int2(gOutputSize * ((clip.xy / clip.w) * .5 + .5));
    if (any(pixelCoord < 0) || any(pixelCoord >= gOutputSize) || cameraPos.z > 0)
        return PathSample();

    float3 toCamera = gCameraPosition - vertex.mPosition;
    const float dist = length(toCamera);
    toCamera /= dist;

    if (Occluded(vertex, toCamera, dist))
        return PathSample();

    const float3 localDirOut = vertex.ToLocal(toCamera);
    const float cosOut = abs(localDirOut.z);

    ReflectanceEvalRecord f = EvaluateReflectance(vertex, bShift ? localDirOut : s.localDirIn, bShift ? s.localDirIn : localDirOut, !bShift);
    const float contProb = GetContinuationProbability(vertex);
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    const float cosAtCamera = abs(normalize(cameraPos).z);

    // Compute pdf conversion factor from image plane area to surface area
    const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
    const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;
    const float cameraPdfA = imageToSolidAngleFactor * cosOut / sqr(dist);

    const float3 contrib = s.throughput * (bShift ? f.mReflectance : (f.mReflectance / cosOut));
	if (!any(contrib > 0))
		return PathSample();

	const float samplePdf = s.pdfW * (1 / cameraPdfA) * gLightSubpathCount;
	if (!(samplePdf > 0))
		return PathSample();

	#ifdef gLightTraceOnly
	const float misWeight = 1;
	#else
	const float wLight = Mis(cameraPdfA / gLightSubpathCount) * (gMisVmWeightFactor + s.dVCM + s.dVC * Mis(f.mRevPdfW));
	const float misWeight = 1.f / (wLight + 1.f);
	#endif

    weight = misWeight / samplePdf;

    return PathSample(
        contrib,
        1,
        s.bounces + 1, max(s.bounces,2)-2,
        s.seed, s.id,
        ReconnectionVertex(),
        PathShiftType::eLightPath,
        s.bounces + 2);
}

// ---------------------------------------------------------------------------------------------------------
// -------------------------------------------- Path generation --------------------------------------------
// ---------------------------------------------------------------------------------------------------------

PathState<bShift> InitializeCameraPath<let bShift : bool>(const uint seed, const uint2 id, const PathVertex vertex, const float3 dir) {
    PathState<bShift> s;
    s.localDirIn = vertex.ToLocal(-dir);
    s.seed = seed;
    s.id = id;
    s.throughput = 1;
    s.pdfW = 1;
    s.bounces = 0;
    s.rcvBounces = 0;
    s.isLightPath = false;
    if (gBidirectional) {
        const float cosAtCamera = abs(normalize(TransformVector(gWorldToCamera, dir)).z);
        const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
        const float cameraPdfW = sqr(imagePointToCameraDist) / cosAtCamera;
        s.dVCM = Mis(gLightSubpathCount / cameraPdfW);
        s.dVC = 0;
        s.dVM = 0;

        const float cosIn = abs(s.localDirIn.z);
		s.dVCM *= Mis(sqr(length(gCameraPosition - vertex.mPosition)) / cosIn);
    }
    return s;
}
PathState<bShift> InitializeLightPath<let bShift : bool>(const uint seed, const uint2 id, out PathVertex vertex) {
    PathState<bShift> s;
    s.seed = seed;
    s.id = id;
    s.isLightPath = true;

    s.bounces = -1; // for rng
    vertex = SampleEmission(GetRandomFloat(s, PerVertexRandomNumbers::eDirectLight), s.pdfW);
    s.bounces = 0;
    s.rcvBounces = 0;
    s.throughput = vertex.mMaterial.Emission();

    return s;
}
void SampleFirstLightVertex<let bShift : bool>(inout PathState<bShift> s, inout PathVertex vertex) {
    const bool isFinite = vertex.mIsSurface;

    const float directPdfA = s.pdfW;
	float emissionPdfW = directPdfA;
	float cosLight;

    float3 origin;
    float3 direction;

    s.bounces = -1; // for rng
    const float2 dirRng = GetRandomFloat(s, PerVertexRandomNumbers::eDirection).xy;
    s.bounces = 0;

    if (isFinite) {
		// area light
		const float3 dir = SampleCosHemisphere(dirRng.xy);
        cosLight = dir.z;
		direction = vertex.ToWorld(dir);
		origin = OffsetRayOrigin(vertex, direction);
		emissionPdfW *= CosHemispherePdfW(cosLight);
		s.throughput *= cosLight;
	} else {
		// environment light
		const float2 xy = SampleConcentricDisc(dirRng.xy);
		const float3x3 onb = MakeOrthonormal(vertex.mPosition);
		origin = gSceneSphere.xyz + gSceneSphere.w * (onb[0] * xy.x + onb[1] * xy.y + vertex.mPosition);
		direction = -vertex.mPosition;
		emissionPdfW *= ConcentricDiscPdfA() / sqr(gSceneSphere.w);
		cosLight = 1;
	}

	s.pdfW = emissionPdfW;
    s.dVCM = Mis(directPdfA / emissionPdfW);
    s.dVC = isFinite ? Mis(cosLight / emissionPdfW) : 0;
    s.dVM = s.dVC * gMisVcWeightFactor;

    float lightPdf, dist;
    vertex = TraceRay(MakeRay(origin, direction), true, lightPdf, dist);
    if (vertex.mIsSurface) {
        s.localDirIn = vertex.ToLocal(-direction);
        const float cosIn = abs(s.localDirIn.z);
        if (isFinite)
            s.dVCM *= Mis(sqr(dist));
        s.dVCM /= Mis(cosIn);
        s.dVC  /= Mis(cosIn);
        s.dVM  /= Mis(cosIn);
    } else
		s.throughput = 0;
}

void ProcessCanonicalSample(inout PathReservoir r, const float mergeRnd, const PathSample ps, const float weight) {
    if (weight <= 0)
        return;

	#ifdef gDebugPathLengths
	if (gDebugTotalVertices != ps.mBounces+2 || gDebugLightVertices != ps.mLightVertices)
        return;
	#endif

    PathReservoir c = PathReservoir(ps, weight, 1);
	c.PrepareMerge();
	r.Merge(mergeRnd, c);
}
PathReservoir SampleCameraPath(PathVertex vertex, const uint2 id) {
    PathReservoir r = PathReservoir();
    PathState<false> s = InitializeCameraPath<false>(gRandomSeed, id, vertex, normalize(vertex.mPosition - gCameraPosition));

    while (s.bounces + 1 <= gMaxBounces && any(s.throughput > 0)) {
        const float4 mergeRnds = GetRandomFloat(s, PerVertexRandomNumbers::eMerge);

        if (IsDiffuse(vertex)) {
            float weight = 0;

			#ifdef BIDIRECTIONAL

			#ifdef VERTEX_MERGING
            {
                int3 o = lerp(int3(0, 0, 0), int3(-1, -1, -1), frac(vertex.mPosition / gLightVertexGrid.mMinCellSize) < 0.5);
				float cameraW = 1 / (s.pdfW * gEtaVCM);
				for (int i = 0; i < 8; i++) {
					const uint2 range = gLightVertexGrid.GetCellDataRange(gLightVertexGrid.FindCellIndex<false>(vertex.mPosition, 0, int3((i&1), (i&2)>>1, (i&4)>>2) + o) );
					for (uint i = range.x; i < range.y; i++) {
						const PackedLightVertex v = gLightVertices[gLightVertexGrid.Get(i)];
						float lightW;
						PathSample ps = MergeWithLightVertex(s, vertex, v, lightW);
						ProcessCanonicalSample(r, mergeRnds.y, ps, cameraW * lightW);
					}
				}

				#ifdef VERTEX_MERGING_ONLY
				break;
				#endif
			}
			#endif

			#ifdef WAVEFRONT_CONNECTION_SELECTION
			if (WaveActiveSum(mergeRnds.w) / WaveActiveSum(1) >= gDirectLightProb)
			#endif
            {
                const uint count = gLightVertexCount[0];
                if (count > 0) {
                    PathSample ps = ConnectToLightSubpath(s, vertex, gLightVertices[GetRandomUint(s, PerVertexRandomNumbers::eBidirectionalConnection).x % count], weight);
                    weight *= count / (float)gLightSubpathCount;
					#ifdef WAVEFRONT_CONNECTION_SELECTION
					weight /= 1 - gDirectLightProb;
					#endif
					ProcessCanonicalSample(r, mergeRnds.y, ps, weight);
                }
			}
			#ifdef WAVEFRONT_CONNECTION_SELECTION
			else
			#endif

			#endif // BIDIRECTIONAL

            if (gSampleLights) {
                PathSample ps = ConnectToLight(s, vertex, weight);
				#if defined(BIDIRECTIONAL) && defined(WAVEFRONT_CONNECTION_SELECTION)
				weight /= gDirectLightProb;
				#endif
				ProcessCanonicalSample(r, mergeRnds.y, ps, weight);
            }

		}

        {
            float weight;
            const PathSample ps = SampleNextVertex(s, vertex, weight);
            ProcessCanonicalSample(r, mergeRnds.x, ps, weight);
    	}
    }

    r.FinalizeMerge();
	r.mM = 1;
	if (r.mW > 0 && r.mSample.mShiftType == PathShiftType::eReconnection)
        IncrementCounter(DebugCounterType::eReconnectionVertices);
    return r;
}
void SampleLightPath(const uint seed, const uint2 id) {
    PathVertex vertex;
    PathState<false> s = InitializeLightPath<false>(pcg(gRandomSeed), id, vertex);
    float3 prevPos = vertex.mPosition;
    SampleFirstLightVertex(s, vertex);

    while (any(s.throughput > 0)) {
		// store vertex
        if (IsDiffuse(vertex)) {
			#ifdef BIDIRECTIONAL
			#ifndef VERTEX_MERGING_ONLY
            {
                float weight;
                int2 pixelCoord;
                PathSample ps = ConnectToCamera(s, vertex, pixelCoord, weight);

				#ifdef DEBUG_PIXEL
                if (pixelCoord.y * gOutputSize.x + pixelCoord.x == gDebugPixel) {
                    printf("connecting %f %f %f, %f %f %f to camera\n", prevPos.x, prevPos.y, prevPos.z, vertex.mPosition.x, vertex.mPosition.y, vertex.mPosition.z);
                }
				#endif

				#ifdef LIGHT_TRACE_RESERVOIRS
                if (weight > 0) {
					uint idx;
					InterlockedAdd(gLightVertexCount[1], 1, idx);
					StoreReservoir(gLightTraceReservoirs, idx, gLightTraceReservoirGrid.mMaxSize, PathReservoir(ps, weight, 1));
					gLightTraceReservoirGrid.Append(float3(pixelCoord, 0), idx, 1, true);
                }
				#endif
			}
			#endif

			PackedLightVertex v;
			v.mVertex = PackVertex(vertex);
			v.mThroughput = s.throughput;
            v.mDirection = vertex.ToWorld(s.localDirIn);
            v.dVC  = s.dVC;
            v.dVM  = s.dVM;
			v.dVCM = s.dVCM;
			v.mSubpathPdf = s.pdfW;
            v.mRngIdx  = s.id;
            v.mBounces = s.bounces;

			uint idx;
			InterlockedAdd(gLightVertexCount[0], 1, idx);
			gLightVertices[idx] = v;

			#ifdef VERTEX_MERGING
			gLightVertexGrid.Append(vertex.mPosition, idx);
			#endif

			#endif
        }

        if (s.bounces + 1 == gMaxBounces)
            break;

        prevPos = vertex.mPosition;

        float weight;
        SampleNextVertex(s, vertex, weight);
    }
}

PathSample ShiftPath(const PathSample basePath, const PackedVertex dstVertex, const float3 cameraPos, out float jacobian) {
    jacobian = 0;

	#if defined(BIDIRECTIONAL) && defined(gLightTraceOnly) && defined(gNoLightTraceResampling)
    return PathSample();
	#endif

    if (dstVertex.mInstanceIndex == INVALID_INSTANCE)
        return PathSample();

	#if defined(BIDIRECTIONAL) && defined(gLightTraceOnly)
    const bool isLight = true;
    const bool hasRcv = false;
	#else
    const bool isLight = gBidirectional && basePath.mShiftType == PathShiftType::eLightPath;
    const bool hasRcv  = basePath.mShiftType == PathShiftType::eReconnection;
	#endif

    IncrementCounter(DebugCounterType::eShiftAttempts);
    if (hasRcv)  IncrementCounter(DebugCounterType::eReconnectionAttempts);
    if (isLight) IncrementCounter(DebugCounterType::eLightShiftAttempts);

	// initialize path state

    PathVertex vertex;
    PathState<true> s;
    if (isLight) {
        s = InitializeLightPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex);
        if (basePath.mBounces > 1)
            SampleFirstLightVertex(s, vertex);
		if (!any(s.throughput > 0))
			return PathSample();
    } else {
		vertex = UnpackVertex(dstVertex);
		s = InitializeCameraPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex, normalize(vertex.mPosition - cameraPos));
    }

	PathSample shifted;

    // trace the new prefix
    while (s.bounces < basePath.mReplayBounces && any(s.throughput > 0)) {
        float weight;
        shifted = SampleNextVertex(s, vertex, weight);
        DebugPrintf("Sampled next vertex (%s/%s): %f %f %f\n", s.bounces, basePath.mReplayBounces, vertex.mPosition.x, vertex.mPosition.y, vertex.mPosition.z);
    }

	// reconnection strategies
    if (basePath.mShiftType != PathShiftType::eReplay) {
		if (!vertex.mIsSurface || !any(s.throughput > 0)) {
			DebugPrintf("Prefix mismatch (%u replay bounces)\n", basePath.mReplayBounces);
			return PathSample();
		}

		if (hasRcv)
			shifted = ConnectToBasePath(s, vertex, basePath);
		#ifdef VERTEX_MERGING
        else if (basePath.mShiftType == PathShiftType::eMerge) {
            float weight;
            shifted = MergeWithLightVertex(s, vertex, reinterpret<PackedLightVertex>(basePath.mReconnectionVertex), weight);
		}
		#endif
		else if (isLight) {
            // connect vertex to cameraVertex

            DebugPrintf("Shift: Connecting %f %f %f, %f %f %f, to camera\n", vertex.mPosition.x, vertex.mPosition.y, vertex.mPosition.z, dstVertex.mPosition.x, dstVertex.mPosition.y, dstVertex.mPosition.z);

            float3 toCameraVertex = dstVertex.mPosition - vertex.mPosition;
			const float dist = length(toCameraVertex);
			toCameraVertex /= dist;

            const float3 localDirOut = vertex.ToLocal(toCameraVertex);

            if (basePath.mBounces == 1) {
                if (localDirOut.z <= 0)
                    return PathSample();
                s.throughput *= localDirOut.z;
			} else {
				s.throughput *= EvaluateReflectance(vertex, s.localDirIn, localDirOut, true).mReflectance;
                if (!any(s.throughput > 0)) {
					DebugPrintf("Zero vertex reflectance (%f %f)\n", s.localDirIn.z, localDirOut.z);
                    return PathSample();
                }
                s.bounces++;
			}

			if (Occluded(vertex, toCameraVertex, dist)) {
				DebugPrintf("Camera vertex occluded\n");
				return PathSample();
			}

            // connect cameraVertex to camera
           	vertex = UnpackVertex(dstVertex);
            s.localDirIn = vertex.ToLocal(-toCameraVertex);

            float weight;
            int2 pixelCoord;
            shifted = ConnectToCamera(s, vertex, pixelCoord, weight);
		} else
			return PathSample();
    }

    if (!any(shifted.mRadiance > 0))
        return PathSample();

	jacobian = basePath.mReplayPdfW / shifted.mReplayPdfW;

	IncrementCounter(DebugCounterType::eShiftSuccesses);
	if (isLight)
		IncrementCounter(DebugCounterType::eLightShiftSuccesses);
	if (hasRcv) {
		IncrementCounter(DebugCounterType::eReconnectionSuccesses);
        if (basePath.mReconnectionVertex.mVertex.mInstanceIndex != INVALID_INSTANCE)
			jacobian *= (shifted.mReconnectionVertex.mCos / basePath.mReconnectionVertex.mCos) * sqr(basePath.mReconnectionVertex.mDist / shifted.mReconnectionVertex.mDist);
	}

    return shifted;
}
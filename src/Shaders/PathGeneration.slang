#pragma once

#include "BRDF.slang"
#include "HashGrid.slang"
#include "Intersection.slang"
#include "Light.slang"
#include "PathReservoir.slang"

[[vk::push_constant]]
cbuffer PushConstants {
    uint gSpatialReusePass;
    uint gSpatialReuseSamples;
    float gSpatialReuseRadius;
    uint gReservoirIndex;
    uint2 gOutputSize;
    uint gRandomSeed;
};

static const float4 gSceneSphere = float4(gScene.mSceneMax + gScene.mSceneMin, length(gScene.mSceneMax - gScene.mSceneMin)) / 2;

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif

#ifdef BIDIRECTIONAL
static const bool gBidirectional = true;
ParameterBlock<HashGrid<PackedLightVertex>> gVisibleLightVertices;
RWStructuredBuffer<uint> gLightVertexCount;
RWStructuredBuffer<PackedLightVertex> gLightVertices;
#else
static const bool gBidirectional = false;
#endif

typedef RWByteAddressBuffer ReservoirBuffer;
ReservoirBuffer gPathReservoirs[2];

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

uniform float3 gCameraPosition;
uniform float3 gPrevCameraPosition;
uniform float4x4 gWorldToCamera;
uniform float4x4 gProjection;

uniform float gReconnectionDistance;
uniform float gReconnectionRoughness;
uniform uint gMaxBounces;
uniform uint gLightSubpathCount;
uniform float gCameraImagePlaneDist;

uniform uint gDebugViewVertices;
uniform uint gDebugLightVertices;

static const float gMisVmWeightFactor = 0;

float Mis(float a) { return a * a; }
float Mis(float a, float b) {
    a = Mis(a);
    return SafeDivide(a, a + Mis(b));
}

struct ReflectanceEvalRecord {
    float3 mReflectance;
    float mFwdPdfW;
    float mRevPdfW;
};
struct DirectionReflectanceSample {
    float3 mDirection;
    float mEta;
    float mRoughness;
    ReflectanceEvalRecord f;

    bool isSingular() { return mRoughness == 0; }
};
ReflectanceEvalRecord EvaluateReflectance(const PathVertex vertex, const float3 localDirIn, const float3 localDirOut, const bool adjoint) {
    ReflectanceEvalRecord r;
    r.mReflectance = EvalBrdfLocal(vertex.mMaterial, localDirIn, localDirOut, adjoint) * vertex.GetShadingNormalCorrection(localDirIn, localDirOut, adjoint);
    r.mFwdPdfW = EvalBrdfPdfWLocal(vertex.mMaterial, localDirIn, localDirOut);
    r.mRevPdfW = EvalBrdfPdfWLocal(vertex.mMaterial, localDirOut, localDirIn);
    return r;
}
DirectionReflectanceSample SampleDirection(const PathVertex vertex, const float4 rnd, const float3 localDirIn, const bool adjoint) {
    DirectionReflectanceSample r;
    DirectionSampleRecord s;
    SampleBrdfLocal(vertex.mMaterial, rnd, localDirIn, s);
    r.mDirection = s.mDirection;
    r.mEta = s.mEta;
    r.mRoughness = s.mRoughness;
	r.f = EvaluateReflectance(vertex, localDirIn, r.mDirection, adjoint);
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
    uint rcvPackedDirOut;
    float3 throughputAtRcv;
    float pdfAtRcv;
    float rcvCos;
    float rcvDist;

    property float3 rcvDirOut {
        get { return UnpackNormal(rcvPackedDirOut); }
        set { rcvPackedDirOut = PackNormal(newValue); }
	}

    property uint bounces {
        get { return BF_GET(packedBounces, 0, 8); }
        set { BF_SET(packedBounces, newValue, 0, 8); }
    }
    property uint rcvBounces {
        get { return BF_GET(packedBounces, 8, 8); }
        set { BF_SET(packedBounces, newValue, 8, 8); }
    }
    property bool isSpecular {
        get { return (bool)BF_GET(packedBounces, 16, 1); }
        set { BF_SET(packedBounces, (uint)newValue, 16, 1); }
    }
    property bool isLightPath {
        get { return (bool)BF_GET(packedBounces, 17, 1); }
        set { BF_SET(packedBounces, (uint)newValue, 17, 1); }
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
    eMerge,
    eTerminate,
    eDirection,
    eDirectLight,
    eBidirectionalConnection,
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

PathSample SampleNextVertex<let bShift : bool>(inout PathState<bShift> s, inout PathVertex vertex, out float samplePdf, out float misWeight) {
	const float prevPdf = s.pdfW;
	const bool diffuse = IsDiffuse(vertex);

	samplePdf = 0;
	misWeight = 0;

    const float contProb = diffuse ? GetContinuationProbability(vertex) : 1;
    if (!bShift && contProb < 1 && (contProb == 0 || GetRandomFloat(s, PerVertexRandomNumbers::eTerminate).x > contProb)) {
		s.throughput = 0;
		return PathSample();
	}

    DirectionReflectanceSample r = SampleDirection(vertex, GetRandomFloat(s, PerVertexRandomNumbers::eDirection), s.localDirIn, s.isLightPath);
    r.f.mFwdPdfW *= contProb;
    r.f.mRevPdfW *= contProb;

    s.throughput *= r.f.mReflectance;
    s.pdfW *= r.f.mFwdPdfW;
	if (all(s.throughput <= 0))
		return PathSample();

    if (gBidirectional) {
		if (diffuse) {
			s.isSpecular = false;
			s.dVC = Mis(abs(r.mDirection.z) / r.f.mFwdPdfW) * (s.dVC * Mis(r.f.mRevPdfW) + s.dVCM + gMisVmWeightFactor);
			s.dVCM = Mis(1.f / r.f.mFwdPdfW);
		} else {
			s.dVCM = 0.f;
			s.dVC *= Mis(abs(r.mDirection.z));
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
        samplePdf = s.pdfW;
		misWeight = 1;
        if (diffuse) {
            if (gBidirectional) {
                const float wCamera = Mis(directPdfA) * s.dVCM + Mis(emissionPdfW) * s.dVC;
				misWeight = 1 / (1 + wCamera);
            } else if (gSampleLights)
                misWeight = Mis(r.f.mFwdPdfW, lightPdfW);
		}

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
            hasRcv ? PathShiftType::eReconnection : PathShiftType::eReplay);
    }
	return PathSample();
}

PathSample ConnectToLight<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, out float samplePdf, out float misWeight) {
	samplePdf = 0;
    misWeight = 0;

    float lightPdf;
    const PathVertex l = SampleEmission(GetRandomFloat(s, PerVertexRandomNumbers::eDirectLight), lightPdf);

	float3 le = l.mMaterial.Emission();
	if (!any(le > 0))
		return PathSample();

    float3 toLight;
    float dist, cosLight, AtoW, directPdfW;
    if (l.mIsSurface) {
        toLight = l.mPosition - vertex.mPosition;
        dist = length(toLight);
        toLight /= dist;
        cosLight = max(0, -dot(toLight, l.mShadingNormal));
        if (cosLight <= 0)
            return PathSample();
        AtoW = sqr(dist) / cosLight;
		directPdfW = lightPdf*AtoW;
    } else {
        toLight = l.mPosition;
        dist = POS_INFINITY;
        AtoW = 1;
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

    samplePdf = s.pdfW * directPdfW;

    if (gBidirectional) {
		const float emissionPdfW = l.mIsSurface ? lightPdf * CosHemispherePdfW(cosLight) : (lightPdf * ConcentricDiscPdfA() / sqr(gSceneSphere.w));
		const float wLight = Mis(f.mFwdPdfW / directPdfW);
		const float wCamera = Mis(emissionPdfW * abs(localDirOut.z) / (directPdfW * cosLight)) * (gMisVmWeightFactor + s.dVCM + s.dVC * Mis(f.mRevPdfW));
		misWeight = 1 / (wLight + 1 + wCamera);
    } else {
		misWeight = Mis(directPdfW, f.mFwdPdfW);
    }

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
        PathShiftType::eReconnection);
}
PathSample ConnectToLightSubpath<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, const PackedLightVertex v, out float samplePdf, out float misWeight) {
	samplePdf = 0;
    misWeight = 0;

    const uint bounces = s.bounces + v.mBounces + 2;
    if (bounces > gMaxBounces)
        return PathSample();

    float cosLight, dist;
    float3 toLight;
    float3 lightDirOut;
    ReflectanceEvalRecord r;
    {
        const PathVertex lightVertex = UnpackVertex(v.mVertex);
        toLight = lightVertex.mPosition - vertex.mPosition;
        dist = length(toLight);
        toLight /= dist;

        lightDirOut = lightVertex.ToWorld(v.mDirection);
        const float3 localDirOut = lightVertex.ToLocal(-toLight);
        cosLight = abs(localDirOut.z);
        r = EvaluateReflectance(lightVertex, v.mDirection, localDirOut, true);
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
    if (all(r.mReflectance <= 0))
        return PathSample();

    if (Occluded(vertex, toLight, dist))
        return PathSample();

    const float3 contrib = (s.throughput * v.mThroughput) * (f.mReflectance * (r.mReflectance / cosLight));

    // Convert pdfs to area pdf
    const float invDist2 = 1 / sqr(dist);
    const float cameraBsdfDirPdfA = f.mFwdPdfW * cosLight * invDist2;
    const float lightBsdfDirPdfA  = r.mFwdPdfW * cosCam   * invDist2;

    // Partial light sub-path MIS weight [tech. rep. (40)]
    const float wLight = Mis(cameraBsdfDirPdfA) * (gMisVmWeightFactor + v.dVCM + v.dVC * Mis(r.mRevPdfW));

    // Partial eye sub-path MIS weight [tech. rep. (41)]
    const float wCamera = Mis(lightBsdfDirPdfA) * (gMisVmWeightFactor + s.dVCM + s.dVC * Mis(f.mRevPdfW));

    misWeight = 1 / (wLight + 1.f + wCamera);

    const float AtoW = sqr(dist) / cosLight;
    samplePdf = s.pdfW * (v.mSubpathPdf * AtoW);

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
        rcv.mRadiance = v.mThroughput * (r.mReflectance / cosLight);
        rcv.mDirOut = lightDirOut;
        rcv.mCos = cosLight;
        rcv.mDist = dist;
    }

    return PathSample(
        contrib,
        hasRcv ? s.pdfAtRcv : s.pdfW,
        bounces, hasRcv ? s.rcvBounces : s.bounces,
        s.seed, s.id,
        rcv,
		PathShiftType::eReconnection);
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
        DebugPrintf("Reconnection failed: zero le (%f * %f)\n", max3(s.throughput), max3(basePath.mReconnectionVertex.mRadiance));
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
        DebugPrintf("Reconnection failed: zero rcv reflectance\n");
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

float3 ConnectToCamera(const PathVertex vertex, const float3 localDirIn, const float maxDist, out float cameraPdfA, out float revPdfW, const bool adjoint = true) {
    cameraPdfA = 0;
    revPdfW = 0;

    const float3 cameraPos = TransformPoint(gWorldToCamera, vertex.mPosition);
    const float cosAtCamera = abs(normalize(cameraPos).z);

    float4 clip = mul(gProjection, float4(cameraPos, 1));
    float2 pixelCoord = gOutputSize * ((clip.xy / clip.w) * .5 + .5);
    if (any(pixelCoord < 0) || any(pixelCoord >= gOutputSize) || clip.z <= 0) {
        DebugPrintf("Camera vertex not visible");
        return 0;
    }

    float3 toCamera = gCameraPosition - vertex.mPosition;
    const float dist = length(toCamera);
    toCamera /= dist;

    // if (Occluded(vertex, toCamera, dist))
    if (dist > maxDist) {
        DebugPrintf("Camera occluded");
        return 0;
    }

    const float3 localDirOut = vertex.ToLocal(toCamera);
    const float cosOut = abs(localDirOut.z);

    ReflectanceEvalRecord f = EvaluateReflectance(vertex, adjoint ? localDirIn : localDirOut, adjoint ? localDirOut : localDirIn, adjoint);
    const float contProb = GetContinuationProbability(vertex);
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    if (!any(f.mReflectance > 0))
        DebugPrintf("zero camera bsdf: %f   %f \n", localDirIn.z, localDirOut.z);

    // Compute pdf conversion factor from image plane area to surface area
    const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
    const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;
    cameraPdfA = imageToSolidAngleFactor * cosOut / sqr(dist);
    revPdfW = f.mRevPdfW;

    return adjoint ? f.mReflectance / cosOut : f.mReflectance;
}

#ifdef BIDIRECTIONAL
float3 ProcessVisibleLightVertices(const uint2 id, const float dist, inout PathReservoir r) {
    RandomSampler rng = RandomSampler(gRandomSeed, id + int2(0, gOutputSize.y * 20));
    const uint2 range = gVisibleLightVertices.GetCellDataRange(gVisibleLightVertices.FindCellIndex<false>(float3(int2(id), 0), 1, 0, true));
	#ifdef gNoLightTraceResampling
    float3 total = 0;
	#else
    r.PrepareMerge();
	#endif
    for (uint i = range.x; i < range.y; i++) {
        const PackedLightVertex lv = gVisibleLightVertices.Get(i);

		#ifdef gDebugPathLengths
		if (gDebugViewVertices != 1 || gDebugLightVertices != lv.mBounces + 2)
			continue;
		#endif

        float cameraPdfA, revPdfW;
        const float3 contrib = lv.mThroughput * ConnectToCamera(UnpackVertex(lv.mVertex), lv.mDirection, dist, cameraPdfA, revPdfW);
        if (!any(contrib > 0))
            continue;

        const float samplePdf = lv.mSubpathPdf * (1 / cameraPdfA) * gLightSubpathCount;
        if (!(samplePdf > 0))
            continue;

		#ifdef gLightTraceOnly
		const float misWeight = 1;
		#else
        const float wLight = Mis(cameraPdfA / gLightSubpathCount) * (gMisVmWeightFactor + lv.dVCM + lv.dVC * Mis(revPdfW));
		const float misWeight = 1.f / (wLight + 1.f);
		#endif

		#ifdef gNoLightTraceResampling
		total += contrib * (misWeight / samplePdf);
		#else
		PathReservoir c = PathReservoir(
			PathSample(contrib, 1, lv.mBounces + 1, lv.mBounces + 1, gRandomSeed, uint2(lv.mRngIdx % gOutputSize.x, lv.mRngIdx / gOutputSize.x), ReconnectionVertex(), PathShiftType::eLightPath),
			1/samplePdf,
			1);
		c.PrepareMerge(misWeight);
		r.Merge(rng.NextFloat().x, c);
		#endif
    }
	#ifdef gNoLightTraceResampling
    return total;
	#else
    r.FinalizeMerge();
    r.mM = 1;
    return 0;
	#endif
}
#endif

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
    s.isSpecular = true;
    s.isLightPath = false;
    if (gBidirectional) {
        const float cosAtCamera = abs(normalize(TransformVector(gWorldToCamera, dir)).z);
        const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
        const float cameraPdfW = sqr(imagePointToCameraDist) / cosAtCamera;
        s.dVCM = Mis(gLightSubpathCount / cameraPdfW);
        s.dVC = 0;

        const float cosIn = abs(s.localDirIn.z);
		s.dVCM *= Mis(sqr(length(gCameraPosition - vertex.mPosition)) / cosIn);
    }
    return s;
}
PathState<bShift> InitializeLightPath<let bShift : bool>(const uint seed, const uint2 id, out PathVertex vertex, const bool extend = true) {
    PathState<bShift> s;
    s.seed = seed;
    s.id = id;
    s.isLightPath = true;

    s.bounces = -1; // set to -1 for rng
    float directPdfA;
    vertex = SampleEmission(GetRandomFloat(s, PerVertexRandomNumbers::eDirectLight), directPdfA);
    s.bounces = 0;
    s.rcvBounces = 0;

    s.throughput = vertex.mMaterial.Emission();
    s.pdfW = directPdfA;

    if (!extend)
        return s;

    const bool isFinite = vertex.mIsSurface;

    s.isSpecular = !isFinite;

	float emissionPdfW = directPdfA;
	float cosLight;

    float3 origin;
    float3 direction;

    s.bounces = -1; // set to -1 for rng
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

    float lightPdf, dist;
    vertex = TraceRay(MakeRay(origin, direction), true, lightPdf, dist);
    if (vertex.mIsSurface) {
        s.localDirIn = vertex.ToLocal(-direction);
        const float cosIn = abs(s.localDirIn.z);
        if (isFinite)
            s.dVCM *= Mis(sqr(dist));
        s.dVCM /= Mis(cosIn);
        s.dVC  /= Mis(cosIn);
    } else
		s.throughput = 0;

    return s;
}


void ProcessCanonicalSample(const uint2 id, const float mergeRnd, const PathSample ps, const float samplePdf, float misWeight) {
    if (samplePdf <= 0)
        return;

	#ifdef gDebugPathLengths
	if (gDebugViewVertices != ps.mBounces || gDebugLightVertices != ps.mLightVertices)
        return;
	#endif

	PathReservoir c = PathReservoir(ps, 1 / samplePdf, 1);
	c.PrepareMerge(misWeight);

	PathReservoir res = LoadReservoir(gPathReservoirsOut, id);
	res.Merge(mergeRnd, c);
	StoreReservoir(gPathReservoirsOut, id, res);
}
void SampleCameraPath(PathVertex vertex, const uint2 id) {
    PathState<false> s = InitializeCameraPath<false>(gRandomSeed, id, vertex, normalize(vertex.mPosition - gCameraPosition));

    while (s.bounces + 1 <= gMaxBounces && any(s.throughput > 0)) {
        const float3 mergeRnds = GetRandomFloat(s, PerVertexRandomNumbers::eMerge).xyz;

        if (IsDiffuse(vertex)) {
            if (gSampleLights) {
        		float samplePdf, misWeight;
                const PathSample ps = ConnectToLight(s, vertex, samplePdf, misWeight);
                ProcessCanonicalSample(id, mergeRnds.y, ps, samplePdf, misWeight);
            }
			#ifdef BIDIRECTIONAL
            {
                const uint count = gLightVertexCount[0];
                if (count > 0) {
                    float samplePdf, misWeight;
                    const PathSample ps = ConnectToLightSubpath(s, vertex, gLightVertices[GetRandomUint(s, PerVertexRandomNumbers::eBidirectionalConnection).x % count], samplePdf, misWeight);
					ProcessCanonicalSample(id, mergeRnds.z, ps, samplePdf, misWeight * (count / (float)gLightSubpathCount));
                }
			}
			#endif
		}

        {
            float samplePdf, misWeight;
            const PathSample ps = SampleNextVertex(s, vertex, samplePdf, misWeight);
            ProcessCanonicalSample(id, mergeRnds.x, ps, samplePdf, misWeight);
    	}
    }

	PathReservoir res = LoadReservoir(gPathReservoirsOut, id);
    res.FinalizeMerge();
	res.mM = 1;
	if (res.mW > 0 && res.mSample.mShiftType == PathShiftType::eReconnection)
		IncrementCounter(DebugCounterType::eReconnectionVertices);
    StoreReservoir(gPathReservoirsOut, id, res);
}
void SampleLightPath(const uint seed, const uint2 id) {
    PathVertex vertex;
    PathState<false> s = InitializeLightPath<false>(gRandomSeed, id + uint2(0, gOutputSize.y), vertex);

    while (any(s.throughput > 0)) {
        if (IsDiffuse(vertex)) {
			PackedLightVertex v;
			v.mVertex = PackVertex(vertex);
			v.mThroughput = s.throughput;
			v.mDirection = s.localDirIn;
			v.dVC = s.dVC;
			v.dVCM = s.dVCM;
			v.mSubpathPdf = s.pdfW;
            v.mRngIdx = s.id.y * gOutputSize.x + s.id.x;
            v.mBounces = s.bounces;

			#ifdef BIDIRECTIONAL
			const float4 clip = mul(gProjection, float4(TransformPoint(gWorldToCamera, vertex.mPosition), 1));
			const int2 pixelCoord = int2(gOutputSize * ((clip.xy / clip.w) * .5 + .5));
			if (all(pixelCoord >= 0) && all(pixelCoord < gOutputSize) && clip.z*sign(clip.w) > 0 )
				gVisibleLightVertices.Append(float3(pixelCoord, 0), v, 1, true);
			#ifndef gLightTraceOnly
			uint idx;
			InterlockedAdd(gLightVertexCount[0], 1, idx);
			gLightVertices[idx] = v;
			#endif
			#endif
        }

        if (s.bounces + 1 == gMaxBounces)
            break;

		float samplePdf, misWeight;
		SampleNextVertex(s, vertex, samplePdf, misWeight);
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
    if (isLight)
		s = InitializeLightPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex, basePath.mBounces > 1);
	else {
		vertex = UnpackVertex(dstVertex);
		s = InitializeCameraPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex, normalize(vertex.mPosition - cameraPos));
    }

	PathSample shifted;

    // trace the new prefix
    while (s.bounces < basePath.mReplayBounces && any(s.throughput > 0)) {
        float samplePdf, misWeight;
        shifted = SampleNextVertex(s, vertex, samplePdf, misWeight);
    }

	// reconnection strategies
    if (basePath.mShiftType != PathShiftType::eReplay) {
		if (!vertex.mIsSurface || !any(s.throughput > 0)) {
			DebugPrintf("Prefix mismatch\n");
			return PathSample();
		}

		if (hasRcv)
			shifted = ConnectToBasePath(s, vertex, basePath);
		else if (isLight) {
			const PathVertex cameraVertex = UnpackVertex(dstVertex);
			float3 toCameraVertex = cameraVertex.mPosition - vertex.mPosition;
			const float dist = length(toCameraVertex);
			toCameraVertex /= dist;
			if (Occluded(vertex, toCameraVertex, dist)) {
				DebugPrintf("Camera vertex occluded\n");
				return PathSample();
			}

			float3 contrib = s.throughput;
			if (basePath.mBounces > 1) {
				contrib *= EvaluateReflectance(vertex, s.localDirIn, vertex.ToLocal(toCameraVertex), true).mReflectance;
				if (!any(contrib > 0))
					return PathSample();
			} else if (vertex.ToLocal(toCameraVertex).z <= 0)
				return PathSample();

			const float3 localDirIn = cameraVertex.ToLocal(-toCameraVertex);

			float cameraPdfA, revPdfW;
			contrib *= ConnectToCamera(cameraVertex, localDirIn, POS_INFINITY, cameraPdfA, revPdfW, false);
			if (!any(contrib > 0)) {
                DebugPrintf("Camera connection failed\n")
				return PathSample();
			}
			shifted = PathSample(contrib, sqr(dist)/abs(localDirIn.z), basePath.mBounces, basePath.mReplayBounces, basePath.mRngSeed, basePath.mRngIdx, ReconnectionVertex(), basePath.mShiftType);
		} else
			return PathSample();
    }

    if (!any(shifted.mRadiance > 0)) {
        return PathSample();
    }

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
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
    float gDirectLightProb;
};

static const float4 gSceneSphere = float4(gScene.mSceneMax + gScene.mSceneMin, length(gScene.mSceneMax - gScene.mSceneMin)) / 2;

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif

#ifdef BIDIRECTIONAL
static const bool gBidirectional = true;
RWStructuredBuffer<PackedLightVertex> gLightVertices;
RWStructuredBuffer<uint> gLightVertexCount;
#ifdef PROCESS_LIGHT_VERTICES
ParameterBlock<HashGrid<uint>> gVisibleLightVertices;
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
			s.isSpecular = false;
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

float3 ConnectToCamera(const PathVertex vertex, const float3 localDirIn, const float maxDist, out float cameraPdfA, out float revPdfW, const bool adjoint = true) {
    cameraPdfA = 0;
    revPdfW = 0;

    const float3 cameraPos = TransformPoint(gWorldToCamera, vertex.mPosition);
    const float cosAtCamera = abs(normalize(cameraPos).z);

    float4 clip = mul(gProjection, float4(cameraPos, 1));
    float2 pixelCoord = gOutputSize * ((clip.xy / clip.w) * .5 + .5);
    if (any(pixelCoord < 0) || any(pixelCoord >= gOutputSize) || clip.z <= 0)
        return 0;

    float3 toCamera = gCameraPosition - vertex.mPosition;
    const float dist = length(toCamera);
    toCamera /= dist;

	if (dist > maxDist)
    //if (Occluded(vertex, toCamera, dist))
        return 0;

    const float3 localDirOut = vertex.ToLocal(toCamera);
    const float cosOut = abs(localDirOut.z);

    ReflectanceEvalRecord f = EvaluateReflectance(vertex, adjoint ? localDirIn : localDirOut, adjoint ? localDirOut : localDirIn, adjoint);
    const float contProb = GetContinuationProbability(vertex);
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    // Compute pdf conversion factor from image plane area to surface area
    const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
    const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;
    cameraPdfA = imageToSolidAngleFactor * cosOut / sqr(dist);
    revPdfW = f.mRevPdfW;

    return adjoint ? f.mReflectance / cosOut : f.mReflectance;
}

#if defined(BIDIRECTIONAL) && defined(PROCESS_LIGHT_VERTICES)
float3 ProcessVisibleLightVertices(const uint2 id, const float dist, inout PathReservoir r) {
    RandomSampler rng = RandomSampler(gRandomSeed, id + int2(0, gOutputSize.y * 20));
    const uint2 range = gVisibleLightVertices.GetCellDataRange(gVisibleLightVertices.FindCellIndex<false>(float3(int2(id), 0), 1, 0, true));
	#ifdef gNoLightTraceResampling
    float3 total = 0;
	#else
    r.PrepareMerge();
	#endif
    for (uint i = range.x; i < range.y; i++) {
        const PackedLightVertex lv = gLightVertices[gVisibleLightVertices.Get(i)];

		#ifdef gDebugPathLengths
		if (gDebugTotalVertices != lv.mBounces+3 || gDebugLightVertices != lv.mBounces + 2)
			continue;
		#endif

        float cameraPdfA, revPdfW;
        const PathVertex lightVertex = UnpackVertex(lv.mVertex);
        const float3 contrib = lv.mThroughput * ConnectToCamera(lightVertex, lightVertex.ToLocal(lv.mDirection), dist, cameraPdfA, revPdfW);
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
			PathSample(
				contrib,
				1,
				lv.mBounces + 1, lv.mBounces + 1,
				gRandomSeed, uint2(lv.mRngIdx % gOutputSize.x, lv.mRngIdx / gOutputSize.x),
				ReconnectionVertex(),
				PathShiftType::eLightPath,
				lv.mBounces + 2),
			misWeight/samplePdf,
			1);
		c.PrepareMerge();
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
        s.dVM = 0;

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

    return s;
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
            PathSample ps = PathSample();

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

            // if (mergeRnds.w >= gDirectLightProb) {
            if (WaveActiveSum(mergeRnds.w) / WaveActiveSum(1) >= gDirectLightProb) {
                const uint count = gLightVertexCount[0];
                if (count > 0) {
                    ps = ConnectToLightSubpath(s, vertex, gLightVertices[GetRandomUint(s, PerVertexRandomNumbers::eBidirectionalConnection).x % count], weight);
                    weight *= count / (float)gLightSubpathCount;
					weight /= 1 - gDirectLightProb;
                }
			} else
			#endif // BIDIRECTIONAL

            if (gSampleLights) {
                ps = ConnectToLight(s, vertex, weight);
				if (gBidirectional)
					weight /= gDirectLightProb;
            }

			ProcessCanonicalSample(r, mergeRnds.y, ps, weight);
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
    PathState<false> s = InitializeLightPath<false>(gRandomSeed, id + uint2(0, gOutputSize.y), vertex);

    while (any(s.throughput > 0)) {
		// store vertex
        if (IsDiffuse(vertex)) {
			PackedLightVertex v;
			v.mVertex = PackVertex(vertex);
			v.mThroughput = s.throughput;
            v.mDirection = vertex.ToWorld(s.localDirIn);
            v.dVC = s.dVC;
            v.dVM = s.dVM;
			v.dVCM = s.dVCM;
			v.mSubpathPdf = s.pdfW;
            v.mRngIdx = s.id.y * gOutputSize.x + s.id.x;
            v.mBounces = s.bounces;

			#ifdef BIDIRECTIONAL
			uint idx;
			InterlockedAdd(gLightVertexCount[0], 1, idx);
			gLightVertices[idx] = v;

			#if defined(PROCESS_LIGHT_VERTICES) && !defined(VERTEX_MERGING_ONLY)
			const float4 clip = mul(gProjection, float4(TransformPoint(gWorldToCamera, vertex.mPosition), 1));
			const int2 pixelCoord = int2(gOutputSize * ((clip.xy / clip.w) * .5 + .5));
			if (all(pixelCoord >= 0) && all(pixelCoord < gOutputSize) && clip.z*sign(clip.w) > 0 )
				gVisibleLightVertices.Append(float3(pixelCoord, 0), idx, 1, true);
			#endif

			#ifdef VERTEX_MERGING
			gLightVertexGrid.Append(vertex.mPosition, idx);
			#endif

			#endif
        }

        if (s.bounces + 1 == gMaxBounces)
            break;

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
    if (isLight)
		s = InitializeLightPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex, basePath.mBounces > 1);
	else {
		vertex = UnpackVertex(dstVertex);
		s = InitializeCameraPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex, normalize(vertex.mPosition - cameraPos));
    }

	PathSample shifted;

    // trace the new prefix
    while (s.bounces < basePath.mReplayBounces && any(s.throughput > 0)) {
        float weight;
        shifted = SampleNextVertex(s, vertex, weight);
    }

	// reconnection strategies
    if (basePath.mShiftType != PathShiftType::eReplay) {
		if (!vertex.mIsSurface || !any(s.throughput > 0)) {
			DebugPrintf("Prefix mismatch\n");
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
			shifted = PathSample(
				contrib,
				sqr(dist)/abs(localDirIn.z),
				basePath.mBounces, basePath.mReplayBounces,
				basePath.mRngSeed, basePath.mRngIdx,
				ReconnectionVertex(),
				basePath.mShiftType,
				basePath.mLightVertices);
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
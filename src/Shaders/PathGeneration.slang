#pragma once

#include "PathReservoir.slang"
#include "Intersection.slang"
#include "Light.slang"
#include "BRDF.slang"
#include "HashGrid.slang"

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif
#ifdef RECONNECTION
static const bool gReconnection = true;
#else
static const bool gReconnection = false;
#endif
#ifdef BIDIRECTIONAL
static const bool gBidirectional = true;
ParameterBlock<HashGrid<PackedLightVertex>> gVisibleLightVertices;
RWStructuredBuffer<uint> gLightVertexCount;
RWStructuredBuffer<PackedLightVertex> gLightVertices;
#else
static const bool gBidirectional = false;
#endif

uniform float3 gCameraPosition;
uniform float3 gPrevCameraPosition;
uniform float4x4 gWorldToCamera;
uniform float4x4 gProjection;

uniform float gReconnectionDistance;
uniform float gReconnectionRoughness;
uniform uint gMaxBounces;
uniform uint gLightSubpathCount;
uniform float4 gSceneSphere;
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

RWByteAddressBuffer gScratchReconnectionVertices;

ReconnectionVertex LoadRcv(const uint2 id, out float3 throughputAtRcv, out float prefixPdfW) {
    const uint address = FlattenIndex(id) * (sizeof(ReconnectionVertex) + sizeof(float4));
    const float4 data = gScratchReconnectionVertices.Load<float4>(address + sizeof(ReconnectionVertex));
    throughputAtRcv = data.rgb;
    prefixPdfW = data.w;
    return gScratchReconnectionVertices.Load<ReconnectionVertex>(address);
}
void StoreRcv(const uint2 id, const ReconnectionVertex rcv, const float3 throughputAtRcv, const float prefixPdfW) {
    const uint address = FlattenIndex(id) * (sizeof(ReconnectionVertex) + sizeof(float4));
    gScratchReconnectionVertices.Store<ReconnectionVertex>(address, rcv);
    gScratchReconnectionVertices.Store<float4>(address + sizeof(ReconnectionVertex), float4(throughputAtRcv, prefixPdfW));
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
        get { return rcvBounces > 0; }
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

    if (!bShift && gReconnection && !s.isLightPath && s.rcvFound && s.rcvBounces == s.bounces) {
        float3 throughputAtRcv;
        float prefixPdfW;
        ReconnectionVertex rcv = LoadRcv(s.id, throughputAtRcv, prefixPdfW);
		rcv.mDirOut = r.mDirection;
        throughputAtRcv = s.throughput;
        StoreRcv(s.id, rcv, throughputAtRcv, prefixPdfW);
	}

	s.bounces++;

    const bool prevReconnectable = diffuse && vertex.mMaterial.Roughness() >= gReconnectionRoughness;

    float directPdfA, dist;
    vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, r.mDirection), r.mDirection), true, directPdfA, dist);

    float3 le = vertex.mMaterial.Emission();
    float lightPdfW = directPdfA;
    float emissionPdfW = directPdfA;
	if (vertex.mIsSurface) {
        s.localDirIn = vertex.ToLocal(-r.mDirection);

		const float cosIn = abs(s.localDirIn.z);
        lightPdfW = directPdfA * sqr(dist) / abs(cosIn);
		if (cosIn <= 0) le = 0;

        emissionPdfW *= CosHemispherePdfW(cosIn);

        if (gBidirectional) {
            if (s.isLightPath) {
                const float G = abs(cosIn) / sqr(dist);
				s.throughput *= G;
				s.pdfW *= G;
			}
			s.dVCM *= Mis(sqr(dist) / cosIn);
			s.dVC  /= Mis(cosIn);
		}

        // store reconnection vertex if one hasn't been found yet
        if (!bShift && gReconnection && !s.isLightPath && !s.rcvFound && prevReconnectable && dist >= gReconnectionDistance && vertex.mMaterial.Roughness() >= gReconnectionRoughness && (IsDiffuse(vertex) || any(le > 0))) {
            ReconnectionVertex rcv;
			rcv.mVertex = PackVertex(vertex);
			rcv.mCos = abs(cosIn);
			rcv.mDist = dist;
            rcv.mPrefixBounces = s.bounces;
            StoreRcv(s.id, rcv, 0, prevPdf);
            s.rcvBounces = s.bounces;
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

		float prefixPdfW;
		ReconnectionVertex rcv;
		if (!bShift && gReconnection && s.rcvFound) {
			float3 throughputAtRcv;
			rcv = LoadRcv(s.id, throughputAtRcv, prefixPdfW);
			if (rcv.mPrefixBounces == s.bounces)
				rcv.mRadiance = le;
			else
				rcv.mRadiance = le * float3(SafeDivide(s.throughput.r, throughputAtRcv.r), SafeDivide(s.throughput.g, throughputAtRcv.g), SafeDivide(s.throughput.b, throughputAtRcv.b));
        }
        return PathSample(s.throughput * le, !bShift && gReconnection && s.rcvFound ? prefixPdfW : s.pdfW, s.bounces, 0, s.seed, s.id, rcv, !bShift && gReconnection && s.rcvFound);
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

    float prefixPdfW;
    ReconnectionVertex rcv;
	if (!bShift && gReconnection && s.rcvFound) {
		float3 throughputAtRcv;
		rcv = LoadRcv(s.id, throughputAtRcv, prefixPdfW);
		if (rcv.mPrefixBounces == s.bounces) {
			rcv.mDirOut = toLight;
			rcv.mRadiance = le;
		} else
			rcv.mRadiance = float3(SafeDivide(contrib.r, throughputAtRcv.r), SafeDivide(contrib.g, throughputAtRcv.g), SafeDivide(contrib.b, throughputAtRcv.b));
	}

    return PathSample(contrib, !bShift && gReconnection && s.rcvFound ? prefixPdfW : (s.pdfW * AtoW), s.bounces + 1, 1, s.seed, s.id, rcv, !bShift && gReconnection && s.rcvFound);
}
PathSample ConnectToLightSubpath<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, const PackedLightVertex v, out float samplePdf, out float misWeight) {
	samplePdf = 0;
    misWeight = 0;

    const uint bounces = s.bounces + 1 + v.mPathLength;
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
    samplePdf = s.pdfW * (v.mPdfA * AtoW);

    float prefixPdfW;
    ReconnectionVertex rcv;
    if (!bShift && gReconnection && s.rcvFound) {
        float3 throughputAtRcv;
        rcv = LoadRcv(s.id, throughputAtRcv, prefixPdfW);
        if (rcv.mPrefixBounces == s.bounces) {
            rcv.mDirOut = toLight;
            rcv.mRadiance = v.mThroughput * (r.mReflectance / cosLight);
        } else
            rcv.mRadiance = float3(SafeDivide(contrib.r, throughputAtRcv.r), SafeDivide(contrib.g, throughputAtRcv.g), SafeDivide(contrib.b, throughputAtRcv.b));
    } else
        rcv = reinterpret<ReconnectionVertex>(v);

    return PathSample(contrib, !bShift && gReconnection && s.rcvFound ? prefixPdfW : (s.pdfW * AtoW), bounces, v.mPathLength+1, s.seed, s.id, rcv, !bShift && gReconnection && s.rcvFound);
}

PathSample ConnectToBasePath<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, const PathSample basePath) {
    if (vertex.mMaterial.Roughness() < gReconnectionRoughness)
        return PathSample();

    ReconnectionVertex rcv = reinterpret<ReconnectionVertex>(basePath.mReconnectionData);
	const PathVertex baseRcv = UnpackVertex(rcv.mVertex);

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

	float3 le = s.throughput * EvaluateReflectance(vertex, s.localDirIn, vertex.ToLocal(toRcv), false).mReflectance * rcv.mRadiance;
	if (!any(le > 0))
		return PathSample();

	if (rcv.mPrefixBounces < basePath.mBounces)
		le *= EvaluateReflectance(baseRcv, baseRcv.ToLocal(-toRcv), baseRcv.ToLocal(rcv.mDirOut), false).mReflectance;
	else if (rcv.mPrefixBounces == basePath.mBounces && dot(toRcv, baseRcv.mShadingNormal) > 0)
		return PathSample();

	if (!any(le > 0))
		return PathSample();
	if (Occluded(vertex, toRcv, dist))
		return PathSample();

	PathSample ps = basePath;
	ps.mRadiance = le;
	ps.mReplayPdfW = s.pdfW;
	rcv.mCos = abs(dot(baseRcv.mShadingNormal, toRcv));
    rcv.mDist = dist;
    ps.mReconnectionData = reinterpret<uint4[3]>(rcv);
	return ps;
}

PathSample ConnectToCamera(const PackedLightVertex lv, const float maxDist, out float samplePdf, out float misWeight) {
    misWeight = 0;
    samplePdf = 0;

    const PathVertex vertex = UnpackVertex(lv.mVertex);

    const float3 cameraPos = TransformPoint(gWorldToCamera, vertex.mPosition);
    const float cosAtCamera = abs(normalize(cameraPos).z);

	float4 clip = mul(gProjection, float4(cameraPos, 1));
    float2 pixelCoord = gOutputSize * ((clip.xy / clip.w) * .5 + .5);
    if (any(pixelCoord < 0) || any(pixelCoord >= gOutputSize) || clip.z <= 0)
        return PathSample();

    float3 toCamera = gCameraPosition - vertex.mPosition;
    const float dist = length(toCamera);
    toCamera /= dist;

    //if (Occluded(vertex, toCamera, dist))
	if (dist > maxDist)
        return PathSample();

	const float3 localDirOut = vertex.ToLocal(toCamera);
	const float cosOut = abs(localDirOut.z);

    const float3 localDirIn = lv.mDirection;
	ReflectanceEvalRecord f = EvaluateReflectance(vertex, localDirIn, localDirOut, true);
	const float contProb = GetContinuationProbability(vertex);
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

	// Compute pdf conversion factor from image plane area to surface area
	const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
	const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;
	const float imageToSurfaceFactor = imageToSolidAngleFactor * cosOut / sqr(dist);

	// We put the virtual image plane at such a distance from the camera origin
	// that the pixel area is one and thus the image plane sampling pdf is 1.
	// The area pdf of aHitpoint as sampled from the camera is then equal to
	// the conversion factor from image plane area density to surface area density
	const float cameraPdfA = imageToSurfaceFactor;

	const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

    const float3 contrib = (lv.mThroughput * f.mReflectance) / cosOut;

    if (!any(contrib > 0))
        return PathSample();

    // convert the pdf from surface area to image plane area, w.r.t. which the
    // pixel integral is actually defined. We also divide by the number of samples
    // this technique makes, which is equal to the number of light sub-paths
    samplePdf = lv.mPdfA * surfaceToImageFactor * gLightSubpathCount;

	#ifdef gLightTraceOnly
	misWeight = 1;
	#else
	const float wLight = Mis(cameraPdfA / gLightSubpathCount) * (gMisVmWeightFactor + lv.dVCM + lv.dVC * Mis(f.mRevPdfW));
	misWeight = 1.f / (wLight + 1.f);
	#endif

    return PathSample(contrib, lv.mPdfA, lv.mPathLength, lv.mPathLength + 1, gRandomSeed, uint2(lv.mRngIdx % gOutputSize.x, lv.mRngIdx / gOutputSize.x), ReconnectionVertex(), false);
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
        const PackedLightVertex v = gVisibleLightVertices.Get(i);

        float pdf, misWeight;
        const PathSample s = ConnectToCamera(v, dist*1.01, pdf, misWeight);
        if (!(pdf > 0))
            continue;

		PathReservoir c = PathReservoir(s, 1/pdf, 1);
		#ifdef gNoLightTraceResampling
		total += c.mSample.mRadiance * c.mW * misWeight;
		#else
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
    }
    return s;
}
PathState<bShift> InitializeLightPath<let bShift : bool>(const uint seed, const uint2 id, out PathVertex vertex) {
    PathState<bShift> s;
    s.seed = seed;
    s.id = id + uint2(gOutputSize.x, 0);
    s.isLightPath = true;

    s.bounces = -1; // set to -1 for rng

    float directPdfA;
	const PathVertex lightVertex = SampleEmission(GetRandomFloat(s, PerVertexRandomNumbers::eDirectLight), directPdfA);

    s.throughput = lightVertex.mMaterial.Emission();
    const bool isFinite = lightVertex.mIsSurface;

    s.isSpecular = !isFinite;

	float emissionPdfW = directPdfA;
	float cosLight;

    float3 origin;
    float3 direction;

    const float2 dirRng = GetRandomFloat(s, PerVertexRandomNumbers::eDirection).xy;

    if (isFinite) {
		// area light
		const float3 dir = SampleCosHemisphere(dirRng.xy);
        cosLight = dir.z;
		direction = lightVertex.ToWorld(dir);
		origin = OffsetRayOrigin(lightVertex, direction);
		emissionPdfW *= CosHemispherePdfW(cosLight);
		s.throughput *= cosLight;
	} else {
		// environment light
		const float2 xy = SampleConcentricDisc(dirRng.xy);
		const float3x3 onb = MakeOrthonormal(lightVertex.mPosition);
		origin = gSceneSphere.xyz + gSceneSphere.w * (onb[0] * xy.x + onb[1] * xy.y + lightVertex.mPosition);
		direction = -lightVertex.mPosition;
		emissionPdfW *= ConcentricDiscPdfA() / sqr(gSceneSphere.w);
		cosLight = 1;
	}

    s.bounces = 0;
    s.rcvBounces = 0;

	s.pdfW = emissionPdfW;

    s.dVCM = Mis(directPdfA / emissionPdfW);
    s.dVC = isFinite ? Mis(cosLight / emissionPdfW) : 0;

    float lightPdf, dist;
    vertex = TraceRay(MakeRay(origin, direction), true, lightPdf, dist);
    if (vertex.mIsSurface) {
        s.localDirIn = vertex.ToLocal(-direction);
        const float cosIn = abs(s.localDirIn.z);
        const float G = isFinite ? abs(cosIn) / sqr(dist) : abs(cosIn);
		s.throughput *= G;
		s.pdfW *= G;
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
	if (gReconnection && res.mW > 0 && res.mSample.mHasRcv)
		IncrementCounter(DebugCounterType::eReconnectionVertices);
    StoreReservoir(gPathReservoirsOut, id, res);
}
void SampleLightPath(const uint seed, const uint2 id) {
    PathVertex vertex;
    PathState<false> s = InitializeLightPath<false>(gRandomSeed, id, vertex);

    while (s.bounces + 1 <= gMaxBounces && any(s.throughput > 0)) {
        if (IsDiffuse(vertex)) {
			PackedLightVertex v;
			v.mVertex = PackVertex(vertex);
			v.mThroughput = s.throughput;
			v.mDirection = s.localDirIn;
			v.dVC = s.dVC;
			v.dVCM = s.dVCM;
			v.mPdfA = s.pdfW;
			v.mRngIdx = s.id.y*gOutputSize.x + s.id.x;
            v.mPathLength = s.bounces + 1;

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

        {
            float samplePdf, misWeight;
            SampleNextVertex(s, vertex, samplePdf, misWeight);
        }
    }
}

PathSample ShiftPath(const PathSample basePath, const PackedVertex dstVertex, const float3 cameraPos, out float jacobian) {
    jacobian = 0;

    PathVertex vertex = UnpackVertex(dstVertex);
    if (!vertex.mIsSurface)
        return PathSample();

    IncrementCounter(DebugCounterType::eShiftAttempts);
    if (gReconnection && basePath.mHasRcv)
        IncrementCounter(DebugCounterType::eReconnectionAttempts);

	PathSample res = PathSample();

    PathState<true> s = InitializeCameraPath<true>(basePath.mRngSeed, basePath.mRngIdx, vertex, normalize(vertex.mPosition - cameraPos));
	const bool hasRcv = gReconnection && basePath.mHasRcv;

    while (s.bounces + 1 <= basePath.mBounces && any(s.throughput > 0)) {
        const bool diffuse = IsDiffuse(vertex);

        if (hasRcv) {
            const ReconnectionVertex baseRcv = reinterpret<ReconnectionVertex>(basePath.mReconnectionData);
            if (s.bounces + 1 == baseRcv.mPrefixBounces) {
				if (diffuse)
					res = ConnectToBasePath(s, vertex, basePath);
				break;
            }
        }

        float samplePdf, misWeight;
        if (!hasRcv && basePath.mLightVertices > 0 && s.bounces + basePath.mLightVertices == basePath.mBounces) {
            if (gSampleLights && diffuse && basePath.mLightVertices == 1)
                res = ConnectToLight(s, vertex, samplePdf, misWeight);
            else if (gBidirectional && diffuse && basePath.mLightVertices > 1)
				res = ConnectToLightSubpath(s, vertex, reinterpret<PackedLightVertex>(basePath.mReconnectionData), samplePdf, misWeight);
            break;
		}

        res = SampleNextVertex(s, vertex, samplePdf, misWeight);
        if (s.bounces != basePath.mBounces || basePath.mLightVertices > 0)
            res.mRadiance = 0;
	}

    if (any(res.mRadiance > 0)) {
		IncrementCounter(DebugCounterType::eShiftSuccesses);
		jacobian = basePath.mReplayPdfW / res.mReplayPdfW;
        if (hasRcv) {
            IncrementCounter(DebugCounterType::eReconnectionSuccesses);
            const ReconnectionVertex baseRcv = reinterpret<ReconnectionVertex>(basePath.mReconnectionData);
            const ReconnectionVertex rcv = reinterpret<ReconnectionVertex>(res.mReconnectionData);
            if (baseRcv.mVertex.mInstanceIndex != INVALID_INSTANCE)
                jacobian *= (rcv.mCos / baseRcv.mCos) * sqr(baseRcv.mDist / rcv.mDist);
        }
    }

    return res;
}
#include "Intersection.slang"
#include "Random.slang"
#include "Light.slang"
#include "BRDF.slang"
#include "HashGrid.slang"

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
    uint gLightSubpathCount;
    float gImagePlaneDist;
    uint gMinDepth;
    uint gMaxDepth;
    uint gDebugPathLengths;
    uint gRandomSeed;

    uint gLVCCanonicalCandidates;
    uint gLVCReuseCandidates;
    uint gLVCMCap;
    float gLVCJitterRadius;
};

static const float4 gSceneSphere = float4(gScene.mSceneMax + gScene.mSceneMin, length(gScene.mSceneMax - gScene.mSceneMin)) / 2;

#define gRadianceQuantizationFactor 16384

struct PathState {
    float3 mRayOrigin;
    float mPdf;
    float3 mThroughput;
    float mDirPdfW;
    RandomSampler mRng;
	uint4 mPackedData;

    property float3 mRayDirection {
        get { return UnpackNormal(mPackedData[0]); }
        set { mPackedData[0] = PackNormal(newValue); }
    }
	#ifdef gMultiDispatch
    property float dVCM {
        get { return f16tof32(BF_GET(mPackedData[2], 0, 16)); }
        set { BF_SET(mPackedData[2], f32tof16(newValue), 0, 16); }
    }
    property float dVC {
        get { return f16tof32(BF_GET(mPackedData[2], 16, 16)); }
        set { BF_SET(mPackedData[2], f32tof16(newValue), 16, 16); }
    }
    property float dVM {
        get { return f16tof32(BF_GET(mPackedData[3], 0, 16)); }
        set { BF_SET(mPackedData[3], f32tof16(newValue), 0, 16); }
    }
	#else
    float dVCM;
    float dVC;
    float dVM;
	#endif

    property uint mPathLength {
        get { return BF_GET(mPackedData[3], 16, 13); }
        set { BF_SET(mPackedData[3], newValue, 16, 13); }
    }

	// specifies if a reconnection vertex has been found
    property bool isRcvValid {
        get { return bool(BF_GET(mPackedData[3], 30, 1)); }
        set { BF_SET(mPackedData[3], (newValue ? 1 : 0), 30, 1); }
    }

    // specifies if a light path came from a finite emitter (e.g. not the environment)
    property bool isFiniteLight {
        get { return bool(BF_GET(mPackedData[3], 31, 1)); }
        set { BF_SET(mPackedData[3], (newValue ? 1 : 0), 31, 1); }
    }
	// specifies if a view path is specular
    property bool isSpecular {
        get { return bool(BF_GET(mPackedData[3], 31, 1)); }
        set { BF_SET(mPackedData[3], (newValue ? 1 : 0), 31, 1); }
    }
};

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
    property uint mSubpathLength {
        get { return BF_GET(mPackedRngLength, 0, 8); }
        set { BF_SET(mPackedRngLength, newValue, 0, 8); }
    }

    property float3 mDirection {
        get { return UnpackNormal(mPackedDirection); }
        set { mPackedDirection = PackNormal(newValue); }
    }
};

struct ShadowRay {
    float3 mContribution;
    float mWeight;
    float3 mRayOrigin;
    float mDistance;
    float3 mRayDirection;
    uint mPixelIndex;
    uint mViewVertices;
    uint mLightVertices;
    uint pad0;
    uint pad1;

    uint2 GetPixelCoord() {
        return uint2(mPixelIndex % gOutputSize.x, mPixelIndex / gOutputSize.x);
    }
};

RWStructuredBuffer<ShadowRay> gShadowRays;
RWStructuredBuffer<uint> gCounters; // 0 -> light vertices, 1 -> shadow rays

Texture2D<uint4> gVertices;
RWTexture2D<float4> gOutput;
RWByteAddressBuffer gOutputAtomic;
uniform float4x4 gWorldToCamera;
uniform float4x4 gProjection;
uniform float3 gCameraPosition;

static PathState sPathState;
static uint2 sPixelIndex;

#ifdef gMultiDispatch
RWStructuredBuffer<PathState> gPathStates;
void LoadPathState(const uint threadIndex) {
    sPathState = gPathStates[threadIndex];
}
void StorePathState(const uint threadIndex) {
    gPathStates[threadIndex] = sPathState;
}
#endif

#ifdef gUseVM
ParameterBlock<HashGrid<PackedLightVertex>> gLightVertices;
#define gLightVertexCounter gLightVertices.GetCurrentElementCount()
#define LoadLightVertex(i) gLightVertices.mAppendData[i]
#define gMergeRadius (gLightVertices.mMinCellSize*0.5)
#else
RWStructuredBuffer<PackedLightVertex> gLightVertices;
#define gLightVertexCounter gCounters[0]
#define LoadLightVertex(i) gLightVertices[i]
#define gMergeRadius 0
#endif

#ifdef gLVCResampling
#include "Reservoir.slang"
struct LightVertexSample : ReservoirSample {
    PackedLightVertex mLightVertex;
    PackedVertex mCameraVertex;
    uint mPackedLocalDirIn;
    float mCachedTargetPdf;
    float mCachedMisWeight;
    float mG;

    property float3 mLocalDirIn {
		get { return UnpackNormal(mPackedLocalDirIn); }
        set { mPackedLocalDirIn = PackNormal(newValue); }
    }

	float GetTargetPdf() { return mCachedTargetPdf; }
};
typedef Reservoir<LightVertexSample> LightVertexReservoir;
struct PackedLightVertexReservoir {
    uint4 data0[3];
    PackedVertex data1;
    float4 data2;
    float2 data3;
    float2 pad;

    __init(const LightVertexReservoir r) {
        data0 = reinterpret<uint4[3]>(r.mSample.mLightVertex);
        data1 = r.mSample.mCameraVertex;
        data2 = float4(asfloat(r.mSample.mPackedLocalDirIn), r.mSample.mCachedTargetPdf, r.mSample.mCachedMisWeight, r.mSample.mG);
        data3 = float2(r.mW, r.mM);
	}
    LightVertexReservoir Unpack() {
        LightVertexReservoir r;
        r.mSample.mLightVertex = reinterpret<PackedLightVertex>(data0);
        r.mSample.mCameraVertex = data1;
        r.mSample.mPackedLocalDirIn = asuint(data2.x);
        r.mSample.mCachedTargetPdf = data2.y;
        r.mSample.mCachedMisWeight = data2.z;
        r.mSample.mG = data2.w;
        r.mW = data2.x;
        r.mM = data2.y;
        return r;
	}
}
static const uint PackedLightVertexReservoirSize = sizeof(PackedLightVertexReservoir);
ParameterBlock<HashGrid<PackedLightVertexReservoir>> gLightVertexHashGrid;
ParameterBlock<HashGrid<PackedLightVertexReservoir>> gPrevLightVertexHashGrid;
#endif

static const float gEtaVCM = sqr(gMergeRadius) * M_PI * gLightSubpathCount;
#ifdef gUseVM
static const float gMisVmWeightFactor = mis(gEtaVCM);
#else
static const float gMisVmWeightFactor = 0;
#endif
#ifdef gUseVC
static const float gMisVcWeightFactor = mis(1 / gEtaVCM);
#else
static const float gMisVcWeightFactor = 0;
#endif


void AddRadianceInterlocked(uint2 id, float3 c) {
    const uint index = id.y * gOutputSize.x + id.x;
    uint3 ci = uint3(c * gRadianceQuantizationFactor);
	uint3 p;
	gOutputAtomic.InterlockedAdd(16*index+0, ci.r, p.r);
	gOutputAtomic.InterlockedAdd(16*index+4, ci.g, p.g);
	gOutputAtomic.InterlockedAdd(16*index+8, ci.b, p.b);

	uint overflowMask = 0;
	for (uint i = 0; i < 3; i++)
		if (0xFFFFFFFF - p.r < ci.r)
			overflowMask |= BIT(i);
	gOutputAtomic.InterlockedOr(16*index + 12, overflowMask);
}
void AddRadianceInterlocked(const uint2 index, const float3 radiance, const float weight, const uint viewVertices, const uint lightVertices) {
	if (weight <= 0 || all(radiance <= 0))
		return;

	#ifdef gDebugPaths
		if (BF_GET(gDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gDebugPathLengths, 16, 16) == lightVertices) {
			#ifdef gDebugPathWeights
				AddRadianceInterlocked(index, radiance * weight);
			#else
				AddRadianceInterlocked(index, radiance);
			#endif
		}
	#else
		AddRadianceInterlocked(index, radiance * weight);
	#endif
}

// called when a full path is found
void OutputRadiance(const float3 radiance, const float weight, const uint viewVertices, const uint lightVertices) {
	if (weight <= 0 || all(radiance <= 0))
		return;

	#ifdef gDebugPaths
	if (BF_GET(gDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gDebugPathLengths, 16, 16) == lightVertices) {
		#ifdef gDebugPathWeights
			gOutput[sPixelIndex] += float4(radiance * weight, 0);
		#else
			gOutput[sPixelIndex] += float4(radiance, 0);
		#endif
    }
    return;
	#endif

	gOutput[sPixelIndex] += float4(radiance * weight, 0);
}

float AdjustIntersectionDistance(float x) { return x * (1 - 1 / 512.0); }

void TraceConnectionRay(const float3 rayOrigin, const float3 rayDirection, const float rayDistance, float3 contrib, const float weight, const uint viewVertices, const uint lightVertices) {
    #ifdef gDeferShadowRays
        ShadowRay ray;
        ray.mContribution = contrib;
        ray.mWeight = weight;
        ray.mRayOrigin = rayOrigin;
        ray.mViewVertices = viewVertices;
        ray.mRayDirection = rayDirection;
        ray.mDistance = rayDistance;
        ray.mPixelIndex = sPixelIndex.y * gOutputSize.x + sPixelIndex.x;
        ray.mLightVertices = lightVertices;
		uint appendIndex;
		InterlockedAdd(gCounters[1], 1, appendIndex);
		gShadowRays[appendIndex] = ray;
    #else
        float lightPdf, dist;
        if (TraceRay(MakeRay(rayOrigin, rayDirection, 0, rayDistance), false, lightPdf, dist).mIsSurface)
            return;

        OutputRadiance(contrib, weight, viewVertices, lightVertices);
    #endif
}

float mis(const float a) {
    return a * a;
}
float mis(const float a, const float b) {
    const float a2 = mis(a);
    return a2 / (a2 + mis(b));
}

// ---------------------------------------------------------------------------------------------------------
// ----------------------------------- Bidirectional vertex connections ------------------------------------
// ---------------------------------------------------------------------------------------------------------

void StoreLightVertex(const PathVertex vertex, const float3 localDirIn) {
    PackedLightVertex v;
	v.mVertex = PackVertex(vertex);
    v.mThroughput = sPathState.mThroughput;
    v.mDirection = vertex.ToWorld(localDirIn);
    v.mSubpathLength = sPathState.mPathLength;
    v.dVCM = sPathState.dVCM;
    v.dVC  = sPathState.dVC;
    v.dVM  = sPathState.dVM;
    v.mSubpathPdf = sPathState.mPdf;
	#ifdef gUseVM
    gLightVertices.Append(vertex.mPosition, v);
	#else
    uint idx;
    InterlockedAdd(gCounters[0], 1, idx);
    gLightVertices[idx] = v;
	#endif
}

// note: does not assign mCameraVertex
float3 ConnectToLightSubpath(const PathVertex vertex, const float3 localDirIn, const PackedLightVertex v, out RayDesc visibilityRay, out float misWeight, out float G) {
    const uint pathLength = sPathState.mPathLength + 1 + v.mSubpathLength;
    if (pathLength < gMinDepth || pathLength > gMaxDepth) {
		visibilityRay = {};
        misWeight = 0;
        G = 0;
		return 0;
    }

    float3 contrib;
    float3 toLight;
    float dist;
    float cosLight;

    ReflectanceEvalRecord r;
    {
        contrib = sPathState.mThroughput * v.mThroughput;
		const PathVertex lightVertex = UnpackVertex(v.mVertex);
		toLight = lightVertex.mPosition - vertex.mPosition;
		dist = length(toLight);
		toLight /= dist;

        const float3 lightLocalDirIn  = lightVertex.ToLocal(v.mDirection);
        const float3 lightLocalDirOut = lightVertex.ToLocal(-toLight);
        cosLight = abs(lightLocalDirOut.z);
        r = EvalBrdf(lightVertex.mMaterial, lightLocalDirIn, lightLocalDirOut, true);
        r.mReflectance *= lightVertex.GetShadingNormalCorrection(lightLocalDirIn, lightLocalDirOut, true);
		const float contProb = GetContinuationProbability(lightVertex);
		r.mFwdPdfW *= contProb;
		r.mRevPdfW *= contProb;
	}

    {
		const float3 localDirOut = vertex.ToLocal(toLight);

		ReflectanceEvalRecord f = EvalBrdf(vertex.mMaterial, localDirIn, localDirOut, false);
		f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
		const float contProb = GetContinuationProbability(vertex);
		f.mFwdPdfW *= contProb;
		f.mRevPdfW *= contProb;

        contrib *= f.mReflectance * r.mReflectance;

        const float invDist2 = 1 / sqr(dist);
        contrib *= invDist2;

        G = cosLight * invDist2;

		// Convert pdfs to area pdf
		const float cameraBsdfDirPdfA = f.mFwdPdfW * cosLight * invDist2;
		const float lightBsdfDirPdfA  = r.mFwdPdfW * abs(localDirOut.z) * invDist2;

		// Partial light sub-path MIS weight [tech. rep. (40)]
		const float wLight = mis(cameraBsdfDirPdfA) * (gMisVmWeightFactor + v.dVCM + v.dVC * mis(r.mRevPdfW));

		// Partial eye sub-path MIS weight [tech. rep. (41)]
		const float wCamera = mis(lightBsdfDirPdfA) * (gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

		misWeight = 1 / (wLight + 1.f + wCamera);
    }

	visibilityRay = MakeRay(OffsetRayOrigin(vertex, toLight), toLight, 0, AdjustIntersectionDistance(dist));

    return contrib;
}
#ifdef gLVCResampling
extension LightVertexSample {
    __init(const PathVertex vertex, const float3 localDirIn, const PackedLightVertex lightVertex, out RayDesc visibilityRay, out float3 contrib) {
        mLightVertex = lightVertex;
        contrib = ConnectToLightSubpath(vertex, localDirIn, mLightVertex, visibilityRay, mCachedMisWeight, mG);
        mCachedTargetPdf = Luminance(contrib) * mCachedMisWeight;
    }
}
LightVertexSample Shift(const LightVertexSample baseSample, const PathVertex vertex, const float3 localDirIn, out float3 contrib, out float jacobian) {
    RayDesc ray;
    LightVertexSample shifted = LightVertexSample(vertex, localDirIn, baseSample.mLightVertex, ray, contrib);
    if (any(contrib > 0)) {
        float lightPdf, dist;
		if (!TraceRay(ray, false, lightPdf, dist).mIsSurface) {
			jacobian = baseSample.mG / shifted.mG;
			return shifted;
        }
    }
	shifted.mCachedTargetPdf = 0;
	contrib = 0;
	jacobian = 0;
    return shifted;
}
#endif
void ConnectToLightSubpath(const PathVertex vertex, const float3 localDirIn) {
	#ifdef gLVCResampling
    {
        LightVertexReservoir r = LightVertexReservoir();
        float3 selectedContribution;

        // canonical samples
        if (gLightVertexCounter > 0) {
			RayDesc visibilityRay;
            for (uint i = 0; i < gLVCCanonicalCandidates; i++) {
				LightVertexReservoir candidate;
                RayDesc candidateRay;
                float3 candidateContribution;
                candidate.mSample = LightVertexSample(vertex, localDirIn, LoadLightVertex(sPathState.mRng.Next().x % gLightVertexCounter), candidateRay, candidateContribution);
                candidate.mW = 1 / candidate.mSample.mLightVertex.mPdf;
                candidate.mM = 1;

				candidate.PrepareMerge();
                if (r.Merge(sPathState.mRng.NextFloat().x, candidate)) {
                    visibilityRay = candidateRay;
                    selectedContribution = candidateContribution;
                }
            }

            float lightPdf, dist;
            if (TraceRay(visibilityRay, false, lightPdf, dist).mIsSurface) {
                r.mW = 0;
            } else {
                r.FinalizeMerge();
                r.mW /= gLVCCanonicalCandidates;
                r.mW *= gLightVertexCounter / (float)gLightSubpathCount;
            }
        }

        // reuse
		#ifdef gLVCResamplingReuse
		#ifdef gLVCJitter
        const float3 jitter = vertex.ToWorld(float3(gLVCJitterRadius * SampleConcentricDisc(sPathState.mRng.NextFloat().xy), 0));
		#else
        const float3 jitter = 0;
		#endif
        const uint2 range = gPrevLightVertexHashGrid.GetCellDataRange(gPrevLightVertexHashGrid.FindCellIndex<false>(vertex.mPosition + jitter));
        const uint count = range.y - range.x;
        //const bool doAll = gLVCReuseCandidates >= count;
        //for (uint i = 0; i < min(count, gLVCReuseCandidates); i++) {
        if (count > 0) {
            LightVertexReservoir candidate = gPrevLightVertexHashGrid.Get(range.x + sPathState.mRng.Next().x % count).Unpack();
            const float candidateTargetPdf = candidate.mSample.mCachedTargetPdf;

            // shift candidate to vertex
            float3 candidateContrib;
            float jacobianToVertex;
            candidate.mSample = Shift(candidate.mSample, vertex, localDirIn, candidateContrib, jacobianToVertex);

			candidate.PrepareMerge(candidate.mM / (candidate.mM + r.mM), jacobianToVertex);
            if (r.mW > 0) {
                float jacobianToCandidate;
                float3 tmpContrib;
                const float shiftedTargetPdf = Shift(r.mSample, UnpackVertex(candidate.mSample.mCameraVertex), candidate.mSample.mLocalDirIn, tmpContrib, jacobianToCandidate).mCachedTargetPdf;
                r.PrepareMerge(shiftedTargetPdf * jacobianToCandidate > 0 ? r.mM / (r.mM + candidate.mM) : 1);
			}

            if (r.Merge(sPathState.mRng.NextFloat().x, candidate)) {
                selectedContribution = candidateContrib;
            }
        }

		r.FinalizeMerge();

        if (!(r.mW > 0) || !any(selectedContribution > 0))
            return;

        r.mSample.mLocalDirIn = localDirIn;
        r.mSample.mCameraVertex = PackVertex(vertex);
        if (gLVCMCap > 0) r.mM = min(gLVCMCap, r.mM);
		gLightVertexHashGrid.Append(vertex.mPosition, PackedLightVertexReservoir(r));
		#endif

        OutputRadiance(selectedContribution, r.mSample.mCachedMisWeight * r.mW, sPathState.mPathLength + 1, r.mSample.mLightVertex.mPathLength + 1);
	}
	#else
    {
        if (gLightVertexCounter == 0)
            return;
        const PackedLightVertex v = LoadLightVertex(sPathState.mRng.Next().x % gLightVertexCounter);
        float misWeight, G;
        RayDesc visibilityRay;
        const float3 contrib = ConnectToLightSubpath(vertex, localDirIn, v, visibilityRay, misWeight, G);
        if (any(contrib > 0)) {
            misWeight *= gLightVertexCounter / (float)gLightSubpathCount;
			TraceConnectionRay(visibilityRay.Origin, visibilityRay.Direction, visibilityRay.TMax, contrib, misWeight / (sPathState.mPdf * v.mSubpathPdf), sPathState.mPathLength + 1, v.mSubpathLength + 1);
        }
    }
	#endif
}

float3 MergeWithLightVertex(const PathVertex vertex, const float3 localDirIn, const PackedLightVertex v, out float w) {
	w = 0;
	const uint pathLength = sPathState.mPathLength + v.mSubpathLength;
	if (pathLength < gMinDepth || pathLength > gMaxDepth)
		return 0;

	if (length(v.mVertex.mPosition - vertex.mPosition) >= gMergeRadius)
		return 0;

    const float3 localDirOut = vertex.ToLocal(v.mDirection);
	ReflectanceEvalRecord f = EvalBrdf(vertex.mMaterial, localDirIn, localDirOut, false);
	f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
	const float contProb = GetContinuationProbability(vertex);
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

	// Partial light sub-path MIS weight [tech. rep. (38)]
	const float wLight = v.dVCM * gMisVcWeightFactor + v.dVM * mis(f.mFwdPdfW);

	// Partial eye sub-path MIS weight [tech. rep. (39)]
	const float wCamera = sPathState.dVCM * gMisVcWeightFactor + sPathState.dVM * mis(f.mRevPdfW);

	// Full path MIS weight [tech. rep. (37)]. No MIS for PPM
	#ifdef gUsePpm
	const float misWeight = 1;
	#else
	const float misWeight = 1.f / (wLight + 1.f + wCamera);
	#endif

	w = misWeight / v.mSubpathPdf;
    return v.mThroughput * f.mReflectance;
}
void MergeWithLightVertices(const PathVertex vertex, const float3 localDirIn) {
	#ifdef gUseVM
    int3 o = lerp(int3(0, 0, 0), int3(-1, -1, -1), frac(vertex.mPosition / gLightVertices.mMinCellSize) < 0.5);
    float cameraW = 1 / (sPathState.mPdf * gEtaVCM);
    for (int i = 0; i < 8; i++) {
		const uint2 range = gLightVertices.GetCellDataRange( gLightVertices.FindCellIndex<false>(vertex.mPosition, 0, int3((i&1), (i&2)>>1, (i&4)>>2) + o) );
		for (uint i = range.x; i < range.y; i++) {
			float lightW;
			const PackedLightVertex v = gLightVertices.Get(i);
			float3 c = MergeWithLightVertex(vertex, localDirIn, v, lightW);
			OutputRadiance(sPathState.mThroughput * c, lightW*cameraW, sPathState.mPathLength+1, v.mSubpathLength+1);
		}
	}
	#endif

}

// ---------------------------------------------------------------------------------------------------------
// --------------------------------------------- Light tracing ---------------------------------------------
// ---------------------------------------------------------------------------------------------------------

void ConnectToCamera(const PathVertex vertex, const float3 localDirIn) {
    const float3 cameraPos = TransformPoint(gWorldToCamera, vertex.mPosition);
	float4 clip = mul(gProjection, float4(cameraPos, 1));
	float2 pixelCoord = gOutputSize * ((clip.xy/clip.w) * .5 + .5);
    if (any(pixelCoord < 0) || any(pixelCoord >= gOutputSize) || clip.z <= 0)
        return;

    const float cosAtCamera = abs(normalize(cameraPos).z);

	float3 toCamera = gCameraPosition - vertex.mPosition;
    const float dist = length(toCamera);
    toCamera /= dist;

    const float3 localDirOut = vertex.ToLocal(toCamera);
	const float cosOut = abs(localDirOut.z);

    ReflectanceEvalRecord f = EvalBrdf(vertex.mMaterial, localDirIn, localDirOut, true);
	f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, true);
    const float contProb = GetContinuationProbability(vertex);
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    // Compute pdf conversion factor from image plane area to surface area
    const float imagePointToCameraDist = gImagePlaneDist / cosAtCamera;
    const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;
    const float imageToSurfaceFactor = imageToSolidAngleFactor * cosOut / sqr(dist);

    // We put the virtual image plane at such a distance from the camera origin
    // that the pixel area is one and thus the image plane sampling pdf is 1.
    // The area pdf of aHitpoint as sampled from the camera is then equal to
    // the conversion factor from image plane area density to surface area density
    const float cameraPdfA = imageToSurfaceFactor;

    const float surfaceToImageFactor = 1.f / imageToSurfaceFactor;

    // We divide the contribution by surfaceToImageFactor to convert the (already
    // divided) pdf from surface area to image plane area, w.r.t. which the
    // pixel integral is actually defined. We also divide by the number of samples
    // this technique makes, which is equal to the number of light sub-paths
    float3 contrib = (sPathState.mThroughput * f.mReflectance) / (sPathState.mPdf * gLightSubpathCount * surfaceToImageFactor * cosOut);

	#if defined(gUseVC) || defined(gUseVM)
    // Partial light sub-path weight [tech. rep. (46)]. Note the division by
    // mLightPathCount, which is the number of samples this technique uses.
    // This division also appears a few lines below in the framebuffer accumulation.
    const float wLight = mis(cameraPdfA / gLightSubpathCount) * (gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

	// Partial eye sub-path weight is 0 [tech. rep. (47)]

	// Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
	const float misWeight = 1.f / (wLight + 1.f);
	#else
    const float misWeight = 1;
	#endif

    #ifdef gDeferShadowRays
        ShadowRay ray;
        ray.mContribution = contrib;
        ray.mWeight = misWeight;
        ray.mRayOrigin = OffsetRayOrigin(vertex, toCamera);
        ray.mViewVertices = 1;
        ray.mRayDirection = toCamera;
        ray.mDistance = dist;
        ray.mPixelIndex = int(pixelCoord.y) * gOutputSize.x + int(pixelCoord.x);
        ray.mLightVertices = (int)sPathState.mPathLength + 1;
		uint appendIndex;
		InterlockedAdd(gCounters[1], 1, appendIndex);
		gShadowRays[appendIndex] = ray;
    #else
        float lightPdf, hitDist;
        if (TraceRay(MakeRay(OffsetRayOrigin(vertex, toCamera), toCamera, 0, dist), false, lightPdf, hitDist).mIsSurface)
            return;

        AddRadianceInterlocked(uint2(pixelCoord), contrib, misWeight, 1, (int)sPathState.mPathLength + 1);
    #endif
}

// ---------------------------------------------------------------------------------------------------------
// ------------------------------ Direct illumination / Next Event Estimation ------------------------------
// ---------------------------------------------------------------------------------------------------------

void ConnectToLight(const PathVertex vertex, const float3 localDirIn) {
	float lightPdf;
    const PathVertex lightVertex = SampleEmission(sPathState.mRng.NextFloat(), lightPdf);
    if (all(lightVertex.mMaterial.Emission() <= 0) || lightPdf <= 0)
		return;

    float3 toLight;
    float dist;
    float cosLight;
    if (lightVertex.mIsSurface) {
        toLight = lightVertex.mPosition - vertex.mPosition;
        dist = length(toLight);
        toLight /= dist;
        cosLight = -dot(lightVertex.mShadingNormal, toLight);
        if (cosLight <= 0) return;
    } else {
        toLight = lightVertex.mPosition;
        dist = POS_INFINITY;
        cosLight = 1;
    }

    float AtoW = 1;
    if (lightVertex.mIsSurface)
		AtoW = sqr(dist) / cosLight;
    const float directPdfW = lightPdf * AtoW;

    const float3 localDirOut = vertex.ToLocal(toLight);

    ReflectanceEvalRecord f = EvalBrdf(vertex.mMaterial, localDirIn, localDirOut, false);
    f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
	const float contProb = GetContinuationProbability(vertex);
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

    const float3 contrib = (sPathState.mThroughput * lightVertex.mMaterial.Emission() * f.mReflectance);

    // compute mis weight

    #if defined(gUseVC) || defined(gUseVM)
    const float emissionPdfW = lightVertex.mIsSurface ? lightPdf * CosHemispherePdfW(cosLight) : (lightPdf * ConcentricDiscPdfA() / sqr(gSceneSphere.w));

    // Partial light sub-path MIS weight [tech. rep. (44)].
    // Note that wLight is a ratio of area pdfs. But since both are on the
    // light source, their distance^2 and cosine terms cancel out.
    // Therefore we can write wLight as a ratio of solid angle pdfs,
    // both expressed w.r.t. the same shading point.
    const float wLight = mis(f.mFwdPdfW / directPdfW);

    // Partial eye sub-path MIS weight [tech. rep. (45)].
    //
    // In front of the sum in the parenthesis we have Mis(ratio), where
    //    ratio = emissionPdfA / directPdfA,
    // with emissionPdfA being the product of the pdfs for choosing the
    // point on the light source and sampling the outgoing direction.
    // What we are given by the light source instead are emissionPdfW
    // and directPdfW. Converting to area pdfs and plugging into ratio:
    //    emissionPdfA = emissionPdfW * cosToLight / dist^2
    //    directPdfA   = directPdfW * cosAtLight / dist^2
    //    ratio = (emissionPdfW * cosToLight / dist^2) / (directPdfW * cosAtLight / dist^2)
    //    ratio = (emissionPdfW * cosToLight) / (directPdfW * cosAtLight)
    //
    // Also note that both emissionPdfW and directPdfW should be
    // multiplied by lightPickProb, so it cancels out.
    const float wCamera = mis(emissionPdfW * abs(localDirOut.z) / (directPdfW * cosLight)) * (gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

    const float misWeight = 1 / (wLight + 1 + wCamera);
    #elif defined(gSampleDirectIlluminationOnly)
    const float misWeight = 1;
    #else
    // mis with BSDF sampling
    const float misWeight = mis(directPdfW, f.mFwdPdfW);
    #endif

	// evaluate transmittance along shadow ray

    TraceConnectionRay(OffsetRayOrigin(vertex, toLight), toLight, AdjustIntersectionDistance(dist), contrib, misWeight / (sPathState.mPdf * directPdfW), sPathState.mPathLength + 1, 1);
}


// ---------------------------------------------------------------------------------------------------------
// ------------------------------------- Direct radiance contribution --------------------------------------
// ---------------------------------------------------------------------------------------------------------

void AddEmission(const float3 emission, const float directPdfA, const float emissionPdfW, const float directPdfW) {
	float w = 1;
	#if defined(gUseVC) || defined(gUseVM)
	if (sPathState.mPathLength > 1 && !sPathState.isSpecular) {
		// Partial eye sub-path MIS weight [tech. rep. (43)].
		// If the last hit was specular, then dVCM == 0.
		const float wCamera = mis(directPdfA) * sPathState.dVCM + mis(emissionPdfW) * sPathState.dVC;
		w = 1 / (1 + wCamera);
	}
	#elif defined(gSampleDirectIllumination)
	w = sPathState.mDirPdfW == 0 ? 1 : mis(sPathState.mDirPdfW, directPdfW);
	#endif

    OutputRadiance(
		emission,
		w/sPathState.mPdf,
		sPathState.mPathLength + 1,
		0);
}



// ---------------------------------------------------------------------------------------------------------
// --------------------------------------------- Path tracing ----------------------------------------------
// ---------------------------------------------------------------------------------------------------------

bool InitializeCameraPath(const uint2 index) {
    sPathState.mRng = RandomSampler(gRandomSeed, index);

    PackedVertex v = reinterpret<PackedVertex>(gVertices[index]);
	if (v.mInstanceIndex == INVALID_INSTANCE) return false;
	const float3 dir = normalize(v.mPosition - gCameraPosition);
    float2 uv;
    const float3 localDir = TransformVector(gWorldToCamera, dir);

    sPathState.mRayOrigin = gCameraPosition;
    sPathState.mPathLength = 0;
    sPathState.mThroughput = 1;
    sPathState.mPdf = 1;
    sPathState.mRayDirection = dir;
	sPathState.mDirPdfW = 0;

	sPathState.isSpecular = true;
	sPathState.isRcvValid = false;

	#if defined(gUseVC) || defined(gUseVM)
	// Compute pdf conversion factor from area on image plane to solid angle on ray
	const float cosAtCamera = abs(localDir.z);
	const float imagePointToCameraDist = gImagePlaneDist / cosAtCamera;
	const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

	// We put the virtual image plane at such a distance from the camera origin
	// that the pixel area is one and thus the image plane sampling pdf is 1.
	// The solid angle ray pdf is then equal to the conversion factor from
	// image plane area density to ray solid angle density
	const float cameraPdfW = imageToSolidAngleFactor;


	// Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
	// The evaluation is completed after tracing the camera ray in the eye sub-path loop.
	sPathState.dVCM = mis(gLightSubpathCount / cameraPdfW);
	sPathState.dVC = 0;
	sPathState.dVM = 0;
	#endif

	return true;
}
void InitializeLightPath(const uint2 index) {
    sPathState.mRng = RandomSampler(gRandomSeed, index + gOutputSize);
	float directPdfA;
    const PathVertex lightVertex = SampleEmission(sPathState.mRng.NextFloat(), directPdfA);
    sPathState.isFiniteLight = lightVertex.mIsSurface;
	sPathState.isRcvValid = false;

    sPathState.mThroughput = lightVertex.mMaterial.Emission();

    float emissionPdfW = directPdfA;
	float cosLight;

    const float2 dirRng = sPathState.mRng.NextFloat().xy;
    if (sPathState.isFiniteLight) {
		// area light
		const float3 dir = SampleCosHemisphere(dirRng.xy);
		sPathState.mRayDirection = lightVertex.ToWorld(dir);
        sPathState.mRayOrigin = OffsetRayOrigin(lightVertex, sPathState.mRayDirection);
        sPathState.mThroughput *= dir.z;
        emissionPdfW *= CosHemispherePdfW(dir.z);
        cosLight = dir.z;
    } else {
        // environment light
        const float2 xy = SampleConcentricDisc(dirRng.xy);
        float3 t, b;
        MakeOrthonormal(lightVertex.mPosition, t, b);
        sPathState.mRayOrigin = gSceneSphere.xyz + gSceneSphere.w * (t * xy.x + b * xy.y + lightVertex.mPosition);
        sPathState.mRayDirection = -lightVertex.mPosition;
        emissionPdfW *= ConcentricDiscPdfA() / sqr(gSceneSphere.w);
        cosLight = 1;
	}

	#if defined(gUseVC) || defined(gUseVM)
    sPathState.dVCM = mis(directPdfA / emissionPdfW);
    sPathState.dVC = lightVertex.mIsSurface ? mis(cosLight / emissionPdfW) : 0;
    sPathState.dVM = sPathState.dVC * gMisVcWeightFactor;
	#endif

    sPathState.mPathLength = 0;
    sPathState.mPdf = emissionPdfW;
}

bool ExtendPath<let bFromLight : bool>() {
	float isectLightPdf, isectDist;
	const PathVertex vertex = TraceRay(MakeRay(sPathState.mRayOrigin, sPathState.mRayDirection), true, isectLightPdf, isectDist);
    sPathState.mPathLength++;

    if (!vertex.mIsSurface) {
        if (!bFromLight && sPathState.mPathLength >= gMinDepth) {
			const float3 emission = sPathState.mThroughput * vertex.mMaterial.Emission();
            if (any(emission > 0)) {
                AddEmission(emission,
					ConcentricDiscPdfA() / sqr(gSceneSphere.w),
					isectLightPdf,
					isectLightPdf);
            }
    	}
		return false;
	}


    const float3 localDirIn = vertex.ToLocal(-sPathState.mRayDirection);

	#if defined(gUseVC) || defined(gUseVM)
    // Update the MIS quantities. Implement equations
    // [tech. rep. (31)-(33)] or [tech. rep. (34)-(36)], respectively.

    // Infinite lights use MIS handled via solid angle integration,
    // so do not divide by the distance for such lights [tech. rep. Section 5.1]
    if (!bFromLight || (sPathState.mPathLength > 1 || sPathState.isFiniteLight == 1))
		sPathState.dVCM *= mis(sqr(isectDist));
	sPathState.dVCM /= mis(abs(localDirIn.z));
	sPathState.dVC  /= mis(abs(localDirIn.z));
	sPathState.dVM  /= mis(abs(localDirIn.z));
	#endif

    if (!bFromLight) {
		// surface lights are one-sided
        if (sPathState.mPathLength >= gMinDepth && localDirIn.z > 0) {
            const float3 emission = sPathState.mThroughput * vertex.mMaterial.Emission();
            if (any(emission > 0)) {
                AddEmission(emission,
					isectLightPdf,
					isectLightPdf * CosHemispherePdfW(localDirIn.z),
					isectLightPdf * sqr(isectDist)/abs(localDirIn.z));
			}
        }
    }

    if (sPathState.mPathLength >= gMaxDepth)
        return false;

    if (!any(vertex.mMaterial.BaseColor() > 0))
        return false;

    if (IsDiffuse(vertex)) {
		if (bFromLight) {
			#ifndef gUsePpm
			if (sPathState.mPathLength + 1 >= gMinDepth)
                ConnectToCamera(vertex, localDirIn);
			#endif

			#ifndef gTraceOffsetPath
            #ifdef gUseVC
			if (sPathState.mPathLength + 2 <= gMaxDepth)
				StoreLightVertex(vertex, localDirIn);
			#elif defined(gUseVM)
			if (sPathState.mPathLength + 1 <= gMaxDepth)
				StoreLightVertex(vertex, localDirIn);
			#endif
			#endif
        } else {
			#if defined(gSampleDirectIllumination) || defined(gUseVC)
			if (sPathState.mPathLength + 1 >= gMinDepth)
				ConnectToLight(vertex, localDirIn);
			#endif

			#ifdef gUseVC
				ConnectToLightSubpath(vertex, localDirIn);
			#endif

            #ifdef gUseVM
                MergeWithLightVertices(vertex, localDirIn);
            #ifdef gUsePpm
            return false;
            #endif
            #endif
        }
    }

	#ifdef gSampleDirectIlluminationOnly
    return false;
	#endif

    if (bFromLight && sPathState.mPathLength + 1 == gMaxDepth)
        return false;

	const float continueProb = min(1, GetContinuationProbability(vertex));
	if (sPathState.mRng.NextFloat().x > continueProb)
		return false;

    DirectionSampleRecord s = SampleBrdf(vertex.mMaterial, sPathState.mRng.NextFloat(), localDirIn, bFromLight);
    s.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, s.mDirection, bFromLight);
    s.mFwdPdfW *= continueProb;
    s.mRevPdfW *= continueProb;

	sPathState.mThroughput *= s.mReflectance;
	sPathState.mPdf *= s.mFwdPdfW;

	#if defined(gUseVC) || defined(gUseVM)
    // Sub-path MIS quantities for the next vertex. Only partial - the
    // evaluation is completed when the actual hit point is known,
    // i.e. after tracing the ray, in the sub-path loop.
    const float cosOut = abs(s.mDirection.z);
    if (s.mRoughness <= 1e-4) {
        // Specular scattering case [tech. rep. (53)-(55)] (partially, as noted above)
        sPathState.dVCM = 0.f;
        sPathState.dVC *= mis(cosOut);
        sPathState.dVM *= mis(cosOut);
    } else {
		sPathState.isSpecular = false;
        // Implements [tech. rep. (34)-(36)] (partially, as noted above)
        sPathState.dVCM = mis(1.f / s.mFwdPdfW);
        sPathState.dVC = mis(cosOut / s.mFwdPdfW) * (sPathState.dVC * mis(s.mRevPdfW) + sPathState.dVCM + gMisVmWeightFactor);
        sPathState.dVM = mis(cosOut / s.mFwdPdfW) * (sPathState.dVM * mis(s.mRevPdfW) + sPathState.dVCM * gMisVcWeightFactor + 1);
	}
	#elif defined(gSampleDirectIllumination)
	sPathState.mDirPdfW = s.mFwdPdfW;
	#endif

	sPathState.mRayDirection = vertex.ToWorld(s.mDirection);
	sPathState.mRayOrigin = OffsetRayOrigin(vertex, sPathState.mRayDirection);

    return !any(isnan(sPathState.mThroughput));
}

#ifdef gTraceFromLight
#define bFromLight true
#else
#define bFromLight false
#endif

[shader("compute")]
[numthreads(8, 4, 1)]
void Render(uint3 index: SV_DispatchThreadID) {
    sPixelIndex = index.xy;
    InitDebugPixel(bFromLight ? -1 : sPixelIndex, gOutputSize);

	#ifdef gTraceFromLight
		if (sPixelIndex.y * gOutputSize.x + sPixelIndex.x >= gLightSubpathCount)
			return;
		InitializeLightPath(sPixelIndex);
	#else
		if (any(sPixelIndex >= gOutputSize))
			return;
		if (!InitializeCameraPath(sPixelIndex))
			return;
	#endif


	#ifdef gMultiDispatch

		if (!ExtendPath<bFromLight>())
			sPathState.mThroughput = 0;
		StorePathState(sPixelIndex.y * gOutputSize.x + sPixelIndex.x);

	#else

        while (sPathState.mPathLength + 1 <= gMaxDepth) {
			if (!ExtendPath<bFromLight>()) {
				sPathState.mThroughput = 0;
				break;
			}
		}

	#endif
}

#ifdef gMultiDispatch

[shader("compute")]
[numthreads(8, 4, 1)]
void RenderIteration(uint3 index: SV_DispatchThreadID) {
	sPixelIndex = index.xy;
	InitDebugPixel(bFromLight ? -1 : sPixelIndex, gOutputSize);

	#ifdef gTraceFromLight
    if (sPixelIndex.y * gOutputSize.x + sPixelIndex.x >= gLightSubpathCount)
		return;
	#else
    if (any(sPixelIndex >= gOutputSize))
        return;
	#endif

    LoadPathState(sPixelIndex.y * gOutputSize.x + sPixelIndex.x);

    if (all(sPathState.mThroughput <= 0) || sPathState.mPathLength + 1 > gMaxDepth)
		return;

	if (!ExtendPath<bFromLight>())
		sPathState.mThroughput = 0;

	StorePathState(sPixelIndex.y * gOutputSize.x + sPixelIndex.x);
}

#endif

[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessShadowRays(uint3 index: SV_DispatchThreadID) {
    InitDebugPixel(-1, gOutputSize);
    const uint idx = index.y * gOutputSize.x + index.x;
    if (idx >= gCounters[1]) return;

    const ShadowRay ray = gShadowRays[idx];

	float tmp1, tmp2;
    if (TraceRay(MakeRay(ray.mRayOrigin, ray.mRayDirection, 0, ray.mDistance), false, tmp1, tmp2).mIsSurface)
        return;

    AddRadianceInterlocked(ray.GetPixelCoord(), ray.mContribution, ray.mWeight, ray.mViewVertices, ray.mLightVertices);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessAtomicOutput(uint3 index: SV_DispatchThreadID) {
    InitDebugPixel(-1, gOutputSize);
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

    float3 color = 0;

	#ifdef gCopyAtomic
    uint4 c = gOutputAtomic.Load4(16 * (id.y * gOutputSize.x + id.x));
	if ((c.w & BIT(0)) != 0) c.r = 0xFFFFFFFF;
	if ((c.w & BIT(1)) != 0) c.g = 0xFFFFFFFF;
    if ((c.w & BIT(2)) != 0) c.b = 0xFFFFFFFF;
	color += c.rgb / (float)gRadianceQuantizationFactor;
	#endif

	gOutput[id].rgb += color;
}

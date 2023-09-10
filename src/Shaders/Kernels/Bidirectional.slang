#include "Intersection.slang"
#include "Random.slang"
#include "Light.slang"
#include "BRDF.slang"

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
    uint gLightSubpathCount;
    float gImagePlaneDist;
    uint gMaxDepth;
    uint gDebugPathLengths;
	uint gRandomSeed;
};

static const float4 gSceneSphere = float4(gScene.mSceneMax + gScene.mSceneMin, length(gScene.mSceneMax - gScene.mSceneMin)) / 2;

#define gRadianceQuantizationFactor 16384

struct PackedLightVertex {
    PackedVertex mVertex;
    float3 mThroughput;
    uint mPackedLocalDirIn;
    uint mPathLength;
    float dVC;
    float dVCM;
    float pad;

    uint getInstanceIndex()  { return mVertex.mInstanceIndex; }
    uint getPrimitiveIndex() { return mVertex.mPrimitiveIndex; }
    float3 getLocalDirIn() { return UnpackNormal(mPackedLocalDirIn); }
};

struct PathState {
    float3 mRayOrigin;
    uint mPathLength;
    float3 mThroughput;
	uint mInstancePrimitiveIndex;
    RandomSampler mRng;
	uint4 mPackedData;

    property float dVCM {
        get { return f16tof32(BF_GET(mPackedData[1], 0, 16)); }
        set { BF_SET(mPackedData[1], f32tof16(newValue), 0, 16); }
    }
    property float dVC {
        get { return f16tof32(BF_GET(mPackedData[1], 16, 16)); }
        set { BF_SET(mPackedData[1], f32tof16(newValue), 16, 16); }
    }

    property float mIntersectionDistance {
        get { return f16tof32(BF_GET(mPackedData[2], 0, 16)); }
        set { BF_SET(mPackedData[2], f32tof16(newValue), 0, 16); }
    }
    property float mIntersectionPdfA {
        get { return f16tof32(BF_GET(mPackedData[2], 16, 16)); }
        set { BF_SET(mPackedData[2], f32tof16(newValue), 16, 16); }
    }

    // fwd pdf of sampling the ray direction at last bounce
    property float mDirPdfW {
        get { return f16tof32(BF_GET(mPackedData[3], 0, 16)); }
        set { BF_SET(mPackedData[3], f32tof16(newValue), 0, 16); }
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

    property float3 mRayDirection {
        get { return UnpackNormal(mPackedData[0]); }
        set { mPackedData[0] = PackNormal(newValue); }
    }
    property uint mInstanceIndex  { get { return BF_GET(mInstancePrimitiveIndex, 0, 16); } }
    property uint mPrimitiveIndex { get { return BF_GET(mInstancePrimitiveIndex, 16, 16); } }
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

    uint2 getPixelCoord() {
        return uint2(mPixelIndex % gOutputSize.x, mPixelIndex / gOutputSize.x);
    }
};

RWStructuredBuffer<PathState> gPathStates;
RWStructuredBuffer<PackedLightVertex> gLightVertices;
RWStructuredBuffer<ShadowRay> gShadowRays;
RWStructuredBuffer<uint> gCounters; // 0 -> light vertices, 1 -> shadow rays

Texture2D<uint4> gVertices;
RWTexture2D<float4> gOutput;
RWByteAddressBuffer gOutputAtomic;
uniform float4x4 gCameraToWorld;
uniform float4x4 gWorldToCamera;
uniform float4x4 gProjection;
uniform float3 gCameraPosition;

static PathState sPathState;
static uint2 sPixelIndex;

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
			overflowMask |= (1 << i);
	gOutputAtomic.InterlockedOr(16*index + 12, overflowMask);
}

void AddRadiance(const uint2 index, const float3 radiance, const float weight, const uint viewVertices, const uint lightVertices) {
	if (weight <= 0 || all(radiance <= 0))
		return;

	#ifdef gDebugPaths
		if (BF_GET(gDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gDebugPathLengths, 16, 16) == lightVertices) {
			#ifdef gDebugPathWeights
				gOutput[index] += float4(weight, weight, weight, 0);
			#else
				gOutput[index] += float4(radiance, 0);
			#endif
		}
	#else
		gOutput[index] += float4(radiance * weight, 0);
	#endif
}

void AddRadianceInterlocked(const uint2 index, const float3 radiance, const float weight, const uint viewVertices, const uint lightVertices) {
	if (weight <= 0 || all(radiance <= 0))
		return;

	#ifdef gDebugPaths
		if (BF_GET(gDebugPathLengths, 0, 16) == viewVertices + lightVertices && BF_GET(gDebugPathLengths, 16, 16) == lightVertices) {
			#ifdef gDebugPathWeights
				AddRadianceInterlocked(index, weight);
			#else
				AddRadianceInterlocked(index, radiance);
			#endif
		}
	#else
		AddRadianceInterlocked(index, radiance * weight);
	#endif
}

float mis(const float a) {
    return a * a;
}
float mis(const float a, const float b) {
    const float a2 = mis(a);
    return a2 / (a2 + mis(b));
}

#define gMisVmWeightFactor 0

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

        AddRadiance(sPixelIndex, contrib, weight, viewVertices, lightVertices);
    #endif
}


struct ReflectanceEvalRecord {
    float3 mReflectance;
    float mFwdPdfW;
    float mRevPdfW;
};
struct DirectionReflectanceSample {
    float3 mDirection;
    float3 mReflectance;
    float mFwdPdfW;
    float mRevPdfW;
    float mEta;
    float mRoughness;

    bool isSingular() { return mRoughness == 0; }
};
ReflectanceEvalRecord EvaluateReflectance<let bAdjoint : bool>(const PathVertex vertex, const float3 localDirIn, const float3 localDirOut) {
    ReflectanceEvalRecord r;
	r.mReflectance = EvalBrdfLocal(vertex.mMaterial, localDirIn, localDirOut, bAdjoint);
	r.mFwdPdfW = EvalBrdfPdfWLocal(vertex.mMaterial, localDirIn, localDirOut);
	r.mRevPdfW = EvalBrdfPdfWLocal(vertex.mMaterial, localDirOut, localDirIn);
	return r;
}
DirectionReflectanceSample SampleDirection<let bAdjoint : bool>(const PathVertex vertex, const float4 rnd, const float3 localDirIn) {
    DirectionReflectanceSample r;
	DirectionSampleRecord s;
	SampleBrdfLocal(vertex.mMaterial, rnd, localDirIn, s);
	r.mDirection = s.mDirection;
	r.mEta = s.mEta;
	r.mRoughness = s.mRoughness;

	const ReflectanceEvalRecord f = EvaluateReflectance<bAdjoint>(vertex, localDirIn, r.mDirection);
	r.mReflectance = f.mReflectance;
	r.mFwdPdfW = f.mFwdPdfW;
	r.mRevPdfW = f.mRevPdfW;
	return r;
}

// ---------------------------------------------------------------------------------------------------------
// ----------------------------------- Bidirectional vertex connections ------------------------------------
// ---------------------------------------------------------------------------------------------------------

void StoreLightVertex(const PathVertex vertex, const float3 localDirIn) {
    PackedLightVertex v;
	v.mVertex = PackVertex(vertex);
	v.mThroughput = sPathState.mThroughput;
	v.mPackedLocalDirIn = PackNormal(localDirIn);
    v.mPathLength = sPathState.mPathLength;
    v.dVC  = sPathState.dVC;
    v.dVCM = sPathState.dVCM;
    uint idx;
    InterlockedAdd(gCounters[0], 1, idx);
    gLightVertices[idx] = v;
}

struct LightVertexSampleRecord {
    float3 mThroughput;
    float mFwdPdfW;
    float mRevPdfW;
    float3 mReflectance;
    float mCosLight;
    float3 mDirectionToLight;
    float mDistanceToLight;
    uint mSubpathLength;
    float dVC;
    float dVCM;
};
LightVertexSampleRecord SampleLightVertex(const float3 referencePosition, const uint rnd) {
    if (gCounters[0] == 0)
        return { 0 };

    const PackedLightVertex v = gLightVertices[rnd % gCounters[0]];

	if (sPathState.mPathLength + 1 + v.mPathLength > gMaxDepth)
		return { 0 };

    const PathVertex vertex = UnpackVertex(v.mVertex);

    LightVertexSampleRecord r;
	r.mThroughput = v.mThroughput;
    r.mDirectionToLight = vertex.mPosition - referencePosition;
    r.mDistanceToLight = length(r.mDirectionToLight);
    r.mDirectionToLight /= r.mDistanceToLight;
	r.mSubpathLength = v.mPathLength;

    const float3 localDirOut = vertex.ToLocal(-r.mDirectionToLight);
    r.mCosLight = abs(localDirOut.z);
    ReflectanceEvalRecord f = EvaluateReflectance<true>(vertex, v.getLocalDirIn(), localDirOut);
    f.mReflectance *= vertex.GetShadingNormalCorrection(v.getLocalDirIn(), localDirOut, true);
    float contProb = GetContinuationProbability(vertex);

	r.mReflectance = f.mReflectance;
	r.mFwdPdfW = f.mFwdPdfW * contProb;
    r.mRevPdfW = f.mRevPdfW * contProb;

    r.dVC = v.dVC;
    r.dVCM = v.dVCM;

    return r;
}
void ConnectToLightSubpath( const PathVertex vertex, const float3 localDirIn) {
    const LightVertexSampleRecord r = SampleLightVertex(vertex.mPosition, sPathState.mRng.Next().x);
    if (all(r.mThroughput <= 0) || all(r.mReflectance <= 0))
        return;


    const float3 localDirOut = vertex.ToLocal(r.mDirectionToLight);

    ReflectanceEvalRecord f = EvaluateReflectance<false>(vertex, localDirIn, localDirOut);
    f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
    const float contProb = GetContinuationProbability(vertex);
    f.mFwdPdfW *= contProb;
    f.mRevPdfW *= contProb;

    // the LVC paper connects to lightVertexCount/lightPathCount light vertices
    // we only connect to one light vertex, therefore we weight its contribution by lightVertexCount/lightPathCount
    const float w = (gCounters[0] / (float)gLightSubpathCount);

    const float invDist2 = 1 / sqr(r.mDistanceToLight);
    const float3 contrib = (sPathState.mThroughput * r.mThroughput) * ((f.mReflectance * r.mReflectance) * invDist2 * w);

	// Convert pdfs to area pdf
	const float cameraBsdfDirPdfA = f.mFwdPdfW * r.mCosLight * invDist2;
	const float lightBsdfDirPdfA  = r.mFwdPdfW * abs(localDirOut.z) * invDist2;

    // Partial light sub-path MIS weight [tech. rep. (40)]
    const float wLight = mis(cameraBsdfDirPdfA) * (gMisVmWeightFactor + r.dVCM + r.dVC * mis(r.mRevPdfW));

    // Partial eye sub-path MIS weight [tech. rep. (41)]
    const float wCamera = mis(lightBsdfDirPdfA) * (gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

	const float misWeight = 1 / (wLight + 1.f + wCamera);

    TraceConnectionRay(OffsetRayOrigin(vertex, r.mDirectionToLight), r.mDirectionToLight, AdjustIntersectionDistance(r.mDistanceToLight), contrib, misWeight, sPathState.mPathLength+1, r.mSubpathLength+1);
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

    ReflectanceEvalRecord f = EvaluateReflectance<true>(vertex, localDirIn, localDirOut);
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
    float3 contrib = (sPathState.mThroughput * f.mReflectance) / (gLightSubpathCount * surfaceToImageFactor * cosOut);

	#ifdef gUseVC
    // Partial light sub-path weight [tech. rep. (46)]. Note the division by
    // mLightPathCount, which is the number of samples this technique uses.
    // This division also appears a few lines below in the framebuffer accumulation.
    const float wLight = mis(cameraPdfA / gLightSubpathCount) * (
		gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

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

    ReflectanceEvalRecord f = EvaluateReflectance<false>(vertex, localDirIn, localDirOut);
    f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
	const float contProb = GetContinuationProbability(vertex);
	f.mFwdPdfW *= contProb;
	f.mRevPdfW *= contProb;

    const float3 contrib = (sPathState.mThroughput * lightVertex.mMaterial.Emission() * f.mReflectance) / directPdfW;

    // compute mis weight

    #ifdef gUseVC
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
    const float wCamera = mis(emissionPdfW * abs(localDirOut.z) / (directPdfW * cosLight)) * (
        gMisVmWeightFactor + sPathState.dVCM + sPathState.dVC * mis(f.mRevPdfW));

    const float misWeight = 1 / (wLight + 1 + wCamera);
    #elif defined(gSampleDirectIlluminationOnly)
    const float misWeight = 1;
    #else
    // mis with BSDF sampling
    const float misWeight = mis(lightPdf, f.mFwdPdfW / AtoW);
    #endif

	// evaluate transmittance along shadow ray

    TraceConnectionRay(OffsetRayOrigin(vertex, toLight), toLight, AdjustIntersectionDistance(dist), contrib, misWeight, sPathState.mPathLength + 1, 1);
}


// ---------------------------------------------------------------------------------------------------------
// ------------------------------------- Direct radiance contribution --------------------------------------
// ---------------------------------------------------------------------------------------------------------

void AddEmission(const float3 emission, const float directPdfA, const float emissionPdfW, const float WtoA) {
	float w = 1;
	if (sPathState.mPathLength > 1 && !sPathState.isSpecular) {
		#ifdef gUseVC
		// Partial eye sub-path MIS weight [tech. rep. (43)].
		// If the last hit was specular, then dVCM == 0.
		const float wCamera = mis(directPdfA) * sPathState.dVCM + mis(emissionPdfW) * sPathState.dVC;
		w = 1 / (1 + wCamera);
		#elif defined(gSampleDirectIllumination)
        if (sPathState.mDirPdfW > 0)
            w = mis(sPathState.mDirPdfW*WtoA, directPdfA);
		#endif
	}

    AddRadiance(
        sPixelIndex,
		emission,
		w,
		sPathState.mPathLength + 1,
		0);
}



// ---------------------------------------------------------------------------------------------------------
// --------------------------------------------- Path tracing ----------------------------------------------
// ---------------------------------------------------------------------------------------------------------

bool InitializeCameraPath(const uint2 index) {
    sPathState.mRng = RandomSampler(gRandomSeed, index);

	#ifdef gPixelJitter
    const float2 offset = sPathState.mRng.nextFloat().xy;
	#else
    const float2 offset = 0.5;
	#endif

    PackedVertex v = reinterpret<PackedVertex>(gVertices[index]);
	if (v.mInstanceIndex == INVALID_INSTANCE) return false;
	const float3 dir = normalize(TransformPoint(gScene.mInstanceTransforms[v.mInstanceIndex], v.mLocalPosition) - gCameraPosition);
    float2 uv;
    const float3 localDir = TransformVector(gWorldToCamera, dir);

    sPathState.mRayOrigin = gCameraPosition;
    sPathState.mPathLength = 0;
    sPathState.mThroughput = 1;
    sPathState.mRayDirection = dir;

	#ifdef gUseVC
		// Compute pdf conversion factor from area on image plane to solid angle on ray
		const float cosAtCamera = abs(localDir.z);
		const float imagePointToCameraDist = gImagePlaneDist / cosAtCamera;
		const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

		// We put the virtual image plane at such a distance from the camera origin
		// that the pixel area is one and thus the image plane sampling pdf is 1.
		// The solid angle ray pdf is then equal to the conversion factor from
		// image plane area density to ray solid angle density
		const float cameraPdfW = imageToSolidAngleFactor;

		sPathState.mDirPdfW = cameraPdfW;

		sPathState.isSpecular = true;

		// Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
		// The evaluation is completed after tracing the camera ray in the eye sub-path loop.
		sPathState.dVCM = mis(gLightSubpathCount / cameraPdfW);
		sPathState.dVC = 0;
	#endif

	return true;
}
void InitializeLightPath(const uint2 index) {
    sPathState.mRng = RandomSampler(gRandomSeed, index + gOutputSize);
	float directPdfA;
    const PathVertex lightVertex = SampleEmission(sPathState.mRng.NextFloat(), directPdfA);
    sPathState.isFiniteLight = lightVertex.mIsSurface;

    sPathState.mThroughput = lightVertex.mMaterial.Emission();

    float emissionPdfW = directPdfA;
	float cosLight;

    const float2 dirRng = sPathState.mRng.NextFloat().xy;
    if (sPathState.isFiniteLight) {
		// area light
		const float3 dir = SampleCosHemisphere(dirRng.xy);
        sPathState.mDirPdfW = CosHemispherePdfW(dir.z);
		sPathState.mRayDirection = lightVertex.ToWorld(dir);
        sPathState.mRayOrigin = OffsetRayOrigin(lightVertex, sPathState.mRayDirection);
        sPathState.mThroughput *= dir.z;
        emissionPdfW *= sPathState.mDirPdfW;
        cosLight = dir.z;
    } else {
        // environment light
        const float2 xy = SampleConcentricDisc(dirRng.xy);
        const float3x3 onb = MakeOrthonormal(lightVertex.mPosition);
        sPathState.mRayOrigin = gSceneSphere.xyz + gSceneSphere.w * (onb[0] * xy.x + onb[1] * xy.y + lightVertex.mPosition);
        sPathState.mRayDirection = -lightVertex.mPosition;
        sPathState.mDirPdfW = directPdfA;
        emissionPdfW *= ConcentricDiscPdfA() / sqr(gSceneSphere.w);
        cosLight = 1;
	}

	sPathState.mThroughput /= emissionPdfW;

	#ifdef gUseVC
    sPathState.dVCM = mis(directPdfA / emissionPdfW);
    sPathState.dVC  = lightVertex.mIsSurface ? mis(cosLight / emissionPdfW) : 0;
	#endif

    sPathState.mPathLength = 0;
}

PathVertex TraceRay<let bFromLight : bool>() {
	float lightPdf, dist;
	const PathVertex v = TraceRay(MakeRay(sPathState.mRayOrigin, sPathState.mRayDirection), true, sPathState.mIntersectionPdfA, sPathState.mIntersectionDistance);
    sPathState.mPathLength++;
    sPathState.mInstancePrimitiveIndex = v.mInstancePrimitiveIndex;

    if (!v.mIsSurface) {
        if (!bFromLight && sPathState.mPathLength > 1) {
			const float3 emission = sPathState.mThroughput * v.mMaterial.Emission();

            if (any(emission > 0)) {
                AddEmission(emission,
					ConcentricDiscPdfA() / sqr(gSceneSphere.w),
					lightPdf,
					1);
            }
    	}
		sPathState.mThroughput = 0;
	}

	return v;
}

bool ProcessInteraction<let bFromLight : bool>(const PathVertex vertex) {
    const float3 localDirIn = vertex.ToLocal(-sPathState.mRayDirection);
    const float cosIn = abs(localDirIn.z);

    if (bFromLight) {
		#ifdef gUseVC
		// Update the MIS quantities before storing them at the vertex.
		// These updates follow the initialization in GenerateLightSample() or
		// SampleScattering(), and together implement equations [tech. rep. (31)-(33)]
		// or [tech. rep. (34)-(36)], respectively.

		// Infinite lights use MIS handled via solid angle integration,
		// so do not divide by the distance for such lights [tech. rep. Section 5.1]
		if (sPathState.mPathLength > 1 || sPathState.isFiniteLight == 1)
			sPathState.dVCM *= mis(sqr(sPathState.mIntersectionDistance));
		sPathState.dVCM /= mis(cosIn);
        sPathState.dVC  /= mis(cosIn);
		#endif
	} else {
		#ifdef gUseVC
		// Update the MIS quantities, following the initialization in
		// GenerateLightSample() or SampleScattering(). Implement equations
		// [tech. rep. (31)-(33)] or [tech. rep. (34)-(36)], respectively.
		sPathState.dVCM *= mis(sqr(sPathState.mIntersectionDistance));
		sPathState.dVCM /= mis(cosIn);
		sPathState.dVC  /= mis(cosIn);
		#endif

		// surface lights are one-sided
        if (sPathState.mPathLength > 1 && localDirIn.z > 0) {
            const float3 emission = sPathState.mThroughput * vertex.mMaterial.Emission();
            if (any(emission > 0)) {
                AddEmission(emission,
					sPathState.mIntersectionPdfA,
					sPathState.mIntersectionPdfA * CosHemispherePdfW(localDirIn.z),
					cosIn / sqr(sPathState.mIntersectionDistance));
			}
        }
    }

    if (sPathState.mPathLength >= gMaxDepth)
        return false;

    if (!any(vertex.mMaterial.BaseColor() > 0))
        return false;

    if (IsDiffuse(vertex)) {
		if (bFromLight) {
            #ifdef gUseVC
				if (sPathState.mPathLength + 2 <= gMaxDepth)
					StoreLightVertex(vertex, localDirIn);
			#endif
                ConnectToCamera(vertex, localDirIn);
        } else {
			#if defined(gSampleDirectIllumination) || defined(gUseVC)
				ConnectToLight(vertex, localDirIn);
			#endif
			#ifdef gUseVC
				ConnectToLightSubpath(vertex, localDirIn);
			#endif
        }
    }

	#ifdef gSampleDirectIlluminationOnly
    return false;
	#endif

	const float continueProb = min(1, GetContinuationProbability(vertex));
	if (sPathState.mRng.NextFloat().x > continueProb)
		return false;

    DirectionReflectanceSample s = SampleDirection<bFromLight>(vertex, sPathState.mRng.NextFloat(), localDirIn);
    s.mFwdPdfW *= continueProb;
    s.mRevPdfW *= continueProb;

    sPathState.mThroughput *= s.mReflectance / s.mFwdPdfW;

    const float cosOut = abs(s.mDirection.z);

    // Sub-path MIS quantities for the next vertex. Only partial - the
    // evaluation is completed when the actual hit point is known,
    // i.e. after tracing the ray, in the sub-path loop.
    if (s.isSingular()) {
        sPathState.mDirPdfW = 0;
		#ifdef gUseVC
			// Specular scattering case [tech. rep. (53)-(55)] (partially, as noted above)
			sPathState.dVCM = 0.f;
			sPathState.dVC *= mis(cosOut);
		#endif
    } else {
		sPathState.mDirPdfW = s.mFwdPdfW;
		#ifdef gUseVC
			#ifndef gTraceFromLight
				sPathState.isSpecular = false;
			#endif
			// Implements [tech. rep. (34)-(36)] (partially, as noted above)
			sPathState.dVC = mis(cosOut / s.mFwdPdfW) * (sPathState.dVC * mis(s.mRevPdfW) + sPathState.dVCM + gMisVmWeightFactor);
			sPathState.dVCM = mis(1.f / s.mFwdPdfW);
		#endif
	}

    sPathState.mThroughput *= vertex.GetShadingNormalCorrection(localDirIn, s.mDirection, bFromLight);
	sPathState.mRayDirection = vertex.ToWorld(s.mDirection);
	sPathState.mRayOrigin = OffsetRayOrigin(vertex, sPathState.mRayDirection);

    return !any(isnan(sPathState.mThroughput));
}


bool ExtendPath<let bFromLight : bool>() {
	PathVertex v = TraceRay<bFromLight>();
	if (all(sPathState.mThroughput <= 0))
		return false;
    return ProcessInteraction<bFromLight>(v);
}


void LoadPathState(const uint threadIndex) {
    sPathState = gPathStates[threadIndex];
}
void StorePathState(const uint threadIndex) {
	gPathStates[threadIndex] = sPathState;
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

[shader("compute")]
[numthreads(8, 4, 1)]
void RenderIteration(uint3 index: SV_DispatchThreadID) {
    sPixelIndex = index.xy;

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


[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessShadowRays(uint3 index: SV_DispatchThreadID) {
    const uint idx = index.y * gOutputSize.x + index.x;
    if (idx >= gCounters[1]) return;

    const ShadowRay ray = gShadowRays[idx];

	float tmp1, tmp2;
    if (TraceRay(MakeRay(ray.mRayOrigin, ray.mRayDirection, 0, ray.mDistance), false, tmp1, tmp2).mIsSurface)
        return;

    AddRadianceInterlocked(ray.getPixelCoord(), ray.mContribution, ray.mWeight, ray.mViewVertices, ray.mLightVertices);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessAtomicOutput(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

    uint4 c = gOutputAtomic.Load4(16 * (id.y * gOutputSize.x + id.x));
	if (c.w & (1 << 0)) c.r = 0xFFFFFFFF;
	if (c.w & (1 << 1)) c.g = 0xFFFFFFFF;
    if (c.w & (1 << 2)) c.b = 0xFFFFFFFF;
    const float3 v = c.rgb / (float)gRadianceQuantizationFactor;

	gOutput[id] += float4(v, 0);
}

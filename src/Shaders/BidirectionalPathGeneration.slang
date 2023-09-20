#pragma once

#include "PathReservoir.slang"
#include "Intersection.slang"
#include "Light.slang"
#include "BRDF.slang"
#include "HashGrid.slang"

#ifdef SAMPLE_LIGHTS
#define gSampleDirectIllumination true
#endif
#ifdef RECONNECTION
static const bool gReconnection = true;
#else
static const bool gReconnection = false;
#endif

#define gRadianceQuantizationFactor 16384

struct PackedLightVertex {
    PackedVertex mVertex;
    float3 mThroughput;
    uint mPackedLocalDirIn;
    uint mPathLength;
    float dVC;
    float dVCM;
    float mPdfA;

    uint getInstanceIndex()  { return mVertex.mInstanceIndex; }
    uint getPrimitiveIndex() { return mVertex.mPrimitiveIndex; }
    float3 getLocalDirIn() { return UnpackNormal(mPackedLocalDirIn); }
};

RWByteAddressBuffer gLightImage;
RWStructuredBuffer<uint> gLightVertexCount;
RWStructuredBuffer<PackedLightVertex> gLightVertices;

uniform float3 gCameraPosition;
uniform uint2 gOutputSize;

uniform float gReconnectionDistance;
uniform float gReconnectionRoughness;
uniform uint gMaxBounces;
uniform uint gRandomSeed;

uniform float4x4 gWorldToCamera;
uniform float4x4 gProjection;

uniform uint gLightSubpathCount;
uniform float3 gCameraForward;
uniform float3 gPrevCameraPosition;
uniform float3 gPrevCameraForward;
uniform float4 gSceneSphere;
uniform float gCameraImagePlaneDist;

uniform uint gDebugViewVertices;
uniform uint gDebugLightVertices;

static const uint gMinDepth = 2;
static const uint gMaxDepth = gMaxBounces + 1;
static const float gMisVmWeightFactor = 0;

float Mis(const float a) {
	return a * a;
}
float Mis(const float a, const float b) {
	const float a2 = Mis(a);
	return a2 / (a2 + Mis(b));
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

float AdjustIntersectionDistance(float x) { return x * (1 - 1 / 512.0); }

struct LightVertexSampleRecord {
	float3 mThroughput;
	float mPdfA;
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
LightVertexSampleRecord SampleLightVertex(const float3 referencePosition, const uint viewPathLength, const PackedLightVertex v) {
	const uint pathLength = viewPathLength + 1 + v.mPathLength;
	if (pathLength < gMinDepth || pathLength > gMaxDepth)
		return { 0 };

	const PathVertex vertex = UnpackVertex(v.mVertex);

	LightVertexSampleRecord r;
	r.mThroughput = v.mThroughput;
	r.mPdfA = v.mPdfA;
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

struct PathGenerator<let bAdjoint : bool, let bShift : bool> {
    float3 mRayOrigin;
    uint mPathLength;
    float3 mRayDirection;
    float mPathPdf;
    float3 mThroughput;
	uint mRngIdx;
    RandomSampler mRng;
    PathReservoir mReservoir;
    float mReplayPdfW;
    float mPrevReplayPdf;

    ReconnectionVertex rcv;
    float3 mThroughputAtRcv;

   	#define mBasePath mReservoir.mSample

    float mIntersectionDistance;
    float mIntersectionPdfA;

    float dVCM;
    float dVC;
	#ifndef gUseVC
    // fwd pdf of sampling the ray direction at last bounce
    float mDirPdfW;
	#endif

    // specifies if a light path came from a finite emitter (e.g. not the environment)
    bool isFiniteLight;
    // specifies if a view path is specular
    bool isSpecular;



	void AddRadianceInterlocked(uint2 id, float3 c) {
		const uint index = id.y * gOutputSize.x + id.x;
		uint3 ci = uint3(c * gRadianceQuantizationFactor);
		uint3 p;
		gLightImage.InterlockedAdd(16 * index + 0, ci.r, p.r);
		gLightImage.InterlockedAdd(16 * index + 4, ci.g, p.g);
		gLightImage.InterlockedAdd(16 * index + 8, ci.b, p.b);

		uint overflowMask = 0;
		for (uint i = 0; i < 3; i++)
			if (0xFFFFFFFF - p.r < ci.r)
				overflowMask |= (1 << i);
		gLightImage.InterlockedOr(16*index + 12, overflowMask);
	}

	void AddRadianceInterlocked(const uint2 index, const float3 radiance, float weight, const uint viewVertices, const uint lightVertices) {
		if (weight <= 0 || all(radiance <= 0))
			return;

		#ifdef gDebugPathLengths
		if (gDebugViewVertices != viewVertices || gDebugLightVertices != lightVertices)
			return;
		weight = 1;
		#endif
		AddRadianceInterlocked(index, radiance * weight);
	}

	[mutating]
	void AddRadiance(const float3 radiance, const float pdf, const float replayPdf, float weight, const uint viewVertices, const uint lightVertices, const float mergeRnd) {
		if (weight <= 0 || all(radiance <= 0))
			return;

		#ifdef gDebugPathLengths
		if (gDebugViewVertices != viewVertices || gDebugLightVertices != lightVertices)
			return;
		weight = 1;
		#endif

        const uint bounces = (viewVertices + lightVertices) - 2;

        if (bShift) {
            if (bounces == mBasePath.mBounces && lightVertices == mBasePath.mLightVertices) {
                mBasePath.mRadiance = radiance;
                mBasePath.mReplayPdfW = replayPdf;
			}
        } else {
            if (gReconnection && rcv.mPrefixBounces > 0)
                rcv.mRadiance = float3(
					SafeDivide(radiance.r, mThroughputAtRcv.r),
					SafeDivide(radiance.g, mThroughputAtRcv.g),
					SafeDivide(radiance.b, mThroughputAtRcv.b));
            PathReservoir c = PathReservoir(PathSample(radiance, replayPdf, bounces, lightVertices, gRandomSeed, uint2(mRngIdx % gOutputSize.x, mRngIdx / gOutputSize.x), rcv), 1/pdf, 1);
            c.PrepareMerge(weight);
            mReservoir.Merge(mergeRnd, c);
		}
	}

    [mutating]
    void TraceConnectionRay(const float3 rayOrigin, const float3 rayDirection, const float rayDistance, float3 contrib, float pdf, const float replayPdf, const float weight, const uint viewVertices, const uint lightVertices, const float mergeRnd) {
        if (!any(contrib > 0) || weight <= 0)
			return;

		float lightPdf, dist;
		if (TraceRay(MakeRay(rayOrigin, rayDirection, 0, rayDistance), false, lightPdf, dist).mIsSurface)
        	return;

        AddRadiance(contrib, pdf, replayPdf, weight, viewVertices, lightVertices, mergeRnd);
	}

	// ---------------------------------------------------------------------------------------------------------
	// ----------------------------------- Bidirectional vertex connections ------------------------------------
	// ---------------------------------------------------------------------------------------------------------

	void StoreLightVertex(const PathVertex vertex, const float3 localDirIn) {
		PackedLightVertex v;
		v.mVertex = PackVertex(vertex);
		v.mThroughput = mThroughput;
		v.mPackedLocalDirIn = PackNormal(localDirIn);
		v.mPathLength = mPathLength;
		v.dVC  = dVC;
		v.dVCM = dVCM;
		v.mPdfA = mPathPdf;
		uint idx;
		InterlockedAdd(gLightVertexCount[0], 1, idx);
		gLightVertices[idx] = v;
	}

	[mutating]
    void ConnectToLightSubpath(const PathVertex vertex, const float3 localDirIn, const float mergeRnd) {
        if (bShift &&
			(mBasePath.mLightVertices < 2 || mPathLength != (mBasePath.mBounces + 2 - mBasePath.mLightVertices) - 1) ||
			(gReconnection && mBasePath.mReconnectionVertex.mPrefixBounces > 0)) {
            mRng.Skip();
            return;
        }

        const uint count = gLightVertexCount[0];
        if (count == 0)
            return;
        const LightVertexSampleRecord r = SampleLightVertex(vertex.mPosition, mPathLength, gLightVertices[mRng.Next().x % count]);
		if (all(r.mThroughput <= 0) || all(r.mReflectance <= 0))
			return;

		const float3 localDirOut = vertex.ToLocal(r.mDirectionToLight);

		ReflectanceEvalRecord f = EvaluateReflectance<false>(vertex, localDirIn, localDirOut);
		f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
		const float contProb = GetContinuationProbability(vertex);
		f.mFwdPdfW *= contProb;
        f.mRevPdfW *= contProb;

        const bool hasRcv = !bShift && gReconnection && rcv.mPrefixBounces > 0;
        if (hasRcv && rcv.mPrefixBounces == mPathLength - 1) {
            mThroughputAtRcv = mThroughput * f.mReflectance;
            rcv.mDirOut = r.mDirectionToLight;
        }

        const float invDist2 = 1 / sqr(r.mDistanceToLight);
		const float3 contrib = (mThroughput * r.mThroughput) * (f.mReflectance * (r.mReflectance / r.mCosLight));

		// Convert pdfs to area pdf
		const float cameraBsdfDirPdfA = f.mFwdPdfW * r.mCosLight * invDist2;
		const float lightBsdfDirPdfA  = r.mFwdPdfW * abs(localDirOut.z) * invDist2;

		// Partial light sub-path MIS weight [tech. rep. (40)]
		const float wLight = Mis(cameraBsdfDirPdfA) * (gMisVmWeightFactor + r.dVCM + r.dVC * Mis(r.mRevPdfW));

		// Partial eye sub-path MIS weight [tech. rep. (41)]
		const float wCamera = Mis(lightBsdfDirPdfA) * (gMisVmWeightFactor + dVCM + dVC * Mis(f.mRevPdfW));

		const float misWeight = 1 / (wLight + 1.f + wCamera);

        const float AtoW = sqr(r.mDistanceToLight) / r.mCosLight;
        const float pdfW = (gLightSubpathCount / (float)count) * r.mPdfA * AtoW;

        TraceConnectionRay(
            OffsetRayOrigin(vertex, r.mDirectionToLight),
            r.mDirectionToLight,
            AdjustIntersectionDistance(r.mDistanceToLight),
            contrib,
            mPathPdf * pdfW,
            mReplayPdfW * (hasRcv ? 1 : AtoW),
            misWeight,
            mPathLength + 1,
            r.mSubpathLength + 1,
            mergeRnd);
	}

	// ---------------------------------------------------------------------------------------------------------
	// --------------------------------------------- Light tracing ---------------------------------------------
	// ---------------------------------------------------------------------------------------------------------

	[mutating]
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
		const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
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
		float3 contrib = (mThroughput * f.mReflectance) / (mPathPdf * gLightSubpathCount * surfaceToImageFactor * cosOut);

		#ifdef gLightTraceOnly
		const float misWeight = 1;
		#else
		// Partial light sub-path weight [tech. rep. (46)]. Note the division by
		// mLightPathCount, which is the number of samples this technique uses.
		// This division also appears a few lines below in the framebuffer accumulation.
		const float wLight = Mis(cameraPdfA / gLightSubpathCount) * (gMisVmWeightFactor + dVCM + dVC * Mis(f.mRevPdfW));

		// Partial eye sub-path weight is 0 [tech. rep. (47)]

		// Full path MIS weight [tech. rep. (37)]. No MIS for traditional light tracing.
		const float misWeight = 1.f / (wLight + 1.f);
		#endif

        if (!any(contrib > 0) || misWeight <= 0)
            return;

		float lightPdf, hitDist;
		if (TraceRay(MakeRay(OffsetRayOrigin(vertex, toCamera), toCamera, 0, dist), false, lightPdf, hitDist).mIsSurface)
			return;

		AddRadianceInterlocked(uint2(pixelCoord), contrib, misWeight, 1, (int)mPathLength + 1);
	}

	// ---------------------------------------------------------------------------------------------------------
	// ------------------------------ Direct illumination / Next Event Estimation ------------------------------
	// ---------------------------------------------------------------------------------------------------------

    [mutating]
    void ConnectToLight(const PathVertex vertex, const float3 localDirIn, const float mergeRnd) {
        if (bShift &&
			(mBasePath.mLightVertices != 1 || mPathLength != mBasePath.mBounces) ||
			(gReconnection && mBasePath.mReconnectionVertex.mPrefixBounces > 0)) {
            mRng.Skip();
            return;
        }

		float lightPdf;
		const PathVertex lightVertex = SampleEmission(mRng.NextFloat(), lightPdf);
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

		float G = 1;
		if (lightVertex.mIsSurface)
			G = cosLight / sqr(dist);
		const float directPdfW = lightPdf / G;

		const float3 localDirOut = vertex.ToLocal(toLight);

		ReflectanceEvalRecord f = EvaluateReflectance<false>(vertex, localDirIn, localDirOut);
		f.mReflectance *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
		const float contProb = GetContinuationProbability(vertex);
		f.mFwdPdfW *= contProb;
		f.mRevPdfW *= contProb;

        const bool hasRcv = !bShift && gReconnection && rcv.mPrefixBounces > 0;
        if (hasRcv && rcv.mPrefixBounces == mPathLength - 1) {
            mThroughputAtRcv = mThroughput * f.mReflectance;
            rcv.mDirOut = toLight;
        }

        const float3 contrib = lightVertex.mMaterial.Emission() * mThroughput * f.mReflectance;

        // compute Mis weight

		#ifdef gUseVC
		const float emissionPdfW = lightVertex.mIsSurface ? lightPdf * CosHemispherePdfW(cosLight) : (lightPdf * ConcentricDiscPdfA() / sqr(gSceneSphere.w));

		// Partial light sub-path MIS weight [tech. rep. (44)].
		// Note that wLight is a ratio of area pdfs. But since both are on the
		// light source, their distance^2 and cosine terms cancel out.
		// Therefore we can write wLight as a ratio of solid angle pdfs,
		// both expressed w.r.t. the same shading point.
		const float wLight = Mis(f.mFwdPdfW / directPdfW);

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
		const float wCamera = Mis(emissionPdfW * abs(localDirOut.z) / (directPdfW * cosLight)) * (gMisVmWeightFactor + dVCM + dVC * Mis(f.mRevPdfW));

		const float misWeight = 1 / (wLight + 1 + wCamera);
		#elif defined(gSampleDirectIlluminationOnly)
		const float misWeight = 1;
		#else
		// Mis with BSDF sampling
		const float misWeight = Mis(lightPdf, f.mFwdPdfW * G);
		#endif

		// evaluate transmittance along shadow ray

        TraceConnectionRay(
            OffsetRayOrigin(vertex, toLight),
            toLight,
            AdjustIntersectionDistance(dist),
            contrib,
            mPathPdf * directPdfW,
			mReplayPdfW * (hasRcv ? 1 : directPdfW),
			misWeight,
			mPathLength + 1,
			1,
			mergeRnd);
    }

    // ---------------------------------------------------------------------------------------------------------
    // --------------------------------------------- Reconnection ----------------------------------------------
    // ---------------------------------------------------------------------------------------------------------

    [mutating]
    void ReconnectToBasePath(const PathVertex vertex, const float3 localDirIn) {
        if (vertex.mMaterial.Roughness() < gReconnectionRoughness)
            return;

        const PathVertex reconnectionVertex = UnpackVertex(mBasePath.mReconnectionVertex.mVertex);

        float3 contrib = mBasePath.mReconnectionVertex.mRadiance * mThroughput;

		float3 toLight;
		float dist;
		float cosLight;
		if (reconnectionVertex.mIsSurface) {
			toLight = reconnectionVertex.mPosition - vertex.mPosition;
			dist = length(toLight);
			toLight /= dist;

            if (dist < gReconnectionDistance)
                return;

            const float3 rcvDirIn  = reconnectionVertex.ToLocal(-toLight);
            cosLight = rcvDirIn.z;

            if (mBasePath.mReconnectionVertex.mPrefixBounces < mBasePath.mBounces) {
				const float3 localDirOut = reconnectionVertex.ToLocal(mBasePath.mReconnectionVertex.mDirOut);
				contrib *= EvaluateReflectance<false>(reconnectionVertex, rcvDirIn, localDirOut).mReflectance;
                contrib *= reconnectionVertex.GetShadingNormalCorrection(rcvDirIn, localDirOut, false);
            } else if (cosLight <= 0)
				return;
		} else {
			toLight = reconnectionVertex.mPosition;
			dist = POS_INFINITY;
			cosLight = 1;
		}

        {
			const float3 localDirOut = vertex.ToLocal(toLight);
			contrib *= EvaluateReflectance<false>(vertex, localDirIn, localDirOut).mReflectance;
			contrib *= vertex.GetShadingNormalCorrection(localDirIn, localDirOut, false);
		}


        mBasePath.mReconnectionVertex.mCos = abs(cosLight);
        mBasePath.mReconnectionVertex.mDist = dist;

        // evaluate transmittance along shadow ray

        TraceConnectionRay(
            OffsetRayOrigin(vertex, toLight),
            toLight,
            AdjustIntersectionDistance(dist),
            contrib,
            1,
            mReplayPdfW,
            1,
            (mBasePath.mBounces + 2) - mBasePath.mLightVertices,
            mBasePath.mLightVertices,
			1);
	}


	// ---------------------------------------------------------------------------------------------------------
	// ------------------------------------- Direct radiance contribution --------------------------------------
	// ---------------------------------------------------------------------------------------------------------

    [mutating]
    void AddEmission(const float3 emission, const float directPdfA, const float emissionPdfW, const float G, const float mergeRng) {
		float w = 1;
		if (mPathLength > 1 && !isSpecular) {
			#ifdef gUseVC
			// Partial eye sub-path MIS weight [tech. rep. (43)].
			// If the last hit was specular, then dVCM == 0.
			const float wCamera = Mis(directPdfA) * dVCM + Mis(emissionPdfW) * dVC;
			w = 1 / (1 + wCamera);
			#elif defined(gSampleDirectIllumination)
			if (mDirPdfW > 0)
				w = Mis(mDirPdfW*G, directPdfA);
			#endif
		}

        AddRadiance(
            emission,
            mPathPdf,
			mReplayPdfW,
            w,
            mPathLength + 1,
            0,
            mergeRng);
	}


	// ---------------------------------------------------------------------------------------------------------
	// --------------------------------------------- Path tracing ----------------------------------------------
	// ---------------------------------------------------------------------------------------------------------

	[mutating]
	PathVertex TraceRay() {
		float lightPdf, dist;
		const PathVertex v = TraceRay(MakeRay(mRayOrigin, mRayDirection), true, mIntersectionPdfA, mIntersectionDistance);
		mPathLength++;

        if (!v.mIsSurface) {
            if (!bAdjoint && mPathLength >= gMinDepth) {
				const float3 emission = mThroughput * v.mMaterial.Emission();

				if (any(emission > 0)) {
					AddEmission(emission,
						ConcentricDiscPdfA() / sqr(gSceneSphere.w),
						lightPdf,
						1,
						mRng.NextFloat().x);
				}
			}
			mThroughput = 0;
		}

		return v;
	}

	[mutating]
	bool ProcessInteraction(const PathVertex vertex) {
		const float3 localDirIn = vertex.ToLocal(-mRayDirection);
        const float cosIn = abs(localDirIn.z);
        const float4 mergeRnds = mRng.NextFloat();
		const bool isDiffuse = IsDiffuse(vertex);

        const float G = abs(cosIn) / sqr(mIntersectionDistance);
        if (bAdjoint) {
			mThroughput *= G;
			mPathPdf    *= G;
		}

        if (bAdjoint) {
			#ifdef gUseVC
			// Update the MIS quantities before storing them at the vertex.
			// These updates follow the initialization in GenerateLightSample() or
			// SampleScattering(), and together implement equations [tech. rep. (31)-(33)]
			// or [tech. rep. (34)-(36)], respectively.

			// Infinite lights use MIS handled via solid angle integration,
			// so do not divide by the distance for such lights [tech. rep. Section 5.1]
			if (mPathLength > 1 || isFiniteLight == 1)
				dVCM *= Mis(sqr(mIntersectionDistance));
			dVCM /= Mis(cosIn);
			dVC  /= Mis(cosIn);
			#endif
		} else {
			#ifdef gUseVC
			// Update the MIS quantities, following the initialization in
			// GenerateLightSample() or SampleScattering(). Implement equations
			// [tech. rep. (31)-(33)] or [tech. rep. (34)-(36)], respectively.
			dVCM *= Mis(sqr(mIntersectionDistance));
			dVCM /= Mis(cosIn);
			dVC  /= Mis(cosIn);
			#endif

			const float3 emission = mPathLength >= gMinDepth && localDirIn.z > 0 ? mThroughput * vertex.mMaterial.Emission() : 0;
			bool isEmissive = any(emission > 0);

            if (!bShift && gReconnection) {
                // store reconnection vertex if one hasn't been found yet
                const bool connectable = mIntersectionDistance >= gReconnectionDistance && (isEmissive || vertex.mMaterial.Roughness() >= gReconnectionRoughness);
                if (rcv.mPrefixBounces == 0 && mPrevReplayPdf > 0 && (isDiffuse || isEmissive) && connectable) {
					rcv.mVertex = PackVertex(vertex);
					rcv.mCos = cosIn;
					rcv.mDist = mIntersectionDistance;
                    rcv.mPrefixBounces = mPathLength - 1;
                    mReplayPdfW = mPrevReplayPdf;
					mThroughputAtRcv = mThroughput;
				}
			}

            // surface lights are one-sided
            if (isEmissive)
				AddEmission(emission,
							mIntersectionPdfA,
							mIntersectionPdfA * CosHemispherePdfW(localDirIn.z),
							cosIn / sqr(mIntersectionDistance),
							mergeRnds.x);
		}

        if (mPathLength >= gMaxDepth)
            return false;
        if (bShift && mPathLength + 1 > (mBasePath.mBounces + 2) - mBasePath.mLightVertices)
            return false; // too many camera vertices in the offset path

		if (!any(vertex.mMaterial.BaseColor() > 0))
			return false;

        // connection strategies + light vertex storage
        if (isDiffuse) {
            if (bAdjoint) {
				if (mPathLength + 1 >= gMinDepth)
					ConnectToCamera(vertex, localDirIn);
				#ifdef gUseVC
				if (mPathLength + 2 <= gMaxDepth)
					StoreLightVertex(vertex, localDirIn);
				#endif
            } else {
                if (bShift && gReconnection && mBasePath.mReconnectionVertex.mPrefixBounces > 0) {
                    if (mPathLength > mBasePath.mReconnectionVertex.mPrefixBounces)
                        return false;
                    else if (mPathLength == mBasePath.mReconnectionVertex.mPrefixBounces) {
                        ReconnectToBasePath(vertex, localDirIn);
						return false;
                    }
                }
				#if defined(gSampleDirectIllumination) || defined(gUseVC)
                if (mPathLength + 1 >= gMinDepth)
					ConnectToLight(vertex, localDirIn, mergeRnds.y);
				#endif
				#ifdef gUseVC
				if (mPathLength + 1 <= gMaxDepth)
					ConnectToLightSubpath(vertex, localDirIn, mergeRnds.z);
				#endif
			}
		}

        mPrevReplayPdf = isDiffuse && vertex.mMaterial.Roughness() >= gReconnectionRoughness ? mReplayPdfW : 0;

        if (bShift && mPathLength + 1 >= mBasePath.mBounces+2 - (max(1, mBasePath.mLightVertices)-1))
            return false; // too many camera vertices on the base path

		#ifdef gSampleDirectIlluminationOnly
		return false;
		#endif

		const float continueProb = min(1, GetContinuationProbability(vertex));
		if (mergeRnds.w > continueProb)
			return false;

        DirectionReflectanceSample s = SampleDirection<bAdjoint>(vertex, mRng.NextFloat(), localDirIn);
		s.mFwdPdfW *= continueProb;
		s.mRevPdfW *= continueProb;

		mThroughput *= s.mReflectance;
        mPathPdf    *= s.mFwdPdfW;
        if (!gReconnection || bShift || rcv.mPrefixBounces == 0)
        	mReplayPdfW *= s.mFwdPdfW;

		const float cosOut = abs(s.mDirection.z);

		// Sub-path MIS quantities for the next vertex. Only partial - the
		// evaluation is completed when the actual hit point is known,
		// i.e. after tracing the ray, in the sub-path loop.
		if (s.isSingular()) {
			#ifdef gUseVC
			// Specular scattering case [tech. rep. (53)-(55)] (partially, as noted above)
			dVCM = 0.f;
			dVC *= Mis(cosOut);
			#else
			mDirPdfW = 0;
			#endif
		} else {
			#ifdef gUseVC
			#ifndef gTraceFromLight
			isSpecular = false;
			#endif
			// Implements [tech. rep. (34)-(36)] (partially, as noted above)
			dVC = Mis(cosOut / s.mFwdPdfW) * (dVC * Mis(s.mRevPdfW) + dVCM + gMisVmWeightFactor);
			dVCM = Mis(1.f / s.mFwdPdfW);
			#else
			mDirPdfW = s.mFwdPdfW;
			#endif
		}

        mThroughput *= vertex.GetShadingNormalCorrection(localDirIn, s.mDirection, bAdjoint);
		mRayDirection = vertex.ToWorld(s.mDirection);
		mRayOrigin = OffsetRayOrigin(vertex, mRayDirection);

        if (!bShift && gReconnection && rcv.mPrefixBounces > 0 && rcv.mPrefixBounces == mPathLength - 1) {
			rcv.mDirOut = mRayDirection;
			mThroughputAtRcv = mThroughput;
		}
		return !any(isnan(mThroughput));
	}

	[mutating]
	bool InitializeCameraPath(const PathVertex vertex, const uint randomSeed, const uint2 index) {
		mRng = RandomSampler(randomSeed, index);
		mRngIdx = gOutputSize.x*index.y + index.x;

		#ifdef gPixelJitter
		const float2 offset = mRng.nextFloat().xy;
		#else
		const float2 offset = 0.5;
		#endif

		const float3 dir = normalize(vertex.mPosition - gCameraPosition);
		float2 uv;
		const float3 localDir = TransformVector(gWorldToCamera, dir);

		mRayOrigin = gCameraPosition;
		mPathLength = 0;
		mThroughput = 1;
        mPathPdf = 1;
        mReplayPdfW = 1;
        mRayDirection = dir;
        mPrevReplayPdf = 0;

		#ifdef gUseVC
			// Compute pdf conversion factor from area on image plane to solid angle on ray
			const float cosAtCamera = abs(localDir.z);
			const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
			const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;

			// We put the virtual image plane at such a distance from the camera origin
			// that the pixel area is one and thus the image plane sampling pdf is 1.
			// The solid angle ray pdf is then equal to the conversion factor from
			// image plane area density to ray solid angle density
			const float cameraPdfW = imageToSolidAngleFactor;

			isSpecular = true;

			// Eye sub-path MIS quantities. Implements [tech. rep. (31)-(33)] partially.
			// The evaluation is completed after tracing the camera ray in the eye sub-path loop.
			dVCM = Mis(gLightSubpathCount / cameraPdfW);
			dVC = 0;
		#endif

		#ifdef gLightTraceOnly
            mThroughput = 0;
		#endif

		mReservoir = PathReservoir();
		rcv = ReconnectionVertex();

		return true;
	}
	[mutating]
	void InitializeLightPath(const uint randomSeed, const uint2 index) {
		mRng = RandomSampler(randomSeed, index + uint2(0, gOutputSize.y*10));
		mRngIdx = gOutputSize.x*index.y + index.x;
		float directPdfA;
		const PathVertex lightVertex = SampleEmission(mRng.NextFloat(), directPdfA);
		isFiniteLight = lightVertex.mIsSurface;

		mThroughput = lightVertex.mMaterial.Emission();

		float emissionPdfW = directPdfA;
		float cosLight;

		const float2 dirRng = mRng.NextFloat().xy;
		if (isFiniteLight) {
			// area light
			const float3 dir = SampleCosHemisphere(dirRng.xy);
			mRayDirection = lightVertex.ToWorld(dir);
			mRayOrigin = OffsetRayOrigin(lightVertex, mRayDirection);
            mThroughput *= dir.z;
            emissionPdfW *= CosHemispherePdfW(dir.z);
			cosLight = dir.z;
		} else {
			// environment light
			const float2 xy = SampleConcentricDisc(dirRng.xy);
			const float3x3 onb = MakeOrthonormal(lightVertex.mPosition);
			mRayOrigin = gSceneSphere.xyz + gSceneSphere.w * (onb[0] * xy.x + onb[1] * xy.y + lightVertex.mPosition);
			mRayDirection = -lightVertex.mPosition;
			emissionPdfW *= ConcentricDiscPdfA() / sqr(gSceneSphere.w);
			cosLight = 1;
		}

		mPathPdf = emissionPdfW;

		#ifdef gUseVC
		dVCM = Mis(directPdfA / emissionPdfW);
		dVC  = lightVertex.mIsSurface ? Mis(cosLight / emissionPdfW) : 0;
		#endif

        mPathLength = 0;
        mPrevReplayPdf = 0;
	}

	[mutating]
	void Run() {
		while (mPathLength + 1 <= gMaxDepth && any(mThroughput > 0)) {
			const PathVertex v = TraceRay();
            if (!ProcessInteraction(v)) {
				mThroughput = 0;
				break;
			}
		}
	}
};

typedef PathGenerator<true,  false> LightPathGenerator;
typedef PathGenerator<false, false> CanonicalPathGenerator;
typedef PathGenerator<false, true > ShiftPathGenerator;

void SampleLightPath(const uint seed, const uint2 rngIdx) {
    LightPathGenerator g;
	g.InitializeLightPath(seed, rngIdx);
	g.Run();
}

PathReservoir SamplePath(const PathVertex vertex, const float3 cameraPos, const uint seed, const uint2 rngIdx) {
    CanonicalPathGenerator g;
	if (!g.InitializeCameraPath(vertex, seed, rngIdx))
		return PathReservoir();

	g.Run();

	g.mReservoir.FinalizeMerge();
	g.mReservoir.mM = 1;

	if (gReconnection && g.mReservoir.mW > 0 && g.mReservoir.mSample.mReconnectionVertex.mPrefixBounces > 0)
		IncrementCounter(DebugCounterType::eReconnectionVertices);

	return g.mReservoir;
}

// when bShiftFromPrevFrame is true, uses the last frame's light vertices
PathSample ShiftPath(const PathSample basePath, const PackedVertex dstVertex, const float3 cameraPos, out float jacobian) {
	jacobian = 0;

	PathVertex vertex = UnpackVertex(dstVertex);
	if (!vertex.mIsSurface)
        return PathSample();

    ShiftPathGenerator g;
	if (!g.InitializeCameraPath(vertex, basePath.mRngSeed, basePath.mRngIdx))
        return PathSample();

    g.mReservoir.mSample = basePath;
    g.mReservoir.mSample.mRadiance = 0;
    g.mReservoir.mSample.mReplayPdfW = 0;

	g.Run();

	const PathSample s = g.mReservoir.mSample;

	IncrementCounter(DebugCounterType::eShiftAttempts);
	if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0)
		IncrementCounter(DebugCounterType::eReconnectionAttempts);

    if (any(s.mRadiance > 0) && s.mReplayPdfW > 0) {
		jacobian = basePath.mReplayPdfW / s.mReplayPdfW;
		IncrementCounter(DebugCounterType::eShiftSuccesses);
		if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0) {
			IncrementCounter(DebugCounterType::eReconnectionSuccesses);
			if (basePath.mReconnectionVertex.mVertex.mInstanceIndex != INVALID_INSTANCE)
				jacobian *= (s.mReconnectionVertex.mCos / basePath.mReconnectionVertex.mCos) * sqr(basePath.mReconnectionVertex.mDist / s.mReconnectionVertex.mDist);
		}
	}

	return s;
}

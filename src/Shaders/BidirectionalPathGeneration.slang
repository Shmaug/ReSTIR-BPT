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
#else
static const bool gBidirectional = false;
#endif

// 48 bytes
struct PackedLightVertex {
    PackedVertex mVertex;
    float3 mThroughput;
    uint mPackedDirIn;
    uint mLightBounces;
    float dVC;
    float pad0;
    float pad1;

    property float3 mDirIn {
		get { return UnpackNormal(mPackedDirIn); }
		set { mPackedDirIn = PackNormal(newValue); }
	};
};

#define gLightImageQuantization 16384

RWByteAddressBuffer gLightImage;
#ifdef LIGHT_VERTEX_HASH_GRID
HashGrid<PackedLightVertex> gLightVertices;
#endif

uniform float gReconnectionDistance;
uniform float gReconnectionRoughness;
uniform uint gMaxBounces;
uniform uint gLightSubpathCount;
uniform float3 gCameraPosition;
uniform float3 gCameraForward;
uniform float4 gSceneSphere;
uniform float gCameraImagePlaneDist;
uniform float4x4 gMVP;

void AddColor(uint2 id, float3 c) {
	const uint index = id.y * gOutputSize.x + id.x;
	uint3 ci = uint3(c * gLightImageQuantization);
	uint3 p;
	gLightImage.InterlockedAdd(16*index+0, ci.r, p.r);
	gLightImage.InterlockedAdd(16*index+4, ci.g, p.g);
	gLightImage.InterlockedAdd(16*index+8, ci.b, p.b);

	uint overflowMask = 0;
	for (uint i = 0; i < 3; i++)
		if (0xFFFFFFFF - p.r < ci.r)
			overflowMask |= (1 << i);
	gLightImage.InterlockedOr(16*index + 12, overflowMask);
}
float3 LoadColor(uint2 id) {
    uint4 c = gLightImage.Load4(16 * (id.y * gOutputSize.x + id.x));
	if (c.w & (1 << 0)) c.r = 0xFFFFFFFF;
	if (c.w & (1 << 1)) c.g = 0xFFFFFFFF;
	if (c.w & (1 << 2)) c.b = 0xFFFFFFFF;
	return c.rgb / (float)gLightImageQuantization;
}


float Mis(float a) {
	return a * a;
}
float Mis(float a, float b) {
    a = Mis(a);
    return a / (a + Mis(b));
}

void ConnectToLightSubpath(const PathVertex vertex, const float3 dirIn, const uint bounces, const uint rnd, out float3 le, out float3 toLight, out float w, out float lightPdf) {
	le = 0;
	w = 0;

	const uint lightVertexCount = gLightVertices.GetCurrentElementCount();
    if (lightVertexCount == 0)
        return;
    const PackedLightVertex v = gLightVertices.mAppendData[rnd % lightVertexCount];

	if (bounces + v.mLightBounces + 1 > gMaxBounces)
		return;

    const PathVertex lightVertex = UnpackVertex(v.mVertex);

    toLight = lightVertex.mPosition - vertex.mPosition;
    float dist = length(toLight);
    toLight /= dist;

    const float cosLight = abs(dot(lightVertex.mShadingNormal, toLight));

	le = v.mThroughput;

	le *= EvalBrdf(lightVertex, v.mDirIn, -toLight, true);
    float lightContProb = GetContinuationProbability(lightVertex);
	float lightFwdPdfW = EvalBrdfPdfW(lightVertex, v.mDirIn, -toLight) * lightContProb;
    float lightRevPdfW = EvalBrdfPdfW(lightVertex, -toLight, v.mDirIn) * lightContProb;

	le *= EvalBrdf(vertex, -dirIn, toLight);
	const float cameraContProb = GetContinuationProbability(vertex);
	const float cameraFwdPdfW = EvalBrdfPdfW(vertex, -dirIn, toLight) * cameraContProb;
	const float cameraRevPdfW = EvalBrdfPdfW(vertex, toLight, -dirIn) * cameraContProb;

	const float invDist2 = 1 / sqr(dist);
	le *= invDist2 * (lightVertexCount / (float)gLightSubpathCount);

	// Convert pdfs to area measure
	const float cameraBsdfDirPdfA = cameraFwdPdfW * cosLight * invDist2;
	const float lightBsdfDirPdfA  = lightFwdPdfW * abs(dot(toLight, vertex.mShadingNormal)) * invDist2;

	const float wLight = Mis(cameraBsdfDirPdfA) * (v.dVCM + v.dVC * Mis(lightRevPdfW));
	const float wCamera = Mis(lightBsdfDirPdfA) * (dVCM + dVC * Mis(cameraRevPdfW));
	w = 1 / (wLight + 1.f + wCamera);

	if (Occluded(OffsetRayOrigin(vertex, toLight), toLight, AdjustIntersectionDistance(dist))) {
		le = 0;
	}
}

PathReservoir SamplePath(PathVertex vertex, const uint seed, const uint2 rngIdx) {
	PathReservoir res = PathReservoir();
    ReconnectionVertex rcv = ReconnectionVertex();
    float3 throughputAtRcv = 0;

	float3 dirIn = vertex.mPosition - gCameraPosition;
	float dist = length(dirIn);
	dirIn /= dist;

	bool isSpecularPath = true;
	float dVCM;
	float dVC;
	if (gBidirectional) {
		const float cosAtCamera = abs(dot(dirIn, gCameraForward));
		const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
		const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;
		const float cameraPdfW = imageToSolidAngleFactor;
		dVCM = Mis(gLightSubpathCount / cameraPdfW);
		dVC = 0;
		dVCM *= Mis(sqr(dist)/abs(dot(vertex.mShadingNormal, dirIn)));
	}

    RandomSampler rng = RandomSampler(seed, rngIdx);

	float3 throughput = 1;
    float pdfW = 1;
    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const float4 rnd = rng.NextFloat();
        const float mergeRnd       = rnd.x;
        const float lightMergeRnd  = rnd.y;
        const float lightMergeRnd2 = rnd.z;
        const float contRnd        = rnd.w;

		const float contProb = GetContinuationProbability(vertex);

        const bool diffuse = IsDiffuse(vertex);
        if (diffuse) {
			// sample direct light
            if (gSampleLights || gBidirectional) {
                const DirectLightSampleRecord l = SampleDirectLight(vertex, rng.NextFloat());
                float3 le = throughput * l.mRadiance;
                if (any(le > 0)) {
                    le *= EvalBrdf(vertex, -dirIn, l.mToLight);
                    if (any(le > 0)) {
						if (!Occluded(vertex, l.mToLight, l.mDistance)) {
							if (gReconnection && rcv.mPrefixBounces > 0) {
								if (rcv.mPrefixBounces == bounces - 1) {
									rcv.mDirOut = l.mToLight;
									rcv.mRadiance = l.mRadiance;
								} else
									rcv.mRadiance = float3(SafeDivide(le.r, throughputAtRcv.r), SafeDivide(le.g, throughputAtRcv.g), SafeDivide(le.b, throughputAtRcv.b));
							}

							const float fwdPdf = contProb * EvalBrdfPdfW(vertex, -dirIn, l.mToLight);

							float w = 1;
							if (gBidirectional) {
								const float revPdf = contProb * EvalBrdfPdfW(vertex, l.mToLight, -dirIn);
								const float emissionPdfW = (l.mPdfW*l.mG) * (isfinite(l.mDistance) ? CosHemispherePdfW(l.mCos) : ConcentricDiscPdfA() / sqr(gSceneSphere.w));

								const float wLight = Mis(fwdPdf / l.mPdfW);
								const float wCamera = Mis(emissionPdfW * abs(dot(vertex.mShadingNormal, l.mToLight)) / (l.mPdfW * l.mCos)) * (dVCM + dVC * Mis(revPdf));

								w = 1 / (wLight + 1 + wCamera);
							} else
								w = Mis(l.mPdfW, fwdPdf);

							PathReservoir c = PathReservoir(PathSample(le, pdfW * l.mPdfW, bounces, true, seed, rngIdx, rcv));
							c.PrepareMerge(w);
							res.Merge(lightMergeRnd, c);
						}
                    }
				}
            }
			if (gBidirectional) {
				float3 radiance, toLight;
				float w, pdf;
				ConnectToLightSubpath(vertex, dirIn, bounces, rng.Next().x, radiance, toLight, w, pdf);
				float3 le = radiance * throughput;
                if (any(le > 0)) {
					if (gReconnection && rcv.mPrefixBounces > 0) {
						if (rcv.mPrefixBounces == bounces - 1) {
							rcv.mDirOut = toLight;
							rcv.mRadiance = radiance;
						} else
							rcv.mRadiance = float3(SafeDivide(le.r, throughputAtRcv.r), SafeDivide(le.g, throughputAtRcv.g), SafeDivide(le.b, throughputAtRcv.b));
					}
					PathReservoir c = PathReservoir(PathSample(le, pdfW * pdf, bounces, false, seed, rngIdx, rcv));
					c.PrepareMerge(w);
					res.Merge(lightMergeRnd2, c);
				}
			}
		}

		if (contRnd >= contProb)
			break;

        const float prevPdf = pdfW;

		// sample brdf
		DirectionSampleRecord r;
        SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection);
        float brdfPdfW = contProb * EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
        pdfW *= brdfPdfW;
        if (all(throughput <= 0)) break;

		// set rcv dirOut at bounce after rcv
        if (gReconnection && rcv.mPrefixBounces > 0 && rcv.mPrefixBounces == bounces-1) {
            rcv.mDirOut = r.mDirection;
            throughputAtRcv = throughput;
		}

		// update bidirectional mis
		if (gBidirectional) {
			const float cosOut = Mis(abs(dot(vertex.mShadingNormal, r.mDirection)));
			if (r.mRoughness == 0) {
				brdfPdfW = 0;
				dVCM = 0.f;
				dVC *= Mis(cosOut);
			} else {
				const float revPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection) * contProb;
				isSpecularPath = false;
				dVC = Mis(cosOut / brdfPdfW) * (dVC * Mis(revPdfW) + dVCM);
				dVCM = Mis(1.f / brdfPdfW);
			}
		}

        float directPdf;
        dirIn = r.mDirection;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, directPdf, dist);

		// update bidirectional mis
		if (vertex.mIsSurface) {
			const float cosMis = Mis(abs(dot(vertex.mShadingNormal, dirIn)));
			dVCM *= Mis(sqr(dist));
			dVCM *= cosMis;
			dVC  *= cosMis;
		}

		float emissionPdf = directPdf;

        float3 le = vertex.mMaterial.Emission();
		// store rcv, convert light sample pdf to solid angle
        if (vertex.mIsSurface) {
            const float cosIn = -dot(vertex.mShadingNormal, dirIn);
			if (gSampleLights && !gBidirectional)
            	directPdf *= sqr(dist) / abs(cosIn);
            if (cosIn <= 0) {
				le = 0;
				directPdf = 0;
				emissionPdf = 0;
			}

            // store reconnection vertex if one hasn't been found yet
            if (gReconnection && rcv.mPrefixBounces == 0 && dist >= gReconnectionDistance && vertex.mMaterial.Roughness() >= gReconnectionRoughness && diffuse && (IsDiffuse(vertex) || any(le > 0))) {
                rcv.mVertex = PackVertex(vertex);
                rcv.mCos = cosIn;
                rcv.mDist = dist;
                rcv.mPrefixPdfW = prevPdf;
                rcv.mPrefixBounces = bounces;
			}
			emissionPdf *= CosHemispherePdfW(cosIn);
        } else
			emissionPdf *= ConcentricDiscPdfA() / sqr(gSceneSphere.w);

		// add emission from new vertex
        if (any(le > 0)) {
            if (gReconnection && rcv.mPrefixBounces > 0) {
                if (rcv.mPrefixBounces == bounces)
                    rcv.mRadiance = le;
				else
	                rcv.mRadiance = le * float3(SafeDivide(throughput.r, throughputAtRcv.r), SafeDivide(throughput.g, throughputAtRcv.g), SafeDivide(throughput.b, throughputAtRcv.b));
			}
			float w = 1;
			if (gSampleLights && !gBidirectional) {
				w = brdfPdfW > 0 ? Mis(brdfPdfW, directPdf) : 1;
			} else if (gBidirectional && !isSpecularPath) {
				const float wCamera = Mis(directPdf) * dVCM + Mis(emissionPdf) * dVC;
				w = 1 / (1 + wCamera);
			}
            PathReservoir c = PathReservoir(PathSample(throughput * le, pdfW, bounces, false, seed, rngIdx, rcv));
            c.PrepareMerge(w);
			res.Merge(mergeRnd, c);
		}
	}
    res.FinalizeMerge();
    res.mM = 1;
    if (gReconnection && res.mW > 0 && res.mSample.mReconnectionVertex.mPrefixBounces > 0)
        IncrementCounter(DebugCounterType::eReconnectionVertices);
    return res;
}

PathSample ShiftPath(const PathSample basePath, const PackedVertex dstVertex, const float3 cameraPos, out float jacobian) {
    jacobian = 0;

    PathVertex vertex = UnpackVertex(dstVertex);
    if (!vertex.mIsSurface)
        return PathSample();

    float3 dirIn = normalize(vertex.mPosition - cameraPos);

    IncrementCounter(DebugCounterType::eShiftAttempts);
    if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0)
    	IncrementCounter(DebugCounterType::eReconnectionAttempts);

    PathSample res = basePath;
    res.mRadiance = 0;
    res.mPdfW = 1;

    RandomSampler rng = RandomSampler(basePath.mRngSeed, basePath.mRngIdx);

	float3 throughput = 1;
    for (res.mBounces = 1; res.mBounces <= basePath.mBounces; res.mBounces++) {
        if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0) {
            // reconnect to base path
            if (res.mBounces == basePath.mReconnectionVertex.mPrefixBounces) {
                if (IsDiffuse(vertex)) {
                    const PathVertex rcv = UnpackVertex(basePath.mReconnectionVertex.mVertex);
                    float3 toRcv;
                    float dist;
					if (rcv.mIsSurface) {
						toRcv = rcv.mPosition - vertex.mPosition;
						dist = length(toRcv);
						toRcv /= dist;
					} else {
						toRcv = rcv.mPosition;
						dist = POS_INFINITY;
					}
                    if (dist >= gReconnectionDistance) {
						float3 le = throughput * EvalBrdf(vertex, -dirIn, toRcv) * res.mReconnectionVertex.mRadiance;
						if (any(le > 0) && basePath.mReconnectionVertex.mPrefixBounces < basePath.mBounces)
							le *= EvalBrdf(rcv, -toRcv, basePath.mReconnectionVertex.mDirOut);
                        else if (basePath.mReconnectionVertex.mPrefixBounces == basePath.mBounces && dot(toRcv, rcv.mShadingNormal) > 0)
							le = 0;
						if (any(le > 0)) {
							if (!Occluded(vertex, toRcv, dist)) {
								res.mReconnectionVertex.mCos = abs(dot(rcv.mShadingNormal, toRcv));
								res.mReconnectionVertex.mDist = dist;
								res.mReconnectionVertex.mPrefixPdfW = res.mPdfW;
								res.mRadiance = le;
								res.mBounces = basePath.mBounces;
							}
						}
                    }
				}
                break;
			}
        }

        rng.Skip(); // skip merge rnds

        if (IsDiffuse(vertex)) {
            if (gSampleLights) {
				if (res.mBounces == basePath.mBounces && basePath.mIsNEE) {
					// sample direct light
                    const DirectLightSampleRecord l = SampleDirectLight(vertex, rng.NextFloat());
                    float3 le = l.mRadiance;
					if (any(le > 0)) {
                        le *= EvalBrdf(vertex, -dirIn, l.mToLight);
                        if (any(le > 0)) {
							if (!Occluded(vertex, l.mToLight, l.mDistance)) {
								res.mPdfW *= l.mPdfW;
								res.mRadiance = throughput * le;
								break;
							}
                        }
                    }
                    break;
				} else
					rng.Skip(); // skip light sample rnd
            }
		}

		const float p = GetContinuationProbability(vertex);
		if (p == 0) break;
		res.mPdfW *= p;

		// sample brdf
		DirectionSampleRecord r;
		SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection);
		const float brdfPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
        res.mPdfW *= brdfPdfW;
		if (all(throughput <= 0)) break;

        dirIn = r.mDirection;

		float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, lightPdf, dist);

        if (res.mBounces == basePath.mBounces) {
			float3 le = vertex.mMaterial.Emission();
			if (vertex.mIsSurface && dot(vertex.mShadingNormal, dirIn) >= 0)
				le = 0;

			res.mRadiance = throughput * le;
			break;
		}
	}

    if (any(res.mRadiance > 0)) {
		IncrementCounter(DebugCounterType::eShiftSuccesses);
        if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0) {
			IncrementCounter(DebugCounterType::eReconnectionSuccesses);
            jacobian = basePath.mReconnectionVertex.mPrefixPdfW / res.mReconnectionVertex.mPrefixPdfW;
			if (basePath.mReconnectionVertex.mVertex.mInstanceIndex != INVALID_INSTANCE)
            	jacobian *= (res.mReconnectionVertex.mCos / basePath.mReconnectionVertex.mCos) * sqr(basePath.mReconnectionVertex.mDist / res.mReconnectionVertex.mDist);
        } else
            jacobian = basePath.mPdfW / res.mPdfW;
    }

    return res;
}

void SampleLightSubpaths(inout RandomSampler rng) {
	float directPdf;
	PathVertex vertex = SampleEmission(rng.NextFloat(), directPdf);

    float3 throughput = vertex.mMaterial.Emission();
    float cosLight = 1;
    float3 origin, dirOut;
    const bool isAreaLight = vertex.mIsSurface;

	float emissionPdf = directPdf;

    const float2 rnd = rng.NextFloat().xy;
    if (isAreaLight) {
        dirOut = SampleCosHemisphere(rnd);
        cosLight = abs(dirOut.z);

        emissionPdf *= CosHemispherePdfW(cosLight);
		throughput *= cosLight;

        dirOut = vertex.ToWorld(dirOut);
        origin = OffsetRayOrigin(vertex, dirOut);
    } else {
        const float2 d = SampleConcentricDisc(rnd);
        dirOut = -vertex.mPosition;
        const float3x3 onb = MakeOrthonormal(dirOut);
       	origin = gSceneSphere.xyz + (onb[0] * d.x + onb[1] * d.y - dirOut) * gSceneSphere.w;

        emissionPdf *= ConcentricDiscPdfA() / sqr(gSceneSphere.w);
	}

	throughput /= emissionPdf;

    float dVC = isAreaLight ? Mis(cosLight / emissionPdf) : 0;
    float dVCM = Mis(directPdf / emissionPdf);

	float isectLightPdf, isectDist;
    vertex = TraceRay(MakeRay(origin, dirOut), true, isectLightPdf, isectDist);

	if (!vertex.mIsSurface)
		return;

	{
		if (isAreaLight)
			dVCM *= Mis(sqr(isectDist));
		const float cosMis = 1/Mis(abs(dot(vertex.mShadingNormal, dirOut)));
		dVCM *= cosMis;
		dVC  *= cosMis;
	}

	float3 dirIn = dirOut;

    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const bool diffuse = IsDiffuse(vertex);
        const float contProb = diffuse ? GetContinuationProbability(vertex) : 1;
        if (diffuse) {
			// store light vertex
			{
				PackedLightVertex v;
				v.mVertex = PackVertex(vertex);
				v.mThroughput = throughput;
				v.mPackedDirIn = PackNormal(-dirIn);
				v.mLightBounces = bounces;
				v.dVC = dVC;
				#ifdef LIGHT_VERTEX_HASH_GRID
				gLightVertices.Append(vertex.mPosition, v);
				#endif
			}

			// connect to camera
			const float4 clip = mul(gMVP, float4(vertex.mPosition, 1));
			const float2 uv = (clip.xy / clip.w) * .5 + .5;
			if (all(uv >= 0) && all(uv < 1) && clip.z > 0) {
				const uint2 pixel = uint2(uv * gOutputSize);

				float3 toCamera = gCameraPosition - vertex.mPosition;
				const float dist = length(toCamera);
				toCamera /= dist;

				const float cosOut = abs(dot(vertex.mShadingNormal, toCamera));

				float3 le = throughput * EvalBrdf(vertex, -dirIn, toCamera, true) / cosOut;
				if (any(le > 0)) {
					const float cameraCos = abs(dot(gCameraForward, -toCamera));
					const float imagePointToCameraDist = gCameraImagePlaneDist / cameraCos;
					const float cameraPdfW = sqr(imagePointToCameraDist) / cameraCos;
					const float cameraPdfA = cameraPdfW * cosOut / sqr(dist);
					const float surfaceToImageFactor = 1 / cameraPdfA;

					le /= (surfaceToImageFactor * gLightSubpathCount);

					const float revPdfW = GetContinuationProbability(vertex) * EvalBrdfPdfW(vertex, toCamera, -dirIn);

					const float wLight = Mis(cameraPdfA / gLightSubpathCount) * (dVCM + dVC * Mis(revPdfW));
					const float misWeight = 1.f / (wLight + 1.f);
					le *= misWeight;

					if (!Occluded(vertex, toCamera, dist))
						AddColor(pixel, le);
				}
			}
		}
        if (contProb == 0 || rng.NextFloat().x > contProb)
            break;

        // sample brdf
        DirectionSampleRecord r;
        SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection, true);
        if (all(throughput <= 0)) break;

		const float fwdDirPdfW = contProb * EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
		throughput /= fwdDirPdfW;

		// Sub-path MIS quantities for the next vertex
		const float cosOut = abs(dot(vertex.mShadingNormal, r.mDirection));
		if (r.mRoughness == 0) {
			dVCM = 0.f;
			dVC *= Mis(cosOut);
		} else {
			const float revDirPdfW = contProb * EvalBrdfPdfW(vertex, r.mDirection, -dirIn);
			dVC = Mis(cosOut / fwdDirPdfW) * (dVC * Mis(revDirPdfW) + dVCM);
			dVCM = Mis(1.f / fwdDirPdfW);
		}

        dirIn = r.mDirection;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, isectLightPdf, isectDist);

		if (vertex.mIsSurface) {
			const float cosMis = 1/Mis(abs(dot(vertex.mShadingNormal, dirIn)));
			dVCM *= Mis(sqr(isectDist)) * cosMis;
			dVC *= cosMis;
		}
    }
}

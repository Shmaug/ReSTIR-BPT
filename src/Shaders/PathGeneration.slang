#pragma once

#include "PathReservoir.slang"
#include "Intersection.slang"
#include "Light.slang"
#include "BRDF.slang"

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

uniform float3 gCameraPosition;
uniform uint gRandomSeed;
uniform float3 gPrevCameraPosition;
uniform uint2 gOutputSize;

uniform float gReconnectionDistance;
uniform float gReconnectionRoughness;
uniform uint gMaxBounces;

float Mis(float a, float b) {
    a = a * a;
    return SafeDivide(a, a + b * b);
}

PathReservoir SamplePath(PathVertex vertex, const float3 cameraPos, const uint seed, const uint2 rngIdx) {
	PathReservoir res = PathReservoir();
    ReconnectionVertex rcv = ReconnectionVertex();
    float3 throughputAtRcv = 0;

	float3 dirIn = normalize(vertex.mPosition - cameraPos);

    RandomSampler rng = RandomSampler(seed, rngIdx);
	float replayPdfW = 0;

	float3 throughput = 1;
    float pdfW = 1;
    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const float3 rnd = rng.NextFloat().xyz;
        const float mergeRnd      = rnd.x;
        const float lightMergeRnd = rnd.y;
        const float contRnd       = rnd.z;

        const float prevPdf = pdfW;

        const bool diffuse = IsDiffuse(vertex);
        if (diffuse) {
			// sample direct light
            if (gSampleLights) {
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
							PathReservoir c = PathReservoir(PathSample(le, replayPdfW > 0 ? replayPdfW : (pdfW/l.mG), bounces, 1, seed, rngIdx, rcv), 1 / (pdfW * l.mPdfW), 1);
							c.PrepareMerge(Mis(l.mPdfW, EvalBrdfPdfW(vertex, -dirIn, l.mToLight)));
							res.Merge(lightMergeRnd, c);
						}
                    }
				}
            }

			if (bounces < gMaxBounces) {
				const float p = GetContinuationProbability(vertex);
				if (p == 0 || contRnd > p)
					break;
				pdfW *= p;
			}
		}

		// sample brdf
		DirectionSampleRecord r;
        SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection);
        const float brdfPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
        pdfW *= brdfPdfW;
        if (all(throughput <= 0)) break;

        if (gReconnection && rcv.mPrefixBounces > 0 && rcv.mPrefixBounces == bounces-1) {
            rcv.mDirOut = r.mDirection;
            throughputAtRcv = throughput;
		}

        dirIn = r.mDirection;

        float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, lightPdf, dist);

        float3 le = vertex.mMaterial.Emission();
        if (vertex.mIsSurface) {
            const float cosIn = -dot(vertex.mShadingNormal, dirIn);
            lightPdf *= sqr(dist) / abs(cosIn);
            if (cosIn <= 0) le = 0;

            // store reconnection vertex if one hasn't been found yet
            if (gReconnection && rcv.mPrefixBounces == 0 && dist >= gReconnectionDistance && vertex.mMaterial.Roughness() >= gReconnectionRoughness && diffuse && (IsDiffuse(vertex) || any(le > 0))) {
                rcv.mVertex = PackVertex(vertex);
                rcv.mCos = cosIn;
                rcv.mDist = dist;
                replayPdfW = prevPdf;
                rcv.mPrefixBounces = bounces;
			}
        }

        if (any(le > 0)) {
            if (gReconnection && rcv.mPrefixBounces > 0) {
                if (rcv.mPrefixBounces == bounces)
                    rcv.mRadiance = le;
				else
	                rcv.mRadiance = le * float3(SafeDivide(throughput.r, throughputAtRcv.r), SafeDivide(throughput.g, throughputAtRcv.g), SafeDivide(throughput.b, throughputAtRcv.b));
			}
            PathReservoir c = PathReservoir(PathSample(throughput * le, replayPdfW > 0 ? replayPdfW : pdfW, bounces, 0, seed, rngIdx, rcv), 1/pdfW, 1);
            c.PrepareMerge(gSampleLights ? Mis(brdfPdfW, lightPdf) : 1);
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
    res.mReplayPdfW = 1;

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
								res.mRadiance = throughput * le;
								res.mReplayPdfW /= l.mG;
								break;
							}
                        }
                    }
                    break;
				} else
					rng.Skip(); // skip light sample rnd
            }

			if (res.mBounces < gMaxBounces) {
				const float p = GetContinuationProbability(vertex);
				if (p == 0) break;
				res.mReplayPdfW *= p;
			}
		}


		// sample brdf
		DirectionSampleRecord r;
		SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection);
		const float brdfPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
        res.mReplayPdfW *= brdfPdfW;
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
		jacobian = basePath.mReplayPdfW / res.mReplayPdfW;
        if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0) {
			IncrementCounter(DebugCounterType::eReconnectionSuccesses);
			if (basePath.mReconnectionVertex.mVertex.mInstanceIndex != INVALID_INSTANCE)
            	jacobian *= (res.mReconnectionVertex.mCos / basePath.mReconnectionVertex.mCos) * sqr(basePath.mReconnectionVertex.mDist / res.mReconnectionVertex.mDist);
        }
    }

    return res;
}
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

float Mis(float a, float b) {
    a = a * a;
    return a / (a + b * b);
}

PathReservoir SamplePath(PathVertex vertex, float3 dirIn, const uint seed, const uint2 rngIdx, const uint maxBounces) {
	PathReservoir res = PathReservoir();
    ReconnectionVertex rcv = ReconnectionVertex();
    float3 throughputAtRcv = 0;

    RandomSampler rng = RandomSampler(seed, rngIdx);

	float3 throughput = 1;
    float pdfW = 1;
    for (uint bounces = 1; bounces <= maxBounces; bounces++) {
        const float3 rnd = rng.NextFloat().xyz;
        const float mergeRnd      = rnd.x;
        const float lightMergeRnd = rnd.y;
        const float contRnd       = rnd.z;

        const bool diffuse = IsDiffuse(vertex);
        if (diffuse) {
			// sample direct light
            if (gSampleLights) {
				DirectLightSampleRecord l = SampleDirectLight(vertex, rng.NextFloat());
                const float3 le = l.mRadiance * EvalBrdf(vertex, -dirIn, l.mToLight);
                if (any(le > 0)) {
                    if (!Occluded(vertex, l.mToLight, l.mDistance)) {
                        if (gReconnection && rcv.mPrefixBounces > 0) {
                            if (rcv.mPrefixBounces == bounces - 1) {
                                rcv.mDirOut = l.mToLight;
                                rcv.mRadiance = l.mRadiance;
                            } else
	                            rcv.mRadiance = le * float3(SafeDivide(throughput.r, throughputAtRcv.r), SafeDivide(throughput.g, throughputAtRcv.g), SafeDivide(throughput.b, throughputAtRcv.b));
                        }
						PathReservoir c = PathReservoir(PathSample(throughput * le, pdfW * l.mPdfW, bounces, true, seed, rngIdx, rcv));
						c.PrepareMerge(Mis(l.mPdfW, EvalBrdfPdfW(vertex, -dirIn, l.mToLight)));
						res.Merge(lightMergeRnd, c);
					}
				}
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

        float G = 1;
        float3 le = vertex.mMaterial.Emission();
        if (vertex.mIsSurface) {
            const float cosIn = -dot(vertex.mShadingNormal, dirIn);
            G = abs(cosIn) / sqr(dist);
            lightPdf /= G;
            if (cosIn <= 0) le = 0;

            // store reconnection vertex if one hasn't been found yet
            if (gReconnection && rcv.mPrefixBounces == 0 && diffuse && (IsDiffuse(vertex) || any(le > 0))) {
                rcv.mVertex = PackVertex(vertex);
                rcv.mG = G;
                rcv.mPrefixPdfW = pdfW / brdfPdfW;
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
            PathReservoir c = PathReservoir(PathSample(throughput * le, pdfW, bounces, false, seed, rngIdx, rcv));
            c.PrepareMerge(gSampleLights ? Mis(brdfPdfW, lightPdf) : 1);
			res.Merge(mergeRnd, c);
		}

        if (bounces < maxBounces) {
        	const float p = GetContinuationProbability(vertex);
			if (p == 0 || contRnd > p)
				break;
			pdfW *= p;
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
    float prevPdfW = 1;

    RandomSampler rng = RandomSampler(basePath.mRngSeed, basePath.mRngIdx);

	float3 throughput = 1;
    for (res.mBounces = 1; res.mBounces <= basePath.mBounces; res.mBounces++) {
        if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0) {
            // reconnect to base path
            if (res.mBounces == basePath.mReconnectionVertex.mPrefixBounces) {
                if (IsDiffuse(vertex)) {
                    const PathVertex rcv = UnpackVertex(basePath.mReconnectionVertex.mVertex);
                    float3 toRcv = rcv.mPosition - vertex.mPosition;
                    const float dist = length(toRcv);
                    toRcv /= dist;
                    if (basePath.mReconnectionVertex.mPrefixBounces < basePath.mBounces || dot(toRcv, rcv.mShadingNormal) < 0) {
						if (!Occluded(vertex, toRcv, dist)) {
							res.mReconnectionVertex.mG = abs(dot(rcv.mShadingNormal, toRcv)) / sqr(dist);
							res.mReconnectionVertex.mPrefixPdfW = prevPdfW;
							res.mRadiance = throughput * res.mReconnectionVertex.mRadiance;
							res.mRadiance *= EvalBrdf(vertex, -dirIn, toRcv);
							if (basePath.mReconnectionVertex.mPrefixBounces < basePath.mBounces)
								res.mRadiance *= EvalBrdf(rcv, -toRcv, basePath.mReconnectionVertex.mDirOut);
							res.mBounces = basePath.mBounces;
							IncrementCounter(DebugCounterType::eReconnectionSuccesses);
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
					DirectLightSampleRecord l = SampleDirectLight(vertex, rng.NextFloat());
					l.mRadiance *= EvalBrdf(vertex, -dirIn, l.mToLight);
					if (any(l.mRadiance > 0)) {
						if (!Occluded(vertex, l.mToLight, l.mDistance)) {
							res.mPdfW *= l.mPdfW;
							res.mRadiance = throughput * l.mRadiance;
							break;
						}
					}
				} else
					rng.Skip(); // skip light sample rnd
            }
		}

		// sample brdf
		DirectionSampleRecord r;
		SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection);
        prevPdfW = res.mPdfW;
		const float brdfPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
        res.mPdfW *= brdfPdfW;
		if (all(throughput <= 0)) break;

        dirIn = r.mDirection;

		float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, lightPdf, dist);

        float3 le = vertex.mMaterial.Emission();
        if (vertex.mIsSurface && dot(vertex.mShadingNormal, dirIn) >= 0)
            le = 0;

        if (res.mBounces == basePath.mBounces && !basePath.mIsNEE && any(le > 0)) {
            res.mRadiance = throughput * le;
            break;
		}

        const float p = GetContinuationProbability(vertex);
        if (p == 0) break;
        res.mPdfW *= p;
	}

    if (any(res.mRadiance > 0)) {
        if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0) {
            jacobian = basePath.mReconnectionVertex.mPrefixPdfW / res.mReconnectionVertex.mPrefixPdfW;
            jacobian *= res.mReconnectionVertex.mG / basePath.mReconnectionVertex.mG;
        } else
            jacobian = basePath.mPdfW / res.mPdfW;
        IncrementCounter(DebugCounterType::eShiftSuccesses);
    }

    return res;
}
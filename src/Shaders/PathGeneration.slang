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

    RandomSampler rng = RandomSampler(seed, rngIdx);

	float3 throughput = 1;
    float pdfW = 1;
    for (uint bounces = 1; bounces < maxBounces; bounces++) {
        const float3 rnd = rng.NextFloat().xyz;
        const float mergeRnd      = rnd.x;
        const float lightMergeRnd = rnd.y;
        const float contRnd       = rnd.z;

        const bool diffuse = IsDiffuse(vertex);
        if (diffuse) {
			// sample direct light
            if (gSampleLights) {
				DirectLightSampleRecord l = SampleDirectLight(vertex, rng.NextFloat());
				l.mRadiance *= EvalBrdf(vertex, -dirIn, l.mToLight);
				if (any(l.mRadiance > 0)) {
					if (!Occluded(vertex, l.mToLight, l.mDistance)) {
						PathReservoir c = PathReservoir(PathSample(throughput * l.mRadiance, pdfW * l.mPdfW, bounces, true, seed, rngIdx, rcv));
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

        float3 le = 0;
        float G = 1;
        float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, r.mDirection), r.mDirection), true, lightPdf, dist);
        dirIn = r.mDirection;

        if (vertex.mIsSurface) {
            const float cosIn = dot(vertex.mShadingNormal, dirIn);
            G = abs(cosIn) / sqr(dist);
            lightPdf /= G;
            le = cosIn < 0 ? vertex.mMaterial.Emission() : 0;
        } else {
            le = EvalBackground(dirIn, lightPdf);
		}

        if (any(le > 0)) {
            PathReservoir c = PathReservoir(PathSample(throughput * le, pdfW, bounces, false, seed, rngIdx, rcv));
            c.PrepareMerge(gSampleLights ? Mis(brdfPdfW, lightPdf) : 1);
			res.Merge(mergeRnd, c);
		}

        const float p = GetContinuationProbability(vertex);
        if (p == 0 || contRnd > p)
			break;
        pdfW *= p;
	}
    res.FinalizeMerge();
    res.mM = 1;
    return res;
}

PathSample ShiftPath(const PathSample basePath, const PackedVertex dstVertex, const float3 cameraPos, out float jacobian) {
    jacobian = 0;

    PathVertex vertex = UnpackVertex(dstVertex);
    if (!vertex.mIsSurface)
        return PathSample();

    float3 dirIn = normalize(vertex.mPosition - cameraPos);

    IncrementCounter(DebugCounterType::eShiftAttempts);

    PathSample res = basePath;
    res.mRadiance = 0;
    res.mPdfW = 1;

    RandomSampler rng = RandomSampler(basePath.mRngSeed, basePath.mRngIdx);

	float3 throughput = 1;
    for (res.mBounces = 1; res.mBounces <= basePath.mBounces; res.mBounces++) {
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
		const float brdfPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
        res.mPdfW *= brdfPdfW;
		if (all(throughput <= 0)) break;

		float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, r.mDirection), r.mDirection), true, lightPdf, dist);
        dirIn = r.mDirection;

        if (!vertex.mIsSurface) {
            if (res.mBounces == basePath.mBounces && !basePath.mIsNEE)
                res.mRadiance = throughput * EvalBackground(dirIn, lightPdf);
			break;
		}

        const float3 le = vertex.mMaterial.Emission();
        if (res.mBounces == basePath.mBounces && !basePath.mIsNEE && any(le > 0)) {
            res.mRadiance = throughput * le;
            break;
		}

        const float p = GetContinuationProbability(vertex);
        if (p == 0) break;
        res.mPdfW *= p;
	}

    if (any(res.mRadiance > 0)) {
        jacobian = basePath.mPdfW / res.mPdfW;
        IncrementCounter(DebugCounterType::eShiftSuccesses);
    }

    return res;
}
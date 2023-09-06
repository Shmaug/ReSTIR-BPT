#pragma once

#include "PathReservoir.slang"
#include "Intersection.slang"
#include "Light.slang"
#include "BRDF.slang"

float Mis(float a, float b) {
    a = a * a;
    return a / (a + b * b);
}

PathReservoir SamplePath(inout RayDesc ray, inout ShadingData sd, inout GpuMaterial m, inout RandomSampler rng, const uint maxBounces) {
    PathReservoir res = PathReservoir();

    res.mSample.mRngSeed = rng;

	float3 throughput = 1;
    float pdfW = 1;
    for (uint bounces = 1; bounces < maxBounces; bounces++) {
        const float3 rnd = rng.NextFloat().xyz;
        const float mergeRnd      = rnd.x;
        const float lightMergeRnd = rnd.y;
        const float contRnd       = rnd.z;

		#ifdef SAMPLE_LIGHTS
		if (IsDiffuse(m)) {
			// sample direct light
			DirectLightSampleRecord l = SampleDirectLight(sd, rng.NextFloat());
			l.mRadiance *= EvalBrdf(sd, m, -ray.Direction, l.mToLight);
			if (any(l.mRadiance > 0)) {
				if (!Occluded(sd, l.mToLight, l.mDistance)) {
					PathReservoir c = PathReservoir(PathSample(throughput * l.mRadiance, pdfW * l.mPdfW, bounces, true, res.mSample.mRngSeed, ReconnectionVertex()));
					c.PrepareMerge(Mis(l.mPdfW, EvalBrdfPdfW(sd, m, -ray.Direction, l.mToLight)));
					res.Merge(lightMergeRnd, c);
				}
			}
		}
		#endif

		// sample brdf
		DirectionSampleRecord r;
		SampleBrdf(sd, m, rng.NextFloat(), -ray.Direction, r);
		throughput *= EvalBrdf(sd, m, -ray.Direction, r.mDirection);
		const float brdfPdfW = EvalBrdfPdfW(sd, m, -ray.Direction, r.mDirection);
        pdfW *= brdfPdfW;
        if (all(throughput <= 0)) break;

        ray = MakeRay(OffsetRayOrigin(sd, r.mDirection), r.mDirection);

        IntersectionResult isect;
		if (!TraceRay(ray, true, sd, isect)) {
            const float3 le = EvalBackground(r.mDirection, isect.mLightPdf);
            PathReservoir c = PathReservoir(PathSample(throughput * le, pdfW, bounces, false, res.mSample.mRngSeed, ReconnectionVertex()));
			#ifdef SAMPLE_LIGHTS
            c.PrepareMerge(Mis(brdfPdfW, isect.mLightPdf));
			#else
			c.PrepareMerge();
			#endif
            res.Merge(mergeRnd, c);
			break;
		}

		m = LoadMaterial(sd);

		const float3 le = m.Emission();
        if (any(le > 0)) {
            isect.mLightPdf *= sqr(isect.mDistance) / abs(dot(ray.Direction, sd.GetGeometryNormal()));
            PathReservoir c = PathReservoir(PathSample(throughput * le, pdfW, bounces, false, res.mSample.mRngSeed, ReconnectionVertex()));
			#ifdef SAMPLE_LIGHTS
			c.PrepareMerge(Mis(brdfPdfW, isect.mLightPdf));
			#else
			c.PrepareMerge();
			#endif
			res.Merge(mergeRnd, c);
		}

        const float p = GetContinuationProbability(m);
        if (contRnd > p)
			break;
		pdfW *= p;
	}
    res.FinalizeMerge();
    res.mM = 1;
    return res;
}

PathSample ShiftPath(const PathSample basePath, const float3 dstVertex, const float3 cameraPos, out float jacobian) {
    jacobian = 0;

    RandomSampler rng = basePath.mRngSeed;
    PathSample res = PathSample(0, 1, 1, basePath.mIsNEE, basePath.mRngSeed, basePath.mReconnectionVertex);

    ShadingData sd;
    IntersectionResult isect;
    RayDesc ray;
    ray.Origin = cameraPos;
    ray.Direction = dstVertex - ray.Origin;
    ray.TMax = length(ray.Direction);
    ray.Direction /= ray.TMax;
    ray.TMin = ray.TMax * 0.95;
    ray.TMax *= 1.05;
    if (!TraceRay(ray, true, sd, isect))
        return res;

    GpuMaterial m = LoadMaterial(sd);

	float3 throughput = 1;
    for (res.mBounces = 1; res.mBounces <= basePath.mBounces; res.mBounces++) {
        rng.Skip(); // skip merge rnds

		#ifdef SAMPLE_LIGHTS
		if (IsDiffuse(m)) {
			if (res.mBounces == basePath.mBounces && basePath.mIsNEE) {
				// sample direct light
				DirectLightSampleRecord l = SampleDirectLight(sd, rng.NextFloat());
				l.mRadiance *= EvalBrdf(sd, m, -ray.Direction, l.mToLight);
				if (any(l.mRadiance > 0)) {
					if (!Occluded(sd, l.mToLight, l.mDistance)) {
						res.mPdfW *= l.mPdfW;
						res.mRadiance = throughput * l.mRadiance;
						break;
					}
				}
			} else
				rng.Skip(); // skip light sample rnd
		}
		#endif

		// sample brdf
		DirectionSampleRecord r;
		SampleBrdf(sd, m, rng.NextFloat(), -ray.Direction, r);
		throughput *= EvalBrdf(sd, m, -ray.Direction, r.mDirection);
		const float brdfPdfW = EvalBrdfPdfW(sd, m, -ray.Direction, r.mDirection);
        res.mPdfW *= brdfPdfW;
		if (all(throughput <= 0)) break;

        ray = MakeRay(OffsetRayOrigin(sd, r.mDirection), r.mDirection);
        if (!TraceRay(ray, true, sd, isect)) {
            if (res.mBounces == basePath.mBounces && !basePath.mIsNEE)
	            res.mRadiance = throughput * EvalBackground(r.mDirection, isect.mLightPdf);
			break;
		}

		m = LoadMaterial(sd);

        const float3 le = m.Emission();
        if (res.mBounces == basePath.mBounces && !basePath.mIsNEE && any(le > 0)) {
            res.mRadiance = throughput * le;
            break;
		}

		res.mPdfW *= GetContinuationProbability(m);
	}

    if (any(res.mRadiance > 0) && res.mBounces == basePath.mBounces)
        jacobian = basePath.mPdfW / res.mPdfW;

    return res;
}
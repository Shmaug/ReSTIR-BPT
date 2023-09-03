#pragma once

#include "PathReservoir.slang"
#include "Light.slang"
#include "Intersection.slang"
#include "BRDF.slang"

#define SAMPLE_LIGHTS

uniform uint gMaxBounces;

struct PathGenerator {
    ShadingData sd;
    GpuMaterial m;
	float3 dirIn;
	RandomSampler rng;
	float3 throughput;
	float pdfW;
	uint bounces;
	uint instanceIndex;
    uint primitiveIndex;

	[mutating]
	void SampleNextVertex() {
		DirectionSampleRecord sr;
		SampleBrdf(sd, m, rng.NextFloat(), -dirIn, sr);
		float3 dirOut = sr.mDirection;

		throughput *= EvalBrdf(sd, m, -dirIn, dirOut);
		pdfW   *= EvalBrdfPdfW(sd, m, -dirIn, dirOut);

		if (!(pdfW > 0) || !any(throughput > 0)) {
			throughput = 0;
			pdfW = 0;
			return;
		}

		IntersectionResult isect;
		if (!TraceRay(MakeRay(OffsetRayOrigin(sd, dirOut), dirOut), true, isect)) {
			m.BaseColor(0);
			float tmp;
			m.Emission(EvalBackground(dirOut, tmp));
		} else
			m = LoadMaterial(sd);
		instanceIndex = isect.mInstanceIndex;
		primitiveIndex = isect.mInstanceIndex;
		dirIn = dirOut;
		bounces++;
    }

    [mutating]
	PathReservoir GeneratePath() {
		PathReservoir r = PathReservoir();
		r.mSample.mRngSeed = rng;

		bool rcvFound = false;
		ReconnectionVertex rcv = ReconnectionVertex();
		float3 throughputAtRcv = 0;

		for (bounces = 0; bounces < gMaxBounces && all(isfinite(sd.mPosition));) {
			#ifdef SAMPLE_LIGHTS
			if (IsDiffuse(m)) {
				DirectLightSampleRecord lr = SampleDirectLight(sd, rng.Next());
				const float3 brdf = EvalBrdf(sd, m, -dirIn, lr.mToLight);
				lr.mRadiance *= brdf;
				if (any(lr.mRadiance > 0)) {
					ReconnectionVertex tmpRcv = ReconnectionVertex();
					#ifdef RECONNECTION
					if (rcvFound) {
						tmpRcv = rcv;
						if (tmpRcv.IsLastVertex()) {
							throughputAtRcv = throughput * brdf;
							tmpRcv.SetDirOut(lr.mToLight);
							tmpRcv.SetIsLastVertex(false);
						}
						tmpRcv.SetRadiance(lr.mRadiance * (throughput / throughputAtRcv));
					}
					#endif

					PathReservoir s = PathReservoir(PathSample(throughput * lr.mRadiance, pdfW * lr.mPdfW, bounces + 1, r.mSample.mRngSeed, tmpRcv));
					s.PrepareMerge();
					r.Merge(rng.NextFloat().x, s);
				} else
					rng.Skip();
			}
			#endif

			bool prevDiffuse = IsDiffuse(m);
			float3 prevPos = sd.mPosition;
			float prevPdfW = pdfW;

			SampleNextVertex();
			if (!any(throughput > 0))
				break;

			#ifdef RECONNECTION
			if (rcvFound && rcv.IsLastVertex()) {
				throughputAtRcv = throughput;
				rcv.SetDirOut(dirIn);
				rcv.SetIsLastVertex(false);
			}
			// store first available reconnection vertex
			if (!rcvFound && prevDiffuse && IsDiffuse(m)) {
				rcvFound = true;
				rcv.SetRadiance(0);
				rcv.SetPrefixBounces(bounces);
				rcv.SetIsLastVertex(true);
				float3 dx = prevPos - sd.mPosition;
				rcv.mG = abs(dot(sd.GetShadingNormal(), normalize(dx))) / dot(dx, dx);
				rcv.mPrefixPdfW = prevPdfW;
                rcv.mVertex.mLocalPosition = TransformPoint(gScene.mInstanceInverseTransforms[instanceIndex], sd.mPosition);
                rcv.mVertex.SetInstanceIndex(instanceIndex);
                rcv.mVertex.SetInstanceIndex(primitiveIndex);
				// throughputAtRcv is assigned in the next iteration
			}
			#endif

			float3 emission = m.Emission();
			if (any(emission > 0)) {
				#ifdef RECONNECTION
				if (rcvFound) {
					if (rcv.IsLastVertex())
						rcv.SetRadiance(emission);
					else
						rcv.SetRadiance(emission * throughput / throughputAtRcv);
				}
				#endif
				PathReservoir s = PathReservoir(PathSample(throughput * emission, pdfW, bounces, r.mSample.mRngSeed, rcv));
				s.PrepareMerge();
				r.Merge(rng.NextFloat().x, s);
			} else
				rng.Skip();

			if (bounces < gMaxBounces && IsDiffuse(m)) {
				float contProb = GetContinuationProbability(m);
				if (rng.NextFloat().x > contProb)
					break;
				pdfW *= contProb;
			}
		}
		r.FinalizeMerge();
		r.mM = 1;
		return r;
    }

    [mutating]
	PathSample GenerateShiftedPath(const PathSample from, out float jacobian) {
		IncrementCounter(DebugCounterType::eShiftAttempts);

		jacobian = 0;

		#ifdef RECONNECTION
		bool hasRcv = any(from.mReconnectionVertex.GetRadiance() > 0);
		if (hasRcv) IncrementCounter(DebugCounterType::eReconnectionAttempts);
		#else
		bool hasRcv = false;
		#endif

		PathSample r = PathSample();

		for (bounces = 0; bounces < from.mBounces && all(isfinite(sd.mPosition));) {
			#ifdef RECONNECTION
			// reconnect to base path
			if (hasRcv) {
				if (bounces >= from.mReconnectionVertex.GetPrefixBounces())
					break;
				if (bounces + 1 == from.mReconnectionVertex.GetPrefixBounces()) {
					if (!IsDiffuse(m))
						break;

                    ShadingData rcvSd = ShadingData::Load(
						from.mReconnectionVertex.mVertex.GetInstanceIndex(),
						from.mReconnectionVertex.mVertex.mLocalPosition,
						from.mReconnectionVertex.mVertex.GetPrimitiveIndex());

					float3 toRcv = rcvSd.mPosition - sd.mPosition;
					const float dist = length(toRcv);
					toRcv /= dist;

					const float3 brdf = EvalBrdf(sd, m, -dirIn, toRcv);
					if (!any(brdf > 0))
						break;

					const GpuMaterial rcvMaterial = LoadMaterial(rcvSd);

					float cosRcv = -dot(rcvSd.GetShadingNormal(), toRcv);

					float3 rcvBrdf = 1;
					if (from.mReconnectionVertex.IsLastVertex())
						rcvBrdf = cosRcv > 0 ? 1 : 0;
					else
						rcvBrdf = EvalBrdf(rcvSd, rcvMaterial, -toRcv, from.mReconnectionVertex.GetDirOut());
					if (!any(rcvBrdf > 0))
						break;

					if (Occluded(sd, toRcv, dist))
						break;

					r.mReconnectionVertex = from.mReconnectionVertex;
					r.mReconnectionVertex.mG = abs(cosRcv) / (dist*dist);
					r.mReconnectionVertex.mPrefixPdfW = pdfW;
					r = PathSample(throughput * brdf * rcvBrdf * from.mReconnectionVertex.GetRadiance(), 0, from.mBounces, from.mRngSeed, r.mReconnectionVertex);
					break;
				}
			}
			#endif

			#ifdef SAMPLE_LIGHTS
			if (IsDiffuse(m)) {
				if (!hasRcv && bounces + 1 == from.mBounces) {
					DirectLightSampleRecord lr = SampleDirectLight(sd, rng.Next());
					lr.mRadiance *= EvalBrdf(sd, m, -dirIn, lr.mToLight);
					rng.Skip();
					if (any(lr.mRadiance > 0)) {
						r = PathSample(throughput * lr.mRadiance, pdfW * lr.mPdfW, bounces + 1, from.mRngSeed, ReconnectionVertex());
						break;
					}
				} else
					rng.Skip(2);
			}
			#endif

			// sample direction

			SampleNextVertex();
			if (!any(throughput > 0))
				break;

			if (!hasRcv && bounces == from.mBounces) {
				float3 emission = m.Emission();
				if (any(emission > 0)) {
					r = PathSample(throughput * emission, pdfW, bounces, from.mRngSeed, ReconnectionVertex());
					break;
				}
			} else
				rng.Skip();

			if (bounces < gMaxBounces && IsDiffuse(m)) {
				float contProb = GetContinuationProbability(m);
				rng.Skip();
				pdfW *= contProb;
			}
		}

		if (any(r.mRadiance > 0)) {
			#ifdef RECONNECTION
			if (hasRcv) {
				jacobian = (from.mReconnectionVertex.mPrefixPdfW / r.mReconnectionVertex.mPrefixPdfW) * (r.mReconnectionVertex.mG / from.mReconnectionVertex.mG);
				if (jacobian > 0)
					IncrementCounter(DebugCounterType::eReconnectionSuccesses);
			} else
			#endif
				jacobian = from.mPdfW / r.mPdfW;
		}

		if (jacobian > 0)
			IncrementCounter(DebugCounterType::eShiftSuccesses);

		return r;
	}
};

PathReservoir SampleRadiance(ShadingData sd, GpuMaterial m, float3 dirIn, const RandomSampler seed) {
    PathGenerator g;
    g.sd = sd;
    g.m = m;
    g.dirIn = dirIn;
    g.rng = seed;
	g.throughput = 1;
	g.pdfW = 1;
	return g.GeneratePath();
}
PathSample ShiftTo(PathSample from, float3 to, float3 cameraPos, out float jacobian) {
    PathGenerator g;
    g.dirIn = normalize(to - cameraPos);
    g.sd = TraceRay(MakeRay(cameraPos, g.dirIn));
    if (!all(isfinite(g.sd.mPosition)))
        return PathSample();
    g.m = LoadMaterial(g.sd);
    g.rng = from.mRngSeed;
    g.throughput = 1;
    g.pdfW = 1;
    return g.GenerateShiftedPath(from, jacobian);
}
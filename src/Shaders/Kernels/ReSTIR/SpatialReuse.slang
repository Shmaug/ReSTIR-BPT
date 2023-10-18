#include "PathGeneration.slang"

Texture2D<uint4> gVertices;

uniform float gMCap;

uint2 GetSampleLocation(const uint2 center, uint index) {
    RandomSampler tmp = RandomSampler(gRandomSeed, center);
	tmp.Skip((2 + gSpatialReusePass) * 1000);
    tmp.Skip(index);
    return center + int2(SampleConcentricDisc(tmp.NextFloat().xy) * gSpatialReuseRadius);
}

// Generalized Talbot RMIS weight
float ComputeSpatialMisWeight(const uint2 center, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex, float candidateTargetPdf) {
    float mi = 0;
    float msum = centerM * sample.mSample.GetTargetPdf() * jacobianToCenter;
    if (candidateIndex == -1)
        mi = msum;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = GetSampleLocation(center, i);
        if (any(p < 0) || any(p >= gOutputSize) || all(p == center))
            continue;

        if (candidateIndex == i) {
            mi = sample.mM * candidateTargetPdf / jacobianToCenter;
            msum += mi;
            continue;
		}

        const PackedVertex vp = reinterpret<PackedVertex>(gVertices[p]);
        if (vp.mInstanceIndex == INVALID_INSTANCE)
            continue;

        const PathReservoir candidate = LoadReservoir(gPathReservoirsIn, p);
        if (!(candidate.mM > 0))
            continue;

        // Determine whether p could have generated sample by shifting sample to p's vertex

        DebugPrintf("Shifting sample (%u %u) to candidate %u\n", sample.mSample.mRngSeed, sample.mSample.mPackedRngIdx, i);

        float jacobian;
        const PathSample shifted = ShiftPath(sample.mSample, vp, gCameraPosition, jacobian);
        DebugPrintf("jacobian: %f\n", jacobian);
        if (jacobian > 0) {
            const float m = candidate.mM * shifted.GetTargetPdf() * jacobian;
            msum += m;
            if (candidateIndex == i)
                mi = m;
        }
    }
    return msum > 0 ? mi / msum : 0;
}
float ComputeSpatialZ(const uint2 center, PathReservoir sample, float centerM, int selectedIndex) {
    float msum = centerM;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = GetSampleLocation(center, i);
        if (any(p < 0) || any(p >= gOutputSize) || all(p == center))
            continue;
        const PackedVertex vp = reinterpret<PackedVertex>(gVertices[p]);
        if (vp.mInstanceIndex == INVALID_INSTANCE)
            continue;

        const PathReservoir candidate = LoadReservoir(gPathReservoirsIn, p);
        if (!(candidate.mM > 0))
            continue;

        if (i == selectedIndex) {
            msum += candidate.mM;
            continue;
        }

        // Determine whether p could have generated sample by shifting sample to p's vertex

        DebugPrintf("Shifting sample (%u %u) to candidate %u\n", sample.mSample.mRngSeed, sample.mSample.mPackedRngIdx, i);

        float jacobian;
        ShiftPath(sample.mSample, vp, gCameraPosition, jacobian);
        DebugPrintf("jacobian: %f\n", jacobian);
        if (jacobian > 0) {
            msum += candidate.mM;
        }
    }
    return msum;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SpatialReuse(uint3 index: SV_DispatchThreadID) {
    uint2 id = index.xy;
    if (any(id >= gOutputSize))
        return;
    InitDebugPixel(id, gOutputSize);

    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE) {
        StoreReservoir(gPathReservoirsOut, id, LoadReservoir(gPathReservoirsIn, id));
        return;
    }

    RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip((2 + gSpatialReusePass) * 10000);
    rng.Skip(gSpatialReuseSamples);

    DebugPrintf("\n\n---------- Performing spatial reuse (%u) ----------\n\n", gRandomSeed);

    PathReservoir r = LoadReservoir(gPathReservoirsIn, id);
    const float centerM = r.mM;

	#ifdef PAIRWISE_RMIS_SPATIAL

    PathReservoir center = r;
    r = PathReservoir();
    uint validNeighbors = 0;
    float mc = 1;

	#elif defined(TALBOT_RMIS_SPATIAL)

	r.PrepareMerge( ComputeSpatialMisWeight(id, r, 1, centerM, -1, r.mSample.GetTargetPdf()) );

	#else

	r.PrepareMerge( centerM );

	#endif

    int selectedIndex = -1;

    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = GetSampleLocation(id, i);
        if (any(p < 0) || any(p >= gOutputSize) || all(p == id))
            continue;

        PathReservoir candidate = LoadReservoir(gPathReservoirsIn, p);

		#ifdef PAIRWISE_RMIS_SPATIAL
        const PackedVertex candidateVertex = reinterpret<PackedVertex>(gVertices[p]);
        if (candidateVertex.mInstanceIndex == INVALID_INSTANCE)
            continue;
        validNeighbors++;
		mc += 1;
        if (center.mW > 0 && candidate.mM > 0) {
            DebugPrintf("Shifting center (%u %u) to candidate %u\n", center.mSample.mRngSeed, center.mSample.mPackedRngIdx, i);
            float jacobianToCandidate;
            const PathSample shifted = ShiftPath(center.mSample, candidateVertex, gCameraPosition, jacobianToCandidate);
            DebugPrintf("center.f: %f -> %f, center.W %f, jacobianToCandidate: %f\n", center.mSample.GetTargetPdf(), shifted.GetTargetPdf(), center.mW, jacobianToCandidate);
            if (jacobianToCandidate > 0) {
                const float f = shifted.GetTargetPdf() * jacobianToCandidate;
                mc -= SafeDivide(candidate.mM * f, candidate.mM * f + (centerM / gSpatialReuseSamples) * center.mSample.GetTargetPdf());
            }
        }
		#endif

        if (!(candidate.mM > 0) || !isfinite(candidate.mM))
            continue;

        if (!(candidate.mW > 0) || !isfinite(candidate.mW)) {
            r.mM += candidate.mM;
            continue;
        }

        IncrementCounter(DebugCounterType::eSpatialSamples);

        const float candidateTargetPdf = candidate.mSample.GetTargetPdf();

        // shift candidate's sample to here
        DebugPrintf("Shifting candidate %u (%d %d) with sample (%u %u) to center\n", i, p.x, p.y, candidate.mSample.mRngSeed, candidate.mSample.mPackedRngIdx);
        float jacobianToCenter;
        candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobianToCenter);

		DebugPrintf("candidate.f: %f -> %f, candidate.W %f, jacobianToCenter: %f\n", candidateTargetPdf, candidate.mSample.GetTargetPdf(), candidate.mW, jacobianToCenter)

        if (jacobianToCenter <= 0 || isnan(jacobianToCenter)) {
            r.mM += candidate.mM;
            continue;
        }

		#ifdef PAIRWISE_RMIS_SPATIAL
        const float f = candidateTargetPdf / jacobianToCenter;
        const float m = SafeDivide(candidate.mM * f, candidate.mM * f + (centerM / gSpatialReuseSamples) * candidate.mSample.GetTargetPdf());
		#elif defined(TALBOT_RMIS_SPATIAL)
		const float m = ComputeSpatialMisWeight(id, candidate, jacobianToCenter, centerM, i, candidateTargetPdf);
		#else
        const float m = candidate.mM;
		#endif

        candidate.PrepareMerge(m, jacobianToCenter);

        DebugPrintf("r ris w: %f\tcandidate ris w: %f\n", r.mW, candidate.mW);

        if (r.Merge(rng.NextFloat().x, candidate)) {
            selectedIndex = i;
            IncrementCounter(DebugCounterType::eSpatialAcceptances);
        }
    }

	#ifdef PAIRWISE_RMIS_SPATIAL
	DebugPrintf("%u valid, canonicalWeight: %f\n", validNeighbors, mc)
    if (center.mW > 0) {
        center.PrepareMerge(mc);
		DebugPrintf("r ris w: %f\tcenter ris w: %f\n", r.mW, center.mW);
        r.Merge(rng.NextFloat().x, center);
    } else
        r.mM += center.mM;
	#endif

	r.FinalizeMerge();

	#ifdef PAIRWISE_RMIS_SPATIAL
	r.mW /= validNeighbors + 1;
	#elif !defined(TALBOT_RMIS_SPATIAL)
	if (r.mW > 0) r.mW = SafeDivide(r.mW, ComputeSpatialZ(id, r, centerM, selectedIndex));
	#endif

    if (gMCap > 0) r.mM = min(r.mM, gMCap);
    StoreReservoir(gPathReservoirsOut, id, r);
}

#define ReadReservoirAtLane(r, lane) \
	r.mW = WaveReadLaneAt(r.mW, lane); \
	r.mM = WaveReadLaneAt(r.mM, lane); \
    r.mSample.mReconnectionVertex.mVertex.mPosition               = WaveReadLaneAt(r.mSample.mReconnectionVertex.mVertex.mPosition              , lane); \
    r.mSample.mReconnectionVertex.mVertex.mInstancePrimitiveIndex = WaveReadLaneAt(r.mSample.mReconnectionVertex.mVertex.mInstancePrimitiveIndex, lane); \
	r.mSample.mReconnectionVertex.mRadiance                       = WaveReadLaneAt(r.mSample.mReconnectionVertex.mRadiance                      , lane); \
	r.mSample.mReconnectionVertex.mPackedDirOut                   = WaveReadLaneAt(r.mSample.mReconnectionVertex.mPackedDirOut                  , lane); \
	r.mSample.mReconnectionVertex.mDist                           = WaveReadLaneAt(r.mSample.mReconnectionVertex.mDist                          , lane); \
	r.mSample.mReconnectionVertex.mCos                            = WaveReadLaneAt(r.mSample.mReconnectionVertex.mCos                           , lane); \
	r.mSample.mRadiance      = WaveReadLaneAt(r.mSample.mRadiance,      lane); \
	r.mSample.mReplayPdfW    = WaveReadLaneAt(r.mSample.mReplayPdfW,    lane); \
	r.mSample.mRngSeed       = WaveReadLaneAt(r.mSample.mRngSeed,       lane); \
	r.mSample.mPackedRngIdx  = WaveReadLaneAt(r.mSample.mPackedRngIdx,  lane); \
	r.mSample.mPackedBounces = WaveReadLaneAt(r.mSample.mPackedBounces, lane); \
	r.mSample.mPacked        = WaveReadLaneAt(r.mSample.mPacked,        lane);

[shader("compute")]
[numthreads(32, 1, 1)]
void WavefrontSpatialReuse(uint3 groupId: SV_GroupID) {
	const uint index = groupId.y * 2048 + groupId.x * 32 + WaveGetLaneIndex();
    const uint pixelIndex     = index / gSpatialReuseSamples;
    const uint candidateIndex = index % gSpatialReuseSamples;
    const uint2 id = uint2(pixelIndex % gOutputSize.x, pixelIndex / gOutputSize.x);
    if (any(id >= gOutputSize))
        return;
    InitDebugPixel(id, gOutputSize);

    PathReservoir center = LoadReservoir(gPathReservoirsIn, id);

    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE) {
		if (candidateIndex == 0)
        	StoreReservoir(gPathReservoirsOut, id, center);
        return;
    }

    if (candidateIndex == 0) {
        DebugPrintf("\n\n---------- Performing spatial reuse (%u) ----------\n\n", gRandomSeed);
    }

    PathReservoir candidate = PathReservoir();
    bool valid = false;
    float mc = 0;

    const int2 p = GetSampleLocation(id, candidateIndex);
    if (!(any(p < 0) || any(p >= gOutputSize) || all(p == id))) {
        const PackedVertex candidateVertex = reinterpret<PackedVertex>(gVertices[p]);
        if (candidateVertex.mInstanceIndex != INVALID_INSTANCE) {
            candidate = LoadReservoir(gPathReservoirsIn, p);
			valid = true;

			// shift center sample to candidate
			if (center.mW > 0) {
				float jacobianToCandidate;
				const float shifted = ShiftPath(center.mSample, candidateVertex, gCameraPosition, jacobianToCandidate).GetTargetPdf();
				if (jacobianToCandidate > 0) {
					const float f = shifted * jacobianToCandidate;
					mc = SafeDivide(candidate.mM * f, candidate.mM * f + (center.mM / gSpatialReuseSamples) * center.mSample.GetTargetPdf());
				}
			}

			if (candidate.mW > 0) {
				IncrementCounter(DebugCounterType::eSpatialSamples);

                DebugPrintf("Shifting candidate %u (%d %d) with sample (%u %u) to center\n", candidateIndex, p.x, p.y, candidate.mSample.mRngSeed, candidate.mSample.mPackedRngIdx);

				// shift candidate to center
				const float candidateTargetPdf = candidate.mSample.GetTargetPdf();
				float jacobianToCenter;
				candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobianToCenter);
				DebugPrintf("candidate.f: %f -> %f, candidate.W %f, jacobianToCenter: %f\n", candidateTargetPdf, candidate.mSample.GetTargetPdf(), candidate.mW, jacobianToCenter)
				if (jacobianToCenter > 0) {
					const float f = candidateTargetPdf / jacobianToCenter;
					const float m = SafeDivide(candidate.mM * f, candidate.mM * f + (center.mM / gSpatialReuseSamples) * candidate.mSample.GetTargetPdf());
					candidate.PrepareMerge(m, jacobianToCenter);
				} else
				candidate.mW = 0;
			}
        }
    }

	RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip((2 + gSpatialReusePass) * 10000);
	rng.Skip(gSpatialReuseSamples);

    uint numValid = 0;
    float canonicalWeight = 0;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        if (valid) {
			numValid++;
            canonicalWeight += 1 - mc;
        }
        mc    = WaveReadLaneAt(mc   , WaveGetLaneIndex() + 1);
        valid = WaveReadLaneAt(valid, WaveGetLaneIndex() + 1);
    }

    center.PrepareMerge(1 + canonicalWeight);
    if (candidateIndex == 0) {
        DebugPrintf("%u valid, canonicalWeight: %f\n", numValid, 1+canonicalWeight)
    }

	for (uint i = 0; i < gSpatialReuseSamples; i++) {
		if (candidateIndex == 0) {
			DebugPrintf("merging with candidate %u: %f\n", i, candidate.mW)
			if (center.Merge(rng.NextFloat().x, candidate))
                IncrementCounter(DebugCounterType::eSpatialAcceptances);
        }
		ReadReservoirAtLane(candidate, WaveGetLaneIndex() + 1);
	}

	if (candidateIndex == 0) {
		center.FinalizeMerge();
		center.mW /= 1 + numValid;

        if (gMCap > 0) center.mM = min(center.mM, gMCap);
        StoreReservoir(gPathReservoirsOut, id, center);
    }
}
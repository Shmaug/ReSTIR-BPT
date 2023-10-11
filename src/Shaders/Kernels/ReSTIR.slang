#include "PathGeneration.slang"

// ---------------------------------------------------------------------------------------------------------
// -------------------------------------------- Canonical paths --------------------------------------------
// ---------------------------------------------------------------------------------------------------------

Texture2D<uint4> gVertices;

[shader("compute")]
[numthreads(8,4,1)]
void SampleCameraPaths(uint3 index : SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    StoreReservoir(gPathReservoirsOut, id, PathReservoir());

    const PathVertex vertex = UnpackVertex(reinterpret<PackedVertex>(gVertices[id]));

    if (vertex.mIsSurface)
        SampleCameraPath(vertex, id);
}

#ifdef BIDIRECTIONAL
[shader("compute")]
[numthreads(8, 4, 1)]
void SampleLightPaths(uint3 index: SV_DispatchThreadID) {
    if (index.y * gOutputSize.x + index.x >= gLightSubpathCount) return;
    InitDebugPixel(-1, gOutputSize);
    SampleLightPath(gRandomSeed, index.xy);
}
#endif

// ---------------------------------------------------------------------------------------------------------
// --------------------------------------------- Temporal reuse --------------------------------------------
// ---------------------------------------------------------------------------------------------------------

ReservoirBuffer gPrevReservoirs;
PathReservoir LoadPrevReservoir(uint2 id) { return LoadReservoir(gPrevReservoirs, id); }

#ifdef gUseDiscardMask
RWTexture2D<float> gHistoryDiscardMask;
#endif
Texture2D<uint4> gPrevVertices;
uniform float4x4 gPrevMVP;
uniform float gMCap;

void DoTemporalReuse(uint2 id, inout PathReservoir r) {
    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    const float4 prevClipPos = mul(gPrevMVP, float4(vertex.mPosition, 1));
    int2 prevPixel = (int2)(((prevClipPos.xy / prevClipPos.w) * .5 + .5) * (float2)gOutputSize);
    if (any(prevPixel < 0) || any(prevPixel >= gOutputSize))
        return;

	#ifdef gCombinedSpatialTemporalReuse
	{
		RandomSampler tmp = RandomSampler(gRandomSeed, id);
		tmp.Skip(0xFFFFF);
		prevPixel += int2(SampleConcentricDisc(tmp.NextFloat().xy) * gTemporalReuseRadius);
	}
	#endif

    const PackedVertex prevVertex = reinterpret<PackedVertex>(gPrevVertices[prevPixel]);
    if (prevVertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    PathReservoir candidate = LoadPrevReservoir(prevPixel);
    if (!(candidate.mM > 0) || isinf(candidate.mM))
        return;

    IncrementCounter(DebugCounterType::eTemporalSamples);

    const float candidateTargetPdf = candidate.mSample.GetTargetPdf();

    float jacobianToCenter = 0;
	float jacobianToPrev = 0;

    if (candidateTargetPdf > 0) {
        DebugPrintf("Shifting candidate to center\n");
		candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobianToCenter);
		if (!(jacobianToCenter > 0)) {
			#ifdef gUseDiscardMask
			gHistoryDiscardMask[id] += candidateTargetPdf * candidate.mM;
			#endif
		}
    }

    DebugPrintf("r.f %f, r.W: %f\ncandidate.f: %f -> %f, candidate.W %f, jacobianToCenter: %f\n", r.mSample.GetTargetPdf(), r.mW, candidateTargetPdf, candidate.mSample.GetTargetPdf(), candidate.mW, jacobianToCenter)

    #ifdef TALBOT_RMIS_TEMPORAL
    // apply mis weight for r.mSample
    {
        float mi = r.mM * r.mSample.GetTargetPdf();
        float msum = mi;
        if (any(r.mSample.mRadiance > 0)) {
            DebugPrintf("Shifting center to candidate\n");
            PathSample shifted = ShiftPath(r.mSample, prevVertex, gPrevCameraPosition, jacobianToPrev);
            msum += candidate.mM * shifted.GetTargetPdf() * jacobianToPrev;
			if (!(jacobianToPrev > 0)) {
				#ifdef gUseDiscardMask
				gHistoryDiscardMask[id] += r.mSample.GetTargetPdf() * candidate.mM;
				#endif
			}
        }

        r.PrepareMerge(SafeDivide(mi, msum));
    }
    // apply mis weight for shifted sample
    {
        float msum = r.mM * candidate.mSample.GetTargetPdf() * jacobianToCenter;
        float mi = SafeDivide(candidate.mM * candidateTargetPdf, jacobianToCenter);
        msum += mi;

        candidate.PrepareMerge(SafeDivide(mi, msum), jacobianToCenter);
    }
	#else
    candidate.PrepareMerge(candidate.mM / (candidate.mM + r.mM), jacobianToCenter);
	if (r.mW > 0) {
		DebugPrintf("Shifting center to candidate\n");
        const float shiftedTargetPdf = ShiftPath(r.mSample, prevVertex, gPrevCameraPosition, jacobianToPrev).GetTargetPdf();
        r.PrepareMerge(r.mM / (r.mM + (shiftedTargetPdf * jacobianToPrev > 0 ? candidate.mM : 0)));
        if (jacobianToPrev <= 0) {
			#ifdef gUseDiscardMask
			gHistoryDiscardMask[id] += r.mSample.GetTargetPdf() * candidate.mM;
			#endif
		}
    }
    #endif

	DebugPrintf("r ris w: %f\ncandidate ris w: %f\n", r.mW, candidate.mW)

    RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip(10000);

    if (r.Merge(rng.NextFloat().x, candidate)) {
        IncrementCounter(DebugCounterType::eTemporalAcceptances);
        DebugPrintf("accepted\n\n");
    } else {
        DebugPrintf("rejected\n\n");
    }

    r.FinalizeMerge();
}

[shader("compute")]
[numthreads(8,4,1)]
void TemporalReuse(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    PathReservoir r = LoadReservoir(id);

    DoTemporalReuse(id, r);

    if (gMCap > 0) r.mM = min(r.mM, gMCap);

    StoreReservoir(id, r);
}


// ---------------------------------------------------------------------------------------------------------
// --------------------------------------------- Spatial reuse ---------------------------------------------
// ---------------------------------------------------------------------------------------------------------

uint2 GetSampleLocation(const uint2 center, uint index) {
    RandomSampler tmp = RandomSampler(gRandomSeed, center);
	tmp.Skip((2 + gSpatialReusePass) * 1000);
    tmp.Skip(index);
    return center + int2(SampleConcentricDisc(tmp.NextFloat().xy) * gSpatialReuseRadius);
}

#ifdef TALBOT_RMIS_SPATIAL
float ComputeSpatialMisWeight(const uint2 center, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex) {
    float mi = 0;
    float msum = centerM * Luminance(sample.mSample.mRadiance) * jacobianToCenter;
    if (candidateIndex == -1)
        mi = msum;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = GetSampleLocation(center, i);
        if (any(p < 0) || any(p >= gOutputSize) || all(p == center))
            continue;
        const PackedVertex vp = reinterpret<PackedVertex>(gVertices[p]);
        if (vp.mInstanceIndex == INVALID_INSTANCE)
            continue;

        const PathReservoir candidate = LoadReservoir(p);
        if (!(candidate.mM > 0))
            continue;

        // Determine whether p could have generated sample by shifting sample to p's vertex

        float jacobian;
        const PathSample shifted = ShiftPath(sample.mSample, vp, gCameraPosition, jacobian);
        if (jacobian > 0) {
            float m = candidate.mM * Luminance(shifted.mRadiance) * jacobian;
            msum += m;
            if (candidateIndex == i)
                mi = m;
        }
    }
    return msum > 0 ? mi / msum : 0;
}
float FinalizeSpatialMisWeight(const uint2 center, PathReservoir sample, float centerM, int selectedIndex) {
    return 1;
}
#else
float ComputeSpatialMisWeight(const uint2 center, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex) {
    return sample.mM;
}
float FinalizeSpatialMisWeight(const uint2 center, PathReservoir sample, float centerM, int selectedIndex) {
    float msum = centerM;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = GetSampleLocation(center, i);
        if (any(p < 0) || any(p >= gOutputSize) || all(p == center))
            continue;
        const PackedVertex vp = reinterpret<PackedVertex>(gVertices[p]);
        if (vp.mInstanceIndex == INVALID_INSTANCE)
            continue;

        const PathReservoir candidate = LoadReservoir(p);
        if (!(candidate.mM > 0))
            continue;

        if (i == selectedIndex) {
            msum += candidate.mM;
            continue;
        }

        // Determine whether p could have generated sample by shifting sample to p's vertex

        float jacobian;
        ShiftPath(sample.mSample, vp, gCameraPosition, jacobian);
        if (jacobian > 0) {
            msum += candidate.mM;
        }
    }
    return msum > 0 ? 1 / msum : 0;
}
#endif

void DoSpatialReuse(uint2 id, inout PathReservoir r) {
    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip((2 + gSpatialReusePass) * 10000);
    rng.Skip(gSpatialReuseSamples);

    int selectedIndex = -1;

    const float centerM = r.mM;
    r.PrepareMerge( ComputeSpatialMisWeight(id, r, 1, centerM, -1) );

    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = GetSampleLocation(id, i);
        if (any(p < 0) || any(p >= gOutputSize) || all(p == id))
            continue;
        const PackedVertex candidateVertex = reinterpret<PackedVertex>(gVertices[p].xyz);
        if (candidateVertex.mInstanceIndex == INVALID_INSTANCE)
            continue;

        PathReservoir candidate = LoadReservoir(p);
        if (!(candidate.mM > 0) || !isfinite(candidate.mM))
            continue;
        if (!(candidate.mW > 0) || !isfinite(candidate.mW)) {
            r.mM += candidate.mM;
            continue;
        }

        IncrementCounter(DebugCounterType::eSpatialSamples);

        // shift candidate's sample to here
        float jacobian;
        candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobian);

        if (jacobian <= 0 || isnan(jacobian)) {
            r.mM += candidate.mM;
            continue;
        }

        candidate.PrepareMerge( ComputeSpatialMisWeight(id, candidate, jacobian, centerM, i), jacobian );

        if (r.Merge(rng.NextFloat().x, candidate)) {
            selectedIndex = i;
            IncrementCounter(DebugCounterType::eSpatialAcceptances);
        }
    }

    if (r.mW > 0) {
        r.mW *= FinalizeSpatialMisWeight(id, r, centerM, selectedIndex);
        r.FinalizeMerge();
    }
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SpatialReuse(uint3 index: SV_DispatchThreadID) {
    uint2 id = index.xy;
    if (any(id >= gOutputSize))
        return;
    InitDebugPixel(id, gOutputSize);

    PathReservoir r = LoadReservoir(id);

	DoSpatialReuse(id, r);

    if (gMCap > 0) r.mM = min(r.mM, gMCap);
    StoreReservoir(id, r);
}


// ---------------------------------------------------------------------------------------------------------
// ------------------------------------------------- Misc --------------------------------------------------
// ---------------------------------------------------------------------------------------------------------


RWTexture2D<float4> gRadiance;

#ifdef BIDIRECTIONAL

[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessCameraConnections(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    const float dist = (1 + 1/512.0)*length(reinterpret<PackedVertex>(gVertices[id]).mPosition.xyz - gCameraPosition);

	#ifdef gNoLightTraceResampling
	{
        PathReservoir r;
        if (isfinite(dist))
            gRadiance[id].rgb += ProcessVisibleLightVertices(id, dist, r);
		#ifdef gLightTraceOnly
		StoreReservoir(id, PathReservoir());
		#endif
	}
	#else
	{
		#ifdef gLightTraceOnly
		PathReservoir r = PathReservoir();
		#else
		PathReservoir r = LoadReservoir(id);
		#endif
		if (isfinite(dist))
			ProcessVisibleLightVertices(id, dist, r);
		StoreReservoir(id, r);
	}
	#endif
}

#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

    float3 radiance = gRadiance[id].rgb;

    const PathReservoir r = LoadReservoir(gPathReservoirsIn, id);
    if (r.mW > 0)
		radiance += r.mSample.mRadiance * r.mW;

	#ifdef DEBUG_PIXEL
    if (gDebugPixel == (gOutputSize.x * id.y + id.x))
        radiance = lerp(radiance, float3(1, 0, 1), 0.25);
	#endif

	gRadiance[id].rgb = radiance;
}

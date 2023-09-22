#ifdef BIDIRECTIONAL
#include "BidirectionalPathGeneration.slang"
#else
#include "PathGeneration.slang"
#endif

[[vk::push_constant]]
cbuffer PushConstants {
    uint gSpatialReusePass;
	uint gSpatialReuseSamples;
    float gSpatialReuseRadius;
    uint gReservoirIndex;
};

ReservoirBuffer gPathReservoirs[2];
#define gPathReservoirsIn gPathReservoirs[gReservoirIndex]
#define gPathReservoirsOut gPathReservoirs[gReservoirIndex^1]

PathReservoir LoadReservoir(const uint2 id) { return LoadReservoir(gPathReservoirsIn, id, gOutputSize); }
void StoreReservoir(const uint2 id, const PathReservoir r) { StoreReservoir(gPathReservoirsOut, r, id, gOutputSize); }


// ---------------------------------------------------------------------------------------------------------
// -------------------------------------------- Canonical paths --------------------------------------------
// ---------------------------------------------------------------------------------------------------------

Texture2D<uint4> gVertices;

[shader("compute")]
[numthreads(8,4,1)]
void SampleCanonicalPaths(uint3 index : SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    sHeatmapPixel = id.y * gOutputSize.x + id.x;

    PathVertex vertex = UnpackVertex(reinterpret<PackedVertex>(gVertices[id]));

    PathReservoir r = PathReservoir();
    if (vertex.mIsSurface)
        r = SamplePath(vertex, gCameraPosition, gRandomSeed, id);

    StoreReservoir(id, r);
}


// ---------------------------------------------------------------------------------------------------------
// --------------------------------------------- Temporal reuse --------------------------------------------
// ---------------------------------------------------------------------------------------------------------

Texture2D<float4> gDepthNormals;
ReservoirBuffer gPrevReservoirs;
PathReservoir LoadPrevReservoir(uint2 id) { return LoadReservoir(gPrevReservoirs, id, gOutputSize); }

#ifdef gUseDiscardMask
RWTexture2D<float> gHistoryDiscardMask;
#endif
Texture2D<uint4> gPrevVertices;
uniform float4x4 gPrevMVP;
uniform float gMCap;
uniform float gReuseX;
bool InScreen(int2 p) { return all(p >= 0) && all(p < gOutputSize) && (gReuseX == 0 || p.x/(float)gOutputSize.x*sign(gReuseX) > gReuseX); }

void DoTemporalReuse(uint2 id, inout PathReservoir r) {
    if (!InScreen(id))
        return;

    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    const float4 prevClipPos = mul(gPrevMVP, float4(gDepthNormals[id].xyz, 1));
    int2 prevPixel = (int2)(((prevClipPos.xy / prevClipPos.w) * .5 + .5) * (float2)gOutputSize);
    if (!InScreen(prevPixel))
        return;

    const PackedVertex prevVertex = reinterpret<PackedVertex>(gPrevVertices[prevPixel]);
    if (prevVertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    PathReservoir candidate = LoadPrevReservoir(prevPixel);
    if (!(candidate.mM > 0) || !isfinite(candidate.mM))
        return;
    if (!(candidate.mW > 0) || !isfinite(candidate.mW)) {
        r.mM += candidate.mM;
        return;
    }

    IncrementCounter(DebugCounterType::eTemporalSamples);

    float candidateTargetPdf = Luminance(candidate.mSample.mRadiance);

    float jacobianToCenter;
    candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobianToCenter);
    if (!(jacobianToCenter > 0)) {
        r.mM += candidate.mM;
		#ifdef gUseDiscardMask
        gHistoryDiscardMask[id] = candidateTargetPdf * candidate.mM;
		#endif
        return;
    }

    #ifdef TALBOT_RMIS_TEMPORAL
    // apply mis weight for r.mSample
    {
        float mi = r.mM * Luminance(r.mSample.mRadiance);
        float msum = mi;
        if (any(r.mSample.mRadiance > 0)) {
            float jacobianToPrev;
            PathSample shifted = ShiftPath(r.mSample, prevVertex, gPrevCameraPosition, jacobianToPrev);
            msum += candidate.mM * Luminance(shifted.mRadiance) * jacobianToPrev;
        }

        r.PrepareMerge(SafeDivide(mi, msum));
    }
    // apply mis weight for candidate.mSample
    {
        float msum = r.mM * Luminance(candidate.mSample.mRadiance) * jacobianToCenter;
        float mi = candidate.mM * candidateTargetPdf / jacobianToCenter;
        msum += mi;

        candidate.PrepareMerge(SafeDivide(mi, msum), jacobianToCenter);
    }
    #else
    float jacobianToPrev = 0;
    if (any(r.mSample.mRadiance > 0))
        ShiftPath(r.mSample, prevVertex, gPrevCameraPosition, jacobianToPrev);
    r.PrepareMerge(Mis(r.mM, jacobianToPrev > 0 ? candidate.mM : 0));
    candidate.PrepareMerge(Mis(candidate.mM, r.mM), jacobianToCenter);
    #endif

    RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip(10000);

    if (r.Merge(rng.NextFloat().x, candidate))
        IncrementCounter(DebugCounterType::eTemporalAcceptances);

    r.FinalizeMerge();
}

[shader("compute")]
[numthreads(8,4,1)]
void TemporalReuse(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    sHeatmapPixel = id.y * gOutputSize.x + id.x;

    PathReservoir r = LoadReservoir(id);

	DoTemporalReuse(id, r);

    if (gMCap > 0) r.mM = min(r.mM, gMCap);

    StoreReservoir(id, r);
}


// ---------------------------------------------------------------------------------------------------------
// --------------------------------------------- Spatial reuse ---------------------------------------------
// ---------------------------------------------------------------------------------------------------------

struct SpatialSampleLocations {
    uint mPackedCenter;

    property int2 mCenter {
        get { return int2(mPackedCenter & 0xFFFF, mPackedCenter >> 16); }
        set { mPackedCenter = (newValue.x & 0xFFFF) | (newValue.y << 16); }
	}

	__init(int2 center) {
		mCenter = center;
	}

    __subscript(uint index) -> int2 {
        get {
            RandomSampler tmp = RandomSampler(gRandomSeed, mCenter);
			tmp.Skip((2 + gSpatialReusePass) * 1000);
			tmp.Skip(index);
			return mCenter + int2(SampleConcentricDisc(tmp.NextFloat().xy) * gSpatialReuseRadius);
        }
    }
};

#ifdef TALBOT_RMIS_SPATIAL
float ComputeSpatialMisWeight(const SpatialSampleLocations sg, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex) {
    float mi = 0;
    float msum = centerM * Luminance(sample.mSample.mRadiance) * jacobianToCenter;
    if (candidateIndex == -1)
        mi = msum;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = sg[i];
        if (!InScreen(p) || all(p == sg.mCenter))
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
float FinalizeSpatialMisWeight(const SpatialSampleLocations sg, PathReservoir sample, float centerM, int selectedIndex) {
    return 1;
}
#else
float ComputeSpatialMisWeight(const SpatialSampleLocations sg, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex) {
    return sample.mM;
}
float FinalizeSpatialMisWeight(const SpatialSampleLocations sg, PathReservoir sample, float centerM, int selectedIndex) {
    float msum = centerM;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        const int2 p = sg[i];
        if (!InScreen(p) || all(p == sg.mCenter))
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

void ComputePairwiseMisWeight(inout PathReservoir r, inout PathReservoir candidate, const PackedVertex candidateVertex, const float candidateTargetPdf, const float jacobianToCenter) {
	// apply mis weight for r.mSample
    {
        float mi = r.mM * Luminance(r.mSample.mRadiance);
        float msum = mi;
        if (any(r.mSample.mRadiance > 0)) {
            float jacobianToCandidate;
            PathSample shifted = ShiftPath(r.mSample, candidateVertex, gCameraPosition, jacobianToCandidate);
            msum += candidate.mM * Luminance(shifted.mRadiance) * jacobianToCandidate;
        }

        r.PrepareMerge(msum > 0 ? mi/msum : 0);
    }
    // apply mis weight for candidate.mSample
    {
        float msum = r.mM * Luminance(candidate.mSample.mRadiance) * jacobianToCenter;
        float mi = candidate.mM * candidateTargetPdf / jacobianToCenter;
        msum += mi;

        candidate.PrepareMerge(msum > 0 ? mi/msum : 0, jacobianToCenter);
    }
}

void DoSpatialReuse(uint2 id, inout PathReservoir r) {
    if (!InScreen(id))
        return;

    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    const SpatialSampleLocations sg = SpatialSampleLocations(id);

    RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip((2 + gSpatialReusePass) * 10000);
    rng.Skip(gSpatialReuseSamples);

    int selectedIndex = -1;

	#ifndef RMIS_PAIRWISE
    const float centerM = r.mM;
    r.PrepareMerge(ComputeSpatialMisWeight(sg, r, 1, centerM, -1));
	#endif

    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        int2 p = sg[i];
        if (!InScreen(p) || all(p == sg.mCenter))
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

        const float candidateTargetPdf = Luminance(candidate.mSample.mRadiance);

        // shift candidate's sample to here
        float jacobian;
        candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobian);

        if (jacobian <= 0 || isnan(jacobian)) {
            r.mM += candidate.mM;
            continue;
        }

        #ifdef RMIS_PAIRWISE
        ComputePairwiseMisWeight(r, candidate, candidateVertex, candidateTargetPdf, jacobian);
		#else
        candidate.PrepareMerge(ComputeSpatialMisWeight(sg, candidate, jacobian, centerM, i), jacobian);
		#endif

        if (r.Merge(rng.NextFloat().x, candidate)) {
            selectedIndex = i;
            IncrementCounter(DebugCounterType::eSpatialAcceptances);
        }

        #ifdef RMIS_PAIRWISE
        r.FinalizeMerge();
		#endif
    }

	#ifndef RMIS_PAIRWISE
    if (r.mW > 0) {
        r.mW *= FinalizeSpatialMisWeight(sg, r, centerM, selectedIndex);
        r.FinalizeMerge();
    }
	#endif
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SpatialReuse(uint3 index: SV_DispatchThreadID) {
    uint2 id = index.xy;
    if (any(id >= gOutputSize))
        return;
    sHeatmapPixel = id.y * gOutputSize.x + id.x;

    PathReservoir r = LoadReservoir(id);
	DoSpatialReuse(id, r);
    if (gMCap > 0) r.mM = min(r.mM, gMCap);
    StoreReservoir(id, r);
}


// ---------------------------------------------------------------------------------------------------------
// ------------------------------------------------- Misc --------------------------------------------------
// ---------------------------------------------------------------------------------------------------------


#ifdef BIDIRECTIONAL

[shader("compute")]
[numthreads(8, 4, 1)]
void TraceLightPaths(uint3 index: SV_DispatchThreadID) {
    const uint id = index.y * gOutputSize.x + index.x;
    if (id >= gLightSubpathCount) return;
    sHeatmapPixel = -1;

    SampleLightPath(gRandomSeed, uint2(id, gOutputSize.y * 10));
}

[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessCameraConnections(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    sHeatmapPixel = id.y * gOutputSize.x + id.x;

    PathReservoir r = LoadReservoir(id);
    ProcessCameraConnections_(id, r);
    StoreReservoir(id, r);
}

#endif

RWTexture2D<float4> gRadiance;

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

    const PathReservoir r = LoadReservoir(gPathReservoirsIn, id, gOutputSize);
    if (r.mW > 0)
        gRadiance[id] += float4(r.mSample.mRadiance * r.mW, 0);
}

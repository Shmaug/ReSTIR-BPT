#include "Common/Math.h"

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
    uint gRandomSeed;
    uint gMaxBounces;
    uint gSpatialReusePass;
};

uniform float4x4 gCameraToWorld;
uniform float4x4 gInverseProjection;
RayDesc GetCameraRay(uint2 id) {
    RayDesc ray;
    ray.TMin = 0;
    ray.TMax = POS_INFINITY;
    ray.Origin = mul(gCameraToWorld, float4(0, 0, 0, 1)).xyz;
    ray.Direction = mul(gInverseProjection, float4((float2(id + 0.5) / float2(gOutputSize)) * 2 - 1, 0, 1)).xyz;
    ray.Direction = normalize(mul(gCameraToWorld, float4(ray.Direction, 0)).xyz);
    return ray;
}

#include "PathGeneration.slang"

StructuredBuffer<PathReservoir> gPathReservoirsIn;
RWStructuredBuffer<PathReservoir> gPathReservoirsOut;

uint FlattenIndex(uint2 id) { return id.y*gOutputSize.x + id.x; }
PathReservoir LoadReservoir(uint2 id) { return gPathReservoirsIn[FlattenIndex(id)]; }
void StoreReservoir(uint2 id, PathReservoir r) { gPathReservoirsOut[FlattenIndex(id)] = r; }


// Canonical paths

RWTexture2D<float4> gRadiance;
RWTexture2D<float4> gAlbedo;
RWTexture2D<float4> gPositions;

[shader("compute")]
[numthreads(8,8,1)]
void SampleCanonicalPaths(uint3 index : SV_DispatchThreadID) {
	const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
	SetHeatmapPixel(id);

    RayDesc ray = GetCameraRay(id);

    IntersectionResult isect;
    ShadingData sd;
    const bool hit = TraceRay(ray, true, sd, isect);
    const bool frontFace = dot(sd.GetShadingNormal(), ray.Direction) < 0;

    gPositions[id] = float4(sd.mPosition, asfloat(PackNormal(sd.GetShadingNormal() * (frontFace ? 1 : -1))));

    PathReservoir r;

    if (hit) {
		GpuMaterial m = LoadMaterial(sd);
		gRadiance[id] = float4(frontFace ? m.Emission() : 0, 1);
		gAlbedo[id]   = float4(m.BaseColor(), 1);
        RandomSampler rng = RandomSampler(gRandomSeed, index.xy);
        r = SamplePath(ray, sd, m, rng, gMaxBounces);
    } else {
		float tmp;
        gRadiance[id] = float4(EvalBackground(ray.Direction, tmp), 1);
        gAlbedo[id]   = 1;
		r = PathReservoir();
	}

    StoreReservoir(id, r);
}


// Temporal reuse

StructuredBuffer<PathReservoir> gPrevReservoirs;
PathReservoir LoadPrevReservoir(uint2 id) { return gPrevReservoirs[FlattenIndex(id)]; }

RWTexture2D<float4> gPrevPositions;
uniform float4x4 gPrevMVP;
uniform float3 gPrevCameraPos;
uniform float gMCap;

uniform float gReuseX;
bool InScreen(int2 p) { return all(p >= 0) && all(p < gOutputSize) && (gReuseX == 0 || p.x/(float)gOutputSize.x*sign(gReuseX) > gReuseX); }

void DoTemporalReuse(uint2 id, inout PathReservoir r) {
    if (!InScreen(id))
        return;

    float4 vertex = gPositions[id];
    if (!all(isfinite(vertex.xyz)))
        return;

    float4 prevClipPos = mul(gPrevMVP, float4(vertex.xyz, 1));
    int2 prevPixel = (int2)(((prevClipPos.xy / prevClipPos.w) * .5 + .5) * (float2)gOutputSize);
    if (!InScreen(prevPixel))
        return;

    float4 prevVertex = gPrevPositions[prevPixel];
    if (!all(isfinite(prevVertex.xyz)))
        return;

    PathReservoir candidate = LoadPrevReservoir(prevPixel);
    if (!(candidate.mM > 0) || !isfinite(candidate.mM))
        return;
    if (!(candidate.mW > 0) || !isfinite(candidate.mW)) {
        r.mM += candidate.mM;
        return;
    }

    IncrementCounter(DebugCounterType::eTemporalSamples);

    float pr = Luminance(candidate.mSample.mRadiance);
    float jacobianToCenter;
    candidate.mSample = ShiftPath(candidate.mSample, vertex.xyz, TransformPoint(gCameraToWorld, 0), jacobianToCenter);
    if (!(jacobianToCenter > 0)) {
        r.mM += candidate.mM;
        return;
    }

    #ifdef TALBOT_RMIS_TEMPORAL
    // apply mis weight for r.mSample
    {
        float mi = r.mM * Luminance(r.mSample.mRadiance);
        float msum = mi;
        if (any(r.mSample.mRadiance > 0)) {
            float jacobianToPrev;
            PathSample shifted = ShiftPath(r.mSample, prevVertex.xyz, gPrevCameraPos, jacobianToPrev);
            msum += candidate.mM * Luminance(shifted.mRadiance) * jacobianToPrev;
        }

        r.PrepareMerge(msum > 0 ? mi/msum : 0);
    }
    // apply mis weight for candidate.mSample
    {
        float msum = r.mM * Luminance(candidate.mSample.mRadiance) * jacobianToCenter;
        float mi = candidate.mM * pr / jacobianToCenter;
        msum += mi;

        candidate.PrepareMerge(msum > 0 ? mi/msum : 0, jacobianToCenter);
    }
    #else
    float jacobianToPrev = 0;
    if (any(r.mSample.mRadiance > 0))
        ShiftPath(r.mSample, prevVertex.xyz, gPrevCameraPos, jacobianToPrev);
    r.PrepareMerge(Mis(r.mM, jacobianToPrev > 0 ? candidate.mM : 0));
    candidate.PrepareMerge(Mis(jacobianToCenter > 0 ? candidate.mM : 0, r.mM), jacobianToCenter);
    #endif

    RandomSampler rng = RandomSampler(gRandomSeed, id, 10000);

    if (r.Merge(rng.NextFloat().x, candidate))
        IncrementCounter(DebugCounterType::eTemporalAcceptances);

    r.FinalizeMerge();
}

[shader("compute")]
[numthreads(8,8,1)]
void TemporalReuse(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    SetHeatmapPixel(id);
    PathReservoir r = LoadReservoir(id);
	DoTemporalReuse(id, r);
    if (gMCap > 0) r.mM = min(r.mM, gMCap);
    StoreReservoir(id, r);
}

// Spatial reuse

uniform uint gSpatialReuseSamples;
uniform float gSpatialReuseRadius;

// Spatial reuse

struct SpatialSampleLocations {
    RandomSampler mRngRandomSeed;
    int2 mCenter;

    int2 Get(uint index) {
        RandomSampler tmp = mRngRandomSeed;
        tmp.Skip(index);
        return mCenter + int2(SampleConcentricDisc(tmp.NextFloat().xy) * gSpatialReuseRadius);
    }
};
SpatialSampleLocations MakeSpatialSampleLocations(inout RandomSampler rng, int2 center) {
    SpatialSampleLocations sg;
    sg.mRngRandomSeed = rng;
    sg.mCenter = center;
    rng.Skip(gSpatialReuseSamples);
    return sg;
}

#ifdef TALBOT_RMIS_SPATIAL
float ComputeSpatialMisWeight(SpatialSampleLocations sg, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex) {
    float mi = 0;
    float msum = centerM * Luminance(sample.mSample.mRadiance) * jacobianToCenter;
    if (candidateIndex == -1)
        mi = msum;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        int2 p = sg.Get(i);
        if (!InScreen(p) || all(p == sg.mCenter))
            continue;
        float4 vp = gPositions[p];
        if (!all(isfinite(vp.xyz)))
            continue;

        PathReservoir candidate = LoadReservoir(p);
        if (!(candidate.mM > 0))
            continue;

        // Determine whether p could have generated sample by shifting sample to p's vertex

        float jacobian;
        PathSample shifted = ShiftPath(sample.mSample, vp.xyz, TransformPoint(gCameraToWorld, 0), jacobian);
        if (jacobian > 0) {
            float m = candidate.mM * Luminance(shifted.mRadiance) * jacobian;
            msum += m;
            if (candidateIndex == i)
                mi = m;
        }
    }
    return msum > 0 ? mi / msum : 0;
}
float FinalizeSpatialMisWeight(SpatialSampleLocations sg, PathReservoir sample, float centerM, int selectedIndex) {
    return 1;
}
#else
float ComputeSpatialMisWeight(SpatialSampleLocations sg, PathReservoir sample, float jacobianToCenter, float centerM, int candidateIndex) {
    return sample.mM;
}
float FinalizeSpatialMisWeight(SpatialSampleLocations sg, PathReservoir sample, float centerM, int selectedIndex) {
    float msum = centerM;
    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        int2 p = sg.Get(i);
        if (!InScreen(p) || all(p == sg.mCenter))
            continue;
        float4 vp = gPositions[p];
        if (!all(isfinite(vp.xyz)))
            continue;

        PathReservoir candidate = LoadReservoir(p);
        if (!(candidate.mM > 0))
            continue;

        if (i == selectedIndex) {
            msum += candidate.mM;
            continue;
        }

        // Determine whether p could have generated sample by shifting sample to p's vertex

        float jacobian;
        PathSample shifted = ShiftPath(sample.mSample, vp.xyz, TransformPoint(gCameraToWorld, 0), jacobian);
        if (jacobian > 0) {
            msum += candidate.mM;
        }
    }
    return msum > 0 ? 1 / msum : 0;
}
#endif

void DoSpatialReuse(uint2 id, inout PathReservoir r) {
    if (!InScreen(id))
        return;

    float4 vertex = gPositions[id];
    if (!all(isfinite(vertex.xyz)))
        return;

    RandomSampler rng = RandomSampler(gRandomSeed, id, (2 + gSpatialReusePass) * 1000);
    SpatialSampleLocations sg = MakeSpatialSampleLocations(rng, id);

    int selectedIndex = -1;

    float centerM = r.mM;
    r.PrepareMerge(ComputeSpatialMisWeight(sg, r, 1, centerM, -1));

    for (uint i = 0; i < gSpatialReuseSamples; i++) {
        int2 p = sg.Get(i);
        if (!InScreen(p) || all(p == sg.mCenter))
            continue;
        if (!all(isfinite(gPositions[p].xyz)))
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
        candidate.mSample = ShiftPath(candidate.mSample, vertex.xyz, TransformPoint(gCameraToWorld, 0), jacobian);

        if (jacobian <= 0 || isnan(jacobian)) {
            r.mM += candidate.mM;
            continue;
        }

        candidate.PrepareMerge(ComputeSpatialMisWeight(sg, candidate, jacobian, centerM, i), jacobian);

        if (r.Merge(rng.NextFloat().x, candidate)) {
            selectedIndex = i;
            IncrementCounter(DebugCounterType::eSpatialAcceptances);
        }
    }

    if (r.mW > 0) {
        r.mW *= FinalizeSpatialMisWeight(sg, r, centerM, selectedIndex);
        r.FinalizeMerge();
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void SpatialReuse(uint3 index: SV_DispatchThreadID) {
    uint2 id = index.xy;
    if (any(id >= gOutputSize))
        return;

    PathReservoir r = LoadReservoir(id);
	DoSpatialReuse(id, r);
    if (gMCap > 0) r.mM = min(r.mM, gMCap);
    StoreReservoir(id, r);
}

// Final output

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

	const PathReservoir r = LoadReservoir(id);
	if (r.mW > 0)
		gRadiance[id] += float4(r.mSample.mRadiance * r.mW, 0);

	#ifdef gEnableDebugCounters
    if (gHeatmapCounterType < (uint)DebugCounterType::eNumDebugCounters) {
		const float total = gDebugCounters[(uint)DebugCounterType::eNumDebugCounters];
		const float val = gHeatmapCounters[id];
		gRadiance[id] = float4(ViridisQuintic(saturate(total > 0 ? val / total : 0)), 1);
    }
	#endif
}
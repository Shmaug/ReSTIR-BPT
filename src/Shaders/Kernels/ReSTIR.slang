#include "Common/Math.h"
uniform float4x4 gCameraToWorld;
uniform float4x4 gInverseProjection;
uniform uint2 gOutputExtent;
RayDesc GetCameraRay(uint2 id) {
    RayDesc ray;
    ray.TMin = 0;
    ray.TMax = POS_INFINITY;
    ray.Origin = mul(gCameraToWorld, float4(0, 0, 0, 1)).xyz;
    ray.Direction = mul(gInverseProjection, float4((float2(id + 0.5) / float2(gOutputExtent)) * 2 - 1, 0, 1)).xyz;
    ray.Direction = normalize(mul(gCameraToWorld, float4(ray.Direction, 0)).xyz);
    return ray;
}

#include "PathGeneration.slang"

StructuredBuffer<PathReservoir> gPathReservoirsIn;
RWStructuredBuffer<PathReservoir> gPathReservoirsOut;
uniform uint gSeed;

uint FlattenIndex(uint2 id) { return id.y*gOutputExtent.x + id.x; }

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
    if (any(id >= gOutputExtent)) return;

    RayDesc ray = GetCameraRay(id);

    IntersectionResult isect;
	const bool hit = TraceRay(ray, true, isect);

    gPositions[id] = float4(isect.mShadingData.mPosition, asfloat(isect.mShadingData.mPackedShadingNormal));

    PathReservoir r = PathReservoir();

    if (hit) {
		const GpuMaterial m = LoadMaterial(isect.mShadingData);
		gRadiance[id]  = float4(m.Emission(), 1);
		gAlbedo[id]    = float4(m.BaseColor(), 1);
        r = SampleRadiance(isect.mShadingData, m, ray.Direction, RandomSampler(gSeed, id, 0));
    } else {
		float tmp;
        gRadiance[id] = float4(EvalBackground(ray.Direction, tmp), 1);
        gAlbedo[id]   = 1;
	}

    StoreReservoir(id, r);
}


uniform float gReuseX;
bool InScreen(int2 p) { return all(p >= 0) && all(p < gOutputExtent) && (gReuseX == 0 || p.x/(float)gOutputExtent.x*sign(gReuseX) > gReuseX); }

// Temporal reuse

StructuredBuffer<PathReservoir> gPrevReservoirs;
PathReservoir LoadPrevReservoir(uint2 id) { return gPrevReservoirs[FlattenIndex(id)]; }

RWTexture2D<float4> gPrevPositions;
uniform float4x4 gPrevWorldToClip;
uniform float3 gPrevCameraPos;
uniform float gMCap;

float Mis(float a, float b) { return a / (a + b); }

void TemporalReuse(uint2 id, inout PathReservoir r) {
    if (!InScreen(id))
        return;

    float4 vertex = gPositions[id];
    if (!all(isfinite(vertex.xyz)))
        return;

    float4 prevClipPos = mul(gPrevWorldToClip, float4(vertex.xyz, 1));
    int2 prevPixel = (int2)(((prevClipPos.xy / prevClipPos.w) * .5 + .5) * (float2)gOutputExtent);
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
    candidate.mSample = ShiftTo(candidate.mSample, vertex.xyz, TransformPoint(gCameraToWorld, 0), jacobianToCenter);
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
            PathSample shifted = ShiftTo(r.mSample, prevVertex, gPrevCameraPos, jacobianToPrev);
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
        ShiftTo(r.mSample, prevVertex.xyz, gPrevCameraPos, jacobianToPrev);
    r.PrepareMerge(Mis(r.mM, jacobianToPrev > 0 ? candidate.mM : 0));
    candidate.PrepareMerge(Mis(jacobianToCenter > 0 ? candidate.mM : 0, r.mM), jacobianToCenter);
    #endif

    RandomSampler rng = RandomSampler(gSeed, id, 1000);

    if (r.Merge(rng.NextFloat().x, candidate)) {
        IncrementCounter(DebugCounterType::eTemporalAcceptances);
    }

    r.FinalizeMerge();
}

[shader("compute")]
[numthreads(8,8,1)]
void TemporalReuse(uint3 index: SV_DispatchThreadID) {
    uint2 id = index.xy;
    if (any(id >= gOutputExtent))
        return;
    PathReservoir r = LoadReservoir(id);
    TemporalReuse(id, r);
    if (gMCap > 0) r.mM = min(r.mM, gMCap);
    StoreReservoir(id, r);
}

// Spatial reuse

uniform uint gSpatialReuseIteration;
uniform uint gSpatialReuseSamples;
uniform float gSpatialReuseRadius;

struct SpatialSampleLocations {
    RandomSampler mRngSeed;
    int2 mCenter;

    int2 Get(uint index) {
        RandomSampler tmp = mRngSeed;
        tmp.Skip(index);
        return mCenter + int2(SampleConcentricDisc(tmp.NextFloat().xy) * gSpatialReuseRadius);
    }
};
SpatialSampleLocations MakeSpatialSampleLocations(inout RandomSampler rng, int2 center) {
    SpatialSampleLocations sg;
    sg.mRngSeed = rng;
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
        PathSample shifted = ShiftTo(sample.mSample, vp.xyz, TransformPoint(gCameraToWorld, 0), jacobian);
        if (jacobian > 0) {
            float m = candidate.mM * Luminance(shifted.mRadiance) * jacobian;
            msum += m;
            if (candidateIndex == i)
                mi = m;
        }
    }
    return msum > 0 ? mi/msum : 0;
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
        PathSample shifted = ShiftTo(sample.mSample, vp.xyz, TransformPoint(gCameraToWorld, 0), jacobian);
        if (jacobian > 0) {
            msum += candidate.mM;
        }
    }
    return msum > 0 ? 1/msum : 0;
}
#endif

void SpatialReuse(uint2 id, inout PathReservoir r) {
    if (!InScreen(id))
        return;

    float4 vertex = gPositions[id];
    if (!all(isfinite(vertex.xyz)))
        return;

    RandomSampler rng = RandomSampler(gSeed, id, (2+gSpatialReuseIteration)*1000);
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
        candidate.mSample = ShiftTo(candidate.mSample, vertex.xyz, TransformPoint(gCameraToWorld, 0), jacobian);

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
[numthreads(8,8,1)]
void SpatialReuse(uint3 index: SV_DispatchThreadID) {
    uint2 id = index.xy;
    if (any(id >= gOutputExtent))
        return;

    PathReservoir r = LoadReservoir(id);
    SpatialReuse(id, r);
    if (gMCap > 0) r.mM = min(r.mM, gMCap);
    StoreReservoir(id, r);
}

[numthreads(8, 8, 1)]
void OutputRadiance(uint3 id: SV_DispatchThreadID) {
    if (all(id.xy < gOutputExtent)) {
        const PathReservoir r = LoadReservoir(id.xy);
        if (r.mW > 0)
            gRadiance[id.xy] += float4(r.mSample.mRadiance * r.mW, 0);
    }
}
#include "PathGeneration.slang"

Texture2D<uint4> gVertices;
RWTexture2D<float4> gRadiance;

[shader("compute")]
[numthreads(8,4,1)]
void SampleCameraPaths(uint3 index : SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    const PathVertex vertex = UnpackVertex(reinterpret<PackedVertex>(gVertices[id]));

    PathReservoir r;
    if (vertex.mIsSurface)
        r = SampleCameraPath(vertex, id);
	else
		r = PathReservoir();

	StoreReservoir(gPathReservoirsOut, id, r);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void OutputRadiance(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

    float3 radiance = gRadiance[id].rgb;

	#if !(defined(gNoLightTraceResampling) && defined(gLightTraceOnly))
    const PathReservoir r = LoadReservoir(gPathReservoirsIn, id);
    if (r.mW > 0)
		radiance += r.mSample.mRadiance * r.mW;
	#endif

	#ifdef DEBUG_PIXEL
    //if (gDebugPixel == (gOutputSize.x * id.y + id.x))
    //    radiance = lerp(radiance, float3(1, 0, 1), 0.25);
	#endif

	gRadiance[id].rgb = radiance;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleLightPaths(uint3 index: SV_DispatchThreadID) {
    if (index.y * gOutputSize.x + index.x >= gLightSubpathCount) return;
    InitDebugPixel(-1, gOutputSize);
    SampleLightPath(gRandomSeed, index.xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessLightTraceReservoirs_(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

	#ifdef gLightTraceOnly
	PathReservoir r = PathReservoir();
	#else
    PathReservoir r = LoadReservoir(gPathReservoirsIn, id);
	#endif

	#if defined(BIDIRECTIONAL) && defined(LIGHT_TRACE_RESERVOIRS)

	#ifndef gNoLightTraceResampling
    r.PrepareMerge();
    RandomSampler rng = RandomSampler(gRandomSeed, id + int2(0, gOutputSize.y));
	#endif

    const uint2 range = gLightTraceReservoirGrid.GetCellDataRange(gLightTraceReservoirGrid.FindCellIndex<false>(float3(int2(id), 0), 1, 0, true));
    for (uint i = range.x; i < range.y; i++) {
        PathReservoir c = LoadReservoir(gLightTraceReservoirs, gLightTraceReservoirGrid.Get(i), gLightTraceReservoirGrid.mMaxSize);
		#ifdef gNoLightTraceResampling
		gRadiance[id].rgb += c.mSample.mRadiance * c.mW;
		#else
		c.PrepareMerge();
		r.Merge(rng.NextFloat().x, c);
		#endif
    }

	#ifndef gNoLightTraceResampling
    r.FinalizeMerge();
    r.mM = 1;
	#endif

	#endif

	StoreReservoir(gPathReservoirsOut, id, r);
}
#include "ReservoirIO.slang"

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
    if (gDebugPixel == (gOutputSize.x * id.y + id.x))
        radiance = lerp(radiance, float3(1, 0, 1), 0.25);
	#endif

	gRadiance[id].rgb = radiance;
}

#ifdef BIDIRECTIONAL

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleLightPaths(uint3 index: SV_DispatchThreadID) {
    if (index.y * gOutputSize.x + index.x >= gLightSubpathCount) return;
    InitDebugPixel(-1, gOutputSize);
    SampleLightPath(gRandomSeed, index.xy);
}

#ifdef PROCESS_LIGHT_VERTICES

[shader("compute")]
[numthreads(8, 4, 1)]
void ProcessCameraConnections(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    const float dist = (1 + 1/512.0)*length(reinterpret<PackedVertex>(gVertices[id]).mPosition.xyz - gCameraPosition);

	#ifdef gLightTraceOnly
	PathReservoir r = PathReservoir();
	#else
    PathReservoir r = LoadReservoir(gPathReservoirsIn, id);
	#endif

    if (isfinite(dist)) {
		#ifdef gNoLightTraceResampling
		gRadiance[id].rgb +=
		#endif
		ProcessVisibleLightVertices(id, dist, r);
	}

	StoreReservoir(gPathReservoirsOut, id, r);
}

#endif
#endif

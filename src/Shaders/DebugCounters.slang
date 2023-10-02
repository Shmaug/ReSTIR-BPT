#pragma once

#include "Common/Enums.h"

RWStructuredBuffer<uint> gDebugCounters;
RWStructuredBuffer<uint> gHeatmap;
uniform int gHeatmapCounterType;
static uint sHeatmapPixel = -1;

#ifdef DEBUG_PIXEL
uniform uint gDebugPixel;
#define DebugPrintf(...) { if (sHeatmapPixel == gDebugPixel) printf(__VA_ARGS__); }
#else
#define DebugPrintf(...) {}
#endif

void InitDebugPixel(const uint2 id, const uint2 extent) {
	sHeatmapPixel = id.y * extent.x + id.x;
}

void IncrementCounter(DebugCounterType counter) {
    #ifdef gEnableDebugCounters
    InterlockedAdd(gDebugCounters[(uint)counter], 1);
    if (gHeatmapCounterType == (int)counter && sHeatmapPixel != -1) {
        uint value;
        InterlockedAdd(gHeatmap[sHeatmapPixel], 1, value);
        InterlockedMax(gDebugCounters[(uint)DebugCounterType::eNumDebugCounters], value + 1);
    }
    #endif
}

#ifdef DEBUG_HEATMAP_SHADER

#include "Common/Math.h"

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
};

RWTexture2D<float4> gRadiance;

[shader("compute")]
[numthreads(8, 8, 1)]
void RenderHeatmap(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    const float val = gHeatmap[id.y * gOutputSize.x + id.x];
    const float total = gDebugCounters[(uint)DebugCounterType::eNumDebugCounters];
    gRadiance[id] = float4(ViridisQuintic(saturate(total > 0 ? val / total : 0)), 1);
}

#endif
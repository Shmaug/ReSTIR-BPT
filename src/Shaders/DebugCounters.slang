#pragma once

#include "Common/Enums.h"

RWStructuredBuffer<uint> gDebugCounters;
RWTexture2D<uint> gHeatmapCounters;
uniform int gHeatmapCounterType;

static uint sHeatmapPixel;
void SetHeatmapPixel(const uint2 idx) {
    sHeatmapPixel = idx.x | (idx.y << 16);
}

void IncrementCounter(DebugCounterType counter) {
    #ifdef gEnableDebugCounters
    InterlockedAdd(gDebugCounters[(uint)counter], 1);
    if (gHeatmapCounterType == (int)counter) {
        uint value;
        InterlockedAdd(gHeatmapCounters[uint2(sHeatmapPixel & 0xFFFF, sHeatmapPixel >> 16)], 1, value);
        InterlockedMax(gDebugCounters[(uint)DebugCounterType::eNumDebugCounters], value + 1);
    }
    #endif
}

#define DebugPrintf(...) { if ((sHeatmapPixel & 0xFFFF) == 500 && (sHeatmapPixel >> 16) == 500) { printf(__VA_ARGS__); }  }
#pragma once

#include "Common/Enums.h"

RWStructuredBuffer<uint> gDebugCounters;
RWTexture2D<uint> gHeatmapCounters;
uniform int gHeatmapCounterType;

void IncrementCounter(DebugCounterType counter) {
    #ifdef gEnableDebugCounters
    InterlockedAdd(gDebugCounters[(uint)counter], 1);
    //if (_HeatmapCounterType == (int)counter) {
    //    uint value;
    //    InterlockedAdd(gHeatmapCounters[DispatchRaysIndex().xy], 1, value);
    //    InterlockedMax(gDebugCounters[DebugCounterType::eNumDebugCounters], value + 1);
    //}
    #endif
}

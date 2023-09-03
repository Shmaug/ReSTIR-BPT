#pragma once

#include "Common.h"

PTVK_NAMESPACE_BEGIN

enum class TonemapMode {
	eRaw,
	eReinhard,
	eReinhardExtended,
	eReinhardLuminance,
	eReinhardLuminanceExtended,
	eUncharted2,
	eFilmic,
	eACES,
	eACESApprox,
	eViridisR,
	eViridisLengthRGB,
	eTonemapModeCount
};
#ifdef __cplusplus
static const char* TonemapModeStrings[] = {
	"Raw",
	"Reinhard",
	"Reinhard Extended",
	"Reinhard Luminance",
	"Reinhard Luminance Extended",
	"Uncharted 2",
	"Filmic",
	"ACES",
	"ACES Approximated",
	"Viridis R",
	"Viridis length(RGB)",
};
#endif

enum class DenoiserDebugMode {
	eNone,
	eSampleCount,
	eVariance,
	eWeightSum,
	eDebugModeCount
};
#ifdef __cplusplus
static const char* DenoiserDebugModeStrings[] = {
	"None",
	"Sample Count",
	"Variance",
	"Weight Sum",
};
#endif


enum DebugCounterType {
    eRays,
    eShadowRays,

    eShiftAttempts,
    eShiftSuccesses,

    eReconnectionAttempts,
    eReconnectionSuccesses,

    eTemporalSamples,
    eTemporalAcceptances,
    eSpatialSamples,
    eSpatialAcceptances,

    eNumDebugCounters
};
#ifdef __cplusplus
static const char* DebugCounterTypeStrings[] = {
    "Rays",
    "Shadow Rays",

    "Shift Attempts",
    "Shift Successes",

    "Reconnection Attempts",
    "Reconnection Successes",

    "Temporal Samples",
    "Temporal Acceptances",
    "Spatial Samples",
    "Spatial Acceptances"
};
#endif


enum class ImageCompareMode {
	eSMAPE,
	eMSE,
	eAverage,
	eNumImageCompareModes
};

enum class FilterKernelType {
	eAtrous,
	eBox3,
	eBox5,
	eSubsampled,
	eBox3Subsampled,
	eBox5Subsampled,
	eFilterKernelTypeCount
};

PTVK_NAMESPACE_END
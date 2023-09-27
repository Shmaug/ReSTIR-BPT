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
	eRejectionMask,
	eRejectionResponse,
	eDebugModeCount
};
#ifdef __cplusplus
static const char* DenoiserDebugModeStrings[] = {
	"None",
	"Sample Count",
	"Variance",
	"Weight Sum",
	"Rejection Mask",
	"Rejection Response",
};
#endif


enum DebugCounterType {
    eRays,
    eShadowRays,

    eShiftAttempts,
    eShiftSuccesses,

    eLightShiftAttempts,
    eLightShiftSuccesses,

	eReconnectionVertices,
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

    "Light Shift Attempts",
    "Light Shift Successes",

    "Reconnection Vertices",
    "Reconnection Attempts",
    "Reconnection Successes",

    "Temporal Samples",
    "Temporal Acceptances",
    "Spatial Samples",
    "Spatial Acceptances",

	"None"
};
#endif

enum class FilterKernel {
	eAtrous,
	eBox3,
	eBox5,
	eSubsampled,
	eBox3Subsampled,
	eBox5Subsampled,
	eFilterKernelTypeCount
};
#ifdef __cplusplus
static const char* FilterKernelStrings[] = {
	"Atrous",
	"Box3",
	"Box5",
	"Subsampled",
	"Box3, then subsampled",
	"Box5, then subsampled",

	"None"
};
#endif

PTVK_NAMESPACE_END
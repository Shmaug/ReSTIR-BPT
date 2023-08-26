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
}

enum class ImageCompareMode {
	eSMAPE,
	eMSE,
	eAverage,
	eNumImageCompareModes
}

enum class FilterKernelType {
	eAtrous,
	eBox3,
	eBox5,
	eSubsampled,
	eBox3Subsampled,
	eBox5Subsampled,
	eFilterKernelTypeCount
}

enum class DenoiserDebugMode {
	eNone,
	eSampleCount,
	eVariance,
	eWeightSum,
	eDebugModeCount
}

PTVK_NAMESPACE_END
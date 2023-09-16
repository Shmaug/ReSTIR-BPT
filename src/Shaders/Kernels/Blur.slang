#include "Common/Enums.h"

[[vk::push_constant]]
cbuffer PushConstants {
    int2 gOutputSize;
    uint gStepSize;
    uint gIteration;
};

Texture2D<float> gInput;
RWTexture2D<float> gOutput;

#ifndef gFilterKernel
#define gFilterKernel (FilterKernel)0
#endif

static int2 sCenter;
static float sSum;
static float sWeightSum;

void Tap(int2 pos, float w) {
    pos = pos * gStepSize + sCenter;
    if (any(pos < 0) || any(pos >= gOutputSize) || all(pos == sCenter))
        return;

	const float v = gInput[pos];

	#ifdef gMaxFilter
    sSum = max(sSum, v);
	#else
    sSum += v*w;
    sWeightSum += w;
	#endif
}

void subsampled() {
    /*
    | | |x| | |
    | |x| |x| |
    |x| |x| |x|
    | |x| |x| |
    | | |x| | |
    */

    if ((gIteration & 1) == 0) {
        /*
        | | | | | |
        | |x| |x| |
        |x| |x| |x|
        | |x| |x| |
        | | | | | |
        */
        Tap(int2(-2, 0), 1.0);
        Tap(int2(2, 0), 1.0);
    } else {
        /*
        | | |x| | |
        | |x| |x| |
        | | |x| | |
        | |x| |x| |
        | | |x| | |
        */
        Tap(int2(0, -2), 1.0);
        Tap(int2(0, 2), 1.0);
    }

    Tap(int2(-1, 1), 1.0);
    Tap(int2(1, 1), 1.0);

    Tap(int2(-1, -1), 1.0);
    Tap(int2(1, -1), 1.0);
}

void box3() {
    const int r = 1;
    for (int yy = -r; yy <= r; yy++)
        for (int xx = -r; xx <= r; xx++)
            if (xx != 0 || yy != 0)
                Tap(int2(xx, yy), 1.0);
}

void box5() {
    const int r = 2;
    for (int yy = -r; yy <= r; yy++)
        for (int xx = -r; xx <= r; xx++)
            if (xx != 0 || yy != 0)
                Tap(int2(xx, yy), 1.0);
}

void atrous() {
    Tap(int2(1, 0), 2.0 / 3.0);
    Tap(int2(0, 1), 2.0 / 3.0);
    Tap(int2(-1, 0), 2.0 / 3.0);
    Tap(int2(0, -1), 2.0 / 3.0);

    Tap(int2(2, 0), 1.0 / 6.0);
    Tap(int2(0, 2), 1.0 / 6.0);
    Tap(int2(-2, 0), 1.0 / 6.0);
    Tap(int2(0, -2), 1.0 / 6.0);

    Tap(int2(1, 1), 4.0 / 9.0);
    Tap(int2(-1, 1), 4.0 / 9.0);
    Tap(int2(-1, -1), 4.0 / 9.0);
    Tap(int2(1, -1), 4.0 / 9.0);

    Tap(int2(1, 2), 1.0 / 9.0);
    Tap(int2(-1, 2), 1.0 / 9.0);
    Tap(int2(-1, -2), 1.0 / 9.0);
    Tap(int2(1, -2), 1.0 / 9.0);

    Tap(int2(2, 1), 1.0 / 9.0);
    Tap(int2(-2, 1), 1.0 / 9.0);
    Tap(int2(-2, -1), 1.0 / 9.0);
    Tap(int2(2, -1), 1.0 / 9.0);

    Tap(int2(2, 2), 1.0 / 36.0);
    Tap(int2(-2, 2), 1.0 / 36.0);
    Tap(int2(-2, -2), 1.0 / 36.0);
    Tap(int2(2, -2), 1.0 / 36.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 index: SV_DispatchThreadID) {
    if (any(index.xy >= gOutputSize))
        return;

    sCenter = index.xy;
    sSum = gInput[sCenter];
	sWeightSum = 1;

    switch (gFilterKernel) {
    case FilterKernel::eAtrous:
        atrous();
        break;
    default:
    case FilterKernel::eBox3:
        box3();
        break;
    case FilterKernel::eBox5:
        box5();
        break;
    case FilterKernel::eSubsampled:
        subsampled();
        break;
    case FilterKernel::eBox3Subsampled:
        if (gIteration == 0)
            box3();
        else
            subsampled();
        break;
    case FilterKernel::eBox5Subsampled:
        if (gIteration == 0)
            box5();
        else
            subsampled();
        break;
    }

    gOutput[index.xy] = sSum / sWeightSum;
}

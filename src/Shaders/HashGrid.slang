#pragma once

#include "Random.slang"

struct HashGrid<T> {
	RWStructuredBuffer<uint> mChecksums;
    // mCellCounters[cellIndex] = number of items in cell
	RWStructuredBuffer<uint> mCellCounters;
    // mOtherCounters[0] = number of appended items
    // mOtherCounters[1] = prefix sum counter
	RWStructuredBuffer<uint> mOtherCounters;

	// stores (cellIndex, indexInCell) for each item in mAppendData
    RWStructuredBuffer<uint2> mAppendDataIndices;
    RWStructuredBuffer<T> mAppendData;
    RWStructuredBuffer<T> mData;

    // mIndices[cellIndex] = offset in mData
	// mCellCounters[cellIndex] = num items in cell
	RWStructuredBuffer<uint> mIndices;

    uniform float mCellPixelRadius;
    uniform float mMinCellSize;
    uniform uint mCellCount;
    uniform uint mMaxSize;
    uniform float3 mCameraPosition;
    uniform float mDistanceScale;

    uint GetCurrentElementCount() { return mOtherCounters[0]; }

    uint2 GetCellDataRange(const uint aCellIndex) {
        if (aCellIndex == -1)
            return 0;
        uint begin = mIndices[aCellIndex];
		return uint2(begin, begin + mCellCounters[aCellIndex]);
	}

    T Get(const uint aIndex) {
        return mData[aIndex];
	}


    float GetCellSize(const float3 aPosition) {
        if (mCellPixelRadius <= 0)
            return mMinCellSize;
        const float cameraDistance = length(mCameraPosition - aPosition);
        const float step = cameraDistance * mDistanceScale;
		return mMinCellSize * (1 << uint(log2(step / mMinCellSize)));
	}

    uint FindCellIndex<let bInsert : bool>(const float3 aPosition, float aCellSize = 0, const int3 aOffset = 0, bool use2dChecksum = false) {
        if (aCellSize == 0) aCellSize = GetCellSize(aPosition);
        // compute index in hash grid
        const int3 p = int3(floor(aPosition / aCellSize)) + aOffset;
        const uint checksum = use2dChecksum ? ((1 + p.x) | (p.y << 16)) : max(1, xxhash32(xxhash32(asuint(aCellSize)) + xxhash32(p.z + xxhash32(p.y + xxhash32(p.x)))));
        const uint baseCellIndex = pcg(pcg(asuint(aCellSize)) + pcg(p.z + pcg(p.y + pcg(p.x)))) % mCellCount;

        // resolve hash collisions with linear probing
        for (uint i = 0; i < 32; i++) {
            const uint cellIndex = (baseCellIndex + i) % mCellCount;
            // find cell with matching checksum, or empty cell if inserting
            if (bInsert) {
                uint prevChecksum;
                InterlockedCompareExchange(mChecksums[cellIndex], 0, checksum, prevChecksum);
                if (prevChecksum == 0 || prevChecksum == checksum)
                    return cellIndex;
            } else {
                if (mChecksums[cellIndex] == checksum)
                    return cellIndex;
            }
        }

        // collision resolution failed - hashgrid full?
        return -1;
    }

    uint Append(const float3 aPosition, const T data, float aCellSize = 0, bool use2dChecksum = false) {
        if (aCellSize == 0)
			aCellSize = GetCellSize(aPosition);

        const uint cellIndex = FindCellIndex<true>(aPosition, aCellSize, 0, use2dChecksum);
        if (cellIndex == -1)
			return -1;

		// store payload
		uint appendIndex;
		InterlockedAdd(mOtherCounters[0], 1, appendIndex);
        if (appendIndex >= mMaxSize)
            return -1;

		// append item to cell by incrementing cell counter
		uint indexInCell;
		InterlockedAdd(mCellCounters[cellIndex], 1, indexInCell);

		mAppendDataIndices[appendIndex] = uint2(cellIndex, indexInCell);
        mAppendData[appendIndex] = data;
        return appendIndex;
	}

    uint2 GetCellDataRange(const float3 aPosition, float aCellSize = 0, const int3 aOffset = 0, bool use2dChecksum = false) {
        const uint cellIndex = FindCellIndex<false>(aPosition, aCellSize, aOffset, use2dChecksum);
        if (cellIndex == -1) return 0;
        return GetCellDataRange(cellIndex);
    }
};

#ifdef HASHGRID_SHADER

#ifndef N
#define N 1
#endif

struct DataType {
    uint data[N];
};

ParameterBlock<HashGrid<DataType>> gHashGrid;

// Prefix sum over cell counter values to determine indices
[shader("compute")]
[numthreads(64, 1, 1)]
void ComputeIndices(uint3 index: SV_DispatchThreadID) {
    const uint cellIndex = index.y * 1024 + index.x;
    if (cellIndex >= gHashGrid.mCellCount) return;

    uint offset;
    InterlockedAdd(gHashGrid.mOtherCounters[1], gHashGrid.mCellCounters[cellIndex], offset);

    gHashGrid.mIndices[cellIndex] = offset;
}

// Sort items from append order into cells
[shader("compute")]
[numthreads(64, 1, 1)]
void Swizzle(uint3 index: SV_DispatchThreadID) {
    const uint elementIndex = index.y * 1024 + index.x;
    if (elementIndex >= gHashGrid.GetCurrentElementCount()) return;

    const uint2 dataIndices = gHashGrid.mAppendDataIndices[elementIndex];
	const uint cellIndex   = dataIndices[0];
    const uint indexInCell = dataIndices[1];
    const DataType data = gHashGrid.mAppendData[elementIndex];

    gHashGrid.mData[gHashGrid.GetCellDataRange(cellIndex).x + indexInCell] = data;
}

#endif
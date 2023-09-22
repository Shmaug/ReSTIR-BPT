#pragma once

#include <Core/PipelineCache.hpp>

namespace ptvk {

struct HashGrid {
private:
	ComputePipelineCache mComputeIndicesPipeline, mSwizzlePipeline;

public:
	uint32_t mSize = 1;
	uint32_t mElementSize = sizeof(uint32_t);
	uint32_t mCellCount = 16384;
	float mCellSize = .01f;
	float mCellPixelRadius = 0;
	ShaderParameterBlock mParameters;

	HashGrid() = default;
	HashGrid(const HashGrid&) = default;
	HashGrid(HashGrid&&) = default;
	HashGrid& operator=(const HashGrid&) = default;
	HashGrid& operator=(HashGrid&&) = default;

	inline HashGrid(const Instance& instance) {
		const std::filesystem::path shaderPath = *instance.GetOption("shader-kernel-path");
		mComputeIndicesPipeline = ComputePipelineCache(shaderPath / "HashGrid.slang", "ComputeIndices", "sm_6_7", { "-O3", "-capability", "spirv_1_5" });
		mSwizzlePipeline        = ComputePipelineCache(shaderPath / "HashGrid.slang", "Swizzle"       , "sm_6_7", { "-O3", "-capability", "spirv_1_5" });
	};

	inline void Prepare(CommandBuffer& commandBuffer, const float3& cameraPos, const float& verticalFov, const uint2 extent) {
		if (!mParameters.Contains("mChecksums") || mParameters.GetBuffer<uint32_t>("mChecksums").size() != mCellCount) {
			mParameters.SetBuffer("mChecksums"    , std::make_shared<Buffer>(commandBuffer.mDevice, "mChecksums",    mCellCount*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst) );
			mParameters.SetBuffer("mIndices"      , std::make_shared<Buffer>(commandBuffer.mDevice, "mIndices",      mCellCount*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer) );
			mParameters.SetBuffer("mCellCounters" , std::make_shared<Buffer>(commandBuffer.mDevice, "mCellCounters", mCellCount*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst) );
			mParameters.SetBuffer("mOtherCounters", std::make_shared<Buffer>(commandBuffer.mDevice, "mOtherCounters",         4*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst) );
		}
		if (!mParameters.Contains("mDataIndices") || mParameters.GetBuffer<uint32_t>("mDataIndices").size() != mSize) {
			mParameters.SetBuffer("mAppendDataIndices", std::make_shared<Buffer>(commandBuffer.mDevice, "mAppendDataIndices", mSize*sizeof(uint2),    vk::BufferUsageFlagBits::eStorageBuffer) );
			mParameters.SetBuffer("mDataIndices"      , std::make_shared<Buffer>(commandBuffer.mDevice, "mDataIndices",       mSize*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer) );
		}
		if (!mParameters.Contains("mAppendData") || mParameters.GetBuffer<std::byte>("mAppendData").size() != mSize*mElementSize) {
			mParameters.SetBuffer("mAppendData", std::make_shared<Buffer>(commandBuffer.mDevice, "mAppendData", mSize*mElementSize, vk::BufferUsageFlagBits::eStorageBuffer) );
		}

		commandBuffer.Fill(mParameters.GetBuffer<uint32_t>("mChecksums"), 0);
		commandBuffer.Fill(mParameters.GetBuffer<uint32_t>("mCellCounters"), 0);
		commandBuffer.Fill(mParameters.GetBuffer<uint32_t>("mOtherCounters"), 0);

		mParameters.SetConstant("mCellPixelRadius", mCellPixelRadius);
		mParameters.SetConstant("mMinCellSize", mCellSize);
		mParameters.SetConstant("mCellCount", mCellCount);
		mParameters.SetConstant("mMaxSize", mSize);
		mParameters.SetConstant("mCameraPosition", cameraPos);
		mParameters.SetConstant("mDistanceScale", tan(mCellPixelRadius * verticalFov * max(1.0f / extent[1], extent[1] / (float)(extent[0]*extent[0]))));
	}

	inline void Build(CommandBuffer& commandBuffer) {
		Defines defs {
			{ "HASHGRID_SHADER", "true" },
			{ "N", std::to_string(mElementSize/4) } };
		const ShaderParameterBlock params = ShaderParameterBlock().SetParameters("gHashGrid", mParameters);
		mComputeIndicesPipeline.Dispatch(commandBuffer, vk::Extent3D{1024, (mCellCount + 1023)/1024, 1}, params, defs);
		mSwizzlePipeline.Dispatch(commandBuffer, vk::Extent3D{1024, (mSize + 1023)/1024, 1}, params, defs);
	}
};

}
#pragma once


#include "ResourceQueue.hpp"
#include "CommandBuffer.hpp"
#include "ShaderParameterBlock.hpp"
#include "Profiler.hpp"

namespace ptvk {

class ComputePipelineCache {
public:
	ComputePipelineCache() = default;
	ComputePipelineCache(const ComputePipelineCache&) = default;
	ComputePipelineCache(ComputePipelineCache&&) = default;
	ComputePipelineCache& operator=(const ComputePipelineCache&) = default;
	ComputePipelineCache& operator=(ComputePipelineCache&&) = default;
	ComputePipelineCache(const std::filesystem::path& sourceFile, const std::string& entryPoint = "main", const std::string& profile = "sm_6_7", const std::vector<std::string>& compileArgs = {}, const PipelineInfo& pipelineInfo = {})
		: mSourceFile(sourceFile), mEntryPoint(entryPoint), mProfile(profile), mCompileArgs(compileArgs), mPipelineInfo(pipelineInfo) {}

	inline const std::shared_ptr<ComputePipeline>& GetPipeline(Device& device, const Defines& defines = {}, const std::optional<PipelineInfo>& info = std::nullopt) {
		auto writeTime = std::filesystem::last_write_time(mSourceFile);
		if (writeTime > mLastWriteTime) {
			mLastWriteTime = writeTime;
			mCachedSpirvHashes.clear();
		}

		size_t infoHash = 0;
		if (info) {
			infoHash = HashArgs(
				info->mStageLayoutFlags,
				info->mLayoutFlags,
				info->mFlags,
				info->mDescriptorSetLayoutFlags );
			for (const auto&[name,s] : info->mImmutableSamplers) infoHash = HashArgs(infoHash, name, HashRange(s));
			for (const auto&[name,f] : info->mBindingFlags)      infoHash = HashArgs(infoHash, name, f);
		}

		size_t key = infoHash;
		for (const auto&[n,v] : defines) key = HashArgs(key, n, v);

		if (auto it = mCachedSpirvHashes.find(key); it != mCachedSpirvHashes.end())
			return mCachedPipelines.at(it->second);

		auto shader = std::make_shared<Shader>(device, mSourceFile, mEntryPoint, mProfile, mCompileArgs, defines);

		size_t spirvHash = HashCombine(shader->SpirvHash(), infoHash);
		mCachedSpirvHashes[key] = spirvHash;

		if (auto it = mCachedPipelines.find(spirvHash); it != mCachedPipelines.end())
			return it->second;

		std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>> descriptorSetLayouts;
		// TODO: descriptorSetLayouts
		auto pipeline = std::make_shared<ComputePipeline>(mSourceFile.stem().string() + "/" + mEntryPoint, shader, info ? *info : mPipelineInfo, descriptorSetLayouts);

		mCachedPipelines[spirvHash] = pipeline;
		return mCachedPipelines[spirvHash];
	}

	inline void Dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const ShaderParameterBlock& params, const Defines& defines = {}, const std::optional<PipelineInfo>& info = std::nullopt) {
		ProfilerScope p("ComputePipelineCache::Dispatch");

		const ComputePipeline& pipeline = *GetPipeline(commandBuffer.mDevice, defines, info);

		// add barriers as needed

		for (const auto& [id, param] : params) {
			auto it = pipeline.GetDescriptors().find(id.first);
			if (it == pipeline.GetDescriptors().end())
				continue;
			const Shader::DescriptorBinding& binding = it->second;

			if (const auto* v = std::get_if<ImageParameter>(&param)) {
				const auto& [image, layout, accessFlags, sampler] = *v;
				commandBuffer.Barrier(image, layout, vk::PipelineStageFlagBits::eComputeShader, accessFlags);
			} else if (const auto* v = std::get_if<BufferParameter>(&param)) {
				const auto& buffer = *v;
				vk::AccessFlags access = vk::AccessFlagBits::eNone;
				switch (binding.mDescriptorType) {
					case vk::DescriptorType::eUniformBuffer:
					case vk::DescriptorType::eUniformBufferDynamic:
					case vk::DescriptorType::eUniformTexelBuffer:
					case vk::DescriptorType::eInlineUniformBlock:
						access = vk::AccessFlagBits::eUniformRead;
						break;
					default:
					case vk::DescriptorType::eStorageBuffer:
					case vk::DescriptorType::eStorageBufferDynamic:
					case vk::DescriptorType::eStorageTexelBuffer:
						access = binding.mWritable ? vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite : vk::AccessFlagBits::eShaderRead;
						break;
				}
				commandBuffer.Barrier(buffer, vk::PipelineStageFlagBits::eComputeShader, access);
			}
		}

		commandBuffer.BindPipeline(pipeline);
		const auto data = mCachedParameters[&pipeline].Get(commandBuffer.mDevice);// std::make_shared<ParameterData>();
		data->SetParameters(commandBuffer, pipeline, params);
		data->Bind(commandBuffer, pipeline);
		commandBuffer.Dispatch(pipeline.GetDispatchDim(dim));
	}

private:
	struct ParameterData {
		std::vector<std::shared_ptr<vk::raii::DescriptorSet>> mDescriptorSets;
		std::vector<std::pair<Buffer::View<std::byte>, Buffer::View<std::byte>>> mUniformBuffers;

		void SetParameters(CommandBuffer& commandBuffer, const Pipeline& pipeline, const ShaderParameterBlock& params);

		inline void Bind(CommandBuffer& commandBuffer, const Pipeline& pipeline) const {
			ProfilerScope p("ComputePipelineCache::ParameterData::Bind");
			for (const auto& ds : mDescriptorSets)
				commandBuffer.HoldResource(ds);

			std::vector<vk::DescriptorSet> descriptorSets(mDescriptorSets.size());
			std::ranges::transform(mDescriptorSets, descriptorSets.begin(), [](const auto& ds) { return **ds; });

			bool isCompute = pipeline.GetShader(vk::ShaderStageFlagBits::eCompute) != nullptr;

			commandBuffer->bindDescriptorSets(
				isCompute ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics,
				**pipeline.GetLayout(),
				0,
				descriptorSets,
				{});
		}
	};

	std::filesystem::path mSourceFile;
	std::string mEntryPoint;
	std::string mProfile;
	std::vector<std::string> mCompileArgs;
	PipelineInfo mPipelineInfo;

	std::filesystem::file_time_type mLastWriteTime;
	std::unordered_map<size_t, size_t> mCachedSpirvHashes;
	std::unordered_map<size_t, std::shared_ptr<ComputePipeline>> mCachedPipelines;

	std::unordered_map<const Pipeline*, ResourceQueue<ParameterData>> mCachedParameters;
};

}
#pragma once


#include "ResourceQueue.hpp"
#include "CommandBuffer.hpp"
#include "ShaderParameterBlock.hpp"
#include "Profiler.hpp"

#include <future>

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
		if (!std::filesystem::exists(mSourceFile))
			throw std::runtime_error("File not found: " + mSourceFile.string());
		auto writeTime = std::filesystem::last_write_time(mSourceFile);
		if (writeTime > mLastWriteTime) {
			mLastWriteTime = writeTime;
			mCachedShaders.clear();
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

		size_t infoDefineHash = infoHash;
		for (const auto&[n,v] : defines)
			infoDefineHash = HashArgs(infoDefineHash, n, v);

		std::shared_ptr<Shader> shader;
		if (auto it = mCachedShaders.find(infoDefineHash); it != mCachedShaders.end())
			shader = it->second;
		else {
			shader = std::make_shared<Shader>(device, mSourceFile, mEntryPoint, mProfile, mCompileArgs, defines);
			mCachedShaders.emplace(infoDefineHash, shader);
		}

		size_t pipelineHash = HashCombine(shader->GetSpirvHash(), infoHash);

		if (auto it = mCachedPipelines.find(pipelineHash); it != mCachedPipelines.end())
			return it->second;

		// TODO: user descriptorSetLayouts
		auto pipeline = std::make_shared<ComputePipeline>(mSourceFile.stem().string() + "/" + mEntryPoint, shader, info ? *info : mPipelineInfo);
		return mCachedPipelines.emplace(pipelineHash, pipeline).first->second;
	}

	inline std::shared_ptr<ComputePipeline> GetPipelineAsync(Device& device, const Defines& defines = {}, const std::optional<PipelineInfo>& info = std::nullopt) {
		auto writeTime = std::filesystem::last_write_time(mSourceFile);
		if (writeTime > mLastWriteTime) {
			mLastWriteTime = writeTime;
			mCachedShaders.clear();
			mShaderCompileJobs.clear();
			mPipelineCompileJobs.clear();
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

		size_t infoDefineHash = infoHash;
		for (const auto&[n,v] : defines)
			infoDefineHash = HashArgs(infoDefineHash, n, v);

		std::shared_ptr<Shader> shader;

		if (auto it = mCachedShaders.find(infoDefineHash); it != mCachedShaders.end()) {
			shader = it->second;
		} else if (auto it = mShaderCompileJobs.find(infoDefineHash); it != mShaderCompileJobs.end()) {
			// shader is compiling
			if (it->second.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready) {
				// compile job completed
				shader = it->second.get();
				mShaderCompileJobs.erase(it);
				mCachedShaders.emplace(infoDefineHash, shader);
			}
		} else {
			// compile the shader asynchronously
			mShaderCompileJobs.emplace(infoDefineHash, std::move(std::async(std::launch::async, [=, this, &device]() {
				return std::make_shared<Shader>(device, mSourceFile, mEntryPoint, mProfile, mCompileArgs, defines);
			})));
		}

		if (!shader)
			return nullptr;

		size_t pipelineHash = HashCombine(shader->GetSpirvHash(), infoHash);

		if (auto it = mCachedPipelines.find(pipelineHash); it != mCachedPipelines.end())
			return it->second;
		else if (auto it = mPipelineCompileJobs.find(pipelineHash); it != mPipelineCompileJobs.end()) {
			// pipeline is compiling
			if (it->second.wait_for(std::chrono::nanoseconds(0)) == std::future_status::ready) {
				// compile job completed
				auto pipeline = it->second.get();
				mPipelineCompileJobs.erase(it);
				mCachedPipelines.emplace(pipelineHash, pipeline);
				return pipeline;
			}
		} else {
			// compile the pipeline asynchronously
			mPipelineCompileJobs.emplace(pipelineHash, std::move(std::async(std::launch::async, [=, this, &device]() {
				return std::make_shared<ComputePipeline>(mSourceFile.stem().string() + "/" + mEntryPoint, shader, info ? *info : mPipelineInfo);
			})));
		}

		return nullptr;
	}

	inline void Dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const ShaderParameterBlock& params, const ComputePipeline& pipeline) {
		ProfilerScope p("ComputePipelineCache::Dispatch");

		commandBuffer.BindPipeline(pipeline);
		const auto data = mCachedParameters[&pipeline].Get(commandBuffer.mDevice);
		data->SetParameters(commandBuffer, pipeline, params);

		// copy push constants and add barriers
		std::vector<std::byte> pushConstants;
		for (const auto& [id, param] : params) {
			if (const auto* v = std::get_if<ConstantParameter>(&param)) {
				auto it = pipeline.GetPushConstants().find(id.first);
				if (it != pipeline.GetPushConstants().end()) {
					if (it->second.mTypeSize != v->size())
						std::cerr << "Warning: Push constant type size mismatch for " << id.first << std::endl;
					size_t s = std::min<size_t>(v->size(), it->second.mTypeSize);
					if (pushConstants.size() < it->second.mOffset + s)
						pushConstants.resize(it->second.mOffset + s);
					std::memcpy(pushConstants.data() + it->second.mOffset, v->data(), s);
				}
				continue;
			}

			auto it = pipeline.GetDescriptors().find(id.first);
			if (it == pipeline.GetDescriptors().end())
				continue;
			const Shader::DescriptorBinding& binding = it->second;

			if        (const auto* v = std::get_if<ImageParameter>(&param)) {
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

		data->Bind(commandBuffer, pipeline);

		if (!pushConstants.empty())
			commandBuffer->pushConstants<std::byte>(**pipeline.GetLayout(), vk::ShaderStageFlagBits::eCompute, 0, vk::ArrayProxy<const std::byte>(pushConstants));
		commandBuffer.Dispatch(pipeline.GetDispatchDim(dim));
	}

	inline void Dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const ShaderParameterBlock& params, const Defines& defines = {}, const std::optional<PipelineInfo>& info = std::nullopt) {
		Dispatch(commandBuffer, dim, params, *GetPipeline(commandBuffer.mDevice, defines, info));
	}

private:
	struct ParameterData {
		std::vector<std::shared_ptr<vk::raii::DescriptorSet>> mDescriptorSets;
		ResourceQueue<std::pair<Buffer::View<std::byte>, Buffer::View<std::byte>>> mCachedUniformBuffers;

		inline void SetParameters(CommandBuffer& commandBuffer, const Pipeline& pipeline, const ShaderParameterBlock& params) {
			ProfilerScope p("ComputePipelineCache::ParameterData::SetParameters");

			// allocate descriptor sets

			if (mDescriptorSets.empty()) {
				ProfilerScope p("Allocate DescriptorSet");

				vk::raii::DescriptorSets sets = nullptr;
				std::vector<vk::DescriptorSetLayout> layouts;
				for (const auto& l : pipeline.GetDescriptorSetLayouts())
					layouts.emplace_back(**l);
				try {
					const std::shared_ptr<vk::raii::DescriptorPool>& descriptorPool = pipeline.mDevice.GetDescriptorPool();
					sets = vk::raii::DescriptorSets(*pipeline.mDevice, vk::DescriptorSetAllocateInfo(**descriptorPool, layouts));
				} catch(vk::OutOfPoolMemoryError e) {
					const std::shared_ptr<vk::raii::DescriptorPool>& descriptorPool = pipeline.mDevice.AllocateDescriptorPool();
					sets = vk::raii::DescriptorSets(*pipeline.mDevice, vk::DescriptorSetAllocateInfo(**descriptorPool, layouts));
				}

				mDescriptorSets.resize(sets.size());
				for (uint32_t i = 0; i < sets.size(); i++) {
					mDescriptorSets[i] = std::make_shared<vk::raii::DescriptorSet>(std::move(sets[i]));
					//std::cout << "Creating descriptor sets for " << pipeline.GetName() << std::endl;
					pipeline.mDevice.SetDebugName(**mDescriptorSets[i], "Pipeline DescriptorSet[" + std::to_string(i) + "]");
				}
			}

			// write descriptors

			union DescriptorInfo {
				vk::DescriptorBufferInfo buffer;
				vk::DescriptorImageInfo image;
				vk::WriteDescriptorSetAccelerationStructureKHR accelerationStructure;
			};

			std::vector<DescriptorInfo> descriptorInfos;
			std::vector<vk::WriteDescriptorSet> writes;
			descriptorInfos.reserve(params.size());
			writes.reserve(params.size());

			std::unordered_set<std::string> unboundDescriptors;
			for (const auto&[id, d] : pipeline.GetDescriptors()) unboundDescriptors.emplace(id);
			for (const auto&[id, c] : pipeline.GetUniforms()) unboundDescriptors.emplace(id);
			for (const auto&[id, c] : pipeline.GetPushConstants()) unboundDescriptors.emplace(id);
			for (const auto&[id, s] : pipeline.GetInfo().mImmutableSamplers)
				unboundDescriptors.erase(id);
			for (const auto&[id, s] : pipeline.GetInfo().mBindingFlags)
				if (s & vk::DescriptorBindingFlagBits::ePartiallyBound)
					unboundDescriptors.erase(id);

			std::unordered_map<std::string, std::vector<std::byte>> uniformData;
			for (const auto&[name,size] : pipeline.GetUniformBufferSizes()) {
				uniformData[name].resize(size);
				unboundDescriptors.erase(name);
			}

			auto msgPrefix = [&]() -> std::ostream& { return std::cerr << "[" << pipeline.GetName() << "] "; };

			for (const auto& [id_index, param] : params) {
				const auto& [name, arrayIndex] = id_index;

				// check if param is a constant/uniform

				if (const auto* v = std::get_if<ConstantParameter>(&param)) {
					const auto& uniforms = pipeline.GetUniforms();
					if (auto it = uniforms.find(name); it != uniforms.end()) {
						if (it->second.mTypeSize != v->size())
							msgPrefix() << "Warning: Writing type size mismatch at " << name << "[" << arrayIndex << "]" << std::endl;

						auto& u = uniformData.at(it->second.mParentDescriptor);
						std::memcpy(u.data() + it->second.mOffset, v->data(), std::min<size_t>(v->size(), it->second.mTypeSize));

						unboundDescriptors.erase(name);
					} else if (pipeline.GetPushConstants().contains(name))
						unboundDescriptors.erase(name);
					continue;
				}

				// check if param is a descriptor

				auto it = pipeline.GetDescriptors().find(name);
				if (it == pipeline.GetDescriptors().end())
					continue;

				const Shader::DescriptorBinding& binding = it->second;

				// write descriptor

				vk::WriteDescriptorSet& w = writes.emplace_back(vk::WriteDescriptorSet(**mDescriptorSets[binding.mSet], binding.mBinding, arrayIndex, 1, binding.mDescriptorType));
				DescriptorInfo& info = descriptorInfos.emplace_back(DescriptorInfo{});

				if        (const auto* v = std::get_if<BufferParameter>(&param)) {
					const auto& buffer = *v;
					if (!buffer) continue;

					commandBuffer.HoldResource(buffer);
					info.buffer = vk::DescriptorBufferInfo(**buffer.GetBuffer(), buffer.Offset(), buffer.SizeBytes());
					w.setBufferInfo(info.buffer);
				} else if (const auto* v = std::get_if<ImageParameter>(&param)) {
					const auto& [image, layout, accessFlags, sampler] = *v;
					if (!image && !sampler) continue;
					if (image)   commandBuffer.HoldResource(image);
					if (sampler) commandBuffer.HoldResource(sampler);
					info.image = vk::DescriptorImageInfo(sampler ? **sampler : nullptr, image ? *image : nullptr, layout);
					w.setImageInfo(info.image);
				} else if (const auto* v = std::get_if<AccelerationStructureParameter>(&param)) {
					if (!*v) continue;
					if (binding.mDescriptorType != vk::DescriptorType::eAccelerationStructureKHR)
						msgPrefix() << "Warning: Invalid descriptor type " << vk::to_string(binding.mDescriptorType) << " at " << name << "[" << arrayIndex << "]" << std::endl;

					commandBuffer.HoldResource(*v);
					info.accelerationStructure = vk::WriteDescriptorSetAccelerationStructureKHR(***v);
					w.descriptorCount = info.accelerationStructure.accelerationStructureCount;
					w.pNext = &info;
				}

				unboundDescriptors.erase(name);
			}


			if (!unboundDescriptors.empty()) {
				msgPrefix() << "Warning: Missing parameter:\t{ ";
				for (const auto& name : unboundDescriptors)
					std::cout << name << ", ";
				std::cout << "}" << std::endl;
			}

			// uniform buffers
			if (!uniformData.empty()) {
				ProfilerScope p("Upload uniforms");
				for (const auto&[name,data] : uniformData) {
					auto&[hostBuf, buf] = *mCachedUniformBuffers.Get(commandBuffer.mDevice);
					if (!hostBuf || hostBuf.SizeBytes() < data.size()) {
						hostBuf = std::make_shared<Buffer>(commandBuffer.mDevice,
							"Pipeline Uniform Buffer (Host)", data.size(),
							vk::BufferUsageFlagBits::eTransferSrc,
							vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
							VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT|VMA_ALLOCATION_CREATE_MAPPED_BIT|VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
						buf = std::make_shared<Buffer>(commandBuffer.mDevice,
							"Pipeline Uniform Buffer", data.size(),
							vk::BufferUsageFlagBits::eUniformBuffer|vk::BufferUsageFlagBits::eTransferDst,
							vk::MemoryPropertyFlagBits::eDeviceLocal,
							VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT);
					}

					std::memcpy(hostBuf.data(), data.data(), data.size());
					commandBuffer.Copy(hostBuf, buf);

					buf.SetState(vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);

					commandBuffer.HoldResource(hostBuf);
					commandBuffer.HoldResource(buf);
					commandBuffer.Barrier(buf,
						pipeline.GetShader(vk::ShaderStageFlagBits::eCompute) ? vk::PipelineStageFlagBits::eComputeShader : vk::PipelineStageFlagBits::eVertexShader,
						vk::AccessFlagBits::eUniformRead);

					vk::WriteDescriptorSet& w = writes.emplace_back(vk::WriteDescriptorSet(**mDescriptorSets[pipeline.GetDescriptors().at(name).mSet], 0, 0, 1, vk::DescriptorType::eUniformBuffer));
					DescriptorInfo& info = descriptorInfos.emplace_back(DescriptorInfo{});
					info.buffer = vk::DescriptorBufferInfo(**buf.GetBuffer(), buf.Offset(), buf.SizeBytes());
					w.setBufferInfo(info.buffer);
				}
			}

			if (!writes.empty()) {
				ProfilerScope p("updateDescriptorSets");
				pipeline.mDevice->updateDescriptorSets(writes, {});
			}
		}

		inline void Bind(CommandBuffer& commandBuffer, const Pipeline& pipeline) const {
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
	// hash(pipeline info, defines) -> shader
	std::unordered_map<size_t, std::shared_ptr<Shader>> mCachedShaders;
	// hash(pipeline info, spir-v) -> pipeline
	std::unordered_map<size_t, std::shared_ptr<ComputePipeline>> mCachedPipelines;

	std::unordered_map<const Pipeline*, ResourceQueue<ParameterData>> mCachedParameters;

	std::unordered_map<size_t, std::future<std::shared_ptr<Shader>>> mShaderCompileJobs;
	std::unordered_map<size_t, std::future<std::shared_ptr<ComputePipeline>>> mPipelineCompileJobs;
};

}
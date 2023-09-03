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

	inline void Dispatch(CommandBuffer& commandBuffer, const vk::Extent3D& dim, const ShaderParameterBlock& params, const Defines& defines = {}, const std::optional<PipelineInfo>& info = std::nullopt) {
		ProfilerScope p("ComputePipelineCache::Dispatch");

		const ComputePipeline& pipeline = *GetPipeline(commandBuffer.mDevice, defines, info);

		// add barriers as needed
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
		if (!pushConstants.empty())
			commandBuffer->pushConstants<std::byte>(**pipeline.GetLayout(), vk::ShaderStageFlagBits::eCompute, 0, vk::ArrayProxy<const std::byte>(pushConstants));
		const auto data = mCachedParameters[&pipeline].Get(commandBuffer.mDevice);
		data->SetParameters(commandBuffer, pipeline, params);
		data->Bind(commandBuffer, pipeline);
		commandBuffer.Dispatch(pipeline.GetDispatchDim(dim));
	}

private:
	struct ParameterData {
		std::vector<std::shared_ptr<vk::raii::DescriptorSet>> mDescriptorSets;
		std::vector<std::pair<Buffer::View<std::byte>, Buffer::View<std::byte>>> mUniformBuffers;

		inline void SetParameters(CommandBuffer& commandBuffer, const Pipeline& pipeline, const ShaderParameterBlock& params) {
			ProfilerScope p("ComputePipelineCache::ParameterData::SetParameters");

			// allocate descriptor sets

			if (mDescriptorSets.empty()) {
				ProfilerScope p("Allocate DescriptorSet");

				vk::raii::DescriptorSets sets = nullptr;
				std::vector<vk::DescriptorSetLayout> layouts = pipeline.GetDescriptorSetLayouts() | std::views::transform([](auto& l){ return **l; }) | std::ranges::to<std::vector<vk::DescriptorSetLayout>>();
				try {
					const std::shared_ptr<vk::raii::DescriptorPool>& descriptorPool = pipeline.mDevice.GetDescriptorPool();
					sets = vk::raii::DescriptorSets(*pipeline.mDevice, vk::DescriptorSetAllocateInfo(**descriptorPool, layouts));
				} catch(vk::OutOfPoolMemoryError e) {
					const std::shared_ptr<vk::raii::DescriptorPool>& descriptorPool = pipeline.mDevice.AllocateDescriptorPool();
					sets = vk::raii::DescriptorSets(*pipeline.mDevice, vk::DescriptorSetAllocateInfo(**descriptorPool, layouts));
				}

				mDescriptorSets.resize(sets.size());
				for (uint32_t i = 0; i < sets.size(); i++) {
					mDescriptorSets[i] = std::move( std::make_shared<vk::raii::DescriptorSet>(std::move(sets[i])) );
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

			std::unordered_set<std::string> unboundDescriptors = pipeline.GetDescriptors() | std::views::keys | std::ranges::to<std::unordered_set<std::string>>();
			for (const auto&[id, s] : pipeline.GetInfo().mImmutableSamplers)
				unboundDescriptors.erase(id);
			for (const auto&[id, s] : pipeline.GetInfo().mBindingFlags)
				if (s & vk::DescriptorBindingFlagBits::ePartiallyBound)
					unboundDescriptors.erase(id);

			std::vector<std::vector<std::byte>> uniformData(pipeline.GetUniformBufferSizes().size());
			for (size_t i = 0; i < pipeline.GetUniformBufferSizes().size(); i++) {
				uniformData[i].resize(pipeline.GetUniformBufferSizes()[i]);
				unboundDescriptors.erase("$Uniforms" + std::to_string(i));
			}

			auto msgPrefix = [&]() -> std::ostream& { return std::cerr << "[" << pipeline.GetName() << "] "; };

			for (const auto& [id_index, param] : params) {
				const auto& [name, arrayIndex] = id_index;

				// check if param is a constant/uniform

				if (const auto* v = std::get_if<ConstantParameter>(&param)) {
					const auto& uniforms = pipeline.GetShader(vk::ShaderStageFlagBits::eCompute)->GetUniforms();
					if (auto it = uniforms.find(name); it != uniforms.end()) {
						if (it->second.mTypeSize != v->size())
							msgPrefix() << "Warning: Writing type size mismatch at " << name << "[" << arrayIndex << "]" << std::endl;

						auto& u = uniformData[it->second.mSetIndex];
						std::memcpy(u.data() + it->second.mOffset, v->data(), std::min<size_t>(v->size(), it->second.mTypeSize));
					}
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
				msgPrefix() << "Warning: Missing descriptors:\t{ ";
				for (const auto& name : unboundDescriptors)
					std::cout << name << ", ";
				std::cout << "}" << std::endl;
			}

			// uniform buffers
			if (!uniformData.empty()) {
				ProfilerScope p("Upload uniforms");
				if (mUniformBuffers.size() < uniformData.size())
					mUniformBuffers.resize(uniformData.size());
				for (size_t i = 0; i < uniformData.size(); i++) {
					auto&[hostBuf, buf] = mUniformBuffers[i];
					if (!hostBuf || hostBuf.SizeBytes() < uniformData[i].size()) {
						hostBuf = std::make_shared<Buffer>(commandBuffer.mDevice, "Pipeline Uniform Buffer (Host)", uniformData[i].size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
						buf     = std::make_shared<Buffer>(commandBuffer.mDevice, "Pipeline Uniform Buffer"       , uniformData[i].size(), vk::BufferUsageFlagBits::eUniformBuffer|vk::BufferUsageFlagBits::eTransferDst);
					}
					commandBuffer.HoldResource(hostBuf);
					commandBuffer.HoldResource(buf);

					std::memcpy(hostBuf.data(), uniformData[i].data(), uniformData[i].size());
					commandBuffer.Copy(hostBuf, buf);

					vk::WriteDescriptorSet& w = writes.emplace_back(vk::WriteDescriptorSet(**mDescriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer));
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
};

}
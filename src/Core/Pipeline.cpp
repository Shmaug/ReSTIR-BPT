#include "Pipeline.hpp"
#include "CommandBuffer.hpp"

#include <map>

namespace ptvk {

Pipeline::Pipeline(Device& device, const std::string& name, const ShaderStageMap& shaders, const PipelineInfo& info, const std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts)
	: mDevice(device), mName(name), mShaders(shaders), mPipeline(nullptr), mInfo(info), mDescriptorSetLayouts(descriptorSetLayouts) {

	// gather descriptorset bindings from shaders

	std::vector<std::map<uint32_t, std::tuple<vk::DescriptorSetLayoutBinding, std::optional<vk::DescriptorBindingFlags>, std::vector<vk::Sampler>>>> bindings;

	uint32_t pushConstantRangeBegin = std::numeric_limits<uint32_t>::max();
	uint32_t pushConstantRangeEnd = 0;
	vk::ShaderStageFlags pushConstantStages = vk::ShaderStageFlags{0};

	for (const auto&[stage, shader] : mShaders) {
		// push constant range

		if (!shader->GetPushConstants().empty()) {
			pushConstantStages |= stage;
			for (const auto& [id, p] : shader->GetPushConstants()) {
				pushConstantRangeBegin = std::min(pushConstantRangeBegin, p.mOffset);
				pushConstantRangeEnd   = std::max(pushConstantRangeEnd  , p.mOffset + p.mTypeSize);
				if (auto it = mPushConstants.find(id); it != mPushConstants.end()) {
					if (it->second.mOffset != p.mOffset || it->second.mTypeSize != p.mTypeSize)
						std::cerr << "Warning: Pipeline push constant " << id << " is specified with different offsets/sizes between shaders" << std::endl;
				} else
					mPushConstants.emplace(id, p);
			}
		}

		// uniforms
		for (const auto&[name,size] :  shader->GetUniformBufferSizes()) {
			if (!mUniformBufferSizes.contains(name))
				mUniformBufferSizes.emplace(name, 0);
			mUniformBufferSizes[name] = std::max<size_t>(mUniformBufferSizes[name], size);
		}

		for (const auto& [id, b] : shader->GetUniforms()) {
			if (auto it = mUniformMap.find(id); it != mUniformMap.end()) {
				if (it->second.mOffset != b.mOffset || it->second.mTypeSize != b.mTypeSize || it->second.mParentDescriptor != b.mParentDescriptor)
					std::cerr << "Warning: Pipeline uniform " << id << " is specified with different offsets/sizes/sets between shaders" << std::endl;
			} else
				mUniformMap.emplace(id, b);
		}

		// descriptors

		for (const auto&[id, binding] : shader->GetDescriptors()) {
			mDescriptorMap[id] = binding;

			// compute total array size
			uint32_t descriptorCount = 1;
			for (const uint32_t v : binding.mArraySize)
				descriptorCount *= v;

			// get binding flags from mInfo
			std::optional<vk::DescriptorBindingFlags> flags;
			if (auto b_it = mInfo.mBindingFlags.find(id); b_it != mInfo.mBindingFlags.end())
				flags = b_it->second;

			// get immutable samplers from mInfo
			std::vector<vk::Sampler> samplers;
			if (auto s_it = mInfo.mImmutableSamplers.find(id); s_it != mInfo.mImmutableSamplers.end()) {
				samplers.resize(s_it->second.size());
				std::ranges::transform(s_it->second, samplers.begin(), [](const std::shared_ptr<vk::raii::Sampler>& s){ return **s; });
			}

			// increase set count if needed
			if (binding.mSet >= bindings.size())
				bindings.resize(binding.mSet + 1);

			// copy bindings

			auto& setBindings = bindings[binding.mSet];

			auto it = setBindings.find(binding.mBinding);
			if (it == setBindings.end())
				it = setBindings.emplace(binding.mBinding,
					std::tuple{
						vk::DescriptorSetLayoutBinding(
							binding.mBinding,
							binding.mDescriptorType,
							descriptorCount,
							shader->GetStage(), {}),
						flags,
						samplers }).first;
			else {
				auto&[setLayoutBinding, flags, samplers] = it->second;

				if (setLayoutBinding.descriptorType != binding.mDescriptorType)
					throw std::logic_error("Shader modules contain descriptors of different types at the same binding");
				if (setLayoutBinding.descriptorCount != descriptorCount)
					throw std::logic_error("Shader modules contain descriptors with different counts at the same binding");

				setLayoutBinding.stageFlags |= shader->GetStage();
			}
		}
	}

	// create DescriptorSetLayouts

	mDescriptorSetLayouts.resize(bindings.size());
	for (uint32_t i = 0; i < bindings.size(); i++) {
		if (mDescriptorSetLayouts[i]) continue;
		std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
		std::vector<vk::DescriptorBindingFlags> bindingFlags;
		bool hasFlags = false;
		for (const auto&[bindingIndex, binding_] : bindings[i]) {
			const auto&[binding, flag, samplers] = binding_;
			if (flag) hasFlags = true;
			bindingFlags.emplace_back(flag ? *flag : vk::DescriptorBindingFlags{});

			auto& b = layoutBindings.emplace_back(binding);
			if (!samplers.empty())
				b.setImmutableSamplers(samplers);
		}

		vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlagsInfo(bindingFlags);
		mDescriptorSetLayouts[i] = std::make_shared<vk::raii::DescriptorSetLayout>(*mDevice, vk::DescriptorSetLayoutCreateInfo(mInfo.mDescriptorSetLayoutFlags, layoutBindings, hasFlags ? &bindingFlagsInfo : nullptr));
		mDevice.SetDebugName(**mDescriptorSetLayouts[i], name + " DescriptorSetLayout[" + std::to_string(i) + "]");
	}

	// create pipelinelayout from descriptors and pushconstants

	std::vector<vk::PushConstantRange> pushConstantRanges;
	if (pushConstantStages != vk::ShaderStageFlags{0})
		pushConstantRanges.emplace_back(pushConstantStages, pushConstantRangeBegin, pushConstantRangeEnd - pushConstantRangeBegin);

	std::vector<vk::DescriptorSetLayout> vklayouts = mDescriptorSetLayouts | std::views::transform([](auto ds) { return **ds; }) | std::ranges::to<std::vector<vk::DescriptorSetLayout>>();
	mLayout = std::make_shared<vk::raii::PipelineLayout>(*mDevice, vk::PipelineLayoutCreateInfo(mInfo.mLayoutFlags, vklayouts, pushConstantRanges));
	mDevice.SetDebugName(**mLayout, name + " Layout");
}

GraphicsPipeline::GraphicsPipeline(const std::string& name, const ShaderStageMap& shaders, const GraphicsPipelineInfo& info, const std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts)
	: Pipeline(shaders.begin()->second->mDevice, name, shaders, info, descriptorSetLayouts) {

	// Pipeline constructor creates mLayout, mDescriptorSetLayouts, and mDescriptorMap

	// create pipeline

	std::vector<vk::PipelineShaderStageCreateInfo> stages;
	for (const auto&[stage, shader] : shaders)
		stages.emplace_back(vk::PipelineShaderStageCreateInfo(mInfo.mStageLayoutFlags, shader->GetStage(), ***shader, "main"));

	vk::PipelineColorBlendStateCreateInfo colorBlendState;
	if (info.mColorBlendState)
		colorBlendState = vk::PipelineColorBlendStateCreateInfo(
			{},
			info.mColorBlendState->mLogicOpEnable,
			info.mColorBlendState->mLogicOp,
			info.mColorBlendState->mAttachments,
			info.mColorBlendState->mBlendConstants);

	vk::PipelineDynamicStateCreateInfo dynamicState({}, info.mDynamicStates);

	vk::PipelineRenderingCreateInfo dynamicRenderingState;
	if (info.mDynamicRenderingState)
		dynamicRenderingState = vk::PipelineRenderingCreateInfo(
			info.mDynamicRenderingState->mViewMask,
			info.mDynamicRenderingState->mColorFormats,
			info.mDynamicRenderingState->mDepthFormat,
			info.mDynamicRenderingState->mStencilFormat);

	vk::PipelineViewportStateCreateInfo viewportState({}, info.mViewports, info.mScissors);

	mPipeline = vk::raii::Pipeline(*mDevice, mDevice.GetPipelineCache(), vk::GraphicsPipelineCreateInfo(
		mInfo.mFlags,
		stages,
		info.mVertexInputState.has_value()   ? &info.mVertexInputState.value() : nullptr,
		info.mInputAssemblyState.has_value() ? &info.mInputAssemblyState.value() : nullptr,
		info.mTessellationState.has_value()  ? &info.mTessellationState.value() : nullptr,
		&viewportState,
		info.mRasterizationState.has_value() ? &info.mRasterizationState.value() : nullptr,
		info.mMultisampleState.has_value()   ? &info.mMultisampleState.value() : nullptr,
		info.mDepthStencilState.has_value()  ? &info.mDepthStencilState.value() : nullptr,
		info.mColorBlendState.has_value()    ? &colorBlendState : nullptr,
		&dynamicState,
		**mLayout,
		info.mRenderPass,
		info.mSubpassIndex, {}, {},
		info.mDynamicRenderingState.has_value() ? &dynamicRenderingState : nullptr));
	mDevice.SetDebugName(*mPipeline, name);
}

ComputePipeline::ComputePipeline(const std::string& name, const std::shared_ptr<Shader>& shader_, const PipelineInfo& info, const std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts)
	: Pipeline(shader_->mDevice, name, { { shader_->GetStage(), shader_ } }, info, descriptorSetLayouts) {

	// Pipeline constructor creates mLayout, mDescriptorSetLayouts, and mDescriptorMap

	mPipeline = vk::raii::Pipeline(*mDevice, mDevice.GetPipelineCache(), vk::ComputePipelineCreateInfo(
		mInfo.mFlags,
		vk::PipelineShaderStageCreateInfo(mInfo.mStageLayoutFlags, vk::ShaderStageFlagBits::eCompute, ***shader_, "main"),
		**mLayout));
	mDevice.SetDebugName(*mPipeline, name);
}

}
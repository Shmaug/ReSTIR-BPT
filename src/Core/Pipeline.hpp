#pragma once

#include <future>
#include <shared_mutex>

#include "Shader.hpp"

namespace ptvk {

struct PipelineInfo {
	vk::PipelineShaderStageCreateFlags mStageLayoutFlags;
	vk::PipelineLayoutCreateFlags mLayoutFlags;
	vk::PipelineCreateFlags mFlags;
	vk::DescriptorSetLayoutCreateFlags mDescriptorSetLayoutFlags;
	std::unordered_map<std::string, std::vector<std::shared_ptr<vk::raii::Sampler>>> mImmutableSamplers;
	std::unordered_map<std::string, vk::DescriptorBindingFlags> mBindingFlags;
};

class Pipeline {
public:
	Device& mDevice;

	using ShaderStageMap = std::unordered_map<vk::ShaderStageFlagBits, std::shared_ptr<Shader>>;

	Pipeline(Device& device, const std::string& name, const ShaderStageMap& shaders, const PipelineInfo& info = {}, const std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});

	inline       vk::raii::Pipeline& operator*()        { return mPipeline; }
	inline const vk::raii::Pipeline& operator*() const  { return mPipeline; }
	inline       vk::raii::Pipeline* operator->()       { return &mPipeline; }
	inline const vk::raii::Pipeline* operator->() const { return &mPipeline; }

	inline std::string GetName() const { return mName; }

	inline const std::shared_ptr<vk::raii::PipelineLayout>& GetLayout() const { return mLayout; }
	inline const std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& GetDescriptorSetLayouts() const { return mDescriptorSetLayouts; }
	inline const PipelineInfo& GetInfo() const { return mInfo; }
	inline const std::unordered_map<std::string, Shader::DescriptorBinding>& GetDescriptors() const { return mDescriptorMap; }
	inline const std::unordered_map<std::string, Shader::ConstantBinding>& GetUniforms() const { return mUniformMap; }
	inline const std::vector<vk::DeviceSize>& GetUniformBufferSizes() const { return mUniformBufferSizes; }
	inline const std::unordered_map<std::string, Shader::ConstantBinding>& GetPushConstants() const { return mPushConstants; }
	inline std::shared_ptr<Shader> GetShader(vk::ShaderStageFlagBits stage) const {
		if (auto it = mShaders.find(stage); it != mShaders.end())
			return it->second;
		return nullptr;
	}

protected:
	vk::raii::Pipeline mPipeline;
	std::string mName;
	PipelineInfo mInfo;
	std::shared_ptr<vk::raii::PipelineLayout> mLayout;
	std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>> mDescriptorSetLayouts;
	std::unordered_map<std::string, Shader::DescriptorBinding> mDescriptorMap;
	std::unordered_map<std::string, Shader::ConstantBinding> mUniformMap;
	std::vector<vk::DeviceSize> mUniformBufferSizes;
	std::unordered_map<std::string, Shader::ConstantBinding> mPushConstants;
	ShaderStageMap mShaders;
};

struct ColorBlendState {
	vk::PipelineColorBlendStateCreateFlags flags = {};
	bool mLogicOpEnable = false;
	vk::LogicOp mLogicOp = vk::LogicOp::eClear;
	std::vector<vk::PipelineColorBlendAttachmentState> mAttachments;
	std::array<float,4> mBlendConstants = { 1, 1, 1, 1 };
};
struct DynamicRenderingState {
	uint32_t mViewMask = 0;
	std::vector<vk::Format> mColorFormats;
	vk::Format mDepthFormat = vk::Format::eUndefined;
	vk::Format mStencilFormat = vk::Format::eUndefined;
};
struct GraphicsPipelineInfo : public PipelineInfo {
	std::optional<vk::PipelineVertexInputStateCreateInfo>   mVertexInputState;
	std::optional<vk::PipelineInputAssemblyStateCreateInfo> mInputAssemblyState;
	std::optional<vk::PipelineTessellationStateCreateInfo>  mTessellationState;
	std::optional<vk::PipelineRasterizationStateCreateInfo> mRasterizationState;
	std::optional<vk::PipelineMultisampleStateCreateInfo>   mMultisampleState;
	std::optional<vk::PipelineDepthStencilStateCreateInfo>  mDepthStencilState;
	std::vector<vk::Viewport> mViewports;
	std::vector<vk::Rect2D> mScissors;
	std::optional<ColorBlendState> mColorBlendState;
	std::vector<vk::DynamicState> mDynamicStates;
	std::optional<DynamicRenderingState> mDynamicRenderingState;
	vk::RenderPass mRenderPass;
	uint32_t mSubpassIndex;
};

class GraphicsPipeline : public Pipeline {
public:
	GraphicsPipeline(const std::string& name, const ShaderStageMap& shaders, const GraphicsPipelineInfo& info = {}, const std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});
};

class ComputePipeline : public Pipeline {
public:
	ComputePipeline(const std::string& name, const std::shared_ptr<Shader>& shader, const PipelineInfo& info = {}, const std::vector<std::shared_ptr<vk::raii::DescriptorSetLayout>>& descriptorSetLayouts = {});

	inline const std::shared_ptr<Shader>& GetShader() const { return mShaders.at(vk::ShaderStageFlagBits::eCompute); }

	inline vk::Extent3D GetDispatchDim(const vk::Extent3D& extent) const {
		const vk::Extent3D& s = GetShader()->WorkgroupSize();
		return {
			(extent.width  + s.width - 1)  / s.width,
			(extent.height + s.height - 1) / s.height,
			(extent.depth  + s.depth - 1)  / s.depth };
	}
};

}
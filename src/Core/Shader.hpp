#pragma once

#include "Device.hpp"

namespace ptvk {

using Defines = std::unordered_map<std::string, std::string>;

class Shader {
public:
	Device& mDevice;

	struct DescriptorBinding {
		uint32_t mSet;
		uint32_t mBinding;
		vk::DescriptorType mDescriptorType;
		std::vector<uint32_t> mArraySize;
		uint32_t mInputAttachmentIndex;
		bool mWritable;
	};
	struct ConstantBinding {
		uint32_t mOffset;
		uint32_t mTypeSize;
		std::string mParentDescriptor;
	};
	struct Variable {
		uint32_t mLocation;
		vk::Format mFormat;
		std::string mSemantic;
		uint32_t mSemanticIndex;
	};

	Shader(Device& device, const std::filesystem::path& sourceFile, const std::string& entryPoint = "main", const std::string& profile = "sm_6_7", const std::vector<std::string>& compileArgs = {}, const Defines& defines = {});
	Shader(Shader&&) = default;
	Shader& operator=(Shader&&) = default;

	inline       vk::raii::ShaderModule& operator*()        { return mModule; }
	inline const vk::raii::ShaderModule& operator*() const  { return mModule; }
	inline       vk::raii::ShaderModule* operator->()       { return &mModule; }
	inline const vk::raii::ShaderModule* operator->() const { return &mModule; }

	inline const vk::ShaderStageFlagBits& GetStage() const { return mStage; }
	inline const std::unordered_map<std::string, DescriptorBinding>& GetDescriptors() const { return mDescriptorMap; }
	inline const std::unordered_map<std::string, ConstantBinding>& GetUniforms() const { return mUniformMap; }
	inline const std::unordered_map<std::string, vk::DeviceSize>& GetUniformBufferSizes() const { return mUniformBufferSizes; }
	inline const std::unordered_map<std::string, ConstantBinding>& GetPushConstants() const { return mPushConstants; }
	inline const std::unordered_map<std::string, Variable>& GetInputVariables() const { return mInputVariables; }
	inline const std::unordered_map<std::string, Variable>& GetOutputVariables() const { return mOutputVariables; }
	inline const vk::Extent3D& WorkgroupSize() const { return mWorkgroupSize; }

	inline size_t GetSpirvHash() const { return mSpirvHash; }

private:
	vk::raii::ShaderModule mModule;
	size_t mSpirvHash;

	vk::ShaderStageFlagBits mStage;
	std::unordered_map<std::string, DescriptorBinding> mDescriptorMap;
	std::unordered_map<std::string, ConstantBinding> mUniformMap;
	std::unordered_map<std::string, vk::DeviceSize> mUniformBufferSizes;
	std::unordered_map<std::string, ConstantBinding> mPushConstants;
	std::unordered_map<std::string, Variable> mInputVariables;
	std::unordered_map<std::string, Variable> mOutputVariables;
	vk::Extent3D mWorkgroupSize;
};

}

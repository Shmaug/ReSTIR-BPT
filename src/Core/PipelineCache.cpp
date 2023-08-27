#include "PipelineCache.hpp"
#include "CommandBuffer.hpp"

namespace ptvk {

void ComputePipelineCache::ParameterData::SetParameters(CommandBuffer& commandBuffer, const Pipeline& pipeline, const ShaderParameterBlock& params) {
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
					std::cerr << "Warning: Writing type size mismatch at " << name << "[" << arrayIndex << "]" << std::endl;

				auto& u = uniformData[it->second.mSetIndex];
				size_t end = it->second.mOffset + it->second.mTypeSize;
				std::memcpy(u.data() + it->second.mOffset, v->data(), std::min<size_t>(v->size(), it->second.mTypeSize));
			}
			continue;
		}

		// check if param is a descriptor

		auto it = pipeline.GetDescriptors().find(name);
		if (it == pipeline.GetDescriptors().end())
			continue;

		unboundDescriptors.erase(name);

		const Shader::DescriptorBinding& binding = it->second;

		// write descriptor

		vk::WriteDescriptorSet& w = writes.emplace_back(vk::WriteDescriptorSet(**mDescriptorSets[binding.mSet], binding.mBinding, arrayIndex, 1, binding.mDescriptorType));
		DescriptorInfo& info = descriptorInfos.emplace_back(DescriptorInfo{});

		if        (const auto* v = std::get_if<BufferParameter>(&param)) {
			const auto& buffer = *v;
			if (!buffer) msgPrefix() << "Warning: Writing null buffer at " << name << "[" << arrayIndex << "]" << std::endl;

			commandBuffer.HoldResource(buffer);
			info.buffer = vk::DescriptorBufferInfo(**buffer.GetBuffer(), buffer.Offset(), buffer.SizeBytes());
			w.setBufferInfo(info.buffer);
		} else if (const auto* v = std::get_if<ImageParameter>(&param)) {
			const auto& [image, layout, accessFlags, sampler] = *v;
			switch (binding.mDescriptorType) {
			case vk::DescriptorType::eSampler:
				if (!sampler) msgPrefix() << "Warning: Writing null sampler at " << name << "[" << arrayIndex << "]" << std::endl;
				break;
			case vk::DescriptorType::eCombinedImageSampler:
				if (!sampler) msgPrefix() << "Warning: Writing null sampler at " << name << "[" << arrayIndex << "]" << std::endl;
			case vk::DescriptorType::eSampledImage:
			case vk::DescriptorType::eStorageImage:
				if (!image) msgPrefix() << "Warning: Writing null image at " << name << "[" << arrayIndex << "]" << std::endl;
				break;
			default:
				msgPrefix() << "Warning: Invalid descriptor type " << vk::to_string(binding.mDescriptorType) << " at " << name << "[" << arrayIndex << "]" << std::endl;
				break;
			}

			if (image) commandBuffer.HoldResource(image);
			if (sampler) commandBuffer.HoldResource(sampler);
			info.image = vk::DescriptorImageInfo(sampler ? **sampler : nullptr, image ? *image : nullptr, layout);
			w.setImageInfo(info.image);
		} else if (const auto* v = std::get_if<AccelerationStructureParameter>(&param)) {
			if (!*v) msgPrefix() << "Warning: Writing null acceleration structure at " << name << "[" << arrayIndex << "]" << std::endl;
			if (binding.mDescriptorType != vk::DescriptorType::eAccelerationStructureKHR)
				msgPrefix() << "Warning: Invalid descriptor type " << vk::to_string(binding.mDescriptorType) << " at " << name << "[" << arrayIndex << "]" << std::endl;

			commandBuffer.HoldResource(*v);
			info.accelerationStructure = vk::WriteDescriptorSetAccelerationStructureKHR(***v);
			w.descriptorCount = info.accelerationStructure.accelerationStructureCount;
			w.pNext = &info;
		}
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

}
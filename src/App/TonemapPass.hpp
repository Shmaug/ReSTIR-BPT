#pragma once

#include <Core/PipelineCache.hpp>
#include <Common/Enums.h>

namespace ptvk {

class TonemapPass {
private:
	ComputePipelineCache mTonemapPipeline;
	ComputePipelineCache mMaxReducePipeline;

	Buffer::View<uint4> mMaxBuf;

	float mExposure = 0;
	bool mGammaCorrect = true;
	TonemapMode mMode = TonemapMode::eRaw;

public:
	inline TonemapPass(Device& device) {
		mExposure = 0.f;

		if (auto arg = device.mInstance.GetOption("exposure"))
			mExposure = atof(arg->c_str());

		const std::filesystem::path shaderPath = *device.mInstance.GetOption("shader-kernel-path");
		mMaxReducePipeline = ComputePipelineCache(shaderPath / "Kernels/Tonemap.slang", "MaxReduce");
		mTonemapPipeline   = ComputePipelineCache(shaderPath / "Kernels/Tonemap.slang", "Tonemap");

		mMaxBuf = std::make_shared<Buffer>(device, "Tonemap max", sizeof(uint4), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
	}

	inline void OnInspectorGui() {
		Gui::EnumDropdown<TonemapMode>("Mode", mMode, TonemapModeStrings);
		ImGui::PushItemWidth(40);
		ImGui::DragFloat("Exposure", &mExposure, .1f, -10, 10);
		ImGui::PopItemWidth();
		ImGui::Checkbox("Gamma correct", &mGammaCorrect);
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& input) {
		ProfilerScope ps("TonemapPass::Render", &commandBuffer);

		Defines defines;
		defines.emplace("gMode", std::to_string((int)mMode));
		if (mGammaCorrect) defines.emplace("gGammaCorrection", "true");

		const vk::Extent3D extent = input.GetExtent();

		// get maximum value in image

		commandBuffer.Fill(mMaxBuf, 0);

		mMaxReducePipeline.Dispatch(commandBuffer, extent,
			ShaderParameterBlock()
				.SetImage("gImage", input, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead)
				.SetBuffer("gMax", mMaxBuf)
			, defines);

		// tonemap

		mTonemapPipeline.Dispatch(commandBuffer, extent,
			ShaderParameterBlock()
				.SetImage("gImage", input, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
				.SetConstant("gExposure", std::pow(2.f, mExposure))
				.SetBuffer("gMax", mMaxBuf)
			, defines);
	}
};

}
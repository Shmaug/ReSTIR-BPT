#pragma once

#include <Common/Common.h>
#include <Core/PipelineCache.hpp>

namespace ptvk {

class Renderer {
public:
	ComputePipelineCache mRenderPipeline;
	ComputePipelineCache mBlurPipeline;
	ComputePipelineCache mPostProcessPipeline;

	float4 mColor0 = float4(0,0,0,1);
	float4 mColor1 = float4(1,1,1,1);
	float mPower = 1;
	float mExposure = 1;

	int mBlurIterations = 3;
	int mBlurRadius = 5;
	float mBlurScale = .01f;

	std::array<Buffer::View<float4>,2> mRadianceBuffers;

	inline Renderer(Device& device) {
		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path")  + "/Kernels/Test.slang";
		mRenderPipeline      = ComputePipelineCache(shaderFile, "Render");
		mBlurPipeline        = ComputePipelineCache(shaderFile, "Blur");
		mPostProcessPipeline = ComputePipelineCache(shaderFile, "PostProcess");
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
		ProfilerScope p("Renderer::Render");

		size_t pixelCount = size_t(renderTarget.GetExtent().width) * size_t(renderTarget.GetExtent().height);

		if (!mRadianceBuffers[0] || mRadianceBuffers[0].size() < pixelCount)
			for (size_t i = 0; i < mRadianceBuffers.size(); i++)
				mRadianceBuffers[i] = std::make_shared<Buffer>(commandBuffer.mDevice, "Radiance" + std::to_string(i), pixelCount*sizeof(float4), vk::BufferUsageFlagBits::eStorageBuffer);

		mRenderPipeline.Dispatch(commandBuffer,
			renderTarget.GetExtent(),
			ShaderParameterBlock()
				.SetBuffer("gRadiance", 0, mRadianceBuffers[0])
				.SetBuffer("gRadiance", 1, mRadianceBuffers[1])
				.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead)
				.SetConstant("gColor0", mColor1)
				.SetConstant("gColor1", mColor0)
				.SetConstant("gPower", mPower)
		);

		for (int i = 0; i < mBlurIterations; i++)
			for (int axis = 0; axis < 2; axis++) {
				mBlurPipeline.Dispatch(commandBuffer,
					renderTarget.GetExtent(),
					ShaderParameterBlock()
						.SetBuffer("gRadiance", 0, mRadianceBuffers[0])
						.SetBuffer("gRadiance", 1, mRadianceBuffers[1])
						.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead)
						.SetConstant("gBlurAxis", axis)
						.SetConstant("gBlurRadius", mBlurRadius)
						.SetConstant("gBlurScale", mBlurScale)
				);
			}

		mPostProcessPipeline.Dispatch(commandBuffer,
			renderTarget.GetExtent(),
			ShaderParameterBlock()
				.SetBuffer("gRadiance", 0, mRadianceBuffers[0])
				.SetBuffer("gRadiance", 1, mRadianceBuffers[1])
				.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetConstant("gExposure", mExposure)
		);
	}

	inline void OnInspectorGui() {
		ImGui::ColorEdit4("Color 0", &mColor0.x);
		ImGui::ColorEdit4("Color 1", &mColor1.x);
		Gui::ScalarField("Power", &mPower);
		Gui::ScalarField("Exposure", &mExposure);
		Gui::ScalarField("Blur iterations", &mBlurIterations);
		Gui::ScalarField("Blur radius", &mBlurRadius);
		Gui::ScalarField("Blur scale", &mBlurScale);
	}
};

}
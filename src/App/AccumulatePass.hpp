#pragma once

#include "VisibilityPass.hpp"

namespace ptvk {

class AccumulatePass {
private:
	ComputePipelineCache mAccumulatePipeline;
	ComputePipelineCache mDemodulatePipeline;
	ComputePipelineCache mBlurPipeline;

	std::array<Image::View,2> mAccumColor;
	std::array<Image::View,2> mAccumMoments;

	uint32_t mBlurPasses = 0;
	FilterKernel mBlurType = FilterKernel::eBox5;
	bool mMaxFilter = false;
	float mDiscardResponse = 1;
	Image::View mBlurImage;

	bool mReproject = true;
	bool mDemodulateAlbedo = true;
	float mHistoryLimit = 16;
	float mNormalReuseCutoff = 8; // degrees
	float mDepthReuseCutoff = 0.01f; // scene units
	DenoiserDebugMode mDebugMode = DenoiserDebugMode::eNone;

	uint32_t mNumAccumulated = 0;
	bool mResetAccumulation = false;

public:
	inline AccumulatePass(Device& device) {
		const std::filesystem::path shaderPath = *device.mInstance.GetOption("shader-kernel-path");
		mAccumulatePipeline = ComputePipelineCache(shaderPath / "Kernels/Accumulate.slang", "Accumulate");
		mDemodulatePipeline = ComputePipelineCache(shaderPath / "Kernels/Demodulate.slang");
		mBlurPipeline       = ComputePipelineCache(shaderPath / "Kernels/Blur.slang");
	}

	inline void OnInspectorGui() {
		ImGui::LabelText("Frames accumulated", "%u", mNumAccumulated);
		if (ImGui::Button("Reset Accumulation")) {
			mResetAccumulation = true;
			mNumAccumulated = 0;
		}
		ImGui::Checkbox("Reproject", &mReproject);
		ImGui::Checkbox("Demodulate albedo", &mDemodulateAlbedo);
		Gui::ScalarField<float>("Sample count", &mHistoryLimit, 0, 16384);
		if (mReproject) {
			Gui::ScalarField<float>("Normal cutoff", &mNormalReuseCutoff,  0, 90, 0);
			Gui::ScalarField<float>("Depth cutoff" , &mDepthReuseCutoff , -10, 10, .1f);
		}

		Gui::ScalarField<uint32_t>("Blur passes", &mBlurPasses, 0, 5, .1f);
		ImGui::Checkbox("Blur max filter", &mMaxFilter);
		Gui::EnumDropdown<FilterKernel>("Blur type", mBlurType, FilterKernelStrings);
		Gui::ScalarField<float>("History discard response", &mDiscardResponse, 0, 10, .01f);

		Gui::EnumDropdown<DenoiserDebugMode>("Debug mode", mDebugMode, DenoiserDebugModeStrings);
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& inputColor, const VisibilityPass& visibility, const Image::View& discardMask) {
		ProfilerScope ps("AccumulatePass::Render", &commandBuffer);

		const vk::Extent3D extent = inputColor.GetExtent();

		Defines defines;
		defines.emplace("gDebugMode", "((DenoiserDebugMode)" + std::to_string((uint32_t)mDebugMode) + ")");
		if (mReproject) defines.emplace("gReproject", "true");
		if (discardMask) defines.emplace("gUseDiscardMask", "true");

		bool reset = mResetAccumulation;
		mResetAccumulation = false;

		if (!mAccumColor[0] || mAccumColor[0].GetExtent() != extent) {
			for (size_t i = 0; i < 2; i++) {
				mAccumColor[i] = std::make_shared<Image>(commandBuffer.mDevice, "gAccumColor" + std::to_string(i), ImageInfo{
					.mFormat = vk::Format::eR32G32B32A32Sfloat,
					.mExtent = extent,
					.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst
				});
				mAccumMoments[i] = std::make_shared<Image>(commandBuffer.mDevice, "gAccumMoments" + std::to_string(i), ImageInfo{
					.mFormat = vk::Format::eR32G32Sfloat,
					.mExtent = extent,
					.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage
				});
			}
			mBlurImage = std::make_shared<Image>(commandBuffer.mDevice, "gBlurImage", ImageInfo{
				.mFormat = vk::Format::eR16Sfloat,
				.mExtent = extent,
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc
			});

			reset = true;
		}

		if (!mReproject && visibility.GetMVP() != visibility.GetPrevMVP())
			reset = true;

		const uint32_t idx = mNumAccumulated & 1;
		const Image::View& accumColor       = mAccumColor[idx];
		const Image::View& prevAccumColor   = mAccumColor[(~idx)&1];
		const Image::View& accumMoments     = mAccumMoments[idx];
		const Image::View& prevAccumMoments = mAccumMoments[(~idx)&1];

		if (reset)
			commandBuffer.ClearColor(prevAccumColor, vk::ClearColorValue{ std::array<float,4>{ 0, 0, 0, 0 } });
	 	else if (discardMask) {
			Defines defs;
			defs.emplace("gFilterKernel", "((FilterKernel)" + std::to_string((uint32_t)mBlurType) + ")");
			if (mMaxFilter) defs.emplace("gMaxFilter", "true");
			ShaderParameterBlock blurParams;
			blurParams.SetConstant("gOutputSize", uint2(extent.width, extent.height));
			std::array<Image::View, 2> imgs = { discardMask, mBlurImage };
			for (uint32_t i = 0; i < mBlurPasses; i++) {
				blurParams.SetConstant("gIteration", i);
				blurParams.SetConstant("gStepSize", 1 << i);
				blurParams.SetImage("gInput",  imgs[i&1],     vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
				blurParams.SetImage("gOutput", imgs[(i&1)^1], vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
				mBlurPipeline.Dispatch(commandBuffer, extent, blurParams, defs);
				if (i == mBlurPasses - 1 && (i&1) == 0)
					commandBuffer.Copy(mBlurImage, discardMask);
			}
		}

		if (mDemodulateAlbedo)
			mDemodulatePipeline.Dispatch(commandBuffer, extent,
				ShaderParameterBlock()
					.SetImage("gImage", inputColor, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
					.SetImage("gAlbedo", visibility.GetAlbedos(), vk::ImageLayout::eGeneral));

		mAccumulatePipeline.Dispatch(commandBuffer, extent,
			ShaderParameterBlock()
				.SetImage("gImage",            inputColor,       vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
				.SetImage("gPositions",        visibility.GetDepthNormals(), vk::ImageLayout::eGeneral)
				.SetImage("gAccumColor",       accumColor,       vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetImage("gAccumMoments",     accumMoments,     vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetImage("gPrevAccumColor",   prevAccumColor,   vk::ImageLayout::eGeneral)
				.SetImage("gPrevAccumMoments", prevAccumMoments, vk::ImageLayout::eGeneral)
				.SetImage("gPrevPositions",    visibility.GetPrevDepthNormals(), vk::ImageLayout::eGeneral)
				.SetImage("gDiscardMask",      discardMask,  vk::ImageLayout::eGeneral)
				.SetConstant("gHistoryLimit", mHistoryLimit)
				.SetConstant("gNormalReuseCutoff", glm::cos(glm::radians(mNormalReuseCutoff)))
				.SetConstant("gDepthReuseCutoff", mDepthReuseCutoff)
				.SetConstant("gPrevWorldToClip", visibility.GetPrevMVP())
				.SetConstant("gDiscardResponse", mDiscardResponse)
			, defines);

		if (mDemodulateAlbedo)
			mDemodulatePipeline.Dispatch(commandBuffer, extent,
				ShaderParameterBlock()
					.SetImage("gImage", inputColor, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
					.SetImage("gAlbedo", visibility.GetAlbedos(), vk::ImageLayout::eGeneral),
				{ {"gModulate", "true"} });

		mNumAccumulated++;
	}
};

}
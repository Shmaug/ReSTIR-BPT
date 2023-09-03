#pragma once

#include <Core/PipelineCache.hpp>
#include <Common/Enums.h>

namespace ptvk {

class AccumulatePass {
private:
	ComputePipelineCache mAccumulatePipeline;
	ComputePipelineCache mDemodulatePipeline;

	std::array<Image::View,2> mAccumColor;
	std::array<Image::View,2> mAccumMoments;
	Image::View mPrevPositions;
	float4x4    mPrevMVP;
	float3      mPrevCameraPosition;

	bool mReproject = true;
	bool mDemodulateAlbedo = true;
	float mHistoryLimit = 16;
	float mNormalReuseCutoff = 8; // degrees
	float mDepthReuseCutoff = 0.01f; // scene units
	DenoiserDebugMode mDebugMode = DenoiserDebugMode::eNone;

	uint32_t mNumAccumulated = 0;
	bool mResetAccumulation = false;
	std::unique_ptr<vk::raii::Event> mPrevFrameDoneEvent;

public:
	inline AccumulatePass(Device& device) {
		const std::filesystem::path shaderPath = *device.mInstance.GetOption("shader-kernel-path");
		mAccumulatePipeline = ComputePipelineCache(shaderPath / "Kernels/Accumulate.slang", "Accumulate");
		mDemodulatePipeline = ComputePipelineCache(shaderPath / "Kernels/Demodulate.slang");
	}

	inline const Image::View& GetPrevPositions() const { return mPrevPositions; }
	inline const float4x4& GetPrevMVP() const { return mPrevMVP; }
	inline const float3& GetPrevCameraPosition() const { return mPrevCameraPosition; }

	inline void OnInspectorGui() {
		ImGui::LabelText("Frames accumulated", "%u", mNumAccumulated);
		if (ImGui::Button("Reset Accumulation")) {
			mResetAccumulation = true;
			mNumAccumulated = 0;
		}
		ImGui::Checkbox("Reproject", &mReproject);
		ImGui::Checkbox("Demodulate albedo", &mDemodulateAlbedo);
		ImGui::PushItemWidth(40);
		ImGui::DragFloat("Sample count", &mHistoryLimit, .1f, 0, 16384);
		if (mReproject) {
			ImGui::DragFloat("Normal cutoff", &mNormalReuseCutoff, .1f, 0, 90);
			ImGui::DragFloat("Depth cutoff", &mDepthReuseCutoff , .1f, -10, 10);
		}
		ImGui::PopItemWidth();
		Gui::EnumDropdown<DenoiserDebugMode>("Debug mode", mDebugMode, DenoiserDebugModeStrings);
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& inputColor, const Image::View& inputAlbedo, const Image::View& inputPositions, const float4x4& cameraToWorld, const float4x4& projection) {
		ProfilerScope ps("AccumulatePass::Render", &commandBuffer);

		Defines defines;
		defines.emplace("gDebugMode", "((DenoiserDebugMode)" + std::to_string((uint32_t)mDebugMode) + ")");
		if (mReproject) defines.emplace("gReproject", "true");

		const vk::Extent3D extent = inputColor.GetExtent();

		bool reset = mResetAccumulation;
		mResetAccumulation = false;

		if (!mAccumColor[0] || mAccumColor[0].GetExtent() != extent) {
			for (size_t i = 0; i < 2; i++) {
				mAccumColor[i] = std::make_shared<Image>(commandBuffer.mDevice, "gAccumColor" + std::to_string(i), ImageInfo{
					.mFormat = vk::Format::eR16G16B16A16Sfloat,
					.mExtent = extent,
					.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst
				});
				mAccumMoments[i] = std::make_shared<Image>(commandBuffer.mDevice, "gAccumMoments" + std::to_string(i), ImageInfo{
					.mFormat = vk::Format::eR16G16Sfloat,
					.mExtent = extent,
					.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage
				});
			}
			mPrevPositions  = std::make_shared<Image>(commandBuffer.mDevice, "gPrevPositions", ImageInfo{
				.mFormat = vk::Format::eR32G32B32A32Sfloat,
				.mExtent = inputColor.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});

			reset = true;
		}

		const float4x4 inputMVP = projection * inverse(cameraToWorld);
		if (!mReproject && inputMVP != mPrevMVP)
			reset = true;

		const uint32_t idx = mNumAccumulated & 1;
		const Image::View& accumColor     = mAccumColor[idx];
		const Image::View& prevAccumColor = mAccumColor[(~idx)&1];
		const Image::View& accumMoments     = mAccumMoments[idx];
		const Image::View& prevAccumMoments = mAccumMoments[(~idx)&1];

		if (mPrevFrameDoneEvent) {
			commandBuffer->waitEvents(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {
				vk::ImageMemoryBarrier{
					vk::AccessFlagBits::eTransferWrite,
					vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal,
					vk::ImageLayout::eGeneral,
					VK_QUEUE_FAMILY_IGNORED,
					VK_QUEUE_FAMILY_IGNORED,
					**mPrevPositions.GetImage(),
					mPrevPositions.GetSubresourceRange(),
				} });
			mPrevPositions.SetSubresourceState(vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
		} else
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{});

		if (reset) {
			commandBuffer.ClearColor(prevAccumColor, vk::ClearColorValue{ std::array<float,4>{ 0, 0, 0, 0 } });
			commandBuffer.ClearColor(mPrevPositions, vk::ClearColorValue{ std::array<float,4>{ POS_INFINITY, POS_INFINITY, POS_INFINITY, 0 } });
		}

		if (mDemodulateAlbedo)
			mDemodulatePipeline.Dispatch(commandBuffer, extent,
				ShaderParameterBlock()
					.SetImage("gImage", inputColor, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
					.SetImage("gAlbedo", inputAlbedo, vk::ImageLayout::eGeneral));

		mAccumulatePipeline.Dispatch(commandBuffer, extent,
			ShaderParameterBlock()
				.SetImage("gImage",            inputColor,       vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
				.SetImage("gPositions",        inputPositions,   vk::ImageLayout::eGeneral)
				.SetImage("gAccumColor",       accumColor,       vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetImage("gAccumMoments",     accumMoments,     vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetImage("gPrevAccumColor",   prevAccumColor,   vk::ImageLayout::eGeneral)
				.SetImage("gPrevAccumMoments", prevAccumMoments, vk::ImageLayout::eGeneral)
				.SetImage("gPrevPositions",    mPrevPositions,   vk::ImageLayout::eGeneral)
				.SetConstant("gHistoryLimit", mHistoryLimit)
				.SetConstant("gNormalReuseCutoff", glm::cos(glm::radians(mNormalReuseCutoff)))
				.SetConstant("gDepthReuseCutoff", mDepthReuseCutoff)
				.SetConstant("gPrevWorldToClip", mPrevMVP)
			, defines);

		if (mDemodulateAlbedo)
			mDemodulatePipeline.Dispatch(commandBuffer, extent,
				ShaderParameterBlock()
					.SetImage("gImage", inputColor, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
					.SetImage("gAlbedo", inputAlbedo, vk::ImageLayout::eGeneral),
				{ {"gModulate", "true"} });

		commandBuffer.Copy(inputPositions, mPrevPositions);

		commandBuffer->setEvent(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer);

		mPrevCameraPosition = TransformPoint(cameraToWorld, float3(0));
		mPrevMVP = inputMVP;
		mNumAccumulated++;
	}
};

}
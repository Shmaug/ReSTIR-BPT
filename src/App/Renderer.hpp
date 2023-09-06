#pragma once

#include "PathTracePass.hpp"
#include "ReSTIRPTPass.hpp"
#include "AccumulatePass.hpp"
#include "TonemapPass.hpp"

namespace ptvk {

class Renderer {
public:
	std::unique_ptr<ReSTIRPTPass>   mPathTracePass;
	std::unique_ptr<TonemapPass>    mTonemapPass;
	std::unique_ptr<AccumulatePass> mAccumulatePass;

	bool mEnableAccumulation = true;
	bool mEnableTonemapper = true;
	float mRenderScale = 1;

	ResourceQueue<Image::View> mCachedRenderTargets;

	Image::View mPrevPositions;
	float4x4    mPrevMVP;
	float3      mPrevCameraPosition;
	std::unique_ptr<vk::raii::Event> mPrevFrameDoneEvent;

	inline Renderer(Device& device) {
		mPathTracePass  = std::make_unique<ReSTIRPTPass>(device);
		mTonemapPass    = std::make_unique<TonemapPass>(device);
		mAccumulatePass = std::make_unique<AccumulatePass>(device);
	}

	inline void OnInspectorGui() {
		if (ImGui::Begin("Renderer")) {
			ImGui::SliderFloat("Render Scale", &mRenderScale, 0.125f, 1.5f);
			if (ImGui::CollapsingHeader("Path Tracing")) {
				ImGui::Indent();
				mPathTracePass->OnInspectorGui();
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("Accumulation")) {
				ImGui::Checkbox("Enable Accumulation", &mEnableAccumulation);
				ImGui::Indent();
				mAccumulatePass->OnInspectorGui();
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("Tonemapper")) {
				ImGui::Checkbox("Enable Tonemapper", &mEnableTonemapper);
				ImGui::Indent();
				mTonemapPass->OnInspectorGui();
				ImGui::Unindent();
			}
		}
		ImGui::End();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& backBuffer, const Scene& scene, const Camera& camera) {
		ProfilerScope p("Renderer::Render");
		vk::Extent3D extent = {
			std::max<uint32_t>(1, uint32_t(backBuffer.GetExtent().width * mRenderScale)),
			std::max<uint32_t>(1, uint32_t(backBuffer.GetExtent().height * mRenderScale)),
			1 };
		Image::View& renderTarget = *mCachedRenderTargets.Get(commandBuffer.mDevice);
		if (!renderTarget || renderTarget.GetExtent() != extent) {
			renderTarget  = std::make_shared<Image>(commandBuffer.mDevice, "Render Target", ImageInfo{
				.mFormat = vk::Format::eR16G16B16A16Sfloat,
				.mExtent = extent,
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst
			});
			mPrevPositions  = std::make_shared<Image>(commandBuffer.mDevice, "gPrevPositions", ImageInfo{
				.mFormat = vk::Format::eR32G32B32A32Sfloat,
				.mExtent = extent,
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});
		}

		// wait for previous frame to finish writing
		if (mPrevFrameDoneEvent) {
			const auto&[layout, stage, access, queue] = mPrevPositions.GetImage()->GetSubresourceState(mPrevPositions.GetSubresourceRange().baseArrayLayer, mPrevPositions.GetSubresourceLayer().mipLevel);
			commandBuffer->waitEvents(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {
				vk::ImageMemoryBarrier{
					access, vk::AccessFlagBits::eShaderRead,
					layout, vk::ImageLayout::eGeneral,
					queue, queue,
					**mPrevPositions.GetImage(),
					mPrevPositions.GetSubresourceRange(),
				} });
			mPrevPositions.SetSubresourceState(vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
		} else
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{});

		const float4x4 cameraToWorld = NodeToWorld(camera.mNode);
		const float4x4 projection = camera.GetProjection() * glm::scale(float3(1,-1,1));
		const float4x4 mvp = projection * inverse(cameraToWorld);

		// render
		mPathTracePass->Render(commandBuffer, renderTarget, scene, cameraToWorld, projection, mPrevPositions, mPrevMVP, mPrevCameraPosition);

		// accumulate/denoise
		if (mEnableAccumulation)
			mAccumulatePass->Render(commandBuffer, renderTarget, mPathTracePass->GetAlbedo(), mPathTracePass->GetPositions(), mvp, mPrevPositions, mPrevMVP);

		// copy gbuffer positions for future frames
		{
			commandBuffer.Copy(mPathTracePass->GetPositions(), mPrevPositions);
			commandBuffer->setEvent(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer);
			mPrevMVP = mvp;
			mPrevCameraPosition = TransformPoint(cameraToWorld, float3(0));
		}

		// tonemap
		if (mEnableTonemapper)
			mTonemapPass->Render(commandBuffer, renderTarget);

		// blit result to back buffer
		commandBuffer.Blit(renderTarget, backBuffer);
	}
};

}

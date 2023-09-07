#pragma once

#include "VisibilityPass.hpp"
#include "AccumulatePass.hpp"
#include "TonemapPass.hpp"
#include "ReSTIRPTPass.hpp"

namespace ptvk {

class Renderer {
public:
	std::unique_ptr<VisibilityPass> mVisibilityPass;
	std::unique_ptr<AccumulatePass> mAccumulatePass;
	std::unique_ptr<TonemapPass>    mTonemapPass;
	std::unique_ptr<ReSTIRPTPass>   mGIPass;

	bool mEnableAccumulation = true;
	bool mEnableTonemapper = true;
	float mRenderScale = 1;

	ResourceQueue<Image::View> mCachedRenderTargets;

	inline Renderer(Device& device) {
		mVisibilityPass = std::make_unique<VisibilityPass>(device);
		mAccumulatePass = std::make_unique<AccumulatePass>(device);
		mTonemapPass    = std::make_unique<TonemapPass>(device);
		mGIPass         = std::make_unique<ReSTIRPTPass>(device);
	}

	inline void OnInspectorGui() {
		if (ImGui::Begin("Renderer")) {
			ImGui::SliderFloat("Render Scale", &mRenderScale, 0.125f, 1.5f);
			if (ImGui::CollapsingHeader("Visibility")) {
				ImGui::Indent();
				mVisibilityPass->OnInspectorGui();
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("Path Tracing")) {
				ImGui::Indent();
				mGIPass->OnInspectorGui();
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
		}

		const float4x4 cameraToWorld = NodeToWorld(camera.mNode);
		const float4x4 projection = camera.GetProjection() * glm::scale(float3(1,-1,1));

		// visibility
		mVisibilityPass->Render(commandBuffer, renderTarget, scene, cameraToWorld, projection);

		// render
		mGIPass->Render(commandBuffer, renderTarget, scene, *mVisibilityPass);

		// accumulate/denoise
		if (mEnableAccumulation)
			mAccumulatePass->Render(commandBuffer, renderTarget, *mVisibilityPass);

		// tonemap
		if (mEnableTonemapper)
			mTonemapPass->Render(commandBuffer, renderTarget);

		mVisibilityPass->PostRender(commandBuffer, renderTarget);

		// blit result to back buffer
		commandBuffer.Blit(renderTarget, backBuffer, vk::Filter::eNearest);
	}
};

}

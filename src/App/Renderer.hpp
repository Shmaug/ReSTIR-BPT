#pragma once

#include "PathTracePass.hpp"
#include "AccumulatePass.hpp"
#include "TonemapPass.hpp"

namespace ptvk {

class Renderer {
public:
	std::unique_ptr<PathTracePass>  mPathTracePass;
	std::unique_ptr<TonemapPass>    mTonemapPass;
	std::unique_ptr<AccumulatePass> mAccumulatePass;

	bool mEnableAccumulation = true;
	bool mEnableTonemapper = true;

	ResourceQueue<Image::View> mCachedRenderTargets;

	inline Renderer(Device& device) {
		mPathTracePass  = std::make_unique<PathTracePass>(device);
		mTonemapPass    = std::make_unique<TonemapPass>(device);
		mAccumulatePass = std::make_unique<AccumulatePass>(device);
	}

	inline void OnInspectorGui() {
		if (ImGui::Begin("Renderer")) {
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
		Image::View& renderTarget = *mCachedRenderTargets.Get(commandBuffer.mDevice);
		if (!renderTarget || renderTarget.GetExtent() != backBuffer.GetExtent()) {
			renderTarget  = std::make_shared<Image>(commandBuffer.mDevice, "Render Target", ImageInfo{
				.mFormat = vk::Format::eR16G16B16A16Sfloat,
				.mExtent = backBuffer.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst
			});
		}

		const float4x4 cameraToWorld = NodeToWorld(camera.mNode);
		const float4x4 projection = camera.GetProjection() * glm::scale(float3(1,-1,1));

		mPathTracePass ->Render(commandBuffer, renderTarget, scene, cameraToWorld, projection);
		if (mEnableAccumulation)
			mAccumulatePass->Render(commandBuffer, renderTarget, mPathTracePass->GetAlbedo(), mPathTracePass->GetPositions(), cameraToWorld, projection);
		if (mEnableTonemapper)
			mTonemapPass->Render(commandBuffer, renderTarget);

		commandBuffer.Blit(renderTarget, backBuffer);
	}
};

}

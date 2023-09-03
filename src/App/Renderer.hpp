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

	Image::View mRenderTarget;

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
				ImGui::Indent();
				mAccumulatePass->OnInspectorGui();
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("Tonemapper")) {
				ImGui::Indent();
				mTonemapPass->OnInspectorGui();
				ImGui::Unindent();
			}
		}
		ImGui::End();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const Camera& camera) {
		ProfilerScope p("Renderer::Render");
		if (!mRenderTarget || mRenderTarget.GetExtent() != renderTarget.GetExtent()) {
			mRenderTarget  = std::make_shared<Image>(commandBuffer.mDevice, "mRenderTarget", ImageInfo{
				.mFormat = vk::Format::eR16G16B16A16Sfloat,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst
			});
		}

		const float4x4 cameraToWorld = NodeToWorld(camera.mNode);
		const float4x4 projection = camera.GetProjection() * glm::scale(float3(1,-1,1));

		mPathTracePass ->Render(commandBuffer, mRenderTarget, scene, cameraToWorld, projection);
		mAccumulatePass->Render(commandBuffer, mRenderTarget, mPathTracePass->GetAlbedo(), mPathTracePass->GetPositions(), cameraToWorld, projection);
		mTonemapPass   ->Render(commandBuffer, mRenderTarget);

		commandBuffer.Blit(mRenderTarget, renderTarget);
	}
};

}

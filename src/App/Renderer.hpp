#pragma once

#include "VisibilityPass.hpp"
#include "AccumulatePass.hpp"
#include "TonemapPass.hpp"
#include "ReSTIRPTPass.hpp"
#include "PathTracePass.hpp"
#include "VCMPass.hpp"
#include "BPTPass.hpp"

namespace ptvk {

class Renderer {
public:
	Device& mDevice;

	std::unique_ptr<VisibilityPass> mVisibilityPass;
	std::unique_ptr<AccumulatePass> mAccumulatePass;
	std::unique_ptr<TonemapPass>    mTonemapPass;
	std::variant<
		std::unique_ptr<PathTracePass>,
		std::unique_ptr<ReSTIRPTPass>,
		std::unique_ptr<BPTPass>,
		std::unique_ptr<VCMPass>
		> mRenderer;

	bool mEnableAccumulation = true;
	bool mEnableTonemapper = true;
	float mRenderScale = 1;

	ResourceQueue<Image::View> mCachedRenderTargets;

	inline Renderer(Device& device) : mDevice(device) {
		mVisibilityPass = std::make_unique<VisibilityPass>(device);
		mAccumulatePass = std::make_unique<AccumulatePass>(device);
		mTonemapPass    = std::make_unique<TonemapPass>(device);
		mRenderer       = std::make_unique<PathTracePass>(device);
	}

	inline void OnInspectorGui() {
		if (ImGui::Begin("Passes")) {
			ImGui::SliderFloat("Render Scale", &mRenderScale, 0.125f, 1.5f);
			if (ImGui::CollapsingHeader("Visibility")) {
				ImGui::Indent();
				mVisibilityPass->OnInspectorGui();
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("Global illumination")) {
				ImGui::Indent();

				const char* labels[] = {
					"Path Tracer",
					"ReSTIR PT",
					"Bidirectional Path Tracer",
					"VCM"
				};
				uint32_t mode = mRenderer.index();
				Gui::EnumDropdown("Type", mode, labels);
				if (mode != mRenderer.index()) {
					mDevice->waitIdle();
					switch (mode) {
						case 0:
							mRenderer = std::make_unique<PathTracePass>(mDevice);
							break;
						case 1:
							mRenderer = std::make_unique<ReSTIRPTPass>(mDevice);
							break;
						case 2:
							mRenderer = std::make_unique<BPTPass>(mDevice);
							break;
						case 3:
							mRenderer = std::make_unique<VCMPass>(mDevice);
							break;
					}
				}

				std::visit(
					[](const auto& p) { p->OnInspectorGui(); },
					mRenderer );

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

		// visibility
		mVisibilityPass->Render(commandBuffer, renderTarget, scene, camera);

		// render
		std::visit(
			[&](const auto& p) { p->Render(commandBuffer, renderTarget, scene, *mVisibilityPass); },
			mRenderer);

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

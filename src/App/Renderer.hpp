#pragma once

#include "VisibilityPass.hpp"
#include "AccumulatePass.hpp"
#include "TonemapPass.hpp"
#include "ReSTIRPTPass.hpp"
#include "PathTracePass.hpp"
#include "BPTPass.hpp"
#include "LightTracePass.hpp"

namespace ptvk {

class Renderer {
public:
	Device& mDevice;

	std::unique_ptr<VisibilityPass> mVisibilityPass;
	std::unique_ptr<AccumulatePass> mAccumulatePass;
	std::unique_ptr<TonemapPass>    mTonemapPass;
	std::tuple<
		std::unique_ptr<PathTracePass>,
		std::unique_ptr<ReSTIRPTPass>,
		std::unique_ptr<BPTPass>,
		std::unique_ptr<LightTracePass>
		> mRenderers;
	inline static const char* const RendererStrings[] = {
		"Path Tracer",
		"ReSTIR PT",
		"Bidirectional Path Tracer",
		"Light Tracer"
	};

	uint32_t mCurrentRenderer = 0;
	float mRenderScale = 1;
	bool mEnableAccumulation = true;
	bool mEnableTonemapper = true;

	ResourceQueue<Image::View> mCachedRenderTargets;

	inline auto CallRendererFn(auto fn) {
		switch (mCurrentRenderer) {
			default:
			case 0: return fn(std::get<0>(mRenderers));
			case 1: return fn(std::get<1>(mRenderers));
			case 2: return fn(std::get<2>(mRenderers));
			case 3: return fn(std::get<3>(mRenderers));
		}
	}
	inline void CreateRenderer() {
		switch (mCurrentRenderer) {
			case 0:
				std::get<0>(mRenderers) = std::make_unique<PathTracePass>(mDevice);
				break;
			case 1:
				std::get<1>(mRenderers) = std::make_unique<ReSTIRPTPass>(mDevice);
				break;
			case 2:
				std::get<2>(mRenderers) = std::make_unique<BPTPass>(mDevice);
				break;
			case 3:
				std::get<3>(mRenderers) = std::make_unique<LightTracePass>(mDevice);
				break;
		}
	}

	inline Renderer(Device& device) : mDevice(device) {
		mVisibilityPass = std::make_unique<VisibilityPass>(device);
		mAccumulatePass = std::make_unique<AccumulatePass>(device);
		mTonemapPass    = std::make_unique<TonemapPass>(device);

		if (auto r = device.mInstance.GetOption("renderer"))
			mCurrentRenderer = std::stoi(*r);
		CreateRenderer();
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

				uint32_t type = mCurrentRenderer;
				Gui::EnumDropdown("Type", mCurrentRenderer, RendererStrings);
				if (mCurrentRenderer != type) {
					if (!CallRendererFn([](const auto& r) -> bool { return (bool)r; })) {
						mDevice->waitIdle();
						CreateRenderer();
					}
				}

				CallRendererFn([](const auto& p) { p->OnInspectorGui(); });

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
		CallRendererFn([&](const auto& p) { p->Render(commandBuffer, renderTarget, scene, *mVisibilityPass); });

		Image::View discardMask;
		if (const auto& r = std::get<std::unique_ptr<ReSTIRPTPass>>(mRenderers); r && mCurrentRenderer == 1) {
			discardMask = r->GetDiscardMask();
		}

		// accumulate/denoise
		if (mEnableAccumulation)
			mAccumulatePass->Render(commandBuffer, renderTarget, *mVisibilityPass, discardMask);

		// tonemap
		if (mEnableTonemapper)
			mTonemapPass->Render(commandBuffer, renderTarget);

		mVisibilityPass->PostRender(commandBuffer, renderTarget);

		// blit result to back buffer
		commandBuffer.Blit(renderTarget, backBuffer, vk::Filter::eNearest);
	}
};

}

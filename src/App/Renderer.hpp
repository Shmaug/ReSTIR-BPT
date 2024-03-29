#pragma once

#include "VisibilityPass.hpp"
#include "AccumulatePass.hpp"
#include "TonemapPass.hpp"
#include "ReSTIRPTPass.hpp"
#include "PathTracePass.hpp"
#include "BPTPass.hpp"
#include "LightTracePass.hpp"
#include "SMSPass.hpp"

namespace ptvk {

using RendererTuple = std::tuple<
	std::unique_ptr<PathTracePass>,
	std::unique_ptr<ReSTIRPTPass>,
	std::unique_ptr<BPTPass>,
	std::unique_ptr<LightTracePass>,
	std::unique_ptr<SMSPass>
	>;
const char* const RendererStrings[] = {
	"Path Tracer",
	"ReSTIR PT",
	"Bidirectional Path Tracer",
	"Light Tracer",
	"Specular Manifold Sampling"
};

class Renderer {
public:
	Device& mDevice;

	std::unique_ptr<VisibilityPass> mVisibilityPass;
	std::unique_ptr<AccumulatePass> mAccumulatePass;
	std::unique_ptr<TonemapPass>    mTonemapPass;
	RendererTuple mRenderers;

	uint32_t mCurrentRenderer = 0;
	bool mEnableAccumulation = true;
	bool mEnableTonemapper   = true;

	bool mPause = false;
	bool mRenderOnce = false;

	ResourceQueue<Image::View> mCachedRenderTargets;
	Image::View mLastRenderTarget;

	inline auto CallRendererFn(auto fn) {
		switch (mCurrentRenderer) {
			default:
			case 0: return fn(std::get<0>(mRenderers));
			case 1: return fn(std::get<1>(mRenderers));
			case 2: return fn(std::get<2>(mRenderers));
			case 3: return fn(std::get<3>(mRenderers));
			case 4: return fn(std::get<4>(mRenderers));
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
			case 4:
				std::get<4>(mRenderers) = std::make_unique<SMSPass>(mDevice);
				break;
		}
	}

	inline Renderer(Device& device) : mDevice(device) {
		mVisibilityPass = std::make_unique<VisibilityPass>(device);
		mAccumulatePass = std::make_unique<AccumulatePass>(device);
		mTonemapPass    = std::make_unique<TonemapPass>(device);

		if (const auto r = device.mInstance.GetOption("renderer"); r.has_value() && !r->empty()) {
			if (r->size() == 0 && std::isdigit(r->at(0))) {
				mCurrentRenderer = std::stoi(*r);
				if (mCurrentRenderer >= std::tuple_size_v<RendererTuple>) {
					std::cout << "Invalid renderer id: " << mCurrentRenderer << std::endl;
					mCurrentRenderer = 0;
				}
			} else {
				const auto it = std::ranges::find(RendererStrings, *r);
				if (it == std::ranges::end(RendererStrings))
					std::cout << "Unknown renderer type: " << *it << std::endl;
				else
					mCurrentRenderer = it - std::ranges::begin(RendererStrings);
			}
		}

		if (const auto r = device.mInstance.GetOption("accumulation"))
			mEnableAccumulation = *r == "on" || *r == "true";
		if (const auto r = device.mInstance.GetOption("tonemapper"))
			mEnableTonemapper = *r == "on" || *r == "true";

		CreateRenderer();
	}

	inline void OnInspectorGui() {
		if (ImGui::Begin("Passes")) {
			ImGui::Checkbox("Pause", &mPause);
			ImGui::SameLine();
			if (ImGui::Button("Render"))
				mRenderOnce = true;

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

	inline Image::View Render(CommandBuffer& commandBuffer, const vk::Extent3D& extent, const Scene& scene, const Camera& camera) {
		ProfilerScope p("Renderer::Render");
		Image::View& renderTarget = *mCachedRenderTargets.Get(commandBuffer.mDevice);
		if (!renderTarget || renderTarget.GetExtent() != extent) {
			renderTarget  = std::make_shared<Image>(commandBuffer.mDevice, "Render Target", ImageInfo{
				.mFormat = vk::Format::eR16G16B16A16Sfloat,
				.mExtent = extent,
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eTransferDst
			});
		}

		if (mPause && !mRenderOnce) {
			return mLastRenderTarget;
		}
		mRenderOnce = false;

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
		mLastRenderTarget = renderTarget;
		return renderTarget;
	}
};

}

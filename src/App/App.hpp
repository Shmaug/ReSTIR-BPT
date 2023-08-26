#pragma once

#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Gui.hpp>
#include <Core/Profiler.hpp>

#include <GLFW/glfw3.h>

#include "Renderer.hpp"

namespace ptvk {

class App {
public:
	std::unique_ptr<Instance> mInstance;
	std::unique_ptr<Window> mWindow;
	std::unique_ptr<Device> mDevice;
	uint32_t mPresentQueueFamily;
	vk::raii::Queue mPresentQueue;

	std::unique_ptr<Swapchain> mSwapchain;
	std::vector<std::unique_ptr<vk::raii::Semaphore>> mSemaphores;
	std::vector<std::unique_ptr<CommandBuffer>> mCommandBuffers;

	int mProfilerHistoryCount = 3;

	std::unique_ptr<Renderer> mRenderer;

	inline App(const std::vector<std::string>& args) : mPresentQueue(nullptr) {
		mInstance = std::make_unique<Instance>(args);

		vk::Extent2D windowSize{ 1600, 900 };
		if (auto arg = mInstance->GetOption("width") ; arg) windowSize.width  = std::stoi(*arg);
		if (auto arg = mInstance->GetOption("height"); arg) windowSize.height = std::stoi(*arg);
		mWindow = std::make_unique<Window>(*mInstance, "Stratum3", windowSize);

		vk::raii::PhysicalDevice physicalDevice = nullptr;
		std::tie(physicalDevice, mPresentQueueFamily) = mWindow->FindSupportedDevice();

		mDevice       = std::make_unique<Device>(*mInstance, physicalDevice);
		mPresentQueue = vk::raii::Queue(**mDevice, mPresentQueueFamily, 0);

		uint32_t minImages = 2;
		if (auto arg = mInstance->GetOption("min-images")) minImages = std::stoi(*arg);

		mSwapchain = std::make_unique<Swapchain>(*mDevice, *mWindow, minImages, vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst);
		CreateSwapchain();

		mRenderer = std::make_unique<Renderer>(*mDevice);
	}
	inline ~App() {
		(*mDevice)->waitIdle();
		Gui::Destroy();
	}

	inline bool CreateSwapchain() {
		ProfilerScope p("App::CreateSwapchain");

		(*mDevice)->waitIdle();
		if (!mSwapchain->Create())
			return false;

		// recreate swapchain-dependent resources

		Gui::Initialize(*mSwapchain, *mPresentQueue, mPresentQueueFamily);

		mCommandBuffers.resize(mSwapchain->GetImageCount());
		uint32_t idx = 0;
		for (auto& cb : mCommandBuffers) {
			if (!cb)
				cb = std::move( std::make_unique<CommandBuffer>(*mDevice, "Frame CommandBuffer " + std::to_string(idx), mPresentQueueFamily) );
			idx++;
		}

		mSemaphores.resize(mSwapchain->GetImageCount());
		idx = 0;
		for (auto& s : mSemaphores) {
			if (!s) {
				s = std::move( std::make_unique<vk::raii::Semaphore>(**mDevice, vk::SemaphoreCreateInfo()) );
				mDevice->SetDebugName(**s, "Frame Semaphore " + std::to_string(idx));
			}
			idx++;
		}

		return true;
	}

	inline void DrawInspectorGui() {
		ProfilerScope p("App::DrawInspectorGui");

		if (ImGui::Begin("Inspector")) {
			mInstance->OnInspectorGui();
			mDevice->OnInspectorGui();
			if (ImGui::CollapsingHeader("Window")) {
				ImGui::Indent();
				mWindow->OnInspectorGui();
				ImGui::Separator();
				mSwapchain->OnInspectorGui();
				ImGui::Unindent();
			}
			if (ImGui::CollapsingHeader("Renderer")) {
				mRenderer->OnInspectorGui();
			}
		}
		ImGui::End();

		// profiler timings

		if (ImGui::Begin("Profiler")) {
			Profiler::DrawFrameTimeGraph();

			ImGui::SliderInt("Count", &mProfilerHistoryCount, 1, 32);
			ImGui::PushID("Show timeline");
			if (ImGui::Button(Profiler::HasHistory() ? "Hide timeline" : "Show timeline"))
				Profiler::ResetHistory(Profiler::HasHistory() ? 0 : mProfilerHistoryCount);
			ImGui::PopID();
		}
		ImGui::End();

		// frame timeline

		if (Profiler::HasHistory()) {
			if (ImGui::Begin("Timeline"))
				Profiler::DrawTimeline();
			ImGui::End();
		}
	}

	inline void Run() {
		// main loop
		while (mWindow->IsOpen()) {
			ProfilerScope p("Frame " + std::to_string(mDevice->GetFrameIndex()));

			{
				ProfilerScope p("glfwPollEvents");
				glfwPollEvents();
			}

			while (!mSwapchain->AcquireImage()) {
				if (mSwapchain->IsDirty())
					CreateSwapchain();
			}

			{
				Profiler::BeginFrame();

				Gui::NewFrame();

				DrawInspectorGui();

				const Image::View& renderTarget = mSwapchain->GetImage();

				const uint32_t commandBufferIndex = mDevice->GetFrameIndex() % mSwapchain->GetImageCount();
				CommandBuffer& commandBuffer = *mCommandBuffers[commandBufferIndex];
				if (auto fence = commandBuffer.GetCompletionFence()) {
					ProfilerScope ps("waitForFences");
					if ((*mDevice)->waitForFences(**fence, true, ~0ull) != vk::Result::eSuccess)
						throw std::runtime_error("waitForFences failed");
				}
				commandBuffer.Reset();
				{
					ProfilerScope p("Build CommandBuffer");
					commandBuffer.ClearColor(renderTarget, vk::ClearColorValue{ std::array<float,4>{0,0,0,0} });

					mRenderer->Render(commandBuffer, renderTarget);

					Gui::Render(commandBuffer, renderTarget);

					commandBuffer.Barrier(renderTarget, vk::ImageLayout::ePresentSrcKHR, vk::PipelineStageFlagBits::eBottomOfPipe, vk::AccessFlagBits::eNone);
				}

				const vk::Semaphore semaphore = **mSemaphores[commandBufferIndex];
				commandBuffer.Submit(
					*mPresentQueue,
					**mSwapchain->GetImageAvailableSemaphore(), (vk::PipelineStageFlags)vk::PipelineStageFlagBits::eColorAttachmentOutput,
					semaphore );

				mSwapchain->Present(mPresentQueue, semaphore);
			}

			mDevice->IncrementFrameIndex();
		}
	}
};

}

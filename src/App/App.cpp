#pragma once

#include "App.hpp"
#include "Renderer.hpp"

#include <Scene/Scene.hpp>
#include <Scene/FlyCamera.hpp>

#include <GLFW/glfw3.h>
#include <portable-file-dialogs.h>

namespace ptvk {

App::App(const std::vector<std::string>& args) : mPresentQueue(nullptr) {
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
	vk::SurfaceFormatKHR surfaceFormat = vk::SurfaceFormatKHR(vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear);
	if (auto arg = mInstance->GetOption("min-images")) minImages = std::stoi(*arg);
	if (auto arg = mInstance->GetOption("surface-format-srgb")) surfaceFormat.format = vk::Format::eB8G8R8A8Srgb;
	if (auto arg = mInstance->GetOption("render-scale")) mRenderScale = std::stof(*arg);

	mSwapchain = std::make_unique<Swapchain>(*mDevice, *mWindow, minImages, vk::ImageUsageFlagBits::eColorAttachment|vk::ImageUsageFlagBits::eTransferDst, surfaceFormat);

	mScene = std::make_unique<Scene>(*mInstance);

	auto cameraNode = mScene->GetRoot()->AddChild("Camera");
	cameraNode->MakeComponent<float4x4>(glm::identity<float4x4>());
	mCamera = cameraNode->MakeComponent<Camera>(*cameraNode);
	mFlyCamera = cameraNode->MakeComponent<FlyCamera>(*cameraNode);

	mRenderer = std::make_unique<Renderer>(*mDevice);

	CreateSwapchain();
}
App::~App() {
	(*mDevice)->waitIdle();
	Gui::Destroy();
}

bool App::CreateSwapchain() {
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

void App::Update() {
	ProfilerScope p("App::Update");

	ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize), ImGuiCond_Always;
	ImGui::Begin("Background", nullptr, ImGuiWindowFlags_NoDocking|ImGuiWindowFlags_NoTitleBar|ImGuiWindowFlags_NoBringToFrontOnFocus|ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize);
	if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DockingEnable) {
		ImGuiID dockSpaceId = ImGui::GetID("Background");
		ImGui::DockSpace(dockSpaceId, ImVec2(0.0f, 0.0f), ImGuiDockNodeFlags_PassthruCentralNode);
	}
	ImGui::End();

	auto now = std::chrono::steady_clock::now();
	const float deltaTime = std::chrono::duration_cast<std::chrono::duration<float>>(now - mLastUpdate).count();
	mLastUpdate = now;

	if (ImGui::Begin("App")) {
		mInstance->OnInspectorGui();
		mDevice->OnInspectorGui();
		if (ImGui::CollapsingHeader("Window")) {
			ImGui::Indent();
			mWindow->OnInspectorGui();
			ImGui::Separator();
			mSwapchain->OnInspectorGui();
			ImGui::Unindent();
		}
		if (ImGui::SliderFloat("Render Scale", &mRenderScale, 0.125f, 1.5f))
			(*mDevice)->waitIdle();
	}
	ImGui::End();

	mRenderer->OnInspectorGui();

	// profiler timings

	if (ImGui::Begin("Profiler"))
		Profiler::DrawFrameTimeGraph();
	ImGui::End();

	// frame timeline

	if (ImGui::Begin("Timeline"))
		Profiler::DrawTimeline();
	ImGui::End();


	//ImGui::ShowMetricsWindow();

	// open file dialog
	if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
		auto f = pfd::open_file("Open scene", "", mScene->LoaderFilters());
		for (const std::string& filepath : f.result())
			mScene->LoadAsync(filepath);
	}

	for (const std::string& file : mWindow->GetDroppedFiles())
		mScene->LoadAsync(file);
	mWindow->GetDroppedFiles().clear();

	mFlyCamera->Update(deltaTime);
}

vk::Semaphore App::Render(const Image::View& renderTarget) {
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

		mScene->Update(commandBuffer);

		if (ImGui::Begin("Viewport")) {
			const float2 s = std::bit_cast<float2>(ImGui::GetWindowContentRegionMax()) - std::bit_cast<float2>(ImGui::GetWindowContentRegionMin());
			mCamera->mAspect = s.x / s.y;
			vk::Extent3D extent {
				std::max<uint32_t>(1, uint32_t(s.x * mRenderScale)),
				std::max<uint32_t>(1, uint32_t(s.y * mRenderScale)),
				1 };
			if (Image::View image = mRenderer->Render(commandBuffer, extent, *mScene, *mCamera)) {

				// copy to tmp image since nearest-neighbor filtering doesnt seem to work for ImGui
				if (mRenderScale != 1) {
					if (!mTmpImage || mTmpImage.GetExtent() != vk::Extent3D(uint32_t(s.x), uint32_t(s.y), 1))
						mTmpImage = std::make_shared<Image>(*mDevice, "mTmpImage", ImageInfo{
							.mFormat = renderTarget.GetImage()->GetFormat(),
							.mExtent = vk::Extent3D(uint32_t(s.x), uint32_t(s.y), 1) });
					commandBuffer.Blit(image, mTmpImage, vk::Filter::eNearest);
					image = mTmpImage;
				}

				ImGui::Image(Gui::GetTextureID(image, vk::Filter::eNearest), ImVec2(s.x, s.y));
				mIsViewportFocused = ImGui::IsItemHovered();
				mViewportRect = float4(std::bit_cast<float2>(ImGui::GetItemRectMin()), std::bit_cast<float2>(ImGui::GetItemRectMax()));
			}
		}
		ImGui::End();

		Gui::Render(commandBuffer, renderTarget);

		commandBuffer.Barrier(renderTarget, vk::ImageLayout::ePresentSrcKHR, vk::PipelineStageFlagBits::eBottomOfPipe, vk::AccessFlagBits::eNone);
	}
	const vk::Semaphore semaphore = **mSemaphores[commandBufferIndex];
	{
		ProfilerScope p("Submit CommandBuffer");
		commandBuffer.Submit(
			*mPresentQueue,
			**mSwapchain->GetImageAvailableSemaphore(), (vk::PipelineStageFlags)vk::PipelineStageFlagBits::eColorAttachmentOutput,
			semaphore );
	}
	return semaphore;
}

void App::Run() {
	// main loop
	while (mWindow->IsOpen()) {
		ProfilerScope p("Frame " + std::to_string(mDevice->GetFrameIndex()));

		{
			ProfilerScope p("Acquire image");
			do {
				{
					ProfilerScope p("glfwPollEvents");
					glfwPollEvents();
				}
				if (mSwapchain->IsDirty())
					CreateSwapchain();
			} while (!mSwapchain->AcquireImage());
		}

		{
			Profiler::BeginFrame();

			Gui::NewFrame();

				Update();
				const vk::Semaphore semaphore = Render(mSwapchain->GetImage());

				mSwapchain->Present(mPresentQueue, semaphore);
			}

			mDevice->IncrementFrameIndex();
		}
	}
}

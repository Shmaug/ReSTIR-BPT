#include "Gui.hpp"
#include <Core/Swapchain.hpp>

#include <imgui/imgui_impl_vulkan.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_internal.h>
#include <ImGuizmo.h>

namespace ptvk {

vk::raii::RenderPass Gui::mRenderPass = nullptr;
uint32_t Gui::mQueueFamily = 0;
std::unordered_map<vk::Image, vk::raii::Framebuffer> Gui::mFramebuffers = {};
std::shared_ptr<vk::raii::DescriptorPool> Gui::mImGuiDescriptorPool = {};
ImFont* Gui::mHeaderFont = nullptr;
std::unordered_set<Image::View> Gui::mFrameTextures = {};
std::unordered_map<std::pair<Image::View, vk::Filter>, std::pair<vk::raii::DescriptorSet, vk::raii::Sampler>, PairHash<Image::View, vk::Filter>> Gui::mTextureIDs = {};

void Gui::ProgressSpinner(const char* label, const float radius, const float thickness, bool center) {
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	ImDrawList* drawList = window->DrawList;
	const ImGuiStyle& style = ImGui::GetStyle();

	ImVec2 pos = window->DC.CursorPos;
	if (center)
    	pos.x += (ImGui::GetContentRegionAvail().x - 2*radius) * .5f;

	const ImRect bb(pos, ImVec2(pos.x + radius*2, pos.y + (radius + style.FramePadding.y)*2));
	ImGui::ItemSize(bb, style.FramePadding.y);
	if (!ImGui::ItemAdd(bb, window->GetID(label)))
		return;

	const float t = ImGui::GetCurrentContext()->Time;

	const int num_segments = drawList->_CalcCircleAutoSegmentCount(radius);

	const int start = abs(sin(t * 1.8f))*(num_segments-5);
	const float a_min = float(M_PI*2) * ((float)start) / (float)num_segments;
	const float a_max = float(M_PI*2) * ((float)num_segments-3) / (float)num_segments;

	const ImVec2 c = ImVec2(pos.x + radius, pos.y + radius + style.FramePadding.y);

	drawList->PathClear();

	for (int i = 0; i < num_segments; i++) {
		const float a = a_min + ((float)i / (float)num_segments) * (a_max - a_min);
		drawList->PathLineTo(ImVec2(
			c.x + cos(a + t*8) * radius,
			c.y + sin(a + t*8) * radius));
	}

	drawList->PathStroke(ImGui::GetColorU32(ImGuiCol_Text), 0, thickness);
}

ImTextureID Gui::GetTextureID(const Image::View& image, const vk::Filter filter) {
	if (!mImGuiDescriptorPool)
		return 0;

	auto it = mTextureIDs.find(std::make_pair(image, filter));
	if (it == mTextureIDs.end()) {
		vk::raii::Sampler sampler(*image.GetImage()->mDevice, vk::SamplerCreateInfo({}, filter, filter, filter == vk::Filter::eLinear ? vk::SamplerMipmapMode::eLinear : vk::SamplerMipmapMode::eNearest));
		vk::raii::DescriptorSet descriptorSet(
			*image.GetImage()->mDevice,
			ImGui_ImplVulkan_AddTexture(*sampler, *image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
			**mImGuiDescriptorPool);
		it = mTextureIDs.emplace(std::make_pair(image, filter), std::make_pair( std::move(descriptorSet), std::move(sampler) )).first;
	}
	mFrameTextures.emplace(image);
	return (VkDescriptorSet)*it->second.first;
}

void Gui::Initialize(const Swapchain& swapchain, const vk::Queue queue, const uint32_t queueFamily) {
	if (*mRenderPass)
		Destroy();
	mQueueFamily = queueFamily;

	// create renderpass
	vk::AttachmentReference attachmentReference(0, vk::ImageLayout::eColorAttachmentOptimal);
	vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, attachmentReference, {}, {}, {});
	vk::AttachmentDescription attachment({},
		swapchain.GetFormat().format,
		vk::SampleCountFlagBits::e1,
		vk::AttachmentLoadOp::eLoad,
		vk::AttachmentStoreOp::eStore,
		vk::AttachmentLoadOp::eDontCare,
		vk::AttachmentStoreOp::eDontCare,
		vk::ImageLayout::eColorAttachmentOptimal,
		vk::ImageLayout::eColorAttachmentOptimal );
	mRenderPass = vk::raii::RenderPass(*swapchain.mDevice, vk::RenderPassCreateInfo({}, attachment, subpass, {}));
	swapchain.mDevice.SetDebugName(*mRenderPass, "Gui::mRenderPass");

	mImGuiDescriptorPool = swapchain.mDevice.GetDescriptorPool();

	ImGui::CreateContext();

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	ImGui::GetStyle().WindowRounding = 4.0f;
	ImGui::GetStyle().GrabRounding   = 4.0f;
	ImGui::GetStyle().IndentSpacing *= 0.75f;
	{
		auto& colors = ImGui::GetStyle().Colors;
		colors[ImGuiCol_WindowBg] = ImVec4{ 0.1f, 0.1f, 0.1f, 0.9f };
		colors[ImGuiCol_DockingEmptyBg] = colors[ImGuiCol_WindowBg];

		colors[ImGuiCol_Header] = colors[ImGuiCol_WindowBg];
		colors[ImGuiCol_HeaderActive]  = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
		colors[ImGuiCol_HeaderHovered] = ImVec4{ 0.20f, 0.20f, 0.20f, 1.0f };

		colors[ImGuiCol_TitleBg]          = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
		colors[ImGuiCol_TitleBgActive]    = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };
		colors[ImGuiCol_TitleBgCollapsed] = colors[ImGuiCol_TitleBg];

		colors[ImGuiCol_Tab]                = colors[ImGuiCol_TitleBgActive];
		colors[ImGuiCol_TabHovered]         = ImVec4{ 0.45f, 0.45f, 0.45f, 1.0f };
		colors[ImGuiCol_TabActive]          = ImVec4{ 0.35f, 0.35f, 0.35f, 1.0f };
		colors[ImGuiCol_TabUnfocused]       = colors[ImGuiCol_TitleBg];
		colors[ImGuiCol_TabUnfocusedActive] = colors[ImGuiCol_TitleBg];

		colors[ImGuiCol_FrameBg]            = ImVec4{ 0.15f, 0.15f, 0.15f, 1.0f };
		colors[ImGuiCol_FrameBgHovered]     = ImVec4{ 0.19f, 0.19f, 0.19f, 1.0f };
		colors[ImGuiCol_FrameBgActive]      = ImVec4{ 0.18f, 0.18f, 0.18f, 1.0f };

		colors[ImGuiCol_Button]             = ImVec4{ 0.2f, 0.2f, 0.2f, 1.0f };
		colors[ImGuiCol_ButtonHovered]      = ImVec4{ 0.25f, 0.25f, 0.25f, 1.0f };
		colors[ImGuiCol_ButtonActive]       = ImVec4{ 0.175f, 0.175f, 0.175f, 1.0f };
		colors[ImGuiCol_CheckMark]          = ImVec4{ 0.75f, 0.75f, 0.75f, 1.0f };
		colors[ImGuiCol_SliderGrab]         = ImVec4{ 0.75f, 0.75f, 0.75f, 1.0f };
		colors[ImGuiCol_SliderGrabActive]   = ImVec4{ 0.8f, 0.8f, 0.8f, 1.0f };

		colors[ImGuiCol_ResizeGrip]        = colors[ImGuiCol_ButtonActive];
		colors[ImGuiCol_ResizeGripActive]  = colors[ImGuiCol_ButtonActive];
		colors[ImGuiCol_ResizeGripHovered] = colors[ImGuiCol_ButtonActive];

		colors[ImGuiCol_DragDropTarget]    = colors[ImGuiCol_ButtonActive];
	}

	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	ImGui::GetIO().ConfigWindowsMoveFromTitleBarOnly = true;

	ImGui_ImplGlfw_InitForVulkan(swapchain.mWindow.GetWindow(), true);

	ImGui_ImplVulkan_InitInfo init_info = {};
	init_info.Instance = **swapchain.mDevice.mInstance;
	init_info.PhysicalDevice = *swapchain.mDevice.GetPhysicalDevice();
	init_info.Device = **swapchain.mDevice;
	init_info.QueueFamily = mQueueFamily;
	init_info.Queue = queue;
	init_info.PipelineCache  = *swapchain.mDevice.GetPipelineCache();
	init_info.DescriptorPool = **mImGuiDescriptorPool;
	init_info.Subpass = 0;
	init_info.MinImageCount = std::max(swapchain.GetMinImageCount(), 2u);
	init_info.ImageCount    = std::max(swapchain.GetImageCount(), 2u);
	init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
	init_info.Allocator = nullptr;
	//init_info.CheckVkResultFn = check_vk_result;
	ImGui_ImplVulkan_Init(&init_info, *mRenderPass);

	float scale = 1;
	if (auto arg = swapchain.mDevice.mInstance.GetOption("gui-scale")) {
		scale = std::stof(*arg);
		ImGui::GetStyle().ScaleAllSizes(scale);
		ImGui::GetStyle().IndentSpacing /= scale;
	}

	// Upload Fonts

	if (auto arg = swapchain.mDevice.mInstance.GetOption("font")) {
		ImGui::GetIO().Fonts->AddFontFromFileTTF(arg->c_str(), scale*16.f);
		mHeaderFont = ImGui::GetIO().Fonts->AddFontFromFileTTF(arg->c_str(), scale*20.f);
	} else
		mHeaderFont = ImGui::GetFont();

	std::shared_ptr<CommandBuffer> commandBufferPtr = std::make_shared<CommandBuffer>(swapchain.mDevice, "ImGui CreateFontsTexture", swapchain.mDevice.FindQueueFamily(vk::QueueFlagBits::eGraphics));
	CommandBuffer& commandBuffer = *commandBufferPtr;
	commandBuffer.Reset();

	ImGui_ImplVulkan_CreateFontsTexture(**commandBuffer);

	commandBuffer.Submit(*swapchain.mDevice->getQueue(commandBuffer.GetQueueFamily(),0));
	if (swapchain.mDevice->waitForFences(**commandBuffer.GetCompletionFence(), true, ~0ull) != vk::Result::eSuccess)
		throw std::runtime_error("waitForFences failed");

	ImGui_ImplVulkan_DestroyFontUploadObjects();
}
void Gui::Destroy() {
	if (*mRenderPass) {
		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
		mRenderPass.clear();
		mFramebuffers.clear();
		mImGuiDescriptorPool.reset();
		mFrameTextures.clear();
		mTextureIDs.clear();
	}
}

void Gui::NewFrame() {
	ImGui_ImplGlfw_NewFrame();
	ImGui_ImplVulkan_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();
}

void Gui::Render(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
	ImGui::Render();
	ImDrawData* drawData = ImGui::GetDrawData();
	if (drawData->DisplaySize.x <= 0.0f || drawData->DisplaySize.y <= 0.0f) return;

	const vk::Extent2D extent((uint32_t)drawData->DisplaySize.x, (uint32_t)drawData->DisplaySize.y);

	// create framebuffer

	auto it = mFramebuffers.find(**renderTarget.GetImage());
	if (it == mFramebuffers.end()) {
		vk::raii::Framebuffer fb(*commandBuffer.mDevice, vk::FramebufferCreateInfo({}, *mRenderPass, *renderTarget, extent.width, extent.height, 1));
		it = mFramebuffers.emplace(**renderTarget.GetImage(), std::move(fb)).first;
	}

	for (const Image::View& v : mFrameTextures)
		commandBuffer.Barrier(v, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eFragmentShader, vk::AccessFlagBits::eShaderRead);
	mFrameTextures.clear();

	// render gui

	commandBuffer.Barrier(renderTarget, vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentRead);
	commandBuffer.FlushBarriers();
	commandBuffer->beginRenderPass(
		vk::RenderPassBeginInfo(*mRenderPass, *it->second, vk::Rect2D({0,0}, extent)),
		vk::SubpassContents::eInline);

	// Record dear imgui primitives into command buffer
	ImGui_ImplVulkan_RenderDrawData(drawData, **commandBuffer);

	// Submit command buffer
	commandBuffer->endRenderPass();
	renderTarget.SetSubresourceState(vk::ImageLayout::eColorAttachmentOptimal, vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::AccessFlagBits::eColorAttachmentWrite);
}

}
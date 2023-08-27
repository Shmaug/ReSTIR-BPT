#include "Swapchain.hpp"
#include "Profiler.hpp"

#include <imgui/imgui.h>

namespace ptvk {

Swapchain::Swapchain(Device& device, Window& window, const uint32_t minImages, const vk::ImageUsageFlags imageUsage, const vk::SurfaceFormatKHR surfaceFormat, const vk::PresentModeKHR presentMode)
	: mDevice(device), mSwapchain(nullptr), mWindow(window), mMinImageCount(minImages), mUsage(imageUsage) {
	// select the format of the swapchain
	const auto formats = mDevice.GetPhysicalDevice().getSurfaceFormatsKHR(*mWindow.GetSurface());
	mSurfaceFormat = formats.front();
	for (const vk::SurfaceFormatKHR& format : formats)
		if (format == surfaceFormat) {
			mSurfaceFormat = format;
			break;
		}

	mPresentMode = vk::PresentModeKHR::eFifo; // required to be supported
	for (const vk::PresentModeKHR& mode : mDevice.GetPhysicalDevice().getSurfacePresentModesKHR(*mWindow.GetSurface()))
		if (mode == presentMode) {
			mPresentMode = mode;
			break;
		}

	Create();
}

bool Swapchain::Create() {
	ProfilerScope ps("Swapchain::Create");

	// get the size of the swapchain
	const vk::SurfaceCapabilitiesKHR capabilities = mDevice.GetPhysicalDevice().getSurfaceCapabilitiesKHR(*mWindow.GetSurface());
	mExtent = capabilities.currentExtent;
	if (mExtent.width == 0 || mExtent.height == 0 || mExtent.width > mDevice.GetLimits().maxImageDimension2D || mExtent.height > mDevice.GetLimits().maxImageDimension2D)
		return false;

	mMinImageCount = max(mMinImageCount, capabilities.minImageCount);

	vk::raii::SwapchainKHR oldSwapchain = std::move( mSwapchain );

	vk::SwapchainCreateInfoKHR info = {};
	info.surface = *mWindow.GetSurface();
	if (*oldSwapchain) info.oldSwapchain = *oldSwapchain;
	info.minImageCount = mMinImageCount;
	info.imageFormat = mSurfaceFormat.format;
	info.imageColorSpace = mSurfaceFormat.colorSpace;
	info.imageExtent = mExtent;
	info.imageArrayLayers = 1;
	info.imageUsage = mUsage;
	info.imageSharingMode = vk::SharingMode::eExclusive;
	info.preTransform = capabilities.currentTransform;
	info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
	info.presentMode = mPresentMode;
	info.clipped = VK_FALSE;
	mSwapchain = std::move( vk::raii::SwapchainKHR(*mDevice, info) );

	oldSwapchain = nullptr;

	const std::vector<VkImage> images = mSwapchain.getImages();

	mDevice.mFramesInFlight = images.size();

	mImages.resize(images.size());
	mImageAvailableSemaphores.resize(images.size());
	for (uint32_t i = 0; i < mImages.size(); i++) {
		ImageInfo m = {};
		m.mFormat = mSurfaceFormat.format;
		m.mExtent = vk::Extent3D(mExtent, 1);
		m.mUsage = info.imageUsage;
		m.mQueueFamilies = mWindow.FindSupportedQueueFamilies(mDevice.GetPhysicalDevice());
		mImages[i] = std::make_shared<Image>(mDevice, "SwapchainImage " + std::to_string(i), images[i], m);
		mImageAvailableSemaphores[i] = std::make_shared<vk::raii::Semaphore>(*mDevice, vk::SemaphoreCreateInfo{});
	}

	mImageIndex = 0;
	mImageAvailableSemaphoreIndex = 0;
	mDirty = false;
	return true;
}

bool Swapchain::AcquireImage(const std::chrono::nanoseconds& timeout) {
	ProfilerScope ps("Swapchain::AcquireImage");

	const uint32_t semaphore = (mImageAvailableSemaphoreIndex + 1) % mImageAvailableSemaphores.size();

	vk::Result result;
	std::tie(result, mImageIndex) = mSwapchain.acquireNextImage(timeout.count(), **mImageAvailableSemaphores[semaphore]);

	if (result == vk::Result::eNotReady || result == vk::Result::eTimeout)
		return false;
	else if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eErrorSurfaceLostKHR) {
		mDirty = true;
		return false;
	} else if (result != vk::Result::eSuccess)
		throw std::runtime_error("Failed to acquire image");

	mImageAvailableSemaphoreIndex = semaphore;

	return true;
}

void Swapchain::Present(const vk::raii::Queue queue, const vk::ArrayProxy<const vk::Semaphore>& waitSemaphores) {
	ProfilerScope ps("Swapchain::present");

	const vk::Result result = queue.presentKHR(vk::PresentInfoKHR(waitSemaphores, *mSwapchain, mImageIndex));
	if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eErrorSurfaceLostKHR)
		mDirty = true;
	mPresentCount++;
}

void Swapchain::OnInspectorGui() {
	vk::SurfaceCapabilitiesKHR capabilities = mDevice.GetPhysicalDevice().getSurfaceCapabilitiesKHR(*mWindow.GetSurface());
	ImGui::SetNextItemWidth(40);
	if (ImGui::DragScalar("Min image count", ImGuiDataType_U32, &mMinImageCount, 1, &capabilities.minImageCount, &capabilities.maxImageCount))
		mDirty = true;
	ImGui::LabelText("Image count", "%u", GetImageCount());

	if (ImGui::BeginCombo("Present mode", to_string(mPresentMode).c_str())) {
		for (auto mode : mDevice.GetPhysicalDevice().getSurfacePresentModesKHR(*mWindow.GetSurface()))
			if (ImGui::Selectable(vk::to_string(mode).c_str(), mPresentMode == mode)) {
				mPresentMode = mode;
				mDirty = true;
			}
		ImGui::EndCombo();
	}

	if (ImGui::CollapsingHeader("Usage flags")) {
		for (uint32_t i = 0; i < 8; i++)
			if (ImGui::CheckboxFlags(to_string((vk::ImageUsageFlagBits)(1 << i)).c_str(), reinterpret_cast<unsigned int*>(&mUsage), 1 << i))
				mDirty = true;
	}

	auto fmt_to_str = [](vk::SurfaceFormatKHR f) { return vk::to_string(f.format) + ", " + vk::to_string(f.colorSpace); };
	if (ImGui::BeginCombo("Surface format", fmt_to_str(mSurfaceFormat).c_str())) {
		for (auto format : mDevice.GetPhysicalDevice().getSurfaceFormatsKHR(*mWindow.GetSurface())) {
			vk::ImageFormatProperties p;
			vk::Result e = (*mDevice.GetPhysicalDevice()).getImageFormatProperties(format.format, vk::ImageType::e2D, vk::ImageTiling::eOptimal, mUsage, {}, &p);
			if (e == vk::Result::eSuccess) {
				if (ImGui::Selectable(fmt_to_str(format).c_str(), mSurfaceFormat == format)) {
					mSurfaceFormat = format;
					mDirty = true;
				}
			}
		}
		ImGui::EndCombo();
	}
}

}
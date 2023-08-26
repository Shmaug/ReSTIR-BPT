#pragma once

#include "Image.hpp"
#include "Window.hpp"

namespace ptvk {

class Swapchain {
public:
	Device& mDevice;
	Window& mWindow;

	Swapchain(Device& device, Window& window,
		const uint32_t minImages = 2,
		const vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
		const vk::SurfaceFormatKHR preferredSurfaceFormat = vk::SurfaceFormatKHR(vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear),
		const vk::PresentModeKHR presentMode = vk::PresentModeKHR::eImmediate);

	inline       vk::raii::SwapchainKHR& operator*()        { return mSwapchain; }
	inline const vk::raii::SwapchainKHR& operator*() const  { return mSwapchain; }
	inline       vk::raii::SwapchainKHR* operator->()       { return &mSwapchain; }
	inline const vk::raii::SwapchainKHR* operator->() const { return &mSwapchain; }

	inline vk::Extent2D GetExtent() const { return mExtent; }
	inline vk::SurfaceFormatKHR GetFormat() const { return mSurfaceFormat; }
	inline vk::PresentModeKHR GetPresentMode() const { return mPresentMode; }
	inline const std::shared_ptr<vk::raii::Semaphore>& GetImageAvailableSemaphore() const { return mImageAvailableSemaphores[mImageAvailableSemaphoreIndex]; }

	inline uint32_t GetMinImageCount() const { return mMinImageCount; }
	inline uint32_t GetImageCount() const { return (uint32_t)mImages.size(); }
	inline uint32_t GetImageIndex() const { return mImageIndex; }
	inline const std::shared_ptr<Image>& GetImage() const { return mImages[mImageIndex]; }
	inline const std::shared_ptr<Image>& GetImage(uint32_t i) const { return mImages[i]; }

	inline bool IsDirty() const { return mDirty || mWindow.GetExtent() != GetExtent(); }
	bool Create();

	bool AcquireImage(const std::chrono::nanoseconds& timeout = std::chrono::nanoseconds(0));
	void Present(const vk::raii::Queue queue, const vk::ArrayProxy<const vk::Semaphore>& waitSemaphores = {});

	// Number of times present has been called
	inline size_t PresentCount() const { return mPresentCount; }

	void OnInspectorGui();

private:
	vk::raii::SwapchainKHR mSwapchain;
	vk::Extent2D mExtent;
	std::vector<std::shared_ptr<Image>> mImages;
	std::vector<std::shared_ptr<vk::raii::Semaphore>> mImageAvailableSemaphores;
	uint32_t mMinImageCount;
	uint32_t mImageIndex;
	uint32_t mImageAvailableSemaphoreIndex;
	vk::ImageUsageFlags mUsage;

	vk::SurfaceFormatKHR mSurfaceFormat;
	vk::PresentModeKHR mPresentMode;
	size_t mPresentCount = 0;
	bool mDirty = false;
};

}
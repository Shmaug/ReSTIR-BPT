#pragma once

#include "Utils.hpp"

struct GLFWwindow;

namespace ptvk {

class Instance;

class Window {
public:
	Instance& mInstance;

	Window(Instance& instance, const std::string& title, const vk::Extent2D& extent);
	~Window();

	inline GLFWwindow* GetWindow() const { return mWindow; }
	inline const std::string& GetTitle() const { return mTitle; }
	inline const vk::raii::SurfaceKHR& GetSurface() const { return mSurface; }
	inline const vk::Extent2D& GetExtent() const {return mClientExtent; }

	std::tuple<vk::raii::PhysicalDevice, uint32_t> FindSupportedDevice() const;
	std::vector<uint32_t> FindSupportedQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice) const;

	bool IsOpen() const;

	void Resize(const vk::Extent2D& extent) const;

	void SetFullscreen(const bool fs);
	inline bool IsFullscreen() const { return mFullscreen; }

	inline bool WantsRepaint() { return mRepaint; }

	std::unordered_set<std::string>& GetDroppedFiles() { return mDroppedFiles; }

	void OnInspectorGui();

private:
	GLFWwindow* mWindow;
	vk::raii::SurfaceKHR mSurface;

	std::string mTitle;
	vk::Extent2D mClientExtent;
	vk::Rect2D mRestoreRect;

	bool mFullscreen = false;
	bool mRecreateSwapchain = false;
	bool mRepaint = false;

	std::unordered_set<std::string> mDroppedFiles;

	void CreateSwapchain();

	static void WindowSizeCallback(GLFWwindow* window, int width, int height);
	static void DropCallback(GLFWwindow* window, int count, const char** paths);
};

}
#pragma once

#include <Core/Swapchain.hpp>
#include <Core/CommandBuffer.hpp>
#include <Core/Gui.hpp>
#include "Common/Common.h"

namespace ptvk {

class Renderer;
class Scene;
class Camera;
class FlyCamera;

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

	int mProfilerHistoryCount = 8;

	std::unique_ptr<Scene> mScene;
	std::shared_ptr<Camera> mCamera;
	std::shared_ptr<FlyCamera> mFlyCamera;

	float mRenderScale = 1;
	std::unique_ptr<Renderer> mRenderer;

	std::chrono::steady_clock::time_point mLastUpdate;

	Image::View mTmpImage;

	inline static float4 mViewportRect = float4(0,0,0,0);
	inline static bool mIsViewportFocused = true;

	App(const std::vector<std::string>& args);
	~App();

	bool CreateSwapchain();

	void Update();

	vk::Semaphore Render(const Image::View& renderTarget);

	void Run();
};

}

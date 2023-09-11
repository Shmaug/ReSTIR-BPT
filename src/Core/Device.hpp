#pragma once

#include <thread>
#include <shared_mutex>

#include <vk_mem_alloc.h>

#include "Instance.hpp"

namespace ptvk {

inline uint32_t FindQueueFamily(vk::raii::PhysicalDevice& physicalDevice, const vk::QueueFlags flags = vk::QueueFlagBits::eGraphics|vk::QueueFlagBits::eCompute|vk::QueueFlagBits::eTransfer) {
	const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();
	for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
		if (queueFamilyProperties[i].queueFlags | flags)
			return i;
	}
	return -1;
}

class Device {
public:
	Instance& mInstance;

	Device(Instance& instance, vk::raii::PhysicalDevice physicalDevice);
	~Device();

	inline       vk::raii::Device& operator*()        { return mDevice; }
	inline const vk::raii::Device& operator*() const  { return mDevice; }
	inline       vk::raii::Device* operator->()       { return &mDevice; }
	inline const vk::raii::Device* operator->() const { return &mDevice; }

	inline vk::raii::PhysicalDevice GetPhysicalDevice() const { return mPhysicalDevice; }
	inline vk::raii::PipelineCache& GetPipelineCache() { return mPipelineCache; }
	inline VmaAllocator GetAllocator() const { return mAllocator; }

	inline const std::unordered_set<std::string>& GetEnabledExtensions() const { return mExtensions; }

	inline const vk::PhysicalDeviceLimits& GetLimits() const { return mLimits; }
	inline const vk::PhysicalDeviceFeatures& GetFeatures() const { return mFeatures; }
	inline const vk::PhysicalDeviceVulkan13Features&                 GetVulkan13Features() const              { return std::get<vk::PhysicalDeviceVulkan13Features>(mFeatureChain); }
	inline const vk::PhysicalDeviceDescriptorIndexingFeatures&       GetDescriptorIndexingFeatures() const    { return std::get<vk::PhysicalDeviceDescriptorIndexingFeatures>(mFeatureChain); }
	inline const vk::PhysicalDeviceBufferDeviceAddressFeatures&      GetBufferDeviceAddressFeatures() const   { return std::get<vk::PhysicalDeviceBufferDeviceAddressFeatures>(mFeatureChain); }
	inline const vk::PhysicalDeviceAccelerationStructureFeaturesKHR& GetAccelerationStructureFeatures() const { return std::get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>(mFeatureChain); }
	inline const vk::PhysicalDeviceRayTracingPipelineFeaturesKHR&    GetRayTracingPipelineFeatures() const    { return std::get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>(mFeatureChain); }
	inline const vk::PhysicalDeviceRayQueryFeaturesKHR&              GetRayQueryFeatures() const              { return std::get<vk::PhysicalDeviceRayQueryFeaturesKHR>(mFeatureChain); }

	template<typename T> requires(std::convertible_to<decltype(T::objectType), vk::ObjectType>)
	inline void SetDebugName(const T& object, const std::string& name) {
		vk::DebugUtilsObjectNameInfoEXT info = {};
		info.objectHandle = *reinterpret_cast<const uint64_t*>(&object);
		info.objectType = T::objectType;
		info.pObjectName = name.c_str();
		mDevice.setDebugUtilsObjectNameEXT(info);
	}

	vk::raii::CommandPool& GetCommandPool(const uint32_t queueFamily);

	const std::shared_ptr<vk::raii::DescriptorPool>& AllocateDescriptorPool();
	const std::shared_ptr<vk::raii::DescriptorPool>& GetDescriptorPool();

	inline uint32_t FindQueueFamily(const vk::QueueFlags flags = vk::QueueFlagBits::eGraphics|vk::QueueFlagBits::eCompute|vk::QueueFlagBits::eTransfer) {
		return ptvk::FindQueueFamily(mPhysicalDevice, flags);
	}

	inline size_t GetFrameIndex() const { return mFrameIndex; }
	inline void IncrementFrameIndex() { mFrameIndex++; }
	inline size_t GetFramesInFlight() const { return mFramesInFlight; }

	void OnInspectorGui();

private:
	vk::raii::Device mDevice;
 	vk::raii::PhysicalDevice mPhysicalDevice;
	vk::raii::PipelineCache mPipelineCache;

	std::unordered_set<std::string> mExtensions;

	mutable std::shared_mutex mCommandPoolMutex;
	std::unordered_map<std::thread::id, std::unordered_map<uint32_t, vk::raii::CommandPool>> mCommandPools;

	mutable std::shared_mutex mDescriptorPoolMutex;
	std::stack<std::shared_ptr<vk::raii::DescriptorPool>> mDescriptorPools;

	VmaAllocator mAllocator;

	size_t mFrameIndex;
	size_t mFramesInFlight; // assigned by Swapchain
	friend class Swapchain;

	vk::PhysicalDeviceFeatures mFeatures;
	vk::StructureChain<
		vk::DeviceCreateInfo,
		vk::PhysicalDeviceVulkan13Features,
		vk::PhysicalDeviceDescriptorIndexingFeatures,
		vk::PhysicalDeviceBufferDeviceAddressFeatures,
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
		vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
		vk::PhysicalDeviceRayQueryFeaturesKHR,
		vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT
	> mFeatureChain;
	vk::PhysicalDeviceLimits mLimits;
};

}
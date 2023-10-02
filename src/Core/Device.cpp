#define VMA_IMPLEMENTATION
#include "Device.hpp"
#include "CommandBuffer.hpp"
#include "Profiler.hpp"

#include <imgui/imgui.h>

namespace ptvk {

Device::Device(Instance& instance, vk::raii::PhysicalDevice physicalDevice) :
	mInstance(instance),
	mPhysicalDevice(physicalDevice),
	mDevice(nullptr),
	mPipelineCache(nullptr),
	mFrameIndex(0),
	mFramesInFlight(1) {
	for (const std::string& s : mInstance.GetOptions("device-extension"))
		mExtensions.emplace(s);
	mExtensions.emplace(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	// load extensions required by other extensions
	if (mExtensions.contains(VK_KHR_RAY_QUERY_EXTENSION_NAME))
		mExtensions.emplace(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
	if (mExtensions.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME))
		mExtensions.emplace(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

	// configure device features
	{
		mFeatures.fillModeNonSolid = true;
		mFeatures.samplerAnisotropy = true;
		mFeatures.shaderImageGatherExtended = true;
		mFeatures.shaderStorageImageExtendedFormats = true;
		mFeatures.wideLines = true;
		mFeatures.largePoints = true;
		mFeatures.sampleRateShading = true;
		mFeatures.shaderInt16 = true;
		//mFeatures.shaderFloat64 = true; // needed by slang?
		mFeatures.shaderStorageBufferArrayDynamicIndexing = true;
		mFeatures.shaderSampledImageArrayDynamicIndexing = true;
		mFeatures.shaderStorageImageArrayDynamicIndexing = true;

		vk::PhysicalDeviceVulkan12Features& vk12features = std::get<vk::PhysicalDeviceVulkan12Features>(mFeatureChain);
		vk12features.shaderStorageBufferArrayNonUniformIndexing = true;
		vk12features.shaderSampledImageArrayNonUniformIndexing = true;
		vk12features.shaderStorageImageArrayNonUniformIndexing = true;
		vk12features.descriptorBindingPartiallyBound = true;
		vk12features.shaderInt8 = true;
		vk12features.storageBuffer8BitAccess = true;
		vk12features.shaderFloat16 = true;
		vk12features.bufferDeviceAddress = mExtensions.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);

		vk::PhysicalDeviceVulkan13Features& vk13features = std::get<vk::PhysicalDeviceVulkan13Features>(mFeatureChain);
		vk13features.dynamicRendering = true;
		vk13features.synchronization2 = true;

		vk::PhysicalDevice16BitStorageFeatures& storageFeatures = std::get<vk::PhysicalDevice16BitStorageFeatures>(mFeatureChain);
		storageFeatures.storageBuffer16BitAccess = true;

		std::get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>(mFeatureChain).accelerationStructure = mExtensions.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);

		auto& rtfeatures = std::get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>(mFeatureChain);
		rtfeatures.rayTracingPipeline = mExtensions.contains(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
		rtfeatures.rayTraversalPrimitiveCulling = rtfeatures.rayTracingPipeline;

		std::get<vk::PhysicalDeviceRayQueryFeaturesKHR>(mFeatureChain).rayQuery = mExtensions.contains(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	}

	// Queue create infos

	std::vector<vk::QueueFamilyProperties> queueFamilyProperties = mPhysicalDevice.getQueueFamilyProperties();
	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	float queuePriority = 1.0f;
	for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
		if (queueFamilyProperties[i].queueFlags & (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer)) {
			queueCreateInfos.emplace_back(vk::DeviceQueueCreateInfo({}, i, 1, &queuePriority));
		}
	}

	// Create logical device

	auto deviceExts       = mExtensions                        | std::views::transform([](const std::string& s) -> const char* { return s.c_str(); }) | std::ranges::to<std::vector<const char*>>();
	auto validationLayers = instance.EnabledValidationLayers() | std::views::transform([](const std::string& s) -> const char* { return s.c_str(); }) | std::ranges::to<std::vector<const char*>>();

	auto& createInfo = mFeatureChain.get<vk::DeviceCreateInfo>();
	createInfo.setQueueCreateInfos(queueCreateInfos);
	createInfo.setPEnabledLayerNames(validationLayers);
	createInfo.setPEnabledExtensionNames(deviceExts);
	createInfo.setPEnabledFeatures(&mFeatures);
	mDevice = mPhysicalDevice.createDevice(createInfo);

	const vk::PhysicalDeviceProperties properties = mPhysicalDevice.getProperties();
	mLimits = properties.limits;
	SetDebugName(*mDevice, "[" + std::to_string(properties.deviceID) + "]: " + properties.deviceName.data());

	// Load pipeline cache

	std::vector<uint8_t> cacheData;
	std::string tmp;
	vk::PipelineCacheCreateInfo cacheInfo = {};
	if (!mInstance.GetOption("no-pipeline-cache")) {
		try {
			cacheData = ReadFile<std::vector<uint8_t>>(std::filesystem::temp_directory_path() / "stm2_pcache");
			if (!cacheData.empty()) {
				cacheInfo.pInitialData = cacheData.data();
				cacheInfo.initialDataSize = cacheData.size();
				std::cout << "Read pipeline cache (" << std::fixed << std::showpoint << std::setprecision(2) << cacheData.size()/1024.f << "KiB)" << std::endl;
			}
		} catch (std::exception& e) {
			std::cerr << "Warning: Failed to read pipeline cache: " << e.what() << std::endl;
		}
	}
	mPipelineCache = vk::raii::PipelineCache(mDevice, cacheInfo);

	// Create VMA allocator

	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = *mPhysicalDevice;
	allocatorInfo.device = *mDevice;
	allocatorInfo.instance = **mInstance;
	allocatorInfo.vulkanApiVersion = mInstance.GetVulkanVersion();
	allocatorInfo.preferredLargeHeapBlockSize = 1024 * 1024;
	allocatorInfo.flags = 0;
	if (mExtensions.contains(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
		allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	if (GetVulkan12Features().bufferDeviceAddress)
		allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	vmaCreateAllocator(&allocatorInfo, &mAllocator);
}
Device::~Device() {
	if (!mInstance.GetOption("no-pipeline-cache")) {
		try {
			const std::vector<uint8_t> cacheData = mPipelineCache.getData();
			if (!cacheData.empty())
				WriteFile(std::filesystem::temp_directory_path() / "stm2_pcache", cacheData);
		} catch (std::exception& e) {
			std::cerr << "Warning: Failed to write pipeline cache: " << e.what() << std::endl;
		}
	}
	vmaDestroyAllocator(mAllocator);
}

vk::raii::CommandPool& Device::GetCommandPool(const uint32_t queueFamily) {
	std::unique_lock l(mCommandPoolMutex);
	auto& pools = mCommandPools[std::this_thread::get_id()];
	if (auto it = pools.find(queueFamily); it != pools.end())
		return it->second;
	return pools.emplace(queueFamily, vk::raii::CommandPool(mDevice, vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily))).first->second;
}

const std::shared_ptr<vk::raii::DescriptorPool>& Device::AllocateDescriptorPool() {
	std::vector<vk::DescriptorPoolSize> poolSizes {
		vk::DescriptorPoolSize(vk::DescriptorType::eSampler,              min(16384u, mLimits.maxDescriptorSetSamplers)),
		vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, min(16384u, mLimits.maxDescriptorSetSampledImages)),
		vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment,      min(16384u, mLimits.maxDescriptorSetInputAttachments)),
		vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage,         min(16384u, mLimits.maxDescriptorSetSampledImages)),
		vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage,         min(16384u, mLimits.maxDescriptorSetStorageImages)),
		vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,        min(16384u, mLimits.maxDescriptorSetUniformBuffers)),
		vk::DescriptorPoolSize(vk::DescriptorType::eUniformBufferDynamic, min(16384u, mLimits.maxDescriptorSetUniformBuffersDynamic)),
		vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,        min(16384u, mLimits.maxDescriptorSetStorageBuffers)),
		vk::DescriptorPoolSize(vk::DescriptorType::eStorageBufferDynamic, min(16384u, mLimits.maxDescriptorSetStorageBuffersDynamic))
	};
	std::unique_lock l(mDescriptorPoolMutex);
	mDescriptorPools.push(std::make_shared<vk::raii::DescriptorPool>(mDevice, vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 8192, poolSizes)));
	return mDescriptorPools.top();
}
const std::shared_ptr<vk::raii::DescriptorPool>& Device::GetDescriptorPool() {
	std::unique_lock l(mDescriptorPoolMutex);
	if (mDescriptorPools.empty()) {
		l.unlock();
		return AllocateDescriptorPool();
	}
	return mDescriptorPools.top();
}

void Device::OnInspectorGui() {
	if (ImGui::CollapsingHeader("Heap budgets")) {
		const bool memoryBudgetExt = mExtensions.contains(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
		vk::StructureChain<vk::PhysicalDeviceMemoryProperties2, vk::PhysicalDeviceMemoryBudgetPropertiesEXT> structureChain;
		if (memoryBudgetExt) {
			const auto tmp = mPhysicalDevice.getMemoryProperties2<vk::PhysicalDeviceMemoryProperties2, vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();
			structureChain = tmp;
		} else {
			structureChain.get<vk::PhysicalDeviceMemoryProperties2>() = mPhysicalDevice.getMemoryProperties2();
		}

		const vk::PhysicalDeviceMemoryProperties2& properties = structureChain.get<vk::PhysicalDeviceMemoryProperties2>();
		const vk::PhysicalDeviceMemoryBudgetPropertiesEXT& budgetProperties = structureChain.get<vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();

		VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
		vmaGetHeapBudgets(mAllocator, budgets);

		for (uint32_t heapIndex = 0; heapIndex < properties.memoryProperties.memoryHeapCount; heapIndex++) {
			const char* isDeviceLocalStr = (properties.memoryProperties.memoryHeaps[heapIndex].flags & vk::MemoryHeapFlagBits::eDeviceLocal) ? " (device local)" : "";

			if (memoryBudgetExt) {
				const auto[usage, usageUnit]   = FormatBytes(budgetProperties.heapUsage[heapIndex]);
				const auto[budget, budgetUnit] = FormatBytes(budgetProperties.heapBudget[heapIndex]);
				ImGui::Text("Heap %u%s (%llu %s / %llu %s)", heapIndex, isDeviceLocalStr, usage, usageUnit, budget, budgetUnit);
			} else
				ImGui::Text("Heap %u%s", heapIndex, isDeviceLocalStr);
			ImGui::Indent();

			// VMA stats
			{
				const auto[usage, usageUnit]   = FormatBytes(budgets[heapIndex].usage);
				const auto[budget, budgetUnit] = FormatBytes(budgets[heapIndex].budget);
				ImGui::Text("%llu %s used, %llu %s budgeted", usage, usageUnit, budget, budgetUnit);

				const auto[allocationBytes, allocationBytesUnit] = FormatBytes(budgets[heapIndex].statistics.allocationBytes);
				ImGui::Text("%u allocations\t(%llu %s)", budgets[heapIndex].statistics.allocationCount, allocationBytes, allocationBytesUnit);

				const auto[blockBytes, blockBytesUnit] = FormatBytes(budgets[heapIndex].statistics.blockBytes);
				ImGui::Text("%u memory blocks\t(%llu %s)", budgets[heapIndex].statistics.blockCount, blockBytes, blockBytesUnit);
			}

			ImGui::Unindent();
		}
	}
}

}
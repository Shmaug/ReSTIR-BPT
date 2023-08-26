#pragma once

#include "Utils.hpp"

namespace ptvk {

class Instance {
public:
	static bool sDisableDebugCallback;

	Instance(const std::vector<std::string>& args);

	inline       vk::raii::Instance& operator*()        { return mInstance; }
	inline const vk::raii::Instance& operator*() const  { return mInstance; }
	inline       vk::raii::Instance* operator->()       { return &mInstance; }
	inline const vk::raii::Instance* operator->() const { return &mInstance; }

	inline std::optional<std::string> GetOption(const std::string& name) const {
		auto[first,last] = mOptions.equal_range(name);
		if (first == last) return std::nullopt;
		return first->second;
	}
	inline auto GetOptions(const std::string& name) const {
		auto[first,last] = mOptions.equal_range(name);
		return std::ranges::subrange(first,last) | std::views::values;
	}

	inline vk::raii::Context& GetVulkanContext() { return mContext; }
	inline const vk::raii::Context& GetVulkanContext() const { return mContext; }
	inline uint32_t GetVulkanVersion() const { return mVulkanApiVersion; }

	inline const std::unordered_set<std::string>& EnabledValidationLayers() const { return mValidationLayers; }

	void OnInspectorGui();

private:
	vk::raii::Context mContext;
	vk::raii::Instance mInstance;

	std::unordered_set<std::string> mValidationLayers;

	std::vector<std::string> mCommandLine;
	std::unordered_multimap<std::string, std::string> mOptions;
	uint32_t mVulkanApiVersion;

	vk::raii::DebugUtilsMessengerEXT mDebugMessenger;
};

}
#pragma once

#include <queue>

#include "Device.hpp"

namespace ptvk {

// Tracks in-flight resources
template<typename ResourceType>
class ResourceQueue {
public:
	inline std::shared_ptr<ResourceType> Get(Device& device, size_t inFlight = 0) {
		std::shared_ptr<ResourceType> resource;

		if (!mResources.empty()) {
			const auto&[frame, ptr] = mResources.front();
			if (device.GetFrameIndex() - frame >= std::max(inFlight, device.GetFramesInFlight())) {
				resource = ptr;
				mResources.pop();
			}
		}

		if (!resource) resource = std::make_shared<ResourceType>();

		mResources.push(std::make_pair(device.GetFrameIndex(), resource));

		return resource;
	}

	inline void Clear() {
		mResources.clear();
	}

private:
	std::queue<std::pair<size_t, std::shared_ptr<ResourceType>>> mResources;
};

}
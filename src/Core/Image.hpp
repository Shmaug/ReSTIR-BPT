#pragma once

#include "Buffer.hpp"

namespace ptvk {

struct ImageInfo {
	vk::ImageCreateFlags mCreateFlags = {};
	vk::ImageType mType = vk::ImageType::e2D;
	vk::Format mFormat;
	vk::Extent3D mExtent;
	uint32_t mLevels = 1;
	uint32_t mLayers = 1;
	vk::SampleCountFlagBits mSamples = vk::SampleCountFlagBits::e1;
	vk::ImageUsageFlags mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	vk::ImageTiling mTiling = vk::ImageTiling::eOptimal;
	vk::SharingMode mSharingMode = vk::SharingMode::eExclusive;
	std::vector<uint32_t> mQueueFamilies;
};

using PixelData = std::tuple<std::shared_ptr<Buffer>, vk::Format, vk::Extent3D>;
PixelData LoadImageFile(Device& device, const std::filesystem::path& filename, const bool srgb = true, int desiredChannels = 0);

class Image {
public:
	using SubresourceLayoutState = std::tuple<vk::ImageLayout, vk::PipelineStageFlags, vk::AccessFlags, uint32_t /*queueFamily*/>;

	class View {
	public:
		View() = default;
		View(const View&) = default;
		View(View&&) = default;
		inline View(const std::shared_ptr<Image>& image, const vk::ImageSubresourceRange& subresource = { vk::ImageAspectFlagBits::eColor, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS }, vk::ImageViewType viewType = vk::ImageViewType::e2D, const vk::ComponentMapping& componentMapping = {})
			: mImage(image), mSubresource(subresource), mType(viewType), mComponentMapping(componentMapping) {
			if (image) {
				if (IsDepthStencil(image->GetFormat()))
					mSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
				if (mSubresource.levelCount == VK_REMAINING_MIP_LEVELS)   mSubresource.levelCount = image->GetLevels();
				if (mSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS) mSubresource.layerCount = image->GetLayers();
				mView = image->GetView(mSubresource, mType, mComponentMapping);
			}
		}
		View& operator=(const View&) = default;
		View& operator=(View&& v) = default;

	inline       vk::ImageView& operator*()        { return mView; }
	inline const vk::ImageView& operator*() const  { return mView; }
	inline       vk::ImageView* operator->()       { return &mView; }
	inline const vk::ImageView* operator->() const { return &mView; }

		inline bool operator==(const View& rhs) const { return mView == rhs.mView; }
		inline bool operator!=(const View& rhs) const { return mView != rhs.mView; }

		inline operator bool() const { return mView; }

		inline const std::shared_ptr<Image>& GetImage() const { return mImage; }
		inline const vk::ImageSubresourceRange& GetSubresourceRange() const { return mSubresource; }
		inline vk::ImageSubresourceLayers GetSubresourceLayer(const uint32_t levelOffset = 0) const {
			return vk::ImageSubresourceLayers(mSubresource.aspectMask, mSubresource.baseMipLevel + levelOffset, mSubresource.baseArrayLayer, mSubresource.layerCount);
		}
		inline const vk::ImageViewType GetType() const { return mType; }
		inline const vk::ComponentMapping& GetComponentMapping() const { return mComponentMapping; }

		inline vk::Extent3D GetExtent(const uint32_t levelOffset = 0) const {
			return mImage->GetExtent(mSubresource.baseMipLevel + levelOffset);
		}

		inline void SetSubresourceState(const SubresourceLayoutState& newState) const {
			mImage->SetSubresourceState(mSubresource, newState);
		}
		inline void SetSubresourceState(const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, const uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) const {
			mImage->SetSubresourceState(mSubresource, { layout, stage, accessMask, queueFamily });
		}

	private:
		friend class Image;
		std::shared_ptr<Image> mImage;
		vk::ImageView mView;
		vk::ImageSubresourceRange mSubresource;
		vk::ImageViewType mType;
		vk::ComponentMapping mComponentMapping;
	};

	Device& mDevice;

	Image(Device& device, const std::string& name, const ImageInfo& info, const vk::MemoryPropertyFlags memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal);
	Image(Device& device, const std::string& name, const vk::Image image, const ImageInfo& info);
	~Image();

	inline       vk::Image& operator*()        { return mImage; }
	inline const vk::Image& operator*() const  { return mImage; }
	inline       vk::Image* operator->()       { return &mImage; }
	inline const vk::Image* operator->() const { return &mImage; }

	inline operator bool() const { return mImage; }

	inline ImageInfo GetInfo() const { return mInfo; }
	inline vk::ImageType GetType() const { return mInfo.mType; }
	inline vk::Format GetFormat() const { return mInfo.mFormat; }
	inline vk::Extent3D GetExtent(const uint32_t level = 0) const {
		uint32_t s = 1 << level;
		const vk::Extent3D& e = mInfo.mExtent;
		return vk::Extent3D(std::max(e.width / s, 1u), std::max(e.height / s, 1u), std::max(e.depth / s, 1u));
	}
	inline uint32_t GetLevels() const { return mInfo.mLevels; }
	inline uint32_t GetLayers() const { return mInfo.mLayers; }
	inline vk::SampleCountFlagBits GetSamples() const { return mInfo.mSamples; }
	inline vk::ImageUsageFlags GetUsage() const { return mInfo.mUsage; }
	inline vk::ImageTiling GetTiling() const { return mInfo.mTiling; }
	inline vk::SharingMode GetSharingMode() const { return mInfo.mSharingMode; }
	inline const std::vector<uint32_t>& GetQueueFamilies() const { return mInfo.mQueueFamilies; }

	const vk::ImageView GetView(const vk::ImageSubresourceRange& subresource, const vk::ImageViewType viewType = vk::ImageViewType::e2D, const vk::ComponentMapping& componentMapping = {});


	inline const SubresourceLayoutState& GetSubresourceState(const uint32_t arrayLayer, const uint32_t level) const {
		return mSubresourceStates[arrayLayer][level];
	}
	inline void SetSubresourceState(const vk::ImageSubresourceRange& subresource, const SubresourceLayoutState& newState) {
		const uint32_t maxLayer = std::min(GetLayers(), subresource.baseArrayLayer + subresource.layerCount);
		const uint32_t maxLevel = std::min(GetLevels(), subresource.baseMipLevel   + subresource.levelCount);
		for (uint32_t arrayLayer = subresource.baseArrayLayer; arrayLayer < maxLayer; arrayLayer++) {
			for (uint32_t level = subresource.baseMipLevel; level < maxLevel; level++) {
				mSubresourceStates[arrayLayer][level] = newState;
			}
		}
	}
	inline void SetSubresourceState(const vk::ImageSubresourceRange& subresource, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, const uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) {
		SetSubresourceState(subresource, { layout, stage, accessMask, queueFamily });
	}

private:
	vk::Image mImage;
	VmaAllocation mAllocation;
	ImageInfo mInfo;
	std::unordered_map<
		std::tuple<vk::ImageSubresourceRange, vk::ImageViewType, vk::ComponentMapping>,
		vk::raii::ImageView,
		TupleHash<vk::ImageSubresourceRange, vk::ImageViewType, vk::ComponentMapping>> mViews;
	std::vector<std::vector<SubresourceLayoutState>> mSubresourceStates; // mSubresourceStates[arrayLayer][level]
};

}

namespace std {

template<>
struct hash<ptvk::Image::View> {
	inline size_t operator()(const ptvk::Image::View& v) const {
		return hash<vk::ImageView>()(*v);
	}
};

}

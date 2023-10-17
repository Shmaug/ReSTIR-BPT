#pragma once

#include <variant>

#include "Image.hpp"
#include "Pipeline.hpp"

namespace ptvk {

class CommandBuffer {
public:
	Device& mDevice;

	inline CommandBuffer(Device& device, const std::string& name, const uint32_t queueFamily)
		: mDevice(device), mCommandBuffer(nullptr), mQueueFamily(queueFamily) {
		vk::raii::CommandBuffers commandBuffers(*mDevice, vk::CommandBufferAllocateInfo(*mDevice.GetCommandPool(queueFamily), vk::CommandBufferLevel::ePrimary, 1));
		mCommandBuffer = std::move(commandBuffers[0]);
		device.SetDebugName(*mCommandBuffer, name);
	}

	inline       vk::raii::CommandBuffer& operator*()        { return mCommandBuffer; }
	inline const vk::raii::CommandBuffer& operator*() const  { return mCommandBuffer; }
	inline       vk::raii::CommandBuffer* operator->()       { return &mCommandBuffer; }
	inline const vk::raii::CommandBuffer* operator->() const { return &mCommandBuffer; }

	inline const std::shared_ptr<vk::raii::Fence>& GetCompletionFence() const { return mFence; }
	inline uint32_t GetQueueFamily() const { return mQueueFamily; }

	inline void Reset() {
		mHeldResources.clear();
		mCommandBuffer.reset();
		mCommandBuffer.begin(vk::CommandBufferBeginInfo());
	}

	inline void Submit(
		const vk::Queue queue,
		const vk::ArrayProxy<const vk::Semaphore>& waitSemaphores = {},
		const vk::ArrayProxy<const vk::PipelineStageFlags>& waitStages = {},
		const vk::ArrayProxy<const vk::Semaphore>& signalSemaphores = {}) {

		FlushBarriers();

		mCommandBuffer.end();

		if (!mFence)
			mFence = std::make_shared<vk::raii::Fence>(*mDevice, vk::FenceCreateInfo());
		else
			mDevice->resetFences(**mFence);

		queue.submit(vk::SubmitInfo(waitSemaphores, waitStages, *mCommandBuffer, signalSemaphores), **mFence);
	}

	template<typename T>
	inline void HoldResource(const std::shared_ptr<T>& resource) { mHeldResources.emplace(resource.get(), resource); }
	inline void HoldResource(const Image::View& img) { HoldResource(img.GetImage()); }
	inline void HoldResource(const Buffer::View<std::byte>& buf) { HoldResource(buf.GetBuffer()); }

	#pragma region Barriers

	inline static const vk::AccessFlags gWriteAccesses =
			vk::AccessFlagBits::eShaderWrite |
			vk::AccessFlagBits::eColorAttachmentWrite |
			vk::AccessFlagBits::eDepthStencilAttachmentWrite |
			vk::AccessFlagBits::eTransferWrite |
			vk::AccessFlagBits::eHostWrite |
			vk::AccessFlagBits::eMemoryWrite |
			vk::AccessFlagBits::eAccelerationStructureWriteKHR;

	inline void FlushBarriers() {
		for (const auto&[stages, b] : mBarrierQueue)
			mCommandBuffer.pipelineBarrier(stages.first, stages.second,
				vk::DependencyFlagBits::eByRegion,
				{}, b.first, b.second);
		mBarrierQueue.clear();
	}

	inline void Barrier(const vk::ArrayProxy<const Buffer::View<std::byte>>& buffers, const vk::PipelineStageFlags dstStage, const vk::AccessFlags dstAccess, const uint32_t dstQueue = VK_QUEUE_FAMILY_IGNORED) {
		for (auto& b : buffers) {
			const auto& [ srcStage, srcAccess, srcQueue ] = b.GetState();
			if (srcAccess != vk::AccessFlagBits::eNone && dstAccess != vk::AccessFlagBits::eNone && ((srcAccess & gWriteAccesses) || (dstAccess & gWriteAccesses)))
				mBarrierQueue[std::make_pair(srcStage, dstStage)].first.emplace_back(
					srcAccess, dstAccess,
					srcQueue, dstQueue,
					**b.GetBuffer(), b.Offset(), b.SizeBytes());
			b.SetState(dstStage, dstAccess, dstQueue);
		}
	}

	inline void Barrier(const vk::ArrayProxy<const std::shared_ptr<Image>>& imgs, const vk::ImageSubresourceRange& subresource, const Image::SubresourceLayoutState& newState) {
		const auto& [ newLayout, newStage, dstAccessMask, dstQueueFamilyIndex ] = newState;

		for (const auto& img : imgs) {
			const uint32_t maxLayer = std::min(img->GetLayers(), subresource.baseArrayLayer + subresource.layerCount);
			const uint32_t maxLevel = std::min(img->GetLevels(), subresource.baseMipLevel   + subresource.levelCount);
			for (uint32_t arrayLayer = subresource.baseArrayLayer; arrayLayer < maxLayer; arrayLayer++) {
				for (uint32_t level = subresource.baseMipLevel; level < maxLevel; level++) {
					const auto& oldState = img->GetSubresourceState(arrayLayer, level);
					const auto& [ oldLayout, oldStage, srcAccessMask, srcQueueFamilyIndex ] = oldState;
					vk::ImageSubresourceRange range = { subresource.aspectMask, level, 1, arrayLayer, 1 };
					if (oldState != newState || (srcAccessMask != vk::AccessFlagBits::eNone && dstAccessMask != vk::AccessFlagBits::eNone && ((srcAccessMask & gWriteAccesses) || (dstAccessMask & gWriteAccesses)))) {
						// try to combine barrier with one for previous mip level
						std::vector<vk::ImageMemoryBarrier>& b = mBarrierQueue[std::make_pair(oldStage, newStage)].second;
						if (!b.empty()) {
							vk::ImageMemoryBarrier& prev = b.back();
							if (prev.image == **img,
								prev.oldLayout == oldLayout &&
								prev.srcAccessMask == srcAccessMask &&
								prev.srcQueueFamilyIndex == srcQueueFamilyIndex &&
								prev.subresourceRange.baseArrayLayer == arrayLayer &&
								prev.subresourceRange.baseMipLevel + prev.subresourceRange.levelCount == level) {

								prev.subresourceRange.levelCount++;
								img->SetSubresourceState(range, newState);
								continue;
							}
						}
						b.emplace_back(vk::ImageMemoryBarrier(
							srcAccessMask, dstAccessMask,
							oldLayout, newLayout,
							dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : srcQueueFamilyIndex, srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED ? VK_QUEUE_FAMILY_IGNORED : dstQueueFamilyIndex,
							**img, range ));
					}
					img->SetSubresourceState(range, newState);
				}
			}
		}
	}
	inline void Barrier(const vk::ArrayProxy<const std::shared_ptr<Image>>& imgs, const vk::ImageSubresourceRange& subresource, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) {
		Barrier(imgs, subresource, { layout, stage, accessMask, queueFamily });
	}
	inline void Barrier(const Image::View& img, const Image::SubresourceLayoutState& newState) {
		Barrier(img.GetImage(), img.GetSubresourceRange(), newState);
	}
	inline void Barrier(const Image::View& img, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) {
		Barrier(img.GetImage(), img.GetSubresourceRange(), { layout, stage, accessMask, queueFamily });
	}

	#pragma endregion

	#pragma region Buffer manipulation

	inline void Fill(const Buffer::View<std::byte>& buffer, const uint32_t data, const vk::DeviceSize offset = 0, const vk::DeviceSize size = VK_WHOLE_SIZE) {
		Barrier(buffer, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		FlushBarriers();
		mCommandBuffer.fillBuffer(**buffer.GetBuffer(), offset, size, data);
	}

	inline void Copy(const Buffer::View<std::byte>& src, const Buffer::View<std::byte>& dst) {
		if (dst.SizeBytes() < src.SizeBytes())
			throw std::runtime_error("dst buffer smaller than src buffer");
		Barrier(src, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		Barrier(dst, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		FlushBarriers();
		mCommandBuffer.copyBuffer(**src.GetBuffer(), **dst.GetBuffer(), vk::BufferCopy(src.Offset(), dst.Offset(), src.SizeBytes()));
	}
	inline void Copy(const Buffer::View<std::byte>& src, const std::shared_ptr<Image>& dst, const vk::DeviceSize offset = 0) {
		Barrier(src, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		Barrier(dst, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		FlushBarriers();
		mCommandBuffer.copyBufferToImage(**src.GetBuffer(), **dst, vk::ImageLayout::eTransferDstOptimal,
			vk::BufferImageCopy(offset, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor,0,0,1), {0,0,0}, dst->GetExtent()));
	}
	inline void Copy(const Buffer::View<std::byte>& src, const std::shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::BufferImageCopy>& copies) {
		Barrier(src, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		Barrier(dst, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		FlushBarriers();
		mCommandBuffer.copyBufferToImage(**src.GetBuffer(), **dst, vk::ImageLayout::eTransferDstOptimal, copies);
	}
	inline void Copy(const std::shared_ptr<Image>& src, const Buffer::View<std::byte>& dst, const vk::ArrayProxy<const vk::BufferImageCopy>& copies) {
		Barrier(src, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		Barrier(dst, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		FlushBarriers();
		mCommandBuffer.copyImageToBuffer(**src, vk::ImageLayout::eTransferSrcOptimal, **dst.GetBuffer(), copies);
	}

	template<typename T>
	inline std::shared_ptr<Buffer> Upload(const vk::ArrayProxy<const T>& data, const std::string name, vk::BufferUsageFlags usage, const bool fastAllocate = false) {
		VmaAllocationCreateFlags flag = 0;
		if (fastAllocate) flag = VMA_ALLOCATION_CREATE_STRATEGY_MIN_TIME_BIT;

		if (data.empty()) {
			auto dst = std::make_shared<Buffer>(mDevice, name, sizeof(T), usage|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, flag);
			HoldResource(dst);
			return dst;
		}

		auto tmp = std::make_shared<Buffer>(
			mDevice,
			name + "/Staging",
			data.size()*sizeof(T) ,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | flag);
		auto dst = std::make_shared<Buffer>(
			mDevice,
			name,
			data.size()*sizeof(T),
			usage|vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			flag);

		memcpy(tmp->data(), data.data(), data.size()*sizeof(T));
		Copy(tmp, dst);
		HoldResource(tmp);
		HoldResource(dst);
		return dst;
	}

	#pragma endregion

	#pragma region Image manipulation

	inline void Copy(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageCopy>& regions) {
		for (const vk::ImageCopy& region : regions) {
			const auto& s = region.srcSubresource;
			Barrier(src, vk::ImageSubresourceRange(region.srcSubresource.aspectMask, region.srcSubresource.mipLevel, 1, region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
			Barrier(dst, vk::ImageSubresourceRange(region.dstSubresource.aspectMask, region.dstSubresource.mipLevel, 1, region.dstSubresource.baseArrayLayer, region.dstSubresource.layerCount), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		}
		FlushBarriers();
		mCommandBuffer.copyImage(**src, vk::ImageLayout::eTransferSrcOptimal, **dst, vk::ImageLayout::eTransferDstOptimal, regions);
	}
	inline void Copy(const Image::View& src, const Image::View& dst) {
		vk::ImageCopy c(
				src.GetSubresourceLayer(), vk::Offset3D(0,0,0),
				dst.GetSubresourceLayer(), vk::Offset3D(0,0,0),
				dst.GetExtent());
		Copy(src.GetImage(), dst.GetImage(), c);
	}

	inline void Blit(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageBlit>& regions, const vk::Filter filter) {
		for (const vk::ImageBlit& region : regions) {
			Barrier(src, vk::ImageSubresourceRange(region.srcSubresource.aspectMask, region.srcSubresource.mipLevel, 1, region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
			Barrier(dst, vk::ImageSubresourceRange(region.dstSubresource.aspectMask, region.dstSubresource.mipLevel, 1, region.dstSubresource.baseArrayLayer, region.dstSubresource.layerCount), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		}
		FlushBarriers();
		mCommandBuffer.blitImage(**src, vk::ImageLayout::eTransferSrcOptimal, **dst, vk::ImageLayout::eTransferDstOptimal, regions, filter);
	}
	inline void Blit(const Image::View& src, const Image::View& dst, const vk::Filter filter = vk::Filter::eLinear) {
		vk::ImageBlit c(
				src.GetSubresourceLayer(), { vk::Offset3D(0,0,0), vk::Offset3D(src.GetExtent().width, src.GetExtent().height, src.GetExtent().depth) },
				dst.GetSubresourceLayer(), { vk::Offset3D(0,0,0), vk::Offset3D(dst.GetExtent().width, dst.GetExtent().height, dst.GetExtent().depth) });
		Blit(src.GetImage(), dst.GetImage(), { c }, filter);
	}

	inline void GenerateMipMaps(const std::shared_ptr<Image>& img, const vk::Filter filter = vk::Filter::eLinear, const vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eColor) {
		Barrier(img,
			vk::ImageSubresourceRange(aspect, 1, img->GetLevels()-1, 0, img->GetLayers()),
			vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		vk::ImageBlit blit = {};
		blit.srcOffsets[0] = blit.dstOffsets[0] = vk::Offset3D(0, 0, 0);
		blit.srcOffsets[1] = vk::Offset3D((int32_t)img->GetExtent().width, (int32_t)img->GetExtent().height, (int32_t)img->GetExtent().depth);
		blit.srcSubresource.aspectMask = aspect;
		blit.srcSubresource.baseArrayLayer = 0;
		blit.srcSubresource.layerCount = img->GetLayers();
		blit.dstSubresource = blit.srcSubresource;
		for (uint32_t i = 1; i < img->GetLevels(); i++) {
			Barrier(img,
				vk::ImageSubresourceRange(aspect, i-1, 1, 0, img->GetLayers()),
				vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead );
			blit.srcSubresource.mipLevel = i - 1;
			blit.dstSubresource.mipLevel = i;
			blit.dstOffsets[1].x = std::max(1, blit.srcOffsets[1].x / 2);
			blit.dstOffsets[1].y = std::max(1, blit.srcOffsets[1].y / 2);
			blit.dstOffsets[1].z = std::max(1, blit.srcOffsets[1].z / 2);
			FlushBarriers();
			mCommandBuffer.blitImage(
				**img, vk::ImageLayout::eTransferSrcOptimal,
				**img, vk::ImageLayout::eTransferDstOptimal,
				blit, vk::Filter::eLinear);
			blit.srcOffsets[1] = blit.dstOffsets[1];
		}
	}

	inline void ClearColor(const std::shared_ptr<Image>& img, const vk::ClearColorValue& clearValue, const vk::ArrayProxy<const vk::ImageSubresourceRange>& subresources) {
		for (const auto& subresource : subresources)
			Barrier(img, subresource, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		FlushBarriers();
		mCommandBuffer.clearColorImage(**img, vk::ImageLayout::eTransferDstOptimal, clearValue, subresources);
	}
	inline void ClearColor(const Image::View& img, const vk::ClearColorValue& clearValue) {
		ClearColor(img.GetImage(), clearValue, { img.GetSubresourceRange() });
	}

	#pragma endregion

	#pragma region Pipelines

	inline void BindPipeline(const ComputePipeline& pipeline) {
		mCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **pipeline);
	}

	inline void Dispatch(const vk::Extent3D& dim) {
		FlushBarriers();
		mCommandBuffer.dispatch(dim.width, dim.height, dim.depth);
	}

	#pragma endregion

private:
	vk::raii::CommandBuffer mCommandBuffer;
	std::shared_ptr<vk::raii::Fence> mFence;
	uint32_t mQueueFamily;

	std::unordered_map<
		std::pair<vk::PipelineStageFlags, vk::PipelineStageFlags>,
		std::pair< std::vector<vk::BufferMemoryBarrier>, std::vector<vk::ImageMemoryBarrier> >,
		PairHash<vk::PipelineStageFlags, vk::PipelineStageFlags>> mBarrierQueue;

	using ResourcePointer = std::variant<
		std::shared_ptr<Image>,
		std::shared_ptr<Buffer>,
		std::shared_ptr<Pipeline>,
		std::shared_ptr<vk::raii::Sampler>,
		std::shared_ptr<vk::raii::AccelerationStructureKHR>,
		std::shared_ptr<vk::raii::DescriptorSet> >;
	std::unordered_map<void*, ResourcePointer> mHeldResources;
};

}

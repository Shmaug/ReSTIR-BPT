#pragma once

#include "ShaderParameterBlock.hpp"
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

	#pragma region Buffer manipulation

	inline void Barrier(
		const vk::ArrayProxy<const Buffer::View<std::byte>>& buffers,
		const vk::PipelineStageFlags srcStage, const vk::PipelineStageFlags dstStage,
		const vk::AccessFlags srcAccess, const vk::AccessFlags dstAccess,
		const uint32_t srcQueue = VK_QUEUE_FAMILY_IGNORED, const uint32_t dstQueue = VK_QUEUE_FAMILY_IGNORED) const {

		std::vector<vk::BufferMemoryBarrier> bufferMemoryBarriers =
			buffers |
			std::views::transform([=](const auto& v){ return vk::BufferMemoryBarrier(
				srcAccess, dstAccess,
				srcQueue, dstQueue,
				**v.GetBuffer(), v.Offset(), v.SizeBytes()); }) |
			std::ranges::to<std::vector<vk::BufferMemoryBarrier>>();

		mCommandBuffer.pipelineBarrier(
			srcStage, dstStage,
			vk::DependencyFlagBits::eByRegion,
			{},
			bufferMemoryBarriers,
			{});
	}

	inline void Copy(const Buffer::View<std::byte>& src, const Buffer::View<std::byte>& dst) {
		if (dst.SizeBytes() < src.SizeBytes())
			throw std::runtime_error("dst buffer smaller than src buffer");
		mCommandBuffer.copyBuffer(**src.GetBuffer(), **dst.GetBuffer(), vk::BufferCopy(src.Offset(), dst.Offset(), src.SizeBytes()));
	}
	inline void Copy(const Buffer::View<std::byte>& src, const std::shared_ptr<Image>& dst, const vk::DeviceSize offset = 0) const {
		Barrier(dst, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		mCommandBuffer.copyBufferToImage(**src.GetBuffer(), **dst, vk::ImageLayout::eTransferDstOptimal,
			vk::BufferImageCopy(offset, 0, 0, vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor,0,0,1), {0,0,0}, dst->GetExtent()));
	}
	inline void Copy(const Buffer::View<std::byte>& src, const std::shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::BufferImageCopy>& copies) const {
		Barrier(dst, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		mCommandBuffer.copyBufferToImage(**src.GetBuffer(), **dst, vk::ImageLayout::eTransferDstOptimal, copies);
	}
	inline void Copy(const std::shared_ptr<Image>& src, const Buffer::View<std::byte>& dst, const vk::ArrayProxy<const vk::BufferImageCopy>& copies) const {
		Barrier(src, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
		mCommandBuffer.copyImageToBuffer(**src, vk::ImageLayout::eTransferSrcOptimal, **dst.GetBuffer(), copies);
	}

	inline void Fill(const Buffer::View<std::byte>& buffer, const uint32_t data, const vk::DeviceSize offset = 0, const vk::DeviceSize size = VK_WHOLE_SIZE) const {
		mCommandBuffer.fillBuffer(**buffer.GetBuffer(), offset, size, data);
	}

	#pragma endregion

	#pragma region Image manipulation

	inline void Barrier(const std::shared_ptr<Image>& img, const vk::ImageSubresourceRange& subresource, const Image::SubresourceLayoutState& newState) const {
		const vk::AccessFlags writeAccess =
			vk::AccessFlagBits::eShaderWrite |
			vk::AccessFlagBits::eColorAttachmentWrite |
			vk::AccessFlagBits::eDepthStencilAttachmentWrite |
			vk::AccessFlagBits::eTransferWrite |
			vk::AccessFlagBits::eHostWrite |
			vk::AccessFlagBits::eMemoryWrite |
			vk::AccessFlagBits::eAccelerationStructureWriteKHR;

		std::unordered_map<
			std::pair<vk::PipelineStageFlags, vk::PipelineStageFlags>,
			std::vector<vk::ImageMemoryBarrier>,
			PairHash<vk::PipelineStageFlags, vk::PipelineStageFlags>> barriers;

		const auto& [ newLayout, newStage, dstAccessMask, dstQueueFamilyIndex ] = newState;

		const uint32_t maxLayer = std::min(img->GetLayers(), subresource.baseArrayLayer + subresource.layerCount);
		const uint32_t maxLevel = std::min(img->GetLevels(), subresource.baseMipLevel   + subresource.levelCount);

		for (uint32_t arrayLayer = subresource.baseArrayLayer; arrayLayer < maxLayer; arrayLayer++) {
			for (uint32_t level = subresource.baseMipLevel; level < maxLevel; level++) {
				const auto& oldState = img->GetSubresourceState(arrayLayer, level);
				const auto& [ oldLayout, curStage, srcAccessMask, srcQueueFamilyIndex ] = oldState;
				if (oldState != newState || (srcAccessMask & writeAccess) || (dstAccessMask & writeAccess)) {
					vk::ImageSubresourceRange range = { subresource.aspectMask, level, 1, arrayLayer, 1 };

					std::vector<vk::ImageMemoryBarrier>& b = barriers[std::make_pair(curStage, newStage)];

					// try to combine barrier with one for previous mip level
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

					img->SetSubresourceState(range, newState);
				}
			}
		}

		for (const auto&[stages, b] : barriers)
			mCommandBuffer.pipelineBarrier(stages.first, stages.second, vk::DependencyFlagBits::eByRegion, {}, {}, b);
	}
	inline void Barrier(const std::shared_ptr<Image>& img, const vk::ImageSubresourceRange& subresource, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) const {
		Barrier(img, subresource, { layout, stage, accessMask, queueFamily });
	}
	inline void Barrier(const Image::View& img, const Image::SubresourceLayoutState& newState) const {
		Barrier(img.GetImage(), img.GetSubresourceRange(), newState);
	}
	inline void Barrier(const Image::View& img, const vk::ImageLayout layout, const vk::PipelineStageFlags stage, const vk::AccessFlags accessMask, uint32_t queueFamily = VK_QUEUE_FAMILY_IGNORED) const {
		Barrier(img.GetImage(), img.GetSubresourceRange(), { layout, stage, accessMask, queueFamily });
	}

	inline void Copy(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageCopy>& regions) const {
		for (const vk::ImageCopy& region : regions) {
			const auto& s = region.srcSubresource;
			Barrier(src, vk::ImageSubresourceRange(region.srcSubresource.aspectMask, region.srcSubresource.mipLevel, 1, region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
			Barrier(dst, vk::ImageSubresourceRange(region.dstSubresource.aspectMask, region.dstSubresource.mipLevel, 1, region.dstSubresource.baseArrayLayer, region.dstSubresource.layerCount), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		}
		mCommandBuffer.copyImage(**src, vk::ImageLayout::eTransferSrcOptimal, **dst, vk::ImageLayout::eTransferDstOptimal, regions);
	}
	inline void Copy(const Image::View& src, const Image::View& dst) const {
		vk::ImageCopy c(
				src.GetSubresourceLayer(), vk::Offset3D(0,0,0),
				dst.GetSubresourceLayer(), vk::Offset3D(0,0,0),
				dst.GetExtent());
		Copy(src.GetImage(), dst.GetImage(), c);
	}

	inline void Blit(const std::shared_ptr<Image>& src, const std::shared_ptr<Image>& dst, const vk::ArrayProxy<const vk::ImageBlit>& regions, const vk::Filter filter) const {
		for (const vk::ImageBlit& region : regions) {
			Barrier(src, vk::ImageSubresourceRange(region.srcSubresource.aspectMask, region.srcSubresource.mipLevel, 1, region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount), vk::ImageLayout::eTransferSrcOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferRead);
			Barrier(dst, vk::ImageSubresourceRange(region.dstSubresource.aspectMask, region.dstSubresource.mipLevel, 1, region.dstSubresource.baseArrayLayer, region.dstSubresource.layerCount), vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		}
		mCommandBuffer.blitImage(**src, vk::ImageLayout::eTransferSrcOptimal, **dst, vk::ImageLayout::eTransferDstOptimal, regions, filter);
	}
	inline void Blit(const Image::View& src, const Image::View& dst, const vk::Filter filter = vk::Filter::eLinear) const {
		vk::ImageBlit c(
				src.GetSubresourceLayer(), { vk::Offset3D(0,0,0), vk::Offset3D(src.GetExtent().width, src.GetExtent().height, src.GetExtent().depth) },
				dst.GetSubresourceLayer(), { vk::Offset3D(0,0,0), vk::Offset3D(dst.GetExtent().width, dst.GetExtent().height, dst.GetExtent().depth) });
		Blit(src.GetImage(), dst.GetImage(), { c }, filter);
	}

	inline void GenerateMipMaps(const std::shared_ptr<Image>& img, const vk::Filter filter, const vk::ImageAspectFlags aspect) const {
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
			mCommandBuffer.blitImage(
				**img, vk::ImageLayout::eTransferSrcOptimal,
				**img, vk::ImageLayout::eTransferDstOptimal,
				blit, vk::Filter::eLinear);
			blit.srcOffsets[1] = blit.dstOffsets[1];
		}
	}

	inline void ClearColor(const std::shared_ptr<Image>& img, const vk::ClearColorValue& clearValue, const vk::ArrayProxy<const vk::ImageSubresourceRange>& subresources) const {
		for (const auto& subresource : subresources)
			Barrier(img, subresource, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite);
		mCommandBuffer.clearColorImage(**img, vk::ImageLayout::eTransferDstOptimal, clearValue, subresources);
	}
	inline void ClearColor(const Image::View& img, const vk::ClearColorValue& clearValue) const {
		ClearColor(img.GetImage(), clearValue, { img.GetSubresourceRange() });
	}

	#pragma endregion

	#pragma region Pipelines

	inline void TransitionImages(const ShaderParameterBlock& params, const vk::PipelineStageFlags stage) const {
		// transition image descriptors to specified layout
		for (const auto& [id, param] : params) {
			if (const auto* v = std::get_if<ImageParameter>(&param)) {
				const auto& [image, layout, accessFlags, sampler] = *v;
				Barrier(image, layout, stage, accessFlags, GetQueueFamily());
			}
		}
	}

	inline void BindPipeline(const ComputePipeline& pipeline) {
		mCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, **pipeline);
	}

	inline void Dispatch(const vk::Extent3D& dim) const {
		mCommandBuffer.dispatch(dim.width, dim.height, dim.depth);
	}

	#pragma endregion

private:
	vk::raii::CommandBuffer mCommandBuffer;
	std::shared_ptr<vk::raii::Fence> mFence;
	uint32_t mQueueFamily;

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

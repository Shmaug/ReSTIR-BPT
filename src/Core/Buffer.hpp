#pragma once

#include "Device.hpp"

namespace ptvk {

class Buffer {
public:
	using ResourceState = std::tuple<vk::PipelineStageFlags, vk::AccessFlags, uint32_t>;

	template<typename T = std::byte>
	class View {
	public:
		using value_type = T;
		using size_type = vk::DeviceSize;
		using reference = value_type&;
		using pointer = value_type*;
		using iterator = T*;

		View() = default;
		View(View&&) = default;
		inline View(const View& v, size_t elementOffset = 0, size_t elementCount = VK_WHOLE_SIZE) : mBuffer(v.mBuffer), mOffset(v.mOffset + elementOffset * sizeof(T)) {
			if (mBuffer) {
				mSize = (elementCount == VK_WHOLE_SIZE) ? (v.size() - elementOffset) : elementCount;
				if (mOffset + mSize * sizeof(T) > mBuffer->size())
					throw std::out_of_range("view size out of bounds");
			}
		}
		inline View(const std::shared_ptr<Buffer>& buffer, const vk::DeviceSize byteOffset = 0, const vk::DeviceSize elementCount = VK_WHOLE_SIZE) : mBuffer(buffer), mOffset(byteOffset) {
			if (mBuffer) {
				mSize = (elementCount == VK_WHOLE_SIZE) ? (mBuffer->size() - mOffset) / sizeof(T) : elementCount;
				if (mOffset + mSize * sizeof(T) > mBuffer->size())
					throw std::out_of_range("view size out of bounds");
			}
		}

		View& operator=(const View&) = default;
		View& operator=(View&&) = default;
		bool operator==(const View&) const = default;

		inline operator View<std::byte>() const { return View<std::byte>(mBuffer, mOffset, SizeBytes()); }
		inline operator bool() const { return !empty(); }

		template<typename Ty>
		inline View<Ty> Cast() const {
			if ((SizeBytes() % sizeof(Ty)) != 0)
				throw std::logic_error("Buffer size must be divisible by sizeof(Ty)");
			return View<Ty>(GetBuffer(), Offset(), SizeBytes() / sizeof(Ty));
		}

		inline const std::shared_ptr<Buffer>& GetBuffer() const { return mBuffer; }
		inline vk::DeviceSize Offset() const { return mOffset; }
		inline vk::DeviceSize SizeBytes() const { return mSize * sizeof(T); }
		inline vk::DeviceSize GetDeviceAddress() const { return mBuffer->GetDeviceAddress() + mOffset; }

		inline const ResourceState& GetState() const { return mBuffer->GetState(mOffset, mSize); }
		inline void SetState(const ResourceState& newState) const { mBuffer->SetState(newState, mOffset, mSize); }
		inline void SetState(const vk::PipelineStageFlags stage, const vk::AccessFlags access, uint32_t queue = VK_QUEUE_FAMILY_IGNORED) const {
			mBuffer->SetState({stage, access, queue}, mOffset, mSize);
		}

		inline bool empty() const { return !mBuffer || mSize == 0; }
		inline vk::DeviceSize size() const { return mSize; }
		inline pointer data() const { return reinterpret_cast<pointer>(reinterpret_cast<std::byte*>(mBuffer->data()) + Offset()); }

		inline T& at(size_type index) const { return data()[index]; }
		inline T& operator[](size_type index) const { return at(index); }

		inline reference front() { return at(0); }
		inline reference back() { return at(mSize - 1); }

		inline iterator begin() const { return data(); }
		inline iterator end() const { return data() + mSize; }

	private:
		std::shared_ptr<Buffer> mBuffer;
		vk::DeviceSize mOffset;
		vk::DeviceSize mSize;
	};

	class StrideView : public View<std::byte> {
	public:
		StrideView() = default;
		StrideView(StrideView&&) = default;
		StrideView(const StrideView&) = default;
		inline StrideView(const View<std::byte>& view, vk::DeviceSize stride) : View<std::byte>(view), mStride(stride) {}
		template<typename T>
		inline StrideView(const View<T>& v) : View<std::byte>(v.GetBuffer(), v.Offset(), v.SizeBytes()), mStride(sizeof(T)) {}
		inline StrideView(const std::shared_ptr<Buffer>& buffer, vk::DeviceSize stride, vk::DeviceSize byteOffset = 0, vk::DeviceSize byteLength = VK_WHOLE_SIZE)
			: View<std::byte>(buffer, byteOffset, byteLength), mStride(stride) {}

		StrideView& operator=(const StrideView&) = default;
		StrideView& operator=(StrideView&&) = default;
		bool operator==(const StrideView&) const = default;

		inline vk::DeviceSize Stride() const { return mStride; }

		template<typename T>
		inline operator View<T>() const {
			if (sizeof(T) != mStride) throw std::logic_error("sizeof(T) must match stride");
			return Buffer::View<T>(GetBuffer(), Offset(), SizeBytes() / sizeof(T));
		}

	private:
		vk::DeviceSize mStride;
	};

	Device& mDevice;

	inline Buffer(Device& device, const std::string& name, const vk::BufferCreateInfo& createInfo, const vk::MemoryPropertyFlags memoryFlags, const bool hostRandomAccess)
		: mDevice(device), mName(name), mSize(createInfo.size), mUsage(createInfo.usage), mMemoryFlags(memoryFlags), mSharingMode(createInfo.sharingMode) {
		VmaAllocationCreateInfo allocationCreateInfo;
		allocationCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | (hostRandomAccess ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		allocationCreateInfo.usage = (memoryFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
		allocationCreateInfo.requiredFlags = (VkMemoryPropertyFlags)memoryFlags;
		allocationCreateInfo.memoryTypeBits = 0;
		allocationCreateInfo.pool = VK_NULL_HANDLE;
		allocationCreateInfo.pUserData = VK_NULL_HANDLE;
		allocationCreateInfo.priority = 0;
		vk::Result result = (vk::Result)vmaCreateBuffer(mDevice.GetAllocator(), &(const VkBufferCreateInfo&)createInfo, &allocationCreateInfo, &(VkBuffer&)mBuffer, &mAllocation, &mAllocationInfo);
		if (result != vk::Result::eSuccess)
			vk::throwResultException(result, "vmaCreateBuffer");
		device.SetDebugName(mBuffer, name);
	}
	inline Buffer(Device& device, const std::string& name, const vk::DeviceSize& size, const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags memoryFlags = vk::MemoryPropertyFlagBits::eDeviceLocal, const bool hostRandomAccess = false) :
		Buffer(device, name, vk::BufferCreateInfo({}, size, usage), memoryFlags, hostRandomAccess) {}
	inline ~Buffer() {
		if (mBuffer && mAllocation)
			vmaDestroyBuffer(mDevice.GetAllocator(), mBuffer, mAllocation);
	}

	inline       vk::Buffer& operator*()        { return mBuffer; }
	inline const vk::Buffer& operator*() const  { return mBuffer; }
	inline       vk::Buffer* operator->()       { return &mBuffer; }
	inline const vk::Buffer* operator->() const { return &mBuffer; }
	inline operator bool() const { return mBuffer; }

	inline std::string GetName() const { return mName; }
	inline vk::BufferUsageFlags GetUsage() const { return mUsage; }
	inline vk::MemoryPropertyFlags GetMemoryUsage() const { return mMemoryFlags; }
	inline vk::SharingMode SharingMode() const { return mSharingMode; }
	inline vk::DeviceSize GetDeviceAddress() const { return mDevice->getBufferAddress(mBuffer); }

	inline const ResourceState& GetState(vk::DeviceSize offset, vk::DeviceSize size) {
		if (auto it = mState.find(std::make_pair(offset, size)); it != mState.end())
			return it->second;
		return mState.emplace(std::make_pair(offset, size), ResourceState{vk::PipelineStageFlagBits::eTopOfPipe, vk::AccessFlagBits::eNone, VK_QUEUE_FAMILY_IGNORED}).first->second;
	}
	inline void SetState(const ResourceState& newState, vk::DeviceSize offset, vk::DeviceSize size) {
		mState[std::make_pair(offset, size)] = newState;
	}

	inline void* data() const { return mAllocationInfo.pMappedData; }
	inline vk::DeviceSize size() const { return mSize; }

private:
	vk::Buffer mBuffer;
	std::string mName;
	VmaAllocation mAllocation;
	VmaAllocationInfo mAllocationInfo;
	vk::DeviceSize mSize;
	vk::BufferUsageFlags mUsage;
	vk::MemoryPropertyFlags mMemoryFlags;
	vk::SharingMode mSharingMode;

	std::unordered_map<std::pair<vk::DeviceSize, vk::DeviceSize>, ResourceState, PairHash<vk::DeviceSize, vk::DeviceSize>> mState;
};

}
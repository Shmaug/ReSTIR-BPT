#include "Image.hpp"
#include "Buffer.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#define TINYEXR_USE_MINIZ 1
#include <miniz.h>
#define TINYEXR_IMPLEMENTATION
#include <tiny_exr.h>

#define TINYDDSLOADER_IMPLEMENTATION
#include <tinyddsloader.h>

namespace ptvk {

inline vk::Format dxgiToVulkan(tinyddsloader::DDSFile::DXGIFormat format, const bool alphaFlag) {
	switch (format) {
		case tinyddsloader::DDSFile::DXGIFormat::BC1_UNorm: {
			if (alphaFlag) return vk::Format::eBc1RgbaUnormBlock;
			else return vk::Format::eBc1RgbUnormBlock;
		}
		case tinyddsloader::DDSFile::DXGIFormat::BC1_UNorm_SRGB: {
			if (alphaFlag) return vk::Format::eBc1RgbaSrgbBlock;
			else return vk::Format::eBc1RgbSrgbBlock;
		}

		case tinyddsloader::DDSFile::DXGIFormat::BC2_UNorm:       return vk::Format::eBc2UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC2_UNorm_SRGB:  return vk::Format::eBc2SrgbBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC3_UNorm:       return vk::Format::eBc3UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC3_UNorm_SRGB:  return vk::Format::eBc3SrgbBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC4_UNorm:       return vk::Format::eBc4UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC4_SNorm:       return vk::Format::eBc4SnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC5_UNorm:       return vk::Format::eBc5UnormBlock;
		case tinyddsloader::DDSFile::DXGIFormat::BC5_SNorm:       return vk::Format::eBc5SnormBlock;

		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_UNorm:      return vk::Format::eR8G8B8A8Unorm;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_UNorm_SRGB: return vk::Format::eR8G8B8A8Srgb;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_UInt:       return vk::Format::eR8G8B8A8Uint;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_SNorm:      return vk::Format::eR8G8B8A8Snorm;
		case tinyddsloader::DDSFile::DXGIFormat::R8G8B8A8_SInt:       return vk::Format::eR8G8B8A8Sint;
		case tinyddsloader::DDSFile::DXGIFormat::B8G8R8A8_UNorm:      return vk::Format::eB8G8R8A8Unorm;
		case tinyddsloader::DDSFile::DXGIFormat::B8G8R8A8_UNorm_SRGB: return vk::Format::eB8G8R8A8Srgb;

		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_Float:  return vk::Format::eR16G16B16A16Sfloat;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_SInt:   return vk::Format::eR16G16B16A16Sint;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_UInt:   return vk::Format::eR16G16B16A16Uint;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_UNorm:  return vk::Format::eR16G16B16A16Unorm;
		case tinyddsloader::DDSFile::DXGIFormat::R16G16B16A16_SNorm:  return vk::Format::eR16G16B16A16Snorm;

		default: return vk::Format::eUndefined;
	}
}

PixelData LoadImageFile(Device& device, const std::filesystem::path& filename, const bool srgb, int desiredChannels) {
	if (!std::filesystem::exists(filename))
		throw std::invalid_argument("File does not exist: " + filename.string());
	if (filename.extension() == ".exr") {
		float* data = nullptr;
		int width;
		int height;
		const char* err = nullptr;
		int ret = LoadEXR(&data, &width, &height, filename.string().c_str(), &err);
		if (ret != TINYEXR_SUCCESS) {
			std::cerr << "OpenEXR error: " << err << std::endl;
			FreeEXRErrorMessage(err);
			throw std::runtime_error(std::string("Failure when loading image: ") + filename.string());
		}
		auto buf = std::make_shared<Buffer>(device, filename.stem().string() + "/Staging", width*height*sizeof(float)*4, vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eStorageBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
			VMA_ALLOCATION_CREATE_MAPPED_BIT|VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		memcpy(buf->data(), data, buf->size());
		free(data);
		return PixelData{buf, vk::Format::eR32G32B32A32Sfloat, vk::Extent3D(width,height,1)};
	} else if (filename.extension() == ".dds") {
		using namespace tinyddsloader;
		DDSFile dds;
    	auto ret = dds.Load(filename.string().c_str());
		if (tinyddsloader::Result::tinydds_Success != ret) throw std::runtime_error("Failed to load " + filename.string());
		dds.GetBitsPerPixel(dds.GetFormat());

		dds.Flip();

		const DDSFile::ImageData* img = dds.GetImageData(0, 0);

		auto buf = std::make_shared<Buffer>(device, filename.stem().string() + "/Staging", img->m_memSlicePitch, vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eStorageBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
			VMA_ALLOCATION_CREATE_MAPPED_BIT|VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		memcpy(buf->data(), img->m_mem, buf->size());
		return PixelData{buf, dxgiToVulkan(dds.GetFormat(), desiredChannels == 4), vk::Extent3D(dds.GetWidth(), dds.GetHeight(), dds.GetDepth())};
	} else {
		int x,y,channels;
		stbi_info(filename.string().c_str(), &x, &y, &channels);

		if (channels == 3) desiredChannels = 4;

		std::byte* pixels = nullptr;
		vk::Format format = vk::Format::eUndefined;
		if (stbi_is_hdr(filename.string().c_str())) {
			pixels = (std::byte*)stbi_loadf(filename.string().c_str(), &x, &y, &channels, desiredChannels);
			switch(desiredChannels ? desiredChannels : channels) {
				case 1: format = vk::Format::eR32Sfloat; break;
				case 2: format = vk::Format::eR32G32Sfloat; break;
				case 3: format = vk::Format::eR32G32B32Sfloat; break;
				case 4: format = vk::Format::eR32G32B32A32Sfloat; break;
			}
		} else if (stbi_is_16_bit(filename.string().c_str())) {
			pixels = (std::byte*)stbi_load_16(filename.string().c_str(), &x, &y, &channels, desiredChannels);
			switch(desiredChannels ? desiredChannels : channels) {
				case 1: format = vk::Format::eR16Unorm; break;
				case 2: format = vk::Format::eR16G16Unorm; break;
				case 3: format = vk::Format::eR16G16B16Unorm; break;
				case 4: format = vk::Format::eR16G16B16A16Unorm; break;
			}
		} else {
			pixels = (std::byte*)stbi_load(filename.string().c_str(), &x, &y, &channels, desiredChannels);
			switch (desiredChannels ? desiredChannels : channels) {
				case 1: format = srgb ? vk::Format::eR8Srgb : vk::Format::eR8Unorm; break;
				case 2: format = srgb ? vk::Format::eR8G8Srgb : vk::Format::eR8G8Unorm; break;
				case 3: format = srgb ? vk::Format::eR8G8B8Srgb : vk::Format::eR8G8B8Unorm; break;
				case 4: format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm; break;
			}
		}
		if (!pixels) throw std::invalid_argument("Could not load " + filename.string());
		std::cout << "Loaded " << filename << " (" << x << "x" << y << ")" << std::endl;
		if (desiredChannels) channels = desiredChannels;

		auto buf = std::make_shared<Buffer>(device, filename.stem().string() + "/Staging", x*y*GetTexelSize(format), vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent,
			VMA_ALLOCATION_CREATE_MAPPED_BIT|VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		memcpy(buf->data(), pixels, buf->size());
		stbi_image_free(pixels);
		return PixelData{buf, format, vk::Extent3D(x,y,1)};
	}
}


Image::Image(Device& device, const std::string& name, const ImageInfo& info, const vk::MemoryPropertyFlags memoryFlags, const VmaAllocationCreateFlags allocationFlags) : mDevice(device), mImage(nullptr), mName(name), mInfo(info) {
	VmaAllocationCreateInfo allocationCreateInfo;
	allocationCreateInfo.flags = allocationFlags;
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationCreateInfo.requiredFlags = (VkMemoryPropertyFlags)memoryFlags;
    allocationCreateInfo.memoryTypeBits = 0;
    allocationCreateInfo.pool = VK_NULL_HANDLE;
    allocationCreateInfo.pUserData = VK_NULL_HANDLE;
    allocationCreateInfo.priority = 0;

	vk::ImageCreateInfo createInfo(
		mInfo.mCreateFlags,
		GetType(),
		GetFormat(),
		GetExtent(),
		GetLevels(),
		GetLayers(),
		GetSamples(),
		GetTiling(),
		GetUsage(),
		GetSharingMode(),
		GetQueueFamilies(),
		vk::ImageLayout::eUndefined );

	vk::Result result = (vk::Result)vmaCreateImage(mDevice.GetAllocator(), &(const VkImageCreateInfo&)createInfo, &allocationCreateInfo, &(VkImage&)mImage, &mAllocation, nullptr);
	if (result != vk::Result::eSuccess)
		vk::throwResultException(result, "vmaCreateImage");
	device.SetDebugName(mImage, name);
	mSubresourceStates = std::vector<std::vector<Image::SubresourceLayoutState>>(
		mInfo.mLayers,
		std::vector<Image::SubresourceLayoutState>(
			mInfo.mLevels,
			Image::SubresourceLayoutState{
				vk::ImageLayout::eUndefined,
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::AccessFlagBits::eNone,
				GetQueueFamilies().empty() ? VK_QUEUE_FAMILY_IGNORED : GetQueueFamilies().front() }));
	//std::cout << "Creating image " << mName << " (" << mInfo.mExtent.width << "x" << mInfo.mExtent.height << "x" << mInfo.mExtent.depth << " " << vk::to_string(mInfo.mFormat) << ")" << std::endl;
}
Image::Image(Device& device, const std::string& name, const vk::Image image, const ImageInfo& info) : mDevice(device), mImage(image), mName(name), mInfo(info), mAllocation(nullptr) {
	mAllocation = nullptr;
	if (mImage) device.SetDebugName(mImage, name);
	mSubresourceStates = std::vector<std::vector<Image::SubresourceLayoutState>>(
		mInfo.mLayers,
		std::vector<Image::SubresourceLayoutState>(
			mInfo.mLevels,
			Image::SubresourceLayoutState{
				vk::ImageLayout::eUndefined,
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::AccessFlagBits::eNone,
				GetQueueFamilies().empty() ? VK_QUEUE_FAMILY_IGNORED : GetQueueFamilies().front() }));
}
Image::~Image() {
	if (mImage && mAllocation) {
		vmaDestroyImage(mDevice.GetAllocator(), mImage, mAllocation);
		//std::cout << "Destroying image " << mName << " (" << mInfo.mExtent.width << "x" << mInfo.mExtent.height << "x" << mInfo.mExtent.depth << " " << vk::to_string(mInfo.mFormat) << ")" << std::endl;
	}
}

const vk::ImageView Image::GetView(const vk::ImageSubresourceRange& subresource, const vk::ImageViewType viewType, const vk::ComponentMapping& componentMapping) {
	auto key = std::tie(subresource, viewType, componentMapping);
	auto it = mViews.find(key);
	if (it == mViews.end()) {
		vk::raii::ImageView v(*mDevice, vk::ImageViewCreateInfo({},
			mImage,
			viewType,
			GetFormat(),
			componentMapping,
			subresource));
		mDevice.SetDebugName(*v, mName);
		it = mViews.emplace(key, std::move(v)).first;
	}
	return *it->second;
}

}
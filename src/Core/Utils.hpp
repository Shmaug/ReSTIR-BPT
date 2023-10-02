#pragma once

#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <optional>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <ranges>
#include <utility>

#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "Hash.hpp"

namespace ptvk {

template<std::ranges::contiguous_range R>
inline R ReadFile(const std::filesystem::path& filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) return {};
	R dst;
	dst.resize((size_t)file.tellg()/sizeof(std::ranges::range_value_t<R>));
	if (dst.empty()) return dst;
	file.seekg(0);
	file.clear();
	file.read(reinterpret_cast<char*>(dst.data()), dst.size()*sizeof(std::ranges::range_value_t<R>));
	return dst;
}
template<std::ranges::contiguous_range R>
inline R ReadFile(const std::filesystem::path& filename, R& dst) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) return {};
	size_t sz = (size_t)file.tellg();
	file.seekg(0);
	file.clear();
	file.read(reinterpret_cast<char*>(dst.data()), sz);
	return dst;
}
template<std::ranges::contiguous_range R>
inline void WriteFile(const std::filesystem::path& filename, const R& r) {
	std::ofstream file(filename, std::ios::ate | std::ios::binary);
	file.write(reinterpret_cast<const char*>(r.data()), r.size()*sizeof(std::ranges::range_value_t<R>));
}

// returns bytes,unit
inline std::tuple<size_t, const char*> FormatBytes(size_t bytes) {
	const char* units[] { "B", "KiB", "MiB", "GiB", "TiB" };
	uint32_t i = 0;
	while (bytes > 1024 && i < std::ranges::size(units)-1) {
		bytes /= 1024;
		i++;
	}
	return std::tie(bytes, units[i]);
}
// returns number,unit
inline std::tuple<float, const char*> FormatNumber(float number) {
	const char* units[] { "", "K", "M", "B" };
	uint32_t i = 0;
	while (number > 1000 && i < std::ranges::size(units)-1) {
		number /= 1000;
		i++;
	}
	return std::tie(number, units[i]);
}

inline uint32_t GetMaxMipLevels(const vk::Extent3D& extent) {
	return 32 - (uint32_t)std::countl_zero(std::max(std::max(extent.width, extent.height), extent.depth));
}

inline constexpr bool IsDepthStencil(vk::Format format) {
	return
		format == vk::Format::eS8Uint ||
		format == vk::Format::eD16Unorm ||
		format == vk::Format::eD16UnormS8Uint ||
		format == vk::Format::eX8D24UnormPack32 ||
		format == vk::Format::eD24UnormS8Uint ||
		format == vk::Format::eD32Sfloat ||
		format == vk::Format::eD32SfloatS8Uint;
}

// Size of an element of format, in bytes
template<typename T = uint32_t> requires(std::is_arithmetic_v<T>)
inline constexpr T GetTexelSize(vk::Format format) {
	switch (format) {
	default:
		throw std::runtime_error("Texel size unknown for format " + to_string(format));
	case vk::Format::eR4G4UnormPack8:
	case vk::Format::eR8Unorm:
	case vk::Format::eR8Snorm:
	case vk::Format::eR8Uscaled:
	case vk::Format::eR8Sscaled:
	case vk::Format::eR8Uint:
	case vk::Format::eR8Sint:
	case vk::Format::eR8Srgb:
	case vk::Format::eS8Uint:
		return 1;

	case vk::Format::eR4G4B4A4UnormPack16:
	case vk::Format::eB4G4R4A4UnormPack16:
	case vk::Format::eR5G6B5UnormPack16:
	case vk::Format::eB5G6R5UnormPack16:
	case vk::Format::eR5G5B5A1UnormPack16:
	case vk::Format::eB5G5R5A1UnormPack16:
	case vk::Format::eA1R5G5B5UnormPack16:
	case vk::Format::eR8G8Unorm:
	case vk::Format::eR8G8Snorm:
	case vk::Format::eR8G8Uscaled:
	case vk::Format::eR8G8Sscaled:
	case vk::Format::eR8G8Uint:
	case vk::Format::eR8G8Sint:
	case vk::Format::eR8G8Srgb:
	case vk::Format::eR16Unorm:
	case vk::Format::eR16Snorm:
	case vk::Format::eR16Uscaled:
	case vk::Format::eR16Sscaled:
	case vk::Format::eR16Uint:
	case vk::Format::eR16Sint:
	case vk::Format::eR16Sfloat:
	case vk::Format::eD16Unorm:
		return 2;

	case vk::Format::eR8G8B8Unorm:
	case vk::Format::eR8G8B8Snorm:
	case vk::Format::eR8G8B8Uscaled:
	case vk::Format::eR8G8B8Sscaled:
	case vk::Format::eR8G8B8Uint:
	case vk::Format::eR8G8B8Sint:
	case vk::Format::eR8G8B8Srgb:
	case vk::Format::eB8G8R8Unorm:
	case vk::Format::eB8G8R8Snorm:
	case vk::Format::eB8G8R8Uscaled:
	case vk::Format::eB8G8R8Sscaled:
	case vk::Format::eB8G8R8Uint:
	case vk::Format::eB8G8R8Sint:
	case vk::Format::eB8G8R8Srgb:
	case vk::Format::eD16UnormS8Uint:
		return 3;

	case vk::Format::eR8G8B8A8Unorm:
	case vk::Format::eR8G8B8A8Snorm:
	case vk::Format::eR8G8B8A8Uscaled:
	case vk::Format::eR8G8B8A8Sscaled:
	case vk::Format::eR8G8B8A8Uint:
	case vk::Format::eR8G8B8A8Sint:
	case vk::Format::eR8G8B8A8Srgb:
	case vk::Format::eB8G8R8A8Unorm:
	case vk::Format::eB8G8R8A8Snorm:
	case vk::Format::eB8G8R8A8Uscaled:
	case vk::Format::eB8G8R8A8Sscaled:
	case vk::Format::eB8G8R8A8Uint:
	case vk::Format::eB8G8R8A8Sint:
	case vk::Format::eB8G8R8A8Srgb:
	case vk::Format::eA8B8G8R8UnormPack32:
	case vk::Format::eA8B8G8R8SnormPack32:
	case vk::Format::eA8B8G8R8UscaledPack32:
	case vk::Format::eA8B8G8R8SscaledPack32:
	case vk::Format::eA8B8G8R8UintPack32:
	case vk::Format::eA8B8G8R8SintPack32:
	case vk::Format::eA8B8G8R8SrgbPack32:
	case vk::Format::eA2R10G10B10UnormPack32:
	case vk::Format::eA2R10G10B10SnormPack32:
	case vk::Format::eA2R10G10B10UscaledPack32:
	case vk::Format::eA2R10G10B10SscaledPack32:
	case vk::Format::eA2R10G10B10UintPack32:
	case vk::Format::eA2R10G10B10SintPack32:
	case vk::Format::eA2B10G10R10UnormPack32:
	case vk::Format::eA2B10G10R10SnormPack32:
	case vk::Format::eA2B10G10R10UscaledPack32:
	case vk::Format::eA2B10G10R10SscaledPack32:
	case vk::Format::eA2B10G10R10UintPack32:
	case vk::Format::eA2B10G10R10SintPack32:
	case vk::Format::eR16G16Unorm:
	case vk::Format::eR16G16Snorm:
	case vk::Format::eR16G16Uscaled:
	case vk::Format::eR16G16Sscaled:
	case vk::Format::eR16G16Uint:
	case vk::Format::eR16G16Sint:
	case vk::Format::eR16G16Sfloat:
	case vk::Format::eR32Uint:
	case vk::Format::eR32Sint:
	case vk::Format::eR32Sfloat:
	case vk::Format::eD24UnormS8Uint:
	case vk::Format::eD32Sfloat:
		return 4;

	case vk::Format::eD32SfloatS8Uint:
		return 5;

	case vk::Format::eR16G16B16Unorm:
	case vk::Format::eR16G16B16Snorm:
	case vk::Format::eR16G16B16Uscaled:
	case vk::Format::eR16G16B16Sscaled:
	case vk::Format::eR16G16B16Uint:
	case vk::Format::eR16G16B16Sint:
	case vk::Format::eR16G16B16Sfloat:
		return 6;

	case vk::Format::eR16G16B16A16Unorm:
	case vk::Format::eR16G16B16A16Snorm:
	case vk::Format::eR16G16B16A16Uscaled:
	case vk::Format::eR16G16B16A16Sscaled:
	case vk::Format::eR16G16B16A16Uint:
	case vk::Format::eR16G16B16A16Sint:
	case vk::Format::eR16G16B16A16Sfloat:
	case vk::Format::eR32G32Uint:
	case vk::Format::eR32G32Sint:
	case vk::Format::eR32G32Sfloat:
	case vk::Format::eR64Uint:
	case vk::Format::eR64Sint:
	case vk::Format::eR64Sfloat:
		return 8;

	case vk::Format::eR32G32B32Uint:
	case vk::Format::eR32G32B32Sint:
	case vk::Format::eR32G32B32Sfloat:
		return 12;

	case vk::Format::eR32G32B32A32Uint:
	case vk::Format::eR32G32B32A32Sint:
	case vk::Format::eR32G32B32A32Sfloat:
	case vk::Format::eR64G64Uint:
	case vk::Format::eR64G64Sint:
	case vk::Format::eR64G64Sfloat:
		return 16;

	case vk::Format::eR64G64B64Uint:
	case vk::Format::eR64G64B64Sint:
	case vk::Format::eR64G64B64Sfloat:
		return 24;

	case vk::Format::eR64G64B64A64Uint:
	case vk::Format::eR64G64B64A64Sint:
	case vk::Format::eR64G64B64A64Sfloat:
		return 32;

	}
	return 0;
}

template<typename T = uint32_t> requires(std::is_arithmetic_v<T>)
inline constexpr T GetChannelCount(const vk::Format format) {
	switch (format) {
	default:
		throw std::runtime_error(std::string("Channel count unknown for format ") + vk::to_string(format));
	case vk::Format::eR8Unorm:
	case vk::Format::eR8Snorm:
	case vk::Format::eR8Uscaled:
	case vk::Format::eR8Sscaled:
	case vk::Format::eR8Uint:
	case vk::Format::eR8Sint:
	case vk::Format::eR8Srgb:
	case vk::Format::eR16Unorm:
	case vk::Format::eR16Snorm:
	case vk::Format::eR16Uscaled:
	case vk::Format::eR16Sscaled:
	case vk::Format::eR16Uint:
	case vk::Format::eR16Sint:
	case vk::Format::eR16Sfloat:
	case vk::Format::eR32Uint:
	case vk::Format::eR32Sint:
	case vk::Format::eR32Sfloat:
	case vk::Format::eR64Uint:
	case vk::Format::eR64Sint:
	case vk::Format::eR64Sfloat:
	case vk::Format::eD16Unorm:
	case vk::Format::eD32Sfloat:
	case vk::Format::eD16UnormS8Uint:
	case vk::Format::eD24UnormS8Uint:
	case vk::Format::eX8D24UnormPack32:
	case vk::Format::eS8Uint:
	case vk::Format::eD32SfloatS8Uint:
	case vk::Format::eBc4UnormBlock:
	case vk::Format::eBc4SnormBlock:
		return 1;
	case vk::Format::eR4G4UnormPack8:
	case vk::Format::eR8G8Unorm:
	case vk::Format::eR8G8Snorm:
	case vk::Format::eR8G8Uscaled:
	case vk::Format::eR8G8Sscaled:
	case vk::Format::eR8G8Uint:
	case vk::Format::eR8G8Sint:
	case vk::Format::eR8G8Srgb:
	case vk::Format::eR16G16Unorm:
	case vk::Format::eR16G16Snorm:
	case vk::Format::eR16G16Uscaled:
	case vk::Format::eR16G16Sscaled:
	case vk::Format::eR16G16Uint:
	case vk::Format::eR16G16Sint:
	case vk::Format::eR16G16Sfloat:
	case vk::Format::eR32G32Uint:
	case vk::Format::eR32G32Sint:
	case vk::Format::eR32G32Sfloat:
	case vk::Format::eR64G64Uint:
	case vk::Format::eR64G64Sint:
	case vk::Format::eR64G64Sfloat:
	case vk::Format::eBc5UnormBlock:
	case vk::Format::eBc5SnormBlock:
		return 2;
	case vk::Format::eR4G4B4A4UnormPack16:
	case vk::Format::eB4G4R4A4UnormPack16:
	case vk::Format::eR5G6B5UnormPack16:
	case vk::Format::eB5G6R5UnormPack16:
	case vk::Format::eR8G8B8Unorm:
	case vk::Format::eR8G8B8Snorm:
	case vk::Format::eR8G8B8Uscaled:
	case vk::Format::eR8G8B8Sscaled:
	case vk::Format::eR8G8B8Uint:
	case vk::Format::eR8G8B8Sint:
	case vk::Format::eR8G8B8Srgb:
	case vk::Format::eB8G8R8Unorm:
	case vk::Format::eB8G8R8Snorm:
	case vk::Format::eB8G8R8Uscaled:
	case vk::Format::eB8G8R8Sscaled:
	case vk::Format::eB8G8R8Uint:
	case vk::Format::eB8G8R8Sint:
	case vk::Format::eB8G8R8Srgb:
	case vk::Format::eR16G16B16Unorm:
	case vk::Format::eR16G16B16Snorm:
	case vk::Format::eR16G16B16Uscaled:
	case vk::Format::eR16G16B16Sscaled:
	case vk::Format::eR16G16B16Uint:
	case vk::Format::eR16G16B16Sint:
	case vk::Format::eR16G16B16Sfloat:
	case vk::Format::eR32G32B32Uint:
	case vk::Format::eR32G32B32Sint:
	case vk::Format::eR32G32B32Sfloat:
	case vk::Format::eR64G64B64Uint:
	case vk::Format::eR64G64B64Sint:
	case vk::Format::eR64G64B64Sfloat:
	case vk::Format::eB10G11R11UfloatPack32:
	case vk::Format::eBc1RgbUnormBlock:
	case vk::Format::eBc1RgbSrgbBlock:
	case vk::Format::eBc3UnormBlock:
	case vk::Format::eBc3SrgbBlock:
		return 3;
	case vk::Format::eR5G5B5A1UnormPack16:
	case vk::Format::eB5G5R5A1UnormPack16:
	case vk::Format::eA1R5G5B5UnormPack16:
	case vk::Format::eR8G8B8A8Unorm:
	case vk::Format::eR8G8B8A8Snorm:
	case vk::Format::eR8G8B8A8Uscaled:
	case vk::Format::eR8G8B8A8Sscaled:
	case vk::Format::eR8G8B8A8Uint:
	case vk::Format::eR8G8B8A8Sint:
	case vk::Format::eR8G8B8A8Srgb:
	case vk::Format::eB8G8R8A8Unorm:
	case vk::Format::eB8G8R8A8Snorm:
	case vk::Format::eB8G8R8A8Uscaled:
	case vk::Format::eB8G8R8A8Sscaled:
	case vk::Format::eB8G8R8A8Uint:
	case vk::Format::eB8G8R8A8Sint:
	case vk::Format::eB8G8R8A8Srgb:
	case vk::Format::eA8B8G8R8UnormPack32:
	case vk::Format::eA8B8G8R8SnormPack32:
	case vk::Format::eA8B8G8R8UscaledPack32:
	case vk::Format::eA8B8G8R8SscaledPack32:
	case vk::Format::eA8B8G8R8UintPack32:
	case vk::Format::eA8B8G8R8SintPack32:
	case vk::Format::eA8B8G8R8SrgbPack32:
	case vk::Format::eA2R10G10B10UnormPack32:
	case vk::Format::eA2R10G10B10SnormPack32:
	case vk::Format::eA2R10G10B10UscaledPack32:
	case vk::Format::eA2R10G10B10SscaledPack32:
	case vk::Format::eA2R10G10B10UintPack32:
	case vk::Format::eA2R10G10B10SintPack32:
	case vk::Format::eA2B10G10R10UnormPack32:
	case vk::Format::eA2B10G10R10SnormPack32:
	case vk::Format::eA2B10G10R10UscaledPack32:
	case vk::Format::eA2B10G10R10SscaledPack32:
	case vk::Format::eA2B10G10R10UintPack32:
	case vk::Format::eA2B10G10R10SintPack32:
	case vk::Format::eR16G16B16A16Unorm:
	case vk::Format::eR16G16B16A16Snorm:
	case vk::Format::eR16G16B16A16Uscaled:
	case vk::Format::eR16G16B16A16Sscaled:
	case vk::Format::eR16G16B16A16Uint:
	case vk::Format::eR16G16B16A16Sint:
	case vk::Format::eR16G16B16A16Sfloat:
	case vk::Format::eR32G32B32A32Uint:
	case vk::Format::eR32G32B32A32Sint:
	case vk::Format::eR32G32B32A32Sfloat:
	case vk::Format::eR64G64B64A64Uint:
	case vk::Format::eR64G64B64A64Sint:
	case vk::Format::eR64G64B64A64Sfloat:
	case vk::Format::eE5B9G9R9UfloatPack32:
	case vk::Format::eBc1RgbaUnormBlock:
	case vk::Format::eBc1RgbaSrgbBlock:
	case vk::Format::eBc2UnormBlock:
	case vk::Format::eBc2SrgbBlock:
		return 4;
	}
}

}
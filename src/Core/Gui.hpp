#pragma once

#include <imgui/imgui.h>

#include <Core/CommandBuffer.hpp>

namespace ptvk {

class Swapchain;

class Gui {
public:
	template<typename T>
	inline static ImGuiDataType GetImGuiDataType() {
		if constexpr(std::is_floating_point_v<T>)
			return sizeof(T) == sizeof(float) ? ImGuiDataType_Float : ImGuiDataType_Double;
		else if constexpr (sizeof(T) == sizeof(uint64_t))
			return std::is_signed_v<T> ? ImGuiDataType_S64 : ImGuiDataType_U64;
		else if constexpr (sizeof(T) == sizeof(uint32_t))
			return std::is_signed_v<T> ? ImGuiDataType_S32 : ImGuiDataType_U32;
		else if constexpr (sizeof(T) == sizeof(uint16_t))
			return std::is_signed_v<T> ? ImGuiDataType_S16 : ImGuiDataType_U16;
		else
			return ImGuiDataType_COUNT;
	}

	template<typename T>
	inline static bool ScalarField(const std::string& label, T* ptr, const T& mn = 0, const T& mx = 0, const float dragSpeed = 1) {
		if (dragSpeed == 0 && mn != mx) {
			ImGui::SetNextItemWidth(75);
			return ImGui::SliderScalar(label.c_str(), GetImGuiDataType<T>(), ptr, &mn, &mx);
		} else {
			ImGui::SetNextItemWidth(50);
			return ImGui::DragScalar(label.c_str(), GetImGuiDataType<T>(), ptr, dragSpeed, &mn, &mx);
		}
	}

	inline static bool ScalarField(const std::string& label, const vk::Format format, void* data) {
		static std::unordered_map<vk::Format, std::pair<ImGuiDataType, int>> sFormatMap = {
			{ vk::Format::eR8Uint,             { ImGuiDataType_U8, 1 } },
			{ vk::Format::eR8G8Uint,           { ImGuiDataType_U8, 2 } },
			{ vk::Format::eR8G8B8Uint,         { ImGuiDataType_U8, 3 } },
			{ vk::Format::eR8G8B8A8Uint,       { ImGuiDataType_U8, 4 } },

			{ vk::Format::eR8Sint,             { ImGuiDataType_S8, 1 } },
			{ vk::Format::eR8G8Sint,           { ImGuiDataType_S8, 2 } },
			{ vk::Format::eR8G8B8Sint,         { ImGuiDataType_S8, 3 } },
			{ vk::Format::eR8G8B8A8Sint,       { ImGuiDataType_S8, 4 } },

			{ vk::Format::eR16Uint,            { ImGuiDataType_U16, 1 } },
			{ vk::Format::eR16G16Uint,         { ImGuiDataType_U16, 2 } },
			{ vk::Format::eR16G16B16Uint,      { ImGuiDataType_U16, 3 } },
			{ vk::Format::eR16G16B16A16Uint,   { ImGuiDataType_U16, 4 } },

			{ vk::Format::eR16Sint,            { ImGuiDataType_S16, 1 } },
			{ vk::Format::eR16G16Sint,         { ImGuiDataType_S16, 2 } },
			{ vk::Format::eR16G16B16Sint,      { ImGuiDataType_S16, 3 } },
			{ vk::Format::eR16G16B16A16Sint,   { ImGuiDataType_S16, 4 } },

			{ vk::Format::eR32Uint,            { ImGuiDataType_U32, 1 } },
			{ vk::Format::eR32G32Uint,         { ImGuiDataType_U32, 2 } },
			{ vk::Format::eR32G32B32Uint,      { ImGuiDataType_U32, 3 } },
			{ vk::Format::eR32G32B32A32Uint,   { ImGuiDataType_U32, 4 } },

			{ vk::Format::eR32Sint,            { ImGuiDataType_S32, 1 } },
			{ vk::Format::eR32G32Sint,         { ImGuiDataType_S32, 2 } },
			{ vk::Format::eR32G32B32Sint,      { ImGuiDataType_S32, 3 } },
			{ vk::Format::eR32G32B32A32Sint,   { ImGuiDataType_S32, 4 } },

			{ vk::Format::eR32Sfloat,          { ImGuiDataType_Float, 1 } },
			{ vk::Format::eR32G32Sfloat,       { ImGuiDataType_Float, 2 } },
			{ vk::Format::eR32G32B32Sfloat,    { ImGuiDataType_Float, 3 } },
			{ vk::Format::eR32G32B32A32Sfloat, { ImGuiDataType_Float, 4 } },

			{ vk::Format::eR64Sfloat,          { ImGuiDataType_Double, 1 } },
			{ vk::Format::eR64G64Sfloat,       { ImGuiDataType_Double, 2 } },
			{ vk::Format::eR64G64B64Sfloat,    { ImGuiDataType_Double, 3 } },
			{ vk::Format::eR64G64B64A64Sfloat, { ImGuiDataType_Double, 4 } }
		};
		const auto&[dataType, components] = sFormatMap.at(format);
		ImGui::SetNextItemWidth(50);
		return ImGui::InputScalarN(label.c_str(), dataType, data, components);

	}

	template<typename T>
	inline static bool EnumDropdown(const std::string& label, T& selected, std::span<const char* const> strings) {
		bool ret = false;
		const std::string previewstr = strings[(uint32_t)selected];
		if (ImGui::BeginCombo(label.c_str(), previewstr.c_str())) {
			for (uint32_t i = 0; i < strings.size(); i++) {
				if (ImGui::Selectable(strings[i], (uint32_t)selected == i)) {
					selected = (T)i;
					ret = true;
				}
			}
			ImGui::EndCombo();
		}
		return ret;
	}

	static void ProgressSpinner(const char* label, const float radius = 15, const float thickness = 6, const bool center = true);

	static ImTextureID GetTextureID(const Image::View& image, const vk::Filter filter = vk::Filter::eLinear);
	static ImFont* GetHeaderFont() { return mHeaderFont; }

	/////////////////////////////

	static void Initialize(const Swapchain& swapchain, const vk::Queue queue, const uint32_t queueFamily);
	static void Destroy();

	static void NewFrame();

	// converts renderTarget to ColorAttachmentOptimal before rendering
	static void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget);

private:
	static vk::raii::RenderPass mRenderPass;
	static uint32_t mQueueFamily;
	static std::unordered_map<vk::Image, vk::raii::Framebuffer> mFramebuffers;
	static std::shared_ptr<vk::raii::DescriptorPool> mImGuiDescriptorPool;
	static ImFont* mHeaderFont;

	static std::unordered_set<Image::View> mFrameTextures;
	static std::unordered_map<
		std::pair<Image::View, vk::Filter>,
		std::pair<vk::raii::DescriptorSet, vk::raii::Sampler>,
		PairHash<Image::View, vk::Filter>> mTextureIDs;
};

}
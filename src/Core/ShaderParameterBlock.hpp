#pragma once

#include <variant>

#include "Image.hpp"

namespace ptvk {

// represents a loose constant (eg a float or vector)
class ConstantParameter : public std::vector<std::byte> {
public:
	template<typename T>
	inline ConstantParameter(const T& value) {
		resize(sizeof(value));
		*reinterpret_cast<T*>(data()) = value;
	}

	ConstantParameter() = default;
	ConstantParameter(ConstantParameter&&) = default;
	ConstantParameter(const ConstantParameter&) = default;
	ConstantParameter& operator=(ConstantParameter&&) = default;
	ConstantParameter& operator=(const ConstantParameter&) = default;

	template<typename T>
	inline T& get() {
		if (empty())
			resize(sizeof(T));
		return *reinterpret_cast<T*>(data());
	}

	template<typename T>
	inline const T& get() const {
		return *reinterpret_cast<const T*>(data());
	}


	template<typename T>
	inline T& operator=(const T& value) {
		resize(sizeof(value));
		return *reinterpret_cast<T*>(data()) = value;
	}
};

using BufferParameter                = Buffer::View<std::byte>;
using ImageParameter                 = std::tuple<Image::View, vk::ImageLayout, vk::AccessFlags, std::shared_ptr<vk::raii::Sampler>>;
using AccelerationStructureParameter = std::shared_ptr<vk::raii::AccelerationStructureKHR>;
using ShaderParameterValue = std::variant< ConstantParameter, BufferParameter, ImageParameter, AccelerationStructureParameter >;

class ShaderParameterBlock : public std::unordered_map<std::pair<std::string, uint32_t>, ShaderParameterValue, PairHash<std::string, uint32_t>> {
public:
	inline bool Contains(const std::string& id, const uint32_t arrayIndex = 0) const {
		return contains(std::make_pair(id, arrayIndex));
	}

	inline const ShaderParameterValue& operator()(const std::string& id, const uint32_t arrayIndex = 0) const {
		return at(std::make_pair(id, arrayIndex));
	}
	inline       ShaderParameterValue& operator()(const std::string& id, const uint32_t arrayIndex = 0) {
		return operator[](std::make_pair(id, arrayIndex));
	}

	template<typename T>
	inline T& GetConstant(const std::string& id, const uint32_t arrayIndex = 0) {
		return std::get<ConstantParameter>(operator()(id, arrayIndex)).get<T>();
	}
	template<typename T>
	inline const T& GetConstant(const std::string& id, const uint32_t arrayIndex = 0) const {
		return std::get<ConstantParameter>(operator()(id, arrayIndex)).get<T>();
	}
	inline ShaderParameterBlock& SetConstant(const std::string& id, const uint32_t arrayIndex, const ConstantParameter& v) {
		operator()(id,arrayIndex) = v;
		return *this;
	}
	inline ShaderParameterBlock& SetConstant(const std::string& id, const ConstantParameter& v) { return SetConstant(id, 0, v); }

	template<typename T>
	inline Buffer::View<T> GetBuffer(const std::string& id, const uint32_t arrayIndex = 0) const {
		return std::get<BufferParameter>(operator()(id, arrayIndex)).Cast<T>();
	}
	inline ShaderParameterBlock& SetBuffer(const std::string& id, const uint32_t arrayIndex, const BufferParameter& v) {
		operator()(id,arrayIndex) = v;
		return *this;
	}
	inline ShaderParameterBlock& SetBuffer(const std::string& id, const BufferParameter& v) { return SetBuffer(id, 0, v); }

	inline const ImageParameter& GetImage(const std::string& id, const uint32_t arrayIndex = 0) const {
		return std::get<ImageParameter>(operator()(id, arrayIndex));
	}
	inline ShaderParameterBlock& SetImage(const std::string& id, const uint32_t arrayIndex, const ImageParameter& v) {
		operator()(id,arrayIndex) = v;
		return *this;
	}
	inline ShaderParameterBlock& SetImage(const std::string& id, const ImageParameter& v) { return SetImage(id, 0, v); }
	inline ShaderParameterBlock& SetImage(const std::string& id, const uint32_t arrayIndex, const Image::View image, const vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal, const vk::AccessFlags access = vk::AccessFlagBits::eShaderRead, const std::shared_ptr<vk::raii::Sampler>& sampler = {}) {
		return SetImage(id, arrayIndex, ImageParameter{ image, layout, access, sampler });
	}
	inline ShaderParameterBlock& SetImage(const std::string& id, const Image::View image, const vk::ImageLayout layout = vk::ImageLayout::eShaderReadOnlyOptimal, const vk::AccessFlags access = vk::AccessFlagBits::eShaderRead, const std::shared_ptr<vk::raii::Sampler>& sampler = {}) {
		return SetImage(id, ImageParameter{ image, layout, access, sampler });
	}

	inline const AccelerationStructureParameter& GetAccelerationStructure(const std::string& id, const uint32_t arrayIndex = 0) const {
		return std::get<AccelerationStructureParameter>(operator()(id, arrayIndex));
	}
	inline ShaderParameterBlock& SetAccelerationStructure(const std::string& id, const uint32_t arrayIndex, const AccelerationStructureParameter& v) {
		operator()(id,arrayIndex) = v;
		return *this;
	}
	inline ShaderParameterBlock& SetAccelerationStructure(const std::string& id, const AccelerationStructureParameter& v) { return SetAccelerationStructure(id, 0, v); }

	inline ShaderParameterBlock& SetParameters(const ShaderParameterBlock& params) {
		for (const auto&[key, val] : params)
			operator[](key) = val;
		return *this;
	}
	inline ShaderParameterBlock& SetParameters(const std::string& id, const ShaderParameterBlock& params) {
		for (const auto&[key, val] : params)
			operator[]({id + "." + key.first, key.second}) = val;
		return *this;
	}
};

}
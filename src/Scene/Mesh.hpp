#pragma once

#include <Core/CommandBuffer.hpp>

namespace ptvk {

class Mesh {
public:
	enum class VertexAttributeType {
		ePosition,
		eNormal,
		eTangent,
		eBinormal,
		eColor,
		eTexcoord,
		ePointSize,
		eBlendIndex,
		eBlendWeight
	};
	struct VertexAttributeDescription {
		uint32_t mStride;
		vk::Format mFormat;
		uint32_t mOffset;
		vk::VertexInputRate mInputRate;
	};
	using VertexAttributeData = std::pair<Buffer::View<std::byte>, VertexAttributeDescription>;

	struct VertexLayoutDescription {
		std::unordered_map<VertexAttributeType, std::vector<std::pair<VertexAttributeDescription, uint32_t/*binding index*/>>> mAttributes;
		vk::PrimitiveTopology mTopology;
		vk::IndexType mIndexType;

		VertexLayoutDescription() = default;
		VertexLayoutDescription(const VertexLayoutDescription&) = default;
		VertexLayoutDescription(VertexLayoutDescription&&) = default;
		VertexLayoutDescription& operator=(const VertexLayoutDescription&) = default;
		VertexLayoutDescription& operator=(VertexLayoutDescription&&) = default;
		inline VertexLayoutDescription(vk::PrimitiveTopology topo, vk::IndexType indexType = vk::IndexType::eUint16) : mTopology(topo), mIndexType(indexType) {}
	};

	class Vertices : public std::unordered_map<VertexAttributeType, std::vector<VertexAttributeData>> {
	public:
		inline std::optional<VertexAttributeData> find(const VertexAttributeType t, uint32_t index = 0) const {
			auto it = std::unordered_map<VertexAttributeType, std::vector<VertexAttributeData>>::find(t);
			if (it != end() && it->second.size() > index)
				return it->second[index];
			else
				return std::nullopt;
		}

		void Bind(CommandBuffer& commandBuffer) const;

		vk::AabbPositionsKHR mAabb;
	};

	Mesh() = default;
	Mesh(const Mesh&) = default;
	Mesh(Mesh&&) = default;
	Mesh& operator=(const Mesh&) = default;
	Mesh& operator=(Mesh&&) = default;
	inline Mesh(const Vertices& vertices, const Buffer::StrideView& indices, const vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList)
		: mVertices(vertices), mIndices(indices), mTopology(topology) {}
	inline Mesh(Vertices&& vertices, const Buffer::StrideView& indices, const vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList)
		: mVertices(move(vertices)), mIndices(indices), mTopology(topology) {}

	inline const Vertices& GetVertices() const { return mVertices; }
	inline const Buffer::StrideView& GetIndices() const { return mIndices; }
	inline const vk::PrimitiveTopology GetTopology() const { return mTopology; }
	inline const vk::IndexType GetIndexType() const {
		return (mIndices.Stride() == sizeof(uint32_t)) ? vk::IndexType::eUint32 : (mIndices.Stride() == sizeof(uint16_t)) ? vk::IndexType::eUint16 : vk::IndexType::eUint8EXT;
	}

	VertexLayoutDescription GetVertexLayout(const Shader& vertexShader) const;

	inline void Bind(CommandBuffer& commandBuffer) const{
		mVertices.Bind(commandBuffer);
		commandBuffer->bindIndexBuffer(**mIndices.GetBuffer(), mIndices.Offset(), GetIndexType());
	}

private:
	Vertices mVertices;
	Buffer::StrideView mIndices;
	vk::PrimitiveTopology mTopology = vk::PrimitiveTopology::eTriangleList;
};

}

namespace std {
template<>
struct hash<ptvk::Mesh::VertexAttributeDescription> {
	inline size_t operator()(const ptvk::Mesh::VertexAttributeDescription& v) const {
		return ptvk::HashArgs(v.mFormat, v.mOffset, v.mInputRate);
	}
};

template<>
struct hash<ptvk::Mesh::VertexLayoutDescription> {
	inline size_t operator()(const ptvk::Mesh::VertexLayoutDescription& v) const {
		size_t h = 0;
		for (const auto[type, attribs] : v.mAttributes) {
			h = ptvk::HashArgs(h, type);
			for (const auto&[a,i] : attribs)
				h = ptvk::HashArgs(h, a, i);
		}
		return ptvk::HashArgs(h, v.mTopology, v.mIndexType);
	}
};

inline std::string to_string(const ptvk::Mesh::VertexAttributeType& value) {
	switch (value) {
		case ptvk::Mesh::VertexAttributeType::ePosition:    return "Position";
		case ptvk::Mesh::VertexAttributeType::eNormal:      return "Normal";
		case ptvk::Mesh::VertexAttributeType::eTangent:     return "Tangent";
		case ptvk::Mesh::VertexAttributeType::eBinormal:    return "Binormal";
		case ptvk::Mesh::VertexAttributeType::eBlendIndex:  return "BlendIndex";
		case ptvk::Mesh::VertexAttributeType::eBlendWeight: return "BlendWeight";
		case ptvk::Mesh::VertexAttributeType::eColor:       return "Color";
		case ptvk::Mesh::VertexAttributeType::ePointSize:   return "PointSize";
		case ptvk::Mesh::VertexAttributeType::eTexcoord:    return "Texcoord";
		default: return "invalid ( " + vk::toHexString( static_cast<uint32_t>( value ) ) + " )";
	}
}
}
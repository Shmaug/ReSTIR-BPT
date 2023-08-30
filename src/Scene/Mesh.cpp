#include "Mesh.hpp"

namespace ptvk {

Mesh::VertexLayoutDescription Mesh::GetVertexLayout(const Shader& vertexShader) const {
	struct stride_view_hash {
		inline size_t operator()(const Buffer::StrideView& v) const {
			return HashArgs(v.GetBuffer().get(), v.Offset(), v.SizeBytes(), v.Stride());
		}
	};

	std::unordered_map<Buffer::StrideView, uint32_t, stride_view_hash> uniqueBuffers;

	VertexLayoutDescription layout(mTopology, GetIndexType());

	for (const auto&[id, v] : vertexShader.GetInputVariables()) {
		static const std::unordered_map<std::string, VertexAttributeType> attributeTypeMap {
			{ "position",    VertexAttributeType::ePosition },
			{ "normal",      VertexAttributeType::eNormal },
			{ "tangent",     VertexAttributeType::eTangent },
			{ "binormal",    VertexAttributeType::eBinormal },
			{ "color",       VertexAttributeType::eColor },
			{ "texcoord",    VertexAttributeType::eTexcoord },
			{ "pointsize",   VertexAttributeType::ePointSize },
			{ "blendindex",  VertexAttributeType::eBlendIndex },
			{ "blendweight", VertexAttributeType::eBlendWeight }
		};

		VertexAttributeType attributeType = VertexAttributeType::eTexcoord;
		std::string s = v.mSemantic;
		std::ranges::transform(v.mSemantic, s.begin(), static_cast<int(*)(int)>(std::tolower));
		if (auto it = attributeTypeMap.find(s); it != attributeTypeMap.end())
			attributeType = it->second;
		else
			std::cout << "Warning, unknown variable semantic " << v.mSemantic << std::endl;

		std::optional<VertexAttributeData> attrib = mVertices.find(attributeType, v.mSemanticIndex);
		if (!attrib) throw std::logic_error("Mesh does not contain required shader input " + std::to_string(attributeType) + "." + std::to_string(v.mSemanticIndex));

		// get/create attribute in attribute array
		auto& dstAttribs = layout.mAttributes[attributeType];
		if (dstAttribs.size() <= v.mSemanticIndex)
		dstAttribs.resize(v.mSemanticIndex + 1);

		auto&[vertexBuffer, attributeDescription] = *attrib;

		// store attribute description
		dstAttribs[v.mSemanticIndex].first = attributeDescription;

		// get unique binding index for buffer
		if (auto it = uniqueBuffers.find(vertexBuffer); it != uniqueBuffers.end())
			dstAttribs[v.mSemanticIndex].second = it->second;
		else {
			dstAttribs[v.mSemanticIndex].second = (uint32_t)uniqueBuffers.size();
			uniqueBuffers.emplace(vertexBuffer, dstAttribs[v.mSemanticIndex].second);
		}
	}

	return layout;
}

}
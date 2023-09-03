#include <Scene/Scene.hpp>

#define TINYGLTF_USE_CPP14
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>

namespace ptvk {

Material Scene::CreateMetallicRoughnessMaterial(CommandBuffer& commandBuffer, const ImageValue3& baseColor, const ImageValue4& metallicRoughness, const ImageValue3& emission) {
	Material m;
	m.mMaterial.BaseColor(baseColor.first);
	m.mMaterial.Emission(emission.first);
	m.mMaterial.Metallic(metallicRoughness.first.z);
	m.mMaterial.Roughness(metallicRoughness.first.y);
	m.mMaterial.Anisotropic(0);
	m.mMaterial.Subsurface(0);
	m.mMaterial.Clearcoat(0);
	m.mMaterial.ClearcoatGloss(0);
	m.mMaterial.Transmission(0);
	m.mMaterial.Eta(1.5f);
	m.mMaterial.Sheen(0);
	m.mMaterial.Specular(0.5f);
	m.mMaterial.AlphaCutoff(0.5f);
	m.mMaterial.BumpScale(1);
	m.mBaseColor = baseColor.second;
	m.mEmission = emission.second;
	/*
	if (baseColor.second) {
		Buffer::View<uint32_t> minAlpha = std::make_shared<Buffer>(commandBuffer.mDevice, "mMinAlpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc);
		commandBuffer.HoldResource(minAlpha);
		commandBuffer.Fill(minAlpha, 255);
		m.mMinAlpha = std::make_shared<Buffer>(commandBuffer.mDevice, "mMinAlpha", sizeof(uint32_t), vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		mComputeMinAlphaPipeline.Dispatch(commandBuffer, baseColor.second.GetExtent(), ShaderParameterBlock()
			.SetImage("gDiffuse", baseColor.second, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead)
			.SetBuffer("gOutputMinAlpha", minAlpha)
		);
		commandBuffer.Copy(minAlpha, m.mMinAlpha);
	}
	if (metallicRoughness.second) {
		ImageInfo md;
		md.mExtent = metallicRoughness.second.GetExtent();
		md.mLevels = GetMaxMipLevels(md.mExtent);
		md.mFormat = vk::Format::eR8G8B8A8Unorm;
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;
		m.mPackedParams = std::make_shared<Image>(commandBuffer.mDevice, "PackedMaterialData", md);

		mConvertMetallicRoughnessPipeline.Dispatch(commandBuffer, metallicRoughness.second.GetExtent(), ShaderParameterBlock()
			.SetImage("gSpecular", metallicRoughness.second, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead)
			.SetImage("gOutput", m.mPackedParams, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
		);

		commandBuffer.GenerateMipMaps(m.mPackedParams.GetImage());
	}*/
	return m;
}

std::shared_ptr<SceneNode> Scene::LoadGltf(CommandBuffer& commandBuffer, const std::filesystem::path& filename) {
	std::cout << "Loading " << filename << std::endl;

	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	std::string err, warn;
	if (
		(filename.extension() == ".glb" && !loader.LoadBinaryFromFile(&model, &err, &warn, filename.string())) ||
		(filename.extension() == ".gltf" && !loader.LoadASCIIFromFile(&model, &err, &warn, filename.string())) )
		throw std::runtime_error(filename.string() + ": " + err);
	if (!warn.empty()) std::cerr << filename.string() << ": " << warn << std::endl;

	Device& device = commandBuffer.mDevice;

	std::cout << "Loading buffers..." << std::endl;

	std::vector<std::shared_ptr<Buffer>> buffers(model.buffers.size());
	vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eVertexBuffer|vk::BufferUsageFlagBits::eIndexBuffer|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eTransferSrc;
	if (commandBuffer.mDevice.GetAccelerationStructureFeatures().accelerationStructure) {
		bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
		bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	}
	std::ranges::transform(model.buffers, buffers.begin(), [&](const tinygltf::Buffer& buffer) {
		Buffer::View<unsigned char> tmp = std::make_shared<Buffer>(device, buffer.name+"/Staging", buffer.data.size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		std::ranges::copy(buffer.data, tmp.begin());
		Buffer::View<unsigned char> dst = std::make_shared<Buffer>(device, buffer.name, buffer.data.size(), bufferUsage, vk::MemoryPropertyFlagBits::eDeviceLocal);
		commandBuffer.HoldResource(tmp);
		commandBuffer.HoldResource(dst);
		commandBuffer.Copy(tmp, dst);
		return dst.GetBuffer();
	});

	std::cout << "Loading materials..." << std::endl;
	std::vector<Image::View> images(model.images.size());
	auto GetImage = [&](const uint32_t textureIndex, const bool srgb) -> Image::View {
		if (textureIndex >= model.textures.size()) return {};
		const uint32_t index = model.textures[textureIndex].source;
		if (index >= images.size()) return {};
		if (images[index]) return images[index];

		const tinygltf::Image& image = model.images[index];
		Buffer::View<unsigned char> pixels = std::make_shared<Buffer>(device, image.name+"/Staging", image.image.size(), vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
		std::ranges::uninitialized_copy(image.image, pixels);


		ImageInfo md = {};
		if (srgb) {
			static const std::array<vk::Format,4> formatMap { vk::Format::eR8Srgb, vk::Format::eR8G8Srgb, vk::Format::eR8G8B8Srgb, vk::Format::eR8G8B8A8Srgb };
			md.mFormat = formatMap.at(image.component - 1);
		} else {
			static const std::unordered_map<int, std::array<vk::Format,4>> formatMap {
				{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE,  { vk::Format::eR8Unorm, vk::Format::eR8G8Unorm, vk::Format::eR8G8B8Unorm, vk::Format::eR8G8B8A8Unorm } },
				{ TINYGLTF_COMPONENT_TYPE_BYTE,           { vk::Format::eR8Snorm, vk::Format::eR8G8Snorm, vk::Format::eR8G8B8Snorm, vk::Format::eR8G8B8A8Snorm } },
				{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, { vk::Format::eR16Unorm, vk::Format::eR16G16Unorm, vk::Format::eR16G16B16Unorm, vk::Format::eR16G16B16A16Unorm } },
				{ TINYGLTF_COMPONENT_TYPE_SHORT,          { vk::Format::eR16Snorm, vk::Format::eR16G16Snorm, vk::Format::eR16G16B16Snorm, vk::Format::eR16G16B16A16Snorm } },
				{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,   { vk::Format::eR32Uint, vk::Format::eR32G32Uint, vk::Format::eR32G32B32Uint, vk::Format::eR32G32B32A32Uint } },
				{ TINYGLTF_COMPONENT_TYPE_INT,            { vk::Format::eR32Sint, vk::Format::eR32G32Sint, vk::Format::eR32G32B32Sint, vk::Format::eR32G32B32A32Sint } },
				{ TINYGLTF_COMPONENT_TYPE_FLOAT,          { vk::Format::eR32Sfloat, vk::Format::eR32G32Sfloat, vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32A32Sfloat } },
				{ TINYGLTF_COMPONENT_TYPE_DOUBLE,         { vk::Format::eR64Sfloat, vk::Format::eR64G64Sfloat, vk::Format::eR64G64B64Sfloat, vk::Format::eR64G64B64A64Sfloat } }
			};
			md.mFormat = formatMap.at(image.pixel_type).at(image.component - 1);
		}

		md.mExtent = vk::Extent3D(image.width, image.height, 1);
		md.mLevels = GetMaxMipLevels(md.mExtent);
		const std::shared_ptr<Image> img = std::make_shared<Image>(device, image.name, md);

		commandBuffer.Copy(pixels, img);
		commandBuffer.GenerateMipMaps(img);

		commandBuffer.HoldResource(pixels);
		commandBuffer.HoldResource(img);

		images[index] = img;
		return img;
	};

	std::vector<std::shared_ptr<Material>> materials(model.materials.size());
	std::ranges::transform(model.materials, materials.begin(), [&](const tinygltf::Material& material) {
		ImageValue3 emission{
			(float3)double3(material.emissiveFactor[0], material.emissiveFactor[1], material.emissiveFactor[2]),
			GetImage(material.emissiveTexture.index, true) };
		if (material.extensions.find("KHR_materials_emissive_strength") != material.extensions.end())
			emission.first *= (float)material.extensions.at("KHR_materials_emissive_strength").Get("emissiveStrength").GetNumberAsDouble();

		const ImageValue3 baseColor{
			(float3)double3(material.pbrMetallicRoughness.baseColorFactor[0], material.pbrMetallicRoughness.baseColorFactor[1], material.pbrMetallicRoughness.baseColorFactor[2]),
			GetImage(material.pbrMetallicRoughness.baseColorTexture.index, true) };
		const ImageValue4 metallicRoughness{
			(float4)double4(0, material.pbrMetallicRoughness.roughnessFactor, material.pbrMetallicRoughness.metallicFactor, 0),
			GetImage(material.pbrMetallicRoughness.metallicRoughnessTexture.index, false) };
		const float eta = material.extensions.contains("KHR_materials_ior") ? (float)material.extensions.at("KHR_materials_ior").Get("ior").GetNumberAsDouble() : 1.5f;
		const float transmission = material.extensions.contains("KHR_materials_transmission") ? (float)material.extensions.at("KHR_materials_transmission").Get("transmissionFactor").GetNumberAsDouble() : 0;

		Material m = CreateMetallicRoughnessMaterial(commandBuffer, baseColor, metallicRoughness, emission);
		m.mBumpMap = GetImage(material.normalTexture.index, false);
		m.mMaterial.BumpScale(1);

		if (material.extensions.contains("KHR_materials_clearcoat")) {
			const auto& v = material.extensions.at("KHR_materials_clearcoat");
			m.mMaterial.Clearcoat((float)v.Get("clearcoatFactor").GetNumberAsDouble());
		}

		if (material.extensions.contains("KHR_materials_specular")) {
			const auto& v = material.extensions.at("KHR_materials_specular");
			if (v.Has("specularColorFactor")) {
				auto& a = v.Get("specularColorFactor");
				m.mMaterial.Specular(Luminance((float3)double3(a.Get(0).GetNumberAsDouble(), a.Get(1).GetNumberAsDouble(), a.Get(2).GetNumberAsDouble())));
			} else if (v.Has("specularFactor")) {
				m.mMaterial.Specular((float)v.Get("specularFactor").GetNumberAsDouble());
			}
		}

		return std::make_shared<Material>(m);
	});

	std::cout << "Loading meshes...";
	std::vector<std::vector<std::shared_ptr<Mesh>>> meshes(model.meshes.size());
	for (uint32_t i = 0; i < model.meshes.size(); i++) {
		std::cout << "\rLoading meshes " << (i+1) << "/" << model.meshes.size() << "     ";
		meshes[i].resize(model.meshes[i].primitives.size());
		for (uint32_t j = 0; j < model.meshes[i].primitives.size(); j++) {
			const tinygltf::Primitive& prim = model.meshes[i].primitives[j];
			const auto& indicesAccessor = model.accessors[prim.indices];
			const auto& indexBufferView = model.bufferViews[indicesAccessor.bufferView];
			const size_t indexStride = tinygltf::GetComponentSizeInBytes(indicesAccessor.componentType);
			const Buffer::StrideView indexBuffer = Buffer::StrideView(buffers[indexBufferView.buffer], indexStride, indexBufferView.byteOffset + indicesAccessor.byteOffset, indicesAccessor.count * indexStride);

			Mesh::Vertices vertexData;

			vk::PrimitiveTopology topology;
			switch (prim.mode) {
				case TINYGLTF_MODE_POINTS: 			topology = vk::PrimitiveTopology::ePointList; break;
				case TINYGLTF_MODE_LINE: 			topology = vk::PrimitiveTopology::eLineList; break;
				case TINYGLTF_MODE_LINE_LOOP: 		topology = vk::PrimitiveTopology::eLineStrip; break;
				case TINYGLTF_MODE_LINE_STRIP: 		topology = vk::PrimitiveTopology::eLineStrip; break;
				case TINYGLTF_MODE_TRIANGLES: 		topology = vk::PrimitiveTopology::eTriangleList; break;
				case TINYGLTF_MODE_TRIANGLE_STRIP: 	topology = vk::PrimitiveTopology::eTriangleStrip; break;
				case TINYGLTF_MODE_TRIANGLE_FAN: 	topology = vk::PrimitiveTopology::eTriangleFan; break;
			}

			for (const auto&[attribName,attribIndex] : prim.attributes) {
				const tinygltf::Accessor& accessor = model.accessors[attribIndex];

				static const std::unordered_map<int, std::unordered_map<int, vk::Format>> formatMap {
					{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR8Uint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR8G8Uint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR8G8B8Uint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR8G8B8A8Uint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_BYTE, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR8Sint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR8G8Sint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR8G8B8Sint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR8G8B8A8Sint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR16Uint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR16G16Uint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR16G16B16Uint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR16G16B16A16Uint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_SHORT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR16Sint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR16G16Sint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR16G16B16Sint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR16G16B16A16Sint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR32Uint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR32G32Uint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR32G32B32Uint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR32G32B32A32Uint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_INT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR32Sint },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR32G32Sint },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR32G32B32Sint },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR32G32B32A32Sint },
					} },
					{ TINYGLTF_COMPONENT_TYPE_FLOAT, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR32Sfloat },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR32G32Sfloat },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR32G32B32Sfloat },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR32G32B32A32Sfloat },
					} },
					{ TINYGLTF_COMPONENT_TYPE_DOUBLE, {
						{ TINYGLTF_TYPE_SCALAR, vk::Format::eR64Sfloat },
						{ TINYGLTF_TYPE_VEC2, 	vk::Format::eR64G64Sfloat },
						{ TINYGLTF_TYPE_VEC3, 	vk::Format::eR64G64B64Sfloat },
						{ TINYGLTF_TYPE_VEC4, 	vk::Format::eR64G64B64A64Sfloat },
					} }
				};
				vk::Format attributeFormat = formatMap.at(accessor.componentType).at(accessor.type);

				Mesh::VertexAttributeType attributeType;
				uint32_t typeIndex = 0;
				// parse typename & typeindex
				{
					std::string typeName;
					typeName.resize(attribName.size());
					std::ranges::transform(attribName, typeName.begin(), [&](char c) { return tolower(c); });
					size_t c = typeName.find_first_of("0123456789");
					if (c != std::string::npos) {
						typeIndex = stoi(typeName.substr(c));
						typeName = typeName.substr(0, c);
					}
					if (typeName.back() == '_') typeName.pop_back();
					static const std::unordered_map<std::string, Mesh::VertexAttributeType> semanticMap {
						{ "position", 	Mesh::VertexAttributeType::ePosition },
						{ "normal", 	Mesh::VertexAttributeType::eNormal },
						{ "tangent", 	Mesh::VertexAttributeType::eTangent },
						{ "bitangent", 	Mesh::VertexAttributeType::eBinormal },
						{ "texcoord", 	Mesh::VertexAttributeType::eTexcoord },
						{ "color", 		Mesh::VertexAttributeType::eColor },
						{ "psize", 		Mesh::VertexAttributeType::ePointSize },
						{ "pointsize", 	Mesh::VertexAttributeType::ePointSize },
						{ "joints",     Mesh::VertexAttributeType::eBlendIndex },
						{ "weights",    Mesh::VertexAttributeType::eBlendWeight }
					};
					attributeType = semanticMap.at(typeName);
				}

				auto& attribs = vertexData[attributeType];
				if (attribs.size() <= typeIndex) attribs.resize(typeIndex+1);
				const tinygltf::BufferView& bv = model.bufferViews[accessor.bufferView];
				const uint32_t stride = accessor.ByteStride(bv);
				attribs[typeIndex] = {
					Buffer::View<std::byte>(buffers[bv.buffer], bv.byteOffset + accessor.byteOffset, stride*accessor.count),
					Mesh::VertexAttributeDescription(stride, attributeFormat, 0, vk::VertexInputRate::eVertex) };

				if (attributeType == Mesh::VertexAttributeType::ePosition) {
					vertexData.mAabb.minX = (float)accessor.minValues[0];
					vertexData.mAabb.minY = (float)accessor.minValues[1];
					vertexData.mAabb.minZ = (float)accessor.minValues[2];
					vertexData.mAabb.maxX = (float)accessor.maxValues[0];
					vertexData.mAabb.maxY = (float)accessor.maxValues[1];
					vertexData.mAabb.maxZ = (float)accessor.maxValues[2];
				}
			}

			meshes[i][j] = std::make_shared<Mesh>(vertexData, indexBuffer, topology);
		}
	}
	std::cout << std::endl;

	std::cout << "Loading primitives...";
	const std::shared_ptr<SceneNode> rootNode = SceneNode::Create(filename.stem().string());
	std::vector<std::shared_ptr<SceneNode>> nodes(model.nodes.size());
	for (size_t n = 0; n < model.nodes.size(); n++) {
		std::cout << "\rLoading primitives " << (n+1) << "/" << model.nodes.size() << "     ";

		const auto& node = model.nodes[n];
		const std::shared_ptr<SceneNode> dst = rootNode->AddChild(node.name);
		nodes[n] = dst;

		// compute transform

		if (!node.translation.empty() || !node.rotation.empty() || !node.scale.empty()) {
			float4x4 t = glm::identity<float4x4>();
			if (!node.translation.empty()) t = glm::translate((float3)double3(node.translation[0], node.translation[1], node.translation[2]));
			if (!node.rotation.empty())    t = t * glm::mat4_cast(glm::quat((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]));
			if (!node.scale.empty())       t = t * glm::scale(float3((float)node.scale[0]));
			dst->MakeComponent<float4x4>(t);
		} else if (!node.matrix.empty())
			dst->MakeComponent<float4x4>(transpose((float4x4)*reinterpret_cast<const glm::mat<4,4,double>*>(node.matrix.data())));

		// make node for MeshRenderer

		if (node.mesh < model.meshes.size())
			for (uint32_t i = 0; i < model.meshes[node.mesh].primitives.size(); i++) {
				const auto& prim = model.meshes[node.mesh].primitives[i];
				dst->AddChild(model.meshes[node.mesh].name)->MakeComponent<MeshRenderer>(materials[prim.material], meshes[node.mesh][i]);
			}

		auto light_it = node.extensions.find("KHR_lights_punctual");
		if (light_it != node.extensions.end() && light_it->second.Has("light")) {
			const tinygltf::Light& l = model.lights[light_it->second.Get("light").GetNumberAsInt()];
			if (l.type == "point" && l.extras.Has("radius")) {
				auto sphere = dst->AddChild(l.name)->MakeComponent<SphereRenderer>();
				sphere->mRadius = (float)l.extras.Get("radius").GetNumberAsDouble();
				Material m;
				const float3 emission = (float3)double3(l.color[0], l.color[1], l.color[2]);
				m.mMaterial.BaseColor(float3(0));
				m.mMaterial.Emission(emission * (float)(l.intensity / (4*M_PI*sphere->mRadius*sphere->mRadius)));
				sphere->mMaterial = std::make_shared<Material>(m);
			}
		}
	}
	std::cout << std::endl;

	for (size_t i = 0; i < model.nodes.size(); i++)
		for (int c : model.nodes[i].children)
			nodes[i]->AddChild(nodes[c]);

	std::cout << "Loaded " << filename << std::endl;

	return rootNode;
}

}
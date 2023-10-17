#ifdef ENABLE_ASSIMP

#include <Scene/Scene.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <portable-file-dialogs.h>

namespace ptvk {

Material Scene::CreateDiffuseSpecularMaterial(CommandBuffer& commandBuffer, const ImageValue3& diffuse, const ImageValue3& specular, const ImageValue3& emission) {
	Material m;
	m.mMaterial.BaseColor(diffuse.first);
	m.mMaterial.Emission(emission.first);
	m.mMaterial.Metallic(specular.first.x);
	m.mMaterial.Roughness(specular.first.y);
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
	m.mBaseColor = diffuse.second;
	m.mPackedParams = specular.second;
	m.mEmission = emission.second;
	return m;
}

std::shared_ptr<SceneNode> Scene::LoadAssimp(CommandBuffer& commandBuffer, const std::filesystem::path& filename) {
	std::cout << "Loading " << filename << std::endl;

	Assimp::Importer importer;
	uint32_t flags = aiProcessPreset_TargetRealtime_Fast | aiProcess_TransformUVCoords;

	int removeFlags = aiComponent_COLORS;
	for (uint32_t uvLayer = 1; uvLayer < AI_MAX_NUMBER_OF_TEXTURECOORDS; uvLayer++)
		removeFlags |= aiComponent_TEXCOORDSn(uvLayer);
	importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, removeFlags);

	const aiScene* scene = importer.ReadFile(filename.string(), flags);
	if (!scene) {
		std::cout << "Failed to load " << filename << ": " << importer.GetErrorString() << std::endl;
		return nullptr;
	}

	if (scene->HasLights())
		std::cout << "Warning: punctual lights are unsupported" << std::endl;

	std::unordered_map<std::string, Image::View> imageCache;
	auto GetImage = [&](std::filesystem::path path, const bool srgb) -> Image::View {
		if (path.is_relative()) {
			std::filesystem::path cur = std::filesystem::current_path();
			std::filesystem::current_path(filename.parent_path());
			path = std::filesystem::absolute(path);
			std::filesystem::current_path(cur);
		}
		auto it = imageCache.find(path.string());
		if (it != imageCache.end()) return it->second;

		ImageInfo md = {};
		std::shared_ptr<Buffer> pixels;
		std::tie(pixels, md.mFormat, md.mExtent) = LoadImageFile(commandBuffer.mDevice, path, srgb);
		md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
		const std::shared_ptr<Image> img = std::make_shared<Image>(commandBuffer.mDevice, path.filename().string(), md);

		commandBuffer.Copy(pixels, img);
		commandBuffer.HoldResource(pixels);

		imageCache.emplace(path.string(), img);
		return img;
	};

	const std::shared_ptr<SceneNode> root = SceneNode::Create(filename.stem().string());

	std::vector<std::shared_ptr<Material>> materials;
	std::vector<std::shared_ptr<Mesh>> meshes;

	if (scene->HasMaterials()) {
		bool metallicRoughness = false;
		if (filename.extension().string() == ".fbx") {
			//pfd::message n("Load PBR materials?", "Interpret diffuse/specular as glTF basecolor/pbr textures?", pfd::choice::yes_no);
			//interpret_as_pbr = n.result() == pfd::button::yes;
			metallicRoughness = true;
		}

		std::cout << "Loading materials...";

		const std::shared_ptr<SceneNode> materialsNode = root->AddChild("materials");
		for (int i = 0; i < scene->mNumMaterials; i++) {
			std::cout << "\rLoading materials " << (i+1) << "/" << scene->mNumMaterials << "     ";

			aiMaterial* m = scene->mMaterials[i];
			Material& material = *materials.emplace_back(materialsNode->AddChild(m->GetName().C_Str())->MakeComponent<Material>());

			ImageValue3 diffuse  = ImageValue3{ float3(1,1,1), {} };
			ImageValue4 specular = ImageValue4{ metallicRoughness ? float4(1,.5f,0,0) : float4(1,1,1,1), {} };
			ImageValue3 emission = ImageValue3{ float3(0), {} };

            aiColor4D tmp_color;
            if (m->Get(AI_MATKEY_COLOR_DIFFUSE    , tmp_color) == AI_SUCCESS) diffuse.first  = float3(tmp_color.r, tmp_color.g, tmp_color.b);
            if (m->Get(AI_MATKEY_COLOR_SPECULAR   , tmp_color) == AI_SUCCESS) specular.first = float4(tmp_color.r, tmp_color.g, tmp_color.b, tmp_color.a);
            if (m->Get(AI_MATKEY_COLOR_EMISSIVE   , tmp_color) == AI_SUCCESS) emission.first = float3(tmp_color.r, tmp_color.g, tmp_color.b);

			float eta = 1.45f;
			m->Get(AI_MATKEY_REFRACTI, eta);
			material.mMaterial.Eta(eta);

			if (m->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_DIFFUSE, 0, &aiPath);
				diffuse.second = GetImage(aiPath.C_Str(), true);
			}
			if (m->GetTextureCount(aiTextureType_SPECULAR) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_SPECULAR, 0, &aiPath);
				specular.second = GetImage(aiPath.C_Str(), false);
			}
			if (m->GetTextureCount(aiTextureType_EMISSIVE) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_EMISSIVE, 0, &aiPath);
				emission = ImageValue3{float3(1,1,1), GetImage(aiPath.C_Str(), true)};
			}

			if (metallicRoughness)
				material = CreateMetallicRoughnessMaterial(commandBuffer, diffuse, specular, emission);
			else
				material = CreateDiffuseSpecularMaterial(commandBuffer, diffuse, specular, emission);

			if (m->GetTextureCount(aiTextureType_NORMALS) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_NORMALS, 0, &aiPath);
				material.mBumpMap = GetImage(aiPath.C_Str(), false);
			} else if (m->GetTextureCount(aiTextureType_HEIGHT) > 0) {
				aiString aiPath;
				m->GetTexture(aiTextureType_HEIGHT, 0, &aiPath);
				material.mBumpMap = GetImage(aiPath.C_Str(), false);
			}

			float bumpScale = 1;
            m->Get(AI_MATKEY_BUMPSCALING, bumpScale);
			material.mMaterial.BumpScale(bumpScale);
		}
		std::cout << std::endl;
	}

	if (scene->HasMeshes()) {
		std::cout << "Loading mesh data...";

		size_t vertexDataSize = 0;
		size_t indexDataSize = 0;

		std::vector<size_t> positionsOffsets(scene->mNumMeshes);
		std::vector<size_t> normalsOffsets  (scene->mNumMeshes);
		std::vector<size_t> uvsOffsets      (scene->mNumMeshes);
		std::vector<size_t> indicesOffsets  (scene->mNumMeshes);
		for (int i = 0; i < scene->mNumMeshes; i++) {
			const aiMesh* m = scene->mMeshes[i];
			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				continue;

			positionsOffsets[i] = vertexDataSize;
			vertexDataSize += m->mNumVertices*3;

			normalsOffsets[i] = vertexDataSize;
			vertexDataSize += m->mNumVertices*3;

			if (m->GetNumUVChannels() > 0) {
				uvsOffsets[i] = vertexDataSize;
				vertexDataSize += m->mNumVertices*2;
			}

			indicesOffsets[i] = indexDataSize;
			indexDataSize += m->mNumFaces*3;
		}

		std::vector<float> vertices(vertexDataSize);
		std::vector<uint> indices(indexDataSize);

		std::vector<std::future<void>> work(scene->mNumMeshes);
		for (uint i = 0; i < scene->mNumMeshes; i++) {
			work[i] = std::move(std::async(std::launch::async, [&,i]() {
				const aiMesh* m = scene->mMeshes[i];
				if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
					return;

				const size_t offsetPos  = positionsOffsets[i];
				const size_t offsetNorm = normalsOffsets[i];
				const size_t offsetUv   = uvsOffsets[i];
 				const size_t offsetIdx  = indicesOffsets[i];

				for (uint vi = 0; vi < m->mNumVertices; vi++)
					for (uint j = 0; j < 3; j++)
						vertices[offsetPos + 3*vi + j] = (float)m->mVertices[vi][j];

				for (uint vi = 0; vi < m->mNumVertices; vi++)
					for (uint j = 0; j < 3; j++)
						vertices[offsetNorm + 3*vi + j] = (float)m->mNormals[vi][j];

				if (m->GetNumUVChannels() > 0) {
					for (uint vi = 0; vi < m->mNumVertices; vi++)
						for (uint j = 0; j < 2; j++)
							vertices[offsetUv + 2*vi + j] = (float)m->mTextureCoords[0][vi][j];
				}

				for (uint fi = 0; fi < m->mNumFaces; fi++)
					for (uint j = 0; j < 3; j++)
						indices[offsetIdx + 3*fi + j] = m->mFaces[fi].mIndices[j];
			}));
		}
		for (auto& w : work)
			w.wait();

		vk::BufferUsageFlags bufferUsage = vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer;
		if (commandBuffer.mDevice.GetAccelerationStructureFeatures().accelerationStructure) {
			bufferUsage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
			bufferUsage |= vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
		}

		auto vertexBuffer = commandBuffer.Upload<float>(vertices, filename.stem().string() + "/Vertices", bufferUsage|vk::BufferUsageFlagBits::eVertexBuffer);
		auto indexBuffer  = commandBuffer.Upload<uint> (indices , filename.stem().string() + "/Indices" , bufferUsage|vk::BufferUsageFlagBits::eIndexBuffer);

		// construct meshes

		const std::shared_ptr<SceneNode>& meshesNode = root->AddChild("meshes");
		for (int i = 0; i < scene->mNumMeshes; i++) {
			std::cout << "\rCreating meshes " << (i+1) << "/" << scene->mNumMeshes;

			const aiMesh* m = scene->mMeshes[i];

			if (!(m->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) || (m->mPrimitiveTypes & ~aiPrimitiveType_TRIANGLE) != 0)
				continue;

			Mesh::Vertices meshVertices;

			meshVertices[Mesh::VertexAttributeType::ePosition].emplace_back(
				Buffer::View<std::byte>(vertexBuffer, positionsOffsets[i]*sizeof(float), m->mNumVertices*3*sizeof(float)),
				Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });

			meshVertices[Mesh::VertexAttributeType::eNormal].emplace_back(
				Buffer::View<std::byte>(vertexBuffer, normalsOffsets[i]*sizeof(float), m->mNumVertices*3*sizeof(float)),
				Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float3), vk::Format::eR32G32B32Sfloat, 0, vk::VertexInputRate::eVertex });

			if (m->GetNumUVChannels() > 0) {
				meshVertices[Mesh::VertexAttributeType::eTexcoord].emplace_back(
					Buffer::View<std::byte>(vertexBuffer, uvsOffsets[i]*sizeof(float), m->mNumVertices*2*sizeof(float)),
					Mesh::VertexAttributeDescription{ (uint32_t)sizeof(float2), vk::Format::eR32G32Sfloat, 0, vk::VertexInputRate::eVertex } );
			}

			meshVertices.mAabb.minX = (float)m->mAABB.mMin.x;
			meshVertices.mAabb.minY = (float)m->mAABB.mMin.y;
			meshVertices.mAabb.minZ = (float)m->mAABB.mMin.z;
			meshVertices.mAabb.maxX = (float)m->mAABB.mMax.x;
			meshVertices.mAabb.maxY = (float)m->mAABB.mMax.y;
			meshVertices.mAabb.maxZ = (float)m->mAABB.mMax.z;

			const std::shared_ptr<SceneNode>& meshNode = meshesNode->AddChild(m->mName.C_Str());
			meshes.emplace_back( meshNode->MakeComponent<Mesh>(
				meshVertices,
				Buffer::View<uint32_t>(indexBuffer, indicesOffsets[i]*sizeof(uint32_t), m->mNumFaces*3),
				vk::PrimitiveTopology::eTriangleList) );
		}
		std::cout << std::endl;
	}

	std::stack<std::pair<aiNode*, SceneNode*>> nodes;
	nodes.push(std::make_pair(scene->mRootNode, root->AddChild(scene->mRootNode->mName.C_Str()).get()));
	while (!nodes.empty()) {
		auto[an, n] = nodes.top();
		nodes.pop();

		if (sizeof(aiMatrix4x4) == sizeof(float4x4)) {
			float4x4 m;
			memcpy(&m, &an->mTransformation, sizeof(float4x4));
			n->MakeComponent<float4x4>( transpose(m) );
		} else {
			double4x4 m;
			memcpy(&m, &an->mTransformation, sizeof(double4x4));
			n->MakeComponent<float4x4>( transpose((float4x4)m) );
		}

		if (an->mNumMeshes == 1)
			n->MakeComponent<MeshRenderer>(materials[scene->mMeshes[an->mMeshes[0]]->mMaterialIndex], meshes[an->mMeshes[0]]);
		else if (an->mNumMeshes > 1)
			for (int i = 0; i < an->mNumMeshes; i++)
				n->AddChild(scene->mMeshes[an->mMeshes[i]]->mName.C_Str())->MakeComponent<MeshRenderer>(materials[scene->mMeshes[an->mMeshes[i]]->mMaterialIndex], meshes[an->mMeshes[i]]);

		for (int i = 0; i < an->mNumChildren; i++)
			nodes.push(std::make_pair(an->mChildren[i], n->AddChild(an->mChildren[i]->mName.C_Str()).get()));
	}

	std::cout << "Loaded " << filename << std::endl;
	return root;
}

}

#endif
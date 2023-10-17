#pragma once

#include <nanovdb/util/GridHandle.h>
#include <Common/Material.h>
#include <Common/SceneTypes.h>
#include <Core/PipelineCache.hpp>
#include "SceneNode.hpp"
#include "Mesh.hpp"

namespace ptvk {

struct Material {
    PackedMaterialParameters mMaterial;
	Image::View mBaseColor;
	Image::View mPackedParams;
	Image::View mEmission;
	Image::View mBumpMap;
	Buffer::View<uint> mMinAlpha;
};

struct MeshRenderer {
	std::shared_ptr<Material> mMaterial;
	std::shared_ptr<Mesh> mMesh;

	void OnInspectorGui(SceneNode& node);
};

struct SphereRenderer {
	std::shared_ptr<Material> mMaterial;
	float mRadius;

	void OnInspectorGui(SceneNode& node);
};

struct VolumeRenderer {
	float3 mDensityScale;
	float mAnisotropy;
	float3 mAlbedoScale;
	std::shared_ptr<nanovdb::GridHandle<nanovdb::HostBuffer>> mDensityGrid, mAlbedoGrid;
	Buffer::View<std::byte> mDensityBuffer, mAlbedoBuffer;

	void OnInspectorGui(SceneNode& node);
};

struct EnvironmentMap {
	float3 mColor;
	Image::View mImage;

    void OnInspectorGui(SceneNode &node);
};

struct Camera {
	SceneNode& mNode;
	float mVerticalFov = glm::radians(70.f);
	float mNearPlane = 0.01f;
	float mAspect = 1;

	void OnInspectorGui();

	inline float4x4 GetProjection() const { return glm::infinitePerspectiveRH(mVerticalFov, mAspect, mNearPlane); }
};

class Scene {
public:
	struct RenderData {
		std::unordered_map<const void* /* renderer address */, std::pair<float4x4, uint32_t /* instance index */ >> mInstanceTransformMap;
		std::vector<std::weak_ptr<SceneNode>> mInstanceNodes;
		Buffer::View<uint32_t> mInstanceIndexMap;

		// see SceneConstants struct in Scene.slang
		ShaderParameterBlock mShaderParameters;

		inline void Reset() {
			mInstanceTransformMap.clear();
			mInstanceNodes.clear();
			mInstanceIndexMap = {};
			mShaderParameters.clear();
		}
	};

	Scene(Instance& instance);

	void CreatePipelines();

	void Update(CommandBuffer& commandBuffer);
	inline std::chrono::high_resolution_clock::time_point GetLastUpdate() const { return mLastUpdate; }
	inline const RenderData& GetRenderData() const { return mRenderData; }
	inline const std::shared_ptr<SceneNode>& GetRoot() const { return mRootNode; }

	inline void InspectorSelect(SceneNode& n) { mInspectedNode = &n; }

	#pragma region Scene loading

	std::shared_ptr<SceneNode> LoadEnvironmentMap(CommandBuffer& commandBuffer, const std::filesystem::path& filename);
	std::shared_ptr<SceneNode> LoadGltf          (CommandBuffer& commandBuffer, const std::filesystem::path& filename);
	//std::shared_ptr<SceneNode> LoadMitsuba       (CommandBuffer& commandBuffer, const std::filesystem::path& filename);
	//std::shared_ptr<SceneNode> LoadVol           (CommandBuffer& commandBuffer, const std::filesystem::path& filename);
	//std::shared_ptr<SceneNode> LoadNvdb          (CommandBuffer& commandBuffer, const std::filesystem::path& filename);
#ifdef ENABLE_ASSIMP
	std::shared_ptr<SceneNode> LoadAssimp        (CommandBuffer& commandBuffer, const std::filesystem::path& filename);
#endif
//#ifdef ENABLE_OPENVDB
//	std::shared_ptr<SceneNode> LoadVdb           (CommandBuffer& commandBuffer, const std::filesystem::path& filename);
//#endif

	inline std::vector<std::string> LoaderFilters() {
		return {
			"All Files", "*",
			"Environment Maps (.exr .hdr)", "*.exr *.hdr",
			"glTF Scenes (.gltf .glb)", "*.gltf *.glb",
			//"Mitsuba Volumes (.vol)" , "*.vol",
			//"NVDB Volume (.nvdb)" , "*.nvdb",
			#ifdef ENABLE_ASSIMP
			"Autodesk (.fbx)", "*.fbx",
			"Wavefront Object Files (.obj)", "*.obj",
			"Stanford Polygon Library Files (.ply)", "*.ply",
			"Stereolithography Files (.stl)", "*.stl",
			"Blender Scenes (.blend)", "*.blend",
			#endif
			//#ifdef ENABLE_OPENVDB
			//"VDB Volumes (.vdb)", "*.vdb",
			//#endif
		};
	}
	inline std::shared_ptr<SceneNode> Load(CommandBuffer& commandBuffer, const std::filesystem::path& filename) {
		const std::string& ext = filename.extension().string();
		if      (ext == ".hdr") return LoadEnvironmentMap(commandBuffer, filename);
		else if (ext == ".exr") return LoadEnvironmentMap(commandBuffer, filename);
		else if (ext == ".gltf") return LoadGltf(commandBuffer, filename);
		else if (ext == ".glb")  return LoadGltf(commandBuffer, filename);
		//else if (ext == ".xml") return LoadMitsuba(commandBuffer, filename);
		//else if (ext == ".vol") return LoadVol(commandBuffer, filename);
		//else if (ext == ".nvdb") return LoadNvdb(commandBuffer, filename);
		//#ifdef ENABLE_OPENVDB
		//else if (ext == ".vdb") return LoadVdb(commandBuffer, filename);
		//#endif
		else
		#ifdef ENABLE_ASSIMP
			return LoadAssimp(commandBuffer, filename);
		#else
			throw std::runtime_error("Unknown extension:" + ext);
		#endif
	}

	inline void LoadAsync(const std::filesystem::path& filename) { mToLoad.emplace_back(filename.string()); }

	#pragma endregion
	#pragma region Material conversion

	using ImageValue1 = std::pair<float, Image::View>;
	using ImageValue2 = std::pair<float2, Image::View>;
	using ImageValue3 = std::pair<float3, Image::View>;
	using ImageValue4 = std::pair<float4, Image::View>;

	Material CreateMetallicRoughnessMaterial(CommandBuffer& commandBuffer, const ImageValue3& baseColor, const ImageValue4& metallicRoughness, const ImageValue3& emission);
	Material CreateDiffuseSpecularMaterial  (CommandBuffer& commandBuffer, const ImageValue3& diffuse, const ImageValue3& specular, const ImageValue3& emission);

	#pragma endregion

private:
	using AccelerationStructureData = std::pair<std::shared_ptr<vk::raii::AccelerationStructureKHR>, Buffer::View<std::byte> /* acceleration structure buffer */>;

	std::shared_ptr<SceneNode> mRootNode;
	SceneNode* mInspectedNode;

	// cache aabb BLASs
	std::unordered_map<size_t, AccelerationStructureData> mAABBs;
	// cache mesh BLASs
	std::unordered_map<size_t, AccelerationStructureData> mMeshAccelerationStructures;

	RenderData mRenderData;

	bool DrawNodeGui(SceneNode& node, bool& changed);
	void UpdateRenderData(CommandBuffer& commandBuffer);

	ComputePipelineCache mComputeMinAlphaPipeline;
	ComputePipelineCache mConvertMetallicRoughnessPipeline;

	std::vector<std::string> mToLoad;
	std::vector< std::future<std::pair<std::shared_ptr<SceneNode>, std::shared_ptr<CommandBuffer>>> > mLoading;

	bool mUpdateOnce = false;
	std::chrono::high_resolution_clock::time_point mLastUpdate;
};

}
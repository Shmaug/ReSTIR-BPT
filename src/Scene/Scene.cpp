#include "Scene.hpp"
#include "FlyCamera.hpp"
#include <App/App.hpp>
#include <Core/Gui.hpp>
#include <Core/Window.hpp>

#include <future>
#include <portable-file-dialogs.h>

#include <ImGuizmo.h>

namespace ptvk {

std::tuple<std::shared_ptr<vk::raii::AccelerationStructureKHR>, Buffer::View<std::byte>> BuildAccelerationStructure(CommandBuffer& commandBuffer, const std::string& name, const vk::AccelerationStructureTypeKHR type, const vk::ArrayProxy<const vk::AccelerationStructureGeometryKHR>& geometries, const vk::ArrayProxy<const vk::AccelerationStructureBuildRangeInfoKHR>& buildRanges) {
	vk::AccelerationStructureBuildGeometryInfoKHR buildGeometry(type, vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace, vk::BuildAccelerationStructureModeKHR::eBuild);
	buildGeometry.setGeometries(geometries);

	vk::AccelerationStructureBuildSizesInfoKHR buildSizes;
	if (buildRanges.size() > 0 && buildRanges.front().primitiveCount > 0) {
		std::vector<uint32_t> counts((uint32_t)geometries.size());
		for (uint32_t i = 0; i < geometries.size(); i++)
			counts[i] = (buildRanges.data() + i)->primitiveCount;
		buildSizes = commandBuffer.mDevice->getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometry, counts);
	} else
		buildSizes.accelerationStructureSize = buildSizes.buildScratchSize = 4;

	Buffer::View<std::byte> buffer = std::make_shared<Buffer>(
		commandBuffer.mDevice,
		name + "/Buffer",
		buildSizes.accelerationStructureSize,
		vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress);

	Buffer::View<std::byte> scratchData = std::make_shared<Buffer>(
		commandBuffer.mDevice,
		name + "/scratchData",
		buildSizes.buildScratchSize,
		vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer);

	std::shared_ptr<vk::raii::AccelerationStructureKHR> accelerationStructure = std::make_shared<vk::raii::AccelerationStructureKHR>(*commandBuffer.mDevice, vk::AccelerationStructureCreateInfoKHR({}, **buffer.GetBuffer(), buffer.Offset(), buffer.SizeBytes(), type));
	commandBuffer.mDevice.SetDebugName(**accelerationStructure, name);

	buildGeometry.dstAccelerationStructure = **accelerationStructure;
	buildGeometry.scratchData = scratchData.GetDeviceAddress();

	commandBuffer->buildAccelerationStructuresKHR(buildGeometry, buildRanges.data());

	commandBuffer.HoldResource(buffer);
	commandBuffer.HoldResource(scratchData);
	commandBuffer.HoldResource(accelerationStructure);

	buffer.SetState(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::AccessFlagBits::eAccelerationStructureWriteKHR);

	return std::tie(accelerationStructure, buffer);
}

Scene::Scene(Instance& instance) {
	const std::filesystem::path shaderPath = *instance.GetOption("shader-kernel-path");
	mComputeMinAlphaPipeline          = ComputePipelineCache(shaderPath / "Kernels/MaterialConversion.slang", "ComputeMinAlpha");
	mConvertMetallicRoughnessPipeline = ComputePipelineCache(shaderPath / "Kernels/MaterialConversion.slang", "ConvertMetallicRoughness");

	mRootNode = SceneNode::Create("Root");
	mInspectedNode = nullptr;

	for (const std::string arg : instance.GetOptions("scene"))
		mToLoad.emplace_back(arg);
	mUpdateOnce = true;
}

std::shared_ptr<SceneNode> Scene::LoadEnvironmentMap(CommandBuffer& commandBuffer, const std::filesystem::path& filepath) {
	std::filesystem::path path = filepath;
	if (path.is_relative()) {
		std::filesystem::path cur = std::filesystem::current_path();
		std::filesystem::current_path(path.parent_path());
		path = std::filesystem::absolute(path);
		std::filesystem::current_path(cur);
	}

	ImageInfo md = {};
	std::shared_ptr<Buffer> pixels;
	std::tie(pixels, md.mFormat, md.mExtent) = LoadImageFile(commandBuffer.mDevice, filepath, false);
	md.mUsage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
	md.mLevels = GetMaxMipLevels(md.mExtent);
	const std::shared_ptr<Image> img = std::make_shared<Image>(commandBuffer.mDevice, filepath.filename().string(), md);

	commandBuffer.Copy(pixels, img);
	commandBuffer.GenerateMipMaps(img);

	commandBuffer.HoldResource(pixels);

	const std::shared_ptr<SceneNode> node = SceneNode::Create(filepath.stem().string());
	node->MakeComponent<EnvironmentMap>(float3(1), img);
	return node;
}

// OnInspectorGui functions

void OnInspectorGui(const Image::View& image) {
	const uint32_t w = ImGui::GetWindowSize().x;
	ImGui::Image(Gui::GetTextureID(image), ImVec2(w, w * (float)image.GetExtent().height / (float)image.GetExtent().width));
}

bool OnInspectorGui(SceneNode& node, float4x4& v) {
	// transform not in scene, or no camera
	bool changed = false;
	float matrixTranslation[3], matrixRotation[3], matrixScale[3];
	ImGuizmo::DecomposeMatrixToComponents(&v[0][0], matrixTranslation, matrixRotation, matrixScale);
	if (ImGui::InputFloat3("T", matrixTranslation)) changed = true;
	if (ImGui::InputFloat3("R", matrixRotation)) changed = true;
	if (ImGui::InputFloat3("S", matrixScale)) changed = true;
	if (changed) {
		ImGuizmo::RecomposeMatrixFromComponents(matrixTranslation, matrixRotation, matrixScale, &v[0][0]);
	}
	return true;
}

bool OnInspectorGui(SceneNode& node, Camera& v) {
	bool changed = false;
	if (ImGui::DragFloat("Vertical FoV", &v.mVerticalFov, 0.01f, 1, 179)) changed = true;
	if (ImGui::DragFloat("Near Plane", &v.mNearPlane, 0.01f, -1, 1)) changed = true;
	return changed;
}

bool OnInspectorGui(SceneNode& node, Mesh& mesh) {
	ImGui::LabelText("Topology", "%s", vk::to_string(mesh.GetTopology()).c_str());
	if (mesh.GetIndices())
		ImGui::LabelText("Index stride", "%s", std::to_string(mesh.GetIndices().Stride()).c_str());
	for (const auto& [type, verts] : mesh.GetVertices()) {
		for (uint32_t i = 0; i < verts.size(); i++) {
			const auto&[buf, desc] = verts[i];
			if (buf && ImGui::CollapsingHeader((std::to_string(type) + "_" + std::to_string(i)).c_str())) {
				ImGui::LabelText("Format", "%s", vk::to_string(desc.mFormat).c_str());
				ImGui::LabelText("Stride", "%u", desc.mStride);
				ImGui::LabelText("Offset", "%u", desc.mOffset);
				ImGui::LabelText("Input rate", "%s", vk::to_string(desc.mInputRate).c_str());
			}
		}
	}
	return false;
}

bool OnInspectorGui(SceneNode& node, Material& v) {
	bool changed = false;

	float3 color = v.mMaterial.BaseColor();
	if (ImGui::ColorEdit3("Base Color", &color.x, ImGuiColorEditFlags_Float|ImGuiColorEditFlags_PickerHueBar)) {
		v.mMaterial.BaseColor(color);
		changed = true;
	}
	float3 emission = v.mMaterial.Emission();
	if (ImGui::ColorEdit3("Emission", &emission.x, ImGuiColorEditFlags_Float|ImGuiColorEditFlags_HDR|ImGuiColorEditFlags_PickerHueBar)) {
		v.mMaterial.Emission(emission);
		changed = true;
	}

	ImGui::PushItemWidth(80);
	auto slider = [&](const char* label, float value, auto setter, const float mn = 0, const float mx = 1) {
		if (ImGui::SliderFloat(label, &value, mn, mx)) {
			setter(value);
			changed = true;
		}
	};
	slider("Roughness",        v.mMaterial.Roughness(),      [&](float f){ v.mMaterial.Roughness(f); });
	slider("Subsurface",       v.mMaterial.Subsurface(),     [&](float f){ v.mMaterial.Subsurface(f); });
	slider("Specular",         v.mMaterial.Specular(),       [&](float f){ v.mMaterial.Specular(f); });
	slider("Metallic",         v.mMaterial.Metallic(),       [&](float f){ v.mMaterial.Metallic(f); });
	slider("Anisotropic",      v.mMaterial.Anisotropic(),    [&](float f){ v.mMaterial.Anisotropic(f); });
	slider("Sheen",            v.mMaterial.Sheen(),          [&](float f){ v.mMaterial.Sheen(f); });
	slider("Clearcoat",        v.mMaterial.Clearcoat(),      [&](float f){ v.mMaterial.Clearcoat(f); });
	slider("Clearcoat gloss",  v.mMaterial.ClearcoatGloss(), [&](float f){ v.mMaterial.ClearcoatGloss(f); });
	slider("Transmission",     v.mMaterial.Transmission(),   [&](float f){ v.mMaterial.Transmission(f); });
	slider("Refraction index", v.mMaterial.Eta(),            [&](float f){ v.mMaterial.Eta(f); }, 0.5f, 2);
	if (v.mBumpMap) slider("Bump Strength", v.mMaterial.BumpScale(), [&](float f){ v.mMaterial.BumpScale(f); }, 0, 8);
	if (v.mBaseColor) slider("Alpha cutoff", v.mMaterial.AlphaCutoff(), [&](float f){ v.mMaterial.AlphaCutoff(f); });

	ImGui::PopItemWidth();

	if (v.mBaseColor) {
		ImGui::Text("Base color");
		OnInspectorGui(v.mBaseColor);
	}
	if (v.mMinAlpha) {
		ImGui::Text("Min alpha: %u", v.mMinAlpha ? v.mMinAlpha[0] : 255u);
	}
	if (v.mEmission) {
		ImGui::Text("Emission");
		OnInspectorGui(v.mEmission);
	}
	if (v.mPackedParams) {
		ImGui::Text("Packed parameters");
		OnInspectorGui(v.mPackedParams);
	}
	if (v.mBumpMap) {
		ImGui::Text("Bump");
		OnInspectorGui(v.mBumpMap);
	}

	return changed;
}

bool OnInspectorGui(SceneNode& node, MeshRenderer& v) {
	bool changed = false;
	if (v.mMaterial) {
		if (ImGui::CollapsingHeader("Mesh")) {
			if (OnInspectorGui(node, *v.mMesh))
				changed = true;
		}
		if (ImGui::CollapsingHeader("Material")) {
			if (OnInspectorGui(node, *v.mMaterial))
				changed = true;
		}
	}
	return changed;
}
bool OnInspectorGui(SceneNode& node, SphereRenderer& v) {
	bool changed = false;
	if (ImGui::DragFloat("Radius", &v.mRadius, .01f))
		changed = true;
	if (v.mMaterial) {
		if (ImGui::CollapsingHeader("Material")) {
			if (OnInspectorGui(node, *v.mMaterial))
				changed = true;
		}
	}
	return changed;
}
bool OnInspectorGui(SceneNode& node, VolumeRenderer& v) {
	bool changed = false;

	if (ImGui::ColorEdit3("Density", &v.mDensityScale.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float)) changed = true;
	if (ImGui::ColorEdit3("Albedo", &v.mAlbedoScale.x, ImGuiColorEditFlags_Float)) changed = true;
	if (ImGui::SliderFloat("Anisotropy", &v.mAnisotropy, -.999f, .999f)) changed = true;

	return changed;
}

bool OnInspectorGui(SceneNode& node, EnvironmentMap& v) {
	bool changed = false;
	if (ImGui::ColorEdit3("Color", &v.mColor.x, ImGuiColorEditFlags_Float|ImGuiColorEditFlags_HDR|ImGuiColorEditFlags_PickerHueBar))
		changed = true;
	if (v.mImage) {
		ImGui::Text("Image");
		OnInspectorGui(v.mImage);
	}
	return changed;
}


// scene graph node inspector

bool Scene::DrawNodeGui(SceneNode& n, bool& changed) {
	ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_OpenOnArrow;
	if (mInspectedNode && &n == mInspectedNode) flags |= ImGuiTreeNodeFlags_Selected;
	if (n.GetChildren().empty()) flags |= ImGuiTreeNodeFlags_Leaf;

	// open nodes above selected node
	if (mInspectedNode && n.IsDescendant(*mInspectedNode))
		ImGui::SetNextItemOpen(true, ImGuiCond_Once);

	// tree menu item
	ImGui::PushID(&n);
	const bool open = ImGui::TreeNodeEx(n.GetName().c_str(), flags);
	ImGui::PopID();

	if (ImGui::BeginDragDropSource()) {
		const SceneNode* payload = &n;
		ImGui::SetDragDropPayload("SceneNode", &payload, sizeof(SceneNode*));
		ImGui::Text("%s", n.GetName().c_str());
		ImGui::EndDragDropSource();
	}
	if (ImGui::BeginDragDropTarget()) {
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneNode")) {
			SceneNode* nodeptr = *(SceneNode**)payload->Data;
			if (nodeptr) {
				n.AddChild(nodeptr->GetPtr());
				changed = true;
			}
		}
		if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneComponent")) {
			const auto&[nodeptr, type] = *(std::pair<SceneNode*, std::type_index>*)payload->Data;
			if (nodeptr && !n.HasComponent(type)) {
				std::shared_ptr<void> component = nodeptr->GetComponent(type);
				nodeptr->RemoveComponent(type);
				n.AddComponent(type, component);
				changed = true;
			}
		}
		ImGui::EndDragDropTarget();
	}

	bool erase = false;

	// context menu
	if (ImGui::BeginPopupContextItem()) {
		if (ImGui::Selectable(n.Enabled() ? "Disable" : "Enable")) {
			n.Enabled(!n.Enabled());
			changed = true;
		}
		if (ImGui::Selectable("Add component", false, ImGuiSelectableFlags_DontClosePopups)) {
			ImGui::OpenPopup("Add component");
		}
		if (ImGui::Selectable("Add child", false, ImGuiSelectableFlags_DontClosePopups)) {
			ImGui::OpenPopup("Add node");
		}
		if (ImGui::Selectable("Delete", false)) {
			erase = true;
		}

		// add component dialog
		if (ImGui::BeginPopup("Add component")) {
			if (ImGui::Selectable("Transform", n.HasComponent<float4x4>() ? ImGuiSelectableFlags_Disabled : 0)) {
				n.MakeComponent<float4x4>(glm::identity<float4x4>());
				ImGui::CloseCurrentPopup();
			}
			if (ImGui::Selectable("Environment map", n.HasComponent<EnvironmentMap>() ? ImGuiSelectableFlags_Disabled : 0)) {
				n.MakeComponent<EnvironmentMap>();
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		// add child dialog
		if (ImGui::BeginPopup("Add node")) {
			static char childName[64];
			ImGui::InputText("Child name", childName, 64);
			if (ImGui::Button("Done")) {
				n.AddChild(childName);
				ImGui::CloseCurrentPopup();
			}
			ImGui::EndPopup();
		}

		ImGui::EndPopup();
	}

	if (open) {
		// select node if treenode is clicked
		if (ImGui::IsItemClicked())
			InspectorSelect(n);

		std::unordered_set<SceneNode*> toErase;

		// draw children
		for (const std::shared_ptr<SceneNode>& c : n.GetChildren())
			if (DrawNodeGui(*c, changed))
				toErase.emplace(c.get());

		for (SceneNode* c : toErase) {
			if (mInspectedNode) {
				if (mInspectedNode == c || mInspectedNode->IsDescendant(*c))
					mInspectedNode = nullptr;
			}
			c->RemoveParent();
			changed = true;
		}

		ImGui::TreePop();
	}
	return erase;
}

void Scene::Update(CommandBuffer& commandBuffer) {
	ProfilerScope s("Scene::Update", &commandBuffer);

	struct GizmoDrawer {
		ImGuizmo::OPERATION mCurrentGizmoOperation = ImGuizmo::ROTATE;
		ImGuizmo::MODE mCurrentGizmoMode = ImGuizmo::LOCAL;
		bool useSnap = false;
		float3 snapTranslation = float3(0.05f);
		float3 snapAngle = float3(22.5f);
		float3 snapScale = float3(0.1f);

		inline void Update() {
			if (ImGui::IsKeyPressed(ImGuiKey_T)) mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			if (ImGui::IsKeyPressed(ImGuiKey_R)) mCurrentGizmoOperation = ImGuizmo::ROTATE;
			if (ImGui::IsKeyPressed(ImGuiKey_Y)) mCurrentGizmoOperation = ImGuizmo::SCALE;
			if (ImGui::IsKeyPressed(ImGuiKey_U)) useSnap = !useSnap;
		}

		inline void OnInspectorGui() {
			if (ImGui::RadioButton("Translate (T)", mCurrentGizmoOperation == ImGuizmo::TRANSLATE)) mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Rotate (R)", mCurrentGizmoOperation == ImGuizmo::ROTATE)) mCurrentGizmoOperation = ImGuizmo::ROTATE;
			ImGui::SameLine();
			if (ImGui::RadioButton("Scale (Y)", mCurrentGizmoOperation == ImGuizmo::SCALE)) mCurrentGizmoOperation = ImGuizmo::SCALE;

			if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
				mCurrentGizmoMode = ImGuizmo::LOCAL;
			ImGui::SameLine();
			if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
				mCurrentGizmoMode = ImGuizmo::WORLD;

			ImGui::Checkbox("Snap (U)", &useSnap);
			ImGui::SameLine();
			float3* snap;
			switch (mCurrentGizmoOperation) {
			case ImGuizmo::TRANSLATE:
				snap = &snapTranslation;
				ImGui::SetNextItemWidth(40);
				ImGui::InputFloat3("Snap", &snap->x);
				break;
			case ImGuizmo::ROTATE:
				snap = &snapAngle;
				ImGui::SetNextItemWidth(40);
				ImGui::InputFloat("Angle Snap", &snap->x);
				break;
			case ImGuizmo::SCALE:
				snap = &snapScale;
				ImGui::SetNextItemWidth(40);
				ImGui::InputFloat("Scale Snap", &snap->x);
				break;
			}
		}

		inline bool OnGizmoGui(const float* view, const float* projection, const float4x4& parent, float4x4& localMatrix) {
			ImGuiIO& io = ImGui::GetIO();
			ImGuizmo::SetRect(App::mViewportRect.x, App::mViewportRect.y, App::mViewportRect.z - App::mViewportRect.x, App::mViewportRect.w - App::mViewportRect.y);
			ImGuizmo::SetID(0);

			float3* snap;
			switch (mCurrentGizmoOperation) {
			case ImGuizmo::TRANSLATE:
				snap = &snapTranslation;
				break;
			case ImGuizmo::ROTATE:
				snap = &snapAngle;
				break;
			case ImGuizmo::SCALE:
				snap = &snapScale;
				break;
			}

			localMatrix = parent * localMatrix;
			bool changed = ImGuizmo::Manipulate(view, projection, mCurrentGizmoOperation, mCurrentGizmoMode, &localMatrix[0][0], NULL, useSnap ? &snap->x : NULL);
			localMatrix = inverse(parent) * localMatrix;

			return changed;
		}
	};

	static GizmoDrawer sGizmoData = {};

	sGizmoData.Update();

	bool changed = false;

	// draw gui
	if (ImGui::Begin("Scene Inspector")) {
		if (ImGui::Button("Load file")) {
			auto f = pfd::open_file("Open scene", "", LoaderFilters());
			for (const std::string& filepath : f.result())
				LoadAsync(filepath);
		}

		ImGui::SameLine();
		if (ImGui::Button("Update"))
			changed = true;

		if (ImGui::CollapsingHeader("Scene graph")) {
			const float s = ImGui::GetStyle().IndentSpacing;
			ImGui::GetStyle().IndentSpacing = s/2;
			if (DrawNodeGui(*mRootNode, changed))
				changed = true;
			ImGui::GetStyle().IndentSpacing = s;
		}
	}
	ImGui::End();

	if (ImGui::Begin("Component Inspector")) {
		if (mInspectedNode) {
			sGizmoData.OnInspectorGui();

			ImGui::Text(mInspectedNode->GetName().c_str());
			bool e = mInspectedNode->Enabled();
			if (ImGui::Checkbox("Enabled", &e)) {
				mInspectedNode->Enabled(e);
				changed = true;
			}
			for (const std::type_index compType : mInspectedNode->GetComponents()) {
				if (ImGui::CollapsingHeader(compType.name())) {
					if      (compType == typeid(float4x4))       { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<float4x4>())) changed = true; }
					else if (compType == typeid(Camera))         { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<Camera>())) changed = true; }
					else if (compType == typeid(Mesh))           { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<Mesh>())) changed = true; }
					else if (compType == typeid(MeshRenderer))   { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<MeshRenderer>())) changed = true; }
					else if (compType == typeid(SphereRenderer)) { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<SphereRenderer>())) changed = true; }
					else if (compType == typeid(EnvironmentMap)) { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<EnvironmentMap>())) changed = true; }
					else if (compType == typeid(Material))       { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<Material>())) changed = true; }
					else if (compType == typeid(VolumeRenderer)) { if (OnInspectorGui(*mInspectedNode, *mInspectedNode->GetComponent<VolumeRenderer>())) changed = true; }
					else if (compType == typeid(FlyCamera))      { if (mInspectedNode->GetComponent<FlyCamera>()->OnInspectorGui()) changed = true; }
					ImGui::Separator();
				}
			}
		}
	}
	ImGui::End();

	// transform gizmo
	if (mInspectedNode) {
		std::shared_ptr<SceneNode> cameraNode;
		if (auto camera = mRootNode->FindDescendant<Camera>(&cameraNode)) {
			float4x4 parentTransform = glm::identity<float4x4>();
			if (std::shared_ptr<SceneNode> p = mInspectedNode->GetParent())
				parentTransform = NodeToWorld(*p);

			float4x4 m = glm::identity<float4x4>();
			if (auto v = mInspectedNode->GetComponent<float4x4>())
				m = *v;

			float4x4 view = inverse(NodeToWorld(*cameraNode));
			float4x4 proj = camera->GetProjection();
			if (sGizmoData.OnGizmoGui(&view[0][0], &proj[0][0], parentTransform, m)) {
				changed = true;
				if (auto v = mInspectedNode->GetComponent<float4x4>())
					*v = m;
				else
					mInspectedNode->MakeComponent<float4x4>(m);
			}
		}
	}

	if (changed) mUpdateOnce = true;

	// load input files

	bool loaded = false;
	for (auto it = mLoading.begin(); it != mLoading.end();) {
		if (it->wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
		 	it++;
			continue;
		}
		const auto[node, cb] = it->get();
		mRootNode->AddChild(node);

		it = mLoading.erase(it);

		loaded = true;
		mUpdateOnce = true;
	}

	for (const std::string& file : mToLoad) {
		const std::filesystem::path filepath = file;
		Device& device = commandBuffer.mDevice;
		const uint32_t family = device.FindQueueFamily(vk::QueueFlagBits::eTransfer|vk::QueueFlagBits::eCompute);
		mLoading.emplace_back( std::move(std::async(std::launch::async, [&,filepath,family]() {
			std::shared_ptr<CommandBuffer> cb = std::make_shared<CommandBuffer>(device, "Scene load", family);
			cb->Reset();
			std::shared_ptr<SceneNode> node = Load(*cb, filepath);
			cb->Submit(*device->getQueue(family, 0));
			if (device->waitForFences(**cb->GetCompletionFence(), true, std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess)
				node = nullptr;
			return std::make_pair(node, cb);
		})) );
	}
	mToLoad.clear();

	if (!mUpdateOnce) {
		if (commandBuffer.mDevice.GetAccelerationStructureFeatures().accelerationStructure)
			commandBuffer.HoldResource(mRenderData.mShaderParameters.GetBuffer<std::byte>("mAccelerationStructureBuffer"));
		return;
	}

	// Update scene data based on node graph
	// always update once after load so that motion transforms are valid

	mUpdateOnce = loaded;

	UpdateRenderData(commandBuffer);
}

void Scene::UpdateRenderData(CommandBuffer& commandBuffer) {
	mLastUpdate = std::chrono::high_resolution_clock::now();

	auto prevInstanceTransforms = std::move(mRenderData.mInstanceTransformMap);
	mRenderData.Reset();

	// Construct resources used by renderers (mesh/material data buffers, image arrays, etc.)

	std::vector<InstanceBase> instanceDatas;
	std::vector<float4x4> instanceTransforms;
	std::vector<float4x4> instanceInverseTransforms;
	std::vector<float4x4> instanceMotionTransforms;
	std::vector<VolumeInfo> volumeInfos;
	std::vector<uint32_t> lightInstanceMap; // light index -> instance index
	std::vector<uint32_t> instanceLightMap; // instance index -> light index
	std::vector<uint32_t> instanceIndexMap; // current frame instance index -> previous frame instance index

	std::vector<MeshVertexInfo> meshVertexInfos;
	std::unordered_map<Buffer*, uint32_t> vertexBufferMap;
	uint32_t numVertexBuffers = 0;

	std::unordered_map<Image::View, uint32_t> image2s;
	std::unordered_map<Image::View, uint32_t> image4s;
	std::vector<GpuMaterial> materials;
	std::unordered_map<const void*, uint32_t> materialMap;

	const bool useAccelerationStructure = commandBuffer.mDevice.GetAccelerationStructureFeatures().accelerationStructure;
	std::vector<vk::AccelerationStructureInstanceKHR> instancesAS;
	std::vector<vk::BufferMemoryBarrier> blasBarriers;

	float3 aabbMin = float3( std::numeric_limits<float>::infinity());
	float3 aabbMax = float3(-std::numeric_limits<float>::infinity());

	auto AddImage2 = [&](const Image::View& img) {
		if (!img) return ~(uint32_t)0;
		if (auto it = image2s.find(img); it != image2s.end())
			return it->second;
		else {
			uint32_t c = image2s.size();
			mRenderData.mShaderParameters.SetImage("mImage2s", c, img, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead);
			image2s.emplace(img, c);
			return c;
		}
	};
	auto AddImage4 = [&](const Image::View& img) {
		if (!img) return ~(uint32_t)0;
		if (auto it = image4s.find(img); it != image4s.end())
			return it->second;
		else {
			uint32_t c = image4s.size();
			mRenderData.mShaderParameters.SetImage("mImage4s", c, img, vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlagBits::eShaderRead);
			image4s.emplace(img, c);
			return c;
		}
	};
	auto AddMaterial = [&](const Material& material) {
		// append unique materials to materials list
		auto materialMap_it = materialMap.find(&material);
		if (materialMap_it != materialMap.end())
			return materialMap_it->second;
		materialMap_it = materialMap.emplace(&material, (uint32_t)materials.size()).first;

		GpuMaterial m;
		m.mParameters = material.mMaterial;
		m.SetBaseColorImage(AddImage4(material.mBaseColor));
		m.SetEmissionImage(AddImage4(material.mEmission));
		m.SetPackedParamsImage(AddImage4(material.mPackedParams));
		if (material.mBumpMap) {
			if (GetChannelCount(material.mBumpMap.GetImage()->GetFormat()) == 2) {
				m.SetBumpImage(AddImage2(material.mBumpMap));
				m.SetIsBumpTwoChannel(true);
			} else {
				m.SetBumpImage(AddImage4(material.mBumpMap));
				m.SetIsBumpTwoChannel(false);
			}
		} else
			m.SetBumpImage(~(uint32_t)0);
		materials.emplace_back(m);

		return materialMap_it->second;
	};

	auto AddVertexBuffer = [&](const std::shared_ptr<Buffer>& buf) -> uint32_t {
		if (!buf)
			return 0xFFFF;

		if (auto it = vertexBufferMap.find(buf.get()); it != vertexBufferMap.end())
			return it->second;

		mRenderData.mShaderParameters.SetBuffer( "mVertexBuffers", numVertexBuffers, buf);
		vertexBufferMap.emplace(buf.get(), numVertexBuffers);
		return numVertexBuffers++;
	};

	auto AddInstance = [&](SceneNode& node, const void* primPtr, const auto& instance, const float4x4& transform, const bool isLight) {
		const uint32_t instanceIndex = (uint32_t)instanceDatas.size();
		instanceDatas.emplace_back(std::bit_cast<InstanceBase>(instance));
		mRenderData.mInstanceNodes.emplace_back(node.GetPtr());

		uint32_t& lightIndex = instanceLightMap.emplace_back(INVALID_INSTANCE);
		if (isLight) {
			lightIndex = (uint32_t)lightInstanceMap.size();
			lightInstanceMap.emplace_back(instanceIndex);
		}

		// transforms
		uint32_t& prevInstanceIndex = instanceIndexMap.emplace_back(INVALID_INSTANCE);

		float4x4 prevTransform = transform;
		if (auto it = prevInstanceTransforms.find(primPtr); it != prevInstanceTransforms.end()) {
			uint32_t idx;
			std::tie(prevTransform, prevInstanceIndex) = it->second;
		}
		mRenderData.mInstanceTransformMap.emplace(primPtr, std::make_pair(transform, instanceIndex));

		const float4x4 invTransform = inverse(transform);
		instanceTransforms.emplace_back(transform);
		instanceInverseTransforms.emplace_back(invTransform);
		instanceMotionTransforms.emplace_back(prevTransform * invTransform);
		return instanceIndex;
	};

	auto IsZero = [](const float3 v) { return !(v.r > 0 || v.g > 0 || v.b > 0); };

	auto GetAabbBlas = [&](const float3 mn, const float3 mx, const bool opaque) {
		const size_t key = HashArgs(mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], opaque);
		auto aabb_it = mAABBs.find(key);
		if (aabb_it != mAABBs.end())
			return aabb_it->second;

		Buffer::View<vk::AabbPositionsKHR> aabb = std::make_shared<Buffer>(
			commandBuffer.mDevice,
			"aabb data",
			sizeof(vk::AabbPositionsKHR),
			vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
		aabb[0].minX = mn[0];
		aabb[0].minY = mn[1];
		aabb[0].minZ = mn[2];
		aabb[0].maxX = mx[0];
		aabb[0].maxY = mx[1];
		aabb[0].maxZ = mx[2];
		vk::AccelerationStructureGeometryAabbsDataKHR aabbs(aabb.GetDeviceAddress(), sizeof(vk::AabbPositionsKHR));
		vk::AccelerationStructureGeometryKHR aabbGeometry(vk::GeometryTypeKHR::eAabbs, aabbs, opaque ? vk::GeometryFlagBitsKHR::eOpaque : vk::GeometryFlagBitsKHR{});
		vk::AccelerationStructureBuildRangeInfoKHR range(1);
		commandBuffer.HoldResource(aabb);

		auto [as, asbuf] = BuildAccelerationStructure(commandBuffer, "aabb BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, aabbGeometry, range);

		blasBarriers.emplace_back(
			vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
			VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
			**asbuf.GetBuffer(), asbuf.Offset(), asbuf.SizeBytes());

		return mAABBs.emplace(key, std::make_pair(as, asbuf)).first->second;
	};

	{ // mesh instances
		ProfilerScope s("Process mesh instances", &commandBuffer);
		mRootNode->ForEachDescendant<MeshRenderer>([&](SceneNode& primNode, const std::shared_ptr<MeshRenderer>& prim) {
			if (!primNode.Enabled() || !prim->mMesh || !prim->mMaterial) return;

			if (prim->mMesh->GetTopology() != vk::PrimitiveTopology::eTriangleList ||
				(prim->mMesh->GetIndexType() != vk::IndexType::eUint32 && prim->mMesh->GetIndexType() != vk::IndexType::eUint16) ||
				!prim->mMesh->GetVertices().find(Mesh::VertexAttributeType::ePosition)) {
				std::cout << "Skipping unsupported mesh in node " << primNode.GetName() << std::endl;
				return;
			}

			auto [positions, positionsDesc] = prim->mMesh->GetVertices().at(Mesh::VertexAttributeType::ePosition)[0];

			const uint32_t vertexCount = (uint32_t)((positions.SizeBytes() - positionsDesc.mOffset) / positionsDesc.mStride);
			const uint32_t primitiveCount = prim->mMesh->GetIndices().size() / (prim->mMesh->GetIndices().Stride() * 3);

			// get/build BLAS
			vk::DeviceAddress accelerationStructureAddress;
			if (useAccelerationStructure) {
				const size_t key = HashArgs(positions.GetBuffer(), positions.Offset(), positions.SizeBytes(), positionsDesc, prim->mMaterial->mMaterial.AlphaCutoff() == 0);
				auto it = mMeshAccelerationStructures.find(key);
				if (it == mMeshAccelerationStructures.end()) {
					ProfilerScope ps("Build acceleration structure", &commandBuffer);

					vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
					triangles.vertexFormat = positionsDesc.mFormat;
					triangles.vertexData = positions.GetDeviceAddress();
					triangles.vertexStride = positionsDesc.mStride;
					triangles.maxVertex = vertexCount;
					triangles.indexType = prim->mMesh->GetIndexType();
					triangles.indexData = prim->mMesh->GetIndices().GetDeviceAddress();
					vk::AccelerationStructureGeometryKHR triangleGeometry(vk::GeometryTypeKHR::eTriangles, triangles, prim->mMaterial->mMaterial.AlphaCutoff() == 0 ? vk::GeometryFlagBitsKHR::eOpaque : vk::GeometryFlagBitsKHR{});
					vk::AccelerationStructureBuildRangeInfoKHR range(primitiveCount);

					auto [as, asbuf] = BuildAccelerationStructure(commandBuffer, primNode.GetName() + "/BLAS", vk::AccelerationStructureTypeKHR::eBottomLevel, triangleGeometry, range);

					blasBarriers.emplace_back(
						vk::AccessFlagBits::eAccelerationStructureWriteKHR, vk::AccessFlagBits::eAccelerationStructureReadKHR,
						VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
						**asbuf.GetBuffer(), asbuf.Offset(), asbuf.SizeBytes());

					it = mMeshAccelerationStructures.emplace(key, std::make_pair(as, asbuf)).first;
				}

				accelerationStructureAddress = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**it->second.first);
			}

			// assign vertex buffers
			Buffer::View<std::byte> normals, texcoords;
			Mesh::VertexAttributeDescription normalsDesc = {}, texcoordsDesc = {};
			if (auto attrib = prim->mMesh->GetVertices().find(Mesh::VertexAttributeType::eNormal))
				tie(normals, normalsDesc) = *attrib;
			if (auto attrib = prim->mMesh->GetVertices().find(Mesh::VertexAttributeType::eTexcoord))
				tie(texcoords, texcoordsDesc) = *attrib;

			const uint32_t vertexInfoIndex = (uint32_t)meshVertexInfos.size();

			meshVertexInfos.emplace_back(
				AddVertexBuffer(prim->mMesh->GetIndices().GetBuffer()), (uint32_t)prim->mMesh->GetIndices().Offset(), (uint32_t)prim->mMesh->GetIndices().Stride(),
				AddVertexBuffer(positions.GetBuffer()), (uint32_t)positions.Offset() + positionsDesc.mOffset, positionsDesc.mStride,
				AddVertexBuffer(normals.GetBuffer())  , (uint32_t)normals.Offset()   + normalsDesc.mOffset  , normalsDesc.mStride,
				AddVertexBuffer(texcoords.GetBuffer()), (uint32_t)texcoords.Offset() + texcoordsDesc.mOffset, texcoordsDesc.mStride);

			const uint32_t materialIndex = AddMaterial(*prim->mMaterial);
			const float4x4 transform = NodeToWorld(primNode);
			const uint32_t triCount = prim->mMesh->GetIndices().SizeBytes() / (prim->mMesh->GetIndices().Stride() * 3);

			const uint32_t instanceIdx = AddInstance(primNode, prim.get(), MeshInstance(materialIndex, vertexInfoIndex, primitiveCount), transform, !IsZero(prim->mMaterial->mMaterial.Emission()));

			if (useAccelerationStructure) {
				vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
				float3x4 t = (float3x4)transpose(transform);
				instance.transform = std::bit_cast<vk::TransformMatrixKHR>(t);
				instance.instanceCustomIndex = instanceIdx;
				instance.mask = BVH_FLAG_TRIANGLES;
				instance.accelerationStructureReference = accelerationStructureAddress;
			}

			const vk::AabbPositionsKHR& aabb = prim->mMesh->GetVertices().mAabb;
			for (uint32_t i = 0; i < 8; i++) {
				const int3 idx(i % 2, (i % 4) / 2, i / 4);
				float3 corner(
					idx[0] == 0 ? aabb.minX : aabb.maxX,
					idx[1] == 0 ? aabb.minY : aabb.maxY,
					idx[2] == 0 ? aabb.minZ : aabb.maxZ);
				corner = TransformPoint(transform, corner);
				aabbMin = min(aabbMin, corner);
				aabbMax = max(aabbMax, corner);
			}
		});
	}

	{ // sphere instances
		ProfilerScope s("Process sphere instances", &commandBuffer);
		mRootNode->ForEachDescendant<SphereRenderer>([&](SceneNode& primNode, const std::shared_ptr<SphereRenderer>& prim) {
			if (!primNode.Enabled() || !prim->mMaterial) return;

			const float4x4 transform = glm::translate(TransformPoint(NodeToWorld(primNode), float3(0)));
			const float radius = prim->mRadius * glm::determinant((float3x3)transform);

			vk::DeviceAddress accelerationStructureAddress;
			if (useAccelerationStructure) {
				const auto& [as, asbuf] = GetAabbBlas(-float3(radius), float3(radius), prim->mMaterial->mMaterial.AlphaCutoff() == 0);
				accelerationStructureAddress = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**as);
			}

			const uint32_t materialIndex = AddMaterial(*prim->mMaterial);
			const uint32_t instanceIdx = AddInstance(primNode, prim.get(), SphereInstance(materialIndex, radius), transform, !IsZero(prim->mMaterial->mMaterial.Emission()));

			if (useAccelerationStructure) {
				vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
				float3x4 t = (float3x4)transpose(transform);
				instance.transform = std::bit_cast<vk::TransformMatrixKHR>(t);
				instance.instanceCustomIndex = instanceIdx;
				instance.mask = BVH_FLAG_SPHERES;
				instance.accelerationStructureReference = accelerationStructureAddress;
			}

			const float3 center = TransformPoint(transform, float3(0));
			aabbMin = min(aabbMin, center - float3(radius));
			aabbMax = max(aabbMax, center + float3(radius));
		});
	}
	/*
	{ // medium instances
		ProfilerScope s("Process media", &commandBuffer);
		mRootNode->ForEachDescendant<VolumeRenderer>([&](SceneNode& primNode, const std::shared_ptr<VolumeRenderer>& vol) {
			if (!primNode.Enabled() || !vol) return;

			float3 mn = { -1, -1, -1 };
			float3 mx = {  1,  1,  1 };
			if (vol->mDensityGrid) {
				auto densityGrid = vol->mDensityGrid->grid<float>();
				nanovdb::Vec3R mnd = densityGrid->worldBBox().min();
				nanovdb::Vec3R mxd = densityGrid->worldBBox().max();
				mn = (float3)double3(mnd[0], mnd[1], mnd[2]);
				mx = (float3)double3(mxd[0], mxd[1], mxd[2]);
			}

			vk::DeviceAddress accelerationStructureAddress;
			if (useAccelerationStructure) {
				const auto& [as, asbuf] = GetAabbBlas(mn, mx, true);
				accelerationStructureAddress = commandBuffer.mDevice->getAccelerationStructureAddressKHR(**as);
			}

			const uint32_t materialIndex = AddMaterial(*vol);

			// append to instance list
			const float4x4 transform = NodeToWorld(primNode);
			vk::AccelerationStructureInstanceKHR& instance = instancesAS.emplace_back();
			float3x4::Map(&instance.transform.matrix[0][0]) = transform.to_float3x4();
			instance.instanceCustomIndex = appendInstance(primNode, vol.get(), VolumeInstance(materialIndex, vol->mDensityBuffer ? mRenderData.mMaterialResources.mVolumeDataMap.at({ vol->mDensityBuffer.buffer(),vol->mDensityBuffer.offset() }) : -1), transform, 0, {});
			instance.mask = BVH_FLAG_VOLUME;
			instance.accelerationStructureReference = accelerationStructureAddress;

			volumeInfos.emplace_back(VolumeInfo{mn, instance.instanceCustomIndex, mx, 0u});

			for (uint32_t i = 0; i < 8; i++) {
				const int3 idx(i % 2, (i % 4) / 2, i / 4);
				float3 corner(idx[0] == 0 ? mn[0] : mx[0],
					idx[1] == 0 ? mn[1] : mx[1],
					idx[2] == 0 ? mn[2] : mx[2]);
				corner = TransformPoint(transform, corner);
				mn = min(mn, corner);
				mx = max(mx, corner);
			}
			aabbMin = min(aabbMin, mn);
			aabbMax = max(aabbMax, mx);
		});
	}
	*/
	{ // environment material
		ProfilerScope s("Process environment", &commandBuffer);
		mRenderData.mShaderParameters.SetConstant("mBackgroundColor", float3(0));
		mRenderData.mShaderParameters.SetConstant("mBackgroundImageIndex", ~uint32_t(0));
		mRenderData.mShaderParameters.SetConstant("mBackgroundSampleProbability", 0.f);
		mRootNode->ForEachDescendant<EnvironmentMap>([&](SceneNode& node, const std::shared_ptr<EnvironmentMap> environment) {
			if (!node.Enabled() || IsZero(environment->mColor)) return true;
			mRenderData.mShaderParameters.SetConstant("mBackgroundColor", environment->mColor);
			mRenderData.mShaderParameters.SetConstant("mBackgroundImageIndex", AddImage4(environment->mImage));
			mRenderData.mShaderParameters.SetConstant("mBackgroundSampleProbability", lightInstanceMap.empty() ? 1.0f : 0.5f);
			return false;
		});
	}

	// Build TLAS
	if (useAccelerationStructure) {
		ProfilerScope s("Build TLAS", &commandBuffer);
		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR, vk::DependencyFlagBits::eByRegion, {}, blasBarriers, {});

		vk::AccelerationStructureGeometryKHR geom{ vk::GeometryTypeKHR::eInstances, vk::AccelerationStructureGeometryInstancesDataKHR() };
		vk::AccelerationStructureBuildRangeInfoKHR range{ (uint32_t)instancesAS.size() };
		if (!instancesAS.empty()) {
			std::shared_ptr<Buffer> tmp = std::make_shared<Buffer>(commandBuffer.mDevice, "TLAS instance buffer",
				sizeof(vk::AccelerationStructureInstanceKHR) * instancesAS.size(),
				vk::BufferUsageFlagBits::eTransferSrc,
				vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

			std::shared_ptr<Buffer> buf = std::make_shared<Buffer>(commandBuffer.mDevice, "TLAS instance buffer",
				tmp->size() + 16, // extra 16 bytes for alignment
				vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress);

			const size_t address = (size_t)buf->GetDeviceAddress();
			const size_t offset = (-address & 15); // aligned = unaligned + (-unaligned & (alignment - 1))

			geom.geometry.instances.data = address + offset;
			memcpy(tmp->data(), instancesAS.data(), instancesAS.size()*sizeof(vk::AccelerationStructureInstanceKHR));
			commandBuffer.Copy(tmp, Buffer::View<std::byte>(buf, offset, tmp->size()));
			commandBuffer.HoldResource(tmp);
			commandBuffer.HoldResource(buf);

		}

		const auto&[ as, asbuf ] = BuildAccelerationStructure(commandBuffer, "TLAS", vk::AccelerationStructureTypeKHR::eTopLevel, geom, range);
		mRenderData.mShaderParameters.SetAccelerationStructure("mAccelerationStructure", as);
		mRenderData.mShaderParameters.SetBuffer("mAccelerationStructureBuffer", asbuf);
	}

	{ // upload data
		ProfilerScope s("Upload scene data buffers");
		mRenderData.mShaderParameters.SetBuffer("mInstances",                 commandBuffer.Upload<InstanceBase>  (instanceDatas,             "mInstances", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mInstanceTransforms",        commandBuffer.Upload<float4x4>      (instanceTransforms,        "mInstanceTransforms", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mInstanceInverseTransforms", commandBuffer.Upload<float4x4>      (instanceInverseTransforms, "mInstanceInverseTransforms", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mInstanceMotionTransforms",  commandBuffer.Upload<float4x4>      (instanceMotionTransforms,  "mInstanceMotionTransforms", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mLightInstanceMap",          commandBuffer.Upload<uint32_t>      (lightInstanceMap,          "mLightInstanceMap", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mInstanceLightMap",          commandBuffer.Upload<uint32_t>      (instanceLightMap,          "mInstanceLightMap", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mMeshVertexInfo",            commandBuffer.Upload<MeshVertexInfo>(meshVertexInfos,           "mMeshVertexInfo", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mInstanceVolumeInfo",        commandBuffer.Upload<VolumeInfo>    (volumeInfos,               "mInstanceVolumeInfo", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mShaderParameters.SetBuffer("mMaterials",                 commandBuffer.Upload<GpuMaterial>   (materials,                 "mMaterials", vk::BufferUsageFlagBits::eStorageBuffer));
		mRenderData.mInstanceIndexMap = commandBuffer.Upload<uint32_t>(instanceIndexMap, "mInstanceIndexMap", vk::BufferUsageFlagBits::eStorageBuffer);
	}
	mRenderData.mShaderParameters.SetConstant("mSceneMin", aabbMin);
	mRenderData.mShaderParameters.SetConstant("mSceneMax", aabbMax);
	mRenderData.mShaderParameters.SetConstant("mInstanceCount", (uint32_t)instanceDatas.size());
	mRenderData.mShaderParameters.SetConstant("mLightCount", (uint32_t)lightInstanceMap.size());
}

}
#include "FlyCamera.hpp"

#include "Scene.hpp"

#include <App/App.hpp>
#include <Core/Profiler.hpp>

#include <imgui/imgui.h>
#include <ImGuizmo.h>

namespace ptvk {

bool FlyCamera::OnInspectorGui() {
	ImGui::DragFloat("Move Speed", &mMoveSpeed, 0.1f, 0);
	if (mMoveSpeed < 0) mMoveSpeed = 0;
	ImGui::DragFloat("Rotate Speed", &mRotateSpeed, 0.001f, 0);
	if (mRotateSpeed < 0) mRotateSpeed = 0;

	if (ImGui::DragFloat2("Rotation", &mRotation.x, 0.01f, 0))
		mRotation.x = clamp(mRotation.x, -((float)M_PI) / 2, ((float)M_PI) / 2);
	return false;
}

void FlyCamera::Update(const float deltaTime) {
	ProfilerScope ps("FlyCamera::Update");

	if (!mNode.HasComponent<float4x4>())
		mNode.MakeComponent<float4x4>(glm::identity<float4x4>());

	float4x4& transform = *mNode.GetComponent<float4x4>();

	const ImGuiIO& io = ImGui::GetIO();

	bool update = false;
	float3 pos = TransformPoint(transform, float3(0));
	if (!io.WantCaptureKeyboard) {
		// wasd camera
		float3 mv = float3(0, 0, 0);
		if (ImGui::IsKeyDown(ImGuiKey_D))     mv += float3(1, 0, 0);
		if (ImGui::IsKeyDown(ImGuiKey_A))     mv += float3(-1, 0, 0);
		if (ImGui::IsKeyDown(ImGuiKey_W))     mv += float3(0, 0, -1);
		if (ImGui::IsKeyDown(ImGuiKey_S))     mv += float3(0, 0, 1);
		if (ImGui::IsKeyDown(ImGuiKey_Space)) mv += float3(0, 1, 0);
		if (ImGui::IsKeyDown(ImGuiKey_C))     mv += float3(0, -1, 0);
		if (mv.x != 0 || mv.y != 0 || mv.z != 0) {
			if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) mv *= 5;
			pos += TransformVector(transform, mv * mMoveSpeed * deltaTime);
			update = true;
		}
	}
	if (App::mIsViewportFocused && !ImGuizmo::IsUsing()) {
		if (ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
			// mouse rotation
			if (io.MouseWheel != 0)
				mMoveSpeed *= (1 + io.MouseWheel / 8);

			mRotation[1] -= io.MouseDelta.x * mRotateSpeed;
			mRotation[0] = clamp(mRotation[0] - io.MouseDelta.y * mRotateSpeed, -((float)M_PI) / 2, ((float)M_PI) / 2);

			update = true;
		} else if (ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
			// pan
			ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
			pos += TransformVector(transform, (mMoveSpeed/io.DisplaySize.x) * float3(-io.MouseDelta.x, io.MouseDelta.y, 0) );
			update = true;
		}
	}
	if (update)
		transform = glm::translate(pos) * (glm::rotate(mRotation.y, float3(0, 1, 0)) * glm::rotate(mRotation.x, float3(1, 0, 0)));
}

}

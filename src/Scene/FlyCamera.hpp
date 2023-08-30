#pragma once

#include <Common/Math.h>
#include "SceneNode.hpp"

namespace ptvk {

struct FlyCamera {
	SceneNode& mNode;

	float mMoveSpeed = 1;
	float mRotateSpeed = 0.002f;
	float2 mRotation = float2(0);

	void Update(const float deltaTime);

	bool OnInspectorGui();
};

}
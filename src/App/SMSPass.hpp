#pragma once

#include "VisibilityPass.hpp"

namespace ptvk {

enum StepMode {
	eFixed,
    eHessian,
    eHessianEigenDecomp
};
static const char* StepModeStrings[] = {
	"Fixed",
	"Hessian",
	"Hessian (Eigen Decomp)"
};

class SMSPass {
private:
	ComputePipelineCache mSampleCameraPathsPipeline;
	ComputePipelineCache mCopyDebugImagePipeline;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mForceLambertian = false;
	bool mRussianRoullette = true;

	uint32_t mMaxBounces = 4;
	uint32_t mMinManifoldVertices = 0;
	uint32_t mMaxManifoldVertices = 0;

	bool mFixedSeed = false;
	uint32_t mRandomSeed = 0;

	uint32_t mManifoldSolverIterations = 16;
	float mManifoldSolverStepSize  = 1;
	float mManifoldSolverThreshold = 1; // degrees
	StepMode mManifoldStepMode = StepMode::eHessianEigenDecomp;

	Buffer::View<std::byte> mDebugImage;

public:
	inline SMSPass(Device& device) {
		auto staticSampler = std::make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
			vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
			0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));
		PipelineInfo md;
		md.mImmutableSamplers["gScene.mStaticSampler"]  = { staticSampler };
		md.mBindingFlags["gScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mImage4s"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

		const std::vector<std::string>& args = {
			"-O3",
			"-Wno-30081",
			"-capability", "spirv_1_5",
			"-capability", "GL_EXT_ray_tracing"
		};

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path") + "/Kernels/SMS.slang";
		mSampleCameraPathsPipeline = ComputePipelineCache(shaderFile, "SampleCameraPaths", "sm_6_7", args, md);
		mCopyDebugImagePipeline    = ComputePipelineCache(shaderFile, "CopyDebugImage"   , "sm_6_7", args, md);
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Russian roullette", &mRussianRoullette);
		ImGui::Checkbox("Force lambertian", &mForceLambertian);
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 0, 32, .5f);
		Gui::ScalarField<uint32_t>("Min manifold vertices", &mMinManifoldVertices, 0, 16, .1f);
		Gui::ScalarField<uint32_t>("Max manifold vertices", &mMaxManifoldVertices, 0, 16, .1f);
		if (mMaxManifoldVertices > 0) {
			ImGui::Separator();
			Gui::ScalarField<uint32_t>("Solver iterations", &mManifoldSolverIterations, 0, 1024);
			Gui::ScalarField<float>("Constraint threshold", &mManifoldSolverThreshold, 0, 1, .1f);
			Gui::ScalarField<float>("Step size", &mManifoldSolverStepSize, 0, 10, .01f);
			Gui::EnumDropdown<StepMode>("Step mode", mManifoldStepMode, StepModeStrings);
		}

		ImGui::Checkbox("Fix seed", &mFixedSeed);
		if (mFixedSeed) {
			ImGui::SameLine();
			Gui::ScalarField<uint32_t>("##", &mRandomSeed);
		}

		ImGui::PopID();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope p("SMSPass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);

		if (!mDebugImage || mDebugImage.SizeBytes() != vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y)*sizeof(uint32_t))
			mDebugImage = std::make_shared<Buffer>(commandBuffer.mDevice, "gDebugImage", vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y)*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);

		Defines defs;
		if (mAlphaTest)
			defs.emplace("gAlphaTest", "true");
		if (mShadingNormals)
			defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)
			defs.emplace("gNormalMaps", "true");
		if (mMaxManifoldVertices > 0 && mMaxBounces > 1) {
			defs.emplace("MANIFOLD_SAMPLING", "true");
			if (mMaxManifoldVertices > 1)
				defs.emplace("MANIFOLD_MULTI_BOUNCE", "true");
			defs.emplace("gStepMode", "((StepMode)" + std::to_string((uint32_t)mManifoldStepMode) + ")");
		}
		if (mForceLambertian)
			defs.emplace("FORCE_LAMBERTIAN", "true");
		if (!mRussianRoullette)
			defs.emplace("DISABLE_STOCHASTIC_TERMINATION", "true");
		if (visibility.GetDebugPixel())
			defs.emplace("DEBUG_PIXEL", "true");
		if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
			defs.emplace("gEnableDebugCounters", "true");

		ShaderParameterBlock params;
		params.SetParameters("gScene", scene.GetRenderData().mShaderParameters);
		params.SetParameters(visibility.GetDebugParameters());
		params.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
		params.SetImage("gVertices", visibility.GetVertices(), vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
		params.SetBuffer("gDebugImage", mDebugImage);
		params.SetConstant("gCameraPosition", visibility.GetCameraPosition());
		params.SetConstant("gMVP", visibility.GetMVP());
		params.SetConstant("gOutputSize", extent);
		params.SetConstant("gRandomSeed", mRandomSeed);
		params.SetConstant("gMaxBounces", mMaxBounces);
		params.SetConstant("gMinManifoldVertices", mMinManifoldVertices);
		params.SetConstant("gMaxManifoldVertices", mMaxManifoldVertices);
		params.SetConstant("gManifoldSolverIterations", mManifoldSolverIterations);
		params.SetConstant("gManifoldSolverStepSize", mManifoldSolverStepSize);
		params.SetConstant("gManifoldSolverThreshold", 1 - std::cosf(mManifoldSolverThreshold * (M_PI/180)));

		if (!mFixedSeed) mRandomSeed++;

		if (visibility.GetDebugPixel())
			commandBuffer.Fill(mDebugImage, 0);

		auto pipeline = mSampleCameraPathsPipeline.GetPipelineAsync(commandBuffer.mDevice, defs);
		if (pipeline) {
			ProfilerScope p("Sample Paths", &commandBuffer);
			mSampleCameraPathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *pipeline);
		} else {
			const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
			ImGui::SetNextWindowPos(ImVec2(size.x/2, size.y/2));
			if (ImGui::Begin("Compiling shaders", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoInputs)) {
				Gui::ProgressSpinner("Compiling shaders", 15, 6, false);
			}
			ImGui::End();
		}

		if (visibility.GetDebugPixel()) {
			mCopyDebugImagePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
		}
	}
};

}
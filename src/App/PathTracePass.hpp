#pragma once

#include "VisibilityPass.hpp"

namespace ptvk {

class PathTracePass {
private:
	ComputePipelineCache mSampleCameraPathsPipeline;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mSampleLights = true;
	bool mDisneyBrdf = false;

	uint32_t mMaxBounces = 4;
	uint32_t mAccumulationStart = 0;

public:
	inline PathTracePass(Device& device) {
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

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path") + "/Kernels/PathTracer.slang";
		mSampleCameraPathsPipeline = ComputePipelineCache(shaderFile, "SampleCameraPaths", "sm_6_7", args, md);
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Sample lights", &mSampleLights);
		ImGui::Checkbox("Disney brdf", &mDisneyBrdf);
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 0, 32);
		ImGui::PopID();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope p("PathTracePass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);

		Defines defs;
		if (mAlphaTest)      defs.emplace("gAlphaTest", "true");
		if (mShadingNormals) defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)     defs.emplace("gNormalMaps", "true");
		if (mSampleLights)   defs.emplace("SAMPLE_LIGHTS", "true");
		if (mDisneyBrdf)     defs.emplace("DISNEY_BRDF", "true");
		if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
			defs.emplace("gEnableDebugCounters", "true");

		ShaderParameterBlock params;
		params.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
		params.SetImage("gVertices", visibility.GetVertices(), vk::ImageLayout::eGeneral);
		params.SetConstant("gOutputSize", extent);
		params.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart));
		params.SetConstant("gMaxBounces", mMaxBounces);
		params.SetConstant("gCameraPosition", visibility.GetCameraPosition());
		params.SetParameters("gScene", scene.GetRenderData().mShaderParameters);
		params.SetParameters(visibility.GetDebugParameters());

		int i = 0;
		{
			ProfilerScope p("Sample Paths", &commandBuffer);
			mSampleCameraPathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
			i ^= 1;
		}
	}
};

}
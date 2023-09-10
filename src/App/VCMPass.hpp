#pragma once

#include "VisibilityPass.hpp"

namespace ptvk {

class VCMPass {
private:
	ComputePipelineCache mSampleCameraPathsPipeline;
	ComputePipelineCache mSampleLightPathsPipeline;
	ComputePipelineCache mAddLightImagePipeline;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mSampleLights = true;
	bool mDisneyBrdf = false;

	bool mLightTrace = false;
	bool mLightTraceOnly = false;

	uint32_t mMaxBounces = 4;
	uint32_t mAccumulationStart = 0;

	Buffer::View<uint4> mLightImage;

public:
	inline VCMPass(Device& device) {
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

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path") + "/Kernels/VCM.slang";
		mSampleCameraPathsPipeline = ComputePipelineCache(shaderFile, "SampleCameraPaths", "sm_6_7", args, md);
		mSampleLightPathsPipeline  = ComputePipelineCache(shaderFile, "SampleLightPaths" , "sm_6_7", args, md);
		mAddLightImagePipeline     = ComputePipelineCache(shaderFile, "AddLightImage"    , "sm_6_7", args, md);
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Sample lights", &mSampleLights);
		ImGui::Checkbox("Disney brdf", &mDisneyBrdf);
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 0, 32);
		ImGui::Checkbox("Light trace", &mLightTrace);
		if (mLightTrace)
			ImGui::Checkbox("Light trace only", &mLightTraceOnly);
		ImGui::PopID();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope p("VCMPass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);

		if (!mLightImage || mLightImage.size() != extent.x*extent.y) {
			mLightImage = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightImage", extent.x*extent.y*sizeof(uint4), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
		}

		Defines defs;
		if (mAlphaTest)      defs.emplace("gAlphaTest", "true");
		if (mShadingNormals) defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)     defs.emplace("gNormalMaps", "true");
		if (mDisneyBrdf)     defs.emplace("DISNEY_BRDF", "true");
		if (mSampleLights)   defs.emplace("gSampleLights", "true");
		if (mLightTrace) {
			defs.emplace("gLightTrace", "true");
			if (mLightTraceOnly) defs.emplace("gLightTraceOnly", "true");
		}
		if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
			defs.emplace("gEnableDebugCounters", "true");

		const float imagePlaneDist = extent.y / (2 * std::tan(visibility.GetVerticalFov()/2));

		ShaderParameterBlock params;
		params.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
		params.SetImage("gVertices", visibility.GetVertices(), vk::ImageLayout::eGeneral);
		params.SetBuffer("gLightImage", mLightImage);
		params.SetConstant("gOutputSize", extent);
		params.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart));
		params.SetConstant("gMaxBounces", mMaxBounces);
		params.SetConstant("gCameraPosition", visibility.GetCameraPosition());
		params.SetConstant("gCameraForward", visibility.GetCameraForward());
		params.SetConstant("gCameraImagePlaneDist", imagePlaneDist);
		params.SetConstant("gMVP", visibility.GetMVP());
		params.SetParameters("gScene", scene.GetRenderData().mShaderParameters);
		params.SetParameters(visibility.GetDebugParameters());

		if (mLightTrace) {
			commandBuffer.Fill(mLightImage, 0);
			{
				ProfilerScope p("Sample Light Paths", &commandBuffer);
				mSampleLightPathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
			}

			{
				ProfilerScope p("Add Light Image", &commandBuffer);
				mAddLightImagePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
			}
		}

		if (!mLightTrace || !mLightTraceOnly) {
			ProfilerScope p("Sample Camera Paths", &commandBuffer);
			mSampleCameraPathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
		}
	}
};

}
#pragma once

#include <Common/Common.h>
#include <Core/PipelineCache.hpp>
#include <Scene/Scene.hpp>

namespace ptvk {

class PathTracePass {
private:
	ComputePipelineCache mRenderPipeline;
	Buffer::View<uint32_t> mRayCountBuffer;
	uint32_t mAccumulationStart;

	bool mAlphaTest;
	bool mCountRays;
	bool mShadingNormals;
	bool mNormalMaps;

	Image::View mPositions;
	Image::View mAlbedo;

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

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path")  + "/Kernels/PathTrace.slang";
		mRenderPipeline = ComputePipelineCache(shaderFile, "Render", "sm_6_7", args, md);

		mRayCountBuffer = std::make_shared<Buffer>(device, "gRayCount", 16, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);

		mAccumulationStart = 0;

		mAlphaTest = false;
		mShadingNormals = false;
		mNormalMaps = false;
		mCountRays = false;
	}

	inline Image::View GetPositions() const { return mPositions; }
	inline Image::View GetAlbedo()    const { return mAlbedo; }

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const float4x4& cameraToWorld, const float4x4& projection) {
		ProfilerScope p("PathTracePass::Render");

		uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);

		if (!mPositions || mPositions.GetExtent().width != extent.x || mPositions.GetExtent().height != extent.y) {
			mPositions = std::make_shared<Image>(commandBuffer.mDevice, "gPositions", ImageInfo{
				.mFormat = vk::Format::eR32G32B32A32Sfloat,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc
			});
			mAlbedo = std::make_shared<Image>(commandBuffer.mDevice, "gAlbedo", ImageInfo{
				.mFormat = vk::Format::eR8G8B8A8Unorm,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage
			});
		}

		commandBuffer.Fill(mRayCountBuffer, 0);

		Defines defs;
		if (mAlphaTest)      defs.emplace("gAlphaTest", "true");
		if (mCountRays)      defs.emplace("gCountRays", "true");
		if (mShadingNormals) defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)     defs.emplace("gNormalMaps", "true");

		mRenderPipeline.Dispatch(commandBuffer,
			renderTarget.GetExtent(),
			ShaderParameterBlock()
				.SetImage("gOutput"         , renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetImage("gOutputPositions", mPositions  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetImage("gOutputAlbedo"   , mAlbedo     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetConstant("gOutputSize", extent)
				.SetConstant("gCameraToWorld", cameraToWorld)
				.SetConstant("gInverseProjection", inverse(projection))
				.SetParameters("gScene", scene.GetRenderData().mShaderParameters)
				.SetBuffer("gRayCount", mRayCountBuffer)
				.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart)),
			defs
		);
	}

	inline void OnInspectorGui() {
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Count rays", &mCountRays);
	}
};

}
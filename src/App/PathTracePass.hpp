#pragma once

#include <Common/Enums.h>
#include <Core/PipelineCache.hpp>
#include <Scene/Scene.hpp>

namespace ptvk {

class PathTracePass {
private:
	ComputePipelineCache mRenderMegakernel;
	ComputePipelineCache mRenderVisibility;
	ComputePipelineCache mRenderBounce;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mCountRays = false;
	bool mSampleLights = true;
	bool mDisneyBrdf = true;
	bool mMegakernel = true;

	uint32_t mAccumulationStart = 0;
	uint32_t mMaxBounces = 4;

	Image::View mPositions;
	Image::View mAlbedo;
	Buffer::View<std::byte> mPathState;
	Buffer::View<uint32_t> mDebugCounters;
	Image::View mDebugHeatmap;

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
		mRenderMegakernel = ComputePipelineCache(shaderFile, "RenderMegakernel", "sm_6_7", args, md);
		mRenderVisibility = ComputePipelineCache(shaderFile, "RenderVisibility", "sm_6_7", args, md);
		mRenderBounce     = ComputePipelineCache(shaderFile, "RenderBounce"    , "sm_6_7", args, md);
	}

	inline Image::View GetPositions() const { return mPositions; }
	inline Image::View GetAlbedo()    const { return mAlbedo; }

	inline void OnInspectorGui() {
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Count rays", &mCountRays);
		ImGui::Checkbox("Sample lights", &mSampleLights);
		ImGui::Checkbox("Disney brdf", &mDisneyBrdf);
		ImGui::Checkbox("Megakernel", &mMegakernel);
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 0, 32);
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const float4x4& cameraToWorld, const float4x4& projection) {
		ProfilerScope p("PathTracePass::Render", &commandBuffer);

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
			mDebugHeatmap = std::make_shared<Image>(commandBuffer.mDevice, "gHeatmapCounters", ImageInfo{
				.mFormat = vk::Format::eR32Uint,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});
			mPathState = std::make_shared<Buffer>(commandBuffer.mDevice, "gPathState", 64*vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y), vk::BufferUsageFlagBits::eStorageBuffer);
			mDebugCounters = std::make_shared<Buffer>(commandBuffer.mDevice, "gRayCount", ((uint32_t)DebugCounterType::eNumDebugCounters + 1)*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
		}

		if (mCountRays) {
			commandBuffer.Fill(mDebugCounters, 0);
			commandBuffer.ClearColor(mDebugHeatmap, vk::ClearColorValue{ std::array<uint32_t,4>{0,0,0,0} });
		}

		Defines defs;
		if (mAlphaTest)      defs.emplace("gAlphaTest", "true");
		if (mCountRays)      defs.emplace("gEnableDebugCounters", "true");
		if (mShadingNormals) defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)     defs.emplace("gNormalMaps", "true");
		if (mSampleLights)   defs.emplace("SAMPLE_LIGHTS", "true");
		if (mDisneyBrdf)     defs.emplace("DISNEY_BRDF", "true");

		if (mMegakernel) {
			mRenderMegakernel.Dispatch(commandBuffer,
				renderTarget.GetExtent(),
				ShaderParameterBlock()
					.SetImage("gOutput"         , renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
					.SetImage("gOutputPositions", mPositions  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
					.SetImage("gOutputAlbedo"   , mAlbedo     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
					.SetBuffer("gPathState", mPathState)
					.SetConstant("gOutputSize", extent)
					.SetConstant("gCameraToWorld", cameraToWorld)
					.SetConstant("gInverseProjection", inverse(projection))
					.SetParameters("gScene", scene.GetRenderData().mShaderParameters)
					.SetBuffer("gDebugCounters", mDebugCounters)
					.SetImage("gHeatmapCounters", mDebugHeatmap, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
					.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart))
					.SetConstant("gMaxBounces", mMaxBounces),
				defs
			);
		} else {
			mRenderVisibility.Dispatch(commandBuffer,
				renderTarget.GetExtent(),
				ShaderParameterBlock()
					.SetImage("gOutput"         , renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
					.SetImage("gOutputPositions", mPositions  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
					.SetImage("gOutputAlbedo"   , mAlbedo     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
					.SetBuffer("gPathState", mPathState)
					.SetConstant("gOutputSize", extent)
					.SetConstant("gCameraToWorld", cameraToWorld)
					.SetConstant("gInverseProjection", inverse(projection))
					.SetParameters("gScene", scene.GetRenderData().mShaderParameters)
					.SetBuffer("gDebugCounters", mDebugCounters)
					.SetImage("gHeatmapCounters", mDebugHeatmap, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
					.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart))
					.SetConstant("gMaxBounces", mMaxBounces),
				defs
			);
			for (uint32_t i = 0; i < mMaxBounces; i++) {
				commandBuffer.Barrier(renderTarget, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite|vk::AccessFlagBits::eShaderRead);
				commandBuffer.Barrier(mPathState, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderWrite|vk::AccessFlagBits::eShaderRead);
				mRenderBounce.Dispatch(commandBuffer,
					renderTarget.GetExtent(),
					ShaderParameterBlock()
						.SetImage("gOutput"         , renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite|vk::AccessFlagBits::eShaderRead)
						.SetImage("gOutputPositions", mPositions  , vk::ImageLayout::eGeneral, {})
						.SetImage("gOutputAlbedo"   , mAlbedo     , vk::ImageLayout::eGeneral, {})
						.SetBuffer("gPathState", mPathState)
						.SetConstant("gOutputSize", extent)
						.SetConstant("gCameraToWorld", cameraToWorld)
						.SetConstant("gInverseProjection", inverse(projection))
						.SetParameters("gScene", scene.GetRenderData().mShaderParameters)
						.SetBuffer("gDebugCounters", mDebugCounters)
						.SetImage("gHeatmapCounters", mDebugHeatmap, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
						.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart))
						.SetConstant("gMaxBounces", mMaxBounces),
					defs
				);
			}
		}
	}
};

}
#pragma once

#include <Common/Enums.h>
#include <Core/PipelineCache.hpp>
#include <Scene/Scene.hpp>
#include "AccumulatePass.hpp"

namespace ptvk {

class ReSTIRPTPass {
private:
	ComputePipelineCache mSamplePaths;
	ComputePipelineCache mTemporalReuse;
	ComputePipelineCache mSpatialReuse;
	ComputePipelineCache mOutputRadiance;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mCountRays = false;

	Image::View mPositions;
	Image::View mAlbedo;
	Buffer::View<uint32_t> mDebugCounters;
	Image::View mDebugHeatmap;
	std::array<Buffer::View<std::byte>, 2> mPathReservoirsBuffers;

	uint32_t mAccumulationStart = 0;

	Buffer::View<std::byte> mPrevReservoirs;
	bool mHasHistory;

public:
	inline ReSTIRPTPass(Device& device) {
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

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path")  + "/Kernels/ReSTIR.slang";
		mSamplePaths    = ComputePipelineCache(shaderFile, "SampleCanonicalPaths", "sm_6_7", args, md);
		mTemporalReuse  = ComputePipelineCache(shaderFile, "TemporalReuse"       , "sm_6_7", args, md);
		mSpatialReuse   = ComputePipelineCache(shaderFile, "SpatialReuse"        , "sm_6_7", args, md);
		mOutputRadiance = ComputePipelineCache(shaderFile, "OutputRadiance"      , "sm_6_7", args, md);

		mDebugCounters = std::make_shared<Buffer>(device, "gRayCount", ((uint32_t)DebugCounterType::eNumDebugCounters + 1)*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
	}

	inline Image::View GetPositions() const { return mPositions; }
	inline Image::View GetAlbedo()    const { return mAlbedo; }

	inline void OnInspectorGui() {
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Count rays", &mCountRays);
	}

    inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const float4x4& cameraToWorld, const float4x4& projection, const AccumulatePass& accum) {
		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);

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
				.mFormat = vk::Format::eR32G32B32Uint,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});
            for (int i = 0; i < 2; i++)
                mPathReservoirsBuffers[i] = std::make_shared<Buffer>(commandBuffer.mDevice, "gReservoirs" + std::to_string(i), vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y)*128, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc);
			mPrevReservoirs = std::make_shared<Buffer>(commandBuffer.mDevice, "gPrevReservoirs", mPathReservoirsBuffers[0].SizeBytes(), vk::BufferUsageFlagBits::eStorage|vk::BufferUsageFlagBits::eTransferDst);
			commandBuffer.Fill(mPrevReservoirs, 0);
		}

		if (mCountRays) {
			commandBuffer.Fill(mDebugCounters, 0);
			commandBuffer.ClearColor(mDebugHeatmap, vk::ClearColorValue{ std::array<uint32_t,4>{0,0,0,0} });
		}

		Defines defs;
		if (mAlphaTest)         defs.emplace("gAlphaTest", "true");
		if (mCountRays)         defs.emplace("gDebugCounters", "true");
		if (mShadingNormals)    defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)        defs.emplace("gNormalMaps", "true");
		if (mTalbotMisTemporal) defs.emplace("TALBOT_RMIS_TEMPORAL", "true");
		if (mTalbotMisSpatial)  defs.emplace("TALBOT_RMIS_SPATIAL", "true");
		if (mReconnection)      defs.emplace("RECONNECTION", "true");
		if (mSampleLights)      defs.emplace("SAMPLE_LIGHTS", "true");
		if (mDisneyBrdf)        defs.emplace("DISNEY_BRDF", "true");

		ShaderParameterBlock params;

        params.SetConstant("gHeatmapCounterType", mDebugHeatmapType);
        params.SetImage("gHeatmapCounters", mDebugHeatmap, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
        params.SetBuffer("gDebugCounters", mDebugCounters);
		params.SetParameters("gScene", scene.GetRenderData().mShaderParameters);

        int currentReservoirBuffer = 0;

        // sample canonical paths
        {
            params.SetImage("gRadiance" , renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
            params.SetImage("gAlbedo"   , mAlbedo     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
            params.SetImage("gPositions", mPositions  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
            params.SetConstant("gOutputExtent", extent);
            params.SetConstant("gMaxBounces", mMaxBounces);
            params.SetConstant("gCameraToWorld",           cameraToWorld);
            params.SetConstant("gCameraInverseProjection", inverse(projection));

            params.SetBuffer("gPathReservoirsIn" , mPathReservoirsBuffers[currentReservoirBuffer^1]);
            params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[currentReservoirBuffer]);

            commandBuffer.Dispatch(mSamplePaths, renderTarget.GetExtent(), params, defs);
        }

        params.SetConstant("gReuseX",  mReuseX);
        params.SetConstant("gMCap", mMCap);

        // temporal reuse
        if (mTemporalReuse && mHasHistory) {
            params.SetBuffer("gPrevReservoirs", mPrevReservoirs);
            params.SetImage("gPrevPositions", accum.GetPrevPositions(), vk::ImageLayout::eGeneral);
            params.SetConstant("gPrevWorldToClip", accum.GetPrevMVP());
            params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[currentReservoirBuffer]);
            params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[currentReservoirBuffer^1]);
            params.SetConstant("gPrevCameraPos", accum.GetPrevCameraPosition());
            commandBuffer.Dispatch(mTemporalReuse, renderTarget.GetExtent(), params, defs);
            currentReservoirBuffer ^= 1;
        }

        // spatial reuse
        if (mSpatialReusePasses > 0) {
            cmd.SetConstant("gSpatialReuseSamples", mSpatialReuseSamples);
            cmd.SetConstant("gSpatialReuseRadius",  mSpatialReuseRadius);
            for (int i = 0; i < mSpatialReusePasses; i++) {
                params.SetBuffer("gPathReservoirsIn",  mPathReservoirsBuffers[currentReservoirBuffer]);
                params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[currentReservoirBuffer^1]);
                currentReservoirBuffer ^= 1;
                params.SetConstant("gSpatialReuseIteration", i);
                commandBuffer.Dispatch("SpatialReuse", renderTarget.GetExtent(), params, defs);
            }
        }

        // output radiance from final reservoir
        {
            int otuputRadianceKernel = _CopyReservoirsShader.FindKernel("OutputRadiance");
            params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[currentReservoirBuffer]);
            commandBuffer.Dispatch(mOutputRadiance, renderTarget.GetExtent(), params);
        }

        // copy final reservoirs for future reuse
        if (mTemporalReuse) {
			commandBuffer.Copy(mPathReservoirsBuffers[currentReservoirBuffer], mPrevReservoirs);
        }
    }
};

}
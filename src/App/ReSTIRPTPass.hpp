#pragma once

#include <Common/Enums.h>
#include <Core/PipelineCache.hpp>
#include <Scene/Scene.hpp>

namespace ptvk {

class ReSTIRPTPass {
private:
	ComputePipelineCache mSamplePathsPipeline;
	ComputePipelineCache mTemporalReusePipeline;
	ComputePipelineCache mSpatialReusePipeline;
	ComputePipelineCache mOutputRadiancePipeline;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mCountRays = false;
	bool mSampleLights = true;
	bool mDisneyBrdf = true;

	bool mReconnection = false;
	bool mTalbotMisTemporal = false;
	bool mTalbotMisSpatial = false;

	bool mTemporalReuse = false;
	uint32_t mSpatialReusePasses = 0;
	uint32_t mSpatialReuseSamples = 5;
	float mSpatialReuseRadius = 32;

	DebugCounterType mDebugHeatmapType = DebugCounterType::eNumDebugCounters;

	float mReuseX = 0;
	float mMCap = 4;

	uint32_t mAccumulationStart = 0;
	uint32_t mMaxBounces = 4;

	Image::View mPositions;
	Image::View mAlbedo;
	Buffer::View<uint32_t> mDebugCounters;
	Image::View mDebugHeatmap;
	std::array<Buffer::View<std::byte>, 2> mPathReservoirsBuffers;
	Buffer::View<std::byte> mPrevReservoirs;
	std::unique_ptr<vk::raii::Event> mPrevFrameDoneEvent;

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
		mSamplePathsPipeline    = ComputePipelineCache(shaderFile, "SampleCanonicalPaths", "sm_6_7", args, md);
		mTemporalReusePipeline  = ComputePipelineCache(shaderFile, "TemporalReuse"       , "sm_6_7", args, md);
		mSpatialReusePipeline   = ComputePipelineCache(shaderFile, "SpatialReuse"        , "sm_6_7", args, md);
		mOutputRadiancePipeline = ComputePipelineCache(shaderFile, "OutputRadiance"      , "sm_6_7", args, md);
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
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 0, 32);

		if (mCountRays)
			Gui::EnumDropdown<DebugCounterType>("Heatmap", mDebugHeatmapType, DebugCounterTypeStrings);

		ImGui::Checkbox("Temporal reuse", &mTemporalReuse);
		if (mTemporalReuse)
			ImGui::Checkbox("Talbot RMIS Temporal", &mTalbotMisTemporal);
		Gui::ScalarField<uint32_t>("Spatial Reuse Passes", &mSpatialReusePasses, 0, 32, .01f);
		if (mSpatialReusePasses > 0) {
			ImGui::Checkbox("Talbot RMIS Spatial", &mTalbotMisSpatial);
			Gui::ScalarField<uint32_t>("Spatial Reuse Samples", &mSpatialReuseSamples, 0, 32, .01f);
			Gui::ScalarField<float>("Spatial Reuse Radius", &mSpatialReuseRadius, 0, 1000);
		}
		if (mTemporalReuse || mSpatialReusePasses > 0) {
			ImGui::Checkbox("Reconnection", &mReconnection);
			Gui::ScalarField<float>("M Cap", &mMCap, 0, 32);
			Gui::ScalarField<float>("Screen partition X", &mReuseX, -1, 1, .01f);
		}
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const float4x4& cameraToWorld, const float4x4& projection, const Image::View& prevPositions, const float4x4& prevMVP, const float3& prevCameraPosition) {
		ProfilerScope p("ReSTIRPTPass::Render", &commandBuffer);

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
			mDebugCounters = std::make_shared<Buffer>(commandBuffer.mDevice, "gRayCount", ((uint32_t)DebugCounterType::eNumDebugCounters + 1)*sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
            for (int i = 0; i < 2; i++)
                mPathReservoirsBuffers[i] = std::make_shared<Buffer>(commandBuffer.mDevice, "gReservoirs" + std::to_string(i), vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y)*128, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc);
			mPrevReservoirs = std::make_shared<Buffer>(commandBuffer.mDevice, "gPrevReservoirs", mPathReservoirsBuffers[0].SizeBytes(), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
			commandBuffer.Fill(mPrevReservoirs, 0);
		}

		if (mCountRays) {
			commandBuffer.Fill(mDebugCounters, 0);
			commandBuffer.ClearColor(mDebugHeatmap, vk::ClearColorValue{ std::array<uint32_t,4>{0,0,0,0} });
		}

		if (mPrevFrameDoneEvent) {
			commandBuffer->waitEvents(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {
				vk::BufferMemoryBarrier{
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**mPrevReservoirs.GetBuffer(), mPrevReservoirs.Offset(), mPrevReservoirs.SizeBytes()
				} }, {});
			mPrevReservoirs.SetState(vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
		} else
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{});

		Defines defs;
		if (mAlphaTest)      defs.emplace("gAlphaTest", "true");
		if (mCountRays)      defs.emplace("gEnableDebugCounters", "true");
		if (mShadingNormals) defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)     defs.emplace("gNormalMaps", "true");
		if (mSampleLights)   defs.emplace("SAMPLE_LIGHTS", "true");
		if (mDisneyBrdf)     defs.emplace("DISNEY_BRDF", "true");
		if (mTalbotMisTemporal) defs.emplace("TALBOT_RMIS_TEMPORAL", "true");
		if (mTalbotMisSpatial)  defs.emplace("TALBOT_RMIS_SPATIAL", "true");
		if (mReconnection)      defs.emplace("RECONNECTION", "true");

		ShaderParameterBlock params;
		params.SetImage("gRadiance" , renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
		params.SetImage("gPositions", mPositions  , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
		params.SetImage("gAlbedo"   , mAlbedo     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
		params.SetImage("gPrevPositions", prevPositions, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
		params.SetBuffer("gPrevReservoirs", mPrevReservoirs);
		params.SetConstant("gOutputSize", extent);
		params.SetConstant("gCameraToWorld", cameraToWorld);
		params.SetConstant("gInverseProjection", inverse(projection));
		params.SetParameters("gScene", scene.GetRenderData().mShaderParameters);
		params.SetBuffer("gDebugCounters", mDebugCounters);
		params.SetImage("gHeatmapCounters", mDebugHeatmap, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
		params.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart));
		params.SetConstant("gMaxBounces", mMaxBounces);
		params.SetConstant("gMCap", mMCap);
		params.SetConstant("gReuseX", mReuseX);
		params.SetConstant("gPrevMVP", prevMVP);
		params.SetConstant("gPrevCameraPos", prevCameraPosition);
		params.SetConstant("gHeatmapCounterType", (uint32_t)mDebugHeatmapType);
		params.SetConstant("gSpatialReuseSamples", mSpatialReuseSamples);
		params.SetConstant("gSpatialReuseRadius", mSpatialReuseRadius);

		int i = 0;

		params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
		params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
		mSamplePathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
		i ^= 1;

		if (mTemporalReuse) {
			params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
			params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
			mTemporalReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
			i ^= 1;
		}

		for (int j = 0; j < mSpatialReusePasses; j++) {
			params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
			params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
			mSpatialReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
			i ^= 1;
		}

		params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
		params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
		mOutputRadiancePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);

		commandBuffer.Copy(mPathReservoirsBuffers[i], mPrevReservoirs);
		commandBuffer->setEvent(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer);
	}
};

}
#pragma once

#include "VisibilityPass.hpp"

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
	bool mSampleLights = true;
	bool mDisneyBrdf = false;

	bool mReconnection = false;
	bool mTalbotMisTemporal = true;
	bool mTalbotMisSpatial = false;

	bool mTemporalReuse = false;
	uint32_t mSpatialReusePasses = 0;
	uint32_t mSpatialReuseSamples = 3;
	float mSpatialReuseRadius = 32;

	float mReuseX = 0;
	float mMCap = 20;

	uint32_t mAccumulationStart = 0;
	uint32_t mMaxBounces = 4;

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

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path") + "/Kernels/ReSTIR.slang";
		mSamplePathsPipeline    = ComputePipelineCache(shaderFile, "SampleCanonicalPaths", "sm_6_7", args, md);
		mTemporalReusePipeline  = ComputePipelineCache(shaderFile, "TemporalReuse"       , "sm_6_7", args, md);
		mSpatialReusePipeline   = ComputePipelineCache(shaderFile, "SpatialReuse"        , "sm_6_7", args, md);
		mOutputRadiancePipeline = ComputePipelineCache(*device.mInstance.GetOption("shader-kernel-path") + "/PathReservoir.slang", "OutputRadiance", "sm_6_7");
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Sample lights", &mSampleLights);
		ImGui::Checkbox("Disney brdf", &mDisneyBrdf);
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 0, 32);

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
			ImGui::SliderFloat("Screen partition X", &mReuseX, -1, 1);
		}
		ImGui::PopID();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope p("ReSTIRPTPass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);
		const vk::DeviceSize numReservoirs = vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y);

		if (!mPrevReservoirs || mPrevReservoirs.SizeBytes() != numReservoirs*128) {
            for (int i = 0; i < 2; i++)
                mPathReservoirsBuffers[i] = std::make_shared<Buffer>(commandBuffer.mDevice, "gReservoirs" + std::to_string(i), numReservoirs*128, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc);
			mPrevReservoirs = std::make_shared<Buffer>(commandBuffer.mDevice, "gPrevReservoirs", mPathReservoirsBuffers[0].SizeBytes(), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
			commandBuffer.Fill(mPrevReservoirs, 0);
		} else if (mPrevFrameDoneEvent) {
			commandBuffer->waitEvents(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {
				vk::BufferMemoryBarrier{
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**mPrevReservoirs.GetBuffer(), mPrevReservoirs.Offset(), mPrevReservoirs.SizeBytes()
				}
			}, {});
			mPrevReservoirs.SetState(vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
		}

		Defines defs;
		if (mAlphaTest)      defs.emplace("gAlphaTest", "true");
		if (mShadingNormals) defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)     defs.emplace("gNormalMaps", "true");
		if (mSampleLights)   defs.emplace("SAMPLE_LIGHTS", "true");
		if (mDisneyBrdf)     defs.emplace("DISNEY_BRDF", "true");
		if (mReconnection)   defs.emplace("RECONNECTION", "true");
		if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
			defs.emplace("gEnableDebugCounters", "true");

		ShaderParameterBlock params;
		params.SetImage("gVertices",     visibility.GetVertices()    , vk::ImageLayout::eGeneral);
		params.SetImage("gPrevVertices", visibility.GetPrevVertices(), vk::ImageLayout::eGeneral);
		params.SetImage("gDepthNormals", visibility.GetDepthNormals(), vk::ImageLayout::eGeneral);
		params.SetBuffer("gPrevReservoirs", mPrevReservoirs);
		params.SetConstant("gOutputSize", extent);
		params.SetConstant("gCameraPosition", visibility.GetCameraPosition());
		params.SetParameters("gScene", scene.GetRenderData().mShaderParameters);
		params.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart));
		params.SetConstant("gMaxBounces", mMaxBounces);
		params.SetConstant("gMCap", mMCap);
		params.SetConstant("gReuseX", mReuseX);
		params.SetConstant("gPrevMVP", visibility.GetPrevMVP());
		params.SetConstant("gPrevCameraPosition", visibility.GetPrevCameraPosition());
		params.SetConstant("gSpatialReuseSamples", mSpatialReuseSamples);
		params.SetConstant("gSpatialReuseRadius", mSpatialReuseRadius);
		params.SetParameters(visibility.GetDebugParameters());

		int i = 0;
		{
			ProfilerScope p("Sample Paths", &commandBuffer);
			params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
			params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
			mSamplePathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
			i ^= 1;
		}

		Defines tmpDefs = defs;
		if (mTemporalReuse) {
			ProfilerScope p("Temporal Reuse", &commandBuffer);
			if (mTalbotMisTemporal) tmpDefs.emplace("TALBOT_RMIS_TEMPORAL", "true");
			params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
			params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
			mTemporalReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, tmpDefs);
			i ^= 1;
		}

		if (mSpatialReusePasses > 0) {
			ProfilerScope p("Spatial Reuse", &commandBuffer);
			tmpDefs = defs;
			if (mTalbotMisSpatial) tmpDefs.emplace("TALBOT_RMIS_SPATIAL", "true");
			for (int j = 0; j < mSpatialReusePasses; j++) {
				params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
				params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
				mSpatialReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, tmpDefs);
				i ^= 1;
			}
		}

		{
			ProfilerScope p("Output Radiance", &commandBuffer);
			tmpDefs = { { "OUTPUT_RADIANCE_SHADER", "" } };
			if (mReconnection) tmpDefs.emplace("RECONNECTION", "true");
			mOutputRadiancePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), ShaderParameterBlock()
				.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
				.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i])
				.SetConstant("gOutputSize", extent)
			, tmpDefs);
		}

		commandBuffer.Copy(mPathReservoirsBuffers[i], mPrevReservoirs);

		if (!mPrevFrameDoneEvent)
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{});
		commandBuffer->setEvent(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer);
	}
};

}
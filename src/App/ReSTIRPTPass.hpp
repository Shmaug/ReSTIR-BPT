#pragma once

#include "VisibilityPass.hpp"
#include "HashGrid.hpp"

namespace ptvk {

class ReSTIRPTPass {
private:
	ComputePipelineCache mSamplePathsPipeline;
	ComputePipelineCache mTemporalReusePipeline;
	ComputePipelineCache mSpatialReusePipeline;
	ComputePipelineCache mTraceLightPathsPipeline;
	ComputePipelineCache mOutputRadiancePipeline;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mSampleLights = true;
	bool mDisneyBrdf = true;

	bool mBidirectional = false;
	float mLightSubpathCount = 0;
	bool mLightTraceOnly = false;

	bool mDebugPathLengths = false;
	uint32_t mDebugViewVertices = 2;
	uint32_t mDebugLightVertices = 2;

	bool mReconnection = true;
	float mReconnectionDistance = 0.01f;
	float mReconnectionRoughness = 0.1f;

	bool mTalbotMisTemporal = true;
	bool mTalbotMisSpatial = false;
	bool mPairwiseMisSpatial = false;

	bool mTemporalReuse = true;
	uint32_t mSpatialReusePasses = 1;
	uint32_t mSpatialReuseSamples = 3;
	float mSpatialReuseRadius = 32;

	float mReuseX = 0;
	float mMCap = 20;

	bool mUseHistoryDiscardMask = false;
	Image::View mHistoryDiscardMask;

	uint32_t mAccumulationStart = 0;
	uint32_t mMaxBounces = 4;

	Buffer::View<std::byte> mLightImage;
	std::array<Buffer::View<std::byte>,2> mLightVertices;
	std::array<Buffer::View<std::byte>,2> mLightVertexCount;

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
		mSamplePathsPipeline     = ComputePipelineCache(shaderFile, "SampleCanonicalPaths", "sm_6_7", args, md);
		mTemporalReusePipeline   = ComputePipelineCache(shaderFile, "TemporalReuse"       , "sm_6_7", args, md);
		mSpatialReusePipeline    = ComputePipelineCache(shaderFile, "SpatialReuse"        , "sm_6_7", args, md);
		mTraceLightPathsPipeline = ComputePipelineCache(shaderFile, "TraceLightPaths"     , "sm_6_7", args, md);
		mOutputRadiancePipeline  = ComputePipelineCache(shaderFile, "OutputRadiance"      , "sm_6_7", args, md);
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Sample lights", &mSampleLights);
		ImGui::Checkbox("Disney brdf", &mDisneyBrdf);
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 1, 32);

		ImGui::Checkbox("Reconnection", &mReconnection);
		if (mReconnection) {
			ImGui::Indent();
			Gui::ScalarField<float>("Distance threshold", &mReconnectionDistance);
			Gui::ScalarField<float>("Roughness threshold", &mReconnectionRoughness, 0, 1);
			ImGui::Unindent();
			ImGui::Separator();
		}

		ImGui::Checkbox("Bidirectional", &mBidirectional);
		if (mBidirectional) {
			ImGui::Indent();
			Gui::ScalarField<float>("Light paths", &mLightSubpathCount, 0, 2, 0);
			ImGui::Checkbox("Light trace only", &mLightTraceOnly);

			ImGui::Checkbox("Debug path lengths", &mDebugPathLengths);
			if (mDebugPathLengths) {
				ImGui::Indent();
				Gui::ScalarField<uint32_t>("View vertices", &mDebugViewVertices, 0, 32);
				Gui::ScalarField<uint32_t>("Light vertices", &mDebugLightVertices, 0, 32);
				ImGui::Unindent();
			}
			ImGui::Unindent();
			ImGui::Separator();
		}

		ImGui::Checkbox("Temporal reuse", &mTemporalReuse);
		if (mTemporalReuse) {
			ImGui::Indent();
			ImGui::Checkbox("Talbot RMIS", &mTalbotMisTemporal);
			ImGui::Checkbox("History rejection mask", &mUseHistoryDiscardMask);
			ImGui::Unindent();
			ImGui::Separator();
		}

		Gui::ScalarField<uint32_t>("Spatial Reuse Passes", &mSpatialReusePasses, 0, 32, .01f);
		if (mSpatialReusePasses > 0) {
			ImGui::Indent();
			Gui::ScalarField<uint32_t>("Samples", &mSpatialReuseSamples, 0, 32, .01f);
			Gui::ScalarField<float>("Radius", &mSpatialReuseRadius, 0, 1000);
			ImGui::Checkbox("Pairwise RMIS", &mPairwiseMisSpatial);
			if (!mPairwiseMisSpatial)
				ImGui::Checkbox("Talbot RMIS", &mTalbotMisSpatial);
			ImGui::Unindent();
		}

		if (mTemporalReuse || mSpatialReusePasses > 0) {
			ImGui::Separator();
			Gui::ScalarField<float>("M Cap", &mMCap, 0, 32);
			ImGui::SliderFloat("Screen partition X", &mReuseX, -1, 1);
		}

		ImGui::PopID();
	}

	inline Image::View GetDiscardMask() const { return mTemporalReuse && mUseHistoryDiscardMask ? mHistoryDiscardMask : Image::View{}; }

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope p("ReSTIRPTPass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);
		const vk::DeviceSize pixelCount = vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y);
		const vk::DeviceSize reservoirBufSize = 68*pixelCount;
		const uint32_t lightSubpathCount = mLightSubpathCount*extent.x*extent.y;

		if (!mPrevReservoirs || mPrevReservoirs.SizeBytes() != reservoirBufSize) {
			auto reservoirsBuf = std::make_shared<Buffer>(commandBuffer.mDevice, "gReservoirs", 3*reservoirBufSize, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst);
			mPathReservoirsBuffers[0] = Buffer::View<std::byte>(reservoirsBuf,                  0, reservoirBufSize);
			mPathReservoirsBuffers[1] = Buffer::View<std::byte>(reservoirsBuf,   reservoirBufSize, reservoirBufSize);
			mPrevReservoirs           = Buffer::View<std::byte>(reservoirsBuf, 2*reservoirBufSize, reservoirBufSize);
			commandBuffer.Fill(reservoirsBuf, 0);
			mLightImage         = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightImage", sizeof(uint4)*pixelCount, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
			mHistoryDiscardMask = std::make_shared<Image>(commandBuffer.mDevice, "gHistoryDiscardMask", ImageInfo{
				.mFormat = vk::Format::eR16Sfloat,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});
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

		if (!mLightVertices[0] || mLightVertices[0].SizeBytes() != 48*std::max(1u,lightSubpathCount)*mMaxBounces) {
			mLightVertices[0] = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightVertices", 48*std::max(1u,lightSubpathCount)*mMaxBounces, vk::BufferUsageFlagBits::eStorageBuffer);
			mLightVertices[1] = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightVertices", 48*std::max(1u,lightSubpathCount)*mMaxBounces, vk::BufferUsageFlagBits::eStorageBuffer);
			mLightVertexCount[0] = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightVertexCount0", sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
			mLightVertexCount[1] = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightVertexCount1", sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
			commandBuffer.Fill(mLightVertexCount[0], 0);
			commandBuffer.Fill(mLightVertexCount[1], 0);
		}

		Defines defs;
		if (mAlphaTest)       defs.emplace("gAlphaTest", "true");
		if (mShadingNormals)  defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)      defs.emplace("gNormalMaps", "true");
		if (mSampleLights && (!mBidirectional || mLightSubpathCount == 0)) defs.emplace("SAMPLE_LIGHTS", "true");
		if (mDisneyBrdf)      defs.emplace("DISNEY_BRDF", "true");
		if (mReconnection)    defs.emplace("RECONNECTION", "true");
		if (mBidirectional)   defs.emplace("BIDIRECTIONAL", "true");
		if (mBidirectional && mLightSubpathCount > 0) defs.emplace("gUseVC", "true");
		if (mBidirectional && mLightTraceOnly) defs.emplace("gLightTraceOnly", "true");
		if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
			defs.emplace("gEnableDebugCounters", "true");
		if (mDebugPathLengths) defs.emplace("gDebugPathLengths", "true");

		const ShaderParameterBlock& sceneParams = scene.GetRenderData().mShaderParameters;
		float3 sceneMin = float3(0);
		float3 sceneMax = float3(0);
		if (sceneParams.Contains("gSceneMin")) {
			sceneMin = sceneParams.GetConstant<float3>("gSceneMin");
			sceneMax = sceneParams.GetConstant<float3>("gSceneMax");
		}

		const uint j = commandBuffer.mDevice.GetFrameIndex() & 1;

		ShaderParameterBlock params;
		params.SetParameters("gScene", sceneParams);
		params.SetParameters(visibility.GetDebugParameters());
		params.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
		params.SetImage("gHistoryDiscardMask", mHistoryDiscardMask, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
		params.SetImage("gVertices",     visibility.GetVertices()    , vk::ImageLayout::eGeneral);
		params.SetImage("gPrevVertices", visibility.GetPrevVertices(), vk::ImageLayout::eGeneral);
		params.SetImage("gDepthNormals", visibility.GetDepthNormals(), vk::ImageLayout::eGeneral);
		params.SetBuffer("gLightImage", mLightImage);
		params.SetBuffer("gLightVertices", mLightVertices[j]);
		params.SetBuffer("gLightVertexCount", mLightVertexCount[j]);
		params.SetBuffer("gPrevLightVertices", mLightVertices[j^1]);
		params.SetBuffer("gPrevLightVertexCount", mLightVertexCount[j^1]);
		params.SetBuffer("gPrevReservoirs", mPrevReservoirs);
		params.SetConstant("gOutputSize", extent);
		params.SetConstant("gSceneSphere", float4(sceneMin+sceneMax, length(sceneMax-sceneMin))/2.f);
		params.SetConstant("gCameraForward", visibility.GetCameraForward());
		params.SetConstant("gCameraImagePlaneDist", (extent.y / (2 * std::tan(visibility.GetVerticalFov()/2))));
		params.SetConstant("gCameraPosition", visibility.GetCameraPosition());
		params.SetConstant("gRandomSeed", (uint32_t)(commandBuffer.mDevice.GetFrameIndex() - mAccumulationStart));
		params.SetConstant("gMaxBounces", mMaxBounces);
		params.SetConstant("gLightSubpathCount", lightSubpathCount);
		params.SetConstant("gMCap", mMCap);
		params.SetConstant("gReuseX", mReuseX);
		params.SetConstant("gPrevMVP", visibility.GetPrevMVP());
		params.SetConstant("gProjection", visibility.GetProjection());
		params.SetConstant("gWorldToCamera", inverse(visibility.GetCameraToWorld()));
		params.SetConstant("gPrevCameraPosition", visibility.GetPrevCameraPosition());
		params.SetConstant("gPrevCameraForward", visibility.GetPrevCameraForward());
		params.SetConstant("gSpatialReuseSamples", mSpatialReuseSamples);
		params.SetConstant("gSpatialReuseRadius", mSpatialReuseRadius);
		params.SetConstant("gSpatialReusePass", -1);
		params.SetConstant("gReconnectionDistance", mReconnectionDistance);
		params.SetConstant("gReconnectionRoughness", mReconnectionRoughness);
		params.SetConstant("gDebugViewVertices", mDebugViewVertices);
		params.SetConstant("gDebugLightVertices", mDebugLightVertices);

		auto drawSpinner = [](const char* shader) {
			const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
			ImGui::SetNextWindowPos(ImVec2(size.x/2, size.y/2));
			if (ImGui::Begin("Compiling shaders", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoInputs)) {
				ImGui::Text("Compiling shader: %s", shader);
				Gui::ProgressSpinner("Compiling shaders");
			}
			ImGui::End();
		};

		auto samplePathsPipeline = mSamplePathsPipeline.GetPipelineAsync(commandBuffer.mDevice, defs);

		Defines tmpDefs = defs;
		if (mUseHistoryDiscardMask) tmpDefs.emplace("gUseDiscardMask", "true");
		if (mTalbotMisTemporal) tmpDefs.emplace("TALBOT_RMIS_TEMPORAL", "true");
		auto temporalReusePipeline = mTemporalReusePipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

		tmpDefs = defs;
		if      (mPairwiseMisSpatial) tmpDefs.emplace("RMIS_PAIRWISE", "true");
		else if (mTalbotMisSpatial)   tmpDefs.emplace("TALBOT_RMIS_SPATIAL", "true");
		auto spatialReusePipeline = mSpatialReusePipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

		if (mTemporalReuse && mUseHistoryDiscardMask)
			commandBuffer.ClearColor(mHistoryDiscardMask, vk::ClearColorValue{ std::array<float,4>{0,0,0,0} });

		if (!samplePathsPipeline) {
			drawSpinner("SamplePaths");
			return;
		}

		params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[0]);
		params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[1]);

		if (mBidirectional && mLightSubpathCount > 0) {
			auto traceLightPathsPipeline = mTraceLightPathsPipeline.GetPipelineAsync(commandBuffer.mDevice, defs);
			commandBuffer.Fill(mLightVertexCount[j], 0);
			commandBuffer.Fill(mLightImage, 0);
			if (traceLightPathsPipeline) {
				ProfilerScope p("Trace Light Paths", &commandBuffer);
				mTraceLightPathsPipeline.Dispatch(commandBuffer, { extent.x, (lightSubpathCount + extent.x-1)/extent.x, 1}, params, *traceLightPathsPipeline);
			} else
				drawSpinner("TraceLightPaths");
		}

		int i = 0;
		{
			ProfilerScope p("Sample Paths", &commandBuffer);
			params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
			params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
			mSamplePathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *samplePathsPipeline);
			i ^= 1;
		}

		if (mTemporalReuse) {
			if (temporalReusePipeline) {
				ProfilerScope p("Temporal Reuse", &commandBuffer);
				params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
				params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
				if (mUseHistoryDiscardMask) params.SetImage("gHistoryDiscardMask", mHistoryDiscardMask, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
				mTemporalReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *temporalReusePipeline);
				if (mUseHistoryDiscardMask) params.SetImage("gHistoryDiscardMask", mHistoryDiscardMask, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
				i ^= 1;
			} else {
				drawSpinner("TemporalReuse");
			}
		}

		if (mSpatialReusePasses > 0) {
			if (spatialReusePipeline) {
				ProfilerScope p("Spatial Reuse", &commandBuffer);
				for (int j = 0; j < mSpatialReusePasses; j++) {
					params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
					params.SetBuffer("gPathReservoirsOut", mPathReservoirsBuffers[i^1]);
					params.SetConstant("gSpatialReusePass", j);
					mSpatialReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *spatialReusePipeline);
					i ^= 1;
				}
			} else
				drawSpinner("SpatialReuse");
		}

		{
			ProfilerScope p("Output Radiance", &commandBuffer);
			params.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
			params.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i]);
			mOutputRadiancePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
		}

		commandBuffer.Copy(mPathReservoirsBuffers[i], mPrevReservoirs);

		if (!mPrevFrameDoneEvent)
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{});
		commandBuffer->setEvent(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer);
	}
};

}
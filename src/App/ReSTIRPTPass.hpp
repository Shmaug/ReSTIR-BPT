#pragma once

#include "VisibilityPass.hpp"
#include "HashGrid.hpp"

namespace ptvk {

class ReSTIRPTPass {
private:
	ComputePipelineCache mSamplePathsPipeline;
	ComputePipelineCache mTemporalReusePipeline;
	ComputePipelineCache mSpatialReusePipeline;
	ComputePipelineCache mSampleLightPathsPipeline;
	ComputePipelineCache mOutputRadiancePipeline;
	ComputePipelineCache mConnectToCameraPipeline;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mNormalMaps = true;
	bool mCompressTangentFrame = true;
	bool mRussianRoullette = true;
	bool mSampleLights = true;
	bool mDisneyBrdf = true;

	bool mBidirectional = false;
	bool mVertexMerging = false;
	bool mVertexMergingOnly = false;
	float mLightSubpathCount = 0.25f;
	bool mLightTraceOnly = false;
	bool mNoLightTraceResampling = true;

	bool mDebugPathLengths = false;
	uint32_t mDebugTotalVertices = 4;
	uint32_t mDebugLightVertices = 2;

	float mReconnectionDistance = 0.01f;
	float mReconnectionRoughness = 0.1f;

	float mDirectLightProb = 0.5f;

	bool mTemporalReuse = true;
	float mTemporalReuseRadius = 0;
	bool mTalbotMisTemporal = true;

	uint32_t mSpatialReusePasses = 1;
	uint32_t mSpatialReuseSamples = 3;
	float mSpatialReuseRadius = 32;
	bool mTalbotMisSpatial = false;
	bool mPairwiseMisSpatial = false;

	float mMCap = 20;

	bool mClearReservoirs = false;

	bool mUseHistoryDiscardMask = false;
	Image::View mHistoryDiscardMask;

	bool mFixedSeed = false;
	uint32_t mRandomSeed = 0;
	uint32_t mMaxBounces = 4;

	bool mDebugPixel = false;
	float2 mDebugPixelId; // normalized to 0-1

	HashGrid mVisibleLightVertices;
	HashGrid mLightVertexGrid;
	Buffer::View<std::byte> mLightVertices;
	Buffer::View<std::byte> mLightVertexCount;

	std::array<Buffer::View<std::byte>, 2> mPathReservoirsBuffers;
	Buffer::View<std::byte> mPrevReservoirs;
	std::unique_ptr<vk::raii::Event> mPrevFrameDoneEvent;
	std::vector<vk::BufferMemoryBarrier2> mPrevFrameBarriers;

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
		mSamplePathsPipeline      = ComputePipelineCache(shaderFile, "SampleCameraPaths"       , "sm_6_7", args, md);
		mSampleLightPathsPipeline = ComputePipelineCache(shaderFile, "SampleLightPaths"        , "sm_6_7", args, md);
		mTemporalReusePipeline    = ComputePipelineCache(shaderFile, "TemporalReuse"           , "sm_6_7", args, md);
		mSpatialReusePipeline     = ComputePipelineCache(shaderFile, "SpatialReuse"            , "sm_6_7", args, md);
		mOutputRadiancePipeline   = ComputePipelineCache(shaderFile, "OutputRadiance"          , "sm_6_7", args, md);
		mConnectToCameraPipeline  = ComputePipelineCache(shaderFile, "ProcessCameraConnections", "sm_6_7", args, md);

		mVisibleLightVertices = HashGrid(device.mInstance);
		mVisibleLightVertices.mElementSize = 4;

		mLightVertexGrid = HashGrid(device.mInstance);
		mLightVertexGrid.mElementSize = 4;
		mLightVertexGrid.mCellCount = 100000;
		mLightVertexGrid.mCellSize = 0.02f;
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		if (ImGui::Checkbox("Alpha test", &mAlphaTest)) mClearReservoirs = true;
		if (ImGui::Checkbox("Shading normals", &mShadingNormals)) mClearReservoirs = true;
		if (ImGui::Checkbox("Normal maps", &mNormalMaps)) mClearReservoirs = true;
		if (ImGui::Checkbox("Russian roullette", &mRussianRoullette)) mClearReservoirs = true;
		if (ImGui::Checkbox("Sample lights", &mSampleLights)) mClearReservoirs = true;
		if (ImGui::Checkbox("Compress tangent frame", &mCompressTangentFrame)) mClearReservoirs = true;
		if (ImGui::Checkbox("Disney brdf", &mDisneyBrdf)) mClearReservoirs = true;
		if (Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 1, 32)) mClearReservoirs = true;

		if (ImGui::Checkbox("Fix seed", &mFixedSeed))
			mRandomSeed = 0;
		if (mFixedSeed) {
			ImGui::SameLine();
			if (Gui::ScalarField<uint32_t>("##", &mRandomSeed)) mClearReservoirs = true;
		}

		Gui::ScalarField<float>("Min reconnection distance", &mReconnectionDistance, 0, 0, .01f);
		Gui::ScalarField<float>("Min reconnection roughness", &mReconnectionRoughness, 0, 1, .01f);

		if (ImGui::Checkbox("Bidirectional", &mBidirectional)) mClearReservoirs = true;
		if (mBidirectional) {
			ImGui::Indent();
			Gui::ScalarField<float>("Light paths", &mLightSubpathCount, 0, 2, 0);
			Gui::ScalarField<float>("Direct light probability", &mDirectLightProb, 0, 1, 0);
			ImGui::Checkbox("Vertex merging", &mVertexMerging);
			ImGui::Checkbox("Light trace only", &mLightTraceOnly);
			ImGui::Checkbox("No Light trace resampling", &mNoLightTraceResampling);

			if (mVertexMerging) {
				if (ImGui::Checkbox("Vertex merging only", &mVertexMergingOnly)) mClearReservoirs = true;
				if (Gui::ScalarField<uint32_t>("Grid cell count", &mLightVertexGrid.mCellCount, 1000, 0xFFFFFF)) mClearReservoirs = true;
				if (Gui::ScalarField<float>("Merge diameter", &mLightVertexGrid.mCellSize, 0.001f, 100, 0.01f)) mClearReservoirs = true;
			}

			ImGui::Checkbox("Debug path lengths", &mDebugPathLengths);
			if (mDebugPathLengths) {
				ImGui::Indent();
				Gui::ScalarField<uint32_t>("Total vertices", &mDebugTotalVertices, 0, 32);
				Gui::ScalarField<uint32_t>("Light vertices", &mDebugLightVertices, 0, 32);
				ImGui::Unindent();
			}
			ImGui::Unindent();
			ImGui::Separator();
		}

		if (ImGui::Checkbox("Temporal reuse", &mTemporalReuse)) mClearReservoirs = true;
		if (mTemporalReuse) {
			ImGui::Indent();
			ImGui::PushID("Temporal");
			Gui::ScalarField<float>("Radius", &mTemporalReuseRadius, 0, 1000);
			ImGui::Checkbox("Talbot RMIS", &mTalbotMisTemporal);
			ImGui::Checkbox("History rejection mask", &mUseHistoryDiscardMask);
			ImGui::PopID();
			ImGui::Unindent();
			ImGui::Separator();
		}

		Gui::ScalarField<uint32_t>("Spatial Reuse Passes", &mSpatialReusePasses, 0, 32, .01f);
		if (mSpatialReusePasses > 0) {
			ImGui::Indent();
			ImGui::PushID("Spatial");
			Gui::ScalarField<uint32_t>("Samples", &mSpatialReuseSamples, 0, 32, .01f);
			Gui::ScalarField<float>("Radius", &mSpatialReuseRadius, 0, 1000);
			ImGui::Checkbox("Talbot RMIS", &mTalbotMisSpatial);
			ImGui::Checkbox("Pairwise RMIS", &mPairwiseMisSpatial);
			ImGui::PopID();
			ImGui::Unindent();
		}

		if (mTemporalReuse || mSpatialReusePasses > 0) {
			ImGui::Separator();
			Gui::ScalarField<float>("M Cap", &mMCap, 0, 32);
		}

		ImGui::Separator();
		ImGui::Checkbox("Debug pixel", &mDebugPixel);
		if (mDebugPixel) {
			const ImGuiIO& io = ImGui::GetIO();
			if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
				const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
				mDebugPixelId = float2((uint32_t)io.MousePos.x, (uint32_t)io.MousePos.y) / float2(size.x, size.y);
			}
		}

		ImGui::PopID();
	}

	inline Image::View GetDiscardMask() const { return mTemporalReuse && mUseHistoryDiscardMask ? mHistoryDiscardMask : Image::View{}; }

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope p("ReSTIRPTPass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);
		const vk::DeviceSize pixelCount = vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y);
		const vk::DeviceSize reservoirBufSize = 88*pixelCount;
		const uint32_t lightSubpathCount = mLightSubpathCount*extent.x*extent.y;

		if (!mPrevReservoirs || mPrevReservoirs.SizeBytes() != reservoirBufSize) {
			auto reservoirsBuf           = std::make_shared<Buffer>(commandBuffer.mDevice, "gReservoirs", 3*reservoirBufSize, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst);
			mPathReservoirsBuffers[0] = Buffer::View<std::byte>(reservoirsBuf, 0*reservoirBufSize, reservoirBufSize);
			mPathReservoirsBuffers[1] = Buffer::View<std::byte>(reservoirsBuf, 1*reservoirBufSize, reservoirBufSize);
			mPrevReservoirs           = Buffer::View<std::byte>(reservoirsBuf, 2*reservoirBufSize, reservoirBufSize);
			mClearReservoirs = true;
			mHistoryDiscardMask = std::make_shared<Image>(commandBuffer.mDevice, "gHistoryDiscardMask", ImageInfo{
				.mFormat = vk::Format::eR16Sfloat,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});
		} else if (mPrevFrameDoneEvent && !mPrevFrameBarriers.empty()) {
			commandBuffer->waitEvents2(**mPrevFrameDoneEvent, vk::DependencyInfo{ {}, {},  mPrevFrameBarriers, {} });
			mPrevReservoirs.SetState(vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
		}

		if (mClearReservoirs) {
			commandBuffer.Fill(mPathReservoirsBuffers[0].GetBuffer(), 0);
			mClearReservoirs = false;
			if (!mFixedSeed) mRandomSeed = 0;
		}
		if (mTemporalReuse && mUseHistoryDiscardMask)
			commandBuffer.ClearColor(mHistoryDiscardMask, vk::ClearColorValue{ std::array<float,4>{0,0,0,0} });

		if (!mLightVertices || mLightVertices.SizeBytes() != 48*std::max(1u,lightSubpathCount*mMaxBounces)) {
			mLightVertices    = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightVertices", 48*std::max(1u,lightSubpathCount*mMaxBounces), vk::BufferUsageFlagBits::eStorageBuffer);
			mLightVertexCount = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightVertexCount", sizeof(uint), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
			commandBuffer.Fill(mLightVertexCount, 0);
			mPrevFrameBarriers.clear();
		}

		if (mBidirectional) {
			mVisibleLightVertices.mSize = std::max(1u,lightSubpathCount*mMaxBounces);
			mVisibleLightVertices.mCellCount = pixelCount + 1;
			mVisibleLightVertices.Prepare(commandBuffer, visibility.GetCameraPosition(), visibility.GetVerticalFov(), extent);

			if (mVertexMerging) {
				mLightVertexGrid.mSize = std::max(1u,lightSubpathCount*mMaxBounces);
				mLightVertexGrid.Prepare(commandBuffer, visibility.GetCameraPosition(), visibility.GetVerticalFov(), extent);
			}
		}

		// assign parameters and defines
		Defines defs;
		ShaderParameterBlock params;
		{
			if (mAlphaTest)                                             defs.emplace("gAlphaTest", "true");
			if (mShadingNormals)                                        defs.emplace("gShadingNormals", "true");
			if (mNormalMaps)                                            defs.emplace("gNormalMaps", "true");
			if (mCompressTangentFrame)                                  defs.emplace("COMPRESS_TANGENT_FRAME", "true");
			if (!mRussianRoullette)                                     defs.emplace("DISABLE_STOCHASTIC_TERMINATION", "true");
			if (mSampleLights || mBidirectional)                        defs.emplace("SAMPLE_LIGHTS", "true");
			if (mDisneyBrdf)                                            defs.emplace("DISNEY_BRDF", "true");
			if (mBidirectional)                                         defs.emplace("BIDIRECTIONAL", "true");
			if (mBidirectional && mVertexMerging)                       defs.emplace("VERTEX_MERGING", "true");
			if (mBidirectional && mVertexMerging && mVertexMergingOnly) defs.emplace("VERTEX_MERGING_ONLY", "true");
			if (mBidirectional && mLightTraceOnly)                      defs.emplace("gLightTraceOnly", "true");
			if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
				defs.emplace("gEnableDebugCounters", "true");
			if (mDebugPathLengths) defs.emplace("gDebugPathLengths", "true");
			if (mDebugPixel) defs.emplace("DEBUG_PIXEL", "true");

			const ShaderParameterBlock& sceneParams = scene.GetRenderData().mShaderParameters;
			float3 sceneMin = float3(0);
			float3 sceneMax = float3(0);
			if (sceneParams.Contains("gSceneMin")) {
				sceneMin = sceneParams.GetConstant<float3>("gSceneMin");
				sceneMax = sceneParams.GetConstant<float3>("gSceneMax");
			}

			params.SetParameters("gScene", sceneParams);
			params.SetParameters("gVisibleLightVertices", mVisibleLightVertices.mParameters);
			params.SetParameters("gLightVertexGrid", mLightVertexGrid.mParameters);
			params.SetParameters(visibility.GetDebugParameters());
			params.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
			params.SetImage("gHistoryDiscardMask", mHistoryDiscardMask, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
			params.SetImage("gVertices",     visibility.GetVertices()    , vk::ImageLayout::eGeneral);
			params.SetImage("gPrevVertices", visibility.GetPrevVertices(), vk::ImageLayout::eGeneral);
			params.SetBuffer("gLightVertices", mLightVertices);
			params.SetBuffer("gLightVertexCount", mLightVertexCount);
			params.SetBuffer("gPrevReservoirs", mPrevReservoirs);
			params.SetBuffer("gPathReservoirs", 0, mPathReservoirsBuffers[0]);
			params.SetBuffer("gPathReservoirs", 1, mPathReservoirsBuffers[1]);
			params.SetConstant("gOutputSize", extent);
			params.SetConstant("gCameraImagePlaneDist", (extent.y / (2 * std::tan(visibility.GetVerticalFov()/2))));
			params.SetConstant("gCameraPosition", visibility.GetCameraPosition());
			params.SetConstant("gRandomSeed", mRandomSeed);
			params.SetConstant("gMaxBounces", mMaxBounces);
			params.SetConstant("gLightSubpathCount", lightSubpathCount);
			params.SetConstant("gMCap", mMCap);
			params.SetConstant("gPrevMVP", visibility.GetPrevMVP());
			params.SetConstant("gProjection", visibility.GetProjection());
			params.SetConstant("gWorldToCamera", inverse(visibility.GetCameraToWorld()));
			params.SetConstant("gPrevCameraPosition", visibility.GetPrevCameraPosition());
			params.SetConstant("gSpatialReuseSamples", mSpatialReuseSamples);
			params.SetConstant("gSpatialReuseRadius", mSpatialReuseRadius);
			params.SetConstant("gTemporalReuseRadius", mTemporalReuseRadius);
			params.SetConstant("gSpatialReusePass", -1);
			params.SetConstant("gReconnectionDistance", mReconnectionDistance);
			params.SetConstant("gReconnectionRoughness", mReconnectionRoughness);
			params.SetConstant("gDirectLightProb", mDirectLightProb);
			params.SetConstant("gDebugTotalVertices", mDebugTotalVertices);
			params.SetConstant("gDebugLightVertices", mDebugLightVertices);
			params.SetConstant("gDebugPixel", int32_t(mDebugPixelId.y * extent.y) * extent.x + int32_t(mDebugPixelId.x * extent.x));
		}

		if (!mFixedSeed) mRandomSeed++;

		// get pipelines

		auto drawSpinner = [](const char* shader) {
			const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
			ImGui::SetNextWindowPos(ImVec2(size.x/2, size.y/2));
			if (ImGui::Begin("Compiling shaders", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoInputs)) {
				ImGui::Text("%s", shader);
				Gui::ProgressSpinner("Compiling shaders", 15, 6, false);
			}
			ImGui::End();
		};

		auto samplePathsPipeline = mSamplePathsPipeline.GetPipelineAsync(commandBuffer.mDevice, defs);

		Defines tmpDefs = defs;
		if (mUseHistoryDiscardMask) tmpDefs.emplace("gUseDiscardMask", "true");
		if (mTalbotMisTemporal) tmpDefs.emplace("TALBOT_RMIS_TEMPORAL", "true");
		if (mTemporalReuseRadius > 0) tmpDefs.emplace("gCombinedSpatialTemporalReuse", "true");
		if (mBidirectional && mNoLightTraceResampling) tmpDefs.emplace("gNoLightTraceResampling", "true");
		auto temporalReusePipeline = mTemporalReusePipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

		tmpDefs = defs;
		if (mPairwiseMisSpatial)    tmpDefs.emplace("PAIRWISE_RMIS_SPATIAL", "true");
		else if (mTalbotMisSpatial) tmpDefs.emplace("TALBOT_RMIS_SPATIAL", "true");
		if (mBidirectional && mNoLightTraceResampling) tmpDefs.emplace("gNoLightTraceResampling", "true");
		auto spatialReusePipeline = mSpatialReusePipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

		// ---------------------------------------------------------------------------------------------------------

		int reservoirIndex = 0;
		params.SetConstant("gReservoirIndex", reservoirIndex);

		// light subpaths
		std::shared_ptr<ComputePipeline> traceLightPathsPipeline, connectToCameraPipeline;
		if (mBidirectional && mLightSubpathCount > 0) {
			tmpDefs = defs;
			tmpDefs.emplace("PROCESS_LIGHT_VERTICES", "true");
			traceLightPathsPipeline = mSampleLightPathsPipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

			if (mBidirectional && mNoLightTraceResampling) tmpDefs.emplace("gNoLightTraceResampling", "true");
			connectToCameraPipeline = mConnectToCameraPipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

			commandBuffer.Fill(mLightVertexCount, 0);

			if (traceLightPathsPipeline) {
				{
					ProfilerScope p("Trace Light Paths", &commandBuffer);
					mSampleLightPathsPipeline.Dispatch(commandBuffer, { extent.x, (lightSubpathCount + extent.x-1)/extent.x, 1}, params, *traceLightPathsPipeline);
				}
				mVisibleLightVertices.Build(commandBuffer);

				if (mVertexMerging)
					mLightVertexGrid.Build(commandBuffer);
			} else
				drawSpinner("TraceLightPaths");
		}

		if (!samplePathsPipeline) {
			drawSpinner("SamplePaths");
			return;
		}

		// camera subpaths
		if (!(mBidirectional && mLightTraceOnly)) {
			ProfilerScope p("Sample Paths", &commandBuffer);
			params.SetConstant("gReservoirIndex", reservoirIndex);
			mSamplePathsPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *samplePathsPipeline);
			reservoirIndex ^= 1;
		}

		// connect light subpaths to the camera
		if (traceLightPathsPipeline) {
			if (connectToCameraPipeline) {
				ProfilerScope p("Process camera connections", &commandBuffer);
				if (mNoLightTraceResampling)
					params.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
				params.SetConstant("gReservoirIndex", reservoirIndex);
				mConnectToCameraPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *connectToCameraPipeline);
				if (mNoLightTraceResampling)
					params.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);

				reservoirIndex ^= 1;
			} else
				drawSpinner("ProcessCameraConnections");
		}

		if (mTemporalReuse) {
			if (temporalReusePipeline) {
				ProfilerScope p("Temporal Reuse", &commandBuffer);
				params.SetConstant("gReservoirIndex", reservoirIndex);
				if (mUseHistoryDiscardMask) params.SetImage("gHistoryDiscardMask", mHistoryDiscardMask, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite);
				mTemporalReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *temporalReusePipeline);
				if (mUseHistoryDiscardMask) params.SetImage("gHistoryDiscardMask", mHistoryDiscardMask, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
				reservoirIndex ^= 1;
			} else {
				drawSpinner("TemporalReuse");
			}
		}

		if (mSpatialReusePasses > 0) {
			if (spatialReusePipeline) {
				ProfilerScope p("Spatial Reuse", &commandBuffer);
				for (int j = 0; j < mSpatialReusePasses; j++) {
					params.SetConstant("gReservoirIndex", reservoirIndex);
					params.SetConstant("gSpatialReusePass", j);
					mSpatialReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *spatialReusePipeline);
					reservoirIndex ^= 1;
				}
			} else
				drawSpinner("SpatialReuse");
		}

		// copy reservoir sample to the output image
		{
			ProfilerScope p("Output Radiance", &commandBuffer);
			params.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
			params.SetConstant("gReservoirIndex", reservoirIndex);
			mOutputRadiancePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, defs);
		}

		// copy reservoirs for future reuse
		if (mTemporalReuse) {
			commandBuffer.Copy(mPathReservoirsBuffers[reservoirIndex], mPrevReservoirs);
			mPrevFrameBarriers = {
				vk::BufferMemoryBarrier2{
					vk::PipelineStageFlagBits2::eTransfer,      vk::AccessFlagBits2::eTransferWrite,
					vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**mPrevReservoirs.GetBuffer(), mPrevReservoirs.Offset(), mPrevReservoirs.SizeBytes()
				} };
		} else
			mPrevFrameBarriers.clear();

		if (!mPrevFrameDoneEvent)
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{ vk::EventCreateFlagBits::eDeviceOnly });
		commandBuffer->setEvent2(**mPrevFrameDoneEvent, vk::DependencyInfo{ {}, {},  mPrevFrameBarriers, {} });
	}
};

}
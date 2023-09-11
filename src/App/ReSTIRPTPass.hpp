#pragma once

#include "VisibilityPass.hpp"
#include "HashGrid.hpp"

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

	bool mStructuredBuffers = false;

	bool mReconnection = true;
	float mReconnectionDistance = 0.01f;
	float mReconnectionRoughness = 0.1f;

	bool mTalbotMisTemporal = true;
	bool mTalbotMisSpatial = true;
	bool mPairwiseMisSpatial = false;

	bool mTemporalReuse = true;
	uint32_t mSpatialReusePasses = 1;
	uint32_t mSpatialReuseSamples = 3;
	float mSpatialReuseRadius = 32;

	float mReuseX = 0;
	float mMCap = 20;

	uint32_t mAccumulationStart = 0;
	uint32_t mMaxBounces = 4;

	bool mWorldSpaceReuse = false;
	std::array<HashGrid,2> mReservoirHashGrids;
	uint32_t mCurHashGrid = 0;

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

		mReservoirHashGrids[0] = HashGrid(device.mInstance);
		mReservoirHashGrids[1] = HashGrid(device.mInstance);
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Normal maps", &mNormalMaps);
		ImGui::Checkbox("Sample lights", &mSampleLights);
		ImGui::Checkbox("Disney brdf", &mDisneyBrdf);
		Gui::ScalarField<uint32_t>("Max bounces", &mMaxBounces, 0, 32);

		ImGui::Separator();

		ImGui::Checkbox("Use structured buffers", &mStructuredBuffers);

		ImGui::Checkbox("Reconnection", &mReconnection);
		if (mReconnection) {
			ImGui::Indent();
			Gui::ScalarField<float>("Distance threshold", &mReconnectionDistance);
			Gui::ScalarField<float>("Roughness threshold", &mReconnectionRoughness, 0, 1);
			ImGui::Unindent();
		}
		Gui::ScalarField<float>("M Cap", &mMCap, 0, 32);
		ImGui::SliderFloat("Screen partition X", &mReuseX, -1, 1);

		ImGui::Checkbox("World Space Reuse", &mWorldSpaceReuse);
		if (mWorldSpaceReuse) {
			ImGui::Indent();
			Gui::ScalarField<uint32_t>("Cell count", &mReservoirHashGrids[0].mCellCount);
			Gui::ScalarField<float>("Min cell size", &mReservoirHashGrids[0].mCellSize);
			Gui::ScalarField<float>("Cell pixel radius", &mReservoirHashGrids[0].mCellPixelRadius);
			mReservoirHashGrids[1].mCellCount       = mReservoirHashGrids[0].mCellCount;
			mReservoirHashGrids[1].mCellSize        = mReservoirHashGrids[0].mCellSize;
			mReservoirHashGrids[1].mCellPixelRadius = mReservoirHashGrids[0].mCellPixelRadius;
			ImGui::Unindent();
		}

		ImGui::Checkbox("Temporal reuse", &mTemporalReuse);
		if (mTemporalReuse) {
			ImGui::Indent();
			ImGui::Checkbox("Talbot RMIS Temporal", &mTalbotMisTemporal);
			ImGui::Unindent();
			ImGui::Separator();
		}

		Gui::ScalarField<uint32_t>("Spatial Reuse Passes", &mSpatialReusePasses, 0, 32, .01f);
		if (mSpatialReusePasses > 0) {
			ImGui::Indent();
			ImGui::Checkbox("Pairwise RMIS Spatial", &mPairwiseMisSpatial);
			if (!mPairwiseMisSpatial)
				ImGui::Checkbox("Talbot RMIS Spatial", &mTalbotMisSpatial);
			Gui::ScalarField<uint32_t>("Spatial Reuse Samples", &mSpatialReuseSamples, 0, 32, .01f);
			Gui::ScalarField<float>("Spatial Reuse Radius", &mSpatialReuseRadius, 0, 1000);
			ImGui::Unindent();
		}
		ImGui::PopID();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope p("ReSTIRPTPass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);
		const vk::DeviceSize numReservoirs = vk::DeviceSize(extent.x)*vk::DeviceSize(extent.y);
		const vk::DeviceSize reservoirSize = 128;

		if (!mPrevReservoirs || mPrevReservoirs.SizeBytes() != numReservoirs*reservoirSize) {
            for (int i = 0; i < 2; i++)
                mPathReservoirsBuffers[i] = std::make_shared<Buffer>(commandBuffer.mDevice, "gReservoirs" + std::to_string(i), numReservoirs*reservoirSize, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc);
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
		if (mAlphaTest)       defs.emplace("gAlphaTest", "true");
		if (mShadingNormals)  defs.emplace("gShadingNormals", "true");
		if (mNormalMaps)      defs.emplace("gNormalMaps", "true");
		if (mSampleLights)    defs.emplace("SAMPLE_LIGHTS", "true");
		if (mDisneyBrdf)      defs.emplace("DISNEY_BRDF", "true");
		if (mReconnection)    defs.emplace("RECONNECTION", "true");
		if (mWorldSpaceReuse) defs.emplace("gWorldSpaceReuse", "true");
		if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
			defs.emplace("gEnableDebugCounters", "true");

		if (mStructuredBuffers)
			defs.emplace("RESERVOIR_STRUCTURED_BUFFERS", "true");

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
		params.SetConstant("gSpatialReusePass", -1);
		params.SetConstant("gReconnectionDistance", mReconnectionDistance);
		params.SetConstant("gReconnectionRoughness", mReconnectionRoughness);
		params.SetParameters(visibility.GetDebugParameters());

		if (mWorldSpaceReuse && mTemporalReuse && mSpatialReusePasses > 0) {
			mReservoirHashGrids[0].mSize = numReservoirs;
			mReservoirHashGrids[1].mSize = numReservoirs;
			mReservoirHashGrids[0].mElementSize = reservoirSize;
			mReservoirHashGrids[1].mElementSize = reservoirSize;
			mReservoirHashGrids[mCurHashGrid].Prepare(commandBuffer, visibility.GetCameraPosition(), visibility.GetVerticalFov(), extent);
			params.SetParameters("gReservoirHashGrid", mReservoirHashGrids[mCurHashGrid].mParameters);
			params.SetParameters("gPrevReservoirHashGrid", mReservoirHashGrids[mCurHashGrid^1].mParameters);
		}

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
		if (mTalbotMisTemporal) tmpDefs.emplace("TALBOT_RMIS_TEMPORAL", "true");
		auto temporalReusePipeline = mTemporalReusePipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

		tmpDefs = defs;
		if (mPairwiseMisSpatial)    tmpDefs.emplace("RMIS_PAIRWISE", "true");
		else if (mTalbotMisSpatial) tmpDefs.emplace("TALBOT_RMIS_SPATIAL", "true");
		auto spatialReusePipeline = mSpatialReusePipeline.GetPipelineAsync(commandBuffer.mDevice, tmpDefs);

		if (!samplePathsPipeline) {
			drawSpinner("SamplePaths");
			return;
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
				mTemporalReusePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), params, *temporalReusePipeline);
				i ^= 1;
			} else
				drawSpinner("TemporalReuse");
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
			Defines tmpDefs = { { "OUTPUT_RADIANCE_SHADER", "" } };
			if (mReconnection) tmpDefs.emplace("RECONNECTION", "true");
			if (mStructuredBuffers) tmpDefs.emplace("RESERVOIR_STRUCTURED_BUFFERS", "true");
			mOutputRadiancePipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), ShaderParameterBlock()
				.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
				.SetBuffer("gPathReservoirsIn", mPathReservoirsBuffers[i])
				.SetConstant("gOutputSize", extent)
			, tmpDefs);
		}

		if (mWorldSpaceReuse && mTemporalReuse && mSpatialReusePasses > 0) {
			mReservoirHashGrids[mCurHashGrid].Build(commandBuffer);
			mCurHashGrid ^= 1;
		}

		commandBuffer.Copy(mPathReservoirsBuffers[i], mPrevReservoirs);

		if (!mPrevFrameDoneEvent)
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{});
		commandBuffer->setEvent(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer);
	}
};

}
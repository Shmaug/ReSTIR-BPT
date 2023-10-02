#pragma once

#include "VisibilityPass.hpp"
#include "HashGrid.hpp"

namespace ptvk {

class BPTPass {
public:
	std::unordered_map<std::string, ComputePipelineCache> mPipelines;
	ShaderParameterBlock mParameters;
	std::unordered_map<std::string, bool> mDefines;

	float mLightSubpathCount = 1;
	bool mLightTrace = false;

	Buffer::View<std::byte> mPathStates;
	Buffer::View<std::byte> mAtomicOutput;
	Buffer::View<std::byte> mLightVertices;
	Buffer::View<std::byte> mCounters;
	Buffer::View<std::byte> mShadowRays;

	HashGrid mLightVertexHashGrid;
	std::array<HashGrid, 2> mLightVertexHashGrids;
	uint mHashGridIndex = 0;

	std::unique_ptr<vk::raii::Event> mPrevFrameDoneEvent;
	std::vector<vk::BufferMemoryBarrier2> mPrevFrameBarriers;

	inline BPTPass(Device& device) {
		mDefines = {
			{ "gAlphaTest", true },
			{ "gNormalMaps", true },
			{ "gShadingNormals", true },
			{ "gPixelJitter", false },
			{ "DISNEY_BRDF", true },
			{ "gDebugFastBRDF", false },
			{ "gDebugPaths", false },
			{ "gDebugPathWeights", false },
			{ "gMultiDispatch", true },
			{ "gDeferShadowRays", true },
			{ "gSampleDirectIllumination", false },
			{ "gSampleDirectIlluminationOnly", false },
			{ "gUseVC", true },
			{ "gUseVM", false },
			{ "gUsePpm", false },
			{ "gLVCResampling", false },
			{ "gLVCResamplingReuse", false },
			{ "DEBUG_PIXEL", false }
		};

		mParameters.SetConstant("gMinDepth", 2u);
		mParameters.SetConstant("gMaxDepth", 5u);
		mParameters.SetConstant("gDebugPathLengths", 3 | (1<<16));
		mParameters.SetConstant("gLVCCanonicalCandidates", 3u);
		mParameters.SetConstant("gLVCReuseCandidates", 1u);
		mParameters.SetConstant("gLVCJitterRadius", 0.1f);
		mParameters.SetConstant("gLVCMCap", 20u);

		mParameters.SetConstant("gDebugPixel", -1);

		mLightVertexHashGrid = HashGrid(device.mInstance);
		mLightVertexHashGrid.mElementSize = 48;
		mLightVertexHashGrids[0] = HashGrid(device.mInstance);
		mLightVertexHashGrids[1] = HashGrid(device.mInstance);

		auto staticSampler = std::make_shared<vk::raii::Sampler>(*device, vk::SamplerCreateInfo({},
			vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat,
			0, true, 8, false, vk::CompareOp::eAlways, 0, VK_LOD_CLAMP_NONE));
		device.SetDebugName(**staticSampler, "BPTPass/Sampler");

		PipelineInfo md;
		md.mImmutableSamplers["gScene.mStaticSampler"]  = { staticSampler };
		md.mBindingFlags["gScene.mVertexBuffers"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mImage1s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mImage2s"] = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mImage4s"]  = vk::DescriptorBindingFlagBits::ePartiallyBound;
		md.mBindingFlags["gScene.mVolumes"] = vk::DescriptorBindingFlagBits::ePartiallyBound;

		std::vector<std::string> args = {
			"-O3",
			"-Wno-30081",
			"-capability", "spirv_1_5",
			"-capability", "GL_EXT_ray_tracing"
		};

		const std::filesystem::path shaderPath = *device.mInstance.GetOption("shader-kernel-path");
		mPipelines.emplace("Render",              ComputePipelineCache(shaderPath / "Kernels/Bidirectional.slang", "Render"             , "sm_6_7", args, md));
		mPipelines.emplace("RenderIteration",     ComputePipelineCache(shaderPath / "Kernels/Bidirectional.slang", "RenderIteration"    , "sm_6_7", args, md));
		mPipelines.emplace("ProcessShadowRays",   ComputePipelineCache(shaderPath / "Kernels/Bidirectional.slang", "ProcessShadowRays"  , "sm_6_7", args, md));
		mPipelines.emplace("ProcessAtomicOutput", ComputePipelineCache(shaderPath / "Kernels/Bidirectional.slang", "ProcessAtomicOutput", "sm_6_7", args, md));
	}

	inline void OnInspectorGui() {
		bool changed = false;

		ImGui::PushID(this);
		if (ImGui::Checkbox("Light tracing", &mLightTrace)) changed = true;

		if (ImGui::CollapsingHeader("Defines")) {
			for (auto&[define, enabled] : mDefines) {
				if (ImGui::Checkbox(define.c_str(), &enabled)) changed = true;
			}

			if (changed) {
				// make sure defines are consistent

				if (mDefines.at("gDebugPathWeights"))
					mDefines.at("gDebugPaths") = true;

				if (mLightTrace) {
					mDefines.at("gUseVC") = false;
					mDefines.at("gUseVM") = false;
					mDefines.at("gUsePpm") = false;
					mDefines.at("gSampleDirectIllumination") = false;
					mDefines.at("gSampleDirectIlluminationOnly") = false;
				}

				if (mDefines.at("gUseVC")) {
					mDefines.at("gSampleDirectIllumination") = false;
					mDefines.at("gSampleDirectIlluminationOnly") = false;
				} else
					mDefines.at("gLVCResampling") = false;

				if (!mDefines.at("gLVCResampling"))
					mDefines.at("gLVCResamplingReuse") = false;

				if (mDefines.at("gUsePpm")) {
					mDefines.at("gUseVM") = true;
					mDefines.at("gUseVC") = false;
					mDefines.at("gSampleDirectIllumination") = false;
					mDefines.at("gSampleDirectIlluminationOnly") = false;
				}
			}
		}

		if (ImGui::CollapsingHeader("Path Tracing")) {
			if (Gui::ScalarField<uint32_t>("Min depth", &mParameters.GetConstant<uint32_t>("gMinDepth"), 1, 0, .2f)) changed = true;
			if (Gui::ScalarField<uint32_t>("Max depth", &mParameters.GetConstant<uint32_t>("gMaxDepth"), 1, 0, .2f)) changed = true;
			if (mDefines.at("gUseVM") || mDefines.at("gUseVC") || mLightTrace) {
				if (Gui::ScalarField<float>("Light subpath count", &mLightSubpathCount, 0, 2, 0)) changed = true;
			}

			if (mDefines.at("gDebugPaths")) {
				ImGui::SetNextItemWidth(40);
				if (ImGui::DragScalarN("Length, light vertices", ImGuiDataType_U16, &mParameters.GetConstant<uint32_t>("gDebugPathLengths"), 2, .2f)) changed = true;
			}
		}

		if (mDefines.at("gUseVM")) {
			if (ImGui::CollapsingHeader("Vertex merging")) {
				if (Gui::ScalarField<uint32_t>("Cell count", &mLightVertexHashGrid.mCellCount, 1000, 0xFFFFFF)) changed = true;
				if (Gui::ScalarField<float>("Cell size", &mLightVertexHashGrid.mCellSize, 0.001f, 100, 0.01f)) changed = true;
			}
		}

		if (mDefines.at("gUseVC") && mDefines.at("gLVCResampling")) {
			if (ImGui::CollapsingHeader("Light Vertex Resampling")) {
				if (Gui::ScalarField<uint32_t>("Canonical samples", &mParameters.GetConstant<uint32_t>("gLVCCanonicalCandidates"), 1, 100, 0.1f)) changed = true;
				if (mDefines.at("gLVCResamplingReuse")) {
					if (Gui::ScalarField<uint32_t>("Reuse samples", &mParameters.GetConstant<uint32_t>("gLVCReuseCandidates"), 0, 100, 0.5f)) changed = true;
					if (Gui::ScalarField<uint32_t>("M Cap", &mParameters.GetConstant<uint32_t>("gLVCMCap"), 0, 1000, 0.05f)) changed = true;
					if (Gui::ScalarField<float>("Jitter radius", &mParameters.GetConstant<float>("gLVCJitterRadius"), 0, 100, 0.05f)) changed = true;
					if (Gui::ScalarField<uint32_t>("Cell count", &mLightVertexHashGrids[0].mCellCount, 0, 0xFFFFFF)) changed = true;
					if (Gui::ScalarField<float>("Cell size", &mLightVertexHashGrids[0].mCellSize, 0, 100, 0.05f)) changed = true;
					if (Gui::ScalarField<float>("Cell pixel radius", &mLightVertexHashGrids[0].mCellPixelRadius, 0, 100, 0.05f)) changed = true;

					mLightVertexHashGrids[1].mCellCount       = mLightVertexHashGrids[0].mCellCount;
					mLightVertexHashGrids[1].mCellSize        = mLightVertexHashGrids[0].mCellSize;
					mLightVertexHashGrids[1].mCellPixelRadius = mLightVertexHashGrids[0].mCellPixelRadius;
				}
			}
		}

		ImGui::PopID();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const VisibilityPass& visibility) {
		ProfilerScope ps("Bidirectional::render", &commandBuffer);

		const vk::Extent3D extent = renderTarget.GetExtent();

		// assign descriptors

		const uint32_t maxShadowRays = mDefines.at("gDeferShadowRays") ? (mParameters.GetConstant<uint32_t>("gMaxDepth")-1)*(
			extent.width*extent.height*(mDefines.at("gUseVC") ? 2 : 1) + (mLightTrace || mDefines.at("gUseVC") ? mParameters.GetConstant<uint32_t>("gLightSubpathCount") : 0))
			: 0;

		vk::DeviceSize sz = sizeof(float4x4) * (mDefines.at("gMultiDispatch") ? extent.width*extent.height : 1);
		if (!mPathStates || mPathStates.SizeBytes() != sz) mPathStates = std::make_shared<Buffer>(commandBuffer.mDevice, "gPathStates", sz, vk::BufferUsageFlagBits::eStorageBuffer);

		sz = sizeof(uint4) * ((mDefines.at("gDeferShadowRays")||mDefines.at("gUseVC")||mLightTrace) ? extent.width*extent.height : 1);
		if (!mAtomicOutput || mAtomicOutput.SizeBytes() != sz) mAtomicOutput = std::make_shared<Buffer>(commandBuffer.mDevice, "gOutputAtomic", sz, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);

		sz = 48 * (mDefines.at("gUseVC") && !mDefines.at("gUseVM") ? max(1u, mParameters.GetConstant<uint32_t>("gLightSubpathCount")*(mParameters.GetConstant<uint32_t>("gMaxDepth")-1)) : 1);
		if (!mLightVertices || mLightVertices.SizeBytes() != sz) mLightVertices = std::make_shared<Buffer>(commandBuffer.mDevice, "gLightVertices", sz, vk::BufferUsageFlagBits::eStorageBuffer);

		sz = sizeof(float4)*4 * max(1u, maxShadowRays);
		if (!mShadowRays || mShadowRays.SizeBytes() != sz) mShadowRays = std::make_shared<Buffer>(commandBuffer.mDevice, "gShadowRays", sz, vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, 0);

		sz = sizeof(uint32_t) * (2 + mDefines.at("gUseVC") ? extent.width*extent.height : 1);
		if (!mCounters || mCounters.SizeBytes() != sz) mCounters = std::make_shared<Buffer>(commandBuffer.mDevice, "gCounters", sz, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal, 0);

		const float imagePlaneDist = extent.height / (2 * std::tan(visibility.GetVerticalFov()/2));

		mParameters.SetParameters("gScene", scene.GetRenderData().mShaderParameters);
		mParameters.SetParameters(visibility.GetDebugParameters());

		mParameters.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite);
		mParameters.SetImage("gVertices", visibility.GetVertices(), vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead);
		mParameters.SetBuffer("gPathStates", mPathStates);
		mParameters.SetBuffer("gOutputAtomic", mAtomicOutput);
		if (!mDefines.at("gUseVM"))
			mParameters.SetBuffer("gLightVertices", mLightVertices);
		mParameters.SetBuffer("gCounters", mCounters);
		mParameters.SetBuffer("gShadowRays", mShadowRays);

		mParameters.SetConstant("gOutputSize", uint2(extent.width, extent.height));
		mParameters.SetConstant("gLightSubpathCount", max(1u, uint32_t(extent.width * extent.height * mLightSubpathCount)));
		mParameters.SetConstant("gRandomSeed", (uint32_t)commandBuffer.mDevice.GetFrameIndex());
		mParameters.SetConstant("gCameraToWorld", visibility.GetCameraToWorld());
		mParameters.SetConstant("gWorldToCamera", inverse(visibility.GetCameraToWorld()));
		mParameters.SetConstant("gProjection", visibility.GetProjection());
		mParameters.SetConstant("gCameraPosition", visibility.GetCameraPosition());
		mParameters.SetConstant("gImagePlaneDist", imagePlaneDist);

		if (mDefines.at("DEBUG_PIXEL")) {
			const ImGuiIO& io = ImGui::GetIO();
			if (ImGui::IsMouseDown(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
				const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
				uint2 p = uint2(float2((uint32_t)io.MousePos.x, (uint32_t)io.MousePos.y) * float2(extent.width, extent.height) / float2(size.x, size.y));
				mParameters.SetConstant("gDebugPixel", p.y * extent.width + p.x);
			}
		}

		auto& lightVertexGrid = mLightVertexHashGrids[mHashGridIndex];
		auto& prevLightVertexGrid = mLightVertexHashGrids[mHashGridIndex^1];
		mHashGridIndex ^= 1;

		// setup shader defines

		Defines defines;
		for (const auto&[define,enabled] : mDefines)
			if (enabled)
				defines[define] = std::to_string(enabled);

		if (visibility.HeatmapCounterType() != DebugCounterType::eNumDebugCounters)
			defines.emplace("gEnableDebugCounters", "true");

		// render

		if (mDefines.at("gDeferShadowRays") || mDefines.at("gUseVC")) {
			commandBuffer.Fill(mCounters, 0);
		}


		std::vector<const char*> loading;
		auto DispatchIfLoaded = [&](const char* name, const vk::Extent3D& extent, const Defines& defs) {
			auto pipeline = mPipelines.at(name).GetPipelineAsync(commandBuffer.mDevice, defs);
			if (pipeline)
				mPipelines.at(name).Dispatch(commandBuffer, extent, mParameters, *pipeline);
			else
				loading.push_back(name);
		};

		// hash grids
		if (!mLightTrace) {
			if (mDefines.at("gUseVM")) {
				mLightVertexHashGrid.mSize = max(1u, mParameters.GetConstant<uint32_t>("gLightSubpathCount") * (max(1u, mParameters.GetConstant<uint32_t>("gMaxDepth"))-1));
				mLightVertexHashGrid.Prepare(commandBuffer, visibility.GetCameraPosition(), visibility.GetVerticalFov(), uint2(extent.width, extent.height));
				mParameters.SetParameters("gLightVertices", mLightVertexHashGrid.mParameters);
			}
			if (mDefines.at("gLVCResampling")) {
				if (mPrevFrameDoneEvent && !mPrevFrameBarriers.empty()) {
					commandBuffer->waitEvents2(**mPrevFrameDoneEvent, vk::DependencyInfo{ {}, {},  mPrevFrameBarriers, {} });
				}
				lightVertexGrid.mSize = max(1u, extent.width * extent.height * (max(2u, mParameters.GetConstant<uint32_t>("gMaxDepth"))-2));
				lightVertexGrid.mElementSize = 96;
				lightVertexGrid.Prepare(commandBuffer, visibility.GetCameraPosition(), visibility.GetVerticalFov(), uint2(extent.width, extent.height));
				if (prevLightVertexGrid.mParameters.empty()) {
					lightVertexGrid.mSize        = lightVertexGrid.mSize;
					lightVertexGrid.mElementSize = lightVertexGrid.mElementSize;
					prevLightVertexGrid.Prepare(commandBuffer, visibility.GetCameraPosition(), visibility.GetVerticalFov(), uint2(extent.width, extent.height));
				}

				mParameters.SetParameters("gLightVertexHashGrid", lightVertexGrid.mParameters);
				mParameters.SetParameters("gPrevLightVertexHashGrid", prevLightVertexGrid.mParameters);
			}
			mPrevFrameBarriers.clear();
		}

		// light paths
		if (mDefines.at("gUseVC") || mDefines.at("gUseVM") || mLightTrace) {
			ProfilerScope ps("Light paths", &commandBuffer);

			commandBuffer.Fill(mAtomicOutput, 0);

			Defines tmpDefs = defines;
			tmpDefs["gTraceFromLight"] = "true";
			if (mParameters.GetConstant<uint32_t>("gLVCJitterRadius") > 0)
				tmpDefs["gLVCJitter"] = "true";
			if (mDefines.at("gMultiDispatch"))
				tmpDefs["gMultiDispatchFirst"] = "true";

			const vk::Extent3D lightExtent = { extent.width, (mParameters.GetConstant<uint32_t>("gLightSubpathCount") + extent.width-1)/extent.width, 1 };
			DispatchIfLoaded("Render", lightExtent, tmpDefs);

			if (mDefines.at("gMultiDispatch")) {
				tmpDefs.erase("gMultiDispatchFirst");
				for (uint32_t i = 1; i < mParameters.GetConstant<uint32_t>("gMaxDepth"); i++) {
					DispatchIfLoaded("RenderIteration", lightExtent, tmpDefs);
				}
			}

			if (mDefines.at("gUseVM"))
				mLightVertexHashGrid.Build(commandBuffer);
		}

		// view paths
		if (!mLightTrace) {
			ProfilerScope ps("View paths", &commandBuffer);

			Defines tmpDefs = defines;
			if (mDefines.at("gMultiDispatch"))
				tmpDefs["gMultiDispatchFirst"] = "true";

			DispatchIfLoaded("Render", extent, tmpDefs);
			if (mDefines.at("gMultiDispatch")) {
				tmpDefs.erase("gMultiDispatchFirst");
				for (uint32_t i = 1; i < mParameters.GetConstant<uint32_t>("gMaxDepth"); i++) {
					DispatchIfLoaded("RenderIteration", extent, tmpDefs);
				}
			}

			// build hash grid for next frame
			if (mDefines.at("gLVCResampling")) {
				lightVertexGrid.Build(commandBuffer);
				mPrevFrameBarriers.clear();
				for (const auto& b : {
					lightVertexGrid.mParameters.GetBuffer<std::byte>("mIndices"),
					lightVertexGrid.mParameters.GetBuffer<std::byte>("mCellCounters"),
					lightVertexGrid.mParameters.GetBuffer<std::byte>("mData") })
					mPrevFrameBarriers.emplace_back( vk::BufferMemoryBarrier2{
						vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
						vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
						VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
						**b.GetBuffer(), b.Offset(), b.SizeBytes() });
				if (!mPrevFrameDoneEvent)
					mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{ vk::EventCreateFlagBits::eDeviceOnly });
				commandBuffer->setEvent2(**mPrevFrameDoneEvent, vk::DependencyInfo{ {}, {},  mPrevFrameBarriers, {} });
			}
		}

		// shadow rays
		if (mDefines.at("gDeferShadowRays")) {
			ProfilerScope ps("Shadow rays", &commandBuffer);
			if (!mDefines.at("gUseVC") && !mLightTrace)
				commandBuffer.Fill(mAtomicOutput, 0);
			DispatchIfLoaded("ProcessShadowRays", vk::Extent3D{extent.width, (maxShadowRays + extent.width-1) / extent.width, 1}, defines);
		}

		// copy light image
		if (mDefines.at("gDeferShadowRays") || mDefines.at("gUseVC") || mLightTrace) {
			DispatchIfLoaded("ProcessAtomicOutput", extent, Defines{ { "gClearImage", std::to_string(mLightTrace) }});
		}

		if (!loading.empty()) {
			const ImVec2 size = ImGui::GetMainViewport()->WorkSize;
			ImGui::SetNextWindowPos(ImVec2(size.x/2, size.y/2));
			if (ImGui::Begin("Compiling shaders", nullptr, ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoNav|ImGuiWindowFlags_NoDecoration|ImGuiWindowFlags_NoInputs)) {
				for (const char* l : loading)
					ImGui::Text(l);
				Gui::ProgressSpinner("Compiling shaders", 15, 6, false);
			}
			ImGui::End();
		}
	}
};

}
#pragma once

#include "VisibilityPass.hpp"

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

	inline BPTPass(Device& device) {
		mDefines = {
			{ "gAlphaTest", true },
			{ "gNormalMaps", true },
			{ "gShadingNormals", true },
			{ "gPixelJitter", false },
			{ "DISNEY_BRDF", false },
			{ "gDebugFastBRDF", false },
			{ "gDebugPaths", false },
			{ "gDebugPathWeights", false },
			{ "gMultiDispatch", true },
			{ "gDeferShadowRays", true },
			{ "gSampleDirectIllumination", false },
			{ "gSampleDirectIlluminationOnly", false },
			{ "gUseVC", true },
			{ "gEvalAllLightVertices", false }
		};

		mParameters.SetConstant("gMinDepth", 2u);
		mParameters.SetConstant("gMaxDepth", 5u);
		mParameters.SetConstant("gDebugPathLengths", 3 | (1<<16));

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
					mDefines.at("gSampleDirectIllumination") = false;
					mDefines.at("gSampleDirectIlluminationOnly") = false;
				}

				if (mDefines.at("gUseVC")) {
					mDefines.at("gSampleDirectIllumination") = false;
					mDefines.at("gSampleDirectIlluminationOnly") = false;
				} else
					mDefines.at("gEvalAllLightVertices") = false;
			}
		}

		if (ImGui::CollapsingHeader("Configuration")) {
			if (mDefines.at("gDebugPaths")) {
				ImGui::SetNextItemWidth(40);
				if (ImGui::DragScalarN("Length, light vertices", ImGuiDataType_U16, &mParameters.GetConstant<uint32_t>("gDebugPathLengths"), 2, .2f)) changed = true;
			}

			if (Gui::ScalarField<uint32_t>("Min depth", &mParameters.GetConstant<uint32_t>("gMinDepth"), 1, 0, .2f)) changed = true;
			if (Gui::ScalarField<uint32_t>("Max depth", &mParameters.GetConstant<uint32_t>("gMaxDepth"), 1, 0, .2f)) changed = true;
			if (mDefines.at("gUseVC") || mLightTrace) {
				if (Gui::ScalarField<float>("Light subpath count", &mLightSubpathCount, 0, 2, 0)) changed = true;
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

		sz = 48 * (mDefines.at("gUseVC") ? max(1u, mParameters.GetConstant<uint32_t>("gLightSubpathCount")*(mParameters.GetConstant<uint32_t>("gMaxDepth")-1)) : 1);
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

		// light paths
		if (mDefines.at("gUseVC") || mLightTrace) {
			ProfilerScope ps("Light paths", &commandBuffer);

			commandBuffer.Fill(mAtomicOutput, 0);

			Defines tmpDefs = defines;
			tmpDefs["gTraceFromLight"] = "true";
			if (mDefines.at("gMultiDispatch"))
				tmpDefs["gMultiDispatchFirst"] = "true";

			const vk::Extent3D lightExtent = { extent.width, (mParameters.GetConstant<uint32_t>("gLightSubpathCount") + extent.width-1)/extent.width, 1 };
			mPipelines.at("Render").Dispatch(commandBuffer, lightExtent, mParameters, tmpDefs);

			if (mDefines.at("gMultiDispatch")) {
				tmpDefs.erase("gMultiDispatchFirst");
				for (uint32_t i = 1; i < mParameters.GetConstant<uint32_t>("gMaxDepth"); i++) {
					mPipelines.at("RenderIteration").Dispatch(commandBuffer, lightExtent, mParameters, tmpDefs);
				}
			}
		}

		// view paths
		if (!mLightTrace) {
			ProfilerScope ps("View paths", &commandBuffer);

			Defines tmpDefs = defines;
			if (mDefines.at("gMultiDispatch"))
				tmpDefs["gMultiDispatchFirst"] = "true";

			mPipelines.at("Render").Dispatch(commandBuffer, extent, mParameters, tmpDefs);
			if (mDefines.at("gMultiDispatch")) {
				tmpDefs.erase("gMultiDispatchFirst");
				for (uint32_t i = 1; i < mParameters.GetConstant<uint32_t>("gMaxDepth"); i++) {
					mPipelines.at("RenderIteration").Dispatch(commandBuffer, extent, mParameters, tmpDefs);
				}
			}
		}

		if (mDefines.at("gDeferShadowRays")) {
			ProfilerScope ps("Shadow rays", &commandBuffer);
			if (!mDefines.at("gUseVC") && !mLightTrace)
				commandBuffer.Fill(mAtomicOutput, 0);
			mPipelines.at("ProcessShadowRays").Dispatch(commandBuffer, vk::Extent3D{extent.width, (maxShadowRays + extent.width-1) / extent.width, 1}, mParameters, defines);
		}

		if (mDefines.at("gDeferShadowRays") || mDefines.at("gUseVC") || mLightTrace) {
			mPipelines.at("ProcessAtomicOutput").Dispatch(commandBuffer, extent, mParameters, Defines{ { "gClearImage", std::to_string(mLightTrace) }});
		}
	}
};

}
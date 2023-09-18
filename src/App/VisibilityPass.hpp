#pragma once

#include <Common/Enums.h>
#include <Core/PipelineCache.hpp>
#include <Scene/Scene.hpp>

namespace ptvk {

class VisibilityPass {
private:
	ComputePipelineCache mRenderVisibilityPipeline;
	ComputePipelineCache mRenderHeatmapPipeline;

	bool mAlphaTest = true;
	bool mShadingNormals = true;
	bool mRenderAlbedos = false;
	bool mRenderNormals = false;

	Buffer::View<uint32_t> mDebugCounters;
	Buffer::View<uint32_t> mDebugHeatmap;
	DebugCounterType mDebugHeatmapType = DebugCounterType::eNumDebugCounters;

	Image::View mAlbedos;
	Image::View mDepthNormals;
	Image::View mVertices;
	float4x4 mCameraToWorld;
	float4x4 mProjection;
	float mCameraVerticalFov;

	Image::View mPrevDepthNormals;
	Image::View mPrevVertices;
	float3      mPrevCameraPosition;
	float3      mPrevCameraForward;
	float4x4    mPrevMVP;
	std::unique_ptr<vk::raii::Event> mPrevFrameDoneEvent;

public:
	inline VisibilityPass(Device& device) {
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

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path");
		mRenderVisibilityPipeline = ComputePipelineCache(shaderFile + "/Kernels/Visibility.slang", "RenderVisibility", "sm_6_7", args, md);
		mRenderHeatmapPipeline    = ComputePipelineCache(shaderFile + "/DebugCounters.slang", "RenderHeatmap"   , "sm_6_7", args, md);
	}

	inline Image::View GetVertices()     const { return mVertices; }
	inline Image::View GetDepthNormals() const { return mDepthNormals; }
	inline Image::View GetAlbedos()      const { return mAlbedos; }

	inline const Image::View& GetPrevDepthNormals() const { return mPrevDepthNormals; }
	inline const Image::View& GetPrevVertices() const { return mPrevVertices; }
	inline const float4x4& GetCameraToWorld() const { return mCameraToWorld; }
	inline const float4x4& GetProjection() const { return mProjection; }
	inline float3 GetCameraPosition() const { return TransformPoint(mCameraToWorld, float3(0)); }
	inline float3 GetCameraForward() const { return TransformVector(mCameraToWorld, float3(0,0,-1)); }
	inline float GetVerticalFov() const { return mCameraVerticalFov; }
	inline float4x4 GetMVP() const { return mProjection * inverse(mCameraToWorld); }
	inline const float4x4& GetPrevMVP() const { return mPrevMVP; }
	inline const float3& GetPrevCameraPosition() const { return mPrevCameraPosition; }
	inline const float3& GetPrevCameraForward() const { return mPrevCameraForward; }

	inline DebugCounterType HeatmapCounterType() const { return mDebugHeatmapType; }

	inline ShaderParameterBlock GetDebugParameters() const {
		return ShaderParameterBlock()
			.SetBuffer("gDebugCounters", mDebugCounters)
			.SetBuffer("gHeatmap", mDebugHeatmap)
			.SetConstant("gHeatmapCounterType", (uint32_t)mDebugHeatmapType);
	}

	inline void OnInspectorGui() {
		ImGui::PushID(this);
		ImGui::Checkbox("Alpha test", &mAlphaTest);
		ImGui::Checkbox("Shading normals", &mShadingNormals);
		ImGui::Checkbox("Render albedos", &mRenderAlbedos);
		ImGui::Checkbox("Render normals", &mRenderNormals);
		Gui::EnumDropdown<DebugCounterType>("Debug Heatmap", mDebugHeatmapType, DebugCounterTypeStrings);
		ImGui::PopID();
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const Camera& camera) {
		ProfilerScope p("ReSTIRPTPass::Render", &commandBuffer);

		const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);

		if (!mAlbedos || mAlbedos.GetExtent().width != extent.x || mAlbedos.GetExtent().height != extent.y) {
			mAlbedos = std::make_shared<Image>(commandBuffer.mDevice, "gAlbedos", ImageInfo{
				.mFormat = vk::Format::eR8G8B8A8Unorm,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc
			});
			mDepthNormals = std::make_shared<Image>(commandBuffer.mDevice, "gDepthNormals", ImageInfo{
				.mFormat = vk::Format::eR32G32B32A32Sfloat,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc
			});
			mVertices = std::make_shared<Image>(commandBuffer.mDevice, "gVertices", ImageInfo{
				.mFormat = vk::Format::eR32G32B32A32Uint,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferSrc
			});
			mPrevDepthNormals  = std::make_shared<Image>(commandBuffer.mDevice, "gPrevVertices", ImageInfo{
				.mFormat = vk::Format::eR32G32B32A32Sfloat,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});
			mPrevVertices = std::make_shared<Image>(commandBuffer.mDevice, "gPrevVertices", ImageInfo{
				.mFormat = vk::Format::eR32G32B32A32Uint,
				.mExtent = renderTarget.GetExtent(),
				.mUsage = vk::ImageUsageFlagBits::eSampled|vk::ImageUsageFlagBits::eStorage|vk::ImageUsageFlagBits::eTransferDst
			});
			mDebugCounters = std::make_shared<Buffer>(commandBuffer.mDevice, "gDebugCounters", ((uint32_t)DebugCounterType::eNumDebugCounters+1) * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
			mDebugHeatmap = std::make_shared<Buffer>(commandBuffer.mDevice, "gDebugHeatmap", vk::DeviceSize(extent.x) * vk::DeviceSize(extent.y) * sizeof(uint32_t), vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
		} else if (mPrevFrameDoneEvent) {
			commandBuffer->waitEvents(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {
				vk::ImageMemoryBarrier{
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**mPrevDepthNormals.GetImage(), mPrevDepthNormals.GetSubresourceRange()
				},
				vk::ImageMemoryBarrier{
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral,
					VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
					**mPrevVertices.GetImage(), mPrevVertices.GetSubresourceRange()
				}
			});
			mPrevDepthNormals.SetSubresourceState(vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
			mPrevVertices.SetSubresourceState(vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eComputeShader, vk::AccessFlagBits::eShaderRead);
		}

		mCameraToWorld = NodeToWorld(camera.mNode);
		mProjection = glm::scale(camera.GetProjection(), float3(1, -1, 1));
		mCameraVerticalFov = camera.mVerticalFov;

		Defines defs;
		if (mAlphaTest)      defs.emplace("gAlphaTest", "true");
		if (mShadingNormals) defs.emplace("gShadingNormals", "true");
		if (mRenderAlbedos)  defs.emplace("gRenderAlbedos", "true");
		if (mRenderNormals)  defs.emplace("gRenderNormals", "true");
		if (mDebugHeatmapType != DebugCounterType::eNumDebugCounters) {
			defs.emplace("gEnableDebugCounters", "true");
			commandBuffer.Fill(mDebugCounters, 0);
			commandBuffer.Fill(mDebugHeatmap, 0);
		}

		mRenderVisibilityPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), ShaderParameterBlock()
			.SetImage("gRadiance"    , renderTarget , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
			.SetImage("gAlbedos"     , mAlbedos     , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
			.SetImage("gDepthNormals", mDepthNormals, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
			.SetImage("gVertices"    , mVertices    , vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
			.SetConstant("gCameraToWorld", mCameraToWorld)
			.SetConstant("gInverseProjection", inverse(mProjection))
			.SetConstant("gOutputSize", extent)
			.SetParameters("gScene", scene.GetRenderData().mShaderParameters)
			.SetParameters(GetDebugParameters()),
			defs);
	}

	inline void PostRender(CommandBuffer& commandBuffer, const Image::View& renderTarget) {
		commandBuffer.Copy(mDepthNormals, mPrevDepthNormals);
		commandBuffer.Copy(mVertices, mPrevVertices);
		if (!mPrevFrameDoneEvent)
			mPrevFrameDoneEvent = std::make_unique<vk::raii::Event>(*commandBuffer.mDevice, vk::EventCreateInfo{ vk::EventCreateFlagBits::eDeviceOnly });
		commandBuffer->setEvent(**mPrevFrameDoneEvent, vk::PipelineStageFlagBits::eTransfer);
		mPrevCameraPosition = GetCameraPosition();
		mPrevCameraForward = GetCameraForward();
		mPrevMVP = GetMVP();

		if (mDebugHeatmapType != DebugCounterType::eNumDebugCounters) {
			const uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);
			mRenderHeatmapPipeline.Dispatch(commandBuffer, renderTarget.GetExtent(), ShaderParameterBlock()
				.SetConstant("gOutputSize", extent)
				.SetImage("gRadiance", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderRead|vk::AccessFlagBits::eShaderWrite)
				.SetConstant("gOutputSize", extent)
				.SetParameters(GetDebugParameters()),
				{{"DEBUG_HEATMAP_SHADER", ""}});
		}
	}
};

}
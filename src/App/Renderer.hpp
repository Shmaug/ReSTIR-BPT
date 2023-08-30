#pragma once

#include <Common/Common.h>
#include <Core/PipelineCache.hpp>
#include <Scene/Scene.hpp>

namespace ptvk {

class Renderer {
public:
	ComputePipelineCache mRenderPipeline;

	Buffer::View<uint32_t> mRayCountBuffer;

	inline Renderer(Device& device) {
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

		const std::string shaderFile = *device.mInstance.GetOption("shader-kernel-path")  + "/Kernels/Test.slang";
		mRenderPipeline = ComputePipelineCache(shaderFile, "Render", "sm_6_7", args, md);

		mRayCountBuffer = std::make_shared<Buffer>(device, "gRayCount", 16, vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst);
	}

	inline void Render(CommandBuffer& commandBuffer, const Image::View& renderTarget, const Scene& scene, const Camera& camera) {
		ProfilerScope p("Renderer::Render");

		uint2 extent = uint2(renderTarget.GetExtent().width, renderTarget.GetExtent().height);
		size_t pixelCount = size_t(extent.x) * size_t(extent.y);

		const float4x4 cameraToWorld = NodeToWorld(camera.mNode);
		const float4x4 projection = camera.GetProjection();

		commandBuffer.Fill(mRayCountBuffer, 0);

		mRenderPipeline.Dispatch(commandBuffer,
			renderTarget.GetExtent(),
			ShaderParameterBlock()
				.SetImage("gOutput", renderTarget, vk::ImageLayout::eGeneral, vk::AccessFlagBits::eShaderWrite)
				.SetConstant("gOutputSize", extent)
				.SetConstant("gCameraToWorld", cameraToWorld)
				.SetConstant("gWorldToCamera", inverse(cameraToWorld))
				.SetConstant("gProjection", projection)
				.SetConstant("gInverseProjection", inverse(projection))
				.SetParameters("gScene", scene.GetRenderData().mShaderParameters)
				.SetBuffer("gRayCount", mRayCountBuffer)
		);
	}

	inline void OnInspectorGui() {

	}
};

}
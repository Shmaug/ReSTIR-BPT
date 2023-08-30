#pragma once

#define gVertexBufferCount 2048
#define gImageCount 2048
#define gVolumeCount 8

#include "Common/Math.h"
#include "Common/SceneTypes.h"
#include "Common/Material.h"
#include "Sampling.slang"

struct SceneConstants {
    float3 mSceneMin;
    uint mInstanceCount;
    float3 mSceneMax;
    uint mLightCount;
    float3 mBackgroundColor;
    uint mBackgroundImageIndex;
    float mBackgroundSampleProbability;

    bool HasBackground() { return any(mBackgroundColor > 0); }

	#ifndef NO_SCENE_ACCELERATION_STRUCTURE
	RaytracingAccelerationStructure mAccelerationStructure;
	#endif

	StructuredBuffer<InstanceBase> mInstances;
	StructuredBuffer<float4x4> mInstanceTransforms;
	StructuredBuffer<float4x4> mInstanceInverseTransforms;
	StructuredBuffer<float4x4> mInstanceMotionTransforms;

    StructuredBuffer<uint> mLightInstanceMap; // light index -> instance index
    StructuredBuffer<uint> mInstanceLightMap; // instance index -> light index

	StructuredBuffer<MeshVertexInfo> mMeshVertexInfo;
	StructuredBuffer<VolumeInfo> mInstanceVolumeInfo;
	StructuredBuffer<GpuMaterial> mMaterials;

    SamplerState mStaticSampler;

	ByteAddressBuffer mVertexBuffers[gVertexBufferCount];

    Texture2D<float2> mImage2s[gImageCount];
    Texture2D<float4> mImage4s[gImageCount];
	StructuredBuffer<uint> mVolumes[gVolumeCount];
};

ParameterBlock<SceneConstants> gScene;

uint GetVolumeIndex(const float3 position, const uint volumeInfoCount) {
	for (uint i = 0; i < volumeInfoCount; i++) {
		const VolumeInfo info = gScene.mInstanceVolumeInfo[i];
		const float3 localPos = TransformPoint(gScene.mInstanceInverseTransforms[info.mInstanceIndex], position);
		if (all(localPos >= info.mMin) && all(localPos <= info.mMax))
			return info.mInstanceIndex;
	}
	return INVALID_INSTANCE;
}

float3 EvalBackground(const float3 direction, out float pdfW) {
    if (!gScene.HasBackground()) {
        pdfW = 0;
        return 0;
    }

    float3 emission = gScene.mBackgroundColor;

    if (gScene.mBackgroundImageIndex < gImageCount) {
        const float2 uv = CartesianToSphericalUV(direction);
        emission *= gScene.mImage4s[gScene.mBackgroundImageIndex].SampleLevel(gScene.mStaticSampler, uv, 0).rgb;
        pdfW = SampleTexelPdf(gScene.mImage4s[gScene.mBackgroundImageIndex], uv) / (2 * M_PI * M_PI * sqrt(1 - direction.y * direction.y));
    } else {
        pdfW = 1 / (4 * M_PI);
    }

    if (gScene.mLightCount > 0)
        pdfW *= gScene.mBackgroundSampleProbability;

    return emission;
}


uint3 LoadTriangleIndices(const ByteAddressBuffer indices, const uint offset, const uint indexStride, const uint primitiveIndex) {
    const int offsetBytes = (int)(offset + primitiveIndex * 3 * indexStride);
    uint3 tri;
    if (indexStride == 2) {
        // https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/Samples/Desktop/D3D12Raytracing/src/D3D12RaytracingSimpleLighting/Raytracing.hlsl
        const int dwordAlignedOffset = offsetBytes & ~3;
        const uint2 four16BitIndices = indices.Load2(dwordAlignedOffset);
        if (dwordAlignedOffset == offsetBytes) {
            tri.x = four16BitIndices.x & 0xffff;
            tri.y = (four16BitIndices.x >> 16) & 0xffff;
            tri.z = four16BitIndices.y & 0xffff;
        } else {
            tri.x = (four16BitIndices.x >> 16) & 0xffff;
            tri.y = four16BitIndices.y & 0xffff;
            tri.z = (four16BitIndices.y >> 16) & 0xffff;
        }
    } else
        tri = indices.Load3(offsetBytes);
    return tri;
}
uint3 LoadTriangleIndices(const MeshVertexInfo vertexInfo, const uint primitiveIndex) {
	return LoadTriangleIndices(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetIndexBuffer())], vertexInfo.GetIndexOffset(), vertexInfo.GetIndexStride(), primitiveIndex);
}
uint3 LoadTriangleIndicesUniform(const MeshVertexInfo vertexInfo, const uint primitiveIndex) {
	return LoadTriangleIndices(gScene.mVertexBuffers[vertexInfo.GetIndexBuffer()], vertexInfo.GetIndexOffset(), vertexInfo.GetIndexStride(), primitiveIndex);
}

T LoadVertexAttribute<T>(const ByteAddressBuffer vertexBuffer, const uint offset, const uint stride, const uint index) {
    return vertexBuffer.Load<T>(int(offset + stride * index));
}
void LoadTriangleAttribute<T>(const ByteAddressBuffer vertexBuffer, const uint offset, const uint stride, const uint3 tri, out T v0, out T v1, out T v2) {
    v0 = LoadVertexAttribute<T>(vertexBuffer, offset, stride, tri[0]);
    v1 = LoadVertexAttribute<T>(vertexBuffer, offset, stride, tri[1]);
    v2 = LoadVertexAttribute<T>(vertexBuffer, offset, stride, tri[2]);
}

float2 SampleImage2(const uint imageIndex, const float2 uv, const float uvScreenSize = 0) {
    float lod = 0;
    if (uvScreenSize > 0) {
        float w, h;
        gScene.mImage4s[imageIndex].GetDimensions(w, h);
        lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
    }
    return gScene.mImage2s[NonUniformResourceIndex(imageIndex)].SampleLevel(gScene.mStaticSampler, uv, lod);
}
float4 SampleImage(const uint imageIndex, const float2 uv, const float uvScreenSize = 0) {
	float lod = 0;
	if (uvScreenSize > 0) {
		float w, h;
		gScene.mImage4s[imageIndex].GetDimensions(w, h);
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
	}
	return gScene.mImage4s[NonUniformResourceIndex(imageIndex)].SampleLevel(gScene.mStaticSampler, uv, lod);
}

void SampleMaterialImages(inout GpuMaterial m, const float2 uv, const float uvScreenSize = 0) {
	if (m.GetBaseColorImage() < gImageCount) {
		m.BaseColor(m.BaseColor() * SampleImage(m.GetBaseColorImage(), uv, uvScreenSize).rgb);
	}
	if (m.GetEmissionImage() < gImageCount) {
		m.Emission(m.Emission() * SampleImage(m.GetEmissionImage(), uv, uvScreenSize).rgb);
    }
    if (m.GetPackedParamsImage() < gImageCount) {
        const float4 packed = SampleImage(m.GetPackedParamsImage(), uv, uvScreenSize);
		m.Metallic(m.Metallic() * packed.x);
		m.Roughness(1 - (1 - m.Roughness()) * (1 - packed.y));
		m.Specular(m.Specular() * packed.x);
		m.Clearcoat(m.Clearcoat() * packed.z);
	}
}

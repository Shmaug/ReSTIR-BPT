#pragma once

#define gVertexBufferCount 2048
#define gImageCount 2048
#define gVolumeCount 8

#include "Common/Math.h"
#include "Common/SceneTypes.h"
#include "Common/Material.h"

struct SceneConstants {
	#ifndef NO_SCENE_ACCELERATION_STRUCTURE
	RaytracingAccelerationStructure mAccelerationStructure;
	#endif

	StructuredBuffer<RendererInstance> mInstances;
	StructuredBuffer<float4x4> mInstanceTransforms;
	StructuredBuffer<float4x4> mInstanceInverseTransforms;
	StructuredBuffer<float4x4> mInstanceMotionTransforms;

    float3 mSceneMin;
    uint mInstanceCount;
    float3 mSceneMax;
    uint mLightCount;
    float3 mBackgroundColor;
    uint mBackgroundImageIndex;

    StructuredBuffer<uint> mLightInstanceMap; // light index -> instance index
    StructuredBuffer<uint> mInstanceLightMap; // instance index -> light index

	StructuredBuffer<MeshVertexInfo> mMeshVertexInfo;
	StructuredBuffer<VolumeInfo> mInstanceVolumeInfo;
	StructuredBuffer<Material> mMaterials;

	RWStructuredBuffer<uint> mRayCount;
    SamplerState mStaticSampler;

	ByteAddressBuffer mVertexBuffers[gVertexBufferCount];

    Texture2D<float>  mImage1s[gImageCount];
    Texture2D<float2> mImage2s[gImageCount];
    Texture2D<float4> mImage4s[gImageCount];
	StructuredBuffer<uint> mVolumes[gVolumeCount];
};

SceneConstants gScene;

uint GetMediumIndex(const float3 position, const uint volumeInfoCount) {
	for (uint i = 0; i < volumeInfoCount; i++) {
		const VolumeInfo info = gScene.mInstanceVolumeInfo[i];
		const float3 localPos = TransformPoint(gScene.mInstanceInverseTransforms[info.mInstanceIndex], position);
		if (all(localPos >= info.mMin) && all(localPos <= info.mMax))
			return info.mInstanceIndex;
	}
	return INVALID_INSTANCE;
}
uint GetViewIndex(const uint2 index, const uint2 extent, const uint viewCount) {
	// assume views are stacked horizontally
    // return index.x / (extent.x / gPushConstants.mViewCount);
	return 0;
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

float SampleImage1(const uint imageIndex, const float2 uv, const float uvScreenSize = 0) {
	float lod = 0;
	if (uvScreenSize > 0) {
		float w, h;
		gScene.mImage1s[imageIndex].GetDimensions(w, h);
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
	}
	return gScene.mImage1s[NonUniformResourceIndex(imageIndex)].SampleLevel(gScene.mStaticSampler, uv, lod);
}
float2 SampleImage2(const uint imageIndex, const float2 uv, const float uvScreenSize = 0) {
	float lod = 0;
	if (uvScreenSize > 0) {
		float w, h;
		gScene.mImage2s[imageIndex].GetDimensions(w, h);
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
	}
	return gScene.mImage2s[NonUniformResourceIndex(imageIndex)].SampleLevel(gScene.mStaticSampler, uv, lod);
}
float4 SampleImage4(const uint imageIndex, const float2 uv, const float uvScreenSize = 0) {
	float lod = 0;
	if (uvScreenSize > 0) {
		float w, h;
		gScene.mImage4s[imageIndex].GetDimensions(w, h);
		lod = log2(max(uvScreenSize * max(w, h), 1e-6f));
	}
	return gScene.mImage4s[NonUniformResourceIndex(imageIndex)].SampleLevel(gScene.mStaticSampler, uv, lod);
}

Material LoadMaterial(const uint index, const float2 uv, const float uvScreenSize = 0) {
	Material m = gScene.mMaterials[index];

	if (m.mImages.GetBaseColor() < gImageCount) {
		m.BaseColor(m.GetBaseColor() * SampleImage4(m.mImages.GetBaseColor(), uv, uvScreenSize).rgb);
	}
	if (m.mImages.GetEmission() < gImageCount) {
		m.Emission(m.GetEmission() * SampleImage4(m.mImages.GetEmission(), uv, uvScreenSize).rgb);
	}
	if (m.mImages.GetPacked0() < gImageCount) {
		const float4 vi = SampleImage4(m.mImages.GetPacked0(), uv, uvScreenSize);
		m.Metallic(m.Metallic() * vi.x);
		m.Roughness(1 - (1 - m.Roughness()) * (1 - vi.y));
		m.Specular(m.Specular() * vi.x);
		m.Clearcoat(m.ClearCoat() * vi.z);

	}
	if (m.mImages.GetPacked1() < gImageCount) {
		const float4 vi = SampleImage4(m.mImages.GetPacked1(), uv, uvScreenSize);
		m.Sheen(m.Sheen() * vi.x);
		m.Transmission(m.Transmission() * vi.y);
		m.Subsurface(m.Subsurface() * vi.z);
		m.ClearcoatGloss(m.ClearcoatGloss() * vi.w);
	}
	return m;
}

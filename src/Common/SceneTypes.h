#pragma once

#include "PackedTypes.h"

#define INVALID_INSTANCE 0xFFFF
#define INVALID_PRIMITIVE 0xFFFF

#define BVH_FLAG_NONE 0
#define BVH_FLAG_TRIANGLES BIT(0)
#define BVH_FLAG_SPHERES BIT(1)
#define BVH_FLAG_VOLUME BIT(2)

PTVK_NAMESPACE_BEGIN

enum class InstanceType {
	eMesh,
	eSphere,
	eVolume
};


struct InstanceHeader {
	uint mHeader;

    inline InstanceType Type() CPP_CONST { return (InstanceType)BF_GET(mHeader, 0, 4); }
	inline uint MaterialIndex() CPP_CONST { return BF_GET(mHeader, 4, 28); }

	SLANG_CTOR(InstanceHeader)() { mHeader = 0; }

    SLANG_CTOR(InstanceHeader)(const InstanceType type, const uint materialIndex) {
		BF_SET(mHeader, (uint)type, 0, 4);
		BF_SET(mHeader, materialIndex, 4, 28);
	}
};

// All instances are 8 bytes

struct InstanceBase {
	InstanceHeader mHeader;
	uint pad;
};

struct MeshInstance {
	InstanceHeader mHeader;
	uint mData;

	inline uint VertexInfoIndex() CPP_CONST { return BF_GET(mData,  0, 16); }
	inline uint PrimitiveCount() CPP_CONST  { return BF_GET(mData, 16, 16); }

    SLANG_CTOR(MeshInstance)(const uint materialIndex, const uint vertexInfoIndex, const uint primitiveCount) {
        mHeader = InstanceHeader(InstanceType::eMesh, materialIndex);
		mData = 0;
		BF_SET(mData, vertexInfoIndex,  0, 16);
		BF_SET(mData, primitiveCount , 16, 16);
	}
};

struct SphereInstance {
	InstanceHeader mHeader;
	float mRadius;

    SLANG_CTOR(SphereInstance)(const uint materialIndex, const float radius) {
        mHeader = InstanceHeader(InstanceType::eSphere, materialIndex);
		mRadius = radius;
	}
};

struct VolumeInstance {
	InstanceHeader mHeader;
	uint mVolumeIndex;

    SLANG_CTOR(VolumeInstance)(const uint materialIndex, const uint volumeIndex) {
        mHeader = InstanceHeader(InstanceType::eVolume, materialIndex);
		mVolumeIndex = volumeIndex;
	}
};

struct VolumeInfo {
	float3 mMin;
	uint mInstanceIndex;
	float3 mMax;
	uint pad;
};

struct MeshVertexInfo {
	uint2 mPackedBufferIndices;
	uint mPackedStrides;
	uint pad;
	uint4 mPackedOffsets;

	inline uint GetIndexBuffer()    CPP_CONST { return BF_GET(mPackedBufferIndices[0],  0, 16); }
	inline uint GetIndexOffset()    CPP_CONST { return mPackedOffsets[0]; };
	inline uint GetIndexStride()    CPP_CONST { return BF_GET(mPackedStrides,  0, 8); }

	inline uint GetPositionBuffer() CPP_CONST { return BF_GET(mPackedBufferIndices[0], 16, 16); }
	inline uint GetPositionOffset() CPP_CONST { return mPackedOffsets[1]; };
	inline uint GetPositionStride() CPP_CONST { return BF_GET(mPackedStrides,  8, 8); }

	inline uint GetNormalBuffer()   CPP_CONST { return BF_GET(mPackedBufferIndices[1],  0, 16); }
	inline uint GetNormalOffset()   CPP_CONST { return mPackedOffsets[2]; };
	inline uint GetNormalStride()   CPP_CONST { return BF_GET(mPackedStrides, 16, 8); }

	inline uint GetTexcoordBuffer() CPP_CONST { return BF_GET(mPackedBufferIndices[1], 16, 16); }
	inline uint GetTexcoordOffset() CPP_CONST { return mPackedOffsets[3]; };
	inline uint GetTexcoordStride() CPP_CONST { return BF_GET(mPackedStrides, 24, 8); }

	SLANG_CTOR(MeshVertexInfo)(
		const uint indexBuffer   , const uint indexOffset   , const uint indexStride,
		const uint positionBuffer, const uint positionOffset, const uint positionStride,
		const uint normalBuffer  , const uint normalOffset  , const uint normalStride,
		const uint texcoordBuffer, const uint texcoordOffset, const uint texcoordStride) {
		BF_SET(mPackedBufferIndices[0], indexBuffer, 0, 16);
		mPackedOffsets[0] = indexOffset;
		BF_SET(mPackedStrides, indexStride, 0, 8);

		BF_SET(mPackedBufferIndices[0], positionBuffer, 16, 16);
		mPackedOffsets[1] = positionOffset;
		BF_SET(mPackedStrides, positionStride, 8, 8);

		BF_SET(mPackedBufferIndices[1], normalBuffer,  0, 16);
		mPackedOffsets[2] = normalOffset;
		BF_SET(mPackedStrides, normalStride, 16, 8);

		BF_SET(mPackedBufferIndices[1], texcoordBuffer, 16, 16);
		mPackedOffsets[3] = texcoordOffset;
		BF_SET(mPackedStrides, texcoordStride, 24, 8);
	}
};

PTVK_NAMESPACE_END
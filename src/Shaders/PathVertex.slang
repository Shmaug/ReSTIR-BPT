#pragma once

#include "Common/Math.h"
#include "Common/PackedTypes.h"
#include "Common/SceneTypes.h"

#include "Scene.slang"

#ifndef gShadingNormals
#define gShadingNormals false
#endif
#ifndef gNormalMaps
#define gNormalMaps false
#endif

// 56 bytes
struct PathVertex {
    float3 mPosition;
    uint mInstancePrimitiveIndex;

    uint mPackedGeometryNormal;
    uint mPackedShadingNormal;
    uint mPackedTangent;
    uint mFlags;

    PackedMaterialParameters mMaterial;

    property float3 mGeometryNormal {
        get { return UnpackNormal(mPackedGeometryNormal); }
        set { mPackedGeometryNormal = PackNormal(newValue); }
    }
    property float3 mShadingNormal {
        get { return UnpackNormal(mPackedShadingNormal); }
        set { mPackedShadingNormal = PackNormal(newValue); }
    }
    property float3 mTangent {
        get { return UnpackNormal(mPackedTangent); }
        set { mPackedTangent = PackNormal(newValue); }
    }

    property float mBitangentSign {
        get { return BF_GET(mFlags, 30, 1) == 0 ? 1 : -1; }
        set { BF_SET(mFlags, (uint)(newValue < 0), 30, 1); }
    }
    property bool mIsSurface {
        get { return (bool)BF_GET(mFlags, 31, 1); }
        set { BF_SET(mFlags, (uint)newValue, 31, 1); }
    }
    property float mPrimitiveArea {
        get { return f16tof32(BF_GET(mFlags, 0, 16)); }
        set { BF_SET(mFlags, f32tof16(newValue), 0, 16); }
    }

    property uint mInstanceIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 0, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 0, 16); }
	}
    property uint mPrimitiveIndex {
		get { return BF_GET(mInstancePrimitiveIndex, 16, 16); }
		set { BF_SET(mInstancePrimitiveIndex, newValue, 16, 16); }
	}

	// tangent space to world
    float3 ToWorld(float3 v) {
		const float3 n = mShadingNormal;
		const float3 t = mTangent;
        return v.x * t + v.y * cross(n, t) * mBitangentSign + v.z * n;
    }
	// world to tangent space
    float3 ToLocal(const float3 v) {
        const float3 n = mShadingNormal;
        const float3 t = mTangent;
        return float3(dot(v, t), dot(v, cross(n, t) * mBitangentSign), dot(v, n));
    }

    float GetShadingNormalCorrection(const float3 localDirIn, const float3 localDirOut, bool adjoint = false) {
        const float3 localGeometryNormal = ToLocal(mGeometryNormal);
        const float ngdotin = dot(localGeometryNormal, localDirIn);
        const float ngdotout = dot(localGeometryNormal, localDirOut);

        // light leak fix
        if (sign(ngdotout) != sign(localDirOut.z) || sign(ngdotin) != sign(localDirIn.z))
            return 0;

        float G = 1;

        if (adjoint) {
            const float num = ngdotout * localDirIn.z;
            const float denom = localDirOut.z * ngdotin;
            if (abs(denom) > 1e-5)
            G *= abs(num / denom);
        }

        return G;
    }

    __init() {
        mMaterial.BaseColor(0);
        mMaterial.Emission(0);
        mIsSurface = false;
	}

    [mutating]
	void InitMaterial(const uint materialIndex, const float2 uv) {
        GpuMaterial m = LoadMaterial(materialIndex);

		// sample images
		if (m.GetBaseColorImage() < gImageCount) {
			m.mParameters.BaseColor(m.mParameters.BaseColor() * SampleImage(m.GetBaseColorImage(), uv, 0).rgb);
		}
		if (m.GetEmissionImage() < gImageCount) {
			m.mParameters.Emission(m.mParameters.Emission() * SampleImage(m.GetEmissionImage(), uv, 0).rgb);
		}
		if (m.GetPackedParamsImage() < gImageCount) {
			const float4 packed = SampleImage(m.GetPackedParamsImage(), uv, 0);
			m.mParameters.Metallic(m.mParameters.Metallic() * packed.x);
			m.mParameters.Roughness(1 - (1 - m.mParameters.Roughness()) * (1 - packed.y));
			m.mParameters.Specular(m.mParameters.Specular() * packed.x);
			m.mParameters.Clearcoat(m.mParameters.Clearcoat() * packed.z);
		}

		// apply bump map
		if (gNormalMaps && m.GetBumpImage() < gImageCount && m.mParameters.BumpScale() > 0) {
			float3 bump;
			if (m.GetIsBumpTwoChannel()) {
				bump.xy = SampleImage2(m.GetBumpImage(), uv, 0);
				bump.z = 1;
			} else {
				bump = SampleImage(m.GetBumpImage(), uv, 0).rgb;
			}
			bump.xy = bump.xy * 2 - 1;
			bump.y = -bump.y;
			bump.xy *= m.mParameters.BumpScale();

			float3 n = mShadingNormal;
			float3 t = mTangent;

			mShadingNormal = normalize(t * bump.x + cross(n, t) * mBitangentSign * bump.y + n * bump.z);
			mTangent = normalize(t - n * dot(n, t));
		}

		mMaterial = m.mParameters;
	}

	// this is not meant to be used directly
    [mutating]
    void InitFromTriangle_(const uint materialIndex, const float4x4 transform, const MeshVertexInfo vertexInfo, const uint3 tri, const float2 bary, const float3 v0, const float3 v1, const float3 v2) {
        mIsSurface = true;
        mBitangentSign = 1;

		const float3 dPds = TransformVector(transform, v0 - v2);
		const float3 dPdt = TransformVector(transform, v1 - v2);
		float3 ng = cross(dPds, dPdt);
        const float area2 = length(ng);
        ng /= area2;
        mPrimitiveArea = area2/2;
		mGeometryNormal = ng;

		float2 t0, t1, t2;
		if (vertexInfo.GetTexcoordBuffer() < gVertexBufferCount)
			LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetTexcoordBuffer())], vertexInfo.GetTexcoordOffset(), vertexInfo.GetTexcoordStride(), tri, t0, t1, t2);
		else
			t0 = t1 = t2 = 0;

		const float2 duvds = t2 - t0;
		const float2 duvdt = t2 - t1;

		const float2 uv = t0 + (t1 - t0) * bary.x + duvds * bary.y;

		// [du/ds, du/dt]
		// [dv/ds, dv/dt]
		// The inverse of this matrix is
		// (1/det) [ dv/dt, -du/dt]
		//         [-dv/ds,  du/ds]
		// where det = duds * dvdt - dudt * dvds
		const float det = duvds[0] * duvdt[1] - duvdt[0] * duvds[1];
		const float invDet = 1 / det;
		const float dsdu = duvdt[1] * invDet;
		const float dtdu = -duvds[1] * invDet;
		const float dsdv = duvdt[0] * invDet;
		const float dtdv = -duvds[0] * invDet;
		float3 dPdu, dPdv;
		if (det != 0) {
			// Now we just need to do the matrix multiplication
			dPdu = -(dPds * dsdu + dPdt * dtdu);
			dPdv = -(dPds * dsdv + dPdt * dtdv);
		} else {
			const float3x3 onb = MakeOrthonormal(ng);
			dPdu = onb[0];
			dPdv = onb[1];
		}

		bool shadingNormalValid = false;
		float3 shadingNormal;
		float3 n0, n1, n2;
		if (gShadingNormals && vertexInfo.GetNormalBuffer() < gVertexBufferCount) {
			LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetNormalBuffer())], vertexInfo.GetNormalOffset(), vertexInfo.GetNormalStride(), tri, n0, n1, n2);

			shadingNormal = n0 + (n1 - n0) * bary.x + (n2 - n0) * bary.y;
			shadingNormalValid = !(all(shadingNormal.xyz == 0) || any(isnan(shadingNormal)));
		}

		if (!shadingNormalValid) {
			mPackedShadingNormal = mPackedGeometryNormal;
			mTangent = normalize(dPdu);
		} else {
			shadingNormal = normalize(TransformVector(transform, shadingNormal));
			const float3 tangent = normalize(dPdu - shadingNormal * dot(shadingNormal, dPdu));
			mShadingNormal = shadingNormal;
			mTangent = tangent;

			// force geometry normal to agree with shading normal
			if (dot(shadingNormal, ng) < 0)
				mGeometryNormal = -ng;
		}

        InitMaterial(materialIndex, uv);
    }

    [mutating]
    void InitFromTriangle(const MeshInstance instance, const float4x4 transform, const uint primitiveIndex, const float2 bary) {
        mPrimitiveIndex = primitiveIndex;

        const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.VertexInfoIndex()];
		const uint3 tri = LoadTriangleIndices(vertexInfo, primitiveIndex);

		float3 v0, v1, v2;
		LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetPositionBuffer())], vertexInfo.GetPositionOffset(), vertexInfo.GetPositionStride(), tri, v0, v1, v2);

        InitFromTriangle_(instance.mHeader.MaterialIndex(), transform, vertexInfo, tri, bary, v0, v1, v2);
		mPosition = TransformPoint(transform, v0 + (v1 - v0) * bary.x + (v2 - v0) * bary.y);
    }
    [mutating]
    void InitFromTriangle(const MeshInstance instance, const float4x4 transform, const uint primitiveIndex, const float3 localPosition) {
        mPrimitiveIndex = primitiveIndex;

		const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.VertexInfoIndex()];
		const uint3 tri = LoadTriangleIndices(vertexInfo, primitiveIndex);

		float3 v0, v1, v2;
		LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetPositionBuffer())], vertexInfo.GetPositionOffset(), vertexInfo.GetPositionStride(), tri, v0, v1, v2);

		// compute barycentrics from localPosition
		const float3 v1v0 = v1 - v0;
		const float3 v2v0 = v2 - v0;
		const float3 p_v0 = localPosition - v0;
		const float d00 = dot(v1v0, v1v0);
		const float d01 = dot(v1v0, v2v0);
		const float d11 = dot(v2v0, v2v0);
		const float d20 = dot(p_v0, v1v0);
		const float d21 = dot(p_v0, v2v0);
		const float2 bary = float2(d11 * d20 - d01 * d21, d00 * d21 - d01 * d20) / (d00 * d11 - d01 * d01);

        InitFromTriangle_(instance.mHeader.MaterialIndex(), transform, vertexInfo, tri, bary, v0, v1, v2);
		mPosition = TransformPoint(transform, localPosition);
    }

    [mutating]
    void InitFromSphere(const SphereInstance instance, const float4x4 transform, const float3 localPosition) {
        mPrimitiveIndex = INVALID_PRIMITIVE;

        const float3 normal = normalize(TransformVector(transform, localPosition));
        mPosition = TransformPoint(transform, localPosition);
        mIsSurface = true;
        mBitangentSign = 1;
        mPackedGeometryNormal = mPackedShadingNormal = PackNormal(normal);
        mPrimitiveArea = 4 * M_PI * instance.mRadius * instance.mRadius;

		const float2 uv = CartesianToSphericalUV(normalize(localPosition));
        const float3 dpdu = TransformVector(transform, float3(-sin(uv[0]) * sin(uv[1]), 0, cos(uv[0]) * sin(uv[1])));
        const float3 dpdv = TransformVector(transform, float3(cos(uv[0]) * cos(uv[1]), -sin(uv[1]), sin(uv[0]) * cos(uv[1])));
        mTangent = normalize(dpdu - normal * dot(normal, dpdu));

        InitMaterial(instance.mHeader.MaterialIndex(), uv);
    }


    [mutating]
    void InitFromBackground(const float3 direction, const float3 le) {
        mPosition = POS_INFINITY;
        mInstanceIndex = INVALID_INSTANCE;
        mPrimitiveIndex = INVALID_PRIMITIVE;
        mIsSurface = false;
        mPackedShadingNormal = mPackedGeometryNormal = PackNormal(direction);
        mMaterial.BaseColor(0);
        mMaterial.Emission(le);
	}

	[mutating]
    void InitFromPosition(const float3 localPosition, const uint instanceIndex, const uint primitiveIndex = -1) {
        mInstanceIndex = instanceIndex;
        const InstanceBase instance = gScene.mInstances[instanceIndex];
        const float4x4 transform = gScene.mInstanceTransforms[instanceIndex];
        switch (instance.mHeader.Type()) {
        case InstanceType::eMesh:
            InitFromTriangle(reinterpret<MeshInstance>(instance), transform, primitiveIndex, localPosition);
			break;
        default:
        case InstanceType::eSphere:
            InitFromSphere(reinterpret<SphereInstance>(instance), transform, localPosition);
			break;
		/*
        case InstanceType::eVolume:
            InitFromVolume(reinterpret<VolumeInstance>(instance), TransformPoint(transform, localPosition));
			break;
		*/
        }
    }
};

PackedVertex PackVertex(const PathVertex v) {
    PackedVertex packed;
	if (v.mIsSurface)
    	packed.mLocalPosition = TransformPoint(gScene.mInstanceInverseTransforms[v.mInstanceIndex], v.mPosition);
    else
        packed.mLocalPosition = POS_INFINITY;
    packed.mInstancePrimitiveIndex = v.mInstancePrimitiveIndex;
    return packed;
}

PathVertex UnpackVertex(const PackedVertex packed) {
	PathVertex v;
    if (packed.mInstanceIndex == INVALID_INSTANCE)
        v.InitFromBackground(packed.mLocalPosition, 0);
	else
        v.InitFromPosition(packed.mLocalPosition, packed.mInstanceIndex, packed.mPrimitiveIndex);
    return v;
}

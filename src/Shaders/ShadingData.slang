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

// 48 bytes
struct ShadingData {
    float3 mPosition;
    uint mFlagsMaterialIndex;

    uint mPackedGeometryNormal;
    uint mPackedShadingNormal;
    uint mPackedTangent;
    float mShapeArea;

    float2 mTexcoord;
    float mTexcoordScreenSize;
    float mMeanCurvature;

    bool IsSurface()           { return BF_GET(mFlagsMaterialIndex, 0, 1); }
    bool IsBitangentFlipped()  { return BF_GET(mFlagsMaterialIndex, 1, 1); }
    uint GetMaterialIndex()    { return BF_GET(mFlagsMaterialIndex, 4, 28); }
    float3 GetGeometryNormal() { return UnpackNormal(mPackedGeometryNormal); }
    float3 GetShadingNormal()  { return UnpackNormal(mPackedShadingNormal); }
    float3 GetTangent()        { return UnpackNormal(mPackedTangent); }

    [mutating] void IsSurface         (bool newValue)   { BF_SET(mFlagsMaterialIndex, newValue, 0, 1); }
    [mutating] void IsBitangentFlipped(bool newValue)   { BF_SET(mFlagsMaterialIndex, newValue, 1, 1); }
    [mutating] void SetMaterialIndex  (uint newValue)   { BF_SET(mFlagsMaterialIndex, newValue, 4, 28); }
    [mutating] void SetGeometryNormal (float3 newValue) { mPackedGeometryNormal = PackNormal(newValue); }
    [mutating] void SetShadingNormal  (float3 newValue) { mPackedShadingNormal  = PackNormal(newValue); }
    [mutating] void SetTangent        (float3 newValue) { mPackedTangent        = PackNormal(newValue); }

    float3 ToWorld(float3 v) {
		float3 n = GetShadingNormal();
		float3 t = GetTangent();
		return v.x * t + v.y * cross(n, t) + v.z * n;
    }
    float3 ToLocal(const float3 v) {
		float3 n = GetShadingNormal();
		float3 t = GetTangent();
		return float3(dot(v, t), dot(v, cross(n, t) * (IsBitangentFlipped() ? -1 : 1)), dot(v, n));
    }

    float GetShadingNormalCorrection(const float3 localDirIn, const float3 localDirOut, bool adjoint = false) {
        const float3 localGeometryNormal = ToLocal(GetGeometryNormal());
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

    // doesn't assign mPosition
    static ShadingData LoadWithoutPosition(const uint materialIndex, const float4x4 transform, const MeshVertexInfo vertexInfo, const uint3 tri, const float2 bary, const float3 v0, const float3 v1, const float3 v2) {
		ShadingData r;
		r.IsSurface(true);
		r.IsBitangentFlipped(false);
		r.SetMaterialIndex(materialIndex);

		const float3 dPds = TransformVector(transform, v0 - v2);
		const float3 dPdt = TransformVector(transform, v1 - v2);
		float3 geometryNormal = cross(dPds, dPdt);
		const float area2 = length(geometryNormal);
		geometryNormal /= area2;
		r.SetGeometryNormal(geometryNormal);
		r.mShapeArea = area2 / 2;

		float2 t0, t1, t2;
		if (vertexInfo.GetTexcoordBuffer() < gVertexBufferCount)
			LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetTexcoordBuffer())], vertexInfo.GetTexcoordOffset(), vertexInfo.GetTexcoordStride(), tri, t0, t1, t2);
		else
			t0 = t1 = t2 = 0;

		const float2 duvds = t2 - t0;
		const float2 duvdt = t2 - t1;

		r.mTexcoord = t0 + (t1 - t0) * bary.x + duvds * bary.y;

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
			r.mTexcoordScreenSize = 1 / max(length(dPdu), length(dPdv));
		} else {
			const float3x3 onb = MakeOrthonormal(geometryNormal);
			dPdu = onb[0];
			dPdv = onb[1];
			r.mTexcoordScreenSize = 1;
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
			r.mPackedShadingNormal = r.mPackedGeometryNormal;
			r.SetTangent(normalize(dPdu));
			r.mMeanCurvature = 0;
		} else {
			shadingNormal = normalize(TransformVector(transform, shadingNormal));
			const float3 tangent = normalize(dPdu - shadingNormal * dot(shadingNormal, dPdu));
			r.SetShadingNormal(shadingNormal);
			r.SetTangent(tangent);

			// force geometry normal to agree with shading normal
			if (dot(shadingNormal, geometryNormal) < 0)
			r.SetGeometryNormal(-geometryNormal);

			const float3 dNds = n2 - n0;
			const float3 dNdt = n2 - n1;
			const float3 dNdu = dNds * dsdu + dNdt * dtdu;
			const float3 dNdv = dNds * dsdv + dNdt * dtdv;
			const float3 bitangent = normalize(cross(shadingNormal, tangent));
			r.mMeanCurvature = (dot(dNdu, tangent) + dot(dNdv, bitangent)) / 2;
		}
		return r;
    }

    static ShadingData Load(const MeshInstance instance, const float4x4 transform, const uint primitiveIndex, const float2 bary) {
		const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.VertexInfoIndex()];
		const uint3 tri = LoadTriangleIndices(vertexInfo, primitiveIndex);

		float3 v0, v1, v2;
		LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetPositionBuffer())], vertexInfo.GetPositionOffset(), vertexInfo.GetPositionStride(), tri, v0, v1, v2);

		ShadingData r = LoadWithoutPosition(instance.mHeader.MaterialIndex(), transform, vertexInfo, tri, bary, v0, v1, v2);
		r.mPosition = TransformPoint(transform, v0 + (v1 - v0) * bary.x + (v2 - v0) * bary.y);
		return r;
    }
    static ShadingData Load(const MeshInstance instance, const float4x4 transform, const uint primitiveIndex, const float3 localPosition) {
		const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.VertexInfoIndex()];
		const uint3 tri = LoadTriangleIndices(vertexInfo, primitiveIndex);

		float3 v0, v1, v2;
		LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetPositionBuffer())], vertexInfo.GetPositionOffset(), vertexInfo.GetPositionStride(), tri, v0, v1, v2);

		const float3 v1v0 = v1 - v0;
		const float3 v2v0 = v2 - v0;
		const float3 p_v0 = localPosition - v0;
		const float d00 = dot(v1v0, v1v0);
		const float d01 = dot(v1v0, v2v0);
		const float d11 = dot(v2v0, v2v0);
		const float d20 = dot(p_v0, v1v0);
		const float d21 = dot(p_v0, v2v0);
		const float2 bary = float2(d11 * d20 - d01 * d21, d00 * d21 - d01 * d20) / (d00 * d11 - d01 * d01);

		ShadingData r = LoadWithoutPosition(instance.mHeader.MaterialIndex(), transform, vertexInfo, tri, bary, v0, v1, v2);
		r.mPosition = TransformPoint(transform, localPosition);
		return r;
    }
    static ShadingData Load(const SphereInstance instance, const float4x4 transform, const float3 localPosition) {
		ShadingData r;
		const float3 normal = normalize(TransformVector(transform, localPosition));
		r.mPosition = TransformPoint(transform, localPosition);
		r.IsSurface(true);
		r.IsBitangentFlipped(false);
		r.SetMaterialIndex(instance.mHeader.MaterialIndex());
		r.mPackedGeometryNormal = r.mPackedShadingNormal = PackNormal(normal);
		r.mShapeArea = 4 * M_PI * instance.mRadius * instance.mRadius;
		r.mMeanCurvature = 1 / instance.mRadius;
		r.mTexcoord = CartesianToSphericalUV(normalize(localPosition));
		const float3 dpdu = TransformVector(transform, float3(-sin(r.mTexcoord[0]) * sin(r.mTexcoord[1]), 0, cos(r.mTexcoord[0]) * sin(r.mTexcoord[1])));
		const float3 dpdv = TransformVector(transform, float3( cos(r.mTexcoord[0]) * cos(r.mTexcoord[1]), -sin(r.mTexcoord[1]), sin(r.mTexcoord[0]) * cos(r.mTexcoord[1])));
		r.SetTangent(normalize(dpdu - normal * dot(normal, dpdu)));
		r.mTexcoordScreenSize = 1 / max(length(dpdu), length(dpdv));
		return r;
    }
    static ShadingData Load(const VolumeInstance instance, const float3 position) {
		ShadingData r;
		r.mPosition = position;
		r.IsSurface(false);
		r.IsBitangentFlipped(false);
		r.SetMaterialIndex(instance.mHeader.MaterialIndex());
		r.mShapeArea = 0;
		r.mTexcoordScreenSize = 0;
		return r;
    }
    static ShadingData Load(const InstanceBase instance, const float4x4 transform, const float3 localPosition, const uint primitiveIndex = -1) {
		switch (instance.mHeader.Type()) {
		case InstanceType::eMesh:
			return Load(reinterpret<MeshInstance>(instance), transform, primitiveIndex, localPosition);
		default:
		case InstanceType::eSphere:
			return Load(reinterpret<SphereInstance>(instance), transform, localPosition);
		case InstanceType::eVolume:
			return Load(reinterpret<VolumeInstance>(instance), TransformPoint(transform, localPosition));
		}
    }
};

void SampleNormalMap(const GpuMaterial m, inout ShadingData sd) {
	if (!gNormalMaps) return;

    if (m.GetBumpImage() >= gImageCount) return;
    const float scale = m.BumpScale();
	if (scale <= 0) return;

    float3 bump;
    if (m.GetIsBumpTwoChannel()) {
        bump.xy = SampleImage2(m.GetBumpImage(), sd.mTexcoord, sd.mTexcoordScreenSize);
		bump.z = 1;
    } else {
        bump = SampleImage(m.GetBumpImage(), sd.mTexcoord, sd.mTexcoordScreenSize).rgb;
	}
	bump.xy = bump.xy * 2 - 1;
	bump.y = -bump.y;
	bump.xy *= scale;

	float3 n = sd.GetShadingNormal();
	float3 t = sd.GetTangent();

	n = normalize(t * bump.x + cross(n, t) * (sd.IsBitangentFlipped() ? -1 : 1) * bump.y + n * bump.z);
	t = normalize(t - n * dot(n, t));

	sd.SetShadingNormal(n);
	sd.SetTangent(t);
}
GpuMaterial LoadMaterial(inout ShadingData sd) {
    GpuMaterial m = gScene.mMaterials[sd.GetMaterialIndex()];
    SampleMaterialImages(m, sd.mTexcoord, 0);
    SampleNormalMap(m, sd);
    return m;
}


float3 OffsetRayOrigin(const float3 P, const float3 Ng) {
	// This function should be used to compute a modified ray start position for
	// rays leaving from a surface. This is from "A Fast and Robust Method for Avoiding
	// Self-Intersection" see https://research.nvidia.com/publication/2019-03_A-Fast-and
  float int_scale = 256.0;
  int3 of_i = int3(int_scale * Ng);

  float origin = 1.0 / 32.0;
  float float_scale = 1.0 / 65536.0;
  return float3(abs(P.x) < origin ? P.x + float_scale * Ng.x : asfloat(asint(P.x) + ((P.x < 0.0) ? -of_i.x : of_i.x)),
                abs(P.y) < origin ? P.y + float_scale * Ng.y : asfloat(asint(P.y) + ((P.y < 0.0) ? -of_i.y : of_i.y)),
                abs(P.z) < origin ? P.z + float_scale * Ng.z : asfloat(asint(P.z) + ((P.z < 0.0) ? -of_i.z : of_i.z)));
}
float3 OffsetRayOrigin(const float3 P, const float3 Ng, const float3 dir) {
  return OffsetRayOrigin(P, dot(Ng, dir) < 0 ? -Ng : Ng);
}
float3 OffsetRayOrigin(const ShadingData sd, const float3 dir) {
    return sd.IsSurface() ? OffsetRayOrigin(sd.mPosition, sd.GetGeometryNormal(), dir) : sd.mPosition;
}
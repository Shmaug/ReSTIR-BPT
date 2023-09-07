#pragma once

#include "ShadingData.slang"
#include "DebugCounters.slang"

#ifndef gAlphaTest
#define gAlphaTest false
#endif

RayDesc MakeRay(const float3 origin, const float3 direction, const float tmin = 0, const float tmax = POS_INFINITY) {
	RayDesc ray;
	ray.Origin = origin;
	ray.TMin = tmin;
	ray.Direction = direction;
	ray.TMax = tmax;
	return ray;
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


struct IntersectionResult {
	float mLightPdf;
    float mDistance;
};

bool TraceRay(const RayDesc ray, const bool closest, out ShadingData sd, out IntersectionResult isect) {
    IncrementCounter(DebugCounterType::eRays);

	// trace ray

    RayQuery<RAY_FLAG_NONE> rayQuery;
    rayQuery.TraceRayInline(gScene.mAccelerationStructure, closest ? RAY_FLAG_NONE : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, ray);
	while (rayQuery.Proceed()) {
		const uint instanceIndex = rayQuery.CandidateInstanceID();
		switch (rayQuery.CandidateType()) {
			case CANDIDATE_NON_OPAQUE_TRIANGLE: {
				if (!gAlphaTest) {
					rayQuery.CommitNonOpaqueTriangleHit();
					break;
				}

				const MeshInstance instance = reinterpret<MeshInstance>(gScene.mInstances[instanceIndex]);
                uint img = -1;
                float cutoff = 0;
                {
                    const GpuMaterial m = LoadMaterial(instance.mHeader.MaterialIndex());
                    cutoff = m.mParameters.AlphaCutoff();
                    img = m.GetBaseColorImage();
				}

                if (img >= gImageCount) {
					rayQuery.CommitNonOpaqueTriangleHit();
					break;
                }

                const MeshVertexInfo vertexInfo = gScene.mMeshVertexInfo[instance.VertexInfoIndex()];
                const uint3 tri = LoadTriangleIndices(vertexInfo, rayQuery.CandidatePrimitiveIndex());

                float2 v0, v1, v2;
                LoadTriangleAttribute(gScene.mVertexBuffers[NonUniformResourceIndex(vertexInfo.GetTexcoordBuffer())], vertexInfo.GetTexcoordOffset(), vertexInfo.GetTexcoordStride(), tri, v0, v1, v2);
                const float2 bary = rayQuery.CandidateTriangleBarycentrics();
                const float2 uv = v0 + (v1 - v0) * bary.x + (v2 - v0) * bary.y;

                if (SampleImage(img, uv).a >= cutoff)
					rayQuery.CommitNonOpaqueTriangleHit();
				break;
			}
			case CANDIDATE_PROCEDURAL_PRIMITIVE: {
				const InstanceBase instance = gScene.mInstances[instanceIndex];
				switch (instance.mHeader.Type()) {
					case InstanceType::eSphere: {
						const float2 st = RaySphere(rayQuery.CandidateObjectRayOrigin(), rayQuery.CandidateObjectRayDirection(), 0, reinterpret<SphereInstance>(instance).mRadius);
						if (st.x < st.y) {
							const float t = st.x > rayQuery.RayTMin() ? st.x : st.y;
							if (t < rayQuery.CommittedRayT() && t > rayQuery.RayTMin())
								rayQuery.CommitProceduralPrimitiveHit(t);
						}
						break;
					}
					/*
					case InstanceType::eVolume: {
						float3 origin,direction, bbox_min, bbox_max;
						const uint volumeIndex = reinterpret<VolumeInstance>(instance).volumeIndex();
						if (volumeIndex == -1) {
							origin = rayQuery.CandidateObjectRayOrigin();
							direction = rayQuery.CandidateObjectRayDirection();
							bbox_min = -1;
							bbox_max = 1;
						} else {
							pnanovdb_buf_t volumeBuffer = mVolumes[NonUniformResourceIndex(volumeIndex)];
							pnanovdb_grid_handle_t gridHandle = {0};
							pnanovdb_root_handle_t root = pnanovdb_tree_get_root(volumeBuffer, pnanovdb_grid_get_tree(volumeBuffer, gridHandle));
							origin    = pnanovdb_grid_world_to_indexf    (volumeBuffer, gridHandle, rayQuery.CandidateObjectRayOrigin());
							direction = pnanovdb_grid_world_to_index_dirf(volumeBuffer, gridHandle, rayQuery.CandidateObjectRayDirection());
							bbox_min = pnanovdb_root_get_bbox_min(volumeBuffer, root);
							bbox_max = pnanovdb_root_get_bbox_max(volumeBuffer, root) + 1;
						}
						const float3 t0 = (bbox_min - origin) / direction;
						const float3 t1 = (bbox_max - origin) / direction;
						const float2 st = float2(max3(min(t0, t1)), min3(max(t0, t1)));
						const float t = st.x > rayQuery.RayTMin() ? st.x : st.y;
						if (t < rayQuery.CommittedRayT() && t > rayQuery.RayTMin()) {
							float3 localNormal = float3(t1 == t) - float3(t0 == t);
							if (volumeIndex != -1)
								localNormal = pnanovdb_grid_index_to_world_dirf(mVolumes[NonUniformResourceIndex(volumeIndex)], {0}, localNormal);
							const float3 normal = normalize(mInstanceTransforms[instanceIndex].transformVector(localNormal));
							isect.mShadingData.mPackedGeometryNormal = isect.mShadingData.mPackedShadingNormal = packNormal(normal);
							rayQuery.CommitProceduralPrimitiveHit(t);
						}
						break;
					}
					*/
				}
			}
		}
    }

    // create IntersectionResult
    uint status = rayQuery.CommittedStatus();
	if (status == COMMITTED_NOTHING) {
        // ray missed scene
        sd.mPosition = POS_INFINITY;
        sd.SetInstanceIndex(INVALID_INSTANCE);
        sd.SetPrimitiveIndex(INVALID_PRIMITIVE);
        sd.IsSurface(false);
		isect.mDistance = ray.TMax;
        if (gScene.HasBackground())
			isect.mLightPdf = gScene.mLightCount == 0 ? 1 : gScene.mBackgroundSampleProbability;
		else
			isect.mLightPdf = 0;
		return false;
	}

	isect.mDistance = rayQuery.CommittedRayT();

    const uint instanceIndex = rayQuery.CommittedInstanceID();
    const InstanceBase instance = gScene.mInstances[instanceIndex];
    const float4x4 transform = gScene.mInstanceTransforms[instanceIndex];
    float primCount = 1;

	if (status == COMMITTED_TRIANGLE_HIT) {
		const MeshInstance meshInstance = reinterpret<MeshInstance>(instance);
        primCount = meshInstance.PrimitiveCount();
        sd = ShadingData::Load(meshInstance, transform, rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
		sd.SetPrimitiveIndex(rayQuery.CommittedPrimitiveIndex());
	} else if (status == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
		switch (instance.mHeader.Type()) {
			case InstanceType::eSphere:
				sd = ShadingData::Load(reinterpret<SphereInstance>(instance), transform, rayQuery.CommittedObjectRayOrigin() + rayQuery.CommittedObjectRayDirection() * rayQuery.CommittedRayT());
				break;
			/*
			case InstanceType::eVolume: {
				const uint n = sd.mPackedGeometryNormal; // assigned in the rayQuery loop above
				sd = ShadingData(reinterpret<VolumeInstance>(instance), ray.Origin + ray.Direction * rayQuery.CommittedRayT());
				sd.mPackedGeometryNormal = n;
				break;
			}
			*/
		}
    	sd.SetPrimitiveIndex(INVALID_PRIMITIVE);
	}
    sd.SetInstanceIndex(rayQuery.CommittedInstanceID());

    if (gScene.mLightCount > 0) {
        isect.mLightPdf = 1 / (primCount * sd.mShapeArea * gScene.mLightCount);
        if (gScene.HasBackground())
			isect.mLightPdf *= 1 - gScene.mBackgroundSampleProbability;
	} else
		isect.mLightPdf = 0;

	return true;
}

bool Occluded(const ShadingData origin, const float3 dir, const float dist) {
    IncrementCounter(DebugCounterType::eShadowRays);

    ShadingData tmpSd;
    IntersectionResult tmpIsect;
    return TraceRay(MakeRay(OffsetRayOrigin(origin, dir), dir, 0, dist*(1.0 - 1.0/512.0)), true, tmpSd, tmpIsect);
}
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

struct IntersectionResult {
	ShadingData mShadingData;
    uint mInstanceIndex;
    uint mPrimitiveIndex;
	float mLightPdf;
	float mDistance;

	InstanceBase GetInstance  () { return gScene.mInstances         [mInstanceIndex]; }
	float4x4     GetTransform () { return gScene.mInstanceTransforms[mInstanceIndex]; }
	uint         GetLightIndex() { return gScene.mInstanceLightMap  [mInstanceIndex]; }
};

bool TraceRay(const RayDesc ray, const bool closest, out IntersectionResult isect) {
    IncrementCounter(DebugCounterType::eRays);

	// trace ray

	RayQuery<RAY_FLAG_NONE> rayQuery;
	rayQuery.TraceRayInline(gScene.mAccelerationStructure, closest ? RAY_FLAG_NONE : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, ray);
	while (rayQuery.Proceed()) {
		const uint instanceIndex = rayQuery.CandidateInstanceID();
		switch (rayQuery.CandidateType()) {
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
			case CANDIDATE_NON_OPAQUE_TRIANGLE: {
				if (!gAlphaTest) {
					rayQuery.CommitNonOpaqueTriangleHit();
					break;
				}

				const MeshInstance instance = reinterpret<MeshInstance>(gScene.mInstances[instanceIndex]);
				const GpuMaterial m = gScene.mMaterials[instance.mHeader.MaterialIndex()];

				if (m.GetBaseColorImage() >= gImageCount) {
					rayQuery.CommitNonOpaqueTriangleHit();
					break;
				}

				// load ShadingData for mTexcoord
                const ShadingData sd = ShadingData::Load(instance, float4x4(1), rayQuery.CandidatePrimitiveIndex(), rayQuery.CandidateTriangleBarycentrics());

				if (SampleImage(m.GetBaseColorImage(), sd.mTexcoord).a >= m.AlphaCutoff())
					rayQuery.CommitNonOpaqueTriangleHit();
				break;
			}
		}
    }

    // create IntersectionResult
    uint status = rayQuery.CommittedStatus();
	if (status == COMMITTED_NOTHING) {
		// ray missed scene
		isect.mInstanceIndex = INVALID_INSTANCE;
		isect.mDistance = ray.TMax;
        isect.mShadingData.mPosition = POS_INFINITY;
        isect.mShadingData.IsSurface(false);
        if (gScene.HasBackground())
			isect.mLightPdf = gScene.mLightCount == 0 ? 1 : gScene.mBackgroundSampleProbability;
		else
			isect.mLightPdf = 0;
		return false;
	}

	isect.mInstanceIndex = rayQuery.CommittedInstanceID();
	isect.mDistance      = rayQuery.CommittedRayT();

    const InstanceBase instance = isect.GetInstance();
    float primCount = 1;

	if (status == COMMITTED_TRIANGLE_HIT) {
		isect.mPrimitiveIndex = rayQuery.CommittedPrimitiveIndex();
        MeshInstance meshInstance = reinterpret<MeshInstance>(instance);
        primCount = meshInstance.PrimitiveCount();
		isect.mShadingData = ShadingData::Load(meshInstance, isect.GetTransform(), rayQuery.CommittedPrimitiveIndex(), rayQuery.CommittedTriangleBarycentrics());
	} else if (status == COMMITTED_PROCEDURAL_PRIMITIVE_HIT) {
		isect.mPrimitiveIndex = INVALID_PRIMITIVE;
		switch (instance.mHeader.Type()) {
			case InstanceType::eSphere:
				isect.mShadingData = ShadingData::Load(reinterpret<SphereInstance>(instance), isect.GetTransform(), rayQuery.CommittedObjectRayOrigin() + rayQuery.CommittedObjectRayDirection() * rayQuery.CommittedRayT());
				break;
			/*
			case InstanceType::eVolume: {
				const uint n = isect.mShadingData.mPackedGeometryNormal; // assigned in the rayQuery loop above
				isect.mShadingData = ShadingData(reinterpret<VolumeInstance>(instance), ray.Origin + ray.Direction * rayQuery.CommittedRayT());
				isect.mShadingData.mPackedGeometryNormal = n;
				break;
			}
			*/
		}
	}

    if (gScene.mLightCount == 0) {
        isect.mLightPdf = 1 / (primCount * isect.mShadingData.mShapeArea * gScene.mLightCount);
        if (gScene.HasBackground())
			isect.mLightPdf *= 1 - gScene.mBackgroundSampleProbability;
	} else
		isect.mLightPdf = 0;

	return true;
}

ShadingData TraceRay(const RayDesc ray) {
    IntersectionResult r;
    TraceRay(ray, true, r);
    return r.mShadingData;
}

bool Occluded(const ShadingData origin, const float3 dir, const float dist) {
    IncrementCounter(DebugCounterType::eShadowRays);

    IntersectionResult tmp;
    return TraceRay(MakeRay(OffsetRayOrigin(origin, dir), dir, 0, dist*(1.0 - 1.0/512.0)), true, tmp);
}
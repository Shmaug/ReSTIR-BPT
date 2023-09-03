#include "ShadingData.slang"

struct EmissionSampleRecord {
    ShadingData mShadingData;
    float mPdf; // area-measure except for background samples
    bool isSingular; // unused (only area lights are implemented)
    uint mInstanceIndex;
    uint mPrimitiveIndex;
};

struct IlluminationSampleRecord {
    float3 mRadiance;
    float mPdf; // area-measure except for environment lights
    float3 mDirectionToLight;
    float mDistanceToLight;
    float3 mPosition;
    uint mPackedNormal;

    float mCosLight;
    bool isFinite;
    bool isSingular; // unused (only area lights are implemented)

    float3 GetNormal() { return UnpackNormal(mPackedNormal); }
};

// uniformly samples a light instance and primitive index, then uniformly samples the primitive's area
EmissionSampleRecord SampleEmission(float4 rnd) {
	EmissionSampleRecord r;
	r.isSingular = false;

    if (gScene.HasBackground()) {
		if (gScene.mLightCount == 0 || rnd.w < gScene.mBackgroundSampleProbability) {
			// sample environment light
			r.mShadingData.mShapeArea = -1;
			r.mInstanceIndex = INVALID_INSTANCE;

			if (gScene.mBackgroundImageIndex < gImageCount) {
                r.mShadingData.mPosition = SphericalUVToCartesian(SampleTexel(gScene.mImage4s[gScene.mBackgroundImageIndex], rnd.xy, r.mPdf));
                // jacobian from SphericalUVToCartesian
				r.mPdf /= (2 * M_PI * M_PI * sqrt(1 - sqr(r.mShadingData.mPosition.y)));
			} else {
				r.mShadingData.mPosition = SampleUniformSphere(rnd.xy);
				r.mPdf = 1 / (4 * M_PI);
			}

			if (gScene.mLightCount > 0)
				r.mPdf *= gScene.mBackgroundSampleProbability;
			return r;
		}

		// remap rnd.w from [mBackgroundSampleProbability,1] to [0,1]
        if (gScene.mLightCount > 0)
            rnd.w = (rnd.w - gScene.mBackgroundSampleProbability) / (1 - gScene.mBackgroundSampleProbability);
	}

	if (gScene.mLightCount == 0)
		return r;

	r.mInstanceIndex = gScene.mLightInstanceMap[uint(rnd.z * gScene.mLightCount) % gScene.mLightCount];
	r.mPdf = 1 / (float)gScene.mLightCount;

    if (gScene.HasBackground())
        r.mPdf *= 1 - gScene.mBackgroundSampleProbability;

	const InstanceBase instance  = gScene.mInstances[r.mInstanceIndex];
	const float4x4 transform = gScene.mInstanceTransforms[r.mInstanceIndex];

	if (instance.mHeader.Type() == InstanceType::eMesh) {
		// triangle
		const MeshInstance mesh = reinterpret<MeshInstance>(instance);
		r.mPrimitiveIndex = uint(rnd.w * mesh.PrimitiveCount()) % mesh.PrimitiveCount();
		r.mPdf /= (float)mesh.PrimitiveCount();
		r.mShadingData = ShadingData::Load(mesh, transform, r.mPrimitiveIndex, SampleUniformTriangle(rnd.xy));
	} else if (instance.mHeader.Type() == InstanceType::eSphere) {
		// sphere
		r.mPrimitiveIndex = INVALID_PRIMITIVE;
		const SphereInstance sphere = reinterpret<SphereInstance>(instance);
		r.mShadingData = ShadingData::Load(sphere, transform, sphere.mRadius * SampleUniformSphere(rnd.xy));
	} else
		return {}; // volume lights are unsupported

	r.mPdf /= r.mShadingData.mShapeArea;
	return r;
}
// uniformly samples a light instance and primitive index, then uniformly samples the primitive's area
IlluminationSampleRecord SampleIllumination(const float4 rnd, const float3 referencePosition) {
	EmissionSampleRecord emissionVertex = SampleEmission(rnd);

	IlluminationSampleRecord r;
	r.mPdf = emissionVertex.mPdf;
	r.isSingular = emissionVertex.isSingular;

    if (emissionVertex.mInstanceIndex == INVALID_INSTANCE) {
		// sample environment light
		r.mDistanceToLight = POS_INFINITY;
		r.mPosition = POS_INFINITY;
		r.mCosLight = 1;
		r.isFinite = false;

		float tmp;
		r.mRadiance = EvalBackground(emissionVertex.mShadingData.mPosition, tmp);
		return r;
	}

	r.isFinite = true;

	r.mRadiance = LoadMaterial(emissionVertex.mShadingData).Emission();
	r.mPosition = emissionVertex.mShadingData.mPosition;
	r.mPackedNormal = emissionVertex.mShadingData.mPackedShadingNormal;
	r.mDirectionToLight = r.mPosition - referencePosition;
	r.mDistanceToLight  = length(r.mDirectionToLight);
	r.mDirectionToLight /= r.mDistanceToLight;
	r.mCosLight = -dot(r.GetNormal(), r.mDirectionToLight);
	if (r.mCosLight <= 0) {
		// lights are one-sided
		return { 0, 0 };
	}
	return r;
}

struct DirectLightSampleRecord {
    float3 mRadiance;
    float mPdfW;
    float3 mToLight;
    float mDistance;
    float mG;
};
DirectLightSampleRecord SampleDirectLight(const ShadingData sd, const float4 rnd) {
    EmissionSampleRecord l = SampleEmission(rnd);
    DirectLightSampleRecord r;
    if (l.mInstanceIndex == INVALID_INSTANCE) {
        r.mToLight = l.mShadingData.mPosition;
        r.mDistance = POS_INFINITY;
        float tmp;
        r.mRadiance = EvalBackground(r.mToLight, tmp);
        r.mPdfW = l.mPdf;
        r.mG = 1;
    } else {
        r.mToLight = l.mShadingData.mPosition - sd.mPosition;
        r.mDistance = length(r.mToLight);
        r.mToLight /= r.mDistance;
		r.mRadiance = LoadMaterial(l.mShadingData).Emission();
        const float cosLight = max(0, -dot(r.mToLight, l.mShadingData.GetGeometryNormal()));
        if (cosLight <= 0)
            r.mRadiance = 0;
        r.mG = cosLight / sqr(r.mDistance);
        r.mPdfW = l.mPdf / r.mG;
    }
	return r;
}
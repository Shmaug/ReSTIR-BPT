#include "Intersection.slang"
#include "Sampling.slang"

struct EmissionSampleRecord {
    ShadingData mShadingData;
    float mPdf;
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

float3 EvalBackground(const float3 direction, out float pdfW) {
    if (!any(gScene.mBackgroundColor > 0))
        return 0;

    float3 emission = gScene.mBackgroundColor;

    if (gScene.mBackgroundImageIndex < gImageCount) {
        const float2 uv = CartesianToSphericalUV(direction);
        emission *= gScene.mImage4s[gScene.mBackgroundImageIndex].SampleLevel(gScene.mStaticSampler, uv, 0).rgb;
        pdfW = SampleTexelPdf(gScene.mImage4s[gScene.mBackgroundImageIndex], uv) / (2 * M_PI * M_PI * sqrt(1 - direction.y * direction.y));
    } else {
        pdfW = 1 / (4 * M_PI);
    }

    if (gScene.mLightCount > 0)
        pdfW *= gScene.mEnvironmentSampleProbability;

    return emission;
}

// uniformly samples a light instance and primitive index, then uniformly samples the primitive's area
EmissionSampleRecord SampleEmission(float4 rnd) {
	EmissionSampleRecord r;
	r.isSingular = false;

	if (gEnvironmentMaterialAddress != -1) {
		if (gScene.mLightCount == 0 || rnd.w < gEnvironmentSampleProbability) {
			// sample environment light
			r.mShadingData.mShapeArea = -1;
			r.mInstanceIndex = INVALID_INSTANCE;

			if (gScene.mBackgroundImageIndex < gImageCount) {
				r.mShadingData.mPosition = SphericalUVToCartesian(SampleTexel(mImages[gScene.mBackgroundImageIndex], rnd.xy, r.mPdf));
				// jacobian from sphericalUvToCartesian
				r.mPdf /= (2 * M_PI * M_PI * sqrt(1 - pow2(r.mShadingData.mPosition.y)));
			} else {
				r.mShadingData.mPosition = SampleUniformSphere(rnd.x, rnd.y);
				r.mPdf = 1 / (4 * M_PI);
			}

			if (gScene.mLightCount > 0)
				r.mPdf *= gEnvironmentSampleProbability;
			return r;
		}

		// remap rnd.w from [gEnvironmentSampleProbability,1] to [0,1]
		if (gScene.mLightCount > 0)
			rnd.w = (rnd.w - gEnvironmentSampleProbability) / (1 - gEnvironmentSampleProbability);
	}

	if (gScene.mLightCount == 0)
		return {};

	r.mInstanceIndex = mLightInstanceMap[uint(rnd.z * gScene.mLightCount) % gScene.mLightCount];
	r.mPdf = 1 / (float)gScene.mLightCount;

	if (gEnvironmentMaterialAddress != -1)
		r.mPdf *= 1 - gEnvironmentSampleProbability;

	const Instance instance  = gScene.mInstances[r.mInstanceIndex];
	const float4x4 transform = gScene.mInstanceTransforms[r.mInstanceIndex];

	if (instance.mHeader.Type() == InstanceType::eMesh) {
		// triangle
		const MeshInstance mesh = reinterpret<MeshInstance>(instance);
		r.mPrimitiveIndex = uint(rnd.w * mesh.PrimitiveCount()) % mesh.PrimitiveCount();
		r.mPdf /= (float)mesh.PrimitiveCount();
		r.mShadingData = ShadingData(mesh, transform, r.mPrimitiveIndex, SampleUniformTriangle(rnd.x, rnd.y));
	} else if (instance.mHeader.Type() == InstanceType::eSphere) {
		// sphere
		r.mPrimitiveIndex = INVALID_PRIMITIVE;
		const SphereInstance sphere = reinterpret<SphereInstance>(instance);
		r.mShadingData = ShadingData(sphere, transform, sphere.mRadius * SampleUniformSphere(rnd.x, rnd.y));
	} else
		return {}; // volume lights are unsupported

	r.mPdf /= r.mShadingData.mShapeArea;
	return r;
}
// uniformly samples a light instance and primitive index, then uniformly samples the primitive's area
IlluminationSampleRecord SampleIllumination(const float4 rnd, const float3 referencePosition) {
	const EmissionSampleRecord emissionVertex = SampleEmission(rnd);

	IlluminationSampleRecord r;
	r.mPdf = emissionVertex.mPdf;
	r.isSingular = emissionVertex.isSingular;

	if (emissionVertex.mShadingData.isEnvironment()) {
		// sample environment light
		r.mDistanceToLight = POS_INFINITY;
		r.mPosition = POS_INFINITY;
		r.mCosLight = 1;
		r.isFinite = false;

		float tmp;
		r.mRadiance = EvaluateEnvironment(emissionVertex.mShadingData.mPosition, tmp);
		return r;
	}

	r.isFinite = true;

	r.mRadiance = LoadMaterial(emissionVertex.mShadingData).getEmission();
	r.mPosition = emissionVertex.mShadingData.mPosition;
	r.mPackedNormal = emissionVertex.mShadingData.mPackedGeometryNormal;
	r.mDirectionToLight = r.mPosition - referencePosition;
	r.mDistanceToLight  = length(r.mDirectionToLight);
	r.mDirectionToLight /= r.mDistanceToLight;
	r.mCosLight = -dot(r.getNormal(), r.mDirectionToLight);
	if (r.mCosLight <= 0) {
		// lights are one-sided
		return { 0, 0 };
	}
	return r;
}

float LightSamplePdfA(IntersectionResult isect) {
	if (gScene.mLightCount == 0)
		return 0;
	float pdfA = mPrimitivePickPdf / (mShadingData.mShapeArea * gScene.mLightCount);
	if (any(gScene.mBackgroundColor > 0))
		pdfA *= 1 - gEnvironmentSampleProbability;
	return pdfA;
}

#include "PathVertex.slang"

// uniformly samples a light instance and primitive index, then uniformly samples the primitive's area.
// pdf is area measure except for background vertices
PathVertex SampleEmission(float4 rnd, out float pdf) {
    if (gScene.HasBackground()) {
		if (gScene.mLightCount == 0 || rnd.w < gScene.mBackgroundSampleProbability) {
			// sample background light

            float3 dir;
			if (gScene.mBackgroundImageIndex < gImageCount) {
				dir = SphericalUVToCartesian(SampleTexel(gScene.mImage4s[gScene.mBackgroundImageIndex], rnd.xy, pdf));
                // jacobian from SphericalUVToCartesian
				pdf /= (2 * M_PI * M_PI * sqrt(1 - sqr(dir.y)));
			} else {
				dir = SampleUniformSphere(rnd.xy);
				pdf = 1 / (4 * M_PI);
            }

            float tmp;
            PathVertex v;
            v.InitFromBackground(dir, EvalBackground(dir, tmp));

			if (gScene.mLightCount > 0)
				pdf *= gScene.mBackgroundSampleProbability;
			return v;
		}

		// remap rnd.w from [mBackgroundSampleProbability,1] to [0,1]
        if (gScene.mLightCount > 0)
            rnd.w = (rnd.w - gScene.mBackgroundSampleProbability) / (1 - gScene.mBackgroundSampleProbability);
	}

    if (gScene.mLightCount == 0) {
        pdf = 0;
        return PathVertex();
    }

    const uint instanceIndex = gScene.mLightInstanceMap[uint(rnd.z * gScene.mLightCount) % gScene.mLightCount];
	pdf = 1 / (float)gScene.mLightCount;
    if (gScene.HasBackground())
        pdf *= 1 - gScene.mBackgroundSampleProbability;

    const InstanceBase instance = gScene.mInstances[instanceIndex];
    const float4x4 transform = gScene.mInstanceTransforms[instanceIndex];

	PathVertex v;
	v.mInstanceIndex = instanceIndex;
	if (instance.mHeader.Type() == InstanceType::eMesh) {
		// triangle
		const MeshInstance mesh = reinterpret<MeshInstance>(instance);
        const uint primitiveIndex = uint(rnd.w * mesh.PrimitiveCount()) % mesh.PrimitiveCount();
		pdf /= (float)mesh.PrimitiveCount();
		v.InitFromTriangle(mesh, transform, primitiveIndex, SampleUniformTriangle(rnd.xy));
	} else if (instance.mHeader.Type() == InstanceType::eSphere) {
		// sphere
		const SphereInstance sphere = reinterpret<SphereInstance>(instance);
		v.InitFromSphere(sphere, transform, sphere.mRadius * SampleUniformSphere(rnd.xy));
	} else
		return {}; // volume lights are unsupported

	pdf /= v.mPrimitiveArea;
	return v;
}

struct DirectLightSampleRecord {
    float3 mRadiance;
    float mPdfW;
    float3 mToLight;
    float mDistance;
	float mCos;
    float mG;
};
DirectLightSampleRecord SampleDirectLight(const PathVertex vertex, const float4 rnd) {
    DirectLightSampleRecord r;
    const PathVertex l = SampleEmission(rnd, r.mPdfW);
	r.mRadiance = l.mMaterial.Emission();
    if (l.mIsSurface) {
        r.mToLight = l.mPosition - vertex.mPosition;
        r.mDistance = length(r.mToLight);
        r.mToLight /= r.mDistance;
        r.mCos = max(0, -dot(r.mToLight, l.mShadingNormal));
        if (r.mCos <= 0)
            r.mRadiance = 0;
        r.mG = r.mCos / sqr(r.mDistance);
        r.mPdfW /= r.mG;
    } else {
        r.mToLight = l.mPosition;
        r.mDistance = POS_INFINITY;
        r.mG = 1;
		r.mCos = 1;
    }
	return r;
}
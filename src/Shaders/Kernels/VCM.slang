#include "Intersection.slang"
#include "Random.slang"
#include "Light.slang"
#include "BRDF.slang"

[[vk::push_constant]]
cbuffer PushConstants {
	uint2 gOutputSize;
    uint gMaxBounces;
    uint gRandomSeed;
    float3 gCameraPosition;
	float gCameraImagePlaneDist;
    float3 gCameraForward;
};

static const uint gLightSubPathCount = gOutputSize.x * gOutputSize.y;

#ifndef gLightTraceEnabled
#define gLightTraceEnabled false
#endif
#ifndef gLightTraceOnly
#define gLightTraceOnly false
#endif
#ifndef gLightImageQuantization
#define gLightImageQuantization 16384
#endif
#ifndef gSampleLights
#define gSampleLights false
#endif

Texture2D<uint4> gVertices;
RWTexture2D<float4> gOutput;
RWByteAddressBuffer gLightImage;
uniform float4x4 gMVP;


void AddColor(uint2 id, float3 c) {
	const uint index = id.y * gOutputSize.x + id.x;
	uint3 ci = uint3(c * gLightImageQuantization);
	uint3 p;
	gLightImage.InterlockedAdd(16*index+0, ci.r, p.r);
	gLightImage.InterlockedAdd(16*index+4, ci.g, p.g);
	gLightImage.InterlockedAdd(16*index+8, ci.b, p.b);

	uint overflowMask = 0;
	for (uint i = 0; i < 3; i++)
		if (0xFFFFFFFF - p.r < ci.r)
			overflowMask |= (1 << i);
	gLightImage.InterlockedOr(16*index + 12, overflowMask);
}
float3 LoadColor(uint2 id) {
    uint4 c = gLightImage.Load4(16 * (id.y * gOutputSize.x + id.x));
	if (c.w & (1 << 0)) c.r = 0xFFFFFFFF;
	if (c.w & (1 << 1)) c.g = 0xFFFFFFFF;
	if (c.w & (1 << 2)) c.b = 0xFFFFFFFF;
	return c.rgb / (float)gLightImageQuantization;
}

float Mis(float a) { return a*a; }
float Mis(float a, float b) {
    a = Mis(a);
    return a / (a + Mis(b));
}

// expects lightPdf to be area measure when lightVertex.mIsSurface is true, or solid-angle measure if not
float3 ConnectToLight(const PathVertex vertex, const float3 dirIn, const PathVertex lightVertex, float lightPdf) {
    float3 toLight;
	float dist;
    float3 le = lightVertex.mMaterial.Emission();
    if (!any(le > 0))
        return 0;

    if (lightVertex.mIsSurface) {
        toLight = lightVertex.mPosition - vertex.mPosition;
        dist = length(toLight);
        toLight /= dist;
        const float cosLight = max(0, -dot(toLight, lightVertex.mShadingNormal));
        if (cosLight <= 0)
            return 0;
		lightPdf *= sqr(dist) / cosLight;
    } else {
        toLight = lightVertex.mPosition;
        dist = POS_INFINITY;
    }

	le *= EvalBrdf(vertex, -dirIn, toLight);

	if (Occluded(vertex, toLight, dist))
		return 0;

    return le * Mis(lightPdf, EvalBrdfPdfW(vertex, -dirIn, toLight)) / lightPdf;
}

void ConnectToCamera(const PathVertex vertex, const float3 dirIn, const float3 throughput, float fwdPdf, float revPdf) {
	const float4 clip = mul(gMVP, float4(vertex.mPosition, 1));
	const float2 uv = (clip.xy / clip.w) * .5 + .5;
	if (any(uv < 0) || any(uv >= 1) || clip.z < 0)
        return;
    const uint2 pixel = uint2(uv * gOutputSize);

	float3 toCamera = gCameraPosition - vertex.mPosition;
	const float dist = length(toCamera);
	toCamera /= dist;

	const float cosOut = dot(vertex.mShadingNormal, toCamera);

	const float3 le = throughput * EvalBrdf(vertex, -dirIn, toCamera, true) / abs(cosOut);
	if (!any(le > 0))
        return;

    const float cosAtCamera = dot(gCameraForward, -toCamera);
    const float imagePointToCameraDist = gCameraImagePlaneDist / cosAtCamera;
    const float imageToSolidAngleFactor = sqr(imagePointToCameraDist) / cosAtCamera;
    const float imageToSurfaceFactor = imageToSolidAngleFactor * abs(cosOut) / sqr(dist);

	const float surfaceToImageFactor = 1 / imageToSurfaceFactor;

    const float w = 1 / (gLightSubPathCount * surfaceToImageFactor);

    if (!(w > 0))
        return;

	revPdf *= EvalBrdfPdfW(vertex, -dirIn, toCamera);

	if (Occluded(vertex, toCamera, dist))
		return;

	AddColor(pixel, le * w / fwdPdf);
}

float3 SamplePath<let bIsLightPath : bool>(PathVertex vertex, float3 dirIn, RandomSampler rngIn, float3 throughput = 1, float pdfW = 1) {
    RandomSampler rng = rngIn;
    float3 color = 0;
    float revPdfW = 1;
    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const bool diffuse = IsDiffuse(vertex);
        if (diffuse) {
			if (bIsLightPath) {
				ConnectToCamera(vertex, dirIn, throughput, pdfW, revPdfW);
			} else if (!bIsLightPath && gSampleLights) {
                float lightVertexPdf;
                const PathVertex lightVertex = SampleEmission(rng.NextFloat(), lightVertexPdf);
                color += (throughput/pdfW) * ConnectToLight(vertex, dirIn, lightVertex, lightVertexPdf);
			}
		}

        // sample brdf
        DirectionSampleRecord r;
        SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection, bIsLightPath);
        const float brdfPdfW = EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
		if (bIsLightPath || bounces > 1)
			revPdfW *= EvalBrdfPdfW(vertex, r.mDirection, -dirIn);
        pdfW *= brdfPdfW;
        if (all(throughput <= 0)) break;

        dirIn = r.mDirection;

        float lightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, lightPdf, dist);

		// handle vertex emission
		if (!bIsLightPath) {
			float G = 1;
			float3 le = vertex.mMaterial.Emission();
			if (vertex.mIsSurface) {
				const float cosIn = dot(vertex.mShadingNormal, dirIn);
				lightPdf *= sqr(dist) / abs(cosIn);
				if (cosIn > 0)
					le = 0;
			}

			if (any(le > 0))
				color += throughput * le * (gSampleLights ? Mis(brdfPdfW, lightPdf) : 1) / pdfW;
		}

		if (bounces < gMaxBounces) {
			const float p = GetContinuationProbability(vertex);
			if (p == 0 || rng.NextFloat().x > p)
				break;
			pdfW *= p;
			revPdfW *= p;
		}
    }
    return color;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void AddLightImage(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;

	gOutput[id] += float4(LoadColor(id), 0);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleLightPaths(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    sHeatmapPixel = -1;

	RandomSampler rng = RandomSampler(gRandomSeed, id + uint2(gOutputSize.x, 0));

	float emissionPdf;
	PathVertex vertex = SampleEmission(rng.NextFloat(), emissionPdf);

    float3 throughput = vertex.mMaterial.Emission();
	float3 origin = vertex.mPosition;
    float3 dirOut;
    const float4 rnd = rng.NextFloat();
    if (vertex.mIsSurface) {
        dirOut = SampleCosHemisphere(rnd.xy);
        emissionPdf *= CosHemispherePdfW(dirOut.z);
		throughput *= abs(dirOut.z);
        dirOut = vertex.ToWorld(dirOut);
        origin = OffsetRayOrigin(vertex, dirOut);
    } else {
		const float3 sceneCenter = (gScene.mSceneMax + gScene.mSceneMin) / 2;
        const float sceneRadius = length(gScene.mSceneMax - gScene.mSceneMin) / 2;
        const float2 d = SampleConcentricDisc(rnd.xy);
        dirOut = -vertex.mPosition;
        const float3x3 onb = MakeOrthonormal(dirOut);
       	origin = sceneCenter + (onb[0] * d.x + onb[1] * d.y - dirOut) * sceneRadius;
        emissionPdf *= ConcentricDiscPdfA() / sqr(sceneRadius);
	}

	float isectLightPdf, isectDist;
    vertex = TraceRay(MakeRay(origin, dirOut), true, isectLightPdf, isectDist);

    if (vertex.mIsSurface) {
        SamplePath<true>(vertex, dirOut, rng, throughput, emissionPdf);
    }
}

[shader("compute")]
[numthreads(8, 4, 1)]
void SampleCameraPaths(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    sHeatmapPixel = id.y * gOutputSize.x + id.x;

	PathVertex vertex = UnpackVertex(reinterpret<PackedVertex>(gVertices[id]));

	if (vertex.mIsSurface) {
		gOutput[id] += float4(SamplePath<false>(vertex, normalize(vertex.mPosition - gCameraPosition), RandomSampler(gRandomSeed, id)), 1);
	}
}

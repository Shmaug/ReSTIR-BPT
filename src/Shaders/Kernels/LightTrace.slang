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
static const float3 gSceneCenter = (gScene.mSceneMax + gScene.mSceneMin) / 2;
static const float gSceneRadius = length(gScene.mSceneMax - gScene.mSceneMin) / 2;

#ifndef gLightTrace
#define gLightTrace false
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

float GetCameraPdfW(float3 direction) {
    const float cameraCos = abs(dot(gCameraForward, direction));
    const float imagePointToCameraDist = gCameraImagePlaneDist / cameraCos;
    return sqr(imagePointToCameraDist) / cameraCos;
}

void ConnectToCamera(const PathVertex vertex, const float3 dirIn, const float3 throughput, float lightPathPdf, float cameraPathPdf) {
	const float4 clip = mul(gMVP, float4(vertex.mPosition, 1));
	const float2 uv = (clip.xy / clip.w) * .5 + .5;
	if (any(uv < 0) || any(uv >= 1) || clip.z < 0)
        return;
    const uint2 pixel = uint2(uv * gOutputSize);

	float3 toCamera = gCameraPosition - vertex.mPosition;
	const float dist = length(toCamera);
	toCamera /= dist;

	const float cosOut = abs(dot(vertex.mShadingNormal, toCamera));

	float3 le = throughput * EvalBrdf(vertex, -dirIn, toCamera, true) / cosOut;
	if (!any(le > 0))
        return;

	const float cameraPdfW = GetCameraPdfW(-toCamera);
	const float cameraPdfA = cameraPdfW * cosOut / sqr(dist);
	const float surfaceToImageFactor = 1 / cameraPdfA;

	le /= (surfaceToImageFactor * gLightSubPathCount);

	if (!gLightTraceOnly) {
		cameraPathPdf *= EvalBrdfPdfW(vertex, toCamera, -dirIn) * GetContinuationProbability(vertex);

		le *= Mis(lightPathPdf * cameraPdfA, cameraPathPdf);
	}

	if (Occluded(vertex, toCamera, dist))
		return;

	AddColor(pixel, le);
}

float3 ConnectToLight(const PathVertex vertex, const float3 dirIn, const float4 rnd, const float3 throughput, const float contProb) {
	const DirectLightSampleRecord l = SampleDirectLight(vertex, rnd);

	float3 le = l.mRadiance;
	if (!any(le > 0))
		return 0;

	le *= EvalBrdf(vertex, -dirIn, l.mToLight);

	if (!any(le > 0))
		return 0;

	if (Occluded(vertex, l.mToLight, l.mDistance))
		return 0;

	return throughput * le * Mis(l.mPdfW, contProb*EvalBrdfPdfW(vertex, -dirIn, l.mToLight)) / l.mPdfW;
}

float3 SamplePath<let bIsLightPath : bool>(PathVertex vertex, float3 dirIn, RandomSampler rngIn, float3 throughput = 1, float fwdPdf = 1, float revPdf = 1) {
    RandomSampler rng = rngIn;
    float3 color = 0;
    for (uint bounces = 1; bounces <= gMaxBounces; bounces++) {
        const bool diffuse = IsDiffuse(vertex);
        const float contProb = diffuse ? GetContinuationProbability(vertex) : 1;
        if (diffuse) {
			if (!bIsLightPath && gSampleLights) {
				color += ConnectToLight(vertex, dirIn, rng.NextFloat(), throughput, contProb);
			} else if (bIsLightPath) {
				ConnectToCamera(vertex, dirIn, throughput, fwdPdf, revPdf);
			}
		}
        if (contProb == 0 || rng.NextFloat().x > contProb)
            break;

        // sample brdf
        DirectionSampleRecord r;
        SampleBrdf(vertex, rng.NextFloat(), -dirIn, r);
        throughput *= EvalBrdf(vertex, -dirIn, r.mDirection, bIsLightPath);
        if (all(throughput <= 0)) break;

		if (!bIsLightPath && bounces == 1 && !diffuse)
			revPdf = 0;

		const float fwdDirPdfW = contProb * EvalBrdfPdfW(vertex, -dirIn, r.mDirection);
		throughput /= fwdDirPdfW;
		fwdPdf *= fwdDirPdfW;
		if (bIsLightPath || bounces > 1) // dont include pdf towards camera, since camera cant be hit via bsdf sampling
			revPdf *= contProb * EvalBrdfPdfW(vertex, r.mDirection, -dirIn); // geometry term is premultiplied into revPdf

        dirIn = r.mDirection;

		revPdf *= abs(dot(vertex.mShadingNormal, dirIn));

        float directLightPdf, dist;
        vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, dirIn), dirIn), true, directLightPdf, dist);

		float3 le = vertex.mMaterial.Emission();
		float emitPdf = directLightPdf;
		float G = 1;
		if (vertex.mIsSurface) {
			const float cosIn = -dot(vertex.mShadingNormal, dirIn);
			const float rcvDist2 = 1 / sqr(dist);
			G = abs(cosIn) * rcvDist2;
			revPdf *= rcvDist2;

			if (cosIn <= 0) le = 0;
			emitPdf *= CosHemispherePdfW(cosIn);
			fwdPdf *= G;
		} else
			emitPdf *= ConcentricDiscPdfA() / sqr(gSceneRadius);

		if (!bIsLightPath && any(le > 0)) {
			float w = 1;

			if (gLightTrace) {
				if (gSampleLights) {
					w = 0;
				} else
					w = Mis(fwdPdf, revPdf * emitPdf);
			} else if (gSampleLights)
				w = Mis(fwdDirPdfW*contProb, directLightPdf/G);

			color += throughput * le * w;
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
    float cosOut = 1;
    float3 origin, dirOut;
    const bool isSurface = vertex.mIsSurface;

    const float2 rnd = rng.NextFloat().xy;
    if (isSurface) {
        dirOut = SampleCosHemisphere(rnd);
        cosOut = abs(dirOut.z);

        emissionPdf *= CosHemispherePdfW(cosOut);
		throughput *= cosOut;

        dirOut = vertex.ToWorld(dirOut);
        origin = OffsetRayOrigin(vertex, dirOut);
    } else {
        const float2 d = SampleConcentricDisc(rnd);
        dirOut = -vertex.mPosition;
        const float3x3 onb = MakeOrthonormal(dirOut);
       	origin = gSceneCenter + (onb[0] * d.x + onb[1] * d.y - dirOut) * gSceneRadius;

        emissionPdf *= ConcentricDiscPdfA() / sqr(gSceneRadius);
	}

	throughput /= emissionPdf;

	float isectLightPdf, isectDist;
    vertex = TraceRay(MakeRay(origin, dirOut), true, isectLightPdf, isectDist);

    if (vertex.mIsSurface) {
		const float G    = isSurface ? abs(dot(vertex.mShadingNormal, dirOut)) / sqr(isectDist) : 1;
		const float revG = isSurface ? cosOut / sqr(isectDist) : 1;
        SamplePath<true>(vertex, dirOut, rng, throughput, emissionPdf * G, revG);
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
		float3 dir = vertex.mPosition - gCameraPosition;
		const float dist = length(dir);
        dir /= dist;
        const float cameraPdfA = GetCameraPdfW(dir) * abs(dot(vertex.mShadingNormal, dir)) / sqr(dist);
		gOutput[id] += float4(SamplePath<false>(vertex, dir, RandomSampler(gRandomSeed, id), 1, 1, cameraPdfA), 0);
	}
}

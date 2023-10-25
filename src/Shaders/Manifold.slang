#pragma once

[[vk::push_constant]]
cbuffer PushConstants {
    uint2 gOutputSize;
    uint gRandomSeed;
    uint gManifoldSolverIterations;
    float3 gCameraPosition;
    float gManifoldSolverStepSize;
    float gManifoldSolverThreshold;
};

RWStructuredBuffer<uint> gDebugImage;

#define VERTEX_DIFFERENTIALS true

#ifdef MANIFOLD_SAMPLING
static const bool gManifoldSampling = true;
#else
static const bool gManifoldSampling = false;
#endif
#ifdef MULTI_MANIFOLD_SAMPLING
static const bool gMultiManifoldSampling = true;
#else
static const bool gMultiManifoldSampling = false;
#endif

static const uint gManifoldMaxRays = 2;
static const uint gManifoldProjectionAttempts = 2;

#include "Intersection.slang"
#include "Random.slang"
#include "BRDF.slang"

uniform float4x4 gMVP;

void WriteDebugPixel(const float3 worldPos, const uint i) {
    const float4 clip = mul(gMVP, float4(worldPos, 1));
    int2 pixelCoord = int2(gOutputSize * ((clip.xy / clip.w) * .5 + .5));
    if (any(pixelCoord < 0) || any(pixelCoord >= gOutputSize) || clip.z < 0)
        return;
    InterlockedMax(gDebugImage[pixelCoord.y * gOutputSize.x + pixelCoord.x], i);
}

float2x2 inverse(const float2x2 m, const float det) {
    float2x2 r = -m;
    r[0][0] = m[1][1];
    r[1][1] = m[0][0];
    return r / det;
}

[ForwardDifferentiable]
float ComputeConstraint(
    const float3 p0,
	const float2 bary,
    const float3 v0, const float3 dPdu, const float3 dPdv,
    const float3 n0, const float3 dNdu, const float3 dNdv,
    const float3 p2, const bool p2sfc,
    const float eta_) {

    const float3 p = v0 + bary.x * dPdu + bary.y * dPdv;
    const float3 n = normalize(n0 + bary.x * dNdu + bary.y * dNdv);

    const float3 dirIn = normalize(p0 - p);
    const float3 dirOut = p2sfc ? normalize(p2 - p) : p2;

    const float cosIn  = dot(dirIn , n);
    const float cosOut = dot(dirOut, n);
    const bool reflecting = cosIn * cosOut > 0;
    float eta = eta_;
    float etaRcp = 1 / eta;
    if (cosIn < 0) {
        const float tmp = eta;
        eta = etaRcp;
        etaRcp = tmp;
    }

    float3 testDir = dirOut;
    float3 targetDir = reflecting ? reflect(-dirIn, n) : refract(-dirIn, n, eta);
    if (all(targetDir == 0)) {
        testDir = dirIn;
        targetDir = reflecting ? reflect(-dirOut, n) : refract(-dirOut, n, etaRcp);
		if (all(targetDir == 0))
    		return no_diff POS_INFINITY;
    }

    float C = acos(dot(targetDir, testDir));
    if (C < 0) C += M_2PI;
    return C;
}

bool ManifoldSolve(inout PathVertex vertices[3]) {
	float step = gManifoldSolverStepSize;

    DebugPrintf("------------ Solving ------------\n");
    for (uint iteration = 0; iteration < gManifoldSolverIterations; iteration++) {
        IncrementCounter(DebugCounterType::eManifoldWalkIterations);
        if (IsDebugPixel)
            WriteDebugPixel(vertices[1].mPosition, iteration);

		const DifferentialPair<float> dCdu = __fwd_diff(ComputeConstraint)(
            diffPair(vertices[0].mPosition),
            diffPair(vertices[1].barycentrics, float2(1, 0)),
            diffPair(vertices[1].triP0), diffPair(vertices[1].dPdu), diffPair(vertices[1].dPdv),
            diffPair(vertices[1].triN0), diffPair(vertices[1].dNdv), diffPair(vertices[1].dNdv),
            diffPair(vertices[2].mPosition),
            vertices[2].mIsSurface,
            diffPair(vertices[1].mMaterial.Eta()));

        const DifferentialPair<float> dCdv = __fwd_diff(ComputeConstraint)(
            diffPair(vertices[0].mPosition),
            diffPair(vertices[1].barycentrics, float2(0, 1)),
            diffPair(vertices[1].triP0), diffPair(vertices[1].dPdu), diffPair(vertices[1].dPdv),
            diffPair(vertices[1].triN0), diffPair(vertices[1].dNdv), diffPair(vertices[1].dNdv),
            diffPair(vertices[2].mPosition),
            vertices[2].mIsSurface,
            diffPair(vertices[1].mMaterial.Eta()));

        const float C = dCdu.p;
        if (isinf(C)) {
            DebugPrintf("\tFailed: invalid constraint\n");
            return false;
        }
        if (C < gManifoldSolverThreshold) {
            DebugPrintf("\tSucceeded: %f\n", C);
            return true;
        }

        const float2 duv = C * float2(dCdu.d, dCdv.d);
		if (iteration+1 == gManifoldSolverIterations)
        	DebugPrintf("\tC: %f, duv: %f %f\n", C, duv.x, duv.y);

        if (length(duv) < 1e-6) {
            DebugPrintf("Failed: Local minima\n");
            return false;
		}


		// offset vertex and project onto manifold
		// if the projection fails, reduce step size and try again

        bool projected = false;
        PathVertex newVertex;
        for (uint projectionIteration = 0; projectionIteration < gManifoldProjectionAttempts; projectionIteration++) {
            const float3 candidatePos = vertices[1].mPosition - step * (vertices[1].dPdu * duv.x + vertices[1].dPdv * duv.y);
			const float3 direction = normalize(candidatePos - vertices[0].mPosition);
			float3 origin = OffsetRayOrigin(vertices[0], direction);
			for (uint i = 0; i < gManifoldMaxRays; i++) {
				newVertex = TraceRay(MakeRay(origin, direction));
				if (!newVertex.mIsSurface)
					break;

				if (!IsDiffuse(newVertex)) {
					projected = true;
					break;
				}

				// ignore non-specular vertices along the ray
				origin = OffsetRayOrigin(newVertex, direction);
			}
            if (projected)
                break;
            step *= 0.5;
            DebugPrintf("\tReducing step size...\n");
        }
		if (!projected)
			return false;

        vertices[1] = newVertex;
    }
    return false;
}

float ManifoldG(const PathVertex vertices[3]) {
    // A lot of the computation here overlaps with 'compute_step_halfvector', but
    // we need to do a bit more work involving the endpoints to get the full
    // geometric term.

    float3 wi = vertices[0].mPosition - vertices[1].mPosition;
    float ili = length(wi);
    if (ili < 1e-3f) {
        return 0.f;
    }
    ili = rcp(ili);
    wi *= ili;

    float3 wo = vertices[2].mIsSurface ? vertices[2].mPosition - vertices[1].mPosition : vertices[2].mPosition;
    float ilo = length(wo);
    if (ilo < 1e-3f) {
        return 0.f;
    }
    ilo = rcp(ilo);
    wo *= ilo;

    float2x2 dc1_dx0, dc2_dx1, dc2_dx2;
    if (!vertices[2].mIsSurface) {
        /* This case is actually a bit more tricky as we're now in a situation
           with two "specular" constraints. As a consequence, we need to solve
           a bigger matrix system, so we prepare a few additional terms. */

        // Derivative of directional light constraint w.r.t. vertices[1]
        float3 dc2_du1 = ilo * (vertices[1].dPdu - wo * dot(wo, vertices[1].dPdu));
		float3 dc2_dv1 = ilo * (vertices[1].dPdv - wo * dot(wo, vertices[1].dPdv));
        dc2_dx1 = float2x2(
            dot(dc2_du1, vertices[2].dPdu), dot(dc2_dv1, vertices[2].dPdu),
            dot(dc2_du1, vertices[2].dPdv), dot(dc2_dv1, vertices[2].dPdv)
        );

        // Derivative of directional light constraint w.r.t. vertices[2]
        float3 dc2_du2 = -ilo * (vertices[2].dPdu - wo * dot(wo, vertices[2].dPdu));
		float3 dc2_dv2 = -ilo * (vertices[2].dPdv - wo * dot(wo, vertices[2].dPdv));
        dc2_dx2 = float2x2(
            dot(dc2_du2, vertices[2].dPdu), dot(dc2_dv2, vertices[2].dPdu),
            dot(dc2_du2, vertices[2].dPdv), dot(dc2_dv2, vertices[2].dPdv)
        );
    }

    // Setup generalized half-vector
    const float cosIn  = dot(wi, vertices[1].mShadingNormal);
    const bool refraction = cosIn * dot(wo, vertices[1].mShadingNormal) < 0;
    float eta = vertices[1].mMaterial.Eta();
    if (cosIn < 0.f) {
        eta = rcp(eta);
    }
    float3 h = wi + eta * wo;
    if (eta != 1.f) h *= -1.f;
    float ilh = rcp(length(h));
    h *= ilh;

    ilo *= eta * ilh;
    ili *= ilh;

    // Local shading tangent frame
    float dot_dpdu_n = dot(vertices[1].dPdu, vertices[1].mShadingNormal);
	float dot_dpdv_n = dot(vertices[1].dPdv, vertices[1].mShadingNormal);
    float3 s = vertices[1].dPdu - dot_dpdu_n * vertices[1].mShadingNormal;
	float3 t = vertices[1].dPdv - dot_dpdv_n * vertices[1].mShadingNormal;

    float3 dh_du, dh_dv;

    if (!vertices[2].mIsSurface) {
        // Derivative of specular constraint w.r.t. vertices[0]
        dh_du = ili * (vertices[0].dPdu - wi * dot(wi, vertices[0].dPdu));
        dh_dv = ili * (vertices[0].dPdv - wi * dot(wi, vertices[0].dPdv));
        dh_du -= h * dot(dh_du, h);
        dh_dv -= h * dot(dh_dv, h);
        if (eta != 1.f) {
            dh_du *= -1.f;
            dh_dv *= -1.f;
        }
        dc1_dx0 = float2x2(
            dot(dh_du, s), dot(dh_dv, s),
            dot(dh_du, t), dot(dh_dv, t)
        );
    }

    // Derivative of specular constraint w.r.t. vertices[1]
    dh_du = -vertices[1].dPdu * (ili + ilo) + wi * (dot(wi, vertices[1].dPdu) * ili) + wo * (dot(wo, vertices[1].dPdu) * ilo);
    dh_dv = -vertices[1].dPdv * (ili + ilo) + wi * (dot(wi, vertices[1].dPdv) * ili) + wo * (dot(wo, vertices[1].dPdv) * ilo);
    dh_du -= h * dot(dh_du, h);
    dh_dv -= h * dot(dh_dv, h);
    if (refraction) {
        dh_du *= -1.f;
        dh_dv *= -1.f;
    }

    float dot_h_n    = dot(h, vertices[1].mShadingNormal);
	float dot_h_dndu = dot(h, vertices[1].dNdu);
	float dot_h_dndv = dot(h, vertices[1].dNdv);
    float2x2 dc1_dx1 = float2x2(
        dot(dh_du, s) - dot(vertices[1].dPdu, vertices[1].dNdu) * dot_h_n - dot_dpdu_n * dot_h_dndu,
        dot(dh_dv, s) - dot(vertices[1].dPdu, vertices[1].dNdv) * dot_h_n - dot_dpdu_n * dot_h_dndv,
        dot(dh_du, t) - dot(vertices[1].dPdv, vertices[1].dNdu) * dot_h_n - dot_dpdv_n * dot_h_dndu,
        dot(dh_dv, t) - dot(vertices[1].dPdv, vertices[1].dNdv) * dot_h_n - dot_dpdv_n * dot_h_dndv );

    // Derivative of specular constraint w.r.t. vertices[2]
    dh_du = ilo * (vertices[2].dPdu - wo * dot(wo, vertices[2].dPdu));
    dh_dv = ilo * (vertices[2].dPdv - wo * dot(wo, vertices[2].dPdv));
    dh_du -= h * dot(dh_du, h);
    dh_dv -= h * dot(dh_dv, h);
    if (refraction) {
        dh_du *= -1.f;
        dh_dv *= -1.f;
    }
    float2x2 dc1_dx2 = float2x2(
        dot(dh_du, s), dot(dh_dv, s),
        dot(dh_du, t), dot(dh_dv, t) );

    if (!vertices[2].mIsSurface) {
        // Invert 2x2 block matrix system
        float det = determinant(dc2_dx2);
        if (abs(det) < 1e-6f) {
            return 0.f;
        }
        float2x2 Li = inverse(dc2_dx2, det);
        float2x2 tmp = mul(Li, dc2_dx1);
        float2x2 m = dc1_dx1 - mul(dc1_dx2, tmp);
        det = determinant(m);
        if (abs(det) < 1e-6f) {
            return 0.f;
        }
        Li = inverse(m, det);
        float2x2 sol1 = -mul(Li, dc1_dx0);
        float2x2 sol0 = -mul(tmp, sol1);

        float G = abs(determinant(-sol0));
        /* Unfortunately, these geometric terms are very unstable, so to avoid
           severe variance we need to clamp here. */
        G = min(G, float(10.f));
        G /= abs(dot(wi, vertices[0].mShadingNormal)); // Cancel out cosine term that will be added during BSDF evaluation
        G *= sqr(eta);
        return G;
    } else {
        // Invert single 2x2 matrix
        float det = determinant(dc1_dx1);
        if (abs(det) < 1e-6f) {
            return 0.f;
        }
        float2x2 inv_dc1_dx1 = inverse(dc1_dx1, det);
        float dx1_dx2 = abs(determinant(mul(inv_dc1_dx1, dc1_dx2)));
        /* Unfortunately, these geometric terms are very unstable, so to avoid
           severe variance we need to clamp here. */
        dx1_dx2 = min(dx1_dx2, float(1.f));
        float3 d = vertices[0].mPosition - vertices[1].mPosition;
        float inv_r2 = rcp(dot(d,d));
        d *= sqrt(inv_r2);
        float dw0_dx1 = abs(dot(d, vertices[1].mShadingNormal)) * inv_r2;
        float G = dw0_dx1 * dx1_dx2;
        return G;
    }
}

float ManifoldPdf(const PathVertex vertices[3]) {
    return 0;
}
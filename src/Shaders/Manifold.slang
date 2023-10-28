#pragma once

[[vk::push_constant]]
cbuffer PushConstants {
    float4x4 gMVP;
    uint2 gOutputSize;
    uint gRandomSeed;
    uint gMaxBounces;
    float3 gCameraPosition;
    uint gManifoldSolverIterations;
    uint gMinManifoldVertices;
    uint gMaxManifoldVertices;
    float gManifoldSolverThreshold;
    float gManifoldSolverStepSize;
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
static const uint gManifoldProjectionAttempts = 4;

enum StepMode {
	eFixed,
    eHessian,
    eHessianEigenDecomp
};
#ifndef gStepMode
static const StepMode gStepMode = StepMode::eFixed;
#endif

static const

#include "Intersection.slang"
#include "Random.slang"
#include "BRDF.slang"

float2x2 Adjoint(const float2x2 m) {
    return float2x2(
		 m[1][1], -m[0][1],
		-m[1][0],  m[0][0] );
}
float2x2 inverse(const float2x2 m, const float det) {
    return Adjoint(m) / det;
}
bool ComputeEigenvalues(const float2x2 m, const float det, out float2 eigenvalues, out float2x2 eigenvectors) {
    const float trace = m[0][0] + m[1][1];

    float d = sqr(trace) - 4 * det;
    if (d < 0) return false;
    d = sqrt(d);
    eigenvalues = (trace + float2(d, -d)) / 2;

    if (m[0][1] != 0) {
        eigenvectors[0] = normalize( float2( eigenvalues[0] - m[1][1], m[0][1] ) );
        eigenvectors[1] = normalize( float2( eigenvalues[1] - m[1][1], m[0][1] ) );
    } else if (m[1][0] != 0) {
        eigenvectors[0] = normalize( float2( m[1][0], eigenvalues[0] - m[0][0] ) );
        eigenvectors[1] = normalize( float2( m[1][0], eigenvalues[1] - m[0][0] ) );
    } else {
        eigenvectors[0] = float2(1, 0);
        eigenvectors[1] = float2(0, 1);
    }

    return true;
}

void WriteDebugPixel(const float3 worldPos, const uint i) {
    const float4 clip = mul(gMVP, float4(worldPos, 1));
    int2 pixelCoord = int2(gOutputSize * ((clip.xy / clip.w) * .5 + .5));
    if (all(pixelCoord >= 0) && all(pixelCoord < gOutputSize) && clip.z > 0)
    	InterlockedMax(gDebugImage[pixelCoord.y * gOutputSize.x + pixelCoord.x], i);
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

    if (reflecting) {
		return 1 - dot(dirOut, normalize(-dirIn + 2 * cosIn * n));
    } else {
		float eta = eta_;
		float etaRcp = 1 / eta;
		if (cosIn < 0) {
			const float tmp = eta;
			eta = etaRcp;
			etaRcp = tmp;
		}
        if (1 - (1 - sqr(cosIn)) * sqr(etaRcp) > 0)
            return 1 - dot(dirOut, -dirIn * etaRcp + (abs(cosIn) * etaRcp - cosOut) * n);
        else if (1 - (1 - sqr(cosOut)) * sqr(eta) > 0)
        	return 1 - dot(dirIn, -dirOut * eta + (abs(cosOut) * eta - cosIn) * (-n));
		else
			return no_diff POS_INFINITY;
    }
}

float2 ComputeStep(const float3 start, inout PathVertex vertex, const float3 end, bool endIsSurface, out float C) {
    if (gStepMode == StepMode::eFixed) {
		// compute first derivatives

        const DifferentialPair<float> dCdu = __fwd_diff(ComputeConstraint)(
            diffPair(start),
            diffPair(vertex.barycentrics, float2(1, 0)),
            diffPair(vertex.triP0), diffPair(vertex.dPdu), diffPair(vertex.dPdv),
            diffPair(vertex.triN0), diffPair(vertex.dNdv), diffPair(vertex.dNdv),
            diffPair(end), endIsSurface,
            diffPair(vertex.mMaterial.Eta()));

        const DifferentialPair<float> dCdv = __fwd_diff(ComputeConstraint)(
            diffPair(start),
            diffPair(vertex.barycentrics, float2(0, 1)),
            diffPair(vertex.triP0), diffPair(vertex.dPdu), diffPair(vertex.dPdv),
            diffPair(vertex.triN0), diffPair(vertex.dNdv), diffPair(vertex.dNdv),
            diffPair(end), endIsSurface,
            diffPair(vertex.mMaterial.Eta()));

        C = dCdu.p;
        const float2 dCduv = float2(dCdu.d, dCdv.d);
        return C * dCduv;
    } else {
		// compute second derivatives

		const DifferentialPair<DifferentialPair<float>> dC2du2 = __fwd_diff(__fwd_diff(ComputeConstraint))(
			diffPair(diffPair(start)),
			diffPair(diffPair(vertex.barycentrics, float2(1, 0)), diffPair(float2(1, 0), float2(0))),
			diffPair(diffPair(vertex.triP0)), diffPair(diffPair(vertex.dPdu)), diffPair(diffPair(vertex.dPdv)),
			diffPair(diffPair(vertex.triN0)), diffPair(diffPair(vertex.dNdv)), diffPair(diffPair(vertex.dNdv)),
			diffPair(diffPair(end)), endIsSurface,
			diffPair(diffPair(vertex.mMaterial.Eta())));

		const DifferentialPair<DifferentialPair<float>> dC2dv2 = __fwd_diff(__fwd_diff(ComputeConstraint))(
			diffPair(diffPair(start)),
			diffPair(diffPair(vertex.barycentrics, float2(0, 1)), diffPair(float2(0, 1), float2(0))),
			diffPair(diffPair(vertex.triP0)), diffPair(diffPair(vertex.dPdu)), diffPair(diffPair(vertex.dPdv)),
			diffPair(diffPair(vertex.triN0)), diffPair(diffPair(vertex.dNdv)), diffPair(diffPair(vertex.dNdv)),
			diffPair(diffPair(end)), endIsSurface,
			diffPair(diffPair(vertex.mMaterial.Eta())));

		const DifferentialPair<DifferentialPair<float>> dC2duv = __fwd_diff(__fwd_diff(ComputeConstraint))(
			diffPair(diffPair(start)),
			diffPair(diffPair(vertex.barycentrics, float2(1, 0)), diffPair(float2(0, 1), float2(0))),
			diffPair(diffPair(vertex.triP0)), diffPair(diffPair(vertex.dPdu)), diffPair(diffPair(vertex.dPdv)),
			diffPair(diffPair(vertex.triN0)), diffPair(diffPair(vertex.dNdv)), diffPair(diffPair(vertex.dNdv)),
			diffPair(diffPair(end)), endIsSurface,
			diffPair(diffPair(vertex.mMaterial.Eta())));

		C = dC2du2.p.p;
		const float2 dCduv = float2(dC2du2.d.p, dC2dv2.d.p);

		const float2x2 H = float2x2(
			dC2du2.d.d, dC2duv.d.d,
			dC2duv.d.d, dC2dv2.d.d);
		const float detH = determinant(H);

		if (gStepMode == StepMode::eHessian)
			return mul(Adjoint(H), dCduv) / detH;
		else {
			float2 eigenvalues;
			float2x2 Q;
			if (!ComputeEigenvalues(H, detH, eigenvalues, Q))
				return 0;
			eigenvalues[0] = eigenvalues[0] > 0 ? 1 / eigenvalues[0] : 0;
			eigenvalues[1] = eigenvalues[1] > 0 ? 1 / eigenvalues[1] : 0;
			return mul(mul(mul(transpose(Q), float2x2(eigenvalues[0], 0, 0, eigenvalues[1])), Q), dCduv);
		}
    }
}

bool ManifoldSolve(const float3 start, const float3 startNg, inout PathVertex vertex, const float3 end, bool endIsSurface) {
    DebugPrintf("------------ Solving ------------\n");
    for (uint iteration = 0; iteration < gManifoldSolverIterations; iteration++) {
        IncrementCounter(DebugCounterType::eManifoldWalkIterations);
        if (IsDebugPixel)
            WriteDebugPixel(vertex.mPosition, iteration);

        float C;
        const float2 duv = ComputeStep(start, vertex, end, endIsSurface, C);

        if (isinf(C) || any(isnan(duv)) || any(isinf(duv))) {
            DebugPrintf("\tFailed: invalid constraint\n");
            return false;
        }
        if (C < gManifoldSolverThreshold) {
            DebugPrintf("\tSucceeded: %f\n", C);
            return true;
        }

		#ifdef PRINT_C
		DebugPrintf("\tC: %f, duv: %f %f\n", C, duv.x, duv.y);
		#endif

        if (length(duv) < 1e-6) {
            DebugPrintf("Failed: Stuck\n");
            return false;
		}

        #if 0
        // if offset point is on the same triangle, we can just load it directly
        float2 candidateBary = vertex.barycentrics - duv*gManifoldSolverStepSize;
        if (all(candidateBary >= 0) && candidateBary.x + candidateBary.y <= 1) {
            vertex.InitFromTriangle(
                reinterpret<MeshInstance>(gScene.mInstances[vertex.mInstanceIndex]),
                gScene.mInstanceTransforms[vertex.mInstanceIndex],
                vertex.mPrimitiveIndex,
                candidateBary);
        } else
		#endif
		{
			// project offsetted vertex back onto manifold
			// if the projection fails, reduce step size and try again
			bool projected = false;
			PathVertex newVertex;
            for (uint projectionIteration = 0; projectionIteration < gManifoldProjectionAttempts; projectionIteration++) {
                const float3 candidatePos = vertex.mPosition - (gManifoldSolverStepSize / float(1 << projectionIteration)) * (vertex.dPdu * duv.x + vertex.dPdv * duv.y);
				const float3 direction = normalize(candidatePos - start);
				float3 origin = OffsetRayOrigin(start, startNg, direction);
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
			}
			if (!projected) {
				DebugPrintf("\tProjection failed\n");
				return false;
			}

			vertex = newVertex;
        }
    }
    return false;
}
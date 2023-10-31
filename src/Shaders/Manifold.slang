#pragma once

[[vk::push_constant]]
cbuffer PushConstants {
    float4x4 gMVP;
    uint2 gOutputSize;
    uint gRandomSeed;
	#ifndef gMaxBounces
    uint gMaxBounces;
	#endif
    float3 gCameraPosition;
    uint gManifoldSolverIterations;
    uint gMinManifoldVertices;
    float gManifoldSolverThreshold;
    float gManifoldSolverStepSize;
    uint gLightSampleCandidates;
};

RWStructuredBuffer<uint> gDebugImage;

#define VERTEX_DIFFERENTIALS true

#ifndef gMaxManifoldVertices
static const uint gMaxManifoldVertices = 0;
#endif
static const uint gMaxManifoldIntermediateVertices = gMaxManifoldVertices == 0 ? 0 : (gMaxManifoldVertices-1);
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

static uint iteration;
void WriteDebugPixel(const float3 worldPos) {
    const float4 clip = mul(gMVP, float4(worldPos, 1));
    int2 pixelCoord = int2(gOutputSize * ((clip.xy / clip.w) * .5 + .5));
    if (all(pixelCoord >= 0) && all(pixelCoord < gOutputSize) && clip.z > 0)
        InterlockedMax(gDebugImage[pixelCoord.y * gOutputSize.x + pixelCoord.x], iteration);
}

#ifdef g2DConstraint
typedef float2 ConstraintReturnType;
#else
typedef float ConstraintReturnType;
#endif

[ForwardDifferentiable]
ConstraintReturnType GetConstraint(const float3 dir0, const float3 dir1) {
#ifdef g2DConstraint
    float phi0 = atan2(dir0.z, dir0.x);
    float phi1 = atan2(dir1.z, dir1.x);
    if (phi0 < 0) phi0 += M_2PI;
    if (phi1 < 0) phi1 += M_2PI;

    float dphi = phi1 - phi0;
    if (dphi < -M_PI) dphi += M_2PI;
    if (dphi >  M_PI) dphi -= M_2PI;

    const float dtheta = acos(dir0.y) - acos(dir1.y);
	return float2(dtheta, dphi);
#else
	return 1 - dot(dir0, dir1);
#endif
}

[ForwardDifferentiable]
ConstraintReturnType ComputeConstraint(
    const float3 p0,
    const float2 bary,
    const float3 v0, const float3 dPdu, const float3 dPdv,
    const float3 n0, const float3 dNdu, const float3 dNdv,
    const float3 p2, const bool p2sfc,
    const float eta_,
    const uint vertexCount, const uint refractMask ) {

    float3 p = v0 + bary.x * dPdu + bary.y * dPdv;
    float3 n = normalize(n0 + bary.x * dNdu + bary.y * dNdv);

    float3 dirIn = normalize(p0 - p);
	float cosIn = dot(n, dirIn);

    float eta = eta_;

    [ForceUnroll]
    for (uint i = 0; i < gMaxManifoldIntermediateVertices; i++) {
        if (i+1 >= vertexCount)
            break;

        float3 dirOut;
        if ((refractMask & (1 << i)) == 0)
            dirOut = normalize(-dirIn + 2 * cosIn * n);
        else {
            const float etaRcp = cosIn < 0 ? eta : 1 / eta;
            const float refractedCosOut2 = 1 - (1 - (cosIn * cosIn)) * (etaRcp * etaRcp);
            if (refractedCosOut2 <= 0)
                return no_diff POS_INFINITY;
            dirOut = normalize(-dirIn * etaRcp + (abs(cosIn) * etaRcp - sqrt(refractedCosOut2)) * n * sign(cosIn));
        }

        const PathVertex v = no_diff TraceRay(MakeRay(OffsetRayOrigin(p, n, dirOut), dirOut), true);
        if (!v.mIsSurface)
            return no_diff POS_INFINITY;

        if (IsDebugPixel)
            WriteDebugPixel(v.mPosition);

        const float3 vp = p - v.triP0;
        const float3 h = cross(dirOut, v.dPdu);
        const float3 q = cross(vp, v.dPdv);
        const float2 bv = float2(dot(dirOut, q), dot(vp, h)) / dot(v.dPdv, h);
        p = v.triP0 + v.dPdu * bv.x + v.dPdv * bv.y;
        n = normalize(v.triN0 + v.dNdu * bv.x + v.dNdv * bv.y);
        dirIn = -dirOut;
        cosIn = dot(dirIn, n);
        eta = no_diff v.mMaterial.Eta();
	}

	// compute constraint at p2 - p

    const float3 dirOut = p2sfc ? normalize(p2 - p) : p2;
    if ((refractMask & (1 << 0)) == 0) {
        // reflect
        return GetConstraint(dirOut, normalize(-dirIn + 2 * cosIn * n));
    } else {
        // refract
        float etaRcp = 1 / eta;
        if (cosIn < 0) {
            const float tmp = eta;
            eta = etaRcp;
            etaRcp = tmp;
        }
        const float cosOut = dot(dirOut, n);
        const float refractedCosOut2 = 1 - (1 -  (cosIn * cosIn))  * (etaRcp * etaRcp);
        const float refractedCosIn2  = 1 - (1 - (cosOut * cosOut)) * (etaRcp * etaRcp);
        if (refractedCosOut2 > 0)
            return GetConstraint(dirOut, normalize(-dirIn * etaRcp + (abs(cosIn) * etaRcp - sqrt(refractedCosOut2)) * n   * sign(cosIn)));
        else if (refractedCosIn2 > 0)
            return GetConstraint(dirIn, normalize(-dirOut * eta    + (abs(cosOut) * eta   - sqrt(refractedCosIn2))  * (-n * sign(cosOut))));
        else
            return no_diff POS_INFINITY;
    }
}

float2 ComputeStep(const float3 start, inout PathVertex vertex, const uint vertexCount, const uint refractMask, const float3 end, bool endIsSurface, out float C) {
#ifdef g2DConstraint

    const DifferentialPair<float2> dCdu = __fwd_diff(ComputeConstraint)(
        diffPair(start),
        diffPair(vertex.barycentrics, float2(1, 0)),
        diffPair(vertex.triP0), diffPair(vertex.dPdu), diffPair(vertex.dPdv),
        diffPair(vertex.triN0), diffPair(vertex.dNdv), diffPair(vertex.dNdv),
        diffPair(end), endIsSurface,
        diffPair(vertex.mMaterial.Eta()),
        vertexCount, refractMask);

    const DifferentialPair<float2> dCdv = __fwd_diff(ComputeConstraint)(
        diffPair(start),
        diffPair(vertex.barycentrics, float2(0, 1)),
        diffPair(vertex.triP0), diffPair(vertex.dPdu), diffPair(vertex.dPdv),
        diffPair(vertex.triN0), diffPair(vertex.dNdv), diffPair(vertex.dNdv),
        diffPair(end), endIsSurface,
        diffPair(vertex.mMaterial.Eta()),
        vertexCount, refractMask);

    C = length(dCdu.p);

    const float2x2 J = float2x2(dCdu.d.x, dCdv.d.x, dCdu.d.y, dCdv.d.y);
    const float2x2 JtJ = mul(transpose(J), J);
    const float detJtJ = determinant(JtJ);
    if (detJtJ <= 0)
        return 0;
    return mul(mul(inverse(JtJ, detJtJ), transpose(J)), dCdu.p);

#else
    if (gStepMode == StepMode::eFixed) {
		// compute first derivatives

        const DifferentialPair<float> dCdu = __fwd_diff(ComputeConstraint)(
            diffPair(start),
            diffPair(vertex.barycentrics, float2(1, 0)),
            diffPair(vertex.triP0), diffPair(vertex.dPdu), diffPair(vertex.dPdv),
            diffPair(vertex.triN0), diffPair(vertex.dNdv), diffPair(vertex.dNdv),
            diffPair(end), endIsSurface,
            diffPair(vertex.mMaterial.Eta()),
			vertexCount, refractMask);

        const DifferentialPair<float> dCdv = __fwd_diff(ComputeConstraint)(
            diffPair(start),
            diffPair(vertex.barycentrics, float2(0, 1)),
            diffPair(vertex.triP0), diffPair(vertex.dPdu), diffPair(vertex.dPdv),
            diffPair(vertex.triN0), diffPair(vertex.dNdv), diffPair(vertex.dNdv),
            diffPair(end), endIsSurface,
            diffPair(vertex.mMaterial.Eta()),
			vertexCount, refractMask);

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
			diffPair(diffPair(vertex.mMaterial.Eta())),
			vertexCount, refractMask);

		const DifferentialPair<DifferentialPair<float>> dC2dv2 = __fwd_diff(__fwd_diff(ComputeConstraint))(
			diffPair(diffPair(start)),
			diffPair(diffPair(vertex.barycentrics, float2(0, 1)), diffPair(float2(0, 1), float2(0))),
			diffPair(diffPair(vertex.triP0)), diffPair(diffPair(vertex.dPdu)), diffPair(diffPair(vertex.dPdv)),
			diffPair(diffPair(vertex.triN0)), diffPair(diffPair(vertex.dNdv)), diffPair(diffPair(vertex.dNdv)),
			diffPair(diffPair(end)), endIsSurface,
			diffPair(diffPair(vertex.mMaterial.Eta())),
			vertexCount, refractMask);

		const DifferentialPair<DifferentialPair<float>> dC2duv = __fwd_diff(__fwd_diff(ComputeConstraint))(
			diffPair(diffPair(start)),
			diffPair(diffPair(vertex.barycentrics, float2(1, 0)), diffPair(float2(0, 1), float2(0))),
			diffPair(diffPair(vertex.triP0)), diffPair(diffPair(vertex.dPdu)), diffPair(diffPair(vertex.dPdv)),
			diffPair(diffPair(vertex.triN0)), diffPair(diffPair(vertex.dNdv)), diffPair(diffPair(vertex.dNdv)),
			diffPair(diffPair(end)), endIsSurface,
			diffPair(diffPair(vertex.mMaterial.Eta())),
			vertexCount, refractMask);

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
#endif
}

bool ManifoldSolve(const float3 start, const float3 startNg, inout PathVertex vertex, const uint vertexCount, const uint refractMask, const float3 end, bool endIsSurface) {
	#ifdef PRINT_C
    DebugPrintf("------------ Solving ------------\n");
	#endif
    for (iteration = 0; iteration < gManifoldSolverIterations; iteration++) {
        IncrementCounter(DebugCounterType::eManifoldWalkIterations);
        if (IsDebugPixel)
            WriteDebugPixel(vertex.mPosition);

        float C;
        const float2 duv = ComputeStep(start, vertex, vertexCount, refractMask, end, endIsSurface, C);

        if (isinf(C) || isnan(C) || any(isinf(duv)) || any(isnan(duv))) {
            DebugPrintf("\tFailed: invalid constraint (%f, %f %f)\n", C, duv.x, duv.y);
            return false;
        }
        if (abs(C) < gManifoldSolverThreshold) {
            DebugPrintf("\tSucceeded: %f\n", C);
            return true;
        }

		#ifdef PRINT_C
		#ifdef g2DConstraint
		DebugPrintf("\tC: %f\n", C*180*M_1_PI);
		#else
		DebugPrintf("\tC: %f\n", acos(1 - C)*180*M_1_PI);
		#endif
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
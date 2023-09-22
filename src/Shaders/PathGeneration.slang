#pragma once

#include "PathReservoir.slang"
#include "Intersection.slang"
#include "Light.slang"
#include "BRDF.slang"

#ifdef SAMPLE_LIGHTS
static const bool gSampleLights = true;
#else
static const bool gSampleLights = false;
#endif
#ifdef RECONNECTION
static const bool gReconnection = true;
#else
static const bool gReconnection = false;
#endif

uniform float3 gCameraPosition;
uniform float3 gPrevCameraPosition;

uniform float gReconnectionDistance;
uniform float gReconnectionRoughness;
uniform uint gMaxBounces;

float Mis(float a, float b) {
    a = a * a;
    return SafeDivide(a, a + b * b);
}

RWByteAddressBuffer gScratchReconnectionVertices;

ReconnectionVertex LoadRcv(const uint2 id, out float3 throughputAtRcv, out float prefixPdfW) {
    const uint address = FlattenIndex(id) * (sizeof(ReconnectionVertex) + sizeof(float4));
    const float4 data = gScratchReconnectionVertices.Load<float4>(address + sizeof(ReconnectionVertex));
    throughputAtRcv = data.rgb;
    prefixPdfW = data.w;
    return gScratchReconnectionVertices.Load<ReconnectionVertex>(address);
}
void StoreRcv(const uint2 id, const ReconnectionVertex rcv, const float3 throughputAtRcv, const float prefixPdfW) {
    const uint address = FlattenIndex(id) * (sizeof(ReconnectionVertex) + sizeof(float4));
    gScratchReconnectionVertices.Store<ReconnectionVertex>(address, rcv);
    gScratchReconnectionVertices.Store<float4>(address + sizeof(ReconnectionVertex), float4(throughputAtRcv, prefixPdfW));
}


struct PathState<let bShift : bool> {
	float3 throughput;
    float pdfW;

    float3 dirIn;
    uint packedBounces;

    uint seed;
    uint packedId;
    uint pad0;
    uint pad1;

    property uint bounces {
        get { return BF_GET(packedBounces, 0, 8); }
        set { BF_SET(packedBounces, newValue, 0, 8); }
    }
    property uint rcvBounces {
        get { return BF_GET(packedBounces, 8, 8); }
        set { BF_SET(packedBounces, newValue, 8, 8); }
    }
    property bool rcvFound {
        get { return rcvBounces > 0; }
    }
    property uint2 id {
        get { return uint2(packedId & 0xFFFF, packedId >> 16); }
        set { packedId = (newValue.y << 16) | newValue.x; }
	}
};


enum PerVertexRandomNumbers {
    eMerge,
    eTerminate,
	eDirection,
	eDirectLight,
	eNumRandomPerVertex
};

uint4 GetRandomUint<let bShift : bool>(const PathState<bShift> s, const PerVertexRandomNumbers type) {
	RandomSampler rng = RandomSampler(s.seed, s.id);
	rng.Skip(s.bounces * (uint)PerVertexRandomNumbers::eNumRandomPerVertex + (uint)type);
	return rng.Next();
}
float4 GetRandomFloat<let bShift : bool>(const PathState<bShift> s, const PerVertexRandomNumbers type) {
	return UnormToFloat(GetRandomUint<bShift>(s, type));
}


PathSample SampleNextVertex<let bShift : bool>(inout PathState<bShift> s, inout PathVertex vertex, out float samplePdf, out float misWeight) {
	const float prevPdf = s.pdfW;
	const bool prevDiffuse = IsDiffuse(vertex);

	samplePdf = 0;
	misWeight = 0;

	const float contProb = prevDiffuse ? GetContinuationProbability(vertex) : 1;
	if (!bShift && (contProb == 0 || GetRandomFloat(s, PerVertexRandomNumbers::eTerminate).x > contProb)) {
		s.throughput = 0;
		return PathSample();
	}

	DirectionSampleRecord r;
	SampleBrdf(vertex, GetRandomFloat(s, PerVertexRandomNumbers::eDirection), -s.dirIn, r);
	s.throughput *= EvalBrdf(vertex, -s.dirIn, r.mDirection);
	const float brdfPdfW = contProb * EvalBrdfPdfW(vertex, -s.dirIn, r.mDirection);
	s.pdfW *= brdfPdfW;
	if (all(s.throughput <= 0))
		return PathSample();

    if (!bShift && gReconnection && s.rcvFound && s.rcvBounces == s.bounces) {
        float3 throughputAtRcv;
        float prefixPdfW;
        ReconnectionVertex rcv = LoadRcv(s.id, throughputAtRcv, prefixPdfW);
		rcv.mDirOut = r.mDirection;
        throughputAtRcv = s.throughput;
        StoreRcv(s.id, rcv, throughputAtRcv, prefixPdfW);
	}

	s.dirIn = r.mDirection;

	s.bounces++;

	float lightPdfW, dist;
	vertex = TraceRay(MakeRay(OffsetRayOrigin(vertex, s.dirIn), s.dirIn), true, lightPdfW, dist);

	float3 le = vertex.mMaterial.Emission();
	if (vertex.mIsSurface) {
		const float cosIn = -dot(vertex.mShadingNormal, s.dirIn);
		lightPdfW *= sqr(dist) / abs(cosIn);
		if (cosIn <= 0) le = 0;

		// store reconnection vertex if one hasn't been found yet
        if (!bShift && gReconnection && !s.rcvFound && dist >= gReconnectionDistance && vertex.mMaterial.Roughness() >= gReconnectionRoughness && prevDiffuse && (IsDiffuse(vertex) || any(le > 0))) {
            ReconnectionVertex rcv;
			rcv.mVertex = PackVertex(vertex);
			rcv.mCos = abs(cosIn);
			rcv.mDist = dist;
            rcv.mPrefixBounces = s.bounces;
            StoreRcv(s.id, rcv, 0, prevPdf);
            s.rcvBounces = s.bounces;
		}
	}

	if (!any(le > 0))
        return PathSample();

    {
		float prefixPdfW;
		ReconnectionVertex rcv;
		if (!bShift && gReconnection && s.rcvFound) {
			float3 throughputAtRcv;
			rcv = LoadRcv(s.id, throughputAtRcv, prefixPdfW);
			if (rcv.mPrefixBounces == s.bounces)
				rcv.mRadiance = le;
			else
				rcv.mRadiance = le * float3(SafeDivide(s.throughput.r, throughputAtRcv.r), SafeDivide(s.throughput.g, throughputAtRcv.g), SafeDivide(s.throughput.b, throughputAtRcv.b));
		}
		samplePdf = s.pdfW;
		misWeight = gSampleLights ? Mis(brdfPdfW, lightPdfW) : 1;
		return PathSample(s.throughput * le, !bShift && gReconnection && s.rcvFound ? prefixPdfW : s.pdfW, s.bounces, 0, s.seed, s.id, rcv, s.rcvFound);
    }
}

PathSample ConnectToLight<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, out float samplePdf, out float misWeight) {
	samplePdf = 0;
    misWeight = 0;

    float lightPdfW;
    const PathVertex l = SampleEmission(GetRandomFloat(s, PerVertexRandomNumbers::eDirectLight), lightPdfW);

	float3 le = l.mMaterial.Emission();
	if (!any(le > 0))
		return PathSample();

    float3 toLight;
    float dist;
    float AtoW;
    if (l.mIsSurface) {
        toLight = l.mPosition - vertex.mPosition;
        dist = length(toLight);
        toLight /= dist;
        const float cosLight = max(0, -dot(toLight, l.mShadingNormal));
        if (cosLight <= 0)
            return PathSample();
        AtoW = sqr(dist) / cosLight;
        lightPdfW *= AtoW;
    } else {
        toLight = l.mPosition;
        dist = POS_INFINITY;
        AtoW = 1;
    }

    const float3 contrib = s.throughput * le * EvalBrdf(vertex, -s.dirIn, toLight);
    if (!any(contrib > 0))
		return PathSample();

	if (Occluded(vertex, toLight, dist))
		return PathSample();

    samplePdf = s.pdfW * lightPdfW;
    misWeight = Mis(lightPdfW, GetContinuationProbability(vertex) * EvalBrdfPdfW(vertex, -s.dirIn, toLight));

    float prefixPdfW;
    ReconnectionVertex rcv;
	if (!bShift && gReconnection && s.rcvFound) {
		float3 throughputAtRcv;
		rcv = LoadRcv(s.id, throughputAtRcv, prefixPdfW);
		if (rcv.mPrefixBounces == s.bounces) {
			rcv.mDirOut = toLight;
			rcv.mRadiance = le;
		} else
			rcv.mRadiance = float3(SafeDivide(contrib.r, throughputAtRcv.r), SafeDivide(contrib.g, throughputAtRcv.g), SafeDivide(contrib.b, throughputAtRcv.b));
	}

    return PathSample(contrib, !bShift && gReconnection && s.rcvFound ? prefixPdfW : (s.pdfW * AtoW), s.bounces + 1, 1, s.seed, s.id, rcv, s.rcvFound);
}

PathSample ConnectToBasePath<let bShift : bool>(const PathState<bShift> s, const PathVertex vertex, const PathSample basePath) {
	const PathVertex baseRcv = UnpackVertex(basePath.mReconnectionVertex.mVertex);

	float3 toRcv;
	float dist;
	if (baseRcv.mIsSurface) {
		toRcv = baseRcv.mPosition - vertex.mPosition;
		dist = length(toRcv);
		toRcv /= dist;
	} else {
		toRcv = baseRcv.mPosition;
		dist = POS_INFINITY;
	}

	if (dist < gReconnectionDistance)
		return PathSample();

	float3 le = s.throughput * EvalBrdf(vertex, -s.dirIn, toRcv) * basePath.mReconnectionVertex.mRadiance;
	if (!any(le > 0))
		return PathSample();

	if (basePath.mReconnectionVertex.mPrefixBounces < basePath.mBounces)
		le *= EvalBrdf(baseRcv, -toRcv, basePath.mReconnectionVertex.mDirOut);
	else if (basePath.mReconnectionVertex.mPrefixBounces == basePath.mBounces && dot(toRcv, baseRcv.mShadingNormal) > 0)
		return PathSample();

	if (!any(le > 0))
		return PathSample();
	if (Occluded(vertex, toRcv, dist))
		return PathSample();

	PathSample ps = basePath;
	ps.mRadiance = le;
	ps.mReplayPdfW = s.pdfW;
	ps.mReconnectionVertex.mCos = abs(dot(baseRcv.mShadingNormal, toRcv));
	ps.mReconnectionVertex.mDist = dist;
	return ps;
}


void ProcessCanonicalSample(const uint2 id, const float mergeRnd, const PathSample ps, const float samplePdf, const float misWeight) {
    if (samplePdf <= 0)
        return;

	PathReservoir c = PathReservoir(ps, 1 / samplePdf, 1);
	c.PrepareMerge(misWeight);

	PathReservoir res = LoadReservoir(gPathReservoirsOut, id);
	res.Merge(mergeRnd, c);
	StoreReservoir(gPathReservoirsOut, id, res);
}
void SampleCanonicalPath(PathVertex vertex, const uint2 id) {
    PathState<false> s;
    s.dirIn = normalize(vertex.mPosition - gCameraPosition);
    s.seed = gRandomSeed;
    s.id = id;
    s.throughput = 1;
    s.pdfW = 1;
    s.bounces = 0;
    s.rcvBounces = 0;
    while (s.bounces + 1 <= gMaxBounces && any(s.throughput > 0)) {
        const float2 mergeRnds = GetRandomFloat(s, PerVertexRandomNumbers::eMerge).xy;

        if (IsDiffuse(vertex)) {
            if (gSampleLights) {
        		float samplePdf, misWeight;
                const PathSample ps = ConnectToLight(s, vertex, samplePdf, misWeight);
                ProcessCanonicalSample(id, mergeRnds.y, ps, samplePdf, misWeight);
            }
		}

        {
            float samplePdf, misWeight;
            const PathSample ps = SampleNextVertex(s, vertex, samplePdf, misWeight);
            ProcessCanonicalSample(id, mergeRnds.x, ps, samplePdf, misWeight);
    	}
    }

	PathReservoir res = LoadReservoir(gPathReservoirsOut, id);
    res.FinalizeMerge();
	res.mM = 1;
	if (gReconnection && res.mW > 0 && res.mSample.mReconnectionVertex.mPrefixBounces > 0)
		IncrementCounter(DebugCounterType::eReconnectionVertices);
	StoreReservoir(gPathReservoirsOut, id, res);
}

PathSample ShiftPath(const PathSample basePath, const PackedVertex dstVertex, const float3 cameraPos, out float jacobian) {
    jacobian = 0;

    PathVertex vertex = UnpackVertex(dstVertex);
    if (!vertex.mIsSurface)
        return PathSample();

    IncrementCounter(DebugCounterType::eShiftAttempts);
    if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0)
        IncrementCounter(DebugCounterType::eReconnectionAttempts);

    PathSample res = PathSample();

    PathState<true> s;
    s.dirIn = normalize(vertex.mPosition - cameraPos);
    s.seed = basePath.mRngSeed;
    s.id = basePath.mRngIdx;
    s.throughput = 1;
    s.pdfW = 1;
    s.bounces = 0;
    s.rcvBounces = 0;
    while (s.bounces + 1 <= basePath.mBounces && any(s.throughput > 0)) {
        const bool diffuse = IsDiffuse(vertex);

        if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0 && s.bounces + 1 == basePath.mReconnectionVertex.mPrefixBounces) {
			if (diffuse)
				res = ConnectToBasePath(s, vertex, basePath);
			break;
        }

		float samplePdf, misWeight;
        if (gSampleLights && basePath.mIsNEE && s.bounces + 1 == basePath.mBounces) {
			if (diffuse)
                res = ConnectToLight(s, vertex, samplePdf, misWeight);
            break;
		}

        res = SampleNextVertex(s, vertex, samplePdf, misWeight);
        if (s.bounces != basePath.mBounces || basePath.mIsNEE)
            res.mRadiance = 0;
	}

    if (any(res.mRadiance > 0)) {
		IncrementCounter(DebugCounterType::eShiftSuccesses);
		jacobian = basePath.mReplayPdfW / res.mReplayPdfW;
        if (gReconnection && basePath.mReconnectionVertex.mPrefixBounces > 0) {
			IncrementCounter(DebugCounterType::eReconnectionSuccesses);
			if (basePath.mReconnectionVertex.mVertex.mInstanceIndex != INVALID_INSTANCE)
            	jacobian *= (res.mReconnectionVertex.mCos / basePath.mReconnectionVertex.mCos) * sqr(basePath.mReconnectionVertex.mDist / res.mReconnectionVertex.mDist);
        }
    }

    return res;
}
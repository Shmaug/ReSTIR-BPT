#include "ReservoirIO.slang"

Texture2D<uint4> gVertices;
Texture2D<uint4> gPrevVertices;
RWByteAddressBuffer gPrevReservoirs;
#ifdef gUseDiscardMask
RWTexture2D<float> gHistoryDiscardMask;
#endif
uniform float4x4 gPrevMVP;
uniform float gMCap;

void DoTemporalReuse(uint2 id, inout PathReservoir r) {
    const PackedVertex vertex = reinterpret<PackedVertex>(gVertices[id]);
    if (vertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    const float4 prevClipPos = mul(gPrevMVP, float4(vertex.mPosition, 1));
    int2 prevPixel = (int2)(((prevClipPos.xy / prevClipPos.w) * .5 + .5) * (float2)gOutputSize);
    if (any(prevPixel < 0) || any(prevPixel >= gOutputSize))
        return;

	#ifdef gCombinedSpatialTemporalReuse
	{
		RandomSampler tmp = RandomSampler(gRandomSeed, id);
		tmp.Skip(0xFFFFF);
		prevPixel += int2(SampleConcentricDisc(tmp.NextFloat().xy) * gTemporalReuseRadius);
	}
	#endif

    const PackedVertex prevVertex = reinterpret<PackedVertex>(gPrevVertices[prevPixel]);
    if (prevVertex.mInstanceIndex == INVALID_INSTANCE)
        return;

    DebugPrintf("\n\n---------- Performing temporal reuse (%u) ----------\n\n", gRandomSeed);

    PathReservoir candidate = LoadReservoir(gPrevReservoirs, prevPixel);
    if (!(candidate.mM > 0) || isinf(candidate.mM))
        return;

    IncrementCounter(DebugCounterType::eTemporalSamples);

    const float candidateTargetPdf = candidate.mSample.GetTargetPdf();

    float jacobianToCenter = 0;
	float jacobianToPrev = 0;

    if (candidateTargetPdf > 0) {
        DebugPrintf("Shifting candidate to center\n");
		candidate.mSample = ShiftPath(candidate.mSample, vertex, gCameraPosition, jacobianToCenter);
		if (!(jacobianToCenter > 0)) {
			#ifdef gUseDiscardMask
			gHistoryDiscardMask[id] += candidateTargetPdf * candidate.mM;
			#endif
		}
    }

    DebugPrintf("r.f %f, r.W: %f\ncandidate.f: %f -> %f, candidate.W %f, jacobianToCenter: %f\n", r.mSample.GetTargetPdf(), r.mW, candidateTargetPdf, candidate.mSample.GetTargetPdf(), candidate.mW, jacobianToCenter)

    #ifdef TALBOT_RMIS_TEMPORAL
    // apply mis weight for r.mSample
    {
        float mi = r.mM * r.mSample.GetTargetPdf();
        float msum = mi;
        if (any(r.mSample.mRadiance > 0)) {
            DebugPrintf("Shifting center to candidate\n");
            PathSample shifted = ShiftPath(r.mSample, prevVertex, gPrevCameraPosition, jacobianToPrev);
            msum += candidate.mM * shifted.GetTargetPdf() * jacobianToPrev;
			if (!(jacobianToPrev > 0)) {
				#ifdef gUseDiscardMask
				gHistoryDiscardMask[id] += r.mSample.GetTargetPdf() * candidate.mM;
				#endif
			}
        }

        r.PrepareMerge(SafeDivide(mi, msum));
    }
    // apply mis weight for shifted sample
    {
        float msum = r.mM * candidate.mSample.GetTargetPdf() * jacobianToCenter;
        float mi = SafeDivide(candidate.mM * candidateTargetPdf, jacobianToCenter);
        msum += mi;

        candidate.PrepareMerge(SafeDivide(mi, msum), jacobianToCenter);
    }
	#else
    candidate.PrepareMerge(candidate.mM / (candidate.mM + r.mM), jacobianToCenter);
	if (r.mW > 0) {
		DebugPrintf("Shifting center to candidate\n");
        const float shiftedTargetPdf = ShiftPath(r.mSample, prevVertex, gPrevCameraPosition, jacobianToPrev).GetTargetPdf();
        r.PrepareMerge(r.mM / (r.mM + (shiftedTargetPdf * jacobianToPrev > 0 ? candidate.mM : 0)));
        if (jacobianToPrev <= 0) {
			#ifdef gUseDiscardMask
			gHistoryDiscardMask[id] += r.mSample.GetTargetPdf() * candidate.mM;
			#endif
		}
    }
    #endif

	DebugPrintf("r ris w: %f\tcandidate ris w: %f\n", r.mW, candidate.mW)

    RandomSampler rng = RandomSampler(gRandomSeed, id);
	rng.Skip(10000);

    if (r.Merge(rng.NextFloat().x, candidate))
        IncrementCounter(DebugCounterType::eTemporalAcceptances);

    r.FinalizeMerge();
}

[shader("compute")]
[numthreads(8,4,1)]
void TemporalReuse(uint3 index: SV_DispatchThreadID) {
    const uint2 id = index.xy;
    if (any(id >= gOutputSize)) return;
    InitDebugPixel(id, gOutputSize);

    PathReservoir r = LoadReservoir(gPathReservoirsIn, id);

    DoTemporalReuse(id, r);

    if (gMCap > 0) r.mM = min(r.mM, gMCap);

    StoreReservoir(gPathReservoirsOut, id, r);
}
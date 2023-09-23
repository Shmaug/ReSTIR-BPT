#pragma once

interface ReservoirSample {
    float GetTargetPdf();
};
struct Reservoir<T : ReservoirSample> {
    T mSample;
    float mW;
    float mM;

	__init() {
		mW = 0;
        mM = 0;
    }
    __init(const T s, const float W, const float M) {
        mSample = s;
        mW = W;
        mM = M;
    }

	[mutating]
    void PrepareMerge(float misWeight = 1, float jacobian = 1) {
        mW *= mSample.GetTargetPdf() * misWeight * jacobian;
    }

    // note: PrepareMerge must be called on both reservoirs prior to calling Merge.
	// FinalizeMerge must be called after merging.
	[mutating]
    bool Merge(float rnd, Reservoir<T> r) {
        mM += r.mM;

        if (r.mW <= 0 || isnan(r.mW) || isinf(r.mW))
            return false;

        mW += r.mW;
        if (rnd*mW < r.mW) {
            mSample = r.mSample;
            return true;
        }

        return false;
    }

	[mutating]
    void FinalizeMerge() {
        const float p = mSample.GetTargetPdf();
        if (p > 0)
            mW /= p;
        else
            mW = 0;
    }
};

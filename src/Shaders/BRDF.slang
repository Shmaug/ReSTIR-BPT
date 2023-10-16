#pragma once

#include "Microfacet.slang"
#include "PathVertex.slang"
#include "Sampling.slang"

struct ReflectanceEvalRecord {
    float3 mReflectance;
    float mFwdPdfW;
    float mRevPdfW;
    __init() {
        mReflectance = 0;
		mFwdPdfW = 0;
		mRevPdfW = 0;
	}
};
struct DirectionSampleRecord {
    float3 mDirection;
    float mRoughness;
    float mEta;
    uint mLobeIndex;

    float3 mReflectance;
    float mFwdPdfW;
    float mRevPdfW;

    [mutating]
    void SetReflectance(const ReflectanceEvalRecord r) {
        mReflectance = r.mReflectance;
        mFwdPdfW = r.mFwdPdfW;
		mRevPdfW = r.mRevPdfW;
	}
};

#ifdef FORCE_LAMBERTIAN

ReflectanceEvalRecord EvalBrdf(const PackedMaterialParameters vertex, const float3 dirIn, const float3 dirOut, const bool Adjoint = false) {
    if (sign(dirOut.z * dirIn.z) < 0) {
        return ReflectanceEvalRecord();
    }
    ReflectanceEvalRecord r;
	r.mReflectance = (vertex.BaseColor() / M_PI) * abs(dirOut.z);
	r.mFwdPdfW = CosHemispherePdfW(max(0, dirOut.z * sign(dirIn.z)));
	r.mRevPdfW = CosHemispherePdfW(max(0,  dirIn.z * sign(dirOut.z)));
    return r;
}
DirectionSampleRecord SampleBrdf(const PackedMaterialParameters vertex, const float4 rnd, const float3 dirIn, const bool adjoint = false) {
    DirectionSampleRecord r;
	r.mDirection = SampleCosHemisphere(rnd.xy);
	if (dirIn.z < 0)
		r.mDirection = -r.mDirection;
    r.mRoughness = 1;
    r.mEta = 0;
    r.mLobeIndex = 0;
    r.SetReflectance(EvalBrdf(vertex, dirIn, r.mDirection, adjoint));
    return r;
}
bool IsDiffuse(const PathVertex vertex) {
    return true;
}

#else

struct DisneyBSDF {
    float diffuse_weight;
    float sheen_weight;
	float specular_weight;
	float clearcoat_weight;
	float glass_weight;

	float diffuse_prob;
	float specular_prob;
	float clearcoat_prob;
	float glass_prob;

    float aspect;
	float min_alpha;
    float alpha;
	float alpha_x;
	float alpha_y;
    float alpha_c;
    float3 Ctint;
    float eta;

    float3 localDirIn;
    float3 localDirOut;
    float3 half_vector;
    float h_dot_in;
    float h_dot_out;

    property float n_dot_in  { get { return abs(localDirIn.z); } }
    property float n_dot_out { get { return abs(localDirOut.z); } }
    property float n_dot_h   { get { return abs(half_vector.z); } }

	enum LobeFlags {
		eDiffuse      = 1,
		eSpecular     = 2,
		eClearcoat    = 4,
        eTransmission = 8,
        eAll          = 0xF
	};

    __init(const PackedMaterialParameters sd, const float3 localDirIn, Optional<float3> localDirOut_ = none) {
        const float m = sd.Metallic();
        const float t = sd.Transmission();
        const float s = sd.Specular();
        const float mt = (1 - t) * (1 - m);
        diffuse_weight   = lerp(0, 1 - s, mt);
		specular_weight  = lerp(1, s, mt);
		glass_weight     = lerp(t, 0, m);
		sheen_weight     = sd.Sheen() * diffuse_weight;
		clearcoat_weight = sd.Clearcoat();

		const float norm = 1 / (diffuse_weight + specular_weight + glass_weight + clearcoat_weight);
		diffuse_prob   =   diffuse_weight * norm;
		specular_prob  =  specular_weight * norm;
		glass_prob     =     glass_weight * norm;
        clearcoat_prob = clearcoat_weight * norm;

        this.localDirIn = localDirIn;

		// only glass on the inside
        if (localDirIn.z < 0) {
            diffuse_prob   = 0;
            specular_prob  = 0;
            clearcoat_prob = 0;
            if (glass_prob > 0)
                glass_prob = 1;
        }

        aspect = sqrt(1 - sd.Anisotropic() * 0.9);
        min_alpha = 1e-4;
        alpha = sqr(sd.Roughness());
        alpha_x = max(min_alpha, alpha / aspect);
        alpha_y = max(min_alpha, alpha * aspect);
        alpha_c = lerp(0.1, 0.001, sd.ClearcoatGloss());
        const float3 baseColor = sd.BaseColor();
        Ctint = Luminance(baseColor) > 1e-2 ? baseColor / Luminance(baseColor) : 1;

        eta = localDirIn.z > 0 ? sd.Eta() : 1 / sd.Eta();

        if (localDirOut_.hasValue) {
            localDirOut = localDirOut_.value;

            if (localDirIn.z * localDirOut.z < 0) {
                half_vector = normalize(localDirIn + localDirOut * eta);
            } else {
                half_vector = normalize(localDirIn + localDirOut);
            }

            // make half vector on the same side as localDirIn
            if (half_vector.z * localDirIn.z < 0) {
                half_vector = -half_vector;
            }

            h_dot_in = dot(half_vector, localDirIn);
            h_dot_out = dot(half_vector, localDirOut);
		}
    }

	float3 EvalDiffuse(const PackedMaterialParameters sd, out float fwdPdf, out float revPdf) {
		const float Fd90 = float(0.5) + 2 * sd.Roughness() * h_dot_out * h_dot_out;
		const float schlick_n_dot_out = pow5(1 - n_dot_out);
		const float schlick_n_dot_in  = pow5(1 - n_dot_in);
		const float schlick_h_dot_out = pow5(1 - h_dot_out);
		const float base_diffuse = (1 + (Fd90 - 1) * schlick_n_dot_out) * (1 + (Fd90 - 1) * schlick_n_dot_in);

		// The subsurface model
		// Disney's hack to increase the response at grazing angle
		const float Fss90 = h_dot_out * h_dot_out * sd.Roughness();
		const float Fss = (1 + (Fss90 - 1) * schlick_n_dot_out) * (1 + (Fss90 - 1) * schlick_n_dot_in);
		// Lommel-Seeliger law (modified/rescaled)
		const float ss = float(1.25) * (Fss * (1 / (n_dot_out + n_dot_in) - float(0.5)) + float(0.5));

		fwdPdf = CosHemispherePdfW(n_dot_out);
		revPdf = CosHemispherePdfW(n_dot_in);

		return sd.BaseColor() * (lerp(base_diffuse, ss, sd.Subsurface()) / M_PI) * n_dot_out;
    }
    float3 EvalSheen(const PackedMaterialParameters sd, out float fwdPdf, out float revPdf) {
        const float3 Csheen = lerp(1, Ctint, sd.SheenTint());
        const float schlick_h_dot_out = pow5(1 - h_dot_out);

		fwdPdf = CosHemispherePdfW(n_dot_out);
		revPdf = CosHemispherePdfW(n_dot_in);

        return Csheen * schlick_h_dot_out * n_dot_out;
    }
    float3 EvalSpecular(const PackedMaterialParameters sd, out float fwdPdf, out float revPdf) {
        const float spec_f0 = sqr(eta - 1) / sqr(eta + 1);
        const float3 spec_color = lerp(1, Ctint, sd.SpecularTint());
        const float3 Cspec0 = lerp(sd.Specular() * spec_f0 * spec_color, sd.BaseColor(), sd.Metallic());

        const float3 F    = FresnelSchlick(Cspec0, h_dot_out);
        const float D     = GTR2(half_vector, alpha_x, alpha_y);
        const float G_in  = SmithMaskingGTR2(localDirIn , alpha_x, alpha_y);
        const float G_out = SmithMaskingGTR2(localDirOut, alpha_x, alpha_y);
        const float G = G_in * G_out;

        fwdPdf = D * G_in / (4 * n_dot_in);
        revPdf = D * G_out / (4 * n_dot_out);

       	return F * D * G / (4 * n_dot_in);
    }
    float3 EvalClearcoat(const PackedMaterialParameters sd, out float fwdPdf, out float revPdf) {
        const float Fc = FresnelSchlick(float(0.04), h_dot_out);
        // Generalized Trowbridge-Reitz distribution
        const float Dc = GTR1(n_dot_h, alpha_c);
        // SmithG with fixed alpha
        const float Gc_in  = SmithMaskingGTR1(localDirIn);
        const float Gc_out = SmithMaskingGTR1(localDirOut);
        const float Gc = Gc_in * Gc_out;

        fwdPdf = Dc * n_dot_h / (4 * h_dot_out);
        revPdf = Dc * n_dot_h / (4 * h_dot_in);
        return Fc * Dc * Gc / (4 * n_dot_in);
    }
    float3 EvalGlass(const PackedMaterialParameters sd, const bool adjoint, out float fwdPdf, out float revPdf) {
		const float F     = FresnelDielectric(h_dot_in, eta);
		const float Fo    = FresnelDielectric(h_dot_out, 1/eta);
		const float D     = GTR2(half_vector, alpha_x, alpha_y);
		const float G_in  = SmithMaskingGTR2(localDirIn , alpha_x, alpha_y);
		const float G_out = SmithMaskingGTR2(localDirOut, alpha_x, alpha_y);
		const float G = G_in * G_out;

        if (localDirIn.z * localDirOut.z < 0) {
            const float sqrt_denom = h_dot_in + eta * h_dot_out;
            const float eta_factor = adjoint ? (1 / sqr(eta)) : 1;
            const float dh_dout = sqr(eta) * abs(h_dot_out) / sqr(sqrt_denom);
            const float dh_in = sqr(1/eta) * abs(h_dot_in) / sqr(sqrt_denom);
            fwdPdf = (1 - F) * D * G_in * abs(dh_dout * h_dot_in / n_dot_in);
            revPdf = (1 - Fo) * D * G_out * abs(dh_in * h_dot_out / n_dot_out);
			return sqrt(sd.BaseColor()) * ((eta_factor * (1 - F) * D * G * sqr(eta) * abs(h_dot_out * h_dot_in)) / (n_dot_in * sqr(sqrt_denom)));
        } else {
            fwdPdf = F * D * G_in / (4 * abs(n_dot_in));
            revPdf = Fo * D * G_out / (4 * abs(n_dot_out));
			return sd.BaseColor() * ((F * D * G) / (4 * abs(n_dot_in)));
		}
    }

    ReflectanceEvalRecord Eval(const PackedMaterialParameters sd, const bool adjoint) {
        ReflectanceEvalRecord r;
        r.mReflectance = 0;
        r.mFwdPdfW = 0;
        r.mRevPdfW = 0;

        float tmpFwd, tmpRev;

        if (glass_weight > 0) {
            r.mReflectance += glass_weight * EvalGlass(sd, adjoint, tmpFwd, tmpRev);
            r.mFwdPdfW += glass_prob * tmpFwd;
            r.mRevPdfW += glass_prob * tmpRev;
        }

        if (localDirIn.z >= 0 && localDirOut.z >= 0) {
            // Diffuse
            if (diffuse_weight > 0) {
                r.mReflectance += diffuse_weight * EvalDiffuse(sd, tmpFwd, tmpRev);
                r.mFwdPdfW += diffuse_prob * tmpFwd;
                r.mRevPdfW += diffuse_prob * tmpRev;
            }

            // Sheen
            if (sheen_weight > 0) {
                r.mReflectance += sheen_weight * EvalSheen(sd, tmpFwd, tmpRev);
            }

            // Metallic
            if (specular_weight > 0) {
                r.mReflectance += specular_weight * EvalSpecular(sd, tmpFwd, tmpRev);
                r.mFwdPdfW += specular_prob * tmpFwd;
                r.mRevPdfW += specular_prob * tmpRev;
            }

            // Clearcoat
            if (clearcoat_weight > 0) {
                r.mReflectance += clearcoat_weight * EvalClearcoat(sd, tmpFwd, tmpRev);
                r.mFwdPdfW += clearcoat_prob * tmpFwd;
                r.mRevPdfW += clearcoat_prob * tmpRev;
            }
        }

		return r;
    }

    [mutating]
    DirectionSampleRecord Sample(const PackedMaterialParameters sd, const float4 rnd, const bool adjoint) {
        DirectionSampleRecord r;
        r.SetReflectance(ReflectanceEvalRecord());

        if (rnd.z <= glass_prob + specular_prob) {
            // glass / specular

            half_vector = SampleVisibleNormals(localDirIn, alpha_x, alpha_y, rnd.xy);
            if (half_vector.z < 0) {
                half_vector = -half_vector;
            }

            h_dot_in = dot(half_vector, localDirIn);
            float F = FresnelDielectric(h_dot_in, eta);
            if (rnd.z <= glass_prob && rnd.z > F) {
                // Refraction
                float h_dot_out_sq = 1 - (1 - sqr(h_dot_in)) / sqr(eta);
                if (h_dot_out_sq <= 0) {
                    return r;
                }
                // flip half_vector if needed
                if (h_dot_in < 0) {
                    half_vector = -half_vector;
                }
                h_dot_out = sqrt(h_dot_out_sq);
                localDirOut = -localDirIn / eta + (abs(h_dot_in) / eta - h_dot_out) * half_vector;
                r.mEta = eta;
                r.mLobeIndex = (uint)LobeFlags::eTransmission;
            } else {
                // Reflection
                localDirOut = normalize(-localDirIn + 2 * h_dot_in * half_vector);
                h_dot_out = dot(localDirOut, half_vector);
                r.mEta = 0;
                r.mLobeIndex = (uint)LobeFlags::eSpecular;
            }
            r.mRoughness = sd.Roughness();
        } else if (rnd.z <= glass_prob + specular_prob + clearcoat_prob) {
            // clearcoat

            float alpha2 = sqr(alpha_c);
            float cos_h_elevation = sqrt(max(0, (1 - pow(alpha2, 1 - rnd.x)) / (1 - alpha2)));
            float sin_h_elevation = sqrt(max(1 - sqr(cos_h_elevation), 0));
            float h_azimuth = 2 * M_PI * rnd.y;
            half_vector = float3(
                sin_h_elevation * cos(h_azimuth),
                sin_h_elevation * sin(h_azimuth),
                cos_h_elevation);

            h_dot_in = dot(half_vector, localDirIn);
            localDirOut = normalize(-localDirIn + 2 * h_dot_in * half_vector);
            h_dot_out = dot(half_vector, localDirOut);
            r.mEta = 0;
            r.mRoughness = alpha_c;
            r.mLobeIndex = (uint)LobeFlags::eClearcoat;
        } else {
            // diffuse
            if (diffuse_prob <= 0)
                return r;

            localDirOut = SampleCosHemisphere(rnd.xy);
            half_vector = normalize(localDirIn + localDirOut);
            h_dot_in  = dot(half_vector, localDirIn);
            h_dot_out = dot(half_vector, localDirOut);
            r.mEta = 0;
            r.mRoughness = 1;
            r.mLobeIndex = (uint)LobeFlags::eDiffuse;
        }
        r.SetReflectance(Eval(sd, adjoint));
        r.mDirection = localDirOut;
        return r;
	}
}

ReflectanceEvalRecord EvalBrdf(const PackedMaterialParameters sd, const float3 localDirIn, const float3 localDirOut, bool adjoint = false) {
    const DisneyBSDF bsdf = DisneyBSDF(sd, localDirIn, localDirOut);
    return bsdf.Eval(sd, adjoint);
}
DirectionSampleRecord SampleBrdf(const PackedMaterialParameters sd, const float4 rnd, const float3 localDirIn, const bool adjoint = false) {
    DisneyBSDF bsdf = DisneyBSDF(sd, localDirIn);
    return bsdf.Sample(sd, rnd, adjoint);
}
bool IsDiffuse(const PathVertex vertex) {
    return max(max(vertex.mMaterial.Specular(), vertex.mMaterial.Metallic()), vertex.mMaterial.Transmission()) * (1 - vertex.mMaterial.Roughness()) < 0.9;
}

#endif

float GetContinuationProbability(const PathVertex vertex) {
	#ifdef DISABLE_STOCHASTIC_TERMINATION
    return any(vertex.mMaterial.BaseColor() > 0) ? 1 : 0;
	#else
    return max3(vertex.mMaterial.BaseColor()) * 0.75;
	#endif
}
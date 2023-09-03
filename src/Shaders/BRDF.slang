#pragma once

#include "ShadingData.slang"
#include "Sampling.slang"
#include "Microfacet.slang"

struct DirectionSampleRecord {
    float3 mDirection;
    float mRoughness;
    float mEta;
};

#ifdef DISNEY_BRDF

float3 EvalBrdfLocal(const GpuMaterial sd, const float3 localdirIn, const float3 localDirOut, bool Adjoint = false) {
    const float aspect = sqrt(1 - sd.Anisotropic() * 0.9);
    const float min_alpha = 1e-4;
    const float alpha = sqr(sd.Roughness());
    const float alpha_x = max(min_alpha, alpha / aspect);
    const float alpha_y = max(min_alpha, alpha * aspect);

    const float metallic = sd.Metallic();

    if (localdirIn.z * localDirOut.z > 0) {
        // Diffuse component
        const float n_dot_in  = abs(localdirIn.z);
        const float n_dot_out = localDirOut.z;
        float3 half_vector = normalize(localdirIn + localDirOut);
        // Flip half-vector if it's below surface
        if (half_vector.z * localdirIn.z < 0) {
            half_vector = -half_vector;
        }
        const float n_dot_h = half_vector.z;
        const float h_dot_in  = dot(half_vector, localdirIn);
        const float h_dot_out = dot(half_vector, localDirOut);

        float3 contrib = 0;

        // For diffuse, metallic, sheen, and clearcoat, the light bounces
        // only at the top of the surface.
        if (localdirIn.z >= 0 && localDirOut.z >= 0) {
            // Diffuse

            // The base diffuse model
            const float diffuse_weight = (1 - sd.Transmission()) * (1 - metallic);
            if (diffuse_weight > 0) {
                const float Fd90 = float(0.5) + 2 * sd.Roughness() * h_dot_out * h_dot_out;
                const float schlick_n_dot_out = pow5(1 - n_dot_out);
                const float schlick_n_dot_in  = pow5(1 - n_dot_in );
                const float schlick_h_dot_out = pow5(1 - h_dot_out);
                const float base_diffuse = (1 + (Fd90 - 1) * schlick_n_dot_out) * (1 + (Fd90 - 1) * schlick_n_dot_in);

                // The subsurface model
                // Disney's hack to increase the response at grazing angle
                const float Fss90 = h_dot_out * h_dot_out * sd.Roughness();
                const float Fss = (1 + (Fss90 - 1) * schlick_n_dot_out) * (1 + (Fss90 - 1) * schlick_n_dot_in);
                // Lommel-Seeliger law (modified/rescaled)
                const float ss = float(1.25) * (Fss * (1 / (n_dot_out + n_dot_in) - float(0.5)) + float(0.5));

                contrib += diffuse_weight * sd.BaseColor() * (lerp(base_diffuse, ss, sd.Subsurface()) / M_PI) * n_dot_out;
            }

            // Sheen
            const float sheen_weight = sd.Sheen() * (1 - metallic);
            float3 baseColor = sd.BaseColor();
            const float3 Ctint = Luminance(baseColor) > 1e-2 ? baseColor / Luminance(baseColor) : 1;
            if (sheen_weight > 0) {
                const float3 Csheen = lerp(1, Ctint, sd.SheenTint());
                const float schlick_h_dot_out = pow5(1 - h_dot_out);
                contrib += sheen_weight * Csheen * schlick_h_dot_out * n_dot_out;
            }


            // Metallic
            const float spec_weight = 1 - sd.Transmission() * (1 - metallic);
            if (spec_weight > 0 && n_dot_in > 0 && h_dot_out > 0 && n_dot_h > 0) {
                const float eta = sd.Eta(); // we're always going inside
                const float spec_f0 = (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
                const float3 spec_color = lerp(1, Ctint, sd.SpecularTint());
                const float3 Cspec0 = lerp(sd.Specular() * spec_f0 * spec_color, sd.BaseColor(), metallic);

                const float3 F    = FresnelSchlick(Cspec0, h_dot_out);
                const float D     = GTR2(half_vector, alpha_x, alpha_y);
                const float G_in  = SmithMaskingGTR2(localdirIn , alpha_x, alpha_y);
                const float G_out = SmithMaskingGTR2(localDirOut, alpha_x, alpha_y);
                const float G = G_in * G_out;

                contrib += spec_weight * F * D * G / (4 * n_dot_in);
            }

            // Clearcoat component
            if (sd.Clearcoat() > 0 && n_dot_in > 0 && n_dot_h > 0) {
                const float alpha_c = lerp(0.1, 0.001, sd.ClearcoatGloss());
                const float Fc = FresnelSchlick(float(0.04), h_dot_out);
                // Generalized Trowbridge-Reitz distribution
                const float Dc = GTR1(n_dot_h, alpha_c);
                // SmithG with fixed alpha
                const float Gc_in  = SmithMaskingGTR1(localdirIn);
                const float Gc_out = SmithMaskingGTR1(localDirOut);
                const float Gc = Gc_in * Gc_out;

                contrib += sd.Clearcoat() * Fc * Dc * Gc / (4 * n_dot_in);
            }
        }

        // Glass
        // For glass, lights bounce at both sides of the surface.
        const float glass_w = (1 - metallic) * sd.Transmission();
        if (glass_w > 0) {
            const float eta = localdirIn.z > 0 ? sd.Eta() : 1 / sd.Eta();
            const float Fg    = FresnelDielectric(h_dot_in, eta);
            const float D     = GTR2(half_vector, alpha_x, alpha_y);
            const float G_in  = SmithMaskingGTR2(localdirIn, alpha_x, alpha_y);
            const float G_out = SmithMaskingGTR2(localDirOut, alpha_x, alpha_y);
            const float G = G_in * G_out;
            contrib += sd.BaseColor() * (glass_w * (Fg * D * G) / (4 * abs(n_dot_in)));
        }
        return contrib;
    } else {
        const float glass_w = (1 - metallic) * sd.Transmission();
        if (glass_w <= 0)
            return 0;

        // Only the glass component for refraction
        const float eta = localdirIn.z > 0 ? sd.Eta() : 1 / sd.Eta();
        float3 half_vector = normalize(localdirIn + localDirOut * eta);
        // Flip half-vector if it's below surface
        if (half_vector.z * localdirIn.z < 0) {
            half_vector = -half_vector;
        }

        const float eta_factor = Adjoint ? (1 / (eta * eta)) : 1;
        const float h_dot_in   = dot(half_vector, localdirIn);
        const float h_dot_out  = dot(half_vector, localDirOut);
        const float sqrt_denom = h_dot_in + eta * h_dot_out;

        const float Fg    = FresnelDielectric(h_dot_in, eta);
        const float D     = GTR2(half_vector, alpha_x, alpha_y);
        const float G_in  = SmithMaskingGTR2(localdirIn, alpha_x, alpha_y);
        const float G_out = SmithMaskingGTR2(localDirOut, alpha_x, alpha_y);
        const float G = G_in * G_out;

        // Burley propose to take the square root of the base color to preserve albedo
        return sqrt(sd.BaseColor()) * (glass_w *
            (eta_factor * (1 - Fg) * D * G * eta * eta * abs(h_dot_out * h_dot_in)) /
            (abs(localdirIn.z) * sqrt_denom * sqrt_denom));
    }
}
float EvalBrdfPdfWLocal(const GpuMaterial sd, const float3 localDirIn, const float3 localDirOut) {
    float alpha = sqr(sd.Roughness());
    float metallic = sd.Metallic();
    float aspect = sqrt(1 - sd.Anisotropic() * float(0.9));
    float min_alpha = float(0.0001);
    float alpha_x = max(min_alpha, alpha / aspect);
    float alpha_y = max(min_alpha, alpha * aspect);
    float alpha_c = (1 - sd.ClearcoatGloss()) * float(0.1) + sd.ClearcoatGloss() * float(0.001);

    float transmission = sd.Transmission();
    float diffuse_weight = (1 - metallic) * (1 - transmission);
    float metallic_weight = (1 - transmission * (1 - metallic));
    float glass_weight = (1 - metallic) * transmission;
    float clearcoat_weight = sd.Clearcoat();
    float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
    float diffuse_prob = diffuse_weight / total_weight;
    float metallic_prob = metallic_weight / total_weight;
    float glass_prob = glass_weight / total_weight;
    float clearcoat_prob = clearcoat_weight / total_weight;

    if (localDirIn.z < 0) {
        // Our incoming ray is coming from inside,
        // so the probability of sampling the glass lobe is 1 if glass_prob is not 0.
        diffuse_prob = 0;
        metallic_prob = 0;
        clearcoat_prob = 0;
        if (glass_prob > 0) {
            glass_prob = 1;
        }
    }

    if (localDirIn.z * localDirOut.z > 0) {
        // For metallic: visible normal sampling -> D * G_in
        float3 half_vector = normalize(localDirIn + localDirOut);
        // Flip half-vector if it's below surface
        if (half_vector.z < 0) {
            half_vector = -half_vector;
        }
        const float n_dot_in  = localDirIn.z;
        const float n_dot_h   = half_vector.z;
        const float h_dot_in  = dot(half_vector, localDirIn);
        const float h_dot_out = dot(half_vector, localDirOut);

        // For diffuse, metallic, and clearcoat, the light bounces
        // only at the top of the surface.
        if (localDirIn.z >= 0 && localDirOut.z >= 0) {
            diffuse_prob *= max(localDirOut.z, float(0)) / M_PI;

            if (n_dot_in > 0) {
                float D    = GTR2(half_vector, alpha_x, alpha_y);
                float G_in = SmithMaskingGTR2(localDirIn, alpha_x, alpha_y);
                metallic_prob *= (D * G_in / (4 * n_dot_in));
            } else {
                metallic_prob = 0;
            }

            // For clearcoat: D importance sampling
            if (n_dot_h > 0 && h_dot_out > 0) {
                float Dc = GTR1(n_dot_h, alpha_c);
                clearcoat_prob *= (Dc * n_dot_h / (4 * h_dot_out));
            } else {
                clearcoat_prob = 0;
            }
        }

        // For glass: F * visible normal
        const float eta  = localDirIn.z > 0 ? sd.Eta() : 1 / sd.Eta();
        const float Fg   = FresnelDielectric(h_dot_in, eta);
        const float D    = GTR2(half_vector, alpha_x, alpha_y);
        const float G_in = SmithMaskingGTR2(localDirIn, alpha_x, alpha_y);
        glass_prob *= (Fg * D * G_in / (4 * abs(n_dot_in)));
    } else {
        // Only glass component for refraction
        const float eta = localDirIn.z > 0 ? sd.Eta() : 1 / sd.Eta();
        float3 half_vector = normalize(localDirIn + localDirOut * eta);
        // Flip half-vector if it's below surface
        if (half_vector.z < 0) {
            half_vector = -half_vector;
        }
        const float h_dot_in = dot(half_vector, localDirIn);
        const float h_dot_out = dot(half_vector, localDirOut);
        const float D    = GTR2(half_vector, alpha_x, alpha_y);
        const float G_in = SmithMaskingGTR2(localDirIn, alpha_x, alpha_y);
        const float Fg   = FresnelDielectric(h_dot_in, eta);
        const float sqrt_denom = h_dot_in + eta * h_dot_out;
        const float dh_dout = eta * eta * h_dot_out / (sqrt_denom * sqrt_denom);
        glass_prob *= (1 - Fg) * D * G_in * abs(dh_dout * h_dot_in / localDirIn.z);
    }

    return diffuse_prob + metallic_prob + glass_prob + clearcoat_prob;
}
void SampleBrdfLocal(const GpuMaterial sd, const float3 rnd, const float3 localDirIn, out DirectionSampleRecord r) {
    float aspect = sqrt(1 - sd.Anisotropic() * float(0.9));
    float min_alpha = float(0.0001);
    const float alpha = sqr(sd.Roughness());
    float alpha_x = max(min_alpha, alpha / aspect);
    float alpha_y = max(min_alpha, alpha * aspect);
    float alpha_c = (1 - sd.ClearcoatGloss()) * float(0.1) + sd.ClearcoatGloss() * float(0.001);

    const float metallic = sd.Metallic();
    const float transmission = sd.Transmission();
    float diffuse_weight = (1 - metallic) * (1 - transmission);
    float metallic_weight = (1 - transmission * (1 - metallic));
    float glass_weight = (1 - metallic) * transmission;
    float clearcoat_weight = sd.Clearcoat();

    // Two cases: 1) if we are coming from "outside" the surface,
    // sample all lobes
    if (localDirIn.z >= 0) {
        float total_weight = diffuse_weight + metallic_weight + glass_weight + clearcoat_weight;
        float diffuse_prob = diffuse_weight / total_weight;
        float metallic_prob = metallic_weight / total_weight;
        float glass_prob = glass_weight / total_weight;
        // float clearcoat_prob = clearcoat_weight / total_weight;
        if (rnd.z <= diffuse_prob) {
            r.mDirection = SampleCosHemisphere(rnd.xy);
            r.mEta = 0;
            r.mRoughness = 1;
            return;
        } else if (rnd.z <= (diffuse_prob + metallic_prob)) { // metallic
            // Visible normal sampling

            // Convert the incoming direction to local coordinates
            float3 local_localDirIn = localDirIn;
            float3 half_vector = SampleVisibleNormals(local_localDirIn, alpha_x, alpha_y, rnd.xy);

            // Reflect over the world space normal
            r.mDirection = normalize(-localDirIn + 2 * dot(localDirIn, half_vector) * half_vector);
            r.mEta = 0;
            r.mRoughness = sd.Roughness();
            return;
        } else if (rnd.z <= (diffuse_prob + metallic_prob + glass_prob)) { // glass
            if (glass_prob <= 0) {
                // Just to be safe numerically.
                r.mDirection = 0;
                return;
            }
            // Visible normal sampling

            // Convert the incoming direction to local coordinates
            float3 local_localDirIn = localDirIn;
            float3 half_vector = SampleVisibleNormals(local_localDirIn, alpha_x, alpha_y, rnd.xy);

            // Now we need to decide whether to reflect or refract.
            // We do this using the Fresnel term.
            float h_dot_in = dot(half_vector, localDirIn);
            float eta = localDirIn.z > 0 ? sd.Eta() : 1 / sd.Eta();
            float F = FresnelDielectric(h_dot_in, eta);
            // rescale rnd_param_w from
            // (diffuse_prob + metallic_prob, diffuse_prob + metallic_prob + glass_prob]
            // to
            // (0, 1]
            float u = (rnd.z - (diffuse_prob + metallic_prob)) / glass_prob;
            if (u <= F) {
                // Reflect over the world space normal
                r.mDirection = normalize(-localDirIn + 2 * dot(localDirIn, half_vector) * half_vector);
                r.mEta = 0;
                return;
            } else {
                // Refraction
                float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
                if (h_dot_out_sq <= 0) {
                    r.mDirection = 0;
                    return;
                }
                // flip half_vector if needed
                if (h_dot_in < 0) {
                    half_vector = -half_vector;
                }
                float h_dot_out= sqrt(h_dot_out_sq);
                r.mDirection = -localDirIn / eta + (abs(h_dot_in) / eta - h_dot_out) * half_vector;
                r.mEta = eta;
            }
            r.mRoughness = sd.Roughness();
            return;
        } else { // clearcoat
            // Only importance sampling D

            // Appendix B.2 Burley's note
            float alpha2 = alpha_c * alpha_c;
            // Equation 5
            float cos_h_elevation = sqrt(max(float(0), (1 - pow(alpha2, 1 - rnd[0])) / (1 - alpha2)));
            float sin_h_elevation = sqrt(max(1 - cos_h_elevation * cos_h_elevation, float(0)));
            float h_azimuth = 2 * M_PI * rnd[1];
            float3 local_micro_normal = float3(
                sin_h_elevation * cos(h_azimuth),
                sin_h_elevation * sin(h_azimuth),
                cos_h_elevation );

            r.mDirection = normalize(-localDirIn + 2 * dot(localDirIn, local_micro_normal) * local_micro_normal);
            r.mEta = 0;
            r.mRoughness = sqrt(alpha_c);
            return;
        }
    } else {
        // 2) otherwise, only consider the glass lobes.

        // Convert the incoming direction to local coordinates
        float3 local_localDirIn = localDirIn;
        float3 local_micro_normal = SampleVisibleNormals(local_localDirIn, alpha_x, alpha_y, rnd.xy);

        // Transform the micro normal to world space
        float3 half_vector = local_micro_normal;
        // Flip half-vector if it's below surface
        if (half_vector.z < 0) {
            half_vector = -half_vector;
        }

        // Now we need to decide whether to reflect or refract.
        // We do this using the Fresnel term.
        float h_dot_in = dot(half_vector, localDirIn);
        float eta = localDirIn.z > 0 ? sd.Eta() : 1 / sd.Eta();
        float F = FresnelDielectric(h_dot_in, eta);
        float u = rnd.z;
        if (u <= F) {
            // Reflect over the world space normal
            r.mDirection = normalize(-localDirIn + 2 * dot(localDirIn, half_vector) * half_vector);
            r.mEta = 0;
        } else {
            // Refraction
            float h_dot_out_sq = 1 - (1 - h_dot_in * h_dot_in) / (eta * eta);
            if (h_dot_out_sq <= 0) {
                r.mDirection = 0;
                return;
            }
            // flip half_vector if needed
            if (h_dot_in < 0) {
                half_vector = -half_vector;
            }
            float h_dot_out = sqrt(h_dot_out_sq);
            r.mDirection = -localDirIn / eta + (abs(h_dot_in) / eta - h_dot_out) * half_vector;
            r.mEta = eta;
        }
        r.mRoughness = sd.Roughness();
        return;
    }
}

float3 EvalBrdf(const ShadingData sd, const GpuMaterial m, const float3 dirIn, const float3 dirOut, const bool Adjoint = false) {
    return EvalBrdfLocal(m, sd.ToLocal(dirIn), sd.ToLocal(dirOut), Adjoint);
}
float EvalBrdfPdfW(const ShadingData sd, const GpuMaterial m, const float3 dirIn, const float3 dirOut) {
    return EvalBrdfPdfWLocal(m, sd.ToLocal(dirIn), sd.ToLocal(dirOut));
}
void SampleBrdf(const ShadingData sd, const GpuMaterial m, const float4 rnd, const float3 dirIn, out DirectionSampleRecord r) {
    SampleBrdfLocal(m, rnd.xyz, sd.ToLocal(dirIn), r);
    r.mDirection = sd.ToWorld(r.mDirection);
}
bool IsDiffuse(const GpuMaterial sd) {
    return max(max(sd.Specular(), sd.Metallic()), sd.Transmission()) * (1 - sd.Roughness()) < 0.75 || any(sd.Emission() > 0);
}
float GetContinuationProbability(const GpuMaterial m) {
    return max3(m.BaseColor()) * .9;
}

#else

// lambertian brdf

float3 EvalBrdf(const ShadingData sd, const GpuMaterial m, const float3 dirIn, const float3 dirOut) {
    float3 n = sd.GetShadingNormal();
    float cosOut = dot(dirOut, n);
    if (sign(cosOut * dot(dirIn, n)) < 0)
        return 0;
    return (m.BaseColor() / M_PI) * abs(cosOut);
}
float EvalBrdfPdfW(const ShadingData sd, const GpuMaterial m, const float3 dirIn, const float3 dirOut) {
    float3 n = sd.GetShadingNormal();
    return CosHemispherePdfW(max(0, dot(dirOut, n)*sign(dot(dirIn, n))));
}
void SampleBrdf(const ShadingData sd, const GpuMaterial m, const float4 rnd, const float3 dirIn, out DirectionSampleRecord r) {
    float3 localDirOut = SampleCosHemisphere(rnd.xy);
    if (dot(dirIn, sd.GetShadingNormal()) < 0)
        localDirOut = -localDirOut;
    r.mDirection = sd.ToWorld(localDirOut);
    r.mRoughness = 1;
    r.mEta = 0;
}
bool IsDiffuse(const GpuMaterial m) {
    return true;
}
float GetContinuationProbability(const GpuMaterial m) {
    return max3(m.BaseColor()) * .9;
}

#endif
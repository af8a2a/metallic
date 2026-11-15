#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_diffuse_lobe.h. Modified by Metallic, 2026.
namespace Metallic.OpenPBR
{
/***************************************************************************
 * Copyright 2026 Adobe
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

//////////////////////////////////////
// EnergyConservingRoughDiffuseLobe //
//////////////////////////////////////

public struct OpenPBREnergyConservingRoughDiffuseLobe
{
    public float3 normal_ff;
    public float3 diffuse_albedo;
    public float diffuse_roughness;
    public float specular_uv_y;
    public float specular_uv_z;
    public float cached_specular_energy_compensation;
    public float eval_specular_energy_compensation;
};

// -------------------------------
// Low-level BRDF implementation.
// -------------------------------

public static const float OpenPBRFONConstantA = 0.5f - 2.0f / (3.0f * OpenPBRPi);
public static const float OpenPBRFONConstantB = 2.0f / 3.0f - 28.0f / (15.0f * OpenPBRPi);

// Fujii Oren-Nayar (FON) directional albedo (exact).

public float openPBREFONExact(const float mu, const float roughness)
{
    const float AF = 1.0f / (1.0f + OpenPBRFONConstantA * roughness); // FON A coeff.
    const float BF = roughness * AF;                                  // FON B coeff.
    const float Si = sqrt(1.0f - (mu * mu));
    const float G =
        Si * (acos(clamp(mu, -1.0f, 1.0f)) - Si * mu) + (2.0f / 3.0f) * ((Si / mu) * (1.0f - (Si * Si * Si)) - Si);
    return AF + (BF * OpenPBRRcpPi) * G;
}

// Fujii Oren-Nayar (FON) directional albedo (approximate).

public float openPBREFONApprox(const float mu, const float roughness)
{
    const float mucomp = 1.0f - mu;
    const float g1 = 0.0571085289f;
    const float g2 = 0.491881867f;
    const float g3 = -0.332181442f;
    const float g4 = 0.0714429953f;
    const float GoverPi = mucomp * (g1 + mucomp * (g2 + mucomp * (g3 + mucomp * g4)));
    return (1.0f + roughness * GoverPi) / (1.0f + OpenPBRFONConstantA * roughness);
}

// Evaluates the Energy-preserving Fujii Oren-Nayar (EON) BRDF value, given inputs:
//          rho = single-scattering albedo parameter
//    roughness = roughness in [0, 1]
//     wi_local = direction of incident ray (directed away from vertex)
//     wo_local = direction of outgoing ray (directed away from vertex)
//        exact = flag to select exact or fast approx. version

public float3 openPBRFEONApprox(const float3 rho, const float roughness, const float mu_i, const float mu_o,
                                const float wi_dot_wo)
{
    const float s = wi_dot_wo - mu_i * mu_o;                                      // QON s term
    const float sovertF = s > 0.0f ? s / max(mu_i, mu_o) : s;                     // FON s/t
    const float AF = 1.0f / (1.0f + OpenPBRFONConstantA * roughness);             // FON A coeff.
    const float3 f_ss = (rho * OpenPBRRcpPi) * AF * (1.0f + roughness * sovertF); // single-scatter lobe
    const float EFo = openPBREFONApprox(mu_o, roughness);                         // FON w_o albedo (approx)
    const float EFi = openPBREFONApprox(mu_i, roughness);                         // FON w_i albedo (approx)
    const float avgEF = AF * (1.0f + OpenPBRFONConstantB * roughness);            // avg. albedo
    const float3 rho_ms = (rho * rho) * avgEF / (float3(1.0f) - rho * (1.0f - avgEF));
    const float Eps = 1.0e-7f;
    const float3 f_ms = (rho_ms * OpenPBRRcpPi) * max(Eps, 1.0f - EFo) // multi-scatter lobe
                        * max(Eps, 1.0f - EFi) / max(Eps, 1.0f - avgEF);
    return f_ss + f_ms;
}

public float3 openPBRFEON(const float3 rho, const float roughness, const float3 wi_local, const float3 wo_local,
                   const bool exact)
{
    if (!exact)
        return openPBRFEONApprox(rho, roughness, wi_local.z, wo_local.z, dot(wi_local, wo_local));
    const float mu_i = wi_local.z;                                                // input angle cos
    const float mu_o = wo_local.z;                                                // output angle cos
    const float s = dot(wi_local, wo_local) - mu_i * mu_o;                        // QON s term
    const float sovertF = s > 0.0f ? s / max(mu_i, mu_o) : s;                     // FON s/t
    const float AF = 1.0f / (1.0f + OpenPBRFONConstantA * roughness);             // FON A coeff.
    const float3 f_ss = (rho * OpenPBRRcpPi) * AF * (1.0f + roughness * sovertF); // single-scatter lobe
    const float EFo = openPBREFONExact(mu_o, roughness);                          // FON w_o albedo (exact)
    const float EFi = openPBREFONExact(mu_i, roughness);                          // FON w_i albedo (exact)
    const float avgEF = AF * (1.0f + OpenPBRFONConstantB * roughness);            // avg. albedo
    const float3 rho_ms = (rho * rho) * avgEF / (float3(1.0f) - rho * (1.0f - avgEF));
    const float Eps = 1.0e-7f;
    const float3 f_ms = (rho_ms * OpenPBRRcpPi) * max(Eps, 1.0f - EFo) // multi-scatter lobe
                        * max(Eps, 1.0f - EFi) / max(Eps, 1.0f - avgEF);
    return f_ss + f_ms;
}

// ---------------------------
// Top-level wrapper function.
// ---------------------------

// This function calculates the diffuse BRDF value adjusted for the specular energy compensation.
// It handles selection of the appropriate BRDF model and conversion of the view and light directions to local space.

public float3 openPBRDiffuseBrdfValueAdjustedForSpecular<TContext : IOpenPBRContext>(
    TContext context, const OpenPBREnergyConservingRoughDiffuseLobe lobe, const float3 view_direction,
    const float3 light_direction)
{
    const float cos_in = dot(view_direction, lobe.normal_ff);
    const float cos_out = dot(light_direction, lobe.normal_ff);
    const float3 brdf_value =
        openPBRFEONApprox(lobe.diffuse_albedo, lobe.diffuse_roughness, cos_in, cos_out, dot(view_direction, light_direction));

    const float uv_cos_out = openPBRRemapExactIndex(openPBRCosThetaToExactIndex(cos_out));
    const float specular_energy_compensation =
        lobe.eval_specular_energy_compensation *
        context.sample3D(3, float3(uv_cos_out, lobe.specular_uv_y, lobe.specular_uv_z)).r;

    return brdf_value * specular_energy_compensation;
}

// ------------------------
// Required lobe functions.
// ------------------------

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(TContext context,
                                                       inout OpenPBREnergyConservingRoughDiffuseLobe lobe,
                                                       const float3 normal_ff, const float3 view_direction,
                                                       const float3 diffuse_albedo, const float diffuse_roughness,
                                                       const float specular_alpha,
                                                       const float specular_eta_t_over_eta_i)
{
    lobe.normal_ff = normal_ff;
    lobe.diffuse_albedo = diffuse_albedo;
    lobe.diffuse_roughness = diffuse_roughness;
    const float uv_alpha = openPBRRemapExactIndex(openPBRAlphaToExactIndex(specular_alpha));
    const float uv_ior = openPBRRemapExactIndex(openPBRIorToExactIndex(specular_eta_t_over_eta_i));
    lobe.specular_uv_y = uv_alpha;
    lobe.specular_uv_z = uv_ior;

    if (all(diffuse_albedo == float3(0))) {
        lobe.cached_specular_energy_compensation = 0.0f;
        lobe.eval_specular_energy_compensation = 0.0f;
        return;
    }

    const float cos_in = dot(view_direction, lobe.normal_ff);
    const float uv_cos_in = openPBRRemapExactIndex(openPBRCosThetaToExactIndex(cos_in));
    const float ior_scale = openPBRExtrapolateTableValueBeyondIorMaxIfNeeded(1.0f, specular_eta_t_over_eta_i);

    // In the terminology of the original ASM ECD lobe, this calculates (1 - L) / (1 - T).
    // (1 - V) is multiplied in later. TODO: Check terminology.
    // This result is calculated assuming an "untinted" white albedo, and then the
    // underlying BRDF value (e.g., rho / pi for Lambert) is multiplied in later.
    const float untinted_numerator = context.sample3D(3, float3(uv_cos_in, uv_alpha, uv_ior)).r * ior_scale;
    const float untinted_denominator = context.sample2D(4, float2(uv_alpha, uv_ior)).r * ior_scale;
    const float untinted = untinted_numerator / openPBRClampAverageEnergyComplementAboveZero(untinted_denominator);
    lobe.cached_specular_energy_compensation = untinted;
    lobe.eval_specular_energy_compensation = untinted * ior_scale;
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context,
                                                    const OpenPBREnergyConservingRoughDiffuseLobe lobe)
{
    return OpenPBRBSDFLobeTypeDiffuse | OpenPBRBSDFLobeTypeReflection;
}

public OpenPBRDiffuseSpecular
openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                      const OpenPBREnergyConservingRoughDiffuseLobe lobe,
                                                      const float3 view_direction, const float3 light_direction)
{
    // Exact zero albedo makes both EON scattering terms zero. Avoid EON and LUT work.
    if (all(lobe.diffuse_albedo == float3(0))) { return openPBRMakeZeroDiffuseSpecular(); }

    const float cos_out = dot(light_direction, lobe.normal_ff);
    if (cos_out <= 0.0f)
        return openPBRMakeZeroDiffuseSpecular();

    return openPBRMakeDiffuseSpecularFromDiffuse(
        openPBRDiffuseBrdfValueAdjustedForSpecular(context, lobe, view_direction, light_direction) * cos_out);
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context,
                                                          const OpenPBREnergyConservingRoughDiffuseLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    return max(dot(light_direction, lobe.normal_ff), 0.0f) * OpenPBRRcpPi;
}

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBREnergyConservingRoughDiffuseLobe lobe,
                                                   const float3 rand, const float3 view_direction,
                                                   out float3 light_direction, out OpenPBRDiffuseSpecular weight,
                                                   out float pdf, out uint sampled_type)
{
    const float3 light_direction_local = openPBRSampleUnitHemisphereCosine((rand).xy);
    const float cos_out = light_direction_local.z;

    // Implementation conventions dictates that the sampling function must return false if the pdf is zero,
    // which could potentially happen in this case if there are any numerical issues in the cosine sampling.
    if (cos_out == 0.0f) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    // Map to world.
    // TODO: Avoid mapping back and forth.
    light_direction = openPBRLocalToWorld(openPBRMakeBasis(lobe.normal_ff), light_direction_local);

    weight = openPBRMakeDiffuseSpecularFromDiffuse(
        openPBRDiffuseBrdfValueAdjustedForSpecular(context, lobe, view_direction, light_direction) *
        OpenPBRPi); // the pdf is cos_out/pi so the cos_out terms cancels out
    pdf = cos_out * OpenPBRRcpPi;
    sampled_type = openPBRGetLobeType(context, lobe);
    return true;
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context,
                                                                  const OpenPBREnergyConservingRoughDiffuseLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    // TODO: Improve.
    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput,
                                                        lobe.diffuse_albedo * lobe.cached_specular_energy_compensation);
}
} // namespace Metallic.OpenPBR

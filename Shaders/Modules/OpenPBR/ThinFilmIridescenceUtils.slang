#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_thin_film_iridescence_utils.h. Modified by Metallic, 2026.
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

// Iridescence-specific constants
public static const float OpenPBRMinDenomMag = 1e-9f; // Minimum denominator magnitude to prevent division by zero

// Struct for polarized scalar quantities
public struct OpenPBRPolarizedFloat
{
    public float s; // S-polarization component
    public float p; // P-polarization component
};

// Struct for polarized complex numbers
public struct OpenPBRPolarizedComplex
{
    public float2 s; // S-polarization complex component
    public float2 p; // P-polarization complex component
};

// Struct for polarized color quantities (per color channel)
public struct OpenPBRPolarizedColor
{
    public float3 s; // S-polarization components for RGB
    public float3 p; // P-polarization components for RGB
};

// Enforce physical branch: choose cos(theta_t) such that Im(eta_t * cos(theta_t)) >= 0

public float2 openPBREnforceDecayingBranch(const float2 eta_t, float2 cos_theta_t)
{
    const float2 eta_t_cos = openPBRComplexMultiply(eta_t, cos_theta_t);
    if (eta_t_cos.y < 0.0f)
        cos_theta_t = openPBRComplexNegate(cos_theta_t);
    return cos_theta_t;
}

// Unified Snell's Law: works for real or complex eta_t. Handles TIR by producing complex cos_theta_t.

public float2 openPBRSnellCosUnified(const float cos_theta_i, const float eta_i, const float2 eta_t)
{
    const float sin_theta_i = sqrt(saturate(1.0f - cos_theta_i * cos_theta_i));

    // Apply Snell's Law: sin(theta_t) = (eta_i / eta_t) * sin(theta_i)
    const float2 eta_i_over_eta_t =
        openPBRComplexDivide(float2(eta_i, 0.0f), eta_t, OpenPBRMinDenomMag, float2(1.0f, 0.0f));

    const float2 sin_theta_t = openPBRComplexScalarMultiply(sin_theta_i, eta_i_over_eta_t);

    // Calculate cos(theta_t) from sin(theta_t) using sin^2 + cos^2 = 1
    const float2 sin_theta_t_squared = openPBRComplexMultiply(sin_theta_t, sin_theta_t);
    const float2 cos_theta_t_squared = openPBRComplexSubtract(float2(1.0f, 0.0f), sin_theta_t_squared);

    float2 cos_theta_t = openPBRComplexSqrt(cos_theta_t_squared);

    // Enforce physical branch selection: choose sign such that Im(eta_t * cos_theta_t) >= 0
    cos_theta_t = openPBREnforceDecayingBranch(eta_t, cos_theta_t);

    return cos_theta_t;
}

// Unified Fresnel amplitude coefficients (s/p), valid for dielectric or conductor eta_t
// Returns complex r and t, and complex cos_theta_t
// Handles total internal reflection (TIR) by returning complex coefficients with phase information

public void openPBRComputeFresnelUnifiedPolarizedAmplitude(
    const float cos_theta_i, const float eta_i,
    const float2 eta_t, // eta_t = n_t + i * k_t (complex IOR for conductors, or n_t + i*0 for dielectrics)
    out OpenPBRPolarizedComplex r, out OpenPBRPolarizedComplex t, out float2 cos_theta_t)
{
    // Apply Snell's Law to compute complex cos(theta_t), which handles TIR by producing complex angles
    cos_theta_t = openPBRSnellCosUnified(cos_theta_i, eta_i, eta_t);

    // Calculate intermediate products used in Fresnel equations
    const float2 eta_i_cos_theta_i = float2(eta_i * cos_theta_i, 0.0f);
    const float2 eta_t_cos_theta_t = openPBRComplexMultiply(eta_t, cos_theta_t);
    const float2 eta_t_cos_theta_i = openPBRComplexScalarMultiply(cos_theta_i, eta_t);
    const float2 eta_i_cos_theta_t = openPBRComplexScalarMultiply(eta_i, cos_theta_t);

    // Transmission numerator is the same for both s and p polarizations
    const float2 t_numerator = float2(2.0f * eta_i * cos_theta_i, 0.0f);

    // For s-polarization:
    // r_s = (eta_i * cos_theta_i - eta_t * cos_theta_t) / (eta_i * cos_theta_i + eta_t * cos_theta_t)
    // t_s = (2 * eta_i * cos_theta_i) / (eta_i * cos_theta_i + eta_t * cos_theta_t)
    const float2 r_s_numerator = openPBRComplexSubtract(eta_i_cos_theta_i, eta_t_cos_theta_t);
    const float2 denominator_s = openPBRComplexAdd(eta_i_cos_theta_i, eta_t_cos_theta_t);
    r.s = openPBRComplexDivide(r_s_numerator, denominator_s, OpenPBRMinDenomMag, float2(1.0f, 0.0f));
    t.s = openPBRComplexDivide(t_numerator, denominator_s, OpenPBRMinDenomMag, float2(0.0f, 0.0f));

    // For p-polarization:
    // r_p = (eta_t * cos_theta_i - eta_i * cos_theta_t) / (eta_t * cos_theta_i + eta_i * cos_theta_t)
    // t_p = (2 * eta_i * cos_theta_i) / (eta_t * cos_theta_i + eta_i * cos_theta_t)
    const float2 r_p_numerator = openPBRComplexSubtract(eta_t_cos_theta_i, eta_i_cos_theta_t);
    const float2 denominator_p = openPBRComplexAdd(eta_t_cos_theta_i, eta_i_cos_theta_t);
    r.p = openPBRComplexDivide(r_p_numerator, denominator_p, OpenPBRMinDenomMag, float2(1.0f, 0.0f));
    t.p = openPBRComplexDivide(t_numerator, denominator_p, OpenPBRMinDenomMag, float2(0.0f, 0.0f));
}

// Compute amplitude Fresnel coefficients for dielectrics (outputs PolarizedFloat)
// This is a fast path for the purely dielectric case (exterior-to-film interface),
// using real arithmetic instead of the complex arithmetic required by the unified function
// This is more efficient when we know both materials are dielectrics
// For the film-to-base interface, we use the unified function since it may encounter TIR, requiring complex coefficients to preserve phase information
// Note: Could be vectorized to output PolarizedColor, but the TIR branching would add complexity that likely outweighs vectorization benefits

public void openPBRComputeFresnelDielectricPolarizedAmplitude(const float cos_theta_i, const float eta_i, const float eta_t,
                                                       out OpenPBRPolarizedFloat r, // Amplitude reflection coefficients
                                                       out OpenPBRPolarizedFloat
                                                           t, // Amplitude transmission coefficients
                                                       out float cos_theta_t)
{
    const float sin_theta_i = sqrt(saturate(1.0f - cos_theta_i * cos_theta_i));

    // Apply Snell's Law
    const float sin_theta_t = eta_i / eta_t * sin_theta_i;

    // Check for total internal reflection
    if (sin_theta_t >= 1.0f) {
        r.s = 1.0f;
        r.p = 1.0f;
        t.s = 0.0f;
        t.p = 0.0f;
        cos_theta_t = 0.0f;
        return;
    }

    // Calculate the cosine of the angle of refraction, which will always be positive
    cos_theta_t = sqrt(saturate(1.0f - sin_theta_t * sin_theta_t));

    // Calculate intermediate products
    const float eta_i_cos_theta_i = eta_i * cos_theta_i;
    const float eta_t_cos_theta_t = eta_t * cos_theta_t;
    const float eta_t_cos_theta_i = eta_t * cos_theta_i;
    const float eta_i_cos_theta_t = eta_i * cos_theta_t;

    // Calculate denominators
    OpenPBRPolarizedFloat denom;
    denom.s = eta_i_cos_theta_i + eta_t_cos_theta_t;
    denom.p = eta_t_cos_theta_i + eta_i_cos_theta_t;

    // Prevent division by zero
    denom.s =
        (abs(denom.s) < OpenPBRMinDenomMag) ? (denom.s >= 0.0f ? OpenPBRMinDenomMag : -OpenPBRMinDenomMag) : denom.s;
    denom.p =
        (abs(denom.p) < OpenPBRMinDenomMag) ? (denom.p >= 0.0f ? OpenPBRMinDenomMag : -OpenPBRMinDenomMag) : denom.p;

    // Precompute reciprocal denominators to minimize number of divisions
    OpenPBRPolarizedFloat rcp_denom;
    rcp_denom.s = 1.0f / denom.s;
    rcp_denom.p = 1.0f / denom.p;

    // Amplitude reflection coefficients for s and p polarizations
    r.s = (eta_i_cos_theta_i - eta_t_cos_theta_t) * rcp_denom.s;
    r.p = (eta_t_cos_theta_i - eta_i_cos_theta_t) * rcp_denom.p;

    // Amplitude transmission coefficients for s and p polarizations
    t.s = (2.0f * eta_i_cos_theta_i) * rcp_denom.s;
    t.p = (2.0f * eta_i_cos_theta_i) * rcp_denom.p;
}

// Gulbrandsen's helper functions for converting two colors to a complex IOR

public float openPBRNMin(const float clamped_r) { return (1.0f - clamped_r) / (1.0f + clamped_r); }

public float openPBRNMax(const float clamped_r)
{
    const float sqrt_clamped_r = sqrt(clamped_r);
    return (1.0f + sqrt_clamped_r) / (1.0f - sqrt_clamped_r);
}

public float openPBRGetN(const float clamped_r, const float g)
{
    return lerp(openPBRNMax(clamped_r), openPBRNMin(clamped_r), g);
}

public float openPBRGetKSquared(float clamped_r, float n)
{
    const float numerator = openPBRSquare(n + 1.0f) * clamped_r - openPBRSquare(n - 1.0f);
    const float denominator = 1.0f - clamped_r;
    return numerator / denominator;
}

// Compute n and k for a metal using the Gulbrandsen method
// Uses the approach outlined in the paper "Artist Friendly Metallic Fresnel" by Gulbrandsen
// "r" and "g" are terms from the paper standing for "reflectivity" and "edge tint"

public float2 openPBRComputeGulbrandsenNAndK(const float r, const float g)
{
    // Clamp r as in the paper
    const float clamped_r = clamp(r, 0.0f, 0.99f);

    // Compute n and k using Gulbrandsen's method
    const float n = openPBRGetN(clamped_r, g);
    const float k_squared = openPBRGetKSquared(clamped_r, n);

    // Calculate k from k^2
    const float k = sqrt(max(k_squared, 0.0f)); // Ensure k^2 is non-negative

    // Format n and k as a complex number and return
    const float2 n_and_k = float2(n, k);
    return n_and_k;
}

// Perform the Airy summation for a single polarization and wavelength
// This is equivalent to Equation 3 in the Belcour and Barla paper

public float openPBRComputeAiryReflectance(
    const float r12,      // Amplitude reflection coefficient at film-to-exterior interface
    const float t12,      // Amplitude transmission coefficient at film-to-exterior interface
    const float r21,      // Amplitude reflection coefficient from film to exterior (reverse direction)
    const float t21,      // Amplitude transmission coefficient from film to exterior
    const float2 r23,     // Complex amplitude reflection coefficient at base-to-film interface
    const float delta_phi // Phase shift due to optical path difference
)
{
    // Exponential term e^(i * delta_phi)
    const float2 exp_i_delta_phi = openPBRComplexExpI(delta_phi);

    // Numerator: t12 * r23 * t21 * e^(i * delta_phi)
    const float2 numerator = openPBRComplexScalarMultiply(t12 * t21, openPBRComplexMultiply(r23, exp_i_delta_phi));

    // Denominator: 1 - r21 * r23 * e^(i * delta_phi)
    const float2 denominator = openPBRComplexSubtract(
        float2(1.0f, 0.0f), openPBRComplexScalarMultiply(r21, openPBRComplexMultiply(r23, exp_i_delta_phi)));

    // Total complex reflection coefficient: r_total = r12 + numerator / denominator
    // This line completes the evaluation of Equation 3 from the Belcour and Barla paper
    // If division by zero occurs, the complex_divide function returns zero and only r12 contributes
    const float2 r_total = openPBRComplexAdd(
        float2(r12, 0.0f), openPBRComplexDivide(numerator, denominator, OpenPBRMinDenomMag, float2(0.0f)));

    // Reflectance (power coefficient) is the magnitude squared of the total reflection coefficient
    return openPBRComplexMagnitudeSquared(r_total);
}

// Helper function to compute thin-film reflectance for a given base material

public float openPBRComputeCombinedReflectanceForThinFilmAndBase(
    const OpenPBRPolarizedFloat r12, // Amplitude reflection coefficients at film-to-exterior interface (s and p)
    const OpenPBRPolarizedFloat t12, // Amplitude transmission coefficients at film-to-exterior interface (s and p)
    const OpenPBRPolarizedFloat
        r21, // Amplitude reflection coefficients from film to exterior (reverse direction) (s and p)
    const OpenPBRPolarizedFloat t21,   // Amplitude transmission coefficients from film to exterior (s and p)
    const OpenPBRPolarizedComplex r23, // Complex amplitude reflection coefficients at base-to-film interface (s and p)
    const float delta_phi              // Phase shift due to optical path difference
)
{
    // S-polarization
    const float reflectance_s = openPBRComputeAiryReflectance(r12.s, t12.s, r21.s, t21.s, r23.s, delta_phi);

    // P-polarization
    const float reflectance_p = openPBRComputeAiryReflectance(r12.p, t12.p, r21.p, t21.p, r23.p, delta_phi);

    // Average over polarizations to get unpolarized reflectance
    const float reflectance = 0.5f * (reflectance_s + reflectance_p);

    return reflectance;
}

// Helper function to compute thin-film reflectance for a dielectric base

public float3 openPBRComputeDielectricReflectance(const OpenPBRPolarizedFloat r12, const OpenPBRPolarizedFloat t12,
                                           const OpenPBRPolarizedFloat r21, const OpenPBRPolarizedFloat t21,
                                           const float eta_film, const float cos_theta_t_film, const float3 delta_phi,
                                           const float3 eta_base, // Refractive index(es) of the dielectric base
                                           const bool enable_dispersion)
{
    // Retain the complex phase under TIR, but consume each channel before
    // advancing. Avoid dynamically indexing an array of complex structs and
    // keeping three channels live across two separate loops. The first channel
    // is reused exactly when dispersion is disabled; no approximation is made.
    OpenPBRPolarizedComplex first;
    OpenPBRPolarizedComplex unusedTransmission;
    float2 unusedCosine;
    openPBRComputeFresnelUnifiedPolarizedAmplitude(cos_theta_t_film, eta_film,
        float2(eta_base.x, 0.0f), first, unusedTransmission, unusedCosine);
    float3 reflectance_dielectric;
    [unroll]
    for (int channel = 0; channel < OpenPBRNumRGBChannels; ++channel) {
        OpenPBRPolarizedComplex coefficient = first;
        if (enable_dispersion && channel != 0) {
            openPBRComputeFresnelUnifiedPolarizedAmplitude(cos_theta_t_film, eta_film,
                float2(eta_base[channel], 0.0f), coefficient, unusedTransmission, unusedCosine);
        }
        reflectance_dielectric[channel] = openPBRComputeCombinedReflectanceForThinFilmAndBase(
            r12, t12, r21, t21, coefficient, delta_phi[channel]);
    }

    return reflectance_dielectric;
}

// Helper function to compute thin-film reflectance for a metal base

public float3 openPBRComputeMetalReflectance(const OpenPBRPolarizedFloat r12, const OpenPBRPolarizedFloat t12,
                                      const OpenPBRPolarizedFloat r21, const OpenPBRPolarizedFloat t21,
                                      const float eta_film, const float cos_theta_t_film, const float3 delta_phi,
                                      const float3 metal_F0, const float3 metal_F82_tint)
{
    // Compute reflectance for each color channel
    float3 reflectance_metal;
    [unroll]
    for (int color_channel = 0; color_channel < OpenPBRNumRGBChannels; ++color_channel) {
        // Compute complex IOR n and k from metal_F0 and metal_F82_tint
        const float2 n_and_k = openPBRComputeGulbrandsenNAndK(metal_F0[color_channel], metal_F82_tint[color_channel]);

        // Compute Fresnel coefficients at the film-metal interface using unified function
        OpenPBRPolarizedComplex r23_metal;
        OpenPBRPolarizedComplex t23_unused;
        float2 cos_theta_t_base_unused;
        openPBRComputeFresnelUnifiedPolarizedAmplitude(cos_theta_t_film, eta_film, n_and_k, r23_metal, t23_unused,
                                                       cos_theta_t_base_unused);

        // Compute and store reflectance for this color channel
        reflectance_metal[color_channel] = openPBRComputeCombinedReflectanceForThinFilmAndBase(
            r12, t12, r21, t21, r23_metal, delta_phi[color_channel]);
    }

    return reflectance_metal;
}

// Struct to hold the results of the thin film and base reflectance computations
// The caller can weight and combine these results as needed
// The caller can also calculate the transmittance by subtracting the reflectance from 1
public struct OpenPBRThinFilmResults
{
    public float3 reflectance_dielectric; // Reflectance of the dielectric base
    public float3 reflectance_metal; // Reflectance of the metal base
};

// Main function to compute reflectance (RGB power coefficient) of combination of thin film and base surface
// In the rare case that dispersion of the dielectric IOR is needed, this function can be called separately for each color channel

public OpenPBRThinFilmResults openPBRThinFilmAndBaseReflectance(
    const float cos_theta_i,            // Cosine of the angle of incidence from the exterior medium (assumed >= 0)
    const float eta_exterior,           // Refractive index of the exterior medium (e.g., air)
    const float eta_film,               // Refractive index of the thin film
    const float3 eta_base,              // Refractive index(es) of the first dielectric base material
    const bool enable_dispersion,       // Enable dispersion of the dielectric IOR
    const float3 metal_F0,              // Metal reflectivity at normal incidence (power coefficient, color)
    const float3 metal_F82_tint,        // Tint of metal reflectivity at 82-degree angle (power coefficient, color)
    const bool enable_dielectric,       // Enable the first dielectric base
    const bool enable_metal,            // Enable the metal base
    const float thin_film_thickness_nm, // Thickness of the thin film (in nanometers)
    const float3 lambda_rgb_nm          // Wavelengths for R, G, B (in nanometers)
)
{
    // 1. Common Calculations

    // Ensure cos_theta_i is non-negative
    // Otherwise, this function should not have been called

    // Ensure thin_film_thickness_nm is positive
    // Otherwise, there is no thin film to compute
    // The caller should have caught this by checking thin_film_presence_multiplier

    // Default the results to zero in case some base components are disabled
    OpenPBRThinFilmResults results;
    results.reflectance_dielectric = float3(0.0f);
    results.reflectance_metal = float3(0.0f);

    // Return if all base components are disabled
    // If the caller were to check this, this early exit could be replaced with an assertion
    if (!(enable_dielectric || enable_metal))
        return results;

    // Compute Fresnel coefficients at film-to-exterior interface
    float cos_theta_t_film;
    OpenPBRPolarizedFloat r12; // Amplitude reflection coefficients at film-to-exterior interface
    OpenPBRPolarizedFloat t12; // Amplitude transmission coefficients at film-to-exterior interface
    openPBRComputeFresnelDielectricPolarizedAmplitude(cos_theta_i, eta_exterior, eta_film, r12, t12, cos_theta_t_film);

    // Check for total internal reflection on the outside of the film
    if (cos_theta_t_film <= 0.0f) {
        // Total internal reflection occurs on the outside of the film, so set all reflectances to one and return
        if (enable_dielectric)
            results.reflectance_dielectric = float3(1.0f);
        if (enable_metal)
            results.reflectance_metal = float3(1.0f);
        return results;
    }

    // Reflection coefficients for reverse direction
    // r21 = -r12 to account for phase shift by 180 degrees
    OpenPBRPolarizedFloat r21;
    r21.s = -r12.s;
    r21.p = -r12.p;

    // Transmission coefficients for reverse direction
    // t21 is derived from r12 using the appropriate reciprocity relation
    // Note that amplitude transmission coefficients can be greater than one due to refraction
    OpenPBRPolarizedFloat t21;
    const float safe_cos_theta_i =
        max(cos_theta_i, OpenPBRMinDenomMag); // Prevent numerical issues at grazing incidence where cos_theta_i -> 0
    const float t21_scale = (eta_film * cos_theta_t_film) / (eta_exterior * safe_cos_theta_i);
    t21.s = t12.s * t21_scale;
    t21.p = t12.p * t21_scale;

    // Compute optical path difference and phase shift per color channel
    const float opd = 2.0f * eta_film * thin_film_thickness_nm * cos_theta_t_film;
    const float3 delta_phi = 2.0f * OpenPBRPi * opd / lambda_rgb_nm;

    // 2. Dielectric Base Calculations

    // Compute reflectance for dielectric base component
    if (enable_dielectric)
        results.reflectance_dielectric =
            openPBRComputeDielectricReflectance(r12, t12, r21, t21, eta_film, cos_theta_t_film, delta_phi,
                                                eta_base, // Refractive index of the first dielectric base
                                                enable_dispersion);

    // 3. Metal Base Calculations

    // Compute reflectance for metal base component
    if (enable_metal)
        results.reflectance_metal = openPBRComputeMetalReflectance(r12, t12, r21, t21, eta_film, cos_theta_t_film,
                                                                   delta_phi, metal_F0, metal_F82_tint);

    // 4. Return Final Reflectance Results

    return results;
}

// Handle very thin films by scaling the thin-film presence by this thickness-based multiplier

public float openPBRThinFilmPresenceMultiplier(const float thin_film_thickness_nm)
{
    // Thickness below which the thin film is fully invisible
    const float FullInvisibilityThreshold_nm =
        OpenPBRMinDenomMag; // Slightly positive value to avoid potential numerical issues

    // Thickness above which the thin film is fully visible
    const float FullVisibilityThreshold_nm = 30.0f; // 30 nanometers

    return smoothstep(FullInvisibilityThreshold_nm, FullVisibilityThreshold_nm, thin_film_thickness_nm);
}

// Desaturates the thin film reflectance based on thin film thickness and the cosine of the angle of incidence (assumed >= 0).
// This technique is used by the rasterizer to match the path tracer thin film reflectance while using constant RGB wavelengths.

public OpenPBRThinFilmResults openPBRDesaturateThinFilmReflectance(const float3 dielectric_reflectance,
                                                            const float3 metal_reflectance,
                                                            const float thin_film_thickness_nm, const float idoth)
{
    OpenPBRThinFilmResults desaturated_reflectance;
    const float thin_film_desaturation_scale = min(sqrt(thin_film_thickness_nm * 0.001f * idoth), 1.0f);

    desaturated_reflectance.reflectance_dielectric =
        lerp(dielectric_reflectance, float3(openPBRAverage(dielectric_reflectance)), thin_film_desaturation_scale);
    desaturated_reflectance.reflectance_metal =
        lerp(metal_reflectance, float3(openPBRAverage(metal_reflectance)), thin_film_desaturation_scale);

    return desaturated_reflectance;
}
} // namespace Metallic.OpenPBR

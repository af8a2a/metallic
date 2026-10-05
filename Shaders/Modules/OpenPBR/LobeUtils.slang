#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_lobe_utils.h. Modified by Metallic, 2026.
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

// This function ensures that a random number is in [0, 1),
// which may occasionally not be the case due to rounding error or fast-math transformations in remapping calculations.

public void openPBRClampRemappedRandomNumber(inout float rand)
{

    // TODO: Address numerical vulnerabilities, uncomment this assertion, and disable the following clamp.
    // OPENPBR_ASSERT(rand < 1.0f, "It is assumed that the random number is already < 1");
    rand = min(rand, OpenPBRLargestFloatBelowOne);
}

// Applies anisotropy to an isotropic GGX microfacet distribution "alpha" (square roughness)
// using the formulas from the OpenPBR spec.
//
// When applicable, also applies a clamp to the anisotropic alpha values to ensure they are above a minimum value.

public float2 openPBRComputeAnisotropicAlpha(const float isotropic_alpha, const float openPBRAnisotropy,
                                      const bool delta_specular_supported, const float min_microfacet_alpha)
{
    float2 anisotropic_alpha;

    if (openPBRAnisotropy > 0.0f) {
        // Apply the formulas from the OpenPBR spec.
        anisotropic_alpha.x = isotropic_alpha * sqrt(2.0f / (1.0f + openPBRSquare(1.0f - openPBRAnisotropy)));
        anisotropic_alpha.y = (1.0f - openPBRAnisotropy) * anisotropic_alpha.x;
    } else // isotropic
    {
        anisotropic_alpha = float2(isotropic_alpha);
    }

    // Unlike in the isotropic case where we could switch to delta lobes at low roughnesses,
    // we must always use microfacet lobes when there is anisotropy; therefore we need to
    // clamp the alphas to avoid numerical issues.
    //
    // If delta specular is not supported, we also need clamp isotropic alpha values to
    // ensure they are above a minimum value. (If delta specular is supported, we can
    // use the delta specular lobe to handle the case where the alpha is too small.)
    if (openPBRAnisotropy > 0.0f || !delta_specular_supported) {
        anisotropic_alpha = max(anisotropic_alpha, float2(min_microfacet_alpha));
    }

    return anisotropic_alpha;
}

// Evaluate anisotropic GGX.
// Input vector is in local (z-up) space.

public float openPBREvalAnisoGgx(const float3 n, const float2 alpha)
{
    return 1.0f / (OpenPBRPi * alpha.x * alpha.y *
                   openPBRSquare(openPBRSquare(n.x / alpha.x) + openPBRSquare(n.y / alpha.y) + openPBRSquare(n.z)));
}

// Sample VNDF of anisotropic GGX with Smith masking function.
// Input and output vectors are in local (z-up) space.
// Based on the technique described in the paper "Sampling the GGX Distribution of Visible Normals" (http://jcgt.org/published/0007/04/01/).

public float3 openPBRSampleAnisoGgxSmithVndf(const float2 alpha, const float3 incoming, const float2 rand)
{
    // Transform ellipsoid configuration into sphere configuration.
    const float3 ellipsoid_to_hemisphere = float3(alpha.x, alpha.y, 1.0f);
    const float3 incoming_hemisphere = openPBRFastNormalize(ellipsoid_to_hemisphere * incoming);
    const float incoming_hemisphere_xy_length_squared =
        openPBRSquare(incoming_hemisphere.x) + openPBRSquare(incoming_hemisphere.y);
    // Construct an orthonormal basis with transformed incoming direction as normal.
    float3 tangent;
    if (incoming_hemisphere_xy_length_squared > 0.0f) {
        const float incoming_hemisphere_xy_length_inverse = openPBRFastRcpSqrt(incoming_hemisphere_xy_length_squared);
        tangent = float3(-incoming_hemisphere.y, incoming_hemisphere.x, 0.0f) * incoming_hemisphere_xy_length_inverse;
    } else {
        tangent = float3(1.0f, 0.0f, 0.0f);
    }
    const float3 bitangent = cross(incoming_hemisphere, tangent);
    // Uniformly sample a disk.
    const float radius = openPBRFastSqrt(rand.x);
    const float angle = OpenPBRTwoPi * rand.y;
    const float sin_angle = sin(angle);
    const float cos_angle = cos(angle);
    const float bitangent_component_unscaled = radius * sin_angle;
    const float tangent_component = radius * cos_angle;
    // Remap vertical coordinate to warp disk into projected hemisphere.
    const float scale = 0.5f * (1.0f + incoming_hemisphere.z);
    const float bitangent_component = (1.0f - scale) * openPBRFastSqrt(1.0f - openPBRSquare(tangent_component)) +
                                      scale * bitangent_component_unscaled;
    // Project point on projected hemisphere back onto hemisphere.
    float normal_component_squared = 1.0f - (openPBRSquare(tangent_component) + openPBRSquare(bitangent_component));
    // Prevent negative number due to rounding error.
    if (normal_component_squared < 0.0f) {
        normal_component_squared = 0.0f;
    }
    float3 microfacet_normal_hemisphere = tangent * tangent_component + bitangent * bitangent_component +
                                          incoming_hemisphere * openPBRFastSqrt(normal_component_squared);
    // Prevent microfacet normal below ground due to rounding error.
    if (microfacet_normal_hemisphere.z < 0.0f) {
        microfacet_normal_hemisphere.z = 0.0f;
    }
    // The inverse transformation for the sampled normal is the same
    // as the forward transformation for the incoming direction.
    return openPBRFastNormalize(ellipsoid_to_hemisphere * microfacet_normal_hemisphere);
}

// Evaluate anisotropic Smith shadowing or masking function.
// Input vector is in local (z-up) space.
// Negative v.z is intentionally supported to accommodate transmission paths.

public float openPBREvalAnisoSmithG1(const float3 v, const float2 alpha)
{
    // Guard against division by zero; underflow in the square automatically
    // handles cases where v is near the tangent plane.
    const float vz_squared = openPBRSquare(v.z);
    if (vz_squared == 0.0f)
        return 0.0f;

    // This uses an algebraically equivalent form of the Smith G1 function that
    // avoids catastrophic cancellation in the intermediate Lambda computation,
    // ensuring numerical stability when roughness or slopes are near zero.
    return 2.0f /
           (1.0f + openPBRFastSqrt(1.0f + (openPBRSquare(alpha.x * v.x) + openPBRSquare(alpha.y * v.y)) / vz_squared));
}

// Evaluate anisotropic Smith non-correlated shadowing and masking function.
// Input vectors are in local (z-up) space.

public float openPBREvalAnisoSmithG2(const float3 v1, const float3 v2, const float2 alpha)
{
    return openPBREvalAnisoSmithG1(v1, alpha) * openPBREvalAnisoSmithG1(v2, alpha);
}

// Weights a color by the path throughput and then returns the max component of the product.
// Used for estimating the average contribution of a BSDF lobe, which is used for lobe selection.

public float openPBRMaxComponentOfThroughputWeightedColor(const float3 path_throughput, const float3 x)
{

    const float3 throughput_weighted_color = path_throughput * x;
    return openPBRMax3(throughput_weighted_color);
}

// Converts IOR to F0.

public float openPBRF0FromIor(const float eta_t_over_eta_i)
{
    return openPBRSquare((eta_t_over_eta_i - 1.0f) / (eta_t_over_eta_i + 1.0f));
}

// Converts F0 to IOR.

public float openPBRIorFromF0(const float f0)
{

    const float sqrt_f0 = openPBRFastSqrt(f0);
    const float eta_t_over_eta_i = (1.0f + sqrt_f0) / (1.0f - sqrt_f0); // infinity if f0 is 1
    return eta_t_over_eta_i;
}

// Adjusts the given IOR using the OpenPBR "specular weight" param.
// When the "specular weight" is 1.0, this function is a no-op.

public float openPBRApplySpecularWeightToIor(const float eta_t_over_eta_i, const float specular_weight)
{
    const float unscaled_f0 = openPBRF0FromIor(eta_t_over_eta_i);
    const float scaled_f0 = unscaled_f0 * specular_weight;
    const float MaxF0 = 0.9999f;
    const float clamped_scaled_f0 = min(scaled_f0, MaxF0);
    const float external_ior = openPBRIorFromF0(clamped_scaled_f0);
    const bool internal_reflection = eta_t_over_eta_i < 1.0f;
    return internal_reflection ? 1.0f / external_ior : external_ior;
}

// Calculates Schlick Fresnel reflectivity given color F0.

public float3 openPBRSchlick(const float cos_theta, const float3 f0)
{
    return f0 + (float3(1.0f) - f0) * openPBRFifthPower(1.0f - abs(cos_theta));
}

// Calculates Schlick Fresnel reflectivity given scalar F0.

public float openPBRSchlick(const float cos_theta, const float f0)
{
    return f0 + (1.0f - f0) * openPBRFifthPower(1.0f - min(abs(cos_theta), 1.0f));
}

// Calculates real unpolarized Fresnel reflectivity.

public float openPBRFresnel(const float eta_t_over_eta_i, const float cos_theta_i)
{

    if (eta_t_over_eta_i == 1.0f)
        return 0.0f; // indexed-matched media
    const float sin_theta_i_squared = 1.0f - openPBRSquare(cos_theta_i);
    const float sin_theta_t_squared = sin_theta_i_squared / openPBRSquare(eta_t_over_eta_i);
    if (sin_theta_t_squared >= 1.0f)
        return 1.0f; // total internal reflection (or grazing reflection)
    const float cos_theta_t = openPBRFastSqrt(1.0f - sin_theta_t_squared);
    const float eta_t_over_eta_i_cos_theta_i = eta_t_over_eta_i * cos_theta_i;
    const float eta_t_over_eta_i_cos_theta_t = eta_t_over_eta_i * cos_theta_t;
    const float Rs =
        openPBRSquare((cos_theta_i - eta_t_over_eta_i_cos_theta_t) / (cos_theta_i + eta_t_over_eta_i_cos_theta_t));
    const float Rp =
        openPBRSquare((cos_theta_t - eta_t_over_eta_i_cos_theta_i) / (cos_theta_t + eta_t_over_eta_i_cos_theta_i));
    return 0.5f * (Rs + Rp); // unpolarized
}

// This is an RGB version of the Fresnel function.
// The OpenPBR specialization constant for ENABLE_DISPERSION should be passed into enable_dispersion.

public float3 openPBRFresnelRgb(const float3 eta_t_over_eta_i, const float cos_theta_i, const bool enable_dispersion)
{
    if (enable_dispersion)
        return float3(openPBRFresnel(eta_t_over_eta_i.r, cos_theta_i), openPBRFresnel(eta_t_over_eta_i.g, cos_theta_i),
                      openPBRFresnel(eta_t_over_eta_i.b, cos_theta_i));
    else
        return float3(openPBRFresnel(eta_t_over_eta_i.r, cos_theta_i));
}

// Approximates the cosine-weighted average of the Fresnel reflection coefficient over the hemisphere.
// See https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf, slide 18.

public float openPBRAverageFresnel(const float eta_t_over_eta_i)
{
    if (eta_t_over_eta_i > 1.0f)
        return (eta_t_over_eta_i - 1.0f) / (4.08567f + 1.00071f * eta_t_over_eta_i);
    else
        return 0.997118f + 0.1014f * eta_t_over_eta_i - 0.965241f * (eta_t_over_eta_i * eta_t_over_eta_i) -
               0.130607f * (eta_t_over_eta_i * eta_t_over_eta_i) * eta_t_over_eta_i;
}

// Calculates the total reflectance from the front and back of a thin-walled surface,
// including all internal bounces. You can think of this as the reflectance you would
// see from a single surface representing a glass windowpane, for example.

public float openPBRThinWallFresnel(const float eta_t_over_eta_i, const float cos_theta_i)
{
    const float front_reflectance = openPBRFresnel(eta_t_over_eta_i, cos_theta_i);
    const float total_reflectance = (2.0f * front_reflectance) / (1.0f + front_reflectance);
    return total_reflectance;
    // Note that total transmittance = 1 - (2 * R) / (1 + R) = (1 - R) / (1 + R).
}

// This is a modified version of Naty Hoffman's F82 model for metallic Fresnel, which was in turn
// based on the Lazanyi-Schlick approximation. We can call this new formulation the F82-tint model
// because it allows the edge tint to be applied multiplicatively rather than being used directly
// as the reflectivity. This formulation has the new property that it reduces to regular Schlick
// when the edge tint is set to a universal default value of white. Like the original F82 model,
// it can approximate the real Fresnel dip by using a custom reflectivity at approximately
// 82 degrees, and it can also be used for creative control as the edge tint is clearly visible
// regardless of the selected colors.

public float3 openPBRComputeMetalSchlickBFactor(const float3 f0, const float3 f82_tint)
{
    const float CosThetaMax = 1.0f / 7.0f;
    const float OneMinusCosThetaMax = 1.0f - CosThetaMax;
    const float OneMinusCosThetaMaxToTheFifth = openPBRFifthPower(OneMinusCosThetaMax);
    const float OneMinusCosThetaMaxToTheSixth = openPBRSixthPower(OneMinusCosThetaMax);
    const float3 r = f0;       // switch to terminology from the "Fresnel Equations Considered Harmful" slides
    const float3 t = f82_tint; // custom terminology: "t" (for "tint") = h / (r + (1 - r) * (1 - CosThetaMax)^5)
    const float3 white_minus_r = float3(1.0f) - r;
    const float3 white_minus_t = float3(1.0f) - t;
    const float3 b_numerator = (r + white_minus_r * OneMinusCosThetaMaxToTheFifth) * white_minus_t;
    const float BDenominator = CosThetaMax * OneMinusCosThetaMaxToTheSixth;
    const float BDenominatorReciprocal = 1.0f / BDenominator;
    return b_numerator *
           BDenominatorReciprocal; // analogous to "a" in the "Fresnel Equations Considered Harmful" slides
}

public float3 openPBRMetalSchlickWithF82Tint(const float3 f0, const float3 f82_tint, const float cos_theta)
{

    const float3 r = f0; // switch to terminology from the "Fresnel Equations Considered Harmful" slides
    const float3 white_minus_r = float3(1.0f) - r;
    const float3 b = openPBRComputeMetalSchlickBFactor(f0, f82_tint);
    const float one_minus_cos_theta = 1.0f - cos_theta;
    const float3 offset_from_r =
        (white_minus_r - b * cos_theta * one_minus_cos_theta) * openPBRFifthPower(one_minus_cos_theta);
    const float3 f_theta = r + offset_from_r;
    return saturate(f_theta);
}

// Returns the cosine-weighted average of the F82 metallic Fresnel function over the hemisphere.

public float3 openPBRMetalAverageFresnelWithF82Tint(const float3 f0, const float3 f82_tint)
{
    const float3 r = f0; // switch to terminology from the "Fresnel Equations Considered Harmful" slides
    const float3 white_minus_r = float3(1.0f) - r;
    const float3 b = openPBRComputeMetalSchlickBFactor(f0, f82_tint);
    const float3 average_albedo = r + white_minus_r * (1.0f / 21.0f) - b * (1.0f / 126.0f);
    return saturate(average_albedo);
}

// Calculates refracted direction.

public bool openPBRRefract(const float3 wi, const float3 n, const float cos_theta_i, const float eta_t_over_eta_i,
                    out float3 wt)
{
    const float eta_i_over_eta_t = 1.0f / eta_t_over_eta_i;

    // Compute cos_theta_t using Snell's law.
    const float sin2_theta_i = max(0.0f, float(1.0f - openPBRSquare(cos_theta_i)));
    const float sin2_theta_t = openPBRSquare(eta_i_over_eta_t) * sin2_theta_i;

    // Handle total internal reflection for transmission.
    if (sin2_theta_t >= 1.0f) {
        wt = float3(0.0f);
        return false;
    }

    // The use of openpbr_fast_sqrt() here will lead to divide by zero in PDFs
    const float cos_theta_t = sqrt(1.0f - sin2_theta_t);

    // Technically this vector is normalized by construction, but rounding error can be significant.
    wt = openPBRFastNormalize(-wi * eta_i_over_eta_t + n * (eta_i_over_eta_t * cos_theta_i - cos_theta_t));
    return true;
}

// In lobes relying on tabulated data, this function prevents NaNs when dividing by an
// average energy complement that has been decoded from low-precision fixed-point form.

public float openPBRClampAverageEnergyComplementAboveZero(const float average_energy_complement)
{
    const float min_value = 1e-12f;
    return max(average_energy_complement, min_value);
}

public float openPBRCombineCoatAndBasePdfs(const float pdf_coat, const float pdf_base, const float coat_reflection_prob)
{
    return lerp(pdf_base, pdf_coat, coat_reflection_prob);
}

// Clears the output parameters of a lobe sampling function when sampling fails.

public void openPBRClearLobeSamplingOutput(out float3 light_direction, out OpenPBRDiffuseSpecular weight, out float pdf,
                                    out uint sampled_type)
{
    light_direction = float3(0.0f);
    weight = openPBRMakeZeroDiffuseSpecular();
    pdf = 0.0f;
    sampled_type = OpenPBRBSDFLobeTypeNone;
}
} // namespace Metallic.OpenPBR

#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_reflection_transmission_coefficient.h. Modified by Metallic, 2026.
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

// The reflection/transmission class below is designed to handle all of the specular reflection and transmission
// of the base surface of OpenPBR. It has a transparent dielectric component, an opaque dielectric component,
// a metallic component, and a thin-film iridescence component.

////////////////////////////////////////////////////
// ComprehensiveReflectionTransmissionCoefficient //
////////////////////////////////////////////////////

public struct OpenPBRComprehensiveReflectionTransmissionCoefficient
{
    public float3 eta_t_over_eta_i_for_transparent_part; // ignoring the effect of the thin film
    public float3 eta_t_over_eta_i_for_opaque_part; // ignoring the effect of the thin film
    public float3 scale_for_reflection_for_transparent_part;
    public float3 scale_for_reflection_for_opaque_part;
    public float3 transmission;
    public float3 f0_for_metal;
    public float3 f82_tint_for_metal;
    public float metal_amount;
    public float thin_film_weight;
    public float thin_film_thickness_nm;
    public float thin_film_exterior_ior; // absolute IOR - dispersion not supported - TODO: support dispersion
                                  // (note that the "exterior" here can be the interior of an object)
    public float thin_film_ior; // absolute IOR - dispersion not supported
    public float3 thin_film_interior_ior; // absolute IOR - dispersion supported
    public float3 rgb_wavelengths_nm;
    public float3 thin_wall_constant_reflection_albedo; // albedo of the thin-wall specular reflection component
                                                 // "constant" because it does not dependent on idoth
};

// This function returns the degree to which the thin film exists.
// The returned value is used to blend between standard reflection and transmission and thin-film reflection and transmission.

public float openPBRComputeThinFilmPresence(const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff,
                                     const float idoth)
{
    // Only apply thin film when striking the front of a microfacet (surface normal was faceforwarded but microfacet normal was not).
    if (idoth >= 0.0f)
        return refl_trans_coeff.thin_film_weight *
               openPBRThinFilmPresenceMultiplier(refl_trans_coeff.thin_film_thickness_nm);
    else
        return 0.0f;
}

public float3 openPBRReflectionCoefficient<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff, const float idoth)
{
    float3 standard_refl_coeff = float3(0.0f);
    float3 thin_film_refl_coeff = float3(0.0f);

    const float thin_film_presence = openPBRComputeThinFilmPresence(refl_trans_coeff, idoth);

    // Calculate the standard reflection coefficient.
    if (thin_film_presence < 1.0f) {
        if (TContext.enableTranslucency())
            standard_refl_coeff += refl_trans_coeff.scale_for_reflection_for_transparent_part *
                                   openPBRFresnelRgb(refl_trans_coeff.eta_t_over_eta_i_for_transparent_part, idoth,
                                                     TContext.enableDispersion());

        standard_refl_coeff +=
            refl_trans_coeff.scale_for_reflection_for_opaque_part *
            openPBRFresnelRgb(refl_trans_coeff.eta_t_over_eta_i_for_opaque_part, idoth, TContext.enableDispersion());

        if (TContext.enableMetallic())
            standard_refl_coeff += refl_trans_coeff.metal_amount *
                                   openPBRMetalSchlickWithF82Tint(refl_trans_coeff.f0_for_metal,
                                                                  refl_trans_coeff.f82_tint_for_metal, idoth);
    }

    // Calculate the thin-film reflection coefficient.
    if (thin_film_presence > 0.0f) {
        OpenPBRThinFilmResults thin_film_results = openPBRThinFilmAndBaseReflectance(
            idoth, refl_trans_coeff.thin_film_exterior_ior, refl_trans_coeff.thin_film_ior,
            refl_trans_coeff.thin_film_interior_ior, TContext.enableDispersion(), refl_trans_coeff.f0_for_metal,
            refl_trans_coeff.f82_tint_for_metal,
            (TContext.enableTranslucency() &&
             any(((refl_trans_coeff.scale_for_reflection_for_transparent_part) > (float3(0.0f))))) ||
                any(((refl_trans_coeff.scale_for_reflection_for_opaque_part) > (float3(0.0f)))), // dielectric part
            TContext.enableMetallic() && refl_trans_coeff.metal_amount > 0.0f,                   // metal part
            refl_trans_coeff.thin_film_thickness_nm, refl_trans_coeff.rgb_wavelengths_nm);

        if (TContext.enableTranslucency())
            thin_film_refl_coeff +=
                refl_trans_coeff.scale_for_reflection_for_transparent_part * thin_film_results.reflectance_dielectric;

        thin_film_refl_coeff +=
            refl_trans_coeff.scale_for_reflection_for_opaque_part * thin_film_results.reflectance_dielectric;

        if (TContext.enableMetallic())
            thin_film_refl_coeff += refl_trans_coeff.metal_amount * thin_film_results.reflectance_metal;
    }

    const float3 refl_coeff = lerp(standard_refl_coeff, thin_film_refl_coeff, thin_film_presence);

    // Currently, when thin wall is enabled, the thin wall reflection is calculated elsewhere and added here;
    // the tranmission is zeroed out in the thin-wall case using scale_for_reflection_for_transparent_part
    // because it is accounted for by an entirely different lobe.
    // TODO: Combine thin film and thin wall in a better way.
    const float3 thin_wall_aware_refl_coeff = refl_coeff + refl_trans_coeff.thin_wall_constant_reflection_albedo;

    return thin_wall_aware_refl_coeff;
}

public float3 openPBRTransmissionCoefficient<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff, const float idoth)
{
    float3 standard_trans_coeff = float3(0.0f);
    float3 thin_film_trans_coeff = float3(0.0f);

    const float thin_film_presence = openPBRComputeThinFilmPresence(refl_trans_coeff, idoth);

    // Calculate the standard transmission coefficient.
    if (thin_film_presence < 1.0f) {
        if (TContext.enableTranslucency())
            standard_trans_coeff +=
                refl_trans_coeff.transmission *
                (float3(1.0f) - openPBRFresnelRgb(refl_trans_coeff.eta_t_over_eta_i_for_transparent_part, idoth,
                                                  TContext.enableDispersion()));
    }

    // Calculate the thin-film transmission coefficient.
    if (thin_film_presence > 0.0f) {
        OpenPBRThinFilmResults thin_film_results = openPBRThinFilmAndBaseReflectance(
            idoth, refl_trans_coeff.thin_film_exterior_ior, refl_trans_coeff.thin_film_ior,
            refl_trans_coeff.thin_film_interior_ior, TContext.enableDispersion(), refl_trans_coeff.f0_for_metal,
            refl_trans_coeff.f82_tint_for_metal,
            TContext.enableTranslucency() &&
                any(((refl_trans_coeff.scale_for_reflection_for_transparent_part) > (float3(0.0f)))), // dielectric part
            false,                                                                                    // metal part
            refl_trans_coeff.thin_film_thickness_nm, refl_trans_coeff.rgb_wavelengths_nm);

        if (TContext.enableTranslucency())
            thin_film_trans_coeff +=
                refl_trans_coeff.transmission * (float3(1.0f) - thin_film_results.reflectance_dielectric);
    }

    const float3 trans_coeff = lerp(standard_trans_coeff, thin_film_trans_coeff, thin_film_presence);

    return trans_coeff;
}

// The default implementations of the reflection and transmission probabilities use both coefficients.
// Specialized implementations could calculate both based on the same shared factors (such as fresnel and 1-fresnel).

public OpenPBRAllCoefficients openPBRAllCoefficients<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff, const float idoth)
{
    float3 standard_refl_coeff = float3(0.0f);
    float3 standard_trans_coeff = float3(0.0f);
    float3 thin_film_refl_coeff = float3(0.0f);
    float3 thin_film_trans_coeff = float3(0.0f);

    const float thin_film_presence = openPBRComputeThinFilmPresence(refl_trans_coeff, idoth);

    // Calculate the standard reflection coefficient.
    if (thin_film_presence < 1.0f) {
        if (TContext.enableTranslucency()) {
            const float3 fresnel_for_transparent_part = openPBRFresnelRgb(
                refl_trans_coeff.eta_t_over_eta_i_for_transparent_part, idoth, TContext.enableDispersion());
            standard_refl_coeff +=
                refl_trans_coeff.scale_for_reflection_for_transparent_part * fresnel_for_transparent_part;
            standard_trans_coeff += refl_trans_coeff.transmission * (float3(1.0f) - fresnel_for_transparent_part);
        }

        standard_refl_coeff +=
            refl_trans_coeff.scale_for_reflection_for_opaque_part *
            openPBRFresnelRgb(refl_trans_coeff.eta_t_over_eta_i_for_opaque_part, idoth, TContext.enableDispersion());

        if (TContext.enableMetallic())
            standard_refl_coeff += refl_trans_coeff.metal_amount *
                                   openPBRMetalSchlickWithF82Tint(refl_trans_coeff.f0_for_metal,
                                                                  refl_trans_coeff.f82_tint_for_metal, idoth);
    }

    // Calculate the thin-film reflection coefficient.
    if (thin_film_presence > 0.0f) {
        OpenPBRThinFilmResults thin_film_results = openPBRThinFilmAndBaseReflectance(
            idoth, refl_trans_coeff.thin_film_exterior_ior, refl_trans_coeff.thin_film_ior,
            refl_trans_coeff.thin_film_interior_ior, TContext.enableDispersion(), refl_trans_coeff.f0_for_metal,
            refl_trans_coeff.f82_tint_for_metal,
            (TContext.enableTranslucency() &&
             any(((refl_trans_coeff.scale_for_reflection_for_transparent_part) > (float3(0.0f))))) ||
                any(((refl_trans_coeff.scale_for_reflection_for_opaque_part) > (float3(0.0f)))), // dielectric part
            TContext.enableMetallic() && refl_trans_coeff.metal_amount > 0.0f,                   // metal part
            refl_trans_coeff.thin_film_thickness_nm, refl_trans_coeff.rgb_wavelengths_nm);

        if (TContext.enableTranslucency()) {
            thin_film_refl_coeff +=
                refl_trans_coeff.scale_for_reflection_for_transparent_part * thin_film_results.reflectance_dielectric;
            thin_film_trans_coeff +=
                refl_trans_coeff.transmission * (float3(1.0f) - thin_film_results.reflectance_dielectric);
        }

        thin_film_refl_coeff +=
            refl_trans_coeff.scale_for_reflection_for_opaque_part * thin_film_results.reflectance_dielectric;

        if (TContext.enableMetallic())
            thin_film_refl_coeff += refl_trans_coeff.metal_amount * thin_film_results.reflectance_metal;
    }

    const float3 refl_coeff = lerp(standard_refl_coeff, thin_film_refl_coeff, thin_film_presence);
    const float3 trans_coeff = lerp(standard_trans_coeff, thin_film_trans_coeff, thin_film_presence);

    // Currently, when thin wall is enabled, the thin wall reflection is calculated elsewhere and added here;
    // the tranmission is zeroed out in the thin-wall case using scale_for_reflection_for_transparent_part
    // because it is accounted for by an entirely different lobe.
    // TODO: Combine thin film and thin wall in a better way.
    const float3 thin_wall_aware_refl_coeff = refl_coeff + refl_trans_coeff.thin_wall_constant_reflection_albedo;

    return OpenPBRAllCoefficients(thin_wall_aware_refl_coeff, trans_coeff);
}

public float openPBRReflectionProbability<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff,
    const float3 path_throughput, const float idoth)
{
    if (!TContext.enableTranslucency())
        return 1.0f;

    const OpenPBRAllCoefficients coefficients = openPBRAllCoefficients(context, refl_trans_coeff, idoth);
    const float reflection_coefficient_max =
        openPBRMaxComponentOfThroughputWeightedColor(path_throughput, coefficients.reflection_coefficient);
    const float transmission_coefficient_max =
        openPBRMaxComponentOfThroughputWeightedColor(path_throughput, coefficients.transmission_coefficient);
    // Using division rather than subtracting the other probability from one is more numerically stable.
    // openpbr_safe_divide is used to handle the case where the reflection and transmission coefficients are both 0.
    return openPBRSafeDivide(reflection_coefficient_max, reflection_coefficient_max + transmission_coefficient_max,
                             0.0f);
}

public float openPBRTransmissionProbability<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff,
    const float3 path_throughput, const float idoth)
{
    if (!TContext.enableTranslucency())
        return 0.0f;

    const OpenPBRAllCoefficients coefficients = openPBRAllCoefficients(context, refl_trans_coeff, idoth);
    const float reflection_coefficient_max =
        openPBRMaxComponentOfThroughputWeightedColor(path_throughput, coefficients.reflection_coefficient);
    const float transmission_coefficient_max =
        openPBRMaxComponentOfThroughputWeightedColor(path_throughput, coefficients.transmission_coefficient);
    // Using division rather than subtracting the other probability from one is more numerically stable.
    // openpbr_safe_divide is used to handle the case where the reflection and transmission coefficients are both 0.
    return openPBRSafeDivide(transmission_coefficient_max, reflection_coefficient_max + transmission_coefficient_max,
                             0.0f);
}

public OpenPBRAllCoefficientsAndProbabilities openPBRAllCoefficientsAndProbabilities<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff,
    const float3 path_throughput, const float idoth)
{
    const OpenPBRAllCoefficients coefficients = openPBRAllCoefficients(context, refl_trans_coeff, idoth);
    const float reflection_coefficient_max =
        openPBRMaxComponentOfThroughputWeightedColor(path_throughput, coefficients.reflection_coefficient);
    const float transmission_coefficient_max =
        openPBRMaxComponentOfThroughputWeightedColor(path_throughput, coefficients.transmission_coefficient);
    const float coefficient_max_sum = reflection_coefficient_max + transmission_coefficient_max;
    // For the probabilities, using division rather than subtracting the other probability from one
    // or multiplying by the inverse is more numerically precise, and openpbr_safe_divide is used to handle
    // the case where the reflection and transmission coefficients are both 0.
    return OpenPBRAllCoefficientsAndProbabilities(
        coefficients.reflection_coefficient, coefficients.transmission_coefficient,
        TContext.enableTranslucency() ? openPBRSafeDivide(reflection_coefficient_max, coefficient_max_sum, 0.0f) : 1.0f,
        TContext.enableTranslucency() ? openPBRSafeDivide(transmission_coefficient_max, coefficient_max_sum, 0.0f)
                                      : 0.0f);
}

public float openPBREstimateWeight<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff,
    const float3 path_throughput, const float idotn)
{
    const OpenPBRAllCoefficients coefficients = openPBRAllCoefficients(context, refl_trans_coeff, idotn);
    const float3 coefficients_sum = coefficients.reflection_coefficient + coefficients.transmission_coefficient;
    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput, coefficients_sum);
}

public float openPBREstimateWeightWhenAppliedToGgxMicrofacetDistribution<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff,
    const float3 path_throughput, const float idotn, const float isotropic_alpha)
{
    // We don't have a way to calculate the average albedo of this complex reflection/transmission coefficient
    // applied to a GGX microfacet distribution, so we just estimate the weight assuming a smooth surface.
    return openPBREstimateWeight(context, refl_trans_coeff, path_throughput, idotn);
}
} // namespace Metallic.OpenPBR

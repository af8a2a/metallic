#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_comprehensive_microfacet_lobe.h. Modified by Metallic, 2026.
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

///////////////////////////////////////////////////////
// ComprehensiveMicrofacetReflectionTransmissionLobe //
///////////////////////////////////////////////////////

public struct OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe
{
    public float3 normal_ff;
    public OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr;
    public OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff;
    public float3 eta_t_over_eta_i;
    public float3 path_throughput;
};

public float3 openPBRChannelProbabilities(const OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe)
{
    return lobe.path_throughput / openPBRSum(lobe.path_throughput);
}

public bool openPBRValidateHalfVectorForTransmission(const OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe,
                                              const float3 half_vector)
{
    // The computed half vector should point into the less dense medium.
    // If the face-forwarded normal points into the less dense medium, half_vector is valid if it faces forward into the less dense medium.
    // If the face-forwarded normal points into the denser medium, half_vector is valid if it faces backward into the less dense medium.
    return (lobe.eta_t_over_eta_i.r >= 1.0f && dot(lobe.normal_ff, half_vector) > 0.0f) ||
           (lobe.eta_t_over_eta_i.r < 1.0f && dot(lobe.normal_ff, half_vector) < 0.0f);
}

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(
    TContext context, inout OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe, const float3 normal_ff,
    const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr,
    const OpenPBRComprehensiveReflectionTransmissionCoefficient refl_trans_coeff, const float3 eta_t_over_eta_i,
    const float3 path_throughput)
{

    lobe = OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe(normal_ff, microfacet_distr, refl_trans_coeff,
                                                                    eta_t_over_eta_i, path_throughput);
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context,
                                                    const OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe)
{
    return TContext.enableTranslucency()
               ? OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection | OpenPBRBSDFLobeTypeTransmission
               : OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection;
}

public OpenPBRDiffuseSpecular openPBRCalculateLobeValue<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe, const float3 view_direction,
    const float3 light_direction)
{
    const float idotn = dot(lobe.normal_ff, view_direction);
    const float odotn = dot(lobe.normal_ff, light_direction);

    if (idotn * odotn > 0.0f) // reflection
    {
        const float3 half_vector = openPBRFastNormalize(view_direction + light_direction);

        const float idoth = dot(view_direction, half_vector);

        const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
        const float other_terms = 1.0f / (4.0f * idotn);
        const float3 F = openPBRReflectionCoefficient(context, lobe.refl_trans_coeff, abs(idoth));

        const float3 D_other_terms_F = D * other_terms * F;

        const float G = openPBREvalSmithG2(lobe.microfacet_distr, view_direction, light_direction, idotn, odotn);

        return openPBRMakeDiffuseSpecularFromSpecular(G * D_other_terms_F);
    } else if (TContext.enableTranslucency() && idotn * odotn < 0.0f) // transmission
    {
        if (TContext.enableDispersion()) {
            float3 D_other_terms_F;
            [unroll]
            for (int channel = 0; channel < OpenPBRNumRGBChannels; ++channel) {
                const float3 half_vector =
                    -openPBRFastNormalize(view_direction + light_direction * lobe.eta_t_over_eta_i[channel]);
                if (!openPBRValidateHalfVectorForTransmission(lobe, half_vector)) {
                    D_other_terms_F[channel] = 0.0f; // invalid transmission
                    continue;
                }

                const float idoth = dot(view_direction, half_vector);
                const float odoth = dot(light_direction, half_vector);
                if (idoth * odoth >= 0.0f) {
                    D_other_terms_F[channel] = 0.0f; // invalid transmission
                    continue;
                }

                const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
                const float other_terms = openPBRSquare(lobe.eta_t_over_eta_i[channel]) * abs(idoth) * abs(odoth) /
                                          (openPBRSquare(idoth + lobe.eta_t_over_eta_i[channel] * odoth) * idotn);
                const float F = openPBRTransmissionCoefficient(context, lobe.refl_trans_coeff, abs(idoth))[channel];

                D_other_terms_F[channel] = D * other_terms * F;
            }

            const float G = openPBREvalSmithG2(lobe.microfacet_distr, view_direction, light_direction, idotn, odotn);

            return openPBRMakeDiffuseSpecularFromSpecular(G * D_other_terms_F);
        } else {
            const float3 half_vector =
                -openPBRFastNormalize(view_direction + light_direction * lobe.eta_t_over_eta_i.r);
            if (!openPBRValidateHalfVectorForTransmission(lobe, half_vector))
                return openPBRMakeZeroDiffuseSpecular(); // invalid transmission

            const float idoth = dot(view_direction, half_vector);
            const float odoth = dot(light_direction, half_vector);
            if (idoth * odoth >= 0.0f)
                return openPBRMakeZeroDiffuseSpecular(); // invalid transmission

            const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
            const float other_terms = openPBRSquare(lobe.eta_t_over_eta_i.r) * abs(idoth) * abs(odoth) /
                                      (openPBRSquare(idoth + lobe.eta_t_over_eta_i.r * odoth) * idotn);
            const float3 F = openPBRTransmissionCoefficient(context, lobe.refl_trans_coeff, abs(idoth));

            const float3 D_other_terms_F = D * other_terms * F;

            const float G = openPBREvalSmithG2(lobe.microfacet_distr, view_direction, light_direction, idotn, odotn);

            return openPBRMakeDiffuseSpecularFromSpecular(G * D_other_terms_F);
        }
    } else // neither reflection nor transmission
        return openPBRMakeZeroDiffuseSpecular();

    // If cosine term needs to be excluded, code like this could be used:
    //     if (!IncludeCosineTerm)
    //         result /= odotn;
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe, const float3 view_direction,
    const float3 light_direction)
{
    const float idotn = dot(lobe.normal_ff, view_direction);
    const float odotn = dot(lobe.normal_ff, light_direction);

    if (idotn * odotn > 0.0f) // reflection
    {
        const float3 half_vector = openPBRFastNormalize(view_direction + light_direction);

        const float idoth = dot(view_direction, half_vector);
        const float odoth = dot(light_direction, half_vector);

        // Return invalid PDF if half vector rounding error would result in negative PDF
        if (idoth * odoth < 0.0f)
            return 0.0f;

        const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
        const float other_terms = idoth / (4.0f * odoth * idotn);
        const float F_prob =
            openPBRReflectionProbability(context, lobe.refl_trans_coeff, lobe.path_throughput, abs(idoth));

        const float F_prob_D_other_terms = F_prob * D * other_terms;

        const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);

        return G_masking * F_prob_D_other_terms;
    } else if (TContext.enableTranslucency() && idotn * odotn < 0.0f) // transmission
    {
        if (TContext.enableDispersion()) {
            float3 F_prob_D_other_terms;
            [unroll]
            for (int channel = 0; channel < OpenPBRNumRGBChannels; ++channel) {
                const float3 half_vector =
                    -openPBRFastNormalize(view_direction + light_direction * lobe.eta_t_over_eta_i[channel]);
                if (!openPBRValidateHalfVectorForTransmission(lobe, half_vector)) {
                    F_prob_D_other_terms[channel] = 0.0f; // invalid transmission
                    continue;
                }

                const float idoth = dot(view_direction, half_vector);
                const float odoth = dot(light_direction, half_vector);
                if (idoth * odoth >= 0.0f) {
                    F_prob_D_other_terms[channel] = 0.0f; // invalid transmission
                    continue;
                }

                const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
                const float other_terms = openPBRSquare(lobe.eta_t_over_eta_i[channel]) * abs(idoth) * abs(odoth) /
                                          (openPBRSquare(idoth + lobe.eta_t_over_eta_i[channel] * odoth) * idotn);
                const float F_prob =
                    openPBRTransmissionProbability(context, lobe.refl_trans_coeff, lobe.path_throughput, abs(idoth));

                F_prob_D_other_terms[channel] = F_prob * D * other_terms;
            }

            const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);

            return G_masking * dot(F_prob_D_other_terms, openPBRChannelProbabilities(lobe));
        } else {
            const float3 half_vector =
                -openPBRFastNormalize(view_direction + light_direction * lobe.eta_t_over_eta_i.r);
            if (!openPBRValidateHalfVectorForTransmission(lobe, half_vector))
                return 0.0f; // invalid transmission

            const float idoth = dot(view_direction, half_vector);
            const float odoth = dot(light_direction, half_vector);
            if (idoth * odoth >= 0.0f)
                return 0.0f; // invalid transmission

            const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
            const float other_terms = openPBRSquare(lobe.eta_t_over_eta_i.r) * abs(idoth) * abs(odoth) /
                                      (openPBRSquare(idoth + lobe.eta_t_over_eta_i.r * odoth) * idotn);
            const float F_prob =
                openPBRTransmissionProbability(context, lobe.refl_trans_coeff, lobe.path_throughput, abs(idoth));

            const float F_prob_D_other_terms = F_prob * D * other_terms;

            const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);

            return G_masking * F_prob_D_other_terms;
        }
    } else // neither reflection nor transmission
        return 0.0f;
}

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context,
                                                   const OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe,
                                                   const float3 rand, const float3 view_direction,
                                                   out float3 light_direction, out OpenPBRDiffuseSpecular weight,
                                                   out float pdf, out uint sampled_type)
{
    const float idotn = dot(view_direction, lobe.normal_ff);

    // Choose microfacet normal_ff (in the caller's coordinate space).
    const float3 half_vector =
        openPBRSampleGgxSmithVndf(lobe.microfacet_distr, view_direction, lobe.normal_ff, (rand).xy);

    const float idoth = dot(view_direction, half_vector);
    // Sampled microfacet normal is expected to be in incident hemisphere,
    // but occasionally is not due to rounding error.
    if (idoth < 0.0f) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    // Select reflection or transmission.
    OpenPBRAllCoefficientsAndProbabilities coeffs_and_probs =
        openPBRAllCoefficientsAndProbabilities(context, lobe.refl_trans_coeff, lobe.path_throughput, idoth);
    const float3 F_refl = coeffs_and_probs.reflection_coefficient;
    const float3 F_trans = coeffs_and_probs.transmission_coefficient;
    const float P_refl = coeffs_and_probs.reflection_probability;
    const float P_trans = coeffs_and_probs.transmission_probability;

    if (P_refl + P_trans == 0.0f) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    if (!TContext.enableTranslucency() || rand.z < P_refl) // reflection
    {
        // The light direction is normalized by construction, but can be significantly off due to rounding error.
        light_direction = openPBRFastNormalize(-view_direction + half_vector * 2.0f * idoth);

        // Check if reflection is on correct side of surface.
        const float odotn = dot(lobe.normal_ff, light_direction);
        if (idotn * odotn <= 0.0f) {
            openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
            return false;
        }

        const float odoth = dot(light_direction, half_vector);

        // If the input normal and view direction are normalized, this assertion should never fail.

        const float G_shadowing = openPBREvalSmithG1(lobe.microfacet_distr, light_direction, odotn);
        weight = openPBRMakeDiffuseSpecularFromSpecular((F_refl * (1.0f / P_refl)) * G_shadowing);

        const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
        if (D <= 0.0f) {
            openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
            return false;
        }

        const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);
        pdf = P_refl * D * G_masking * idoth / (4.0f * odoth * idotn);

        // It's very important to return the correct type to ensure correct treatment by outside systems.
        sampled_type = OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection;

        return true;
    } else // transmission
    {
        if (TContext.enableDispersion()) {
            const float remapped_rand_z = (rand.z - P_refl) / P_trans;
            int selected_color_channel;
            const float3 color_channel_probabilities = openPBRChannelProbabilities(lobe);
            if (remapped_rand_z < color_channel_probabilities[0])
                selected_color_channel = 0;
            else if (remapped_rand_z < color_channel_probabilities[0] + color_channel_probabilities[1])
                selected_color_channel = 1;
            else
                // TODO: Address numerical vulnerabilities.
                selected_color_channel = 2;

            if (!openPBRRefract(view_direction, half_vector, idoth, lobe.eta_t_over_eta_i[selected_color_channel],
                                light_direction)) {
                openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
                return false;
            }

            // Check if transmission is on correct side of surface.
            const float odotn = dot(light_direction, lobe.normal_ff);
            if (idotn * odotn >= 0.0f) {
                openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
                return false;
            }

            // TODO: Only perform the necessary calculations.
            pdf = openPBRCalculateLobePdf(context, lobe, view_direction, light_direction);
            if (pdf == 0.0f) {
                openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
                return false;
            }
            const OpenPBRDiffuseSpecular bsdf_cos =
                openPBRCalculateLobeValue(context, lobe, view_direction, light_direction);
            weight = openPBRScaleDiffuseSpecular(bsdf_cos, 1.0f / pdf);

            // It's very important to return the correct type to ensure correct treatment by outside systems.
            sampled_type = OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeTransmission;

            return true;
        } else {
            if (!openPBRRefract(view_direction, half_vector, idoth, lobe.eta_t_over_eta_i.r, light_direction)) {
                openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
                return false;
            }

            // Check if transmission is on correct side of surface.
            const float odotn = dot(light_direction, lobe.normal_ff);
            if (idotn * odotn >= 0.0f) {
                openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
                return false;
            }

            const float odoth = dot(light_direction, half_vector);

            const float G_shadowing = openPBREvalSmithG1(lobe.microfacet_distr, light_direction, odotn);
            weight = openPBRMakeDiffuseSpecularFromSpecular((F_trans * (1.0f / P_trans)) * G_shadowing);

            const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
            if (D <= 0.0f) {
                openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
                return false;
            }

            const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);
            pdf = P_trans * D * G_masking * openPBRSquare(lobe.eta_t_over_eta_i.r) * abs(idoth) * abs(odoth) /
                  (openPBRSquare(idoth + lobe.eta_t_over_eta_i.r * odoth) * idotn);

            // It's very important to return the correct type to ensure correct treatment by outside systems.
            sampled_type = OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeTransmission;

            return true;
        }
    }
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe lobe, const float3 view_direction,
    const float3 path_throughput)
{
    const float idotn = dot(view_direction, lobe.normal_ff);
    // Here we assume that the microfacet distribution is a GGX one and calculate the estimated weight accordingly.
    return openPBREstimateWeightWhenAppliedToGgxMicrofacetDistribution(
        context, lobe.refl_trans_coeff, path_throughput, idotn, openPBRGetIsotropicAlpha(lobe.microfacet_distr));
}
} // namespace Metallic.OpenPBR

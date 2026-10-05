#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_microfacet_multiple_scattering_lobes.h. Modified by Metallic, 2026.
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

public static const float OpenPBRMinRoughnessWithVisibleEnergyLoss = 0.04f; // empirically derived
public static const float OpenPBRMinAlphaWithVisibleEnergyLoss =
    OpenPBRMinRoughnessWithVisibleEnergyLoss * OpenPBRMinRoughnessWithVisibleEnergyLoss;

////////////////////////////////////////////////
// DielectricMicrofacetMultipleScatteringLobe //
////////////////////////////////////////////////

public struct OpenPBRDielectricMicrofacetMultipleScatteringLobe
{
    public float3 normal_ff;
    public float alpha;
    public float eta_t_over_eta_i;
    public float3 scale_refl;
    public float3 scale_trans;
    public float reflection_ratio_ti;
    public float energy_complement_ti_idotn;
};

public float openPBRClampedTabulatedFactors<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe, float odotn)
{
    float ior_to_use;
    if (odotn > 0.0f) {
        ior_to_use = lobe.eta_t_over_eta_i;
    } else {
        odotn *= -1.0f;
        const float eta_i_over_eta_t = 1.0f / lobe.eta_t_over_eta_i;
        ior_to_use = eta_i_over_eta_t;
    }

    const float tabulated_factors =
        lobe.energy_complement_ti_idotn *
        openPBRLookUpIdealDielectricEnergyComplement(context, ior_to_use, lobe.alpha, odotn) /
        openPBRClampAverageEnergyComplementAboveZero(
            openPBRLookUpIdealDielectricAverageEnergyComplement(context, ior_to_use, lobe.alpha));

    // This clamping can prevent weights from getting out of control near grazing angles at low roughnesses.
    return min(tabulated_factors, openPBRSafeDivide(1.0f, odotn, 0.0f));
}

public float openPBRClampedTabulatedFactorsIncludingRatio<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe, float odotn)
{
    float ior_to_use;
    float ratio_to_use;
    if (odotn > 0.0f) {
        ior_to_use = lobe.eta_t_over_eta_i;
        ratio_to_use = lobe.reflection_ratio_ti;
    } else {
        odotn *= -1.0f;
        const float eta_i_over_eta_t = 1.0f / lobe.eta_t_over_eta_i;
        ior_to_use = eta_i_over_eta_t;
        ratio_to_use = 1.0f - lobe.reflection_ratio_ti;
    }

    float tabulated_factors = lobe.energy_complement_ti_idotn *
                              openPBRLookUpIdealDielectricEnergyComplement(context, ior_to_use, lobe.alpha, odotn) /
                              openPBRClampAverageEnergyComplementAboveZero(
                                  openPBRLookUpIdealDielectricAverageEnergyComplement(context, ior_to_use, lobe.alpha));
    tabulated_factors *= ratio_to_use;

    // This clamping is necessary to avoid weights from getting out of control near grazing angles at low roughnesses.
    return min(tabulated_factors, openPBRSafeDivide(1.0f, odotn, 0.0f));
}

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(TContext context,
                                                       inout OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe,
                                                       const float3 normal_ff, const float3 view_direction,
                                                       const float alpha, const float eta_t_over_eta_i,
                                                       const float3 scale_refl, const float3 scale_trans)
{
    // Copied properties.
    lobe.normal_ff = normal_ff;
    lobe.alpha = alpha;
    lobe.eta_t_over_eta_i = eta_t_over_eta_i;
    lobe.scale_refl = scale_refl;
    lobe.scale_trans = scale_trans;

    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return;

    // Derived properties.
    lobe.reflection_ratio_ti = openPBRLookUpIdealDielectricReflectionRatio(context, lobe.eta_t_over_eta_i, lobe.alpha);
    const float idotn = dot(view_direction, lobe.normal_ff);
    lobe.energy_complement_ti_idotn =
        openPBRLookUpIdealDielectricEnergyComplement(context, lobe.eta_t_over_eta_i, lobe.alpha, idotn);
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context,
                                                    const OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe)
{
    return OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection | OpenPBRBSDFLobeTypeTransmission;
}

public OpenPBRDiffuseSpecular
openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                      const OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe,
                                                      const float3 view_direction, const float3 light_direction)
{
    // Physical zero multiplier; leave the sampling distribution unchanged.
    if (all(lobe.scale_refl == float3(0)) && all(lobe.scale_trans == float3(0))) { return openPBRMakeZeroDiffuseSpecular(); }

    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return openPBRMakeZeroDiffuseSpecular();

    const float odotn = dot(light_direction, lobe.normal_ff);
    const float unscaled = openPBRClampedTabulatedFactorsIncludingRatio(context, lobe, odotn) * OpenPBRRcpPi;
    const float unscaled_cos = unscaled * abs(odotn);
    if (odotn > 0.0f) {
        return openPBRMakeDiffuseSpecularFromSpecular(lobe.scale_refl * unscaled_cos);
    } else {
        return openPBRMakeDiffuseSpecularFromSpecular(lobe.scale_trans * unscaled_cos);
    }
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context,
                                                          const OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return 0.0f;

    const float odotn = dot(light_direction, lobe.normal_ff);
    if (odotn > 0.0f) {
        return OpenPBRRcpTwoPi * lobe.reflection_ratio_ti;
    } else {
        return OpenPBRRcpTwoPi * (1.0f - lobe.reflection_ratio_ti);
    }
}

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context,
                                                   const OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe,
                                                   const float3 rand, const float3 view_direction,
                                                   out float3 light_direction, out OpenPBRDiffuseSpecular weight,
                                                   out float pdf, out uint sampled_type)
{
    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    // Choose reflection or transmission first.
    float rand_x = rand.x;
    float z;
    if (rand_x < lobe.reflection_ratio_ti) // reflection
    {
        rand_x /= lobe.reflection_ratio_ti; // remap used random number
        openPBRClampRemappedRandomNumber(rand_x);
        z = rand_x;
    } else // transmission
    {
        rand_x = (rand_x - lobe.reflection_ratio_ti) / (1.0f - lobe.reflection_ratio_ti); // remap used random number
        openPBRClampRemappedRandomNumber(rand_x);
        z = -rand_x;
    }

    const float horizontal = sqrt(1.0f - openPBRSquare(z));
    const float azimuth = OpenPBRTwoPi * rand.y;
    const float y = sin(azimuth) * horizontal;
    const float x = cos(azimuth) * horizontal;

    light_direction = openPBRLocalToWorld(openPBRMakeBasis(lobe.normal_ff), float3(x, y, z));

    const float odotn = dot(light_direction, lobe.normal_ff);
    const float unscaled_weight = 2.0f * abs(odotn) * openPBRClampedTabulatedFactors(context, lobe, odotn);

    if (odotn > 0.0f) // reflection
    {
        weight = openPBRMakeDiffuseSpecularFromSpecular(lobe.scale_refl * unscaled_weight);
        pdf = OpenPBRRcpTwoPi * lobe.reflection_ratio_ti;
        sampled_type = OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection;
    } else // transmission
    {
        weight = openPBRMakeDiffuseSpecularFromSpecular(lobe.scale_trans * unscaled_weight);
        pdf = OpenPBRRcpTwoPi * (1.0f - lobe.reflection_ratio_ti);
        sampled_type = OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeTransmission;
    }

    return true;
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRDielectricMicrofacetMultipleScatteringLobe lobe, const float3 view_direction,
    const float3 path_throughput)
{
    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return 0.0f;

    const float3 weighted_average_scale =
        lobe.scale_refl * lobe.reflection_ratio_ti + lobe.scale_trans * (1.0f - lobe.reflection_ratio_ti);
    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput, weighted_average_scale) *
           lobe.energy_complement_ti_idotn;
}

///////////////////////////////////////////
// MetalMicrofacetMultipleScatteringLobe //
///////////////////////////////////////////

public struct OpenPBRMetalMicrofacetMultipleScatteringLobe
{
    public float3 normal_ff;
    public float alpha;
    public float3 scale;
    public float energy_complement_idotn;
};

public float openPBRClampedTabulatedFactors<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRMetalMicrofacetMultipleScatteringLobe lobe, const float odotn)
{
    const float tabulated_factors = lobe.energy_complement_idotn *
                                    openPBRLookUpIdealMetalEnergyComplement(context, lobe.alpha, odotn) /
                                    openPBRClampAverageEnergyComplementAboveZero(
                                        openPBRLookUpIdealMetalAverageEnergyComplement(context, lobe.alpha));
    // This clamping can prevent weights from getting out of control near grazing angles at low roughnesses.
    return min(tabulated_factors, openPBRSafeDivide(1.0f, odotn, 0.0f));
}

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(TContext context,
                                                       inout OpenPBRMetalMicrofacetMultipleScatteringLobe lobe,
                                                       const float3 normal_ff, const float3 view_direction,
                                                       const float alpha, const float3 scale)
{
    // Copied properties.
    lobe.normal_ff = normal_ff;
    lobe.alpha = alpha;
    lobe.scale = scale;

    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return;

    // Derived properties.
    const float idotn = dot(view_direction, lobe.normal_ff);
    lobe.energy_complement_idotn = openPBRLookUpIdealMetalEnergyComplement(context, lobe.alpha, idotn);
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context,
                                                    const OpenPBRMetalMicrofacetMultipleScatteringLobe lobe)
{
    return OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection;
}

public OpenPBRDiffuseSpecular
openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                      const OpenPBRMetalMicrofacetMultipleScatteringLobe lobe,
                                                      const float3 view_direction, const float3 light_direction)
{
    // Physical zero multiplier; leave the sampling distribution unchanged.
    if (all(lobe.scale == float3(0))) { return openPBRMakeZeroDiffuseSpecular(); }

    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return openPBRMakeZeroDiffuseSpecular();

    const float odotn = dot(light_direction, lobe.normal_ff);
    if (odotn > 0.0f) {
        const float unscaled = openPBRClampedTabulatedFactors(context, lobe, odotn) * OpenPBRRcpPi;
        const float unscaled_cos = unscaled * odotn;
        return openPBRMakeDiffuseSpecularFromSpecular(lobe.scale * unscaled_cos);
    } else {
        return openPBRMakeZeroDiffuseSpecular();
    }
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context,
                                                          const OpenPBRMetalMicrofacetMultipleScatteringLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return 0.0f;

    const float odotn = dot(light_direction, lobe.normal_ff);
    if (odotn > 0.0f) {
        return OpenPBRRcpTwoPi;
    } else {
        return 0.0f;
    }
}

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context,
                                                   const OpenPBRMetalMicrofacetMultipleScatteringLobe lobe,
                                                   const float3 rand, const float3 view_direction,
                                                   out float3 light_direction, out OpenPBRDiffuseSpecular weight,
                                                   out float pdf, out uint sampled_type)
{
    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    const float z = rand.x;
    const float horizontal = sqrt(1.0f - openPBRSquare(z));
    const float azimuth = OpenPBRTwoPi * rand.y;
    const float y = sin(azimuth) * horizontal;
    const float x = cos(azimuth) * horizontal;

    light_direction = openPBRLocalToWorld(openPBRMakeBasis(lobe.normal_ff), float3(x, y, z));

    const float odotn = dot(light_direction, lobe.normal_ff);
    const float unscaled_weight = 2.0f * odotn * openPBRClampedTabulatedFactors(context, lobe, odotn);

    weight = openPBRMakeDiffuseSpecularFromSpecular(lobe.scale * unscaled_weight);
    pdf = OpenPBRRcpTwoPi;
    sampled_type = openPBRGetLobeType(context, lobe);

    return true;
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRMetalMicrofacetMultipleScatteringLobe lobe, const float3 view_direction,
    const float3 path_throughput)
{
    if (lobe.alpha < OpenPBRMinAlphaWithVisibleEnergyLoss)
        return 0.0f;

    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput, lobe.scale) * lobe.energy_complement_idotn;
}
} // namespace Metallic.OpenPBR

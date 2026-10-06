#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_aggregate_lobe.h. Modified by Metallic, 2026.
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

///////////////////
// AggregateLobe //
///////////////////

// This lobe stores all of the lobes that constitute the base surface.
// These lobes are assumed to be already weighted so that they can exist side by side without any nesting logic.
// In the names below, the acronym MMS stands for "microfacet multiple scattering".

public static const int OpenPBRInvalidLobeIndex = -1;
public static const int OpenPBRSpecularLobeIndex = 0;
public static const int OpenPBRDielectricMMSLobeIndex = 1;
public static const int OpenPBRMetalMMSLobeIndex = 2;
public static const int OpenPBRDiffuseLobeIndex = 3;
public static const int OpenPBRThinWallSpecularTransLobeIndex = 4;
public static const int OpenPBRThinWallDiffuseTransLobeIndex = 5;
public static const int OpenPBRNumBaseLobes = 6;

public struct OpenPBRAggregateLobe
{
    public OpenPBRComprehensiveMicrofacetReflectionTransmissionLobe specular_lobe;
    public OpenPBRDielectricMicrofacetMultipleScatteringLobe dielectric_mms_lobe;
    public OpenPBRMetalMicrofacetMultipleScatteringLobe metal_mms_lobe;
    public OpenPBREnergyConservingRoughDiffuseLobe diffuse_lobe;
    public OpenPBRThinWallSpecularTransmissionLobe thin_wall_specular_trans_lobe;
    public OpenPBRThinWallDiffuseTransmissionLobe thin_wall_diffuse_trans_lobe;

    public float lobe_weights[OpenPBRNumBaseLobes];
};

public int openPBRSelectLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRAggregateLobe lobe, inout float rand,
                                                  out float selected_lobe_weight, out float total_weight)
{
    total_weight = 0.0f;

    total_weight += lobe.lobe_weights[OpenPBRSpecularLobeIndex];
    if (TContext.enableTranslucency())
        total_weight += lobe.lobe_weights[OpenPBRDielectricMMSLobeIndex];
    if (TContext.enableMetallic())
        total_weight += lobe.lobe_weights[OpenPBRMetalMMSLobeIndex];
    total_weight += lobe.lobe_weights[OpenPBRDiffuseLobeIndex];
    total_weight += lobe.lobe_weights[OpenPBRThinWallSpecularTransLobeIndex];
    total_weight += lobe.lobe_weights[OpenPBRThinWallDiffuseTransLobeIndex];

    // Abort sampling if total_weight is zero, subnormal, or the minimum normal float.
    // If total_weight were subnormal or equal to OpenPBR_FloatMin, then multiplying rand by
    // total_weight -- which always makes rand less than total_weight when the original
    // rand is less than one and total_weight is a normal float -- would occasionally
    // make rand equal to total_weight, resulting in a corrupt sample of the last lobe.
    if (total_weight <= OpenPBRFloatMin) {
        selected_lobe_weight = 0.0f;
        return OpenPBRInvalidLobeIndex;
    }

    rand *= total_weight;

    float weight_so_far = 0.0f;

    // Loop through all but the last lobe index.
    [unroll]
    for (int lobe_index = 0; lobe_index < OpenPBRNumBaseLobes - 1; ++lobe_index) {
        // Skip any lobes that are disabled by specialization constants.
        if ((!TContext.enableTranslucency() && lobe_index == OpenPBRDielectricMMSLobeIndex) ||
            (!TContext.enableMetallic() && lobe_index == OpenPBRMetalMMSLobeIndex))
            continue;

        selected_lobe_weight = lobe.lobe_weights[lobe_index];
        if (rand < weight_so_far + selected_lobe_weight) {
            rand = (rand - weight_so_far) / selected_lobe_weight;
            openPBRClampRemappedRandomNumber(rand);
            return lobe_index;
        }
        weight_so_far += selected_lobe_weight;
    }

    const int OpenPBRLastLobeIndex = OpenPBRNumBaseLobes - 1;
    // The very last lobe is currently the thin-wall diffuse-transmission lobe.
    static_assert(OpenPBRLastLobeIndex == OpenPBRThinWallDiffuseTransLobeIndex, "Unexpected lobe index");
    selected_lobe_weight = lobe.lobe_weights[OpenPBRLastLobeIndex];
    // The logic is simplified on the last iteration.

    rand = (rand - weight_so_far) / selected_lobe_weight;
    openPBRClampRemappedRandomNumber(rand);
    return OpenPBRLastLobeIndex;
}

// Assumes that child lobes have already been initialized.

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(TContext context, inout OpenPBRAggregateLobe lobe,
                                                       const float3 view_direction, const float3 path_throughput)
{
    lobe.lobe_weights[OpenPBRSpecularLobeIndex] =
        openPBREstimateLobeContribution(context, lobe.specular_lobe, view_direction, path_throughput);
    if (TContext.enableTranslucency())
        lobe.lobe_weights[OpenPBRDielectricMMSLobeIndex] =
            openPBREstimateLobeContribution(context, lobe.dielectric_mms_lobe, view_direction, path_throughput);
    if (TContext.enableMetallic())
        lobe.lobe_weights[OpenPBRMetalMMSLobeIndex] =
            openPBREstimateLobeContribution(context, lobe.metal_mms_lobe, view_direction, path_throughput);
    lobe.lobe_weights[OpenPBRDiffuseLobeIndex] =
        openPBREstimateLobeContribution(context, lobe.diffuse_lobe, view_direction, path_throughput);
    lobe.lobe_weights[OpenPBRThinWallSpecularTransLobeIndex] =
        openPBREstimateLobeContribution(context, lobe.thin_wall_specular_trans_lobe, view_direction, path_throughput);
    lobe.lobe_weights[OpenPBRThinWallDiffuseTransLobeIndex] =
        openPBREstimateLobeContribution(context, lobe.thin_wall_diffuse_trans_lobe, view_direction, path_throughput);
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context, const OpenPBRAggregateLobe lobe)
{
    uint result = OpenPBRBSDFLobeTypeNone;

    result |= openPBRGetLobeType(context, lobe.specular_lobe);
    if (TContext.enableTranslucency())
        result |= openPBRGetLobeType(context, lobe.dielectric_mms_lobe);
    if (TContext.enableMetallic())
        result |= openPBRGetLobeType(context, lobe.metal_mms_lobe);
    result |= openPBRGetLobeType(context, lobe.diffuse_lobe);
    result |= openPBRGetLobeType(context, lobe.thin_wall_specular_trans_lobe);
    result |= openPBRGetLobeType(context, lobe.thin_wall_diffuse_trans_lobe);

    return result;
}

public OpenPBRDiffuseSpecular openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                                             const OpenPBRAggregateLobe lobe,
                                                                             const float3 view_direction,
                                                                             const float3 light_direction)
{
    OpenPBRDiffuseSpecular result = openPBRMakeZeroDiffuseSpecular();

    result = openPBRAddDiffuseSpecular(
        result, openPBRCalculateLobeValue(context, lobe.specular_lobe, view_direction, light_direction));
    if (TContext.enableTranslucency())
        result = openPBRAddDiffuseSpecular(
            result, openPBRCalculateLobeValue(context, lobe.dielectric_mms_lobe, view_direction, light_direction));
    if (TContext.enableMetallic())
        result = openPBRAddDiffuseSpecular(
            result, openPBRCalculateLobeValue(context, lobe.metal_mms_lobe, view_direction, light_direction));
    result = openPBRAddDiffuseSpecular(
        result, openPBRCalculateLobeValue(context, lobe.diffuse_lobe, view_direction, light_direction));
    result = openPBRAddDiffuseSpecular(result, openPBRCalculateLobeValue(context, lobe.thin_wall_specular_trans_lobe,
                                                                         view_direction, light_direction));
    result = openPBRAddDiffuseSpecular(
        result, openPBRCalculateLobeValue(context, lobe.thin_wall_diffuse_trans_lobe, view_direction, light_direction));

    return result;
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context, const OpenPBRAggregateLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    float sum = 0.0f;
    float total_weight = 0.0f;

    {
        const float weight = lobe.lobe_weights[OpenPBRSpecularLobeIndex];
        sum += weight * openPBRCalculateLobePdf(context, lobe.specular_lobe, view_direction, light_direction);
        total_weight += weight;
    }
    if (TContext.enableTranslucency()) {
        const float weight = lobe.lobe_weights[OpenPBRDielectricMMSLobeIndex];
        sum += weight * openPBRCalculateLobePdf(context, lobe.dielectric_mms_lobe, view_direction, light_direction);
        total_weight += weight;
    }
    if (TContext.enableMetallic()) {
        const float weight = lobe.lobe_weights[OpenPBRMetalMMSLobeIndex];
        sum += weight * openPBRCalculateLobePdf(context, lobe.metal_mms_lobe, view_direction, light_direction);
        total_weight += weight;
    }
    {
        const float weight = lobe.lobe_weights[OpenPBRDiffuseLobeIndex];
        sum += weight * openPBRCalculateLobePdf(context, lobe.diffuse_lobe, view_direction, light_direction);
        total_weight += weight;
    }
    {
        const float weight = lobe.lobe_weights[OpenPBRThinWallSpecularTransLobeIndex];
        sum += weight *
               openPBRCalculateLobePdf(context, lobe.thin_wall_specular_trans_lobe, view_direction, light_direction);
        total_weight += weight;
    }
    {
        const float weight = lobe.lobe_weights[OpenPBRThinWallDiffuseTransLobeIndex];
        sum += weight *
               openPBRCalculateLobePdf(context, lobe.thin_wall_diffuse_trans_lobe, view_direction, light_direction);
        total_weight += weight;
    }

    return total_weight > 0.0f ? sum / total_weight : 0.0f;
}

// Given a view direction, sample a light direction,
// evaluate the throughput and pdf for this direction,
// and return whether a sample was successfully generated.
// Output arguments are only set when the function returns true.

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRAggregateLobe lobe, const float3 rand,
                                                   const float3 view_direction, out float3 light_direction,
                                                   out OpenPBRDiffuseSpecular weight, out float pdf,
                                                   out uint sampled_type)
{
    float rand_x = rand.x;
    float selected_lobe_weight;
    float total_weight;
    const int selected_lobe_index = openPBRSelectLobe(context, lobe, rand_x, selected_lobe_weight, total_weight);

    if (selected_lobe_index == OpenPBRInvalidLobeIndex) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    const float3 remapped_rand = float3(rand_x, rand.y, rand.z);

    bool valid_sample = false;
    if (selected_lobe_index == OpenPBRSpecularLobeIndex)
        valid_sample = openPBRSampleLobe(context, lobe.specular_lobe, remapped_rand, view_direction, light_direction,
                                         weight, pdf, sampled_type);
    else if (TContext.enableTranslucency() && selected_lobe_index == OpenPBRDielectricMMSLobeIndex)
        valid_sample = openPBRSampleLobe(context, lobe.dielectric_mms_lobe, remapped_rand, view_direction,
                                         light_direction, weight, pdf, sampled_type);
    else if (TContext.enableMetallic() && selected_lobe_index == OpenPBRMetalMMSLobeIndex)
        valid_sample = openPBRSampleLobe(context, lobe.metal_mms_lobe, remapped_rand, view_direction, light_direction,
                                         weight, pdf, sampled_type);
    else if (selected_lobe_index == OpenPBRDiffuseLobeIndex)
        valid_sample = openPBRSampleLobe(context, lobe.diffuse_lobe, remapped_rand, view_direction, light_direction,
                                         weight, pdf, sampled_type);
    else if (selected_lobe_index == OpenPBRThinWallSpecularTransLobeIndex)
        valid_sample = openPBRSampleLobe(context, lobe.thin_wall_specular_trans_lobe, remapped_rand, view_direction,
                                         light_direction, weight, pdf, sampled_type);
    else if (selected_lobe_index == OpenPBRThinWallDiffuseTransLobeIndex)
        valid_sample = openPBRSampleLobe(context, lobe.thin_wall_diffuse_trans_lobe, remapped_rand, view_direction,
                                         light_direction, weight, pdf, sampled_type);
    else {
    }

    if (!valid_sample)
        return false;

    // Compute overall weight and pdf with all matched lobes.
    if (!bool(sampled_type & OpenPBRBSDFLobeTypeSpecular)) {

        OpenPBRDiffuseSpecular bsdf_cos = openPBRScaleDiffuseSpecular(weight, pdf);
        pdf *= selected_lobe_weight;

        if (selected_lobe_index != OpenPBRSpecularLobeIndex) {
            bsdf_cos = openPBRAddDiffuseSpecular(
                bsdf_cos, openPBRCalculateLobeValue(context, lobe.specular_lobe, view_direction, light_direction));
            pdf += lobe.lobe_weights[OpenPBRSpecularLobeIndex] *
                   openPBRCalculateLobePdf(context, lobe.specular_lobe, view_direction, light_direction);
        }
        if (TContext.enableTranslucency() && selected_lobe_index != OpenPBRDielectricMMSLobeIndex) {
            bsdf_cos = openPBRAddDiffuseSpecular(bsdf_cos, openPBRCalculateLobeValue(context, lobe.dielectric_mms_lobe,
                                                                                     view_direction, light_direction));
            pdf += lobe.lobe_weights[OpenPBRDielectricMMSLobeIndex] *
                   openPBRCalculateLobePdf(context, lobe.dielectric_mms_lobe, view_direction, light_direction);
        }
        if (TContext.enableMetallic() && selected_lobe_index != OpenPBRMetalMMSLobeIndex) {
            bsdf_cos = openPBRAddDiffuseSpecular(
                bsdf_cos, openPBRCalculateLobeValue(context, lobe.metal_mms_lobe, view_direction, light_direction));
            pdf += lobe.lobe_weights[OpenPBRMetalMMSLobeIndex] *
                   openPBRCalculateLobePdf(context, lobe.metal_mms_lobe, view_direction, light_direction);
        }
        if (selected_lobe_index != OpenPBRDiffuseLobeIndex) {
            bsdf_cos = openPBRAddDiffuseSpecular(
                bsdf_cos, openPBRCalculateLobeValue(context, lobe.diffuse_lobe, view_direction, light_direction));
            pdf += lobe.lobe_weights[OpenPBRDiffuseLobeIndex] *
                   openPBRCalculateLobePdf(context, lobe.diffuse_lobe, view_direction, light_direction);
        }
        if (selected_lobe_index != OpenPBRThinWallSpecularTransLobeIndex) {
            bsdf_cos = openPBRAddDiffuseSpecular(bsdf_cos,
                                                 openPBRCalculateLobeValue(context, lobe.thin_wall_specular_trans_lobe,
                                                                           view_direction, light_direction));
            pdf +=
                lobe.lobe_weights[OpenPBRThinWallSpecularTransLobeIndex] *
                openPBRCalculateLobePdf(context, lobe.thin_wall_specular_trans_lobe, view_direction, light_direction);
        }
        if (selected_lobe_index != OpenPBRThinWallDiffuseTransLobeIndex) {
            bsdf_cos = openPBRAddDiffuseSpecular(
                bsdf_cos,
                openPBRCalculateLobeValue(context, lobe.thin_wall_diffuse_trans_lobe, view_direction, light_direction));
            pdf += lobe.lobe_weights[OpenPBRThinWallDiffuseTransLobeIndex] *
                   openPBRCalculateLobePdf(context, lobe.thin_wall_diffuse_trans_lobe, view_direction, light_direction);
        }

        pdf /= total_weight;                                        // PDF takes all lobes into account.
        weight = openPBRScaleDiffuseSpecular(bsdf_cos, 1.0f / pdf); // Weight takes all lobes into account.

        // NOTE: The sampled_type is intentionally not updated here -- it represents
        // the type of lobe that directly generated the sample.
    } else {
        // If a delta lobe is sampled, don't attempt to combine its weight with the other lobes
        // (theoretically multiple delta lobes could be combined but we're not doing that).
        weight = openPBRScaleDiffuseSpecular(weight, total_weight / selected_lobe_weight);
    }

    return true;
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context, const OpenPBRAggregateLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    float result = 0.0f;

    [unroll]
    for (int lobe_index = 0; lobe_index < OpenPBRNumBaseLobes; ++lobe_index) {
        if ((!TContext.enableTranslucency() && lobe_index == OpenPBRDielectricMMSLobeIndex) ||
            (!TContext.enableMetallic() && lobe_index == OpenPBRMetalMMSLobeIndex))
            continue;

        result += lobe.lobe_weights[lobe_index];
    }

    return result;
}
} // namespace Metallic.OpenPBR

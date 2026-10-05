#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_coating_lobe.h. Modified by Metallic, 2026.
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

// Specify the type of the lobe underneath the coating.

// Specify the type of lobe representing reflection from the coating.

// Instantiate the generic minimal microfacet lobe template using the above configuration.

// Undefine macros to avoid potential conflicts.

/////////////////
// CoatingLobe //
/////////////////

// Represents a coating layer and whatever is beneath it (the aggregate base lobe in this case).
// Handles energy balance between these layers in a reciprocal way.
// When "inside" is true, represents a coated surface hit from the inside;
// applies a tint but doesn't do much else.
public struct OpenPBRCoatingLobe
{
    // Common properties.
    public OpenPBRAggregateLobe base_lobe;
    public float3 normal_ff;
    public float3 tint;
    public float presence;
    public bool inside;

    // Properties used only when entering the material (not when exiting).
    public OpenPBRDielectricMicrofacetLobe coat_reflection_lobe;
    public float in_reflected;
    public float3 in_base_layer_scale;
};

// Define how light should be tinted as it passes through the coat in one direction.
// Light that enters the coat, reflects from the base layer, and exits the coat will
// be tinted by this function two times, once for each passage.

public float3 openPBRCoatPassageColorMultiplier(const OpenPBRCoatingLobe lobe, const float cosine)
{
    if (cosine > 0.0f && openPBRMin3(lobe.tint) < 1.0f) {
        // The user-specified coat color is defined to be the square of the coat transmission color
        // at normal incidence, so taking the square root of the tint produces the coat
        // transmission color at normal incidence (compensated for the presence of the coat).
        const float3 coat_transmission_at_normal_incidence = sqrt(lobe.tint);
        const float refracted_cosine = sqrt(
            max(0.0f,
                1.0f - (1.0f - openPBRSquare(cosine)) /
                           openPBRSquare(lobe.coat_reflection_lobe.refl_trans_coeff.eta_t_over_eta_i))); // Snell's law
        const float distance_scale =
            1.0f / refracted_cosine; // passage distance relative to coat thickness in normal direction
        const float3 coat_transmission_along_refracted_ray =
            openPBRSafePow(coat_transmission_at_normal_incidence, float3(distance_scale));
        return lerp(float3(1.0f), coat_transmission_along_refracted_ray,
                    lobe.presence); // account for fractional coat coverage
    } else                          // inside or no tint
    {
        return float3(1.0f);
    }
}

public float openPBRProportionReflected<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe,
                                                             const float cosine)
{
    if (cosine > 0.0f) {
        const float proportion_reflected_from_full_coat = openPBRComputeDielectricEnergyReflected(
            context, lobe.coat_reflection_lobe.refl_trans_coeff.eta_t_over_eta_i,
            openPBRGetIsotropicAlpha(lobe.coat_reflection_lobe.microfacet_distr), cosine);
        return proportion_reflected_from_full_coat * lobe.presence;
    } else // inside
    {
        return 0.0f;
    }
}

public float3 openPBRBaseLayerScaleOneSide(const OpenPBRCoatingLobe lobe, const float cosine,
                                    const float proportion_reflected_from_this_side)
{
    return openPBRCoatPassageColorMultiplier(lobe, cosine) * (1.0f - proportion_reflected_from_this_side);
}

public float3 openPBRBaseLayerScaleIncoming(const OpenPBRCoatingLobe lobe) { return lobe.in_base_layer_scale; }

public float3 openPBRBaseLayerScaleOutgoing<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe,
                                                                 const float3 light_direction)
{
    const float odotn = dot(light_direction, lobe.normal_ff);
    const float out_reflected = openPBRProportionReflected(context, lobe, odotn);
    return openPBRBaseLayerScaleOneSide(lobe, odotn, out_reflected);
}

public float3 openPBRBaseLayerScaleComplete<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe,
                                                                 const float3 light_direction)
{
    const float3 incoming = lobe.in_base_layer_scale;
    const float3 outgoing = openPBRBaseLayerScaleOutgoing(context, lobe, light_direction);
    return incoming * outgoing;
}

// Assumes that this is an external hit.

public float openPBRCoatReflectionProbability<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe,
                                                                   const float3 view_direction)
{
    // Here we calculate the expected contribution of the coat reflection lobe relative to the overall lobe
    // and we use that to set the probability of sampling the coat reflection lobe.

    // TODO: Take the real path throughput into account (for both standalone lobes),
    //       either by saving it in the lobe struct or by passing it in.
    const float3 PlaceholderPathThroughput = float3(1.0f);

    // Here we can use lobe.in_reflected instead of calling estimate_lobe_contribution
    // because we know the inner details of the coat reflection lobe.
    const float coat_reflection_lobe_contribution = lobe.in_reflected;

    const float unscaled_base_lobe_contribution =
        openPBREstimateLobeContribution(context, lobe.base_lobe, view_direction, PlaceholderPathThroughput);
    const float base_lobe_scale = openPBRMax3(lobe.in_base_layer_scale);
    const float scaled_base_lobe_contribution = unscaled_base_lobe_contribution * base_lobe_scale;

    return openPBRSafeDivide(coat_reflection_lobe_contribution,
                             coat_reflection_lobe_contribution + scaled_base_lobe_contribution, 0.5f);
}

public OpenPBRDiffuseSpecular openPBRCombineCoatAndBaseEvals<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRCoatingLobe lobe, const OpenPBRDiffuseSpecular eval_coat,
    const OpenPBRDiffuseSpecular eval_base, const float3 light_direction)
{
    return openPBRAddDiffuseSpecular(
        openPBRScaleDiffuseSpecular(eval_coat, lobe.presence),
        openPBRScaleDiffuseSpecular(eval_base, openPBRBaseLayerScaleComplete(context, lobe, light_direction)));
}

// Assumes that child lobe has already been initialized.

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(
    TContext context, inout OpenPBRCoatingLobe lobe, const float3 normal_ff, const float3 view_direction,
    const float3 tint, const float presence,
    const float3 extra_base_layer_scale, // such as darkening due to internal reflections
    const bool inside)
{
    lobe.normal_ff = normal_ff;
    lobe.tint = tint;
    lobe.presence = presence;
    lobe.inside = inside;

    // Precompute certain properties. The order of these calls is important because
    // they use information in the lobe struct that is set by previous calls.
    const float idotn = dot(view_direction, lobe.normal_ff);
    lobe.in_reflected = openPBRProportionReflected(context, lobe, idotn);
    lobe.in_base_layer_scale =
        openPBRBaseLayerScaleOneSide(lobe, idotn, lobe.in_reflected) *
        extra_base_layer_scale; // (We can incorporate the entire extra scale here because it's currently not view-dependent.
                                //  This lets us avoid storing it and allows it to be applied to emission as well.)
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe)
{
    if (lobe.inside) {
        return openPBRGetLobeType(context, lobe.base_lobe);
    } else {
        return openPBRGetLobeType(context, lobe.coat_reflection_lobe) | openPBRGetLobeType(context, lobe.base_lobe);
    }
}

public OpenPBRDiffuseSpecular openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                                             const OpenPBRCoatingLobe lobe,
                                                                             const float3 view_direction,
                                                                             const float3 light_direction)
{
    if (lobe.inside) {
        const float odotn = dot(light_direction, lobe.normal_ff);
        const float3 base_layer_scale =
            openPBRCoatPassageColorMultiplier(lobe, -odotn); // negative to tint only transmission

        return openPBRScaleDiffuseSpecular(
            openPBRCalculateLobeValue(context, lobe.base_lobe, view_direction, light_direction), base_layer_scale);
    } else {
        return openPBRCombineCoatAndBaseEvals(
            context, lobe,
            openPBRCalculateLobeValue(context, lobe.coat_reflection_lobe, view_direction, light_direction),
            openPBRCalculateLobeValue(context, lobe.base_lobe, view_direction, light_direction), light_direction);
    }
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    if (lobe.inside) {
        return openPBRCalculateLobePdf(context, lobe.base_lobe, view_direction, light_direction);
    } else {
        return openPBRCombineCoatAndBasePdfs(
            openPBRCalculateLobePdf(context, lobe.coat_reflection_lobe, view_direction, light_direction),
            openPBRCalculateLobePdf(context, lobe.base_lobe, view_direction, light_direction),
            openPBRCoatReflectionProbability(context, lobe, view_direction));
    }
}

// Given a view direction, sample a light direction,
// evaluate the throughput and pdf for this direction,
// and return whether a sample was successfully generated.
// Output arguments are only set when the function returns true.

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe, const float3 rand,
                                                   const float3 view_direction, out float3 light_direction,
                                                   out OpenPBRDiffuseSpecular weight, out float pdf,
                                                   out uint sampled_type)
{
    if (lobe.inside) {
        const bool success = openPBRSampleLobe(context, lobe.base_lobe, rand, view_direction, light_direction, weight,
                                               pdf, sampled_type);
        if (success) {
            const float odotn = dot(light_direction, lobe.normal_ff);
            const float3 base_layer_scale =
                openPBRCoatPassageColorMultiplier(lobe, -odotn); // negative to tint only transmission

            weight = openPBRScaleDiffuseSpecular(weight, base_layer_scale);
        }

        return success;
    } else {
        bool success;

        const float coat_reflection_prob = openPBRCoatReflectionProbability(context, lobe, view_direction);

        float rand_x = rand.x;
        if (rand_x < coat_reflection_prob) // coat
        {
            const float inverse_coat_reflection_prob = 1.0f / coat_reflection_prob;
            rand_x *= inverse_coat_reflection_prob;
            openPBRClampRemappedRandomNumber(rand_x);

            success = openPBRSampleLobe(context, lobe.coat_reflection_lobe, float3(rand_x, rand.y, rand.z),
                                        view_direction, light_direction, weight, pdf, sampled_type);
            if (success) {
                if (!bool(sampled_type & OpenPBRBSDFLobeTypeSpecular)) // non-delta
                {
                    OpenPBRDiffuseSpecular bsdf_cos = openPBRScaleDiffuseSpecular(weight, pdf);

                    bsdf_cos = openPBRCombineCoatAndBaseEvals(
                        context, lobe, bsdf_cos,
                        openPBRCalculateLobeValue(context, lobe.base_lobe, view_direction, light_direction),
                        light_direction);
                    pdf = openPBRCombineCoatAndBasePdfs(
                        pdf, openPBRCalculateLobePdf(context, lobe.base_lobe, view_direction, light_direction),
                        coat_reflection_prob);

                    weight = openPBRScaleDiffuseSpecular(bsdf_cos, 1.0f / pdf);
                } else // delta
                {
                    weight = openPBRScaleDiffuseSpecular(weight, inverse_coat_reflection_prob);
                }
            }
        } else // base
        {
            const float base_prob = 1.0f - coat_reflection_prob;
            const float inverse_base_prob = 1.0f / base_prob;
            rand_x = (rand_x - coat_reflection_prob) * inverse_base_prob;
            openPBRClampRemappedRandomNumber(rand_x);

            success = openPBRSampleLobe(context, lobe.base_lobe, float3(rand_x, rand.y, rand.z), view_direction,
                                        light_direction, weight, pdf, sampled_type);
            if (success) {
                if (!bool(sampled_type & OpenPBRBSDFLobeTypeSpecular)) // non-delta
                {
                    OpenPBRDiffuseSpecular bsdf_cos = openPBRScaleDiffuseSpecular(weight, pdf);

                    bsdf_cos = openPBRCombineCoatAndBaseEvals(
                        context, lobe,
                        openPBRCalculateLobeValue(context, lobe.coat_reflection_lobe, view_direction, light_direction),
                        bsdf_cos, light_direction);
                    pdf = openPBRCombineCoatAndBasePdfs(
                        openPBRCalculateLobePdf(context, lobe.coat_reflection_lobe, view_direction, light_direction),
                        pdf, coat_reflection_prob);

                    weight = openPBRScaleDiffuseSpecular(bsdf_cos, 1.0f / pdf);
                } else // delta
                {
                    const float3 remaining_scale_factors =
                        openPBRBaseLayerScaleComplete(context, lobe, light_direction) * inverse_base_prob;
                    weight = openPBRScaleDiffuseSpecular(weight, remaining_scale_factors);
                }
            }
        }

        return success;
    }
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context, const OpenPBRCoatingLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    const float unscaled_base_lobe_contribution =
        openPBREstimateLobeContribution(context, lobe.base_lobe, view_direction, path_throughput);

    if (lobe.inside) {
        return unscaled_base_lobe_contribution;
    } else {
        // Here we can use lobe.in_reflected instead of calling estimate_lobe_contribution
        // because we know the inner details of the coat reflection lobe.
        const float coat_reflection_lobe_contribution = lobe.in_reflected;

        const float base_lobe_scale =
            openPBRMaxComponentOfThroughputWeightedColor(path_throughput, lobe.in_base_layer_scale);
        const float scaled_base_lobe_contribution = unscaled_base_lobe_contribution * base_lobe_scale;

        return coat_reflection_lobe_contribution + scaled_base_lobe_contribution;
    }
}
} // namespace Metallic.OpenPBR

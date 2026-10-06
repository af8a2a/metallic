#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_generic_minimal_microfacet_lobe.h. Modified by Metallic, 2026.
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

// Concrete Slang lobe implementations; no macro instantiation.

/////////////////////////////////////
// MinimalMicrofacetReflectionLobe //
/////////////////////////////////////

public struct OpenPBRConstantMicrofacetLobe
{
    public float3 normal_ff;
    public OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr;
    public OpenPBRConstantReflectionCoefficient refl_trans_coeff;
};

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(
    TContext context, inout OpenPBRConstantMicrofacetLobe lobe, const float3 normal_ff,
    const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr,
    const OpenPBRConstantReflectionCoefficient refl_trans_coeff)
{
    lobe = OpenPBRConstantMicrofacetLobe(normal_ff, microfacet_distr, refl_trans_coeff);
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context, const OpenPBRConstantMicrofacetLobe lobe)
{
    return OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection;
}

public OpenPBRDiffuseSpecular openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                                             const OpenPBRConstantMicrofacetLobe lobe,
                                                                             const float3 view_direction,
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
    } else // not reflection
    {
        return openPBRMakeZeroDiffuseSpecular();
    }
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context, const OpenPBRConstantMicrofacetLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
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

        const float D_other_terms = D * other_terms;

        const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);

        return G_masking * D_other_terms;
    } else // not reflection
    {
        return 0.0f;
    }
}

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRConstantMicrofacetLobe lobe,
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

    // The light direction is normalized by construction, but can be significantly off due to rounding error.
    light_direction = openPBRFastNormalize(-view_direction + half_vector * 2.0f * idoth);

    // Check if reflection is on correct side of surface.
    const float odotn = dot(lobe.normal_ff, light_direction);
    if (idotn * odotn <= 0.0f) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    const float odoth = dot(light_direction, half_vector);

    const float3 F_refl = openPBRReflectionCoefficient(context, lobe.refl_trans_coeff, idoth);
    const float G_shadowing = openPBREvalSmithG1(lobe.microfacet_distr, light_direction, odotn);
    weight = openPBRMakeDiffuseSpecularFromSpecular(F_refl * G_shadowing);

    const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
    if (D <= 0.0f) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);
    pdf = D * G_masking * idoth / (4.0f * odoth * idotn);

    // It's very important to return the correct type to ensure correct treatment by outside systems.
    sampled_type = openPBRGetLobeType(context, lobe);

    return true;
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context,
                                                                  const OpenPBRConstantMicrofacetLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    const float idotn = dot(view_direction, lobe.normal_ff);
    // Here we assume that the microfacet distribution is a GGX one and calculate the estimated weight accordingly.
    return openPBREstimateWeightWhenAppliedToGgxMicrofacetDistribution(
        context, lobe.refl_trans_coeff, path_throughput, idotn, openPBRGetIsotropicAlpha(lobe.microfacet_distr));
}
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

// Concrete Slang lobe implementations; no macro instantiation.

/////////////////////////////////////
// MinimalMicrofacetReflectionLobe //
/////////////////////////////////////

public struct OpenPBRDielectricMicrofacetLobe
{
    public float3 normal_ff;
    public OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr;
    public OpenPBRIorReflectionCoefficient refl_trans_coeff;
};

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(
    TContext context, inout OpenPBRDielectricMicrofacetLobe lobe, const float3 normal_ff,
    const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr,
    const OpenPBRIorReflectionCoefficient refl_trans_coeff)
{
    lobe = OpenPBRDielectricMicrofacetLobe(normal_ff, microfacet_distr, refl_trans_coeff);
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context, const OpenPBRDielectricMicrofacetLobe lobe)
{
    return OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection;
}

public OpenPBRDiffuseSpecular openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                                             const OpenPBRDielectricMicrofacetLobe lobe,
                                                                             const float3 view_direction,
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
    } else // not reflection
    {
        return openPBRMakeZeroDiffuseSpecular();
    }
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context, const OpenPBRDielectricMicrofacetLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
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

        const float D_other_terms = D * other_terms;

        const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);

        return G_masking * D_other_terms;
    } else // not reflection
    {
        return 0.0f;
    }
}

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRDielectricMicrofacetLobe lobe,
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

    // The light direction is normalized by construction, but can be significantly off due to rounding error.
    light_direction = openPBRFastNormalize(-view_direction + half_vector * 2.0f * idoth);

    // Check if reflection is on correct side of surface.
    const float odotn = dot(lobe.normal_ff, light_direction);
    if (idotn * odotn <= 0.0f) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    const float odoth = dot(light_direction, half_vector);

    const float3 F_refl = openPBRReflectionCoefficient(context, lobe.refl_trans_coeff, idoth);
    const float G_shadowing = openPBREvalSmithG1(lobe.microfacet_distr, light_direction, odotn);
    weight = openPBRMakeDiffuseSpecularFromSpecular(F_refl * G_shadowing);

    const float D = openPBREvalGGX(lobe.microfacet_distr, half_vector, lobe.normal_ff);
    if (D <= 0.0f) {
        openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
        return false;
    }

    const float G_masking = openPBREvalSmithG1(lobe.microfacet_distr, view_direction, idotn);
    pdf = D * G_masking * idoth / (4.0f * odoth * idotn);

    // It's very important to return the correct type to ensure correct treatment by outside systems.
    sampled_type = openPBRGetLobeType(context, lobe);

    return true;
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context,
                                                                  const OpenPBRDielectricMicrofacetLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    const float idotn = dot(view_direction, lobe.normal_ff);
    // Here we assume that the microfacet distribution is a GGX one and calculate the estimated weight accordingly.
    return openPBREstimateWeightWhenAppliedToGgxMicrofacetDistribution(
        context, lobe.refl_trans_coeff, path_throughput, idotn, openPBRGetIsotropicAlpha(lobe.microfacet_distr));
}
} // namespace Metallic.OpenPBR

#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_microfacet_multiple_scattering_data.h. Modified by Metallic, 2026.
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

public static const int OpenPBREnergyTableSizeMinusOne = 32 - 1;
public static const float OpenPBRIorMax = 2.5f;
public static const float OpenPBRInverseIorMax = 1.0f / OpenPBRIorMax;

public float openPBRIorToExactIndex(const float ior)
{
    const float IorOne = 1.0f;
    const float IorMaxMinusIorOne = OpenPBRIorMax - IorOne;
    const float InverseIorMaxMinusIorOne = 1.0f / IorMaxMinusIorOne;
    const int HalfTableSize = 32 / 2;
    const int HalfTableSizeMinusOne = HalfTableSize - 1;
    if (ior < IorOne) {
        const float inverse_ior = 1.0f / ior;
        const float fraction = (inverse_ior - IorOne) * InverseIorMaxMinusIorOne;
        return HalfTableSizeMinusOne - fraction * HalfTableSizeMinusOne;
    } else {
        const float fraction = (ior - IorOne) * InverseIorMaxMinusIorOne;
        return HalfTableSize + fraction * HalfTableSizeMinusOne;
    }
}

public float openPBRAlphaToExactIndex(const float alpha)
{
    // NOTE: openpbr_fast_sqrt() is not utilized here as it alters roughness visibly
    const float roughness = sqrt(alpha);
    const float exact_index = roughness * OpenPBREnergyTableSizeMinusOne;
    return exact_index;
}

public float openPBRCosThetaToExactIndex(const float cos_theta)
{
    const float exact_index = cos_theta * OpenPBREnergyTableSizeMinusOne;
    return exact_index;
}

public float openPBRRemapExactIndex(const float exact_index)
{
    const float InverseTableSize = 1.0f / 32;
    const float MinCoordinate = 0.5f * InverseTableSize;
    const float MaxCoordinate = 1.0f - MinCoordinate;
    return clamp(MinCoordinate + exact_index * InverseTableSize, MinCoordinate, MaxCoordinate);
}

public float openPBRClampExactIndex(const float exact_index)
{
    return clamp(exact_index, 0.0f, float(OpenPBREnergyTableSizeMinusOne));
}

public float openPBRExtrapolateTableValueBeyondIorMaxIfNeeded(const float table_value, const float ior)
{
    if (ior > OpenPBRIorMax || ior < OpenPBRInverseIorMax) {
        const float F0Max = openPBRF0FromIor(OpenPBRIorMax);
        const float F0ExtrapolationRange = 1.0f - F0Max;
        const float InverseF0ExtrapolationRange = 1.0f / F0ExtrapolationRange;
        const float f0 = openPBRF0FromIor(ior);
        const float progress_toward_zero = (f0 - F0Max) * InverseF0ExtrapolationRange;
        return (1.0f - progress_toward_zero) * table_value;
    } else {
        return table_value;
    }
}

// Unified energy table lookup helpers
// The statically typed context owns the LUT storage and filtering.

public float openPBRLookUpLinear<TContext : IOpenPBRContext>(TContext context, const int lut_id, const float exact_index_x)
{

    const float remapped_exact_index_x = openPBRRemapExactIndex(exact_index_x);
    return context.sample2D(lut_id, float2(remapped_exact_index_x, 0.5f))
        .r; // TODO: Use a 1D texture sampler when available.
}

public float openPBRLookUpBilinear<TContext : IOpenPBRContext>(TContext context, const int lut_id, const float exact_index_x,
                                                        const float exact_index_y)
{

    const float remapped_exact_index_x = openPBRRemapExactIndex(exact_index_x);
    const float remapped_exact_index_y = openPBRRemapExactIndex(exact_index_y);
    return context.sample2D(lut_id, float2(remapped_exact_index_y, remapped_exact_index_x)).r;
}

public float openPBRLookUpTrilinear<TContext : IOpenPBRContext>(TContext context, const int lut_id, const float exact_index_x,
                                                         const float exact_index_y, const float exact_index_z)
{

    const float remapped_exact_index_x = openPBRRemapExactIndex(exact_index_x);
    const float remapped_exact_index_y = openPBRRemapExactIndex(exact_index_y);
    const float remapped_exact_index_z = openPBRRemapExactIndex(exact_index_z);
    return context.sample3D(lut_id, float3(remapped_exact_index_z, remapped_exact_index_y, remapped_exact_index_x)).r;
}

public float openPBRLookUpIdealDielectricEnergyComplement<TContext : IOpenPBRContext>(TContext context, const float ior,
                                                                               const float alpha, const float cos_theta)
{
    const float exact_index_ior = openPBRIorToExactIndex(ior);
    const float exact_index_alpha = openPBRAlphaToExactIndex(alpha);
    const float exact_index_cos_theta = openPBRCosThetaToExactIndex(cos_theta);
    return openPBRLookUpTrilinear(context, 0, exact_index_ior, exact_index_alpha, exact_index_cos_theta);
}

public float openPBRLookUpIdealDielectricAverageEnergyComplement<TContext : IOpenPBRContext>(TContext context, const float ior,
                                                                                      const float alpha)
{
    const float exact_index_ior = openPBRIorToExactIndex(ior);
    const float exact_index_alpha = openPBRAlphaToExactIndex(alpha);
    return openPBRLookUpBilinear(context, 1, exact_index_ior, exact_index_alpha);
}

public float openPBRLookUpIdealDielectricReflectionRatio<TContext : IOpenPBRContext>(TContext context, const float ior,
                                                                              const float alpha)
{
    const float exact_index_ior = openPBRIorToExactIndex(ior);
    const float exact_index_alpha = openPBRAlphaToExactIndex(alpha);
    return openPBRLookUpBilinear(context, 2, exact_index_ior, exact_index_alpha);
}

public float openPBRLookUpOpaqueDielectricEnergyComplement<TContext : IOpenPBRContext>(TContext context, const float ior,
                                                                                const float alpha,
                                                                                const float cos_theta)
{
    const float exact_index_ior = openPBRIorToExactIndex(ior);
    const float exact_index_alpha = openPBRAlphaToExactIndex(alpha);
    const float exact_index_cos_theta = openPBRCosThetaToExactIndex(cos_theta);
    const float table_value =
        openPBRLookUpTrilinear(context, 3, exact_index_ior, exact_index_alpha, exact_index_cos_theta);
    return openPBRExtrapolateTableValueBeyondIorMaxIfNeeded(table_value, ior);
}

public float openPBRLookUpOpaqueDielectricAverageEnergyComplement<TContext : IOpenPBRContext>(TContext context,
                                                                                       const float ior,
                                                                                       const float alpha)
{
    const float exact_index_ior = openPBRIorToExactIndex(ior);
    const float exact_index_alpha = openPBRAlphaToExactIndex(alpha);
    const float table_value = openPBRLookUpBilinear(context, 4, exact_index_ior, exact_index_alpha);
    return openPBRExtrapolateTableValueBeyondIorMaxIfNeeded(table_value, ior);
}

public float openPBRLookUpIdealMetalEnergyComplement<TContext : IOpenPBRContext>(TContext context, const float alpha,
                                                                          const float cos_theta)
{
    const float exact_index_alpha = openPBRAlphaToExactIndex(alpha);
    const float exact_index_cos_theta = openPBRCosThetaToExactIndex(cos_theta);
    return openPBRLookUpBilinear(context, 5, exact_index_alpha, exact_index_cos_theta);
}

public float openPBRLookUpIdealMetalAverageEnergyComplement<TContext : IOpenPBRContext>(TContext context, const float alpha)
{
    const float exact_index_alpha = openPBRAlphaToExactIndex(alpha);
    return openPBRLookUpLinear(context, 6, exact_index_alpha);
}

// This function can be used for the common case of finding the average energy reflected from an opaque rough dielectric.
// It also contains a special case for zero roughness to produce an exact result in that case.
// This function could be relocated somewhere else if it ever needs to be used outside of this file,
// such as in MinimalMicrofacetReflectionLobe's estimate_lobe_contribution function.

public float openPBRComputeDielectricEnergyReflected<TContext : IOpenPBRContext>(TContext context, const float ior,
                                                                          const float alpha, const float cos_theta)
{
    if (alpha == 0.0f)
        return openPBRFresnel(ior, cos_theta);
    else
        return 1.0f - openPBRLookUpOpaqueDielectricEnergyComplement(context, ior, alpha, cos_theta);
}
} // namespace Metallic.OpenPBR

#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_constant_reflection_coefficient.h. Modified by Metallic, 2026.
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

// Below is a simple variation of the reflection/transmission coefficient that only does dielectric reflection based on a constant color.
// It is useful for things like the thin-film specular lobes that are uniformly scaled based on the net reflection and transmission from a window.
// It follows the same pattern as the other reflection/transmission coefficients so it is easily interchangeable with them.

///////////////////////////////////
// ConstantReflectionCoefficient //
///////////////////////////////////

public struct OpenPBRConstantReflectionCoefficient
{
    public float3 color; // the constant color used for reflection
};

public float3 openPBRReflectionCoefficient<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRConstantReflectionCoefficient refl_trans_coeff, const float idoth)
{
    return refl_trans_coeff.color;
}

public float3 openPBRTransmissionCoefficient<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRConstantReflectionCoefficient refl_trans_coeff, const float idoth)
{
    return float3(0.0f);
}

// The default implementations of the reflection and transmission probabilities use both coefficients.
// Specialized implementations could calculate both based on the same shared factors (such as fresnel and 1-fresnel).

public OpenPBRAllCoefficients openPBRAllCoefficients<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRConstantReflectionCoefficient refl_trans_coeff, const float idoth)
{
    return OpenPBRAllCoefficients(refl_trans_coeff.color, float3(0.0f));
}

public float openPBRReflectionProbability<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRConstantReflectionCoefficient refl_trans_coeff, const float3 path_throughput,
    const float idoth)
{
    return 1.0f;
}

public float openPBRTransmissionProbability<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRConstantReflectionCoefficient refl_trans_coeff, const float3 path_throughput,
    const float idoth)
{
    return 0.0f;
}

public OpenPBRAllCoefficientsAndProbabilities openPBRAllCoefficientsAndProbabilities<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRConstantReflectionCoefficient refl_trans_coeff, const float3 path_throughput,
    const float idoth)
{
    return OpenPBRAllCoefficientsAndProbabilities(refl_trans_coeff.color, float3(0.0f), 1.0f, 0.0f);
}

public float openPBREstimateWeight<TContext : IOpenPBRContext>(TContext context,
                                                        const OpenPBRConstantReflectionCoefficient refl_trans_coeff,
                                                        const float3 path_throughput, const float idotn)
{
    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput, refl_trans_coeff.color);
}

public float openPBREstimateWeightWhenAppliedToGgxMicrofacetDistribution<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRConstantReflectionCoefficient refl_trans_coeff, const float3 path_throughput,
    const float idotn, const float isotropic_alpha)
{
    // We don't currently have a way to calculate the average albedo of this constant color reflection coefficient
    // applied to a GGX microfacet distribution, so we just estimate the weight assuming a smooth surface.
    return openPBREstimateWeight(context, refl_trans_coeff, path_throughput, idotn);
}
} // namespace Metallic.OpenPBR

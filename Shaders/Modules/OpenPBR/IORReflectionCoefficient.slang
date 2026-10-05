#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_ior_reflection_coefficient.h. Modified by Metallic, 2026.
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

// Below is a simple variation of the reflection/transmission coefficient that only does dielectric reflection based on an IOR.
// It is useful for things like dielectric coating lobes.
// It follows the same pattern as the other reflection/transmission coefficients so it is easily interchangeable with them.

//////////////////////////////
// IorReflectionCoefficient //
//////////////////////////////

public struct OpenPBRIorReflectionCoefficient
{
    public float eta_t_over_eta_i; // the IOR used for reflection
};

public float3 openPBRReflectionCoefficient<TContext : IOpenPBRContext>(TContext context,
                                                                const OpenPBRIorReflectionCoefficient refl_trans_coeff,
                                                                const float idoth)
{
    return float3(openPBRFresnel(refl_trans_coeff.eta_t_over_eta_i, idoth));
}

public float3 openPBRTransmissionCoefficient<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRIorReflectionCoefficient refl_trans_coeff, const float idoth)
{
    return float3(0.0f);
}

// The default implementations of the reflection and transmission probabilities use both coefficients.
// Specialized implementations could calculate both based on the same shared factors (such as fresnel and 1-fresnel).

public OpenPBRAllCoefficients openPBRAllCoefficients<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRIorReflectionCoefficient refl_trans_coeff, const float idoth)
{
    return OpenPBRAllCoefficients(openPBRReflectionCoefficient(context, refl_trans_coeff, idoth), float3(0.0f));
}

public float openPBRReflectionProbability<TContext : IOpenPBRContext>(TContext context,
                                                               const OpenPBRIorReflectionCoefficient refl_trans_coeff,
                                                               const float3 path_throughput, const float idoth)
{
    return 1.0f;
}

public float openPBRTransmissionProbability<TContext : IOpenPBRContext>(TContext context,
                                                                 const OpenPBRIorReflectionCoefficient refl_trans_coeff,
                                                                 const float3 path_throughput, const float idoth)
{
    return 0.0f;
}

public OpenPBRAllCoefficientsAndProbabilities openPBRAllCoefficientsAndProbabilities<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRIorReflectionCoefficient refl_trans_coeff, const float3 path_throughput,
    const float idoth)
{
    return OpenPBRAllCoefficientsAndProbabilities(openPBRReflectionCoefficient(context, refl_trans_coeff, idoth),
                                                  float3(0.0f), 1.0f, 0.0f);
}

public float openPBREstimateWeight<TContext : IOpenPBRContext>(TContext context,
                                                        const OpenPBRIorReflectionCoefficient refl_trans_coeff,
                                                        const float3 path_throughput, const float idotn)
{
    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput,
                                                        openPBRReflectionCoefficient(context, refl_trans_coeff, idotn));
}

public float openPBREstimateWeightWhenAppliedToGgxMicrofacetDistribution<TContext : IOpenPBRContext>(
    TContext context, const OpenPBRIorReflectionCoefficient refl_trans_coeff, const float3 path_throughput,
    const float idotn, const float isotropic_alpha)
{
    // In this case we are given the alpha of the GGX microfacet distribution being used with our dielectric Fresnel reflection coefficient,
    // so we can look up the energy reflected (the complement of the energy not reflected) directly.
    // If we ever support delta (zero roughness) reflection, we could update this to
    // use the CoatingLobe's compute_rough_dielectric_energy_reflected function.
    const float average_reflectivity = 1.0f - openPBRLookUpOpaqueDielectricEnergyComplement(
                                                  context, refl_trans_coeff.eta_t_over_eta_i, isotropic_alpha, idotn);
    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput, float3(average_reflectivity));
}
} // namespace Metallic.OpenPBR

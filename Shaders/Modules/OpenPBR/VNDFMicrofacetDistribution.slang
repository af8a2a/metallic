#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_vndf_microfacet_distribution.h. Modified by Metallic, 2026.
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

// The classes here abstract away all of the data, function calls, and coordinate-system transformations
// necessary to sample and evaluate microfacet distributions with GGX+Smith visible-normal sampling.
// The client microfacet lobes need not know whether the distribution is isotropic or anisotropic.
// These classes have no virtual functions; duck typing can be used to avoid any abstraction cost.

///////////////////////////////////////////////////
// AnisotropicGGXSmithVNDFMicrofacetDistribution //
///////////////////////////////////////////////////

public struct OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution
{
    public float2 alpha;
    public OpenPBRBasis basis_ff;

    // Can be used by the client lobes to get the original or standard isotropic alpha value,
    // which is needed for auxiliary operations like creating microfacet-multiple-scattering lobes
    // and estimating lobe contributions.
    public float isotropic_alpha;
};

public float3 openPBRSampleGgxSmithVndf(const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr,
                                 const float3 view_direction, const float3 normal_ff, const float2 rand)
{
    // Choose microfacet normal (in local space).
    const float3 half_vector_local = openPBRSampleAnisoGgxSmithVndf(
        microfacet_distr.alpha, openPBRWorldToLocal(microfacet_distr.basis_ff, view_direction), rand);

    // Map half vector to world.
    return openPBRLocalToWorld(microfacet_distr.basis_ff, half_vector_local);
}

public float openPBREvalGGX(const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr,
                     const float3 half_vector, const float3 normal_ff)
{
    return openPBREvalAnisoGgx(openPBRWorldToLocal(microfacet_distr.basis_ff, half_vector), microfacet_distr.alpha);
}

public float openPBREvalSmithG1(const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr,
                         const float3 view_direction_or_light_direction, float iodotn)
{
    return openPBREvalAnisoSmithG1(openPBRWorldToLocal(microfacet_distr.basis_ff, view_direction_or_light_direction),
                                   microfacet_distr.alpha);
}

public float openPBREvalSmithG2(const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr,
                         const float3 view_direction, const float3 light_direction, float idotn, float odotn)
{
    return openPBREvalAnisoSmithG2(openPBRWorldToLocal(microfacet_distr.basis_ff, view_direction),
                                   openPBRWorldToLocal(microfacet_distr.basis_ff, light_direction),
                                   microfacet_distr.alpha);
}

public float openPBRGetIsotropicAlpha(const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr)
{
    return microfacet_distr.isotropic_alpha;
}
} // namespace Metallic.OpenPBR

#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_thin_wall_specular_transmission_lobe.h. Modified by Metallic, 2026.
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

// Specify the type of lobe that is used for the thin-wall specular reflection and transmission.

//////////////////////////////////////
// ThinWallSpecularTransmissionLobe //
//////////////////////////////////////

// A thin-wall specular lobe that only handles transmission via a flipped minimal microfacet lobe.
public struct OpenPBRThinWallSpecularTransmissionLobe
{
    public OpenPBRConstantMicrofacetLobe flipped_lobe;
};

// Initialize the inner specular lobe with flipped normal, flipped microfacet-distribution basis, and reflected view direction.

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(
    TContext context, inout OpenPBRThinWallSpecularTransmissionLobe lobe, const float3 normal_ff,
    const OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution microfacet_distr, const float3 trans_color)
{
    const float3 flipped_normal = -normal_ff;
    OpenPBRAnisotropicGGXSmithVNDFMicrofacetDistribution flipped_microfacet_distr = microfacet_distr;
    flipped_microfacet_distr.basis_ff.n *= -1.0f; // only necessary to flip the normal
    openPBRInitializeLobe(context, lobe.flipped_lobe, flipped_normal, flipped_microfacet_distr,
                          OpenPBRConstantReflectionCoefficient(trans_color));
}
} // namespace Metallic.OpenPBR

#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_thin_wall_diffuse_transmission_lobe.h. Modified by Metallic, 2026.
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

/////////////////////////////////////
// ThinWallDiffuseTransmissionLobe //
/////////////////////////////////////

// This lobe wraps an existing EnergyConservingRoughDiffuseLobe
// but flips it to the opposite hemisphere to represent diffuse transmission.
public struct OpenPBRThinWallDiffuseTransmissionLobe
{
    public OpenPBREnergyConservingRoughDiffuseLobe flipped_lobe;
};

// Initialize the inner diffuse lobe with flipped normal and reflected view direction.

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(TContext context,
                                                       inout OpenPBRThinWallDiffuseTransmissionLobe lobe,
                                                       const float3 normal_ff, const float3 view_direction,
                                                       const float3 diffuse_albedo, const float diffuse_roughness,
                                                       const float specular_alpha,
                                                       const float specular_eta_t_over_eta_i)
{
    const float3 flipped_normal = -normal_ff;
    float3 reflected_view =
        reflect(view_direction, normal_ff); // reflects view across surface (view points away from surface)

    const float dot_reflected_flipped = dot(reflected_view, flipped_normal);
    if (dot_reflected_flipped < 0.0f) {

        // In the rare case where the reflected view is on the wrong side of the surface, we force it to be above the surface.
        // This can happen due to numerical precision issues when the view direction is nearly parallel to the surface
        // (rounding error is introduced by the reflect() function above, so the view can get out of sync with the normal).
        reflected_view *= -1.0f;
    }

    openPBRInitializeLobe(context, lobe.flipped_lobe, flipped_normal, reflected_view, diffuse_albedo, diffuse_roughness,
                          specular_alpha, specular_eta_t_over_eta_i);
}
} // namespace Metallic.OpenPBR

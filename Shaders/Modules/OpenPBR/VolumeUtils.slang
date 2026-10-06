#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_volume_utils.h. Modified by Metallic, 2026.
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

// This file contains the math necessary to calculate
// volume properties based on OpenPBR volumes parameters.

// Helper functions used only in this file.

// Clamping inputs is necessary to handle edge cases that would result in overflow and NaNs,
// and to avoid extreme values that would result in numerical issues during path tracing.
public static const float OpenPBRMinColor = 1e-3f;
public static const float OpenPBRMinDistance = 1e-3f;
public static const float OpenPBRMaxAbsAnisotropy = 0.999f;

public float3 openPBRClampInputColor(const float3 color) { return max(color, float3(OpenPBRMinColor)); }

public float3 openPBRClampInputDistance(const float3 distance) { return max(distance, float3(OpenPBRMinDistance)); }

public float openPBRClampInputDistance(const float distance) { return max(distance, OpenPBRMinDistance); }

public float openPBRClampInputAnisotropy(const float anisotropy)
{
    return clamp(anisotropy, -OpenPBRMaxAbsAnisotropy, OpenPBRMaxAbsAnisotropy);
}

// Functions used by the OpenPBR material to calculate subsurface and interior volume properties.

// Convert artist-friendly subsurface parameters to low-level volume parameters.
// TODO(sss): Figure out whether unit conversion is needed!
// TODO(sss): Incorporate IOR, roughness, and translucent shadows into the color mapping!
// TODO(sss): Gradually apply similarity theory in later bounces to improve efficiency!

public OpenPBRHomogeneousVolume openPBRCreateSubsurfaceVolumeFromOpenpbrParams(const float subsurface_radius,
                                                                        const float3 subsurface_radius_scale,
                                                                        const float3 subsurface_color,
                                                                        const float subsurface_scatter_anisotropy)
{
    // Calculate the extinction coefficient from the subsurface radius and radius scale.
    // Per the OpenPBR spec:
    //     "Unit: where relevant, otherwise assumed to be dimensionless.
    //      Note that lengths (other than thin_film_thickness) can be assumed to be in the world space units of the scene."

    const float3 extinction_mean_free_path = openPBRClampInputDistance(subsurface_radius * subsurface_radius_scale);

    const float3 extinction_coefficient = float3(1.0f) / extinction_mean_free_path;

    // Calculate the single-scattering albedo from the multiple-scattering albedo.
    // This is using the van de Hulst formulas mentioned in the OpenPBR spec.
    // As per the OpenPBR spec, this albedo mapping compensates for the scattering anisotropy, but the extinction mapping does not.
    // TODO(sss): Is the van de Hulst mapping as good as the Hyperion one we were previously using?
    // TODO(sss): Derive an Eclair-specific mapping or at least a general mapping taking into account the IOR and roughness.

    const float clamped_subsurface_anisotropy = openPBRClampInputAnisotropy(subsurface_scatter_anisotropy);

    // This is the van de Hulst mapping:
    const float3 s = float3(4.09712f) + float3(4.20863f) * subsurface_color -
                     sqrt(float3(9.59217f) + float3(41.6808f) * subsurface_color +
                          float3(17.7126f) * openPBRSquare(subsurface_color));
    const float3 single_scattering_albedo =
        (float3(1.0f) - openPBRSquare(s)) / (float3(1.0f) - clamped_subsurface_anisotropy * openPBRSquare(s));

    // This is the Hyperion mapping:
    //         const vec3 A = subsurface_color;
    //         const vec3 AA = A * A;
    //         const vec3 AAA = AA * A;
    //         const vec3 AAAA = AA * AA;
    //         const vec3 single_scattering_albedo = -expm1(-11.43f * A + 15.38f * AA - 13.91f * AAA);
    // TODO(sss): The anisotropy would need to be taken into account if this were to be used.

    // Rounding error occurs in the mapping above; clamp to ensure valid albedo.
    const float3 clamped_single_scattering_albedo = clamp(single_scattering_albedo, float3(0.0f), float3(1.0f));

    // Initialize and return the final homogeneous volume.
    return openPBRMakeVolumeFromExtinctionCoefficientAndAlbedoAndAnisotropy(
        extinction_coefficient, clamped_single_scattering_albedo, clamped_subsurface_anisotropy);
}

// Calculate, combine, and return "absorption" and "scattering" volumes.
// TODO(sss): Figure out whether unit conversion is needed!

public OpenPBRHomogeneousVolume openPBRCreateVolumeFromOpenpbrParams(const float3 transmission_color,
                                                              const float transmission_depth,
                                                              const float3 transmission_scatter,
                                                              const float transmission_anisotropy)
{

    // Early-out if extinction would be zero or invalid.
    if (openPBRMin3(transmission_color) >= 1.0f && openPBRMax3(transmission_scatter) <= 0.0f)
        return openPBRMakeEmptyVolume();

    const float3 clamped_transmission_color = openPBRClampInputColor(transmission_color);
    const float clamped_transmission_depth = openPBRClampInputDistance(transmission_depth);

    const float reciprocal_clamped_transmission_depth = 1.0f / clamped_transmission_depth;

    const float3 extinction_coefficient = -log(clamped_transmission_color) * reciprocal_clamped_transmission_depth;

    const float3 scattering_coefficient = transmission_scatter * reciprocal_clamped_transmission_depth;

    float3 absorption_coefficient = extinction_coefficient - scattering_coefficient;

    // As per the OpenPBR spec:
    //     "If any component of mu_a is negative, then mu_a is shifted by enough gray to make all the components positive"
    if (openPBRMin3(absorption_coefficient) < 0.0f)
        absorption_coefficient -= openPBRMin3(absorption_coefficient);

    return openPBRMakeVolumeFromAbsorptionAndScatteringCoefficientsAndAnisotropy(
        absorption_coefficient, scattering_coefficient, openPBRClampInputAnisotropy(transmission_anisotropy));
}
} // namespace Metallic.OpenPBR

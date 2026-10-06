#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_resolved_inputs.h. Modified by Metallic, 2026.
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

// Fully-resolved, texture-free OpenPBR inputs matching the official
// OpenPBR parameter set with a few modifications (noted in comments).
// Subsystems may only fill/read the subset they need.
public struct OpenPBRResolvedInputs
{
    // Base layer
    public float base_weight;
    public float3 base_color;
    public float base_diffuse_roughness;
    public float base_metalness;

    // Subsurface
    public float subsurface_weight;
    public float3 subsurface_color;
    public float subsurface_radius;
    public float3 subsurface_radius_scale;
    public float subsurface_scatter_anisotropy;

    // Specular
    public float specular_weight;
    public float3 specular_color;
    public float specular_roughness;
    public float specular_roughness_anisotropy;
    public float specular_ior;
    // Optional extension (Eclair): cosine and sine of the anisotropy rotation angle; set to (1, 0) if unused.
    public float2 specular_anisotropy_rotation_cos_sin;

    // Coat
    public float coat_weight;
    public float3 coat_color;
    public float coat_roughness;
    public float coat_roughness_anisotropy;
    public float coat_ior;
    public float coat_darkening;
    // Optional extension (Eclair): cosine and sine of the anisotropy rotation angle; set to (1, 0) if unused.
    public float2 coat_anisotropy_rotation_cos_sin;

    // Fuzz
    public float fuzz_weight;
    public float3 fuzz_color;
    public float fuzz_roughness;

    // Transmission / volume
    public float transmission_weight;
    public float3 transmission_color;
    public float transmission_depth;
    public float3 transmission_scatter;
    public float transmission_scatter_anisotropy;
    public float transmission_dispersion_scale;
    public float transmission_dispersion_abbe_number;

    // Thin-film
    public float thin_film_weight;
    public float thin_film_thickness;
    public float thin_film_ior;

    // Emission
    public float emission_luminance;
    public float3 emission_color;

    // Geometry
    public float geometry_opacity;
    public bool geometry_thin_walled;

    // Precomputed geometry bases (must be provided). geometry_basis replaces
    // geometry_normal/geometry_tangent, and geometry_coat_basis replaces
    // geometry_coat_normal/geometry_coat_tangent. Caller must compute bitangents
    // and handedness correctly.
    public OpenPBRBasis geometry_basis;
    public OpenPBRBasis geometry_coat_basis;
};

// Returns a fully initialized set of resolved inputs using the OpenPBR specification defaults.

public OpenPBRResolvedInputs openPBRMakeDefaultResolvedInputs()
{
    OpenPBRResolvedInputs inputs;

    // Base layer
    inputs.base_weight = 1.0f;
    inputs.base_color = float3(0.8f);
    inputs.base_diffuse_roughness = 0.0f;
    inputs.base_metalness = 0.0f;

    // Subsurface
    inputs.subsurface_weight = 0.0f;
    inputs.subsurface_color = float3(0.8f);
    inputs.subsurface_radius = 1.0f;
    inputs.subsurface_radius_scale = float3(1.0f, 0.5f, 0.25f);
    inputs.subsurface_scatter_anisotropy = 0.0f;

    // Specular
    inputs.specular_weight = 1.0f;
    inputs.specular_color = float3(1.0f);
    inputs.specular_roughness = 0.3f;
    inputs.specular_roughness_anisotropy = 0.0f;
    inputs.specular_ior = 1.5f;
    inputs.specular_anisotropy_rotation_cos_sin = float2(1.0f, 0.0f);

    // Coat
    inputs.coat_weight = 0.0f;
    inputs.coat_color = float3(1.0f);
    inputs.coat_roughness = 0.0f;
    inputs.coat_roughness_anisotropy = 0.0f;
    inputs.coat_ior = 1.6f;
    inputs.coat_darkening = 1.0f;
    inputs.coat_anisotropy_rotation_cos_sin = float2(1.0f, 0.0f);

    // Fuzz
    inputs.fuzz_weight = 0.0f;
    inputs.fuzz_color = float3(1.0f);
    inputs.fuzz_roughness = 0.5f;

    // Transmission / volume
    inputs.transmission_weight = 0.0f;
    inputs.transmission_color = float3(1.0f);
    inputs.transmission_depth = 0.0f;
    inputs.transmission_scatter = float3(0.0f);
    inputs.transmission_scatter_anisotropy = 0.0f;
    inputs.transmission_dispersion_scale = 0.0f;
    inputs.transmission_dispersion_abbe_number = 20.0f;

    // Thin-film
    inputs.thin_film_weight = 0.0f;
    inputs.thin_film_thickness = 0.5f;
    inputs.thin_film_ior = 1.4f;

    // Emission
    inputs.emission_luminance = 0.0f;
    inputs.emission_color = float3(1.0f);

    // Geometry
    inputs.geometry_opacity = 1.0f;
    inputs.geometry_thin_walled = false;

    // Bases
    inputs.geometry_basis = OpenPBRBasis(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
    inputs.geometry_coat_basis = inputs.geometry_basis;

    return inputs;
}
} // namespace Metallic.OpenPBR

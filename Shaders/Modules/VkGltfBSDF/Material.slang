#language slang 2026
implementing VkGltfBSDF;

/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Modified for Metallic: native module ownership, explicit visibility and Slang types.
// Mathematical expressions are retained from the source snapshot; see README.md.

namespace Metallic.VkGltf
{
/*
Evaluated PBR material containing all properties needed for physically based shading and sampling.
This is the result of evaluating a glTF material with its textures and parameters, ready to be used
by the rendering/ray tracing system for accurate material response simulation.
*/

public struct PreparedMaterial
{
    public float3 baseColor;  // base color
    public float  opacity;    // 1 = opaque, 0 = fully transparent
    public float2 roughness;  // Microfacet alpha = perceptual roughness squared (x = U, y = V).
    public float  metallic;   // 0 = dielectric, 1 = metallic
    public float3 emissive;   // emissive color
    public float  occlusion;  // occlusion factor

    public float3 N;   // shading normal
    public float3 T;   // tangent
    public float3 B;   // bitangent
    public float3 Ng;  // geometric normal


    public float ior1;        // index of refraction : current medium (i.e. air)
    public float ior2;        // index of refraction : the other side (i.e. glass)
    public float dispersion;  // KHR_materials_dispersion

    public float  specular;       // weight of the dielectric specular layer
    public float3 specularColor;  // color of the dielectric specular layer
    public float  transmission;   // KHR_materials_transmission
    // Extra tint on the specular transmission (BTDF) lobe only; the lobe is tinted by
    // `baseColor * transmissionTint`. float3(1) is plain glTF behavior. It exists because the
    // thin-walled mode of KHR_materials_scatter needs the transmission lobe and the diffuse
    // reflection lobe to carry different colors, and `baseColor` alone can only express one.
    public float3 transmissionTint;

    public float3 attenuationColor;     // KHR_materials_volume
    public float  attenuationDistance;  //
    public float  thickness;            // KHR_materials_volume thicknessFactor * thicknessTexture (object-space)

    public float3 scatterCoefficient;   // KHR_materials_scatter: σ_s (scattering coefficient)
    public float  scatterAnisotropy;    // KHR_materials_scatter: g parameter for Henyey-Greenstein

    public float  clearcoat;           // KHR_materials_clearcoat
    public float  clearcoatRoughness;  //
    public float3 Nc;                  // clearcoat normal

    public float iridescence;  // KHR_materials_iridescence
    public float iridescenceIor;
    public float iridescenceThickness;

    public float3 sheenColor;  // KHR_materials_sheen
    public float  sheenRoughness;

    public float  diffuseTransmissionFactor;  // KHR_materials_diffuse_transmission
    public float3 diffuseTransmissionColor;

    public float retroreflection;  // KHR_materials_retroreflection [0,1] MRM blend
};

/*
Default PBR material with all properties set to default values.
This function initializes a PreparedMaterial structure with typical values for a diffuse material.
*/
public PreparedMaterial defaultPreparedMaterial()
{
    PreparedMaterial mat;
    mat.baseColor = float3(1.0F);
    mat.opacity   = 1.0F;
    mat.roughness = float2(1.0F);
    mat.metallic  = 1.0F;
    mat.emissive  = float3(0.0F);

    mat.N  = float3(0.0F, 0.0F, 1.0F);
    mat.Ng = float3(0.0F, 0.0F, 1.0F);
    mat.T  = float3(1.0F, 0.0F, 0.0F);
    mat.B  = float3(0.0F, 1.0F, 0.0F);

    mat.ior1       = 1.0F;
    mat.ior2       = 1.5F;
    mat.dispersion = 0.0F;

    mat.specular         = 1.0F;
    mat.specularColor    = float3(1.0F);
    mat.transmission     = 0.0F;
    mat.transmissionTint = float3(1.0F);

    mat.attenuationColor    = float3(1.0F);
    mat.attenuationDistance = 1.0F;
    mat.thickness           = 0.0F;

    mat.scatterCoefficient = float3(0.0F);
    mat.scatterAnisotropy  = 0.0F;

    mat.clearcoat          = 0.0F;
    mat.clearcoatRoughness = 0.01F;
    mat.Nc                 = float3(0.0F, 0.0F, 1.0F);

    mat.iridescence          = 0.0F;
    mat.iridescenceIor       = 1.5F;
    mat.iridescenceThickness = 0.1F;

    mat.sheenColor     = float3(0.0F);
    mat.sheenRoughness = 0.0F;
    mat.occlusion      = 1.0F;

    mat.diffuseTransmissionFactor = 0.0F;
    mat.diffuseTransmissionColor  = float3(1.0F);

    mat.retroreflection = 0.0F;

    return mat;
}

/*
Default PBR material with all properties set to default values.
This function initializes a PreparedMaterial structure with typical values for a diffuse material.
*/
public PreparedMaterial defaultPreparedMaterial(float3 baseColor, float metallic, float roughness, float3 N, float3 Ng)
{
    PreparedMaterial mat = defaultPreparedMaterial();
    mat.baseColor   = baseColor;
    mat.metallic    = metallic;
    mat.roughness   = float2(roughness * roughness);
    mat.Ng          = Ng;
    mat.N           = N;
    orthonormalBasis(mat.N, mat.T, mat.B);

    return mat;
}
} // namespace Metallic.VkGltf

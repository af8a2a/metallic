#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_homogeneous_volume.h. Modified by Metallic, 2026.
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

// Use a hard-coded magic constant to represent the extinction coefficient of the ambient volume.
// The ambient volume is whatever volume exists in the space outside the objects --
// currently the ambient volume is an empty volume overlayed with heterogeneous volumes.
public static const float OpenPBRAmbientVolumeExtinctionCoefficient = -3.402823466e+38f;
public static const float OpenPBRAmbientVolumeAlbedo =
    0.0f; // can be arbitrary because unused; zero can be compressed without special handling
public static const float OpenPBRAmbientVolumeAnisotropy =
    0.0f; // can be arbitrary because unused; zero can be compressed without special handling

// Struct storing the properties of a homogeneous volume.
// Extinction and albedo are used instead of absorption and scattering
// to reduce the number of calculations necessary in certain functions.
public struct OpenPBRHomogeneousVolume
{
    public float3 extinction_coefficient; // extinction coefficient
    public float3 albedo; // single-scattering albedo
    public float anisotropy; // mean cosine of the scattering angle
};

// Initializes a volume from extinction, albedo, and anisotropy.

public OpenPBRHomogeneousVolume openPBRMakeVolumeFromExtinctionCoefficientAndAlbedoAndAnisotropy(float3 extinction_coefficient,
                                                                                          float3 albedo,
                                                                                          float anisotropy)
{
    OpenPBRHomogeneousVolume volume;
    volume.extinction_coefficient = extinction_coefficient;
    volume.albedo = albedo;
    volume.anisotropy = anisotropy;
    return volume;
}

// Initializes a volume from extinction and albedo.

public OpenPBRHomogeneousVolume openPBRMakeVolumeFromExtinctionCoefficientAndAlbedo(float3 extinction_coefficient,
                                                                             float3 albedo)
{
    return openPBRMakeVolumeFromExtinctionCoefficientAndAlbedoAndAnisotropy(extinction_coefficient, albedo, 0.0f);
}

// Initializes an empty volume (vacuum).

public OpenPBRHomogeneousVolume openPBRMakeEmptyVolume()
{
    return openPBRMakeVolumeFromExtinctionCoefficientAndAlbedoAndAnisotropy(float3(0.0f), float3(0.0f), 0.0f);
}

// Initializes a default ambient volume.

public OpenPBRHomogeneousVolume openPBRMakeAmbientVolume()
{
    return openPBRMakeVolumeFromExtinctionCoefficientAndAlbedoAndAnisotropy(
        float3(OpenPBRAmbientVolumeExtinctionCoefficient), float3(OpenPBRAmbientVolumeAlbedo),
        OpenPBRAmbientVolumeAnisotropy);
}

// Initializes a volume from extinction, creating an ambient volume if necessary.

public OpenPBRHomogeneousVolume openPBRMakeVolumeFromExtinctionCoefficient(float3 extinction_coefficient)
{
    if (all(((extinction_coefficient) == (float3(OpenPBRAmbientVolumeExtinctionCoefficient))))) {
        return openPBRMakeAmbientVolume();
    } else {
        return openPBRMakeVolumeFromExtinctionCoefficientAndAlbedo(extinction_coefficient, float3(0.0f));
    }
}

// Initializes a volume from an absorption coefficient and a scattering coefficient.

public OpenPBRHomogeneousVolume
openPBRMakeVolumeFromAbsorptionAndScatteringCoefficientsAndAnisotropy(float3 absorption_coefficient,
                                                                      float3 scattering_coefficient, float anisotropy)
{
    OpenPBRHomogeneousVolume volume;
    volume.extinction_coefficient = absorption_coefficient + scattering_coefficient;
    if (all(((volume.extinction_coefficient) > (float3(0.0f))))) {
        volume.albedo = scattering_coefficient / volume.extinction_coefficient;
    } else {
        for (int color_channel = 0; color_channel < 3; ++color_channel) {
            volume.albedo[color_channel] =
                volume.extinction_coefficient[color_channel] > 0.0f
                    ? scattering_coefficient[color_channel] / volume.extinction_coefficient[color_channel]
                    : 0.0f;
        }
    }
    volume.anisotropy = anisotropy;
    return volume;
}

// Initializes a volume from an absorption coefficient and a scattering coefficient.

public OpenPBRHomogeneousVolume openPBRMakeVolumeFromAbsorptionAndScatteringCoefficients(float3 absorption_coefficient,
                                                                                  float3 scattering_coefficient)
{
    return openPBRMakeVolumeFromAbsorptionAndScatteringCoefficientsAndAnisotropy(absorption_coefficient,
                                                                                 scattering_coefficient, 0.0f);
}

// Initializes a purely absorbing volume from an absorption coefficient alone.

public OpenPBRHomogeneousVolume openPBRMakeVolumeFromAbsorptionCoefficient(float3 absorption_coefficient)
{
    return openPBRMakeVolumeFromExtinctionCoefficientAndAlbedo(absorption_coefficient, float3(0.0f));
}

// Checks if a volume is empty (vacuum).

public bool openPBRIsEmptyVolume(const OpenPBRHomogeneousVolume volume)
{
    return all(((volume.extinction_coefficient) == (float3(0.0f))));
}

// Checks if a volume is an ambient volume.

public bool openPBRIsAmbientVolume(const OpenPBRHomogeneousVolume volume)
{
    return all(((volume.extinction_coefficient) == (float3(OpenPBRAmbientVolumeExtinctionCoefficient)))) &&
           all(((volume.albedo) == (float3(OpenPBRAmbientVolumeAlbedo)))) &&
           volume.anisotropy == OpenPBRAmbientVolumeAnisotropy;
}

// Special version of function above with alternative interface
// for use when only the extinction coefficient is known.

public bool openPBRIsAmbientVolume(const float3 extinction_coefficient)
{
    return all(((extinction_coefficient) == (float3(OpenPBRAmbientVolumeExtinctionCoefficient))));
}

// Checks if a volume is a non-empty non-ambient volume.

public bool openPBRIsHomogeneousVolume(const OpenPBRHomogeneousVolume volume)
{
    return (!openPBRIsEmptyVolume(volume)) && (!openPBRIsAmbientVolume(volume));
}

// Calculates absorption coefficient of volume based on extinction and albedo.

public float3 openPBRCalculateAbsorptionCoefficient(const OpenPBRHomogeneousVolume volume)
{
    return volume.extinction_coefficient * (1.0f - volume.albedo);
}

// Calculates scattering coefficient of volume based on extinction and albedo.

public float3 openPBRCalculateScatteringCoefficient(const OpenPBRHomogeneousVolume volume)
{
    return volume.extinction_coefficient * volume.albedo;
}

// Calculates negative optical depth for volume.

public float3 openPBRCalculateNegativeOpticalDepth(const OpenPBRHomogeneousVolume volume, const float distance)
{
    return volume.extinction_coefficient * -distance;
}

// Calculates volume transmittance at the given finite distance.

public float3 openPBRCalculateTransmittanceAtDistance(const OpenPBRHomogeneousVolume volume, const float distance)
{

    return exp(openPBRCalculateNegativeOpticalDepth(volume, distance));
}

// Special version of function above with alternative interface
// for use when only the extinction coefficient is known.

public float3 openPBRCalculateTransmittanceAtDistance(const float3 extinction_coefficient, const float distance)
{
    const OpenPBRHomogeneousVolume volume = openPBRMakeVolumeFromExtinctionCoefficient(extinction_coefficient);

    // Special case to support ambient volumes and speed up transmittance calculation for empty volumes.
    if (!openPBRIsHomogeneousVolume(volume)) {
        return float3(1.0f);
    }

    return openPBRCalculateTransmittanceAtDistance(volume, distance);
}

// Calculates the transmittance at an infinite distance.

public float3 openPBRCalculateTransmittanceAtInfinity(const OpenPBRHomogeneousVolume volume)
{
    return float3(((volume.extinction_coefficient) == (float3(0.0f))));
}

// Calculates probabilities of sampling a distance based on each color channel.

public float3 openPBRCalculateColorChannelProbability(const OpenPBRHomogeneousVolume volume, const float3 throughput)
{
    const float3 total_contribution = throughput * volume.albedo;
    const float total_contribution_sum = openPBRSum(total_contribution);
    // We check if total_contribution_sum >= FloatMin instead of > 0.0f because
    // 1.0f / total_contribution_sum could overflow if total_contribution_sum is subnormal.
    // In that case Infs and NaNs would result and cause invalid array indexing downstream.
    if (total_contribution_sum >= OpenPBRFloatMin) {
        return total_contribution * (1.0f / total_contribution_sum);
    } else {
        return float3(1.0f / 3.0f);
    }
}

// This function optimally samples a single extinction or transmission event based on the expected
// contribution of each event type, assuming no knowledge of the lighting or geometric configuration
// or the next surface intersection distance. Specifically, this function samples an interaction
// distance from a PDF that combines the PDFs of the individual color channels based on their expected
// contributions. The result is that the weights and the average path throughput remain stable even
// when the volume has highly chromatic extinction and high albedo.
//
// This function sets the distance argument if light interacted with the volume. If there is no
// volume, the distance argument is left as is.

public void openPBRSampleEventDistance(const OpenPBRHomogeneousVolume volume, const float3 throughput, const float rand,
                                inout float distance)
{
    // We could return for empty volumes, but not calling this function for empty volumes saves a random number.

    // Make sure the random number is in [0, 1).

    // Select a color channel.
    const float3 color_channel_probability = openPBRCalculateColorChannelProbability(volume, throughput);
    const float rand_color_channel =
        rand *
        openPBRSum(
            color_channel_probability); // avoid numerical issues (probability sum equals one mathematically but not numerically)
    float cdf = 0.0f;
    int sampled_color_channel = -1;
    float rand_distance = 0.0f;
    for (int i = 0; i < 3; ++i) {
        const float probability = color_channel_probability[i];
        const float cdf_next = cdf + probability;
        if (rand_color_channel < cdf_next) {
            sampled_color_channel = i;
            rand_distance =
                (rand_color_channel - cdf) /
                (cdf_next -
                 cdf); // avoid numerical issues (denominator equals probability mathematically but not numerically)

            if (rand_distance == 1.0f)
                rand_distance =
                    OpenPBRLargestFloatBelowOne; // avoid numerical issues (rand_distance is always less than one mathematically but can
                                                 // be exactly one when calculated in single precision)

            break;
        }
        cdf = cdf_next;
    }

    // Verify that a color channel was selected successfully.

    // Make sure the extinction of the sampled color channel is greater than zero;
    // otherwise, the sampled distance is infinity and the distance argument is left as is.
    const float extinction_of_sampled_color_channel = volume.extinction_coefficient[sampled_color_channel];
    if (extinction_of_sampled_color_channel > 0.0f) {
        // Perfectly sample an event distance for the selected color channel.
        distance =
            -log(1.0f - rand_distance) /
            extinction_of_sampled_color_channel; // one minus rand is never zero, preventing log from being negative infinity
    }
}

// Calculate the weight to apply to a volume interaction that occurs before a surface interaction.

public float3 openPBRCalculateWeightForEventAtDistance(const OpenPBRHomogeneousVolume volume, const float3 throughput,
                                                const float distance)
{
    // We could return zero for empty volumes, but it doesn't make sense to call this function for empty volumes in the first place.

    const float3 color_channel_probability = openPBRCalculateColorChannelProbability(volume, throughput);
    const float3 transmittance = openPBRCalculateTransmittanceAtDistance(volume, distance);

    const float3 color_channel_pdf =
        transmittance * volume.extinction_coefficient; // color-channel pdf at event distance
    const float event_pdf = openPBRSum(color_channel_probability *
                                       color_channel_pdf); // overall pdf based on contributions of all color channels

    // Note: 1.0f/event_pdf could overflow if event_pdf is subnormal.
    return transmittance * openPBRCalculateScatteringCoefficient(volume) *
           (1.0f / event_pdf); // equals albedo in the case of achromatic extinction
}

// Calculate the weight to apply to a surface interaction that occurs before a volume interaction.
// Specifically, this incorporates the volume transmittance and the probability of the sampled
// volume distance falling beyond the provided surface distance.

public float3 openPBRCalculateWeightForSurfaceAtDistance(const OpenPBRHomogeneousVolume volume, const float3 throughput,
                                                  const float distance)
{
    // Special case for non-interaction with an empty volume.
    if (!openPBRIsHomogeneousVolume(volume)) {
        return float3(1.0f);
    }

    const float3 color_channel_probability = openPBRCalculateColorChannelProbability(volume, throughput);
    const float3 transmittance = openPBRCalculateTransmittanceAtDistance(volume, distance);

    const float surface_probability = openPBRSum(
        color_channel_probability *
        transmittance); // overall probability of reaching the surface based on contributions of all color channels

    // Note: 1.0f/surface_probability could overflow if surface_probability is subnormal.
    return transmittance * (1.0f / surface_probability); // equals one in the case of achromatic extinction
}

// Perfectly samples the isotropic phase function. The weight is implicitly one.
// The albedo is not relevant because it was already included in the distance-sampling weight.

public float3 openPBRSampleIsotropicPhaseFunction(const float2 rand) { return openPBRSampleUnitSphereUniform(rand); }

// Returns the value of the isotropic phase function for any direction.

public float openPBRCalculateIsotropicPhaseFunctionValue() { return OpenPBRRcpFourPi; }

// Returns the probability density of sampling any direction from the isotropic phase function.

public float openPBRCalculateIsotropicPhaseFunctionPdf() { return OpenPBRRcpFourPi; }

// Perfectly samples the anisotropic Henyey-Greenstein phase function. The weight is implicitly one.
// The albedo is not relevant because it was already included in the distance-sampling weight.

public float3 openPBRSampleAnisotropicPhaseFunction(const OpenPBRHomogeneousVolume volume, const float3 view_direction,
                                             const float2 rand)
{

    // For details, see https://pages.astro.umd.edu/~jph/HG_note.pdf
    const float g = volume.anisotropy;
    const float s = 2.0f * rand.x - 1.0f;
    const float cos_theta =
        g == 0 ? s : (0.5f / g) * (1.0f + openPBRSquare(g) - openPBRSquare((1.0f - openPBRSquare(g)) / (1.0f + g * s)));

    const float sin_theta = sqrt(max(0.0f, 1.0f - openPBRSquare(cos_theta)));
    const float phi = OpenPBRTwoPi * rand.y;

    const OpenPBRBasis basis = openPBRMakeBasis(-view_direction);
    return cos_theta * basis.n + sin_theta * cos(phi) * basis.t + sin_theta * sin(phi) * basis.b;
}

// Returns the value of the anisotropic Henyey-Greenstein phase function for any direction.

public float openPBRCalculateAnisotropicPhaseFunctionValue(const OpenPBRHomogeneousVolume volume, const float3 view_direction,
                                                    const float3 light_direction)
{

    // For details, see https://pages.astro.umd.edu/~jph/HG_note.pdf
    const float g = volume.anisotropy;
    const float cos_theta = dot(-view_direction, light_direction);
    return OpenPBRRcpFourPi * (1.0f - openPBRSquare(g)) /
           openPBRThreeHalvesPower(1.0f + openPBRSquare(g) - 2.0f * g * cos_theta);
}

// Returns the probability density of sampling any direction from the anisotropic Henyey-Greenstein phase function.

public float openPBRCalculateAnisotropicPhaseFunctionPdf(const OpenPBRHomogeneousVolume volume, const float3 view_direction,
                                                  const float3 light_direction)
{
    return openPBRCalculateAnisotropicPhaseFunctionValue(volume, view_direction, light_direction);
}

public OpenPBRHomogeneousVolume openPBRAddVolumes(const OpenPBRHomogeneousVolume volume_A,
                                           const OpenPBRHomogeneousVolume volume_B, const float volume_A_weight,
                                           const float volume_B_weight)
{

    const float3 total_absorption_coefficient = openPBRCalculateAbsorptionCoefficient(volume_A) * volume_A_weight +
                                                openPBRCalculateAbsorptionCoefficient(volume_B) * volume_B_weight;
    const float3 total_scattering_coefficient = openPBRCalculateScatteringCoefficient(volume_A) * volume_A_weight +
                                                openPBRCalculateScatteringCoefficient(volume_B) * volume_B_weight;
    const float average_anisotropy =
        openPBRSafeDivide(volume_A.anisotropy * volume_A_weight + volume_B.anisotropy * volume_B_weight,
                          volume_A_weight + volume_B_weight, 0.0f);

    return openPBRMakeVolumeFromAbsorptionAndScatteringCoefficientsAndAnisotropy(
        total_absorption_coefficient, total_scattering_coefficient, average_anisotropy);
}

// Returns the value of Eclair's "weighted roughness" heuristic for volumes.

public float openPBRCalculateWeightedRoughnessForVolume(const OpenPBRHomogeneousVolume volume)
{
    // This is empirically derived.
    // TODO(sss): Revisit this.
    return sqrt(1.0f - abs(volume.anisotropy));
}

// This function introduces a weight for shadow rays that reduces or eliminates faceting artifacts that appear on dense volumes inside low-poly
// meshes. The weight is a ratio of cosine terms that compensates for the difference between the projected areas of the imaginary smooth surface and
// the actual geometric surface.
//
// Additional context and details: In the case of shadow rays that originate from a volume and pass straight through the enclosing interface, there is
// no foreshortening cosine term explicitly applied to the weight. However, there is foreshortening cosine term implicitly incorporated into the
// distribution of shadow rays over the surface, and this foreshortening is naturally based on the actual geometric surface; it is unaffected by the
// interpolated normal. In the case of a low-poly mesh containing a dense volume, faceting artifacts that follow the geometric surface become visible.
// What the weight in this function does is divide out the implicitly present cosine term that's based on the geometric normal and multiply in a
// cosine term based on the interpolated normal, which results in the desired smooth appearance across the mesh.

public float openPBRVolumeFacetingCorrection(const bool last_scattered_from_volume, const float side_light,
                                      const float side_view, const float3 light_direction,
                                      const float3 geometric_normal_normalized,
                                      const float3 interpolated_normal_normalized)
{
    if (last_scattered_from_volume && side_light > 0.0f && side_view < 0.0f) // emerging from volume
    {
        return abs(openPBRSafeDivide(dot(light_direction, interpolated_normal_normalized),
                                     dot(light_direction, geometric_normal_normalized), 1.0f));
    } else {
        return 1.0f;
    }
}
} // namespace Metallic.OpenPBR

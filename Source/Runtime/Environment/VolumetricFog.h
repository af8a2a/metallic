#pragma once

#include "ml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace metallic::environment {

struct LocalFogVolume {
    bool enabled = true;
    std::array<float, 3> centerMeters{};
    // Right-handed local-to-world rotation about X, then Y, then Z.
    std::array<float, 3> rotationDegrees{};
    float radiusMeters = 5.0f;
    float radialExtinction = 1.0f;
    float heightExtinction = 1.0f;
    float heightFalloff = 1000.0f;
    float heightOffset = 0.0f; // Local unit-sphere coordinate.
    std::array<float, 3> albedo{1.0f, 1.0f, 1.0f};
    std::array<float, 3> emissive{};
    int sortPriority = 0;
    uint64_t stableId = 0; // Assigned by SceneDocument, omitted from render state comparisons.

    bool operator==(const LocalFogVolume& other) const
    {
        return enabled == other.enabled && centerMeters == other.centerMeters && rotationDegrees == other.rotationDegrees &&
            radiusMeters == other.radiusMeters && radialExtinction == other.radialExtinction &&
            heightExtinction == other.heightExtinction && heightFalloff == other.heightFalloff &&
            heightOffset == other.heightOffset && albedo == other.albedo && emissive == other.emissive &&
            sortPriority == other.sortPriority;
    }
};

inline bool validLocalFogVolume(const LocalFogVolume& volume)
{
    const auto finite = [](float x) { return std::isfinite(x); };
    const auto nonnegative = [](float x) { return x >= 0.0f && std::isfinite(x); };
    return std::ranges::all_of(volume.centerMeters, finite) && std::ranges::all_of(volume.rotationDegrees, finite) &&
        finite(volume.radiusMeters) && volume.radiusMeters > 0.0f && nonnegative(volume.radialExtinction) &&
        nonnegative(volume.heightExtinction) && finite(volume.heightFalloff) && volume.heightFalloff >= 1.0f &&
        finite(volume.heightOffset) && std::ranges::all_of(volume.albedo, [](float x) { return x >= 0.0f && x <= 1.0f; }) &&
        std::ranges::all_of(volume.emissive, nonnegative) && volume.sortPriority >= -127 && volume.sortPriority <= 127;
}

struct VolumetricFogState {
    bool enabled = false;
    // Densities and falloffs use Unreal's authored exponential height fog scale.
    float density = 0.02f;
    float heightFalloff = 0.2f;
    float heightMeters = 0.0f;
    float secondDensity = 0.0f;
    float secondHeightFalloff = 0.2f;
    float secondHeightOffsetMeters = 0.0f;
    float3 albedo{1.0f};
    float3 emissive{0.0f};
    float extinctionScale = 1.0f;
    float scatteringDistribution = 0.2f;
    float directionalScatteringIntensity = 1.0f;
    // Dynamic Sky Light parameters used when no Lumen translucency volume is connected.
    float skyLightVolumetricScatteringIntensity = 1.0f;
    float skyOcclusionMaxDistanceMeters = 10.0f;
    float distanceMeters = 60.0f;
    float startDistanceMeters = 0.0f;
    float nearFadeInDistanceMeters = 0.0f;
    std::vector<LocalFogVolume> localVolumes;

    bool operator==(const VolumetricFogState& other) const
    {
        return enabled == other.enabled && density == other.density && heightFalloff == other.heightFalloff &&
            heightMeters == other.heightMeters && secondDensity == other.secondDensity &&
            secondHeightFalloff == other.secondHeightFalloff && secondHeightOffsetMeters == other.secondHeightOffsetMeters &&
            albedo.x == other.albedo.x && albedo.y == other.albedo.y && albedo.z == other.albedo.z &&
            emissive.x == other.emissive.x && emissive.y == other.emissive.y && emissive.z == other.emissive.z &&
            extinctionScale == other.extinctionScale && scatteringDistribution == other.scatteringDistribution &&
            directionalScatteringIntensity == other.directionalScatteringIntensity &&
            skyLightVolumetricScatteringIntensity == other.skyLightVolumetricScatteringIntensity &&
            skyOcclusionMaxDistanceMeters == other.skyOcclusionMaxDistanceMeters && distanceMeters == other.distanceMeters &&
            startDistanceMeters == other.startDistanceMeters && nearFadeInDistanceMeters == other.nearFadeInDistanceMeters &&
            localVolumes == other.localVolumes;
    }
};

inline bool validVolumetricFog(const VolumetricFogState& fog)
{
    const auto nonnegative = [](float value) { return value >= 0.0f && std::isfinite(value); };
    const auto unit = [](float value) { return value >= 0.0f && value <= 1.0f; };
    return nonnegative(fog.density) && nonnegative(fog.heightFalloff) && std::isfinite(fog.heightMeters) &&
        nonnegative(fog.secondDensity) && nonnegative(fog.secondHeightFalloff) && std::isfinite(fog.secondHeightOffsetMeters) &&
        unit(fog.albedo.x) && unit(fog.albedo.y) && unit(fog.albedo.z) &&
        nonnegative(fog.emissive.x) && nonnegative(fog.emissive.y) && nonnegative(fog.emissive.z) &&
        nonnegative(fog.extinctionScale) && nonnegative(fog.directionalScatteringIntensity) &&
        nonnegative(fog.skyLightVolumetricScatteringIntensity) && nonnegative(fog.skyOcclusionMaxDistanceMeters) &&
        fog.scatteringDistribution >= -0.99f && fog.scatteringDistribution <= 0.99f &&
        nonnegative(fog.startDistanceMeters) && nonnegative(fog.nearFadeInDistanceMeters) &&
        // CalculateGridZParams places the first slice 9.5 cm beyond the start plane.
        std::isfinite(fog.distanceMeters) && fog.distanceMeters > fog.startDistanceMeters + 0.095f &&
        std::ranges::all_of(fog.localVolumes, validLocalFogVolume);
}

} // namespace metallic::environment

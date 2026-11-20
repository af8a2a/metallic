#pragma once

#include "Runtime/Scene/MeshDistanceField.h"

#include <span>

namespace metallic::scene {

struct MeshDistanceFieldBakeOptions {
    float resolutionScale = 1.0f;
    // Keep the reference's centimeter-based voxel density at the bake boundary.
    float voxelDensity = 0.2f;
    uint32_t maxResolution = 256;
};

// Positions are in local meters. Triangle indices use outward CCW winding.
// Supply all single-sided opaque sections of one logical mesh together.
bool bakeMeshDistanceField(std::span<const std::array<float, 3>> positions,
    std::span<const uint32_t> indices, const MeshDistanceFieldBakeOptions& options,
    MeshDistanceField& result, std::string& error);

} // namespace metallic::scene

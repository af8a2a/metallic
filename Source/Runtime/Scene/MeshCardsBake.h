#pragma once

#include "Runtime/Scene/MeshCards.h"

#include <optional>
#include <span>

namespace metallic::scene {

struct MeshCardsBakeOptions {
    uint32_t maxCards = 12;
    // The optional pair is the Mesh SDF's local minimum and maximum in meters.
    std::optional<std::array<std::array<float, 3>, 2>> bounds;
};

// Supply all opaque, single-sided sections of one logical mesh in local meters.
// Triangles use outward CCW winding. Output follows UE's axis order.
bool bakeMeshCards(std::span<const std::array<float, 3>> positions, std::span<const uint32_t> indices,
    const MeshCardsBakeOptions& options, std::vector<MeshCardBuildRecord>& cards, std::string& error);

} // namespace metallic::scene

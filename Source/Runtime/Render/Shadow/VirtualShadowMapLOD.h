#pragma once

#include "Runtime/Render/MeshletLOD.h"

namespace metallic::render {

// The actual shadow projection's homogeneous X, Y and W rows. Resolution is
// the full virtual face width in texels, including its page grid. Perspective
// W is forward distance, and nearPlane uses the same units. Orthographic W=1.
struct VirtualShadowMapLODProjection
{
    std::array<float, 4> rowX{};
    std::array<float, 4> rowY{};
    std::array<float, 4> rowW{0, 0, 0, 1};
    uint32_t resolution = 0;
    float nearPlane = 0;
};

struct VirtualShadowMapLODRequirements
{
    std::vector<uint8_t> needsFine;
    // Maximum nonterminal replacement error tested, not the selected cut's
    // residual error. Terminal FLT_MAX stop sentinels are excluded.
    float maxProjectedError = 0;
};

// Maximum conservative error in shadow texels over all admitted shadow views.
// A sphere outside a view contributes zero demand, never removes the caster.
// Terminal groups have no replacement metric and return zero.
float virtualShadowMapLodPixelError(const MeshletLODGroupRecord& group,
    const GPUSceneGPUInstanceRecord& instance,
    std::span<const VirtualShadowMapLODProjection> projections);

VirtualShadowMapLODRequirements buildVirtualShadowMapLodRequirements(
    std::span<const MeshletLODGroupRecord> groups,
    const GPUSceneGPUInstanceRecord& instance,
    std::span<const VirtualShadowMapLODProjection> projections, float pixelThreshold);

} // namespace metallic::render

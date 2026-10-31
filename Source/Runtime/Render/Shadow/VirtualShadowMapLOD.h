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

// Immutable primitive topology, built at geometry publication rather than per
// instance/frame. Metric leaves exclude terminal stop sentinels; terminal roots
// are always demanded separately. Cluster indices retain the original order.
struct VirtualShadowMapLODHierarchy
{
    std::vector<MeshletLODGroupRecord> groups;
    std::vector<MeshletLODGroupRange> ranges;
    std::vector<uint32_t> refinedGroups;
    std::vector<std::vector<uint32_t>> parents;
    std::vector<uint32_t> clusterGroups;
    std::vector<uint32_t> terminalGroups;
    std::vector<uint32_t> metricGroupIDs;
    std::vector<MeshletLODBVHNode> bvh;
};

bool buildVirtualShadowMapLODHierarchy(std::span<const MeshletLODGroupRecord> groups,
    std::span<const MeshletLODGroupRange> ranges, std::span<const uint32_t> refinedGroups,
    VirtualShadowMapLODHierarchy& hierarchy, std::string& reason);

struct VirtualShadowMapLODSelection
{
    StreamMeshletLODReference cut;
    uint32_t metricGroups = 0;
    uint32_t hierarchyNodes = 0;
    uint32_t prunedGroups = 0;
    uint32_t topologyGroups = 0;
    uint32_t topologyEdges = 0;
    uint32_t clusterTests = 0;
};

// Hierarchical and flat paths use the same group metric and residency rules.
// The flat path is retained as an independent cut/reference and A/B control.
// Dense activeGroups remains part of the result ABI; only demanded topology and
// active cluster ranges are traversed by the hierarchical path.
VirtualShadowMapLODSelection selectVirtualShadowMapLOD(const VirtualShadowMapLODHierarchy& hierarchy,
    std::span<const uint8_t> drawableGroups, const GPUSceneGPUInstanceRecord& instance,
    std::span<const VirtualShadowMapLODProjection> projections, float pixelThreshold,
    bool hierarchical = true);

} // namespace metallic::render

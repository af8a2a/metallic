#pragma once

#include "Runtime/Render/GPUDrivenRaster.h"
#include <array>
#include <span>
#include <string>
#include <vector>

namespace metallic::scene { struct RenderPrimitive; }

namespace metallic::render {

inline constexpr uint32_t kMeshletLODInvalidGroup = UINT32_MAX;
inline constexpr uint32_t kMeshletLODTerminalGroup = 1u;

// Shared replacement metric. All clusters refining this group use this sphere
// and error, never their individual geometry/culling bounds.
struct alignas(16) MeshletLODGroupRecord {
    std::array<float, 4> sphere{};
    float error = 0;
    uint32_t level = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
};

// Preorder BVH4 with stackless escape links and descending group-ID leaves.
// groupCount == 0 identifies an interior; leaf ranges contain at most 4 groups.
// All offsets and escape indices are primitive-local. Scalar layout matches GPU.
struct MeshletLODBVHNode {
    std::array<float, 4> sphere{};
    float maxError = 0;
    uint32_t maxLevel = 0;
    uint32_t flags = 0;
    uint32_t escapeIndex = 0;
    uint32_t groupOffset = 0;
    uint32_t groupCount = 0;
};

struct MeshletLODView {
    // World-space eye / near plane, normalized forward / orthographic flag.
    std::array<float, 4> eye{0, 0, 0, 0.1f};
    std::array<float, 4> forward{0, 0, -1, 0};
    // Internal render height, tan(vertical FOV / 2), ortho height, render-pixel
    // threshold. Convert the user-facing display-pixel error at the CPU boundary.
    std::array<float, 4> projection{1080, 0.577350269f, 10, 1.5f};
};

// Raster, jitter guards and HZB keep using render pixels. Scale the LOD threshold
// instead so the same display-pixel budget selects the same geometry across
// internal resolutions. A zero display height means native resolution.
float meshletLodRenderPixelThreshold(float displayPixelError, uint32_t renderHeight,
    uint32_t displayHeight = 0);

struct MeshletLODGroupRange;
struct GPUSceneGPUInstanceRecord;

// Includes every replacement descendant, padded by its authored error. Used
// only for page demand culling; it does not change the LOD error metric.
struct MeshletLODRefinementBounds {
    std::array<float, 3> min{1, 1, 1};
    std::array<float, 3> max{-1, -1, -1};
};

bool buildMeshletLodRefinementBounds(std::span<const MeshletLODGroupRecord> groups,
    std::span<const MeshletLODGroupRange> ranges, std::span<const uint32_t> refinedGroups,
    std::vector<MeshletLODRefinementBounds>& bounds, std::string& reason);

bool meshletLodBoundsVisible(const MeshletLODRefinementBounds& bounds,
    const GPUSceneGPUInstanceRecord& instance, const MeshletLODView& view,
    std::array<float, 3> cameraUp, float aspect, float farPlane);

// GPU output. recordIndex retains the immutable VBuffer indirection; compacted
// list order is deliberately not a persistent geometry identity.
struct alignas(16) MeshletLODSelection {
    uint32_t instanceIndex = 0;
    uint32_t clusterIndex = 0;
    uint32_t recordIndex = 0;
    uint32_t geometryIndex = 0;
    auto operator<=>(const MeshletLODSelection&) const = default;
};

struct GPUSceneGPUMeshletRecord;

bool buildMeshletLodMetadata(const scene::RenderPrimitive& primitive,
    std::vector<MeshletLODGroupRecord>& groups, std::string& reason);
bool buildMeshletLodBvh(std::span<const MeshletLODGroupRecord> groups,
    std::vector<MeshletLODBVHNode>& nodes, std::string& reason);
// Ordered BVH4 of same-level tiles (<=64 groups). Zero groupCount denotes an
// interior, escapeIndex skips a subtree. Leaves stay in descending group order;
// bounds use the same conservative aggregation as the BVH. No cook change.
bool buildMeshletLodTiles(std::span<const MeshletLODGroupRecord> groups,
    std::vector<MeshletLODBVHNode>& tiles, std::string& reason);
// Disjoint, bounded subtrees of a forest produced by buildMeshletLodTiles.
// Demand has no parent residency dependencies, so these roots can run in any
// order across instances.
std::vector<uint32_t> buildMeshletLodDemandRoots(std::span<const MeshletLODBVHNode> tiles,
    uint32_t maxSubtreeNodes = 8);
std::vector<uint32_t> buildMeshletLodTileParents(std::span<const MeshletLODBVHNode> tiles);
float meshletLodPixelError(const MeshletLODGroupRecord& group,
    const GPUSceneGPUInstanceRecord& instance, const MeshletLODView& view);
bool meshletLodNeedsFine(const MeshletLODGroupRecord& group,
    const GPUSceneGPUInstanceRecord& instance, const MeshletLODView& view,
    uint32_t manualLevel = UINT32_MAX);

// Always-resident topology; refinedGroups contains one group index per cluster.
// Ranges, group indices and selected cluster indices are primitive-local.
struct MeshletLODGroupRange {
    uint32_t clusterOffset = 0;
    uint32_t clusterCount = 0;
};

struct StreamMeshletLODReference {
    std::vector<uint32_t> selectedClusters;
    std::vector<uint32_t> requestedGroups;
    std::vector<uint8_t> activeGroups;
    bool valid = true;
    bool capacityExceeded = false;
    bool capacityFallback = false;
    uint32_t visitedBvhNodes = 0;
    uint32_t testedGroups = 0;
    std::string reason;
};

// A finer group is reachable only when every owner of its coarse replacement
// is active. This keeps missing intermediate pages from exposing descendants
// underneath a coarser fallback. Requests advance the same reachable frontier.
// Capacity is the number of nonempty active groups, matching stream GPU output.
StreamMeshletLODReference selectStreamMeshletLodReference(
    std::span<const MeshletLODGroupRecord> groups,
    std::span<const MeshletLODGroupRange> ranges,
    std::span<const uint32_t> refinedGroups,
    std::span<const uint8_t> drawableGroups,
    const GPUSceneGPUInstanceRecord& instance, const MeshletLODView& view,
    uint32_t manualLevel = UINT32_MAX, uint32_t capacity = UINT32_MAX,
    std::span<const uint8_t> availableGroups = {},
    std::span<const MeshletLODBVHNode> bvh = {});

std::vector<MeshletLODSelection> selectMeshletLodReference(
    std::span<const MeshletLODGroupRecord> groups,
    std::span<const GPUSceneGPUMeshletRecord> clusters,
    std::span<const GPUSceneGPUInstanceRecord> instances,
    std::span<const VisibleClusterRecord> candidates, const MeshletLODView& view,
    uint32_t recordOffset = 0, uint32_t manualLevel = UINT32_MAX);

static_assert(sizeof(MeshletLODGroupRecord) == 32);
static_assert(sizeof(MeshletLODBVHNode) == 40);
static_assert(sizeof(MeshletLODSelection) == 16);
static_assert(sizeof(MeshletLODView) == 48);

} // namespace metallic::render

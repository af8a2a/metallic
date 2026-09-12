#pragma once

#include "Runtime/Render/GPUDrivenRaster.h"
#include <array>
#include <span>
#include <string>
#include <vector>

namespace metallic::scene { struct RenderPrimitive; }

namespace metallic::render {

inline constexpr uint32_t kMeshletLodInvalidGroup = UINT32_MAX;
inline constexpr uint32_t kMeshletLodTerminalGroup = 1u;

// Shared replacement metric. All clusters refining this group use this sphere
// and error, never their individual geometry/culling bounds.
struct alignas(16) MeshletLodGroupRecord {
    std::array<float, 4> sphere{};
    float error = 0;
    uint32_t level = 0;
    uint32_t flags = 0;
    uint32_t reserved = 0;
};

// Preorder BVH4 with stackless escape links and descending group-ID leaves.
// groupCount == 0 identifies an interior; leaf ranges contain at most 4 groups.
// All offsets and escape indices are primitive-local. Scalar layout matches GPU.
struct MeshletLodBvhNode {
    std::array<float, 4> sphere{};
    float maxError = 0;
    uint32_t maxLevel = 0;
    uint32_t flags = 0;
    uint32_t escapeIndex = 0;
    uint32_t groupOffset = 0;
    uint32_t groupCount = 0;
};

struct MeshletLodView {
    // World-space eye / near plane, normalized forward / orthographic flag.
    std::array<float, 4> eye{0, 0, 0, 0.1f};
    std::array<float, 4> forward{0, 0, -1, 0};
    // Internal render height, tan(vertical FOV / 2), ortho height, pixel error.
    std::array<float, 4> projection{1080, 0.577350269f, 10, 1.5f};
};

// GPU output. recordIndex retains the immutable VBuffer indirection; compacted
// list order is deliberately not a persistent geometry identity.
struct alignas(16) MeshletLodSelection {
    uint32_t instanceIndex = 0;
    uint32_t clusterIndex = 0;
    uint32_t recordIndex = 0;
    uint32_t geometryIndex = 0;
    auto operator<=>(const MeshletLodSelection&) const = default;
};

struct GPUSceneGpuMeshletRecord;
struct GPUSceneGpuInstanceRecord;

bool buildMeshletLodMetadata(const scene::RenderPrimitive& primitive,
    std::vector<MeshletLodGroupRecord>& groups, std::string& reason);
bool buildMeshletLodBvh(std::span<const MeshletLodGroupRecord> groups,
    std::vector<MeshletLodBvhNode>& nodes, std::string& reason);
float meshletLodPixelError(const MeshletLodGroupRecord& group,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view);
bool meshletLodNeedsFine(const MeshletLodGroupRecord& group,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view,
    uint32_t manualLevel = UINT32_MAX);

// Always-resident topology; refinedGroups contains one group index per cluster.
// Ranges, group indices and selected cluster indices are primitive-local.
struct MeshletLodGroupRange {
    uint32_t clusterOffset = 0;
    uint32_t clusterCount = 0;
};

struct StreamMeshletLodReference {
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
StreamMeshletLodReference selectStreamMeshletLodReference(
    std::span<const MeshletLodGroupRecord> groups,
    std::span<const MeshletLodGroupRange> ranges,
    std::span<const uint32_t> refinedGroups,
    std::span<const uint8_t> drawableGroups,
    const GPUSceneGpuInstanceRecord& instance, const MeshletLodView& view,
    uint32_t manualLevel = UINT32_MAX, uint32_t capacity = UINT32_MAX,
    std::span<const uint8_t> availableGroups = {},
    std::span<const MeshletLodBvhNode> bvh = {});

std::vector<MeshletLodSelection> selectMeshletLodReference(
    std::span<const MeshletLodGroupRecord> groups,
    std::span<const GPUSceneGpuMeshletRecord> clusters,
    std::span<const GPUSceneGpuInstanceRecord> instances,
    std::span<const VisibleClusterRecord> candidates, const MeshletLodView& view,
    uint32_t recordOffset = 0, uint32_t manualLevel = UINT32_MAX);

static_assert(sizeof(MeshletLodGroupRecord) == 32);
static_assert(sizeof(MeshletLodBvhNode) == 40);
static_assert(sizeof(MeshletLodSelection) == 16);
static_assert(sizeof(MeshletLodView) == 48);

} // namespace metallic::render

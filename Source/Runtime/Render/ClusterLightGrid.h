#pragma once

#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace metallic::render {

struct ClusterLightGridDesc {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tileSize = 64;
    uint32_t depthSliceCount = 32;
    uint32_t maxLightsPerCell = 64;
    float3 eye{0.0f, 0.0f, 0.0f};
    float3 center{0.0f, 0.0f, -1.0f};
    float3 up{0.0f, 1.0f, 0.0f};
    float aspect = 1.0f;
    float fovRadians = 1.0471975512f;
    float zNear = 0.1f;
    float zFar = 1000.0f;
    // Positive values select orthographic projection and linear Z slices.
    float orthoHeight = 0.0f;
    float jitterGuardPixels = 0.0f;
};

// Shared with ClusterLightGridCommon.slang. No exposure or metadata element is
// prepended to the grid light buffer. Punctual GPUScene slots come first, then rectangles.
struct alignas(16) ClusterLightGridParams {
    std::array<uint32_t, 4> grid{}; // x, y, z, tile pixels
    std::array<uint32_t, 4> viewport{}; // width, height, cell capacity, orthographic
    std::array<float, 4> eyeNear{};
    std::array<float, 4> rightFar{};
    std::array<float, 4> upExtent{}; // up.xyz, tan(fov/2) or half ortho height
    std::array<float, 4> forwardExtent{}; // forward.xyz, horizontal extent
    // Perspective: log2(depth * B + O) * S. Orthographic: (depth-near)*S.
    std::array<float, 4> zParams{}; // B, O, S, pixel guard band for temporal jitter
    std::array<uint32_t, 4> counts{}; // clustered local, punctual source slots, unbounded punctual, total source slots
};
static_assert(sizeof(ClusterLightGridParams) == 128);

struct ClusterLightGridCell {
    uint32_t offset = 0;
    uint32_t count = 0; // stored entries, never greater than cell capacity
    uint32_t overflow = 0; // nonzero requires complete bounded-local fallback
    uint32_t totalCount = 0;
};
static_assert(sizeof(ClusterLightGridCell) == 16);

// Camera/configuration math only. Cell-light intersection and list construction
// have no CPU implementation or readback in the runtime.
[[nodiscard]] Result<ClusterLightGridParams> buildClusterLightGridParams(
    const ClusterLightGridDesc& desc,
    std::string& log);
float clusterLightGridSliceDepth(const ClusterLightGridParams& params, uint32_t slice);
bool clusterLightGridCellIndex(const ClusterLightGridParams& params,
    uint32_t pixelX, uint32_t pixelY, float viewDepth, uint32_t& cellIndex);

struct ClusterLightGridSnapshot {
    Buffer* parameters = nullptr;
    Buffer* lights = nullptr;
    // Rect payloads share the light and candidate publication lifetime.
    Buffer* rectLights = nullptr;
    // Clustered locals first, followed by unbounded punctual lights.
    Buffer* candidates = nullptr;
    Buffer* cells = nullptr;
    Buffer* lightIndices = nullptr;
    ClusterLightGridParams params;
    GPUSceneViewId sourceView;
    uint32_t frameSlot = 0;
    uint32_t sourceLightGeneration = 0;
    uint64_t sourceLightRevision = 0;
    uint64_t sourcePrepareCount = 0;
    uint64_t buildRevision = 0;
    uint64_t cellCount() const;
    uint32_t rectLightCount() const { return params.counts[3] - params.counts[1]; }
    bool valid() const;
};

class ClusterLightGrid {
public:
    // Tracked frames permit completed-allocation reuse. Untracked commands own
    // immutable resources/executable state until reset; callers must finish GPU work
    // before resetting the command pool, as required by the underlying RHI.
    Result<> record(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
        const GPUScene& scene, GPUSceneViewId view, uint32_t frameSlot,
        const ClusterLightGridDesc& desc, std::string& log, std::span<const GPURectLight> rectLights = {});
    const ClusterLightGridSnapshot* snapshot(const GPUScene& scene) const;
    // The owner must outlive the staged reload. Preparation leaves its current
    // program and snapshot intact; commit invalidates the snapshot, not buffers.
    [[nodiscard]] Result<std::unique_ptr<RenderSubsystemShaderReload>> prepareShaderReload(
        Device& device,
        std::string& log);
    void clear(RenderSubsystemHost* host = nullptr);

private:
    struct Resources;
    class Publication;
    class ShaderReload;
    ComputeKernel program_;
    // Canonical committed bytecode; failed/discarded reloads do not change the active kernel.
    std::vector<uint32_t> programSpirv_;
    std::shared_ptr<Resources> resources_;
    std::vector<std::shared_ptr<Resources>> resourcePool_;
    std::shared_ptr<SubmissionTransaction> publication_;
    ClusterLightGridSnapshot snapshot_;
    uint64_t nextBuildRevision_ = 1;
};

} // namespace metallic::render

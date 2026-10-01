#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

struct SceneAccelerationStructureBuildOptions {
    // Explicit choice: an unavailable Partitioned backend returns Unsupported.
    RayTracingTopLevelBackend topLevelBackend = RayTracingTopLevelBackend::Standard;
    bool asyncComputePreferred = true;
};

struct SceneAccelerationStructureStats {
    uint32_t blasCount = 0;
    uint32_t instanceCount = 0;
    uint64_t triangleCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t geometryBytes = 0;
    // Final resident BLAS, TLAS, and OMM bytes once the build is Ready.
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
    // BLAS-only compaction accounting. During Phase A, compactedBlasBytes is
    // zero; it becomes the final mixed compact/original resident size when
    // Phase B is submitted.
    uint64_t originalBlasBytes = 0;
    uint64_t compactedBlasBytes = 0;
    uint64_t compactionSavedBytes = 0;
    // Peak simultaneous AS bytes, excluding geometry, instances, and scratch.
    uint64_t peakAccelerationStructureBytes = 0;
    uint32_t opacityMicromapCount = 0;
    uint64_t opacityMicromapTriangleCount = 0;
    uint64_t opacityMicromapBytes = 0;
    RayTracingTopLevelBackend topLevelBackend = RayTracingTopLevelBackend::Standard;
    uint64_t topLevelBytes = 0;
    uint32_t partitionCount = 0;
    uint32_t maxInstancesPerPartition = 0;
    uint64_t operationBytes = 0;
};

enum class SceneAccelerationStructureBuildState : uint8_t {
    Idle,
    Building,
    Ready,
    Failed,
};

class SceneAccelerationStructureBuilder {
public:
    SceneAccelerationStructureBuilder();
    ~SceneAccelerationStructureBuilder();

    SceneAccelerationStructureBuilder(SceneAccelerationStructureBuilder&&) noexcept;
    SceneAccelerationStructureBuilder& operator=(SceneAccelerationStructureBuilder&&) noexcept;

    SceneAccelerationStructureBuilder(const SceneAccelerationStructureBuilder&) = delete;
    SceneAccelerationStructureBuilder& operator=(const SceneAccelerationStructureBuilder&) = delete;

    Result<> build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log,
        const SceneAccelerationStructureBuildOptions& options = {});
    Result<> beginBuild(Device& device, Queue& queue, const scene::Scene& scene, std::string& log,
        const SceneAccelerationStructureBuildOptions& options = {});
    [[nodiscard]] Result<bool> pollBuild(std::string& log);
    SceneAccelerationStructureBuildState buildState() const;
    Result<> updateInstanceTransforms(
        Device& device,
        Queue& queue,
        const scene::Scene& scene,
        std::string& log);
    // The graph owns AS build/read ordering. Preparation only creates a fresh
    // immutable upload; recording retains it until its submission completes.
    Result<> prepareInstanceTransformUpdate(Device& device, const scene::Scene& scene, std::string& log);
    Result<> recordInstanceTransformUpdate(CommandBuffer& commands, std::string& log,
        bool graphManagedSynchronization = true);
    bool hasPendingInstanceTransformUpdate() const;
    Buffer* instanceTransformUpdateBuffer() const;
    Buffer* instanceTransformUpdateScratchBuffer() const;
    void clear();

    bool valid() const;
    RayTracingAccelerationStructure* accelerationStructure() const;
    const SceneAccelerationStructureStats& stats() const;

private:
    struct Impl;
    Result<> buildInternal(
        Device& device,
        Queue& queue,
        const scene::Scene& scene,
        const SceneAccelerationStructureBuildOptions& options,
        bool waitForCompletion,
        std::string& log);
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render

#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

struct SceneClusterAccelerationStructureStats {
    uint32_t clasCount = 0;
    uint32_t clusterBlasCount = 0;
    uint32_t instanceCount = 0;
    uint64_t clusterTriangleCount = 0;
    uint64_t clusterVertexCount = 0;
    uint64_t clusterIndexBytes = 0;
    uint64_t selectedClusterReferenceCount = 0;
    uint64_t geometryBytes = 0;
    uint64_t clasBytes = 0;
    uint64_t clusterBlasBytes = 0;
    uint64_t tlasBytes = 0;
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
};

struct ScenePartitionedAccelerationStructureStats {
    uint32_t blasCount = 0;
    uint32_t instanceCount = 0;
    uint32_t partitionCount = 0;
    uint32_t maxInstancesPerPartition = 0;
    uint64_t triangleCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t geometryBytes = 0;
    uint64_t blasBytes = 0;
    uint64_t ptlasBytes = 0;
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
    uint64_t operationBytes = 0;
};

class SceneClusterAccelerationStructureBuilder {
public:
    SceneClusterAccelerationStructureBuilder();
    ~SceneClusterAccelerationStructureBuilder();

    SceneClusterAccelerationStructureBuilder(
        SceneClusterAccelerationStructureBuilder&&) noexcept;
    SceneClusterAccelerationStructureBuilder& operator=(
        SceneClusterAccelerationStructureBuilder&&) noexcept;

    SceneClusterAccelerationStructureBuilder(
        const SceneClusterAccelerationStructureBuilder&) = delete;
    SceneClusterAccelerationStructureBuilder& operator=(
        const SceneClusterAccelerationStructureBuilder&) = delete;

    Result build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    void clear();

    bool valid() const;
    RayTracingAccelerationStructure* accelerationStructure() const;
    const SceneClusterAccelerationStructureStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ScenePartitionedAccelerationStructureBuilder {
public:
    ScenePartitionedAccelerationStructureBuilder();
    ~ScenePartitionedAccelerationStructureBuilder();

    ScenePartitionedAccelerationStructureBuilder(
        ScenePartitionedAccelerationStructureBuilder&&) noexcept;
    ScenePartitionedAccelerationStructureBuilder& operator=(
        ScenePartitionedAccelerationStructureBuilder&&) noexcept;

    ScenePartitionedAccelerationStructureBuilder(
        const ScenePartitionedAccelerationStructureBuilder&) = delete;
    ScenePartitionedAccelerationStructureBuilder& operator=(
        const ScenePartitionedAccelerationStructureBuilder&) = delete;

    Result build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    void clear();

    bool valid() const;
    PartitionedAccelerationStructure* accelerationStructure() const;
    const ScenePartitionedAccelerationStructureStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render

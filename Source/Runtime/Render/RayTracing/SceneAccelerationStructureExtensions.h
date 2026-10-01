#pragma once

#include "Runtime/Render/GAPI/RHI.h"
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

    Result<> build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    void clear();

    bool valid() const;
    RayTracingAccelerationStructure* accelerationStructure() const;
    const SceneClusterAccelerationStructureStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render

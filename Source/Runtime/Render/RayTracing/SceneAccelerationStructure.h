#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

struct SceneAccelerationStructureStats {
    uint32_t blasCount = 0;
    uint32_t instanceCount = 0;
    uint64_t triangleCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t geometryBytes = 0;
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
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

    Result build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    Result beginBuild(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    bool pollBuild();
    SceneAccelerationStructureBuildState buildState() const;
    Result updateInstanceTransforms(
        Device& device,
        Queue& queue,
        const scene::Scene& scene,
        std::string& log);
    void clear();

    bool valid() const;
    RayTracingAccelerationStructure* accelerationStructure() const;
    const SceneAccelerationStructureStats& stats() const;

private:
    struct Impl;
    Result buildInternal(
        Device& device,
        Queue& queue,
        const scene::Scene& scene,
        bool waitForCompletion,
        std::string& log);
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render

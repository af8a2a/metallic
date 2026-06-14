#pragma once

#include "Runtime/Render/GAPI/rhi.h"
#include "Runtime/Scene/scene.h"

#include <volk.h>

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render::vulkan {

struct SceneRtxStats {
    uint32_t blasCount = 0;
    uint32_t instanceCount = 0;
    uint64_t triangleCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t geometryBytes = 0;
    uint64_t accelerationStructureBytes = 0;
    uint64_t scratchBytes = 0;
};

class SceneRtxBuilder {
public:
    SceneRtxBuilder();
    ~SceneRtxBuilder();

    SceneRtxBuilder(SceneRtxBuilder&&) noexcept;
    SceneRtxBuilder& operator=(SceneRtxBuilder&&) noexcept;

    SceneRtxBuilder(const SceneRtxBuilder&) = delete;
    SceneRtxBuilder& operator=(const SceneRtxBuilder&) = delete;

    Result build(Device& device, Queue& queue, const scene::Scene& scene, std::string& log);
    void clear();

    bool valid() const;
    VkAccelerationStructureKHR tlas() const;
    VkDeviceAddress tlasDeviceAddress() const;
    const SceneRtxStats& stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render::vulkan

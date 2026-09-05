#pragma once

#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Scene/Scene.h"

namespace metallic::render {

// Shared C++/Slang ABI. Element zero contains count and exposure; lights start at 1.
struct GpuPunctualLight {
    float positionRange[4] = {};
    float directionType[4] = {};
    float colorIntensity[4] = {};
    float spot[4] = {};
};
static_assert(sizeof(GpuPunctualLight) == 64);

std::vector<GpuPunctualLight> buildPunctualLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& settings);

class SceneLightResources {
public:
    Result update(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
        const scene::Scene* scene, const scene::LightingSettings& settings);
    Buffer* buffer() const { return buffer_.get(); }
    uint64_t revision() const { return revision_; }

private:
    std::vector<GpuPunctualLight> records_;
    std::shared_ptr<Buffer> buffer_;
    uint64_t revision_ = 0;
};

} // namespace metallic::render

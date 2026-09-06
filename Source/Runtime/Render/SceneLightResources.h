#pragma once

#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Scene/Scene.h"
#include "Runtime/Scene/SceneLighting.h"

#include <cstdint>
#include <span>
#include <vector>

namespace metallic::render {

// Shared C++/Slang ABI. Element zero contains count and exposure; lights start at 1.
struct GpuPunctualLight {
    float positionRange[4] = {};
    float directionType[4] = {};
    float colorIntensity[4] = {};
    float spot[4] = {};
};
static_assert(sizeof(GpuPunctualLight) == 64);

// Stable source slots: imported lights first, then virtual world lights. Inactive
// or invalid sources retain their slot and provenance, with enabled=false.
struct SceneLightRecord {
    GpuPunctualLight gpu;
    int32_t sourceRenderLightIndex = scene::kInvalidSceneIndex;
    int32_t sourceVirtualLightIndex = scene::kInvalidSceneIndex;
    scene::SceneEntity sourceObject = scene::kNullSceneEntity;
    bool enabled = false;
};

std::vector<SceneLightRecord> buildSceneLightRecords(
    std::span<const scene::RenderLight> renderLights,
    std::span<const scene::PunctualLight> virtualLights);

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

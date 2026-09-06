#pragma once

#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Render/ReGIR.h"
#include "Runtime/Scene/Scene.h"
#include "Runtime/Scene/SceneLighting.h"

#include <cstdint>
#include <span>
#include <vector>

namespace metallic::render {

class RenderWorld;

// A resolved/override scene owns its authored imports. Only independent manual
// world lights may accompany a different scene; same-scene world edits win.
scene::LightingSettings resolveSceneLighting(const scene::Scene* actualScene, const RenderWorld* world);

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
// Document-owned imports emit only from their native virtual slot, whose pose
// and visibility are resolved against the matching current RenderLight source.
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
    // Camera-independent full-scene proposal: never use DrawSet/LightGrid's
    // camera-filtered candidates for secondary path vertices.
    Result buildSampling(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
        TextureView& environment, const ReGIRBuildParameters& parameters,
        uint32_t gridSize, uint32_t lightsPerCell, bool buildGrid, std::string& log);
    Buffer* buffer() const { return buffer_.get(); }
    Buffer* reGIRBuffer() const;
    TextureView* lightPdfView() const;
    uint32_t lightCount() const { return records_.empty() ? 0u : static_cast<uint32_t>(records_.size() - 1u); }
    uint64_t revision() const { return revision_; }

private:
    struct SamplingState;
    std::shared_ptr<SamplingState> sampling_;
    std::vector<GpuPunctualLight> records_;
    std::shared_ptr<Buffer> buffer_;
    uint64_t revision_ = 0;
};

} // namespace metallic::render

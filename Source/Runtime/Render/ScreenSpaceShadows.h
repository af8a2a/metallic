#pragma once

#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderView.h"
#include "Runtime/Render/SceneLightResources.h"

namespace metallic::render {

class ScenePathTraceResources;

struct ScreenSpaceShadowSettings {
    bool enabled = true;
    bool denoise = true;
    bool debug = false;
    int32_t lightIndex = -1; // Stable scene light slot; -1 prefers a directional light.
    uint32_t historyLength = 5;
    float maxDistance = 100000.0f;
    float normalBias = 0.01f;
    float angularRadiusDegrees = 0.266f;
    float lightRadius = 0.05f;
    bool operator==(const ScreenSpaceShadowSettings&) const = default;
};

struct ScreenSpaceShadowParameters {
    ViewConstants view;
    GpuPunctualLight light;
    float trace[4]{}; // ray length, reserved, normal bias, tan(angular radius)
    uint32_t control[4]{}; // enabled, reserved, stable light slot, debug
    float shape[4]{}; // local light radius, denoise enabled, reserved, reserved
};
static_assert(sizeof(ScreenSpaceShadowParameters) == 320);

struct ScreenSpaceShadowResult {
    Texture* texture = nullptr;
    TextureView* shadow = nullptr; // SIGMA encoded visibility; unpack by squaring.
    Buffer* parameters = nullptr;
};

// Header followed by stable GPUScene source slots, including disabled slots.
std::vector<GpuPunctualLight> buildScreenSpaceShadowLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& lighting);
// Invalid/disabled requested slots use the same automatic selection as -1.
uint32_t selectScreenSpaceShadowLight(std::span<const GpuPunctualLight> lights, int32_t requestedIndex);

// Full TLAS shadow tracing; the legacy C++ name is retained for existing callers.
// One shadow history for the selected punctual light. The owner serializes frames.
class ScreenSpaceShadows {
public:
    Result record(Device& device, CommandBuffer& commands, Streamer& streamer,
        TextureView& depth, const ViewConstants& view, std::span<const GpuPunctualLight> lights,
        uint64_t sceneRevision, uint64_t transformRevision, const ScreenSpaceShadowSettings& settings,
        ScreenSpaceShadowResult& output, std::string& log, ScenePathTraceResources* geometry);
    void clear();

private:
    struct State;
    std::shared_ptr<State> state_;
    std::array<ComputeProgram, 3> traces_; // conventional textures, NTC, NTC cooperative vector
};

} // namespace metallic::render

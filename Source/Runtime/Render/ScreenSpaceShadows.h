#pragma once

#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/Environment/CelestialLighting.h"

namespace metallic::render {

class ScenePathTraceResources;
struct MeshletStreamDeferredGPUResourcesView;
struct CPUProfileRecorder;

struct ScreenSpaceShadowSettings {
    bool enabled = true;
    bool denoise = true;
    bool debug = false;
    int32_t lightIndex = -1; // Sun=0, Moon=1, local source slots follow; -1 prefers Sun/Moon.
    uint32_t historyLength = 5;
    float maxDistance = 100000.0f;
    float normalBias = 0.01f;
    float lightRadius = 0.05f;
    bool operator==(const ScreenSpaceShadowSettings&) const = default;
};

struct ScreenSpaceShadowParameters {
    ViewConstants view;
    GPUPunctualLight light;
    float trace[4]{}; // ray length, reserved, normal bias, tan(angular radius)
    uint32_t control[4]{}; // enabled, celestial domain, tagged source identity, debug
    float shape[4]{}; // local light radius, denoise enabled, reserved, reserved
};
static_assert(sizeof(ScreenSpaceShadowParameters) == 320);

struct ScreenSpaceShadowResult {
    Texture* texture = nullptr;
    TextureView* shadow = nullptr; // SIGMA encoded visibility; unpack by squaring.
    Buffer* parameters = nullptr;
};

struct ScreenSpaceShadowLightRecord {
    GPUCelestialLight celestial;
    GPUPunctualLight local;
    uint32_t sourceIndex = UINT32_MAX;
    bool isCelestial = false;
    bool enabled = false;
};

// Fixed Sun/Moon slots followed by stable local GPUScene source slots, including
// disabled locals. The selected celestial is adapted only at shadow dispatch.
std::vector<ScreenSpaceShadowLightRecord> buildScreenSpaceShadowLightRecords(
    const scene::Scene* scene, const scene::LightingSettings& lighting,
    const environment::EnvironmentSnapshot& environment);
// Invalid/disabled requested slots use the same automatic selection as -1.
uint32_t selectScreenSpaceShadowLight(std::span<const ScreenSpaceShadowLightRecord> lights, int32_t requestedIndex);

// Full TLAS shadow tracing; the legacy C++ name is retained for existing callers.
// One shadow history for the selected celestial/local light. The owner serializes frames.
class ScreenSpaceShadows {
public:
    [[nodiscard]] Result<ScreenSpaceShadowResult> record(
        Device& device,
        CommandBuffer& commands,
        Streamer& streamer,
        TextureView& depth,
        const ViewConstants& view,
        std::span<const ScreenSpaceShadowLightRecord> lights,
        uint64_t sceneRevision,
        uint64_t transformRevision,
        const ScreenSpaceShadowSettings& settings,
        std::string& log,
        ScenePathTraceResources* geometry,
        const MeshletStreamDeferredGPUResourcesView* streamGeometry = nullptr,
        CPUProfileRecorder* profiler = nullptr,
        RayTracingAccelerationStructure* accelerationStructure = nullptr);
    void clear();

private:
    struct State;
    std::shared_ptr<State> state_;
    std::array<ComputeProgram, 5> traces_; // conventional, NTC, NTC cooperative, streamed TLAS, stream pending
    std::array<uint32_t, 5> traceTextureCounts_{};
};

} // namespace metallic::render

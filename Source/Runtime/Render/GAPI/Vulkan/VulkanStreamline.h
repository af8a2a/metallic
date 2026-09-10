#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <cstdint>
#include <string>

namespace metallic::render::vulkan {

enum class StreamlineReflexMode : uint32_t {
    Off,
    On,
    Boost,
};

struct StreamlineReflexOptions {
    StreamlineReflexMode mode = StreamlineReflexMode::On;
    uint32_t frameLimitUs = 0;
    bool operator==(const StreamlineReflexOptions&) const = default;
};

struct StreamlineReflexStatus {
    bool available = false;
    bool suspended = false;
    bool latencyReportAvailable = false;
    StreamlineReflexOptions options;
    uint64_t reportFrameId = 0;
    double renderLatencyMs = 0.0; // Simulation start to GPU render end; excludes display latency.
    double gpuRenderMs = 0.0;
};

enum class StreamlineLatencyMarker : uint32_t {
    SimulationEnd,
    RenderSubmitStart,
    RenderSubmitEnd,
    PresentStart,
    PresentEnd,
};

StreamlineReflexStatus streamlineReflexStatus();
Result setStreamlineReflexOptions(const StreamlineReflexOptions& options);
void setStreamlineLatencyMarker(StreamlineLatencyMarker marker);

// One scope per application frame, before input polling. DLSS evaluations inside
// the scope share its token. Offscreen callers can continue evaluating without it.
// Must be externally serialized with device lifetime and other frame scopes.
class StreamlineFrameScope {
public:
    explicit StreamlineFrameScope(bool allowLatency = true);
    ~StreamlineFrameScope();
    StreamlineFrameScope(const StreamlineFrameScope&) = delete;
    StreamlineFrameScope& operator=(const StreamlineFrameScope&) = delete;

private:
    bool active_ = false;
};

struct StreamlineDlssRrTextureRef {
    Texture* texture = nullptr;
    TextureView* view = nullptr;
};

enum class StreamlineDlssRrMode : uint32_t {
    Off,
    Dlaa,
    Quality,
    Balanced,
    Performance,
    UltraPerformance,
    UltraQuality,
};

struct StreamlineDlssRrOptimalSettings {
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t renderWidthMin = 0;
    uint32_t renderHeightMin = 0;
    uint32_t renderWidthMax = 0;
    uint32_t renderHeightMax = 0;
};

struct StreamlineDlssRrCamera {
    float eye[3] = {0.0f, 0.0f, 0.0f};
    float center[3] = {0.0f, 0.0f, -1.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float previousEye[3] = {0.0f, 0.0f, 0.0f};
    float previousCenter[3] = {0.0f, 0.0f, -1.0f};
    float previousUp[3] = {0.0f, 1.0f, 0.0f};
    float fovRadians = 0.87266463f;
    float previousFovRadians = 0.87266463f;
    float aspectRatio = 1.0f;
    float previousAspectRatio = 1.0f;
    float zNear = 0.001f;
    float previousZNear = 0.001f;
    float zFar = 10000.0f;
    float previousZFar = 10000.0f;
    float orthoHeight = 1.0f;
    float previousOrthoHeight = 1.0f;
    float jitterOffset[2] = {0.0f, 0.0f};
    bool orthographic = false;
    bool previousOrthographic = false;
    bool previousValid = false;
};

struct StreamlineDlssRrDesc {
    StreamlineDlssRrTextureRef inputColor;
    StreamlineDlssRrTextureRef outputColor;
    StreamlineDlssRrTextureRef albedo;
    StreamlineDlssRrTextureRef specularAlbedo;
    StreamlineDlssRrTextureRef normalRoughness;
    StreamlineDlssRrTextureRef motionVectors;
    StreamlineDlssRrTextureRef linearDepth;
    StreamlineDlssRrTextureRef specularHitDistance;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    StreamlineDlssRrCamera camera;
    StreamlineDlssRrMode mode = StreamlineDlssRrMode::Balanced;
    bool reset = false;
};

using StreamlineDlssSrTextureRef = StreamlineDlssRrTextureRef;
using StreamlineDlssSrMode = StreamlineDlssRrMode;
using StreamlineDlssSrOptimalSettings = StreamlineDlssRrOptimalSettings;
using StreamlineDlssSrCamera = StreamlineDlssRrCamera;

struct StreamlineDlssSrDesc {
    StreamlineDlssSrTextureRef inputColor;
    StreamlineDlssSrTextureRef outputColor;
    StreamlineDlssSrTextureRef motionVectors;
    StreamlineDlssSrTextureRef depth;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    StreamlineDlssSrCamera camera;
    StreamlineDlssSrMode mode = StreamlineDlssSrMode::Balanced;
    bool reset = false;
};

const char* streamlineVulkanLibraryName();
bool streamlineSdkAvailable();
bool streamlineInitialized();
bool streamlineDlssSrSupported();
bool streamlineDlssRrSupported();
Result getStreamlineDlssSrOptimalSettings(
    StreamlineDlssSrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssSrOptimalSettings& settings,
    std::string& log);
Result getStreamlineDlssRrOptimalSettings(
    StreamlineDlssRrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssRrOptimalSettings& settings,
    std::string& log);
Result initializeStreamlinePreDevice(std::string& log);
Result setStreamlineVulkanDevice(
    const NativeDevice& device,
    const NativeQueue& graphicsQueue,
    const NativeQueue& computeQueue,
    std::string& log);
void shutdownStreamline();
// Switch from the engine's descriptor heap to legacy NGX descriptors, including
// direct experimental features that share Streamline's Vulkan device.
void prepareStreamlineNgxCommandBuffer(CommandBuffer& commandBuffer);
Result evaluateStreamlineDlssSr(CommandBuffer& commandBuffer, const StreamlineDlssSrDesc& desc, std::string& log);
Result evaluateStreamlineDlssRr(CommandBuffer& commandBuffer, const StreamlineDlssRrDesc& desc, std::string& log);

} // namespace metallic::render::vulkan

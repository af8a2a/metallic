#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"

#include <array>
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
    // Cached driver report, same reportFrameId as above. Zero raw timestamps or
    // durations can mean unavailable; these are not current-frame GPU queries.
    uint64_t simulationStartUs = 0;
    uint64_t simulationEndUs = 0;
    uint64_t renderSubmitStartUs = 0;
    uint64_t renderSubmitEndUs = 0;
    uint64_t presentStartUs = 0;
    uint64_t presentEndUs = 0;
    uint64_t gpuRenderStartUs = 0;
    uint64_t gpuRenderEndUs = 0;
    uint32_t gpuActiveRenderTimeUs = 0;
    uint32_t gpuFrameTimeUs = 0;
};

enum class StreamlineLatencyMarker : uint32_t {
    SimulationEnd,
    RenderSubmitStart,
    RenderSubmitEnd,
    PresentStart,
    PresentEnd,
};

StreamlineReflexStatus streamlineReflexStatus();
Result<> setStreamlineReflexOptions(const StreamlineReflexOptions& options);
void setStreamlineLatencyMarker(StreamlineLatencyMarker marker);

// Offscreen evaluations do not pass through vkQueuePresentKHR. Advance the
// common plugin's frame bookkeeping once after submitting each such frame.
// Never call this in addition to an actual interposed presentation.
Result<> notifyStreamlineOffscreenFrame();

// Optional CPU wall timings for frame-begin diagnostics. The cached driver report
// is refreshed at the normal cadence, not queried again for instrumentation.
struct StreamlineFrameBeginProfile {
    double totalMs = 0.0;
    double mutexWaitMs = 0.0;
    double tokenMs = 0.0;
    double optionsMs = 0.0;
    double sleepMs = 0.0;
    double statusMs = 0.0;
    double markerMs = 0.0;
    uint64_t frameId = 0;
    bool active = false;
    bool sleepCalled = false;
    bool optionsUpdated = false;
    bool statusRefreshed = false;
    StreamlineReflexOptions effectiveOptions;
    StreamlineReflexStatus cachedStatus;
};

// One scope per application frame, before input polling. DLSS evaluations inside
// the scope share its token. Offscreen callers can continue evaluating without it.
// Must be externally serialized with device lifetime and other frame scopes.
class StreamlineFrameScope {
public:
    explicit StreamlineFrameScope(bool allowLatency = true, StreamlineFrameBeginProfile* profile = nullptr);
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

// Value-only snapshots: no SDK calls or retained GPU resource pointers in the UI.
struct StreamlineDebugResource {
    const char* name = "";
    bool bound = false;
    uint32_t width = 0;
    uint32_t height = 0;
    Format format = Format::Unknown;
};

struct StreamlineDlssDebugStatus {
    uint64_t attempts = 0;
    uint64_t successes = 0;
    uint32_t frameIndex = 0;
    StreamlineDlssRrMode mode = StreamlineDlssRrMode::Off;
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t outputWidth = 0;
    uint32_t outputHeight = 0;
    StreamlineDlssRrCamera camera;
    bool reset = false;
    bool succeeded = false;
    double cpuMs = 0.0; // CPU evaluation wall time, not GPU execution time.
    double ageSeconds = 0.0;
    std::string message;
    std::array<StreamlineDebugResource, 8> resources{};
    uint32_t resourceCount = 0;
};

struct StreamlineDebugStatus {
    bool sdkAvailable = false;
    bool initialized = false;
    bool deviceSet = false;
    bool dlssSrSupported = false;
    bool dlssRrSupported = false;
    bool descriptorHeapWorkaround = false;
    std::string sdkVersion;
    uint32_t frameIndex = 0;
    StreamlineReflexStatus reflex;
    StreamlineDlssDebugStatus sr;
    StreamlineDlssDebugStatus rr;
};

StreamlineDebugStatus streamlineDebugStatus();
const char* streamlineVulkanLibraryName();
bool streamlineSdkAvailable();
bool streamlineInitialized();
bool streamlineDlssSrSupported();
bool streamlineDlssRrSupported();
Result<> getStreamlineDlssSrOptimalSettings(
    StreamlineDlssSrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssSrOptimalSettings& settings,
    std::string& log);
Result<> getStreamlineDlssRrOptimalSettings(
    StreamlineDlssRrMode mode,
    uint32_t outputWidth,
    uint32_t outputHeight,
    StreamlineDlssRrOptimalSettings& settings,
    std::string& log);
Result<> initializeStreamlinePreDevice(std::string& log);
Result<> setStreamlineVulkanDevice(
    const NativeDevice& device,
    const NativeQueue& graphicsQueue,
    const NativeQueue& computeQueue,
    std::string& log);
void shutdownStreamline();
// Switch from the engine's descriptor heap to legacy NGX descriptors, including
// direct experimental features that share Streamline's Vulkan device.
void prepareStreamlineNgxCommandBuffer(CommandBuffer& commandBuffer);
Result<> evaluateStreamlineDlssSr(CommandBuffer& commandBuffer, const StreamlineDlssSrDesc& desc, std::string& log);
Result<> evaluateStreamlineDlssRr(CommandBuffer& commandBuffer, const StreamlineDlssRrDesc& desc, std::string& log);

} // namespace metallic::render::vulkan

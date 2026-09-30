#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>

namespace metallic::render {
class Queue;
class Texture;
} // namespace metallic::render

namespace metallic::render::profiling {

enum class NsightCaptureMode : uint8_t {
    Default,
    GPUTrace,
    GraphicsCapture,
};

// For a process launched by ngfx GPU Trace with SDK start/stop triggers.
// The caller drains all submitted GPU work before either boundary. These do
// not self-inject, and cannot coexist with Graphics Capture in one process.
// Call both functions on the same owner thread.
bool beginExternalNsightGpuTrace(std::string& error);
bool endExternalNsightGpuTrace(std::string& error);

enum class NsightGraphicsCaptureState : uint8_t {
    Unavailable,
    Uninitialized,
    Ready,
    CapturePending,
    CaptureCompleted,
    Error,
};

struct NsightGraphicsCaptureConfig {
    std::filesystem::path installationRoot;
    std::filesystem::path outputDirectory;
    bool showHud = true;
    NsightCaptureMode mode = NsightCaptureMode::GraphicsCapture;
};

struct NsightGraphicsCaptureRequest {
    uint32_t framesBeforeStart = 0;
    uint32_t framesToCapture = 1;
    // Offscreen tests delimit frames explicitly instead of presenting a window.
    bool explicitFrameBoundaries = false;
};

struct NsightGraphicsCapturePollResult {
    NsightGraphicsCaptureState state = NsightGraphicsCaptureState::Unavailable;
    std::filesystem::path capturePath;
    std::string message;
};

// Nsight Graphics owns process-global injection state. Use one instance and call
// every method from the same thread. initializeBeforeGraphics() must run before
// creating the Vulkan instance or any other graphics context.
class NsightGraphicsCapture final {
public:
    NsightGraphicsCapture();
    ~NsightGraphicsCapture();
    NsightGraphicsCapture(const NsightGraphicsCapture&) = delete;
    NsightGraphicsCapture& operator=(const NsightGraphicsCapture&) = delete;

    static bool compiledAvailable();
    // Includes SDK injection and an externally loaded Nsight capture interceptor.
    static bool vulkanInjectionActive();
    static std::filesystem::path defaultInstallationRoot();

    bool initializeBeforeGraphics(const NsightGraphicsCaptureConfig& config, std::string& error);
    bool requestCapture(const NsightGraphicsCaptureRequest& request, std::string& error);
    bool frameBoundary(Queue& queue, Texture* output, std::string& error);
    // Called after the main viewport Present (not secondary ImGui windows).
    bool afterPresent(Queue& queue, std::string& error);
    NsightGraphicsCapturePollResult poll();
    NsightCaptureMode mode() const { return mode_; }

    NsightGraphicsCaptureState state() const;
    const char* statusText() const;
    bool hasOutstandingCapture() const;
    const std::filesystem::path& capturePath() const;
    const std::string& lastError() const;

private:
    bool fail(std::string message, std::string* error = nullptr);

    NsightGraphicsCaptureState state_ = NsightGraphicsCaptureState::Unavailable;
    std::filesystem::path installationRoot_;
    std::filesystem::path outputDirectory_;
    std::filesystem::path capturePath_;
    std::string outputDirectoryUtf8_;
    std::string lastError_;
    uint32_t pendingCaptureIndex_ = 0;
    NsightCaptureMode mode_ = NsightCaptureMode::GraphicsCapture;
    uint32_t traceFramesBeforeStart_ = 0;
    uint32_t traceFramesRemaining_ = 0;
    bool traceStarted_ = false;
    bool traceStopped_ = false;
    std::chrono::steady_clock::time_point traceDeadline_;
    void* traceHostProcess_ = nullptr;
    void* traceHostJob_ = nullptr;
    std::filesystem::path traceHostLog_;
    bool startTraceHost(std::string& error);
};

} // namespace metallic::render::profiling

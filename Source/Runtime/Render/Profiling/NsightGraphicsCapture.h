#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

namespace metallic::render::profiling {

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
};

struct NsightGraphicsCaptureRequest {
    uint32_t framesBeforeStart = 0;
    uint32_t framesToCapture = 1;
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

    static bool compiledAvailable();
    static std::filesystem::path defaultInstallationRoot();

    bool initializeBeforeGraphics(const NsightGraphicsCaptureConfig& config, std::string& error);
    bool requestCapture(const NsightGraphicsCaptureRequest& request, std::string& error);
    NsightGraphicsCapturePollResult poll();

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
};

} // namespace metallic::render::profiling

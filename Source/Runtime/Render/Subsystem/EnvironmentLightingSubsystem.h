#pragma once

#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace metallic::render {

enum class EnvironmentLightingStatus : uint8_t {
    Uninitialized,
    Loading,
    Ready,
    Degraded,
};

struct EnvironmentLightingSnapshot {
    EnvironmentSettings settings;
    EnvironmentLightingStatus status = EnvironmentLightingStatus::Uninitialized;
    TextureView* radianceView = nullptr;
    TextureView* pdfView = nullptr;
    Buffer* sphericalHarmonicsBuffer = nullptr;
    uint32_t width = 1;
    uint32_t height = 1;
    uint64_t settingsRevision = 0;
    uint64_t resourceRevision = 0;
    bool mapAvailable = false;
    std::string error;

    bool valid() const
    {
        return radianceView != nullptr &&
            pdfView != nullptr &&
            sphericalHarmonicsBuffer != nullptr;
    }
};

class EnvironmentLightingSubsystem final : public IRenderSubsystem {
public:
    struct Desc {
        uint32_t maxDecodeJobs = 2;
    };

    static constexpr RenderSubsystemId kSubsystemId = "render.environment";

    EnvironmentLightingSubsystem();
    ~EnvironmentLightingSubsystem() override;

    Result initialize(const RenderSubsystemInitContext& context, std::string& log) override;
    void onWorldChanged(RenderWorld* world) override;
    Result beginFrame(
        const RenderSubsystemFrameContext& context,
        RenderChangeBits& changes,
        std::string& log) override;
    Result recordPreGraph(const RenderSubsystemFrameContext& context, std::string& log) override;
    void shutdown() override;

    const EnvironmentLightingSnapshot& snapshot() const { return snapshot_; }
    uint64_t decodeCount() const { return decodeCount_; }

private:
    struct DecodedEnvironment;
    struct DecodeJob;
    struct GpuPrecompute;
    struct Resources;

    void requestEnvironment(const EnvironmentSettings& settings, uint64_t settingsRevision);
    void startDecodeJob(const std::filesystem::path& path, uint64_t generation);
    void pollDecodeJobs(RenderChangeBits& changes);
    Result publishDecoded(
        const RenderSubsystemFrameContext& context,
        DecodedEnvironment decoded,
        std::string& log);
    void refreshSnapshot();

    Device* device_ = nullptr;
    RenderWorld* world_ = nullptr;
    Desc desc_;
    ImportancePdfCompute pdfCompute_;
    std::unique_ptr<GpuPrecompute> gpuPrecompute_;
    std::shared_ptr<Resources> resources_;
    std::vector<DecodeJob> decodeJobs_;
    std::filesystem::path pendingDecodePath_;
    uint64_t pendingDecodeGeneration_ = 0;
    std::unique_ptr<DecodedEnvironment> readyDecode_;
    EnvironmentLightingSnapshot snapshot_;
    EnvironmentSettings requestedSettings_;
    uint64_t requestedSettingsRevision_ = 0;
    uint64_t requestedGeneration_ = 0;
    uint64_t resourceRevision_ = 0;
    uint64_t decodeCount_ = 0;
    bool requestInitialized_ = false;
};

} // namespace metallic::render

#pragma once

#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <cstdint>
#include <array>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
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
    // Nine RGB irradiance SH coefficients (l <= 2), cosine-convolved on GPU.
    // Evaluate at the normal, multiply by diffuse albedo / pi exactly once.
    Buffer* sphericalHarmonicsBuffer = nullptr;
    // Eight 256x128 lat-long layers, linear perceptual roughness, GGX filtered.
    Buffer* prefilteredSpecularBuffer = nullptr;
    Buffer* celestialLightsBuffer = nullptr;
    Buffer* atmosphereParametersBuffer = nullptr;
    TextureView* transmittanceView = nullptr;
    TextureView* multiScatteringView = nullptr;
    TextureView* skyView = nullptr;
    Buffer* aerialPerspectiveBuffer = nullptr;
    environment::EnvironmentSource source = environment::EnvironmentSource::HDRI;
    // A resolved provider snapshot owns its immutable publication, independently
    // of later source switches and cache eviction.
    std::shared_ptr<void> retainedResources;
    uint64_t celestialResourceRevision = 0;
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

struct CelestialLightingResources {
    std::shared_ptr<Buffer> buffer;
    uint64_t revision = 0;
};

class EnvironmentLightingSubsystem final : public IRenderSubsystem {
public:
    struct Desc {
        uint32_t maxDecodeJobs = 2;
        // Opt-in deterministic capture: await the first decode before recording
        // any placeholder frame. Zero preserves interactive asynchronous loading.
        uint32_t initialDecodeTimeoutMilliseconds = 0;
    };

    static constexpr RenderSubsystemId kSubsystemId = "render.environment";

    EnvironmentLightingSubsystem();
    ~EnvironmentLightingSubsystem() override;

    Result<> initialize(const RenderSubsystemInitContext& context, std::string& log) override;
    void onWorldChanged(RenderWorld* world) override;
    Result<> beginFrame(
        const RenderSubsystemFrameContext& context,
        RenderChangeBits& changes,
        std::string& log) override;
    Result<> recordPreGraph(const RenderSubsystemFrameContext& context, std::string& log) override;
    [[nodiscard]] Result<std::unique_ptr<RenderSubsystemShaderReload>> prepareShaderReload(
        const RenderSubsystemInitContext& context,
        std::string& log) override;
    void shutdown() override;

    const EnvironmentLightingSnapshot& snapshot() const { return snapshot_; }
    Result<CelestialLightingResources> updateCelestial(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
        const environment::EnvironmentSnapshot& environment);
    Result<EnvironmentLightingSnapshot> resolveRadiance(Device& device, CommandBuffer& commands,
        RenderSubsystemHost& host, const environment::EnvironmentSnapshot& environment,
        const std::array<double, 3>& observerWorldMetres, std::string& log);
    uint64_t decodeCount() const { return decodeCount_; }

private:
    struct DecodedEnvironment;
    struct DecodeJob;
    struct GPUPrecompute;
    struct Resources;
    struct PhysicalPublication;
    class ShaderReload;

    void requestEnvironment(const EnvironmentSettings& settings, uint64_t settingsRevision);
    void startDecodeJob(const std::filesystem::path& path, uint64_t generation);
    void pollDecodeJobs(RenderChangeBits& changes);
    Result<> publishDecoded(
        const RenderSubsystemFrameContext& context,
        const DecodedEnvironment& decoded,
        std::string& log);
    void refreshSnapshot();

    Device* device_ = nullptr;
    RenderSubsystemHost* host_ = nullptr;
    RenderWorld* world_ = nullptr;
    Desc desc_;
    ImportancePdfCompute pdfCompute_;
    std::unique_ptr<GPUPrecompute> gpuPrecompute_;
    std::shared_ptr<Resources> resources_;
    std::vector<std::shared_ptr<PhysicalPublication>> physicalPublications_;
    std::mutex physicalMutex_;
    std::shared_ptr<Buffer> celestialLightsBuffer_;
    struct CelestialPublication {
        GPUCelestialLightRecords records;
        CelestialLightingResources resources;
    };
    std::vector<CelestialPublication> celestialPublications_;
    std::mutex celestialMutex_;
    uint64_t nextCelestialResourceRevision_ = 0;
    uint64_t celestialResourceRevision_ = 0;
    std::vector<DecodeJob> decodeJobs_;
    std::filesystem::path pendingDecodePath_;
    uint64_t pendingDecodeGeneration_ = 0;
    std::shared_ptr<DecodedEnvironment> readyDecode_;
    std::shared_ptr<SubmissionTransaction> pendingPublication_;
    EnvironmentLightingSnapshot snapshot_;
    EnvironmentSettings requestedSettings_;
    uint64_t requestedSettingsRevision_ = 0;
    uint64_t requestedGeneration_ = 0;
    uint64_t resourceRevision_ = 0;
    uint64_t decodeCount_ = 0;
    bool requestInitialized_ = false;
};

} // namespace metallic::render

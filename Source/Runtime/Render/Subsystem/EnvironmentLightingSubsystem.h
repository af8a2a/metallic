#pragma once

#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/Environment/CelestialLighting.h"
#include "Runtime/Render/RenderGraph/RenderGraphResourceLifecycle.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"

#include <cstdint>
#include <array>
#include <filesystem>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
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
    // Optional filterable copy of the same publication; the buffer remains the reference.
    TextureView* prefilteredSpecularView = nullptr;
    const SamplerDesc* prefilteredSpecularSampler = nullptr;
    Buffer* celestialLightsBuffer = nullptr;
    Buffer* atmosphereParametersBuffer = nullptr;
    TextureView* transmittanceView = nullptr;
    TextureView* multiScatteringView = nullptr;
    TextureView* skyView = nullptr;
    Buffer* aerialPerspectiveBuffer = nullptr;
    TextureView* cloudShadowView = nullptr;
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

struct EnvironmentRequest {
    environment::EnvironmentSnapshot environment;
    std::array<double, 3> observerWorldMetres{0.0, 0.0, 0.0};
    uint64_t viewId = 0;
    bool needCelestial = true;
};

// Immutable per-execution environment resource bundle resolved by
// EnvironmentLightingSubsystem before pass recording begins.
struct EnvironmentResourceSet {
    LogicalResourceId logicalId;
    ResourceVersionId version;
    uint64_t contentKey = 0;
    uint64_t mediumKey = 0;
    bool cacheHit = false;
    bool mediumShared = false;
    bool producedThisFrame = false;
    bool celestialNeedsHostBarrier = false;
    EnvironmentLightingSnapshot lighting;
    CelestialLightingResources celestial;
    GPUCelestialLightRecords celestialRecords{};
    GPUCompletionPoint producerCompletion;
    uint64_t producerDedupKey = 0;
    std::vector<ExternalResourcePublication> publications;

    bool valid() const { return lighting.valid(); }
};

class EnvironmentLightingSubsystem final : public IRenderSubsystem, public IRenderGraphResourceProvider {
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
    void appendDebugBindings(const RenderSubsystemFrameContext& context, std::vector<DebugResourceBinding>& bindings) override;

    RenderSubsystemId providerSubsystemId() const override { return kSubsystemId; }
    std::optional<RenderGraphExternalDomainContract> externalDomainContract() const override;

    // Pure CPU selection/caching + GPU allocation without recording commands.
    // On cache miss, populates outProducerWork for RenderGraph scheduling.
    Result<std::shared_ptr<const EnvironmentResourceSet>> prepareEnvironmentResourceSet(
        Device& device,
        RenderSubsystemHost& host,
        const EnvironmentRequest& request,
        std::optional<RenderGraphProviderProducerWork>& outProducerWork,
        std::string& log);

    // Declarative RenderGraph provider entry point called during
    // RenderGraphPass::declareResourceRequests().
    Result<std::shared_ptr<const EnvironmentResourceSet>> declareEnvironmentRequest(
        const RenderGraphResourceRequestContext& context,
        RenderGraphResourceRequestBuilder& builder,
        const EnvironmentRequest& request,
        std::string& log);

    const EnvironmentLightingSnapshot& snapshot() const { return snapshot_; }
    Result<CelestialLightingResources> updateCelestial(Device& device, CommandBuffer& commands, RenderSubsystemHost& host,
        const environment::EnvironmentSnapshot& environment);
    Result<EnvironmentLightingSnapshot> resolveRadiance(Device& device, CommandBuffer& commands,
        RenderSubsystemHost& host, const environment::EnvironmentSnapshot& environment,
        const std::array<double, 3>& observerWorldMetres, std::string& log);
    uint64_t decodeCount() const { return decodeCount_; }
    uint64_t physicalProducerCount() const { return physicalProducerCount_; }
    uint64_t physicalCacheHitCount() const { return physicalCacheHitCount_; }
    uint64_t mediumShareCount() const { return mediumShareCount_; }
    uint64_t physicalAllocationGeneration() const { return nextPhysicalAllocationGeneration_; }
    uint64_t mediumAllocationGeneration() const { return nextMediumAllocationGeneration_; }

private:
    struct DecodedEnvironment;
    struct DecodeJob;
    struct GPUPrecompute;
    struct Resources;
    struct PhysicalPublication;
    struct MediumPublication;
    class ShaderReload;

    Result<CelestialLightingResources> prepareCelestialResources(
        Device& device,
        RenderSubsystemHost& host,
        const environment::EnvironmentSnapshot& environment,
        GPUCelestialLightRecords& outRecords,
        bool& outNewlyUploaded);
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
    std::vector<std::shared_ptr<PhysicalPublication>> pendingFramePublications_;
    std::vector<MediumPublication> mediumPublications_;
    std::mutex physicalMutex_;
    uint64_t nextPhysicalAllocationGeneration_ = 0;
    uint64_t nextMediumAllocationGeneration_ = 0;
    uint64_t nextContentRevision_ = 0;
    uint64_t nextMediumContentRevision_ = 0;
    std::unordered_map<uint64_t, uint64_t> contentRevisionByKey_;
    std::unordered_map<uint64_t, uint64_t> mediumContentRevisionByKey_;
    uint64_t physicalProducerCount_ = 0;
    uint64_t physicalCacheHitCount_ = 0;
    uint64_t mediumShareCount_ = 0;
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
    uint64_t frameSerial_ = 0;
    bool requestInitialized_ = false;
};

} // namespace metallic::render

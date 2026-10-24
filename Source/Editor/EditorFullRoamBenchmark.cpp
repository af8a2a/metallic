#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Editor/EditorApplication.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Profiling/PacingTrace.h"
#include "Editor/EditorRasterWorkload.h"
#include "Editor/EditorRayQualityCapture.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/StreamerSubsystem.h"
#include "Runtime/Render/Streamer/TextureMipTransition.h"
#include "imgui.h"
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>

namespace metallic {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

// The streaming regression snapshots live cut records only outside timed frames.
// Source residency is frozen, but demand/frontier/BLAS execution remains live.
class StreamingRegressionCapture final : public render::IRenderDebugObserver {
public:
    bool capture = false;
    bool semanticSource = false;
    RayQualityCapture* images = nullptr;
    void compiled(Json) override {}
    void beginExecution(render::Device& device, debug::DebugEvidenceStamp stamp, render::RenderSubsystemHost* host) override
    {
        device_ = &device;
        if (images) { images->beginExecution(device, stamp, host); }
    }
    void endExecution(bool success) override { if (images) { images->endExecution(success); } }
    void boundary(render::CommandBuffer& commands, std::string_view checkpoint, uint32_t index,
        std::string_view pass, std::span<const render::DebugResourceBinding> resources, const Json& values) override
    {
        if (!capture) { return; }
        if (images) { images->boundary(commands, checkpoint, index, pass, resources, values); }
        if (checkpoint != "AfterPass" || pass != "VBuffer") { return; }
        using namespace render;
        for (const auto& resource : resources) {
            if (!resource.buffer || !resource.id.starts_with("streaming.VBuffer.") ||
                copies_.contains(resource.id)) { continue; }
            const auto name = resource.id.substr(resource.id.find_last_of('.') + 1);
            if (name != "pageTable" && name != "activeHeader" && name != "activeGroups" &&
                name != "rtRepresentationHeader" && name != "rtRepresentationGroups" && name != "rtRepresentations" &&
                name != "tlasInstances" && name != "requestHeader" && name != "loadRequests" && name != "demandStats" &&
                !(semanticSource && (name == "instances" || name == "clasPageTable" ||
                    name == "fallbackBlasAddresses" || name == "rtRepresentationState"))) { continue; }
            const uint64_t bytes = resource.size ? resource.size : resource.buffer->desc().size - resource.offset;
            if (bytes > 128ull * 1024 * 1024) { throw std::runtime_error("Streaming regression snapshot exceeds bounded MiniZorah size"); }
            Copy copy;
            copy.metadata = {{"resource",resource.id},{"layout",resource.layout},{"bytes",bytes},{"metadata",resource.metadata}};
            if (!device_->createBuffer({.size = bytes, .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { copy.buffer = std::move(value); })) {
                throw std::runtime_error("Streaming regression readback allocation failed");
            }
            BufferBarrierDesc barrier{.buffer = resource.buffer,
                .before = resourceSyncScope(resource.state, PipelineStageBits::AllCommands),
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}, .range = {.offset = resource.offset, .size = bytes}};
            if (!commands.synchronize({.buffers = {&barrier, 1}}) ||
                !resource.buffer->slice({resource.offset, bytes}).and_then([&](const auto& source) {
                    return copy.buffer->slice().and_then([&](const auto& destination) { return commands.copyBuffer(source, destination); });
                })) { throw std::runtime_error("Streaming regression snapshot copy failed"); }
            std::swap(barrier.before, barrier.after);
            if (!commands.synchronize({.buffers = {&barrier, 1}})) { throw std::runtime_error("Streaming regression snapshot restore failed"); }
            copies_.emplace(resource.id, std::move(copy));
        }
    }
    Json exportAfterDrain(const std::filesystem::path& directory)
    {
        std::filesystem::create_directories(directory);
        Json result = Json::array();
        for (auto& [name, copy] : copies_) {
            copy.buffer->invalidate();
            const void* data = copy.buffer->map();
            if (!data) { throw std::runtime_error("Streaming regression snapshot map failed"); }
            std::ofstream file(directory / (name + ".bin"), std::ios::binary);
            file.write(static_cast<const char*>(data), static_cast<std::streamsize>(copy.buffer->desc().size));
            copy.buffer->unmap();
            if (!file) { throw std::runtime_error("Streaming regression snapshot export failed"); }
            copy.metadata["file"] = name + ".bin";
            result.push_back(copy.metadata);
        }
        if (result.size() != (semanticSource ? 14u : 10u)) { throw std::runtime_error("Streaming regression missing required stream snapshots"); }
        std::ofstream manifest(directory / "Buffers.json");
        manifest << result.dump(2) << '\n';
        if (!manifest) { throw std::runtime_error("Streaming regression buffer manifest failed"); }
        return result;
    }
private:
    struct Copy { std::unique_ptr<render::Buffer> buffer; Json metadata; };
    render::Device* device_ = nullptr;
    std::map<std::string, Copy> copies_;
};

Json streamSample(const render::SceneStreamingProfile& s)
{
    return {{"pass",s.passName},{"asset",s.assetPath},
        {"softwareRaster",s.softwareRasterIdentity.empty() ? Json(nullptr) : Json::parse(s.softwareRasterIdentity)},
        {"diagnosticSourceIdentity",s.diagnosticSourceIdentity.empty() ? Json(nullptr) : Json(s.diagnosticSourceIdentity)},
        {"diagnosticTextureSemanticIdentity",s.diagnosticTextureSemanticIdentity.empty() ? Json(nullptr) : Json::parse(s.diagnosticTextureSemanticIdentity)},
        {"nodeDemandC1",s.nodeDemandC1},
        {"nodeDemandC2",s.nodeDemandC2},
        {"rtDemandMode",s.rtDemandMode},{"rtHistoryPacked",s.rtHistoryPacked},{"rtLodHistoryBytes",s.rtLodHistoryBytes},
        {"rtHistorySparseWrites",s.rtHistorySparseWrites},
        {"frontierWave32",s.frontierWave32},
        {"terminalOnlyMask",s.terminalOnlyMask},{"terminalMaskDiagnostics",s.terminalMaskDiagnostics},
        {"maskShaderIdentity",{{"module",render::kMeshletStreamShaderModuleName},
            {"entryPoint",render::kMeshletStreamCooperativeBuildEntryPoint},
            {"spirvHash",s.maskShaderHash},{"workgroupSize",64u},{"subgroupSize",s.frontierSubgroupSize}}},
        {"frontierShaderIdentity",{{"module",render::kMeshletStreamShaderModuleName},
            {"entryPoint",s.frontierWave32 ? render::kMeshletStreamCooperativeBuild32EntryPoint : render::kMeshletStreamCooperativeBuildEntryPoint},
            {"spirvHash",s.frontierShaderHash},{"workgroupSize",s.frontierWave32 ? 32u : 64u},
            {"subgroupSize",s.frontierSubgroupSize}}},
        {"frame",s.frameIndex},{"feedbackFrame",s.feedbackFrame},{"generation",s.generation},
        {"gpuLoadRequests",s.gpuLoadRequests},{"geometryPendingTasks",s.geometryPendingTasks},
        {"geometryPendingPatches",s.geometryPendingPatches},
        {"geometryPublicationRevision",s.geometryPublicationRevision},{"clasPublicationRevision",s.clasPublicationRevision},
        {"lodTransitions", {{"enabled",s.lodTransitionTelemetryEnabled},{"historyBytes",s.lodTransitionHistoryBytes},
            {"demandedGroups",s.lodDemandedGroups},{"ownPageBlockedGroups",s.lodOwnPageBlockedGroups},
            {"dependencyBlockedGroups",s.lodDependencyBlockedGroups},{"catchupActivatedGroups",s.lodCatchupActivatedGroups},
            {"catchupSelectedGroups",s.lodCatchupSelectedGroups},{"catchupSelectedClusters",s.lodCatchupSelectedClusters},
            {"thresholdSelectedGroups",s.lodThresholdSelectedGroups},{"thresholdSelectedClusters",s.lodThresholdSelectedClusters},
            {"unclassifiedSelectedGroups",s.lodUnclassifiedSelectedGroups},{"unclassifiedSelectedClusters",s.lodUnclassifiedSelectedClusters}}},
        {"prefetch", {{"predictiveEnabled",s.predictivePrefetchEnabled},{"forecastActive",s.prefetchForecastActive},
            {"enabled",s.prefetchEnabled},{"memoryWatermarkBlocked",s.prefetchMemoryWatermarkBlocked},{"queueBlocked",s.prefetchQueueBlocked},
            {"horizonMilliseconds",s.prefetchHorizonMilliseconds},{"translationDistance",s.prefetchTranslationDistance},
            {"rotationDegrees",s.prefetchRotationDegrees},{"measuredLatency",s.prefetchMeasuredLatency},
            {"latencySamples",s.prefetchLatencySamples},{"demandLatencyP95Milliseconds",s.prefetchDemandLatencyP95Milliseconds},
            {"totalAdmitted",s.totalPrefetchAdmitted},{"totalUsed",s.totalPrefetchUsed},{"totalDeferred",s.totalPrefetchDeferred},
            {"gpuRequests",s.prefetchGpuRequests},{"gpuDropped",s.prefetchGpuDropped}}},
        {"geometryBytes",s.geometryUsedBytes},{"geometryCapacity",s.geometryBudgetBytes},
        {"immutableMetadata", {{"deviceRequested",s.deviceImmutableMetadata},{"ready",s.immutableMetadataReady},
            {"bytes",s.immutableMetadataBytes},{"allocatedBytes",s.immutableMetadataAllocatedBytes},
            {"submittedBytes",s.immutableMetadataSubmittedBytes},{"stagingBytes",s.immutableMetadataStagingBytes},
            {"uploadBatches",s.immutableMetadataUploadBatches},{"groupBytes",s.immutableGroupBytes},
            {"topologyBytes",s.immutableTopologyBytes}}},
        {"retention", {{"adaptive",s.adaptivePageRetentionEnabled},
            {"reclaimReserveBytes",s.geometryReclaimReserveBytes},{"demandReserveBytes",s.geometryDemandReserveBytes},
            {"coldResidentBytes",s.coldResidentBytes},{"pendingFreeBytes",s.pendingFreeBytes},
            {"evictedGeometryBytes",s.evictedGeometryBytes},{"evictedPrefetchPages",s.evictedPrefetchPages},
            {"pressureEvictions",s.cpuWork.coldPressureScheduled},{"retentionEvictions",s.cpuWork.coldRetentionScheduled},
            {"ageRejected",s.cpuWork.coldAgeRejected},{"scheduleFailed",s.cpuWork.coldScheduleFailed}}},
        {"clasBytes",s.clasUsedBytes},{"clasCapacity",s.clasCapacityBytes},{"clasAllocatedBytes",s.clasAllocatedBytes},{"clasStorageChunks",s.clasStorageChunks},{"clasScratchBytes",s.clasScratchBytes},
        {"clasStartBytes",s.clasStartBytes},{"clasGrowBytes",s.clasGrowBytes},{"clasEmptyBytes",s.clasEmptyBytes},
        {"clasGrowthCount",s.clasGrowthCount},{"clasReleasedBytes",s.clasReleasedBytes},
        {"clasPersistentGrowBytes",s.clasPersistentGrowBytes},{"clasPersistentAllocatedBytes",s.clasPersistentAllocatedBytes},{"clasPersistentUsedBytes",s.clasPersistentUsedBytes},
        {"clasTransientAllocatedBytes",s.clasTransientAllocatedBytes},{"clasTransientUsedBytes",s.clasTransientUsedBytes},
        {"clasFragmentedFreeBytes",s.clasFragmentedFreeBytes},
        {"clasVirtualBytes",s.clasVirtualBytes},{"clasSizeReadbackBytes",s.clasSizeReadbackBytes},
        {"clasAllocatorPages",s.clasAllocatorPages},{"clasReadyLatencySamples",s.clasReadyLatencySamples},
        {"clasReadyLatencyFrameTotal",s.clasReadyLatencyFrameTotal},{"clasReadyLatencyFrameMax",s.clasReadyLatencyFrameMax},
        {"clasReadyLatencyFramesP50",s.clasReadyLatencyFramesP50},{"clasReadyLatencyFramesP95",s.clasReadyLatencyFramesP95},
        {"clasReadyLatencyHistogram",s.clasReadyLatencyHistogram},
        {"clasBuiltClusters",s.clasBuiltClusters},{"clasMovedClusters",s.clasMovedClusters},
        {"clasTotalBuiltPages",s.clasTotalBuiltPages},{"clasTotalBuiltClusters",s.clasTotalBuiltClusters},
        {"clasPendingPages",s.clasPendingPages},{"clasRetiringPages",s.clasRetiringPages},
        {"residentPages",s.residentPages},{"pendingPages",s.pendingPages},
        {"ioQueued",s.ioQueued},{"ioActive",s.ioActive},{"uploadQueued",s.uploadQueued},
        {"requests",s.requests},{"uploads",s.uploads},{"evictions",s.evictions},{"uploadBytes",s.uploadBytes},
        {"totalUploadBytes",s.totalUploadBytes},
        {"requestOverflows",s.requestOverflows},{"allocationFailures",s.allocationFailures},{"loadFailures",s.loadFailures},
        {"allocationAttempts",s.cpuWork.allocationAttempts},{"budgetRetrySuppressed",s.cpuWork.budgetRetrySuppressed},
        {"requestDuplicatesMerged",s.cpuWork.requestDuplicatesMerged},{"admissionCandidates",s.cpuWork.admissionCandidates},
        {"admissionPriorityPops",s.cpuWork.admissionPriorityPops},{"admissionCalls",s.cpuWork.admissionCalls},
        {"demandVisited",s.cpuWork.demandVisited},{"demandTransitions",s.cpuWork.demandTransitions},
        {"demandEpochUpdates",s.cpuWork.demandEpochUpdates},{"demandStaleBatches",s.cpuWork.demandStaleBatches},
        {"demandMembershipTests",s.cpuWork.demandMembershipTests},{"demandPrefetchVisited",s.cpuWork.demandPrefetchVisited},
        {"coldCandidateTests",s.cpuWork.coldCandidateTests},
        {"priorityRecomputed",s.cpuWork.priorityRecomputed},{"priorityReused",s.cpuWork.priorityReused},
        {"admissionBypassed",s.cpuWork.admissionBypassed},
        {"blasFeedbackAvailable",s.blasFeedbackAvailable},{"blasFeedbackFrame",s.blasFeedbackFrame},
        {"blasDiagnosticResetEpoch",s.blasDiagnosticResetEpoch},
        {"blasBuildCount",s.blasBuildCount},{"blasClusterReferences",s.blasClusterReferences},{"blasOverflowCount",s.blasOverflowCount},
        {"blasRequestedClusterReferences",s.blasRequestedClusterReferences},
        {"blasRequestedInstances",s.blasRequestedInstances},
        {"blasReferenceBudgetRejected",s.blasReferenceBudgetRejected},
        {"blasBuildBudgetRejected",s.blasBuildBudgetRejected},
        {"blasOversizedInstances",s.blasOversizedInstances},
        {"blasMissingClasInstances",s.blasMissingClasInstances},
        {"blasInvalidGroups",s.blasInvalidGroups},
        {"blasLiveClusterReferences",s.blasLiveClusterReferences},{"blasAdmittedInstances",s.blasAdmittedInstances},
        {"blasInstanceReuseEnabled",s.blasInstanceReuseEnabled},
        {"blasDirtyInstances",s.blasDirtyInstances},
        {"blasReusedInstances",s.blasReusedInstances},
        {"blasStorageRejected",s.blasStorageRejected},
        {"blasArenaUsedBytes",s.blasArenaUsedBytes},
        {"blasArenaRepack",s.blasArenaRepack},
        {"blasGPUAllocatorEnabled",s.blasGPUAllocatorEnabled},
        {"blasPriorityEnabled",s.blasPriorityEnabled},
        {"blasPriorityCacheEnabled",s.blasPriorityCacheEnabled},
        {"blasMergedSharingEnabled",s.blasMergedSharingEnabled},
        {"blasFallbackPolicy",s.blasFallbackPolicy},
        {"blasMixedTileFastPathEnabled",s.blasMixedTileFastPathEnabled},
        {"blasPrioritySortReused",s.blasPrioritySortReused},
        {"blasPriorityAdmissionReused",s.blasPriorityAdmissionReused},
        {"blasRequestedVisibleConsumers",s.blasRequestedVisibleConsumers},
        {"blasRequestedOffscreenConsumers",s.blasRequestedOffscreenConsumers},
        {"blasRejectedVisibleConsumers",s.blasRejectedVisibleConsumers},
        {"blasRejectedOffscreenConsumers",s.blasRejectedOffscreenConsumers},
        {"blasAdmittedOffscreenConsumers",s.blasAdmittedOffscreenConsumers},
        {"blasPriorityEvictedOwners",s.blasPriorityEvictedOwners},
        {"blasCompactionTargetBytes",s.blasCompactionTargetBytes},
        {"blasWaitingBuildCount",s.blasWaitingBuildCount},
        {"blasPostBuildBudgetRejected",s.blasPostBuildBudgetRejected},
        {"blasPhysicalAllocationFailed",s.blasPhysicalAllocationFailed},
        {"blasWaitingClaimBytes",s.blasWaitingClaimBytes},
        {"blasMaxWaitingBytes",s.blasMaxWaitingBytes},
        {"blasUnknownBuildCount",s.blasUnknownBuildCount},
        {"blasProjectedRetryDeferredCount",s.blasProjectedRetryDeferredCount},
        {"blasProjectedDirtyClaimBytes",s.blasProjectedDirtyClaimBytes},
        {"blasRejectedMaxErrorPixels",s.blasRejectedMaxErrorPixels},
        {"blasRejectedVisibleMaxErrorPixels",s.blasRejectedVisibleMaxErrorPixels},
        {"blasEncodedBytes",s.blasEncodedBytes},{"blasAllocatedBytes",s.blasAllocatedBytes},
        {"blasFreeBytes",s.blasFreeBytes},{"blasLargestFreeBytes",s.blasLargestFreeBytes},
        {"blasRelocatedCount",s.blasRelocatedCount},{"blasRelocatedBytes",s.blasRelocatedBytes},
        {"blasSuppressedBuilds",s.blasSuppressedBuilds},{"blasFailedAllocations",s.blasFailedAllocations},
        {"blasPersistentBytes",s.blasPersistentBytes},{"blasStagingBytes",s.blasStagingBytes},
        {"blasMoveScratchBytes",s.blasMoveScratchBytes},
        {"blasRelocationStagingBytes",s.blasRelocationStagingBytes},
        {"blasPublicationInvalidated",s.blasPublicationInvalidated},
        {"blasSemanticChangeCount",s.blasSemanticChangeCount},
        {"blasRepresentationsEnabled",s.blasRepresentationsEnabled},
        {"blasRepresentationCount",s.blasRepresentationCount},
        {"blasSharingInstances",s.blasSharingInstances},
        {"blasMergedInstances",s.blasMergedInstances},
        {"blasCachedInstances",s.blasCachedInstances},
        {"blasRepresentationGroupCount",s.blasRepresentationGroupCount},
        {"blasRepresentationGroupOverflow",s.blasRepresentationGroupOverflow},
        {"blasCacheDependencyInvalidations",s.blasCacheDependencyInvalidations},
        {"textureStreaming",s.textureStreaming},
        {"textureResidentBytes",s.textureResidentBytes},{"textureRetiredBytes",s.textureRetiredBytes},
        {"texturePendingBytes",s.texturePendingBytes},{"textureBudgetBytes",s.textureBudgetBytes},
        {"textureRefinedImages",s.textureRefinedImages},{"textureRequestedImages",s.textureRequestedImages},
        {"texturePendingImages",s.texturePendingImages},{"textureFeedbackFrames",s.textureFeedbackFrames},
        {"textureUploadBytes",s.textureUploadBytes},
        {"textureUpgrades",s.textureUpgrades},{"textureDowngrades",s.textureDowngrades},
        {"textureBudgetDeferrals",s.textureBudgetDeferrals},{"textureMaxRequestFrames",s.textureMaxRequestFrames},
        {"lastTextureUploadSequence",s.textureUpload.sequence}};
}

Json streamSnapshot(const render::RenderGraphExecutionStats& execution)
{
    Json samples = Json::array();
    for (const auto& sample : execution.streaming) { samples.push_back(streamSample(sample)); }
    return {{"executionId",execution.executionId},{"graphGeneration",execution.graphGeneration},{"streaming",std::move(samples)}};
}

void recordFrozenCounters(Json& ranges, const Json& sample)
{
    auto entry = std::find_if(ranges.begin(), ranges.end(), [&](const Json& value) {
        return value.at("pass") == sample.at("pass") && value.at("asset") == sample.at("asset") &&
            value.at("generation") == sample.at("generation");
    });
    if (entry == ranges.end()) {
        ranges.push_back({{"pass",sample.at("pass")},{"asset",sample.at("asset")},
            {"generation",sample.at("generation")},{"samples",0u},{"counters",Json::object()}});
        entry = std::prev(ranges.end());
    }
    (*entry)["samples"] = entry->at("samples").get<uint64_t>() + 1;
    auto& counters = (*entry)["counters"];
    for (const char* key : {"residentPages", "geometryBytes", "clasBytes", "clasPendingPages",
        "clasTotalBuiltPages", "clasTotalBuiltClusters", "pendingPages", "ioQueued", "ioActive", "uploadQueued",
        "totalUploadBytes", "textureResidentBytes", "textureRetiredBytes", "texturePendingBytes",
        "textureRefinedImages", "textureRequestedImages", "texturePendingImages", "textureUpgrades",
        "textureDowngrades", "textureBudgetDeferrals", "textureFeedbackFrames", "textureUploadBytes", "lastTextureUploadSequence"}) {
        const uint64_t value = sample.at(key).get<uint64_t>();
        if (!counters.contains(key)) { counters[key] = {{"first",value},{"last",value},{"min",value},{"max",value}}; }
        else {
            auto& range = counters[key];
            range["last"] = value;
            range["min"] = std::min(range.at("min").get<uint64_t>(), value);
            range["max"] = std::max(range.at("max").get<uint64_t>(), value);
        }
    }
}

Json uploadSample(const render::TextureUploadProfile& s)
{
    return {{"sequence",s.sequence},{"requestFrame",s.requestFrame},{"submitFrame",s.submitFrame},
        {"completionFrame",s.completionFrame},{"bytes",s.bytes},{"images",s.images},{"queue","graphics"},
        {"gpuMs",s.gpuTimingAvailable ? Json(s.gpuMilliseconds) : Json(nullptr)},
        {"completionObservedMs",s.completionObservedMilliseconds}};
}
} // namespace

bool EditorApplication::runZorahFullRoamBenchmark()
{
    const char* outputEnvironment = std::getenv("METALLIC_FULL_ROAM_OUTPUT");
    const std::filesystem::path output = outputEnvironment ? std::filesystem::path(outputEnvironment) :
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Captures" / "FullRoam" /
        std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    if (std::filesystem::exists(output / "Capture.json")) {
        spdlog::error("[Full Roam] Capture already exists: {}", output.string());
        return false;
    }
    Json report{{"protocol","zorah-full-editor-roam-v1"},{"status","failed"},
        {"frameTimeDefinition","start-to-start editor loop; includes wait, events, recording, submit, present and capture bookkeeping"},
        {"gpuTimeDefinition","RenderGraph envelope excludes editor composite/present and independent texture submissions; do not sum nested scopes"},
        {"textureLatencyDefinition","request enqueue to publish frames; excludes pre-admission budget waiting"}};
    fullRoamActive_ = true;
    fullRoamWidth_ = viewportTextureWidth_;
    fullRoamHeight_ = viewportTextureHeight_;
    bool passed = false;
    RasterWorkloadObserver workloadObserver;
    auto qualityCapture = std::make_unique<RayQualityCapture>();
    std::unique_ptr<StreamingRegressionCapture> demandCapture;
    struct FrozenProperty { uint32_t node; std::string key; bool existed; Json value; };
    std::vector<FrozenProperty> frozenProperties;
    std::optional<scene::LightingSettings> controlledLighting;
    const auto drain = [&]() {
        if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { throw std::runtime_error("GPU drain failed"); }
        std::vector<render::RenderGraphExecutionStats> completed;
        if (!graphExecutor_->collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) { throw std::runtime_error("GPU query resolve failed"); }
        for (const auto& stats : completed) { profiler_.updateRenderGraphGpuStats(stats); }
    };
    try {
        std::filesystem::create_directories(output);
        Json config = Json::object();
        if (const char* path = std::getenv("METALLIC_FULL_ROAM_CONFIG")) {
            std::ifstream input(path);
            config = Json::parse(input);
        }
        if (config.value("rasterComparison", false)) { return runZorahFullRasterComparison(config, output); }
        if (config.contains("benchmarkFreezeStreaming") && !config.at("benchmarkFreezeStreaming").is_boolean()) {
            throw std::runtime_error("benchmarkFreezeStreaming must be a boolean");
        }
        const bool freezeStreaming = config.value("benchmarkFreezeStreaming", false);
        if (config.contains("detailDemandComparison") || config.contains("blasQualityComparison")) {
            throw std::runtime_error("Retired streaming A/B configuration; use streamingRegression for the production path");
        }
        const Json streamingRegression = config.value("streamingRegression", Json::object());
        if (!streamingRegression.is_object()) { throw std::runtime_error("streamingRegression must be an object"); }
        const bool controlledStreaming = !streamingRegression.empty();
        if (controlledStreaming && freezeStreaming) {
            throw std::runtime_error("Streaming regression requires its own residency-only freeze");
        }
        constexpr uint32_t kFreezeSettleFrames = 16;
        const Json deferredOverrides = config.value("deferredOverrides", Json::object());
        if (!deferredOverrides.is_object()) { throw std::runtime_error("deferredOverrides must be an object"); }
        const std::set<std::string> deferredBoolOverrides{"debugDisableDirectLighting", "debugDisableMaterialTextures",
            "debugDisableNormalMap", "debugForceGeometryNormal", "materialBinning", "stochasticTextureFiltering"};
        const std::set<std::string> deferredDebugViews{"final", "baseColor", "geometryNormal", "shadingNormal", "tangent", "material"};
        for (auto it = deferredOverrides.begin(); it != deferredOverrides.end(); ++it) {
            if (it.key() == "debugView") {
                if (!it.value().is_string() || !deferredDebugViews.contains(it.value().get<std::string>())) {
                    throw std::runtime_error("Unsupported deferredOverrides debugView");
                }
            } else if (!deferredBoolOverrides.contains(it.key()) || !it.value().is_boolean()) {
                throw std::runtime_error("Unsupported key or non-boolean value in deferredOverrides: " + it.key());
            }
        }
        profilerOpen_ = config.value("profilerOpen", profilerOpen_);
        const bool focusProfiler = config.value("focusProfiler", false);
        report["profilerOpen"] = profilerOpen_;
        report["focusProfiler"] = focusProfiler;
        report["blasStorageRejectedSemantics"] = "physical-and-known-storage-only-v2";
        const uint32_t workloadEvery = config.value("workloadEvery", 0u);
        const bool captureQuality = config.value("captureQuality", false);
        const bool captureReflections = config.value("captureReflections", false);
        const bool capturePathDiagnostics = config.contains("diagnosticPixels");
        if (capturePathDiagnostics && (!controlledStreaming || !captureReflections)) {
            throw std::runtime_error("diagnosticPixels requires the controlled MiniZorah streaming regression");
        }
        const uint32_t reflectionSamples = config.value("reflectionSamples", 8u);
        if (reflectionSamples == 0u || reflectionSamples > 16u) { throw std::runtime_error("reflectionSamples must be [1,16]"); }
        if (captureReflections && !captureQuality) { throw std::runtime_error("captureReflections requires captureQuality"); }
        if (captureQuality && workloadEvery) { throw std::runtime_error("Quality and workload captures require separate runs"); }
        report["captureQuality"] = captureQuality;
        report["captureReflections"] = captureReflections;
        qualityCapture->captureReflections = captureReflections;
        qualityCapture->captureRasterInputs = controlledStreaming;
        qualityCapture->capturePathDiagnostics = capturePathDiagnostics;
        qualityCapture->diagnosticRows = capturePathDiagnostics ? 16u * uint32_t(config.at("diagnosticPixels").size()) : 0u;
        workloadObserver.captureCull = config.value("classifyCounters", false);
        if (workloadEvery && workloadEvery < 10) { throw std::runtime_error("workloadEvery must be 0 or at least 10"); }
        report["workloadEvery"] = workloadEvery;
        report["graphicsCaptureInjected"] = render::profiling::NsightGraphicsCapture::vulkanInjectionActive();
        const auto shaderDebugMode = render::slangShaderDebugMode();
        report["shaderDebugMode"] = shaderDebugMode == render::SlangShaderDebugMode::ShaderDebug ? "unoptimized-debug"
            : shaderDebugMode == render::SlangShaderDebugMode::CaptureSymbols ? "capture-symbols" : "disabled";
        report["diagnosticRun"] = captureQuality || freezeStreaming || !deferredOverrides.empty() || workloadEvery != 0 || config.value("enableLodTransitionTelemetry", false) ||
            report["graphicsCaptureInjected"].get<bool>() || shaderDebugMode != render::SlangShaderDebugMode::Disabled;
        report["lodTransitionDefinition"] = "Instance-groups in the emitted geometry cut, before raster occlusion and independent of CLAS. Threshold requires continuous auto-visible frames and newly demanded SSE; catchup requires prior-frame demand blocked by own geometry page or dependency. First/reentry and other selected changes are unclassified. History adds 8 bytes per instance-group plus 4 per instance; feedback may lag.";
        const double duration = config.value("durationSeconds",180.0);
        const double warmup = config.value("warmupSeconds",5.0);
        const double distance = config.value("distance",6.0);
        if (!std::isfinite(duration) || duration < 1 || duration > 1800 ||
            !std::isfinite(warmup) || warmup < 0 || warmup > 120 ||
            !std::isfinite(distance) || distance < 0 || distance > 100) { throw std::runtime_error("Invalid route configuration"); }
        if (config.contains("width") || config.contains("height")) {
            const int width = config.value("width",0), height = config.value("height",0);
            if (width < 64 || height < 64 || width > 8192 || height > 8192) { throw std::runtime_error("Explicit viewport requires width and height in [64,8192]"); }
            fullRoamWidth_ = uint32_t(width); fullRoamHeight_ = uint32_t(height);
        }
        // Fractions of route duration, forward distance, yaw degrees. Actual
        // absolute camera keyframes are exported for exact reuse and inspection.
        const Json points = config.value("keyframes",Json::array({
            {{"t",0.0},{"forward",0.0},{"yaw",0.0},{"stage","static"}},
            {{"t",1.0/12},{"forward",0.0},{"yaw",0.0},{"stage","first-visit"}},
            {{"t",1.0/3},{"forward",1.0},{"yaw",0.0},{"stage","near-wall"}},
            {{"t",4.0/9},{"forward",1.0},{"yaw",65.0},{"stage","turn"}},
            {{"t",2.0/3},{"forward",1.0},{"yaw",-65.0},{"stage","return"}},
            {{"t",8.0/9},{"forward",0.0},{"yaw",0.0},{"stage","settle"}},
            {{"t",1.0},{"forward",0.0},{"yaw",0.0},{"stage","end"}}}));
        if (!points.is_array() || points.size() < 2 || points.size() > 128 ||
            points.front().at("t").get<double>() != 0 || points.back().at("t").get<double>() != 1) {
            throw std::runtime_error("Route keyframes must span [0,1]");
        }
        double lastT = -1;
        for (const auto& point : points) {
            const double t=point.at("t"), f=point.at("forward"), yaw=point.at("yaw");
            if (!std::isfinite(t) || !std::isfinite(f) || !std::isfinite(yaw) || t <= lastT || t > 1 ||
                std::abs(f)>10 || std::abs(yaw)>360 || !point.at("stage").is_string()) { throw std::runtime_error("Invalid route keyframe"); }
            lastT=t;
        }
        if (workloadEvery) { graphExecutor_->setDebugObserver(&workloadObserver); }
        if (captureQuality) { graphExecutor_->setDebugObserver(qualityCapture.get()); }
        if (controlledStreaming) {
            // Compile transfer-readable, non-aliasing allocations before the
            // source is frozen. Attaching a late readback observer could rebuild
            // the graph and invalidate the source identity before the production capture.
            // The inactive observer records no copies during loading/warmup.
            demandCapture = std::make_unique<StreamingRegressionCapture>();
            graphExecutor_->setDebugObserver(demandCapture.get());
        }
        const std::string sampleId = config.value("sample", std::string(render::kGPUDrivenZorahFullSampleId));
        if (controlledStreaming && (sampleId != "gpu-driven-sample" || workloadEvery ||
            captureQuality != captureReflections)) {
            throw std::runtime_error("Detail streaming regression requires MiniZorah and separate timing or shadow/reflection quality modes");
        }
        if (!render::isGPUDrivenSceneSample(sampleId)) { throw std::runtime_error("Roam requires MiniZorah or ZorahFull sample"); }
        loadBuiltInSample(sampleId.c_str());
        if (config.contains("shadowPointLight")) {
            const auto& source = config.at("shadowPointLight");
            const auto position = source.at("position").get<std::array<float, 3>>();
            scene::PunctualLight light;
            light.name = "RT representation benchmark light";
            light.position = {position[0], position[1], position[2]};
            light.properties.intensity = source.at("intensity").get<double>();
            auto lighting = renderWorld_.lighting();
            lighting.lights.push_back(std::move(light));
            if (!scene::validLightingSettings(lighting)) { throw std::runtime_error("Invalid shadowPointLight"); }
            renderWorld_.setLighting(std::move(lighting));
            report["shadowPointLight"] = source;
        }
        if (config.contains("shadowOverrides")) {
            const auto* shadows = renderGraph_.findNode("Shadows");
            if (!shadows || !config.at("shadowOverrides").is_object()) { throw std::runtime_error("Invalid shadowOverrides"); }
            for (auto it = config.at("shadowOverrides").begin(); it != config.at("shadowOverrides").end(); ++it) {
                if (!renderGraph_.setNodeRuntimeProperty(shadows->id, it.key(), it.value())) {
                    throw std::runtime_error("Failed to apply shadow override: " + it.key());
                }
            }
            report["shadowOverrides"] = config.at("shadowOverrides");
        }
        const auto* vbuffer = renderGraph_.findNode("VBuffer");
        if (!vbuffer) { throw std::runtime_error("Missing VBuffer"); }
        const uint32_t groupSize = config.value("softwareGroupSize", 0u);
        if (groupSize != 0 && groupSize != 32 && groupSize != 64 && groupSize != 128) {
            throw std::runtime_error("Unsupported experimental SW group size");
        }
        if (groupSize) {
            const char* experiment = std::getenv("METALLIC_SW_GROUP_EXPERIMENT");
            if (groupSize != 32 && (!experiment || std::string_view(experiment) != "1")) {
                throw std::runtime_error("Experimental SW group size requires explicit process opt-in");
            }
            renderGraph_.setNodeRuntimeProperty(vbuffer->id, "benchmarkSoftwareGroupSize", groupSize);
        }
        renderGraph_.setNodeRuntimeProperty(vbuffer->id, "cullHardwareClassification", config.value("cullHardwareClassification", false));
        renderGraph_.setNodeRuntimeProperty(vbuffer->id, "metadataFastClassification", config.value("metadataFastClassification", true));
        renderGraph_.setNodeRuntimeProperty(vbuffer->id, "initialLoad", config.value("initialLoad", true));
        for (const char* key : {"maxClasBytes", "maxResidentBytes"}) {
            if (config.contains(key)) {
                const auto bytes = config.at(key).get<uint64_t>();
                if (bytes == 0u) { throw std::runtime_error(std::string(key) + " must be nonzero"); }
                renderGraph_.setNodeRuntimeProperty(vbuffer->id, key, bytes);
            }
        }
        renderGraph_.setNodeRuntimeProperty(vbuffer->id, "deviceImmutableMetadata", config.value("deviceImmutableMetadata", true));
        renderGraph_.setNodeRuntimeProperty(vbuffer->id, "predictivePrefetch", config.value("predictivePrefetch", true));
        renderGraph_.setNodeRuntimeProperty(vbuffer->id, "adaptivePageRetention", config.value("adaptivePageRetention", true));
        renderGraph_.setNodeRuntimeProperty(vbuffer->id, "enableLodTransitionTelemetry", config.value("enableLodTransitionTelemetry", false));
        if (controlledStreaming) {
            // Keep the distributed production path fixed for a reproducible capture.
            renderGraph_.setNodeRuntimeProperty(vbuffer->id, "distributedDemandMinGroups", 0u);
        }
        const auto* deferred = renderGraph_.findNode("Deferred");
        const auto* dlss = renderGraph_.findNode("DLSSSR");
        if (!deferred || !dlss) { throw std::runtime_error("Missing Full shading/reconstruction nodes"); }
        renderGraph_.setNodeRuntimeProperty(deferred->id, "stochasticTextureFiltering", config.value("stochasticTextureFiltering", false));
        const auto applyDeferredOverrides = [&]() {
            for (auto it = deferredOverrides.begin(); it != deferredOverrides.end(); ++it) {
                if (!renderGraph_.setNodeRuntimeProperty(deferred->id, it.key(), it.value())) {
                    throw std::runtime_error("Failed to apply Deferred override: " + it.key());
                }
            }
        };
        // Frozen variants warm the same default shading workload before overrides.
        if (!freezeStreaming) { applyDeferredOverrides(); }
        const std::string dlssMode = config.value("dlssMode", std::string("Quality"));
        if (dlssMode != "Off" && dlssMode != "DLAA" && dlssMode != "Quality" &&
            dlssMode != "Balanced" && dlssMode != "Performance" && dlssMode != "UltraPerformance" &&
            dlssMode != "UltraQuality") { throw std::runtime_error("Unsupported Full dlssMode"); }
        renderGraph_.setNodeRuntimeProperty(dlss->id, "mode", dlssMode);
        if (config.contains("maxRasterCandidates")) {
            renderGraph_.setNodeRuntimeProperty(vbuffer->id, "maxRasterCandidates", config.at("maxRasterCandidates").get<uint32_t>());
        }
        if (config.contains("temporalJitter")) {
            auto view = viewportCameraProperties();
            const bool jitter = config.at("temporalJitter").get<bool>();
            view["temporalJitter"] = jitter;
            applyViewportCameraProperties(view, nullptr);
            viewportView_.setTemporalJitter(jitter);
            renderGraph_.setNodeRuntimeProperty(vbuffer->id, "temporalJitter", jitter);
        }
        if (captureReflections) {
            // Reuse the raster producer's scene binding, stream budgets and
            // global camera. An asset-bound sample must not accidentally trace
            // a different editor document through a new World binding.
            Json properties = vbuffer->properties;
            properties.update(vbuffer->runtimeProperties, true);
            properties["viewBinding"] = "global";
            properties["bsdf"] = "openpbr";
            properties["cacheMode"] = "off";
            properties["samples"] = reflectionSamples;
            properties["maxDepth"] = 3;
            properties["accumulate"] = false;
            properties["outputLinear"] = true;
            properties["exportDenoiserGuides"] = false;
            properties["diagnosticSampleFrame"] = 0u;
            properties["diagnosticUseVisibilityStream"] = true;
            properties["stochasticTextureFiltering"] = false;
            if (capturePathDiagnostics) { properties["diagnosticPixels"] = config.at("diagnosticPixels"); }
            if (!renderGraph_.addNode("ScenePathTracePass", "ReflectionQuality", properties) ||
                !renderGraph_.addEdge("VBuffer.rasterInfo", "ReflectionQuality.rasterInfo") ||
                !renderGraph_.addEdge("VBuffer.accelerationStructure", "ReflectionQuality.accelerationStructure") ||
                !renderGraph_.markOutput("FinalBlit.color") || !renderGraph_.markOutput("ReflectionQuality.color")) {
                throw std::runtime_error("Failed to add path-traced reflection quality diagnostic");
            }
            if (capturePathDiagnostics && !renderGraph_.markOutput("ReflectionQuality.pathDiagnostics")) {
                throw std::runtime_error("Failed to retain path diagnostics");
            }
            // addNode can reallocate the graph's node storage.
            vbuffer = renderGraph_.findNode("VBuffer");
            deferred = renderGraph_.findNode("Deferred");
            dlss = renderGraph_.findNode("DLSSSR");
            report["reflectionDiagnostic"] = {{"pass", "ReflectionQuality"}, {"properties", properties},
                {"scope", "Three-vertex OpenPBR path tracing of VBuffer's published CLAS TLAS including material reflections; not isolated specular radiance"},
                {"geometrySource", "VBuffer.accelerationStructure and its published visibility stream; no separate streaming session"},
                {"performanceComparable", false}};
        }
        const uint32_t routeFrames = config.value("routeFrames", 0u);
        if (routeFrames && (routeFrames < 8 || routeFrames > 10000)) { throw std::runtime_error("routeFrames must be 0 or [8,10000]"); }
        std::set<uint32_t> qualitySamples;
        const Json requestedQualitySamples = config.value("qualitySamples", Json::array());
        if (!requestedQualitySamples.is_array()) { throw std::runtime_error("qualitySamples must be an array of route sample indices"); }
        for (const auto& value : requestedQualitySamples) {
            if (!captureQuality || !routeFrames || !value.is_number_integer() ||
                (!value.is_number_unsigned() && value.get<int64_t>() < 0) ||
                value.get<uint64_t>() >= routeFrames || !qualitySamples.insert(value.get<uint32_t>()).second) {
                throw std::runtime_error("qualitySamples requires captureQuality, deterministic routeFrames and unique indices within the route");
            }
        }
        report["qualitySamples"] = Json::array();
        report["sample"] = sampleId;
        report["trajectoryClock"] = routeFrames ? "deterministic frame index" : "wall time";
        const auto workloadSetting = [&](bool enabled) {
            const auto* node = renderGraph_.findNode("VBuffer");
            if (!node) { throw std::runtime_error("Missing VBuffer"); }
            renderGraph_.setNodeRuntimeProperty(node->id, "softwareRasterWorkload", enabled);
        };
        render::vulkan::StreamlineFrameBeginProfile frameBegin;
        const auto draw = [&]() {
            auto frame = profiler_.beginFrame();
            if (!waitForFrameSlotBeforeInput()) { return false; }
            const render::vulkan::StreamlineFrameScope streamlineFrame(
                (SDL_GetWindowFlags(window_) & SDL_WINDOW_MINIMIZED)==0 && ImGui::GetPlatformIO().Viewports.Size<=1, &frameBegin);
            if (frameBegin.sleepCalled) { profiler_.addIdleSample("Reflex Frame Pacing", frameBegin.sleepMs); }
            { auto scope = profiler_.scope("Poll Events"); pollEvents(); }
            if (!running_ || SDL_GetKeyboardState(nullptr)[SDL_SCANCODE_ESCAPE]) { return false; }
            auto scope = profiler_.scope("Render Frame");
            return renderFrame() && !viewportCompileFailed_;
        };
        const auto loadingStart = Clock::now();
        auto readyAt = Clock::time_point{};
        uint32_t loadingFrames = 0;
        while (true) {
            if (!draw()) { throw std::runtime_error("Full loading or rendering failed/cancelled"); }
            ++loadingFrames;
            if (loadingFrames == 1 && focusProfiler && profilerOpen_) {
                ImGui::SetWindowFocus("Profiler");
            }
            const auto readiness = subsystemHost_.get<render::StreamerSubsystem>()->sceneReadiness();
            const auto now=Clock::now();
            if (readiness.ready && readiness.requiredPages && viewportPreviewValid_) {
                if (readyAt == Clock::time_point{}) { readyAt=now; }
                if (std::chrono::duration<double>(now-readyAt).count() >= warmup) { break; }
            }
            if (std::chrono::duration<double>(now-loadingStart).count()>600) { throw std::runtime_error("Full readiness timed out"); }
        }
        if (workloadEvery) { graphExecutor_->setDebugObserver(nullptr); }
        report["streamingFreeze"] = {{"enabled",freezeStreaming},{"settleFrames",freezeStreaming ? kFreezeSettleFrames : 0u},
            {"scope","Freeze geometry/CLAS/cut/TLAS and texture publication/mip reveal; camera route remains configured"},
            {"pendingDefinition","GPU drain resolves submitted work, not CPU decode or texture migration publication; pending work may remain frozen"},
            {"workingSetDefinition","CPU aggregate counters only; no per-image mip distribution or working-set hash, so equality does not establish identical texture residency across processes"}};
        if (freezeStreaming) {
            drain();
            report["streamingFreeze"]["beforeFreeze"] = streamSnapshot(graphExecutor_->executionStats());
            for (const auto* node : {vbuffer, deferred}) {
                const bool existed = node->runtimeProperties.contains("benchmarkFreezeStreaming");
                frozenProperties.push_back({node->id, "benchmarkFreezeStreaming", existed, existed ? node->runtimeProperties.at("benchmarkFreezeStreaming") : Json(nullptr)});
                if (!renderGraph_.setNodeRuntimeProperty(node->id, "benchmarkFreezeStreaming", true)) {
                    throw std::runtime_error("Failed to freeze streaming");
                }
            }
            applyDeferredOverrides();
            const auto settleStart = Clock::now();
            for (uint32_t i = 0; i < kFreezeSettleFrames; ++i) {
                if (!draw()) { throw std::runtime_error("Frozen streaming settle failed/cancelled"); }
            }
            drain();
            report["streamingFreeze"]["settleSeconds"] = std::chrono::duration<double>(Clock::now()-settleStart).count();
            report["streamingFreeze"]["afterFreezeSettle"] = streamSnapshot(graphExecutor_->executionStats());
        }
        if (!fullRoamWidth_) { fullRoamWidth_=viewportTextureWidth_; fullRoamHeight_=viewportTextureHeight_; }
        const auto original = viewportCameraProperties();
        const auto cameraAt = [&](double forward, double yaw) {
            auto camera = original;
            const auto& base = original.at("camera");
            const double dx=base["center"][0].get<double>()-base["eye"][0].get<double>();
            const double dz=base["center"][2].get<double>()-base["eye"][2].get<double>();
            const double length=std::hypot(dx,dz);
            if (length<1e-6) { throw std::runtime_error("Route requires a horizontal view direction"); }
            const double angle=yaw*3.14159265358979323846/180;
            camera["camera"]["eye"][0]=base["eye"][0].get<double>()+dx/length*distance*forward;
            camera["camera"]["eye"][2]=base["eye"][2].get<double>()+dz/length*distance*forward;
            camera["camera"]["center"][0]=camera["camera"]["eye"][0].get<double>()+std::cos(angle)*dx+std::sin(angle)*dz;
            camera["camera"]["center"][2]=camera["camera"]["eye"][2].get<double>()-std::sin(angle)*dx+std::cos(angle)*dz;
            return camera;
        };
        if (controlledStreaming) {
            const uint32_t settleFrames = streamingRegression.value("settleFrames", 16u);
            const uint32_t sampleFrames = streamingRegression.value("sampleFrames", 64u);
            const bool prepareProfile = streamingRegression.value("prepareProfile", false);
            const bool cpuThreadDiagnostics = streamingRegression.value("cpuThreadDiagnostics", false);
            const bool deferDiagnostics = !captureQuality;
            const auto seeds = streamingRegression.value("seeds", std::vector<uint32_t>{0u, 1u});
            if (settleFrames < 4u || settleFrames > 16u || sampleFrames < 16u || sampleFrames > 64u ||
                (prepareProfile && captureQuality) ||
                (cpuThreadDiagnostics && !prepareProfile) ||
                seeds.empty() || seeds.size() > 2u || std::set<uint32_t>(seeds.begin(), seeds.end()).size() != seeds.size() ||
                viewportView_.temporalJitter() || config.value("stochasticTextureFiltering", false) ||
                (captureQuality && (dlssMode != "Off" || !config.contains("shadowOverrides") ||
                    config.at("shadowOverrides").value("sigmaDenoise", true)))) {
                throw std::runtime_error("Invalid bounded deterministic MiniZorah streaming regression configuration");
            }
            const auto setProperty = [&](const std::string& nodeName, const std::string& key, const Json& value) {
                const auto* node = renderGraph_.findNode(nodeName);
                if (!node) { throw std::runtime_error("Missing streaming regression node " + nodeName); }
                if (std::none_of(frozenProperties.begin(), frozenProperties.end(), [&](const auto& saved) {
                    return saved.node == node->id && saved.key == key;
                })) {
                    const bool existed = node->runtimeProperties.contains(key);
                    frozenProperties.push_back({node->id, key, existed, existed ? node->runtimeProperties.at(key) : Json(nullptr)});
                }
                if (!renderGraph_.setNodeRuntimeProperty(node->id, key, value)) {
                    throw std::runtime_error("Failed streaming regression property " + key);
                }
            };
            const auto snapshot = [&]() {
                const auto& profiles = graphExecutor_->executionStats().streaming;
                const auto found = std::find_if(profiles.begin(), profiles.end(), [](const auto& s) { return s.passName == "VBuffer"; });
                if (found == profiles.end()) { throw std::runtime_error("Missing streaming regression stream profile"); }
                return streamSample(*found);
            };
            setProperty("VBuffer", "benchmarkPrepareProfile", prepareProfile);
            setProperty("VBuffer", "benchmarkCPUThreadDiagnostics", cpuThreadDiagnostics);
            report["prepareProfile"] = prepareProfile;
            report["cpuThreadDiagnostics"] = cpuThreadDiagnostics;
            drain();
            controlledLighting = renderWorld_.lighting();
            auto fixedLighting = *controlledLighting;
            fixedLighting.autoExposure.enabled = false;
            renderWorld_.setLighting(std::move(fixedLighting));
            {
                // Readiness covers terminal fallback only. Keep CPU maintenance,
                // CLAS publication and texture reveal live until their work and
                // the actual submitted sampling metadata have stopped changing.
                setProperty("VBuffer", "benchmarkCaptureSemanticSource", true);
                Json observations = Json::array();
                Json previousIdentity;
                uint64_t previousFeedback = UINT64_MAX;
                uint32_t stableFeedbacks = 0;
                const auto start = Clock::now();
                auto stableSince = start;
                while (true) {
                    if (!draw()) { throw std::runtime_error("Demand convergence frame failed"); }
                    const auto state = snapshot();
                    bool idle = true;
                    for (const char* key : {"gpuLoadRequests", "geometryPendingTasks", "geometryPendingPatches",
                        "pendingPages", "ioQueued", "ioActive", "uploadQueued", "uploads", "evictions",
                        "clasPendingPages", "clasRetiringPages", "clasBuiltClusters", "clasMovedClusters",
                        "texturePendingImages", "textureRequestedImages", "texturePendingBytes", "textureRetiredBytes"}) {
                        idle = idle && state.at(key).get<uint64_t>() == 0u;
                    }
                    for (const char* key : {"loadFailures", "allocationFailures", "requestOverflows"}) {
                        if (state.at(key).get<uint64_t>() != 0u) { throw std::runtime_error("Demand convergence resource failure: " + std::string(key)); }
                    }
                    const Json identity = {state.at("geometryPublicationRevision"), state.at("clasPublicationRevision"),
                        state.at("residentPages"), state.at("clasTotalBuiltPages"), state.at("diagnosticTextureSemanticIdentity")};
                    const auto now = Clock::now();
                    if (!idle || identity != previousIdentity) { stableFeedbacks = 0; stableSince = now; }
                    const uint64_t feedback = state.at("feedbackFrame").get<uint64_t>();
                    const bool fresh = feedback != UINT64_MAX && feedback != previousFeedback;
                    if (idle && fresh) { ++stableFeedbacks; }
                    observations.push_back({{"frame",state.at("frame")},{"feedbackFrame",feedback},{"idle",idle},
                        {"gpuLoadRequests",state.at("gpuLoadRequests")},{"geometryPendingTasks",state.at("geometryPendingTasks")},
                        {"uploadQueued",state.at("uploadQueued")},{"clasPendingPages",state.at("clasPendingPages")},
                        {"residentPages",state.at("residentPages")},{"stableFeedbacks",stableFeedbacks}});
                    previousFeedback = feedback;
                    previousIdentity = identity;
                    const double stableSeconds = std::chrono::duration<double>(now-stableSince).count();
                    report["demandStabilization"] = {{"completed",false},{"observations",observations},
                        {"requiredFreshFeedbacks",8},{"minimumStableSeconds",0.2}};
                    if (stableFeedbacks >= 8u && stableSeconds >= 0.2) {
                        report["demandStabilization"]["completed"] = true;
                        report["demandStabilization"]["stableFeedbacks"] = stableFeedbacks;
                        report["demandStabilization"]["stableSeconds"] = stableSeconds;
                        report["demandStabilization"]["finalState"] = state;
                        break;
                    }
                    if (std::chrono::duration<double>(now-start).count() > 20.0) { throw std::runtime_error("Demand source did not converge within 20 seconds"); }
                }
                setProperty("VBuffer", "benchmarkCaptureSemanticSource", false);
                drain();
            }
            for (const char* node : {"VBuffer", "Deferred"}) { setProperty(node, "benchmarkFreezeResidency", true); }
            if (captureReflections) { setProperty("ReflectionQuality", "benchmarkFreezeResidency", true); }
            setProperty("VBuffer", "benchmarkDemandTraversal", true);
            {
                // Loaded pages arrive in different batches. Start admission from
                // one empty owner/history state after the source is frozen.
                const auto before = snapshot();
                setProperty("VBuffer", "benchmarkBlasStorageBytes", before.at("blasPersistentBytes"));
                setProperty("VBuffer", "benchmarkBlasResetEpoch", 1u);
                Json resetFrames = Json::array();
                uint64_t previousFeedback = UINT64_MAX;
                uint32_t stableFeedbacks = 0;
                for (uint32_t frame = 0; frame < 64u; ++frame) {
                    if (!draw()) { throw std::runtime_error("Canonical owner reset frame failed"); }
                    const auto state = snapshot();
                    resetFrames.push_back(state);
                    const uint64_t feedback = state.at("blasFeedbackFrame").get<uint64_t>();
                    const bool idle = state.at("blasDiagnosticResetEpoch") == 1u && state.at("blasFeedbackAvailable") == true &&
                        state.at("blasBuildCount") == 0u && state.at("blasSemanticChangeCount") == 0u &&
                        state.at("blasArenaRepack") == 0u && state.at("blasPrioritySortReused") == 1u &&
                        state.at("blasPriorityAdmissionReused") == 1u;
                    if (!idle) { stableFeedbacks = 0; }
                    else if (feedback != previousFeedback) { ++stableFeedbacks; }
                    previousFeedback = feedback;
                    report["canonicalOwnerReset"] = {{"completed",false},{"epoch",1},{"historyResetRequested",true},{"frames",resetFrames}};
                    if (stableFeedbacks >= 8u) {
                        report["canonicalOwnerReset"]["completed"] = true;
                        report["canonicalOwnerReset"]["stableFeedbacks"] = stableFeedbacks;
                        break;
                    }
                }
                if (!report["canonicalOwnerReset"].at("completed").get<bool>()) { throw std::runtime_error("Canonical owner reset did not settle within 64 frames"); }
                drain();
            }
            report["protocol"] = "minizorah-streaming-regression-v1";
            report["diagnosticRun"] = captureQuality || prepareProfile;
            report["config"] = config;
            report["frames"] = 0u;
            report["loadingFrames"] = loadingFrames;
            report["loadingSeconds"] = std::chrono::duration<double>(readyAt-loadingStart).count();
            report["outputExtent"] = {fullRoamWidth_,fullRoamHeight_};
            auto* depth = graphExecutor_->outputResource("VBuffer.depth");
            if (!depth) { throw std::runtime_error("Missing streaming regression render extent"); }
            report["renderExtent"] = {depth->desc.width,depth->desc.height};
            report["validationRequested"] = debugRuntime_ && std::getenv("METALLIC_DEBUG_VALIDATION");
            report["streamingRegression"] = {
                {"scope","Converged frozen source; canonical owner/history reset; production implementation only"},
                {"prepareProfile",prepareProfile},{"timingDiagnosticsDeferred",deferDiagnostics},
                {"cpuThreadDiagnostics",cpuThreadDiagnostics},{"statisticalUnit","One bounded production run; no A/B or speedup inference"},
                {"distributedDemandMinGroups",0u},{"fixedExposureEV100",controlledLighting->exposureEV100},
                {"performanceComparable",!captureQuality && !prepareProfile},{"camera",viewportCameraProperties()}};
            graphExecutor_->setDebugObserver(nullptr);
            std::string sourceIdentity;
            const auto checkState = [&](bool maskDiagnostics = false) {
                auto state = snapshot();
                const auto identity = state.at("diagnosticSourceIdentity").get<std::string>();
                if (sourceIdentity.empty()) { sourceIdentity = identity; }
                if (identity.empty() || identity != sourceIdentity || state.at("nodeDemandC1") != true ||
                    state.at("nodeDemandC2") != true || state.at("rtDemandMode") != 1u ||
                    state.at("rtHistoryPacked") != false || state.at("rtHistorySparseWrites") != false ||
                    state.at("terminalOnlyMask") != true || state.at("terminalMaskDiagnostics") != maskDiagnostics) {
                    throw std::runtime_error("Streaming regression source or production implementation changed");
                }
                return state;
            };
            const auto exportTiming = [&](const std::vector<uint64_t>& frameIds) {
                const auto& frames = profiler_.capturedFrames();
                const auto directory = output / "steady";
                std::filesystem::create_directories(directory);
                std::ofstream file(directory / "Frames.jsonl");
                file.exceptions(std::ios::badbit | std::ios::failbit);
                for (size_t i = 0; i < frameIds.size(); ++i) {
                    const auto found = std::lower_bound(frames.begin(), frames.end(), frameIds[i],
                        [](const auto& frame, uint64_t id) { return frame.index < id; });
                    if (found == frames.end()) { throw std::runtime_error("Streaming regression timing frame missing"); }
                    const auto& frame = *found;
                    if (frame.index != frameIds[i] || frame.profilingOverflow) { throw std::runtime_error("Streaming regression timing frame identity mismatch"); }
                    Json row{{"frame",frame.index},{"sample","steady"},{"diagnostic",prepareProfile},{"scopes",Json::array()},{"streaming",Json::array()}};
                    std::vector<std::string> paths;
                    bool completeGpu = false;
                    for (size_t n = 0; n < frame.nodes.size(); ++n) {
                        const auto& node = frame.nodes[n];
                        paths.push_back((n && node.parent < n ? paths[node.parent] + "/" : "") + node.name);
                        Json work = Json::object();
                        for (const auto& count : node.cpuWork) { work[count.name] = count.value; }
                        row["scopes"].push_back({{"path",paths.back()},{"cpuMs",node.cpuMilliseconds},
                            {"gpuMs",node.gpuTimingAvailable ? Json(node.gpuMilliseconds) : Json(nullptr)},
                            {"cpuThread",node.cpuThreadProfile ? Json(*node.cpuThreadProfile) : Json(nullptr)},
                            {"cpuWork",std::move(work)}});
                        if (node.name == "RenderGraph GPU envelope") { completeGpu = node.gpuTimingAvailable; }
                    }
                    if (!completeGpu) { throw std::runtime_error("Streaming regression incomplete GPU timing coverage"); }
                    for (const auto& profile : frame.streaming) { row["streaming"].push_back(streamSample(profile)); }
                    file << row.dump() << '\n';
                }
            };
            const auto captureSnapshot = [&](Json& sample) {
                for (uint32_t seed : captureQuality ? seeds : std::vector<uint32_t>{0u}) {
                    drain();
                    if (captureQuality) {
                        qualityCapture->capturePostprocessInputs = streamingRegression.value("capturePostprocessInputs", false);
                        setProperty("ReflectionQuality", "diagnosticSampleFrame", seed);
                        setProperty("Shadows", "diagnosticSampleFrame", seed);
                        qualityCapture->capture = true;
                    }
                    demandCapture = std::make_unique<StreamingRegressionCapture>();
                    demandCapture->capture = true;
                    demandCapture->semanticSource = true;
                    demandCapture->images = captureQuality ? qualityCapture.get() : nullptr;
                    setProperty("VBuffer", "benchmarkCaptureSemanticSource", true);
                    setProperty("VBuffer", "benchmarkTerminalMaskStats", true);
                    graphExecutor_->setDebugObserver(demandCapture.get());
                    if (!draw()) { throw std::runtime_error("Streaming regression snapshot frame failed"); }
                    setProperty("VBuffer", "benchmarkCaptureSemanticSource", false);
                    setProperty("VBuffer", "benchmarkTerminalMaskStats", false);
                    qualityCapture->capture = false;
                    auto state = checkState(true);
                    drain();
                    const std::string directory = std::string("steady/seed") + std::to_string(seed);
                    Json capture{{"seed",seed},{"directory",directory},{"state",state},
                        {"buffers",demandCapture->exportAfterDrain(output / directory)}};
                    if (captureQuality) { capture["images"] = qualityCapture->exportAfterDrain(output / directory); }
                    sample["captures"].push_back(std::move(capture));
                    graphExecutor_->setDebugObserver(nullptr);
                    demandCapture.reset();
                    qualityCapture = std::make_unique<RayQualityCapture>();
                    qualityCapture->captureReflections = captureReflections;
                    qualityCapture->captureRasterInputs = true;
                    qualityCapture->capturePathDiagnostics = capturePathDiagnostics;
                    qualityCapture->diagnosticRows = capturePathDiagnostics ? 16u * uint32_t(config.at("diagnosticPixels").size()) : 0u;
                }
            };
            setProperty("VBuffer", "benchmarkTerminalMaskStats", false);
            Json sample{{"name","steady"},{"settle",Json::array()},{"captures",Json::array()}};
            for (uint32_t frame = 0; frame < settleFrames; ++frame) {
                if (!draw()) { throw std::runtime_error("Streaming regression settle failed"); }
                sample["settle"].push_back(checkState());
            }
            const auto implementation = checkState();
            sample["frontierShaderIdentity"] = implementation.at("frontierShaderIdentity");
            sample["maskShaderIdentity"] = implementation.at("maskShaderIdentity");
            if (!captureQuality) {
                std::vector<uint64_t> frameIds;
                frameIds.reserve(sampleFrames);
                profiler_.beginCapture();
                for (uint32_t frame = 0; frame < sampleFrames; ++frame) {
                    frameIds.push_back(profiler_.nextFrameIndex());
                    if (!draw()) { throw std::runtime_error("Streaming regression timed frame failed"); }
                }
                profiler_.endCapture();
                drain();
                if (profiler_.capturedFrames().size() != sampleFrames) {
                    throw std::runtime_error("Streaming regression timing frame count mismatch");
                }
                exportTiming(frameIds);
                sample["timings"] = "steady/Frames.jsonl";
                report["frames"] = sampleFrames;
                sample["diagnosticSettle"] = Json::array();
                for (uint32_t frame = 0; frame < settleFrames; ++frame) {
                    if (!draw()) { throw std::runtime_error("Streaming diagnostic recovery failed"); }
                    sample["diagnosticSettle"].push_back(checkState());
                }
            }
            captureSnapshot(sample);
            report["streamingRegression"]["sample"] = std::move(sample);
            report["streamingRegression"]["sourceIdentity"] = sourceIdentity;
            report["graph"] = Json::array();
            for (const auto& node : renderGraph_.nodes()) {
                auto properties = node.properties;
                properties.merge_patch(node.runtimeProperties);
                report["graph"].push_back({{"name",node.name},{"type",node.type},{"properties",properties}});
            }
        } else {
        report["config"]={{"durationSeconds",duration},{"warmupSeconds",warmup},{"distance",distance},{"keyframes",points},
            {"routeFrames",routeFrames},{"sample",sampleId},{"cullHardwareClassification",config.value("cullHardwareClassification",false)},
            {"metadataFastClassification",config.value("metadataFastClassification",true)},
            {"initialLoad",config.value("initialLoad",true)},
            {"deviceImmutableMetadata",config.value("deviceImmutableMetadata",true)},
            {"predictivePrefetch",config.value("predictivePrefetch",true)},
            {"adaptivePageRetention",config.value("adaptivePageRetention",true)},
            {"enableLodTransitionTelemetry",config.value("enableLodTransitionTelemetry",false)},
            {"stochasticTextureFiltering",config.value("stochasticTextureFiltering",false)},
            {"benchmarkFreezeStreaming",freezeStreaming},{"deferredOverrides",deferredOverrides},
            {"dlssMode",dlssMode}, {"textureMipTransitionMilliseconds",1000.0 * render::TextureMipTransition::kDurationSeconds},
            {"temporalJitter",viewportView_.temporalJitter()}, {"softwareGroupSize",groupSize},
            {"qualitySamples",requestedQualitySamples}};
        report["absoluteKeyframes"]=Json::array();
        for (const auto& p : points) { report["absoluteKeyframes"].push_back({{"seconds",p.at("t").get<double>()*duration},
            {"stage",p.at("stage")},{"camera",cameraAt(p.at("forward"),p.at("yaw"))}}); }
        report["outputExtent"]={fullRoamWidth_,fullRoamHeight_};
        auto* depth=graphExecutor_->outputResource("VBuffer.depth");
        if (!depth) { throw std::runtime_error("Missing VBuffer render extent"); }
        report["renderExtent"]={depth->desc.width,depth->desc.height};
        report["hidden"]=std::getenv("METALLIC_FULL_ROAM_HIDDEN")!=nullptr;
        const char* noVsync = std::getenv("METALLIC_FULL_ROAM_NO_VSYNC");
        report["vsync"]=!(noVsync && std::string_view(noVsync) == "1");
        report["frameSlots"]=frameSlots_.size();
        report["loadingFrames"]=loadingFrames;
        report["loadingSeconds"]=std::chrono::duration<double>(readyAt-loadingStart).count();
        report["validationRequested"]=debugRuntime_ && std::getenv("METALLIC_DEBUG_VALIDATION");
        report["graph"]=Json::array();
        for (const auto& node : renderGraph_.nodes()) { auto properties=node.properties; properties.merge_patch(node.runtimeProperties);
            report["graph"].push_back({{"name",node.name},{"type",node.type},{"properties",properties}});
            if (node.name == "Deferred") {
                Json actual = Json::object();
                for (auto it = deferredOverrides.begin(); it != deferredOverrides.end(); ++it) { actual[it.key()] = properties.at(it.key()); }
                report["effectiveDeferredOverrides"] = std::move(actual);
            }
        }
        const auto graphGeneration = graphExecutor_->executionStats().graphGeneration;
        struct Sample { uint64_t frame; double seconds, ms; size_t segment; uint64_t availableBytes; Json graphPreparation; bool diagnostic = false; render::vulkan::StreamlineFrameBeginProfile frameBegin; };
        std::vector<Sample> samples;
        samples.reserve(size_t(duration*120));
        profiler_.beginCapture();
        render::profiling::pacingTrace("CaptureBegin");
        const auto start=Clock::now();
        auto previous=start;
        while (true) {
            const auto now=Clock::now();
            const double seconds = routeFrames ? duration * double(samples.size()) / double(routeFrames - 1) :
                std::chrono::duration<double>(now-start).count();
            if (!samples.empty()) { samples.back().ms=std::chrono::duration<double,std::milli>(now-previous).count(); }
            if (routeFrames ? samples.size() >= routeFrames : seconds >= duration) { break; }
            previous=now;
            size_t segment=0;
            while (segment+2<points.size() && seconds/duration >= points[segment+1]["t"].get<double>()) { ++segment; }
            const auto& a=points[segment]; const auto& b=points[segment+1];
            const double u=std::clamp((seconds/duration-a["t"].get<double>())/(b["t"].get<double>()-a["t"].get<double>()),0.0,1.0);
            const auto blend=[&](const char* key) { return a[key].get<double>()*(1-u)+b[key].get<double>()*u; };
            applyViewportCameraProperties(cameraAt(blend("forward"),blend("yaw")),nullptr);
            samples.push_back({profiler_.nextFrameIndex(),seconds,0,segment,device_->memoryBudget().availableBytes});
            const bool diagnostic = workloadEvery && (samples.size() - 1) % workloadEvery == 0;
            samples.back().diagnostic = diagnostic || captureQuality || freezeStreaming || !deferredOverrides.empty();
            qualityCapture->capture = qualitySamples.contains(static_cast<uint32_t>(samples.size() - 1));
            if (workloadEvery) {
                workloadSetting(diagnostic && config.value("softwareWorkloadCounters", true));
                workloadObserver.capture = diagnostic;
                workloadObserver.editorFrame = samples.back().frame;
                workloadObserver.camera = viewportCameraProperties();
                graphExecutor_->setDebugObserver(diagnostic ? &workloadObserver : nullptr);
            }
            if (!draw()) { throw std::runtime_error("Roam rendering failed/cancelled"); }
            samples.back().frameBegin = frameBegin;
            const auto& execution = graphExecutor_->executionStats();
            samples.back().graphPreparation = {{"executionId", execution.executionId},
                {"drainReasonMask", execution.drainReasonMask}, {"externalCompletionCount", execution.externalCompletionCount},
                {"overlapBlockingPasses", execution.overlapBlockingPasses}};
            if (qualityCapture->capture) {
                // Quality-only routes copy the requested execution without an
                // extra draw. Their drains/export time is never performance data.
                qualityCapture->capture = false;
                drain();
                const uint32_t sample = static_cast<uint32_t>(samples.size() - 1);
                const auto directory = std::filesystem::path("Quality") / ("sample" + std::to_string(sample));
                report["qualitySamples"].push_back({{"routeSample",sample},{"frame",samples.back().frame},
                    {"camera",viewportCameraProperties()},{"directory",directory.generic_string()},
                    {"images",qualityCapture->exportAfterDrain(output / directory)}});
                // Release completed copies before collecting the next sample;
                // on failure the current observer survives the outer GPU drain.
                qualityCapture = std::make_unique<RayQualityCapture>();
                qualityCapture->captureReflections = captureReflections;
                graphExecutor_->setDebugObserver(qualityCapture.get());
            }
            if (profiler_.captureOverflow()) { throw std::runtime_error("Profiler capture limit exceeded"); }
            if (viewportTextureWidth_!=fullRoamWidth_ || viewportTextureHeight_!=fullRoamHeight_ ||
                graphExecutor_->executionStats().graphGeneration!=graphGeneration) { throw std::runtime_error("Viewport or graph changed during capture"); }
        }
        render::profiling::pacingTrace("CaptureEnd");
        profiler_.endCapture();
        // Resolve outstanding queries after measurement; no per-frame readback wait.
        drain();
        if (freezeStreaming) { report["streamingFreeze"]["afterCapture"] = streamSnapshot(graphExecutor_->executionStats()); }
        if (workloadEvery) {
            report["workloads"] = workloadObserver.takeAfterDrain();
            graphExecutor_->setDebugObserver(nullptr);
            workloadSetting(false);
        }
        const auto& frames=profiler_.capturedFrames();
        if (frames.size()!=samples.size()) { throw std::runtime_error("CPU frame count mismatch"); }
        std::ofstream frameFile(output/"Frames.jsonl"), uploadFile(output/"Uploads.jsonl");
        frameFile.exceptions(std::ios::badbit|std::ios::failbit); uploadFile.exceptions(std::ios::badbit|std::ios::failbit);
        std::map<std::string,uint32_t> scopeIds;
        std::set<std::pair<uint64_t,uint64_t>> uploads;
        Json definitions=Json::array();
        Json frozenCounters = Json::array();
        size_t missingGpu=0;
        for (size_t i=0; i<frames.size(); ++i) {
            const auto& f=frames[i]; const auto& sample=samples[i];
            if (f.index!=sample.frame || f.profilingOverflow) { throw std::runtime_error("Frame identity/section overflow"); }
            Json row{{"frame",f.index},{"seconds",sample.seconds},{"frameMs",sample.ms},
                {"routeSample",i},{"stage",points[sample.segment]["stage"]},{"availableBytes",sample.availableBytes},
                {"graphPreparation",sample.graphPreparation},{"diagnostic",sample.diagnostic},
                {"scopes",Json::array()},{"streaming",Json::array()}};
            const auto& begin = sample.frameBegin;
            row["streamlineBegin"] = {{"totalMs", begin.totalMs}, {"mutexWaitMs", begin.mutexWaitMs},
                {"tokenMs", begin.tokenMs}, {"optionsMs", begin.optionsMs}, {"sleepMs", begin.sleepMs},
                {"statusMs", begin.statusMs}, {"markerMs", begin.markerMs}, {"frameId", begin.frameId},
                {"active", begin.active}, {"sleepCalled", begin.sleepCalled}, {"optionsUpdated", begin.optionsUpdated},
                {"statusRefreshed", begin.statusRefreshed}, {"effectiveMode", static_cast<uint32_t>(begin.effectiveOptions.mode)},
                {"frameLimitUs", begin.effectiveOptions.frameLimitUs}, {"available", begin.cachedStatus.available},
                {"suspended", begin.cachedStatus.suspended}, {"reportAvailable", begin.cachedStatus.latencyReportAvailable},
                {"reportFrameId", begin.cachedStatus.reportFrameId}, {"renderLatencyMs", begin.cachedStatus.renderLatencyMs},
                {"gpuRenderMs", begin.cachedStatus.gpuRenderMs},
                {"driverTiming", {{"simulationStartUs", begin.cachedStatus.simulationStartUs},
                    {"simulationEndUs", begin.cachedStatus.simulationEndUs},
                    {"renderSubmitStartUs", begin.cachedStatus.renderSubmitStartUs},
                    {"renderSubmitEndUs", begin.cachedStatus.renderSubmitEndUs},
                    {"presentStartUs", begin.cachedStatus.presentStartUs},
                    {"presentEndUs", begin.cachedStatus.presentEndUs},
                    {"gpuRenderStartUs", begin.cachedStatus.gpuRenderStartUs},
                    {"gpuRenderEndUs", begin.cachedStatus.gpuRenderEndUs},
                    {"gpuActiveRenderTimeUs", begin.cachedStatus.gpuActiveRenderTimeUs},
                    {"gpuFrameTimeUs", begin.cachedStatus.gpuFrameTimeUs}}}};
            std::vector<std::string> paths;
            bool graphGpu=false;
            for (size_t n=0; n<f.nodes.size(); ++n) {
                const auto& node=f.nodes[n];
                const std::string path=(n && node.parent<n ? paths[node.parent]+"/" : "")+node.name;
                paths.push_back(path);
                auto [it,inserted]=scopeIds.emplace(path,uint32_t(scopeIds.size()));
                if (inserted) { definitions.push_back({{"id",it->second},{"path",path},{"cpuOnly",node.cpuOnly},{"cpuIdle",node.cpuIdle},{"cpuEditor",node.cpuEditor},
                    {"queue",(node.cpuOnly || node.renderGraphExecutionId==UINT64_MAX) ? "cpu" : node.queue==render::QueueType::Compute ? "compute" : node.queue==render::QueueType::Copy ? "copy" : "graphics"}}); }
                row["scopes"].push_back({{"id",it->second},{"cpuMs",node.cpuMilliseconds},
                    {"gpuMs",node.gpuTimingAvailable ? Json(node.gpuMilliseconds) : Json(nullptr)},
                    {"executionId",node.renderGraphExecutionId==UINT64_MAX ? Json(nullptr) : Json(node.renderGraphExecutionId)}});
                if (node.name=="RenderGraph GPU envelope") { graphGpu=node.gpuTimingAvailable; }
            }
            if (!graphGpu) { ++missingGpu; }
            for (const auto& s : f.streaming) {
                if (groupSize) {
                    const auto identity = s.softwareRasterIdentity.empty() ? Json::object() : Json::parse(s.softwareRasterIdentity);
                    if (identity.value("groupSize", 0u) != groupSize) {
                        throw std::runtime_error("Live roam did not bind requested SW group size");
                    }
                }
                auto streaming = streamSample(s);
                if (freezeStreaming) { recordFrozenCounters(frozenCounters, streaming); }
                row["streaming"].push_back(std::move(streaming));
                if (s.textureUpload.sequence && uploads.insert({s.generation,s.textureUpload.sequence}).second) {
                    auto upload=uploadSample(s.textureUpload); upload["streamGeneration"]=s.generation;
                    uploadFile<<upload.dump()<<'\n';
                }
            }
            frameFile<<row.dump()<<'\n';
        }
        frameFile.close(); uploadFile.close();
        report["scopes"]=std::move(definitions);
        report["frames"]=frames.size(); report["missingGpuFrames"]=missingGpu;
        report["uploadSamples"]=uploads.size(); report["graphGeneration"]=graphGeneration;
        if (freezeStreaming) { report["streamingFreeze"]["captureCounterRanges"] = std::move(frozenCounters); }
        if (missingGpu) { throw std::runtime_error("Incomplete GPU timing coverage"); }
        if (captureQuality) {
            // Readback is deliberately a separate, untimed execution after the
            // measured route. Its completion owns all exported image bytes.
            qualityCapture->capture = true;
            if (!draw()) { throw std::runtime_error("Quality capture rendering failed"); }
            qualityCapture->capture = false;
            drain();
            report["qualityCamera"] = viewportCameraProperties();
            report["qualityImages"] = qualityCapture->exportAfterDrain(output / "Quality");
            report["qualitySamplingScope"] = "Optional requested deterministic route executions plus one extra final-route execution. Readback drains perturb quality runs; none of their timings are performance acceptance data.";
            report["qualityScope"] = captureReflections
                ? "One untimed final-route frame; shadow mask, deferred lighting, final output and fixed-sample OpenPBR path tracing including reflections. The diagnostic graph is excluded from performance comparisons."
                : "One untimed final-route frame; shadow mask, deferred lighting and final output. This scene has no secondary reflection pass.";
        }
        }
        report["status"]="capture_complete";
        passed=true;
    } catch (const std::exception& error) {
        report["error"]=error.what();
        spdlog::error("[Full Roam] {}",error.what());
    }
    profiler_.endCapture();
    // Keep readback allocations alive until all submitted copies complete, also on failure.
    if (!frameSubmissions_.wait() || !graphExecutor_->waitForSubmittedWork()) { passed = false; report["status"] = "failed"; }
    for (const auto& property : frozenProperties) {
        if (const auto* node = renderGraph_.findNode(property.node)) {
            auto properties = node->runtimeProperties;
            if (property.existed) { properties[property.key] = property.value; }
            else { properties.erase(property.key); }
            if (!renderGraph_.setNodeRuntimeProperties(property.node, std::move(properties))) { passed = false; report["status"] = "failed"; }
        } else { passed = false; report["status"] = "failed"; }
    }
    graphExecutor_->setDebugObserver(debugRuntime_.get());
    if (controlledLighting) { renderWorld_.setLighting(std::move(*controlledLighting)); }
    fullRoamActive_=false; fullRoamWidth_=fullRoamHeight_=0;
    profiler_.beginCapture(); profiler_.endCapture(); // Release retained capture samples after export.
    try {
        std::ofstream file(output/"Capture.json");
        file.exceptions(std::ios::badbit|std::ios::failbit);
        file<<report.dump(2)<<'\n';
    } catch (const std::exception& error) { spdlog::error("[Full Roam] Export failed: {}",error.what()); passed=false; }
    spdlog::info("[Full Roam] {}: {}",passed ? "Capture complete" : "Failed",output.string());
    return passed;
}
} // namespace metallic

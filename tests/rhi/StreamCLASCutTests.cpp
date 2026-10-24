#include "RHITest.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Subsystem/GPUScene.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace metallic::tests {
namespace {

using namespace render;

class StreamHistoryLayoutEnvironment {
public:
    explicit StreamHistoryLayoutEnvironment(bool packed)
    {
        if (const char* previous = std::getenv("METALLIC_STREAM_RT_HISTORY_PACKED")) { previous_ = previous; }
        set(packed ? "1" : "0");
    }
    ~StreamHistoryLayoutEnvironment() { set(previous_ ? previous_->c_str() : nullptr); }
private:
    static void set(const char* value)
    {
#ifdef _WIN32
        _putenv_s("METALLIC_STREAM_RT_HISTORY_PACKED", value ? value : "");
#else
        if (value) { setenv("METALLIC_STREAM_RT_HISTORY_PACKED", value, 1); }
        else { unsetenv("METALLIC_STREAM_RT_HISTORY_PACKED"); }
#endif
    }
    std::optional<std::string> previous_;
};

class StreamCLASCutReadinessTest final : public RHITest {
public:
    StreamCLASCutReadinessTest()
    {
        type = RHITestType::Rendering;
        name = "stream_clas_cut_readiness";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const auto require = [](bool condition, const std::string& reason) {
            if (!condition) { throw std::runtime_error(reason); }
        };
        std::atomic_uint validationErrors = 0;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "CLAS-ready streaming cut",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true, .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true, .enableClusterAccelerationStructure = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}}).transform([&](auto value) { device = std::move(value); });
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("CLAS unavailable")
                : RHITestResult::fail("CLAS cut device creation failed");
        }
        try {
            const auto sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
            const auto assetPath = std::filesystem::absolute(context.outputDirectory / "CLASCut.meshstream.bin");
            scene::Scene scene;
            require(scene.load(sourcePath), scene.lastLoadResult().error);
            std::string log;
            require(scene::buildMeshletStreamAsset({.scene = &scene, .sourcePath = sourcePath,
                .outputPath = assetPath, .compressionMode = scene::MeshletStreamPayloadCompression::ByteRle}, log), log);
            scene::MeshletStreamAsset asset;
            require(asset.open(assetPath, log), log);
            require(asset.primitiveCount() == 1 && asset.instanceCount() == 1 &&
                asset.primitives()[0].lodLevelCount >= 3, "Fixture needs one multi-level refinement hierarchy");
            const uint32_t intermediate = std::min(2u, asset.primitives()[0].lodLevelCount - 2u);
            // Runtime reserves the cooked maximum, which can exceed every
            // actual page's population after meshlet grouping.
            const uint32_t maxPageClusters = asset.maxPageClusters();
            require(maxPageClusters != 0, "Fixture has no clusters");
            std::vector<MeshletLODGroupRecord> metrics;
            std::vector<MeshletLODGroupRange> ranges;
            for (const auto& group : asset.groups()) {
                MeshletLODGroupRecord metric;
                std::copy_n(group.boundsCenterRadius, 4, metric.sphere.begin());
                metric.error = group.maxQuadricError;
                metric.level = group.lodLevel;
                metric.flags = group.flags;
                metrics.push_back(metric);
                ranges.push_back({group.clusterRefinedOffset, group.clusterCount});
            }
            std::vector<MeshletLODRefinementBounds> refinementBounds;
            require(buildMeshletLodRefinementBounds(metrics, ranges, asset.refinedGroups(), refinementBounds, log), log);
            GPUSceneGPUInstanceRecord metricInstance;
            std::copy_n(asset.instances()[0].worldMatrix, 16, metricInstance.worldMatrix.begin());
            uint32_t targetGroup = UINT32_MAX;
            for (uint32_t group = 0; group < metrics.size(); ++group) {
                if ((metrics[group].flags & kMeshletLODTerminalGroup) == 0 && metrics[group].error > 0 &&
                    (targetGroup == UINT32_MAX || metrics[group].level > metrics[targetGroup].level)) {
                    targetGroup = group;
                }
            }
            require(targetGroup != UINT32_MAX, "Fixture has no finite nonterminal demand group");
            nlohmann::json report = nlohmann::json::array();
            for (uint32_t layoutCase = 0; layoutCase < 4; ++layoutCase) {
                const bool compact = (layoutCase & 1u) != 0u;
                const bool packedHistory = (layoutCase & 2u) != 0u;
                StreamHistoryLayoutEnvironment historyEnvironment(packedHistory);
                for (bool distributed : {false, true}) {
                    MeshletStreamRuntime runtime;
                    runtime.setDebugReadbackEnabled(true);
                    const uint32_t groupCapacity = std::min(2048u, asset.groupCount());
                    constexpr uint64_t kGeometryBudget = 32ull << 20;
                    constexpr uint64_t kCLASBudget = 64ull << 20;
                    // This test compares the per-instance frontier directly
                    // with BLAS semantic changes. Shared/discrete owners may
                    // legally publish a different cut and are covered by the
                    // representation tests and full-scene A/B validation.
                    MeshletStreamRuntimeDesc desc{.sourcePath = sourcePath, .streamAssetPath = assetPath,
                        .maxResidentBytes = kGeometryBudget, .maxResidentPages = 1024,
                        .maxPageUploadsPerFrame = 256, .maxGpuPageRequests = 1024, .maxGpuPageUnloadRequests = 1024,
                        .maxActiveGroups = groupCapacity, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
                        .pageLoadConcurrency = 0, .maxPageLoadsInFlight = 256, .queuedFrameCount = 2,
                        .enableClusterRtx = true, .enableClas = true, .compactClas = compact,
                        .maxClasBytes = kCLASBudget, .growClasBytes = 4ull << 20, .maxClasBuildClusters = maxPageClusters,
                        .maxBlasClusterReferences = 8192, .maxBlasBytes = 16ull << 20, .maxBlasBuilds = 4,
                        .enableBlasRepresentations = false,
                        .maxFallbackBlasBytes = 4ull << 20, .distributedPageDemand = distributed,
                        .distributedDemandMinGroups = 0, .prefetchPages = false, .predictivePrefetch = false,
                        .enableLodTransitionTelemetry = compact && distributed};
                    require(bool(runtime.initialize(*device, desc, log)), log);
                    const uint64_t initialShadingRevision = runtime.shadingRevision();
                    auto& queue = *device->getQueue(QueueType::Graphics);
                    QueueSubmissionTracker tracker;
                    RenderFrameContext frame;
                    std::unique_ptr<CommandPool> pool;
                    std::unique_ptr<CommandBuffer> commands;
                    std::unique_ptr<Streamer> streamer;
                    std::unique_ptr<Buffer> readback;
                    const uint64_t groupsOffset = sizeof(MeshletStreamGPUBLASHeader) + sizeof(MeshletStreamGPUActiveHeader);
                    const uint64_t pagesOffset = groupsOffset + groupCapacity * sizeof(MeshletStreamGPUActiveGroup);
                    const uint64_t lodOffset = pagesOffset + asset.pageCount() * sizeof(StreamPageTableEntry);
                    const uint32_t historyOffset = 4u + asset.groupCount() * 3u + 4u;
                    const uint32_t historyStride = packedHistory ? 1u : 2u;
                    const uint32_t lodWords = historyOffset + 3u + asset.groupCount() * historyStride;
                    const uint64_t tlasOffset = lodOffset + lodWords * sizeof(uint32_t);
                    const uint64_t clasOffset = tlasOffset + sizeof(RayTracingGPUInstance);
                    const uint64_t readbackBytes = clasOffset + asset.pageCount() * sizeof(MeshletStreamCLASPageEntry);
                    StreamerDesc streamerDesc;
                    streamerDesc.constantBufferSize = 4096;
                    streamerDesc.dynamicBufferSizePerFrame = 1024;
                    streamerDesc.queuedFrameCount = 2;
                    streamerDesc.dynamicBufferDesc.usage = BufferUsageBits::TransferSource;
                    require(bool(tracker.initialize(*device, queue)) &&
                        bool(device->createCommandPool(queue).transform([&](auto value) { pool = std::move(value); })) &&
                        bool(pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); })) &&
                        bool(createStreamer(*device, streamerDesc).transform([&](auto value) { streamer = std::move(value); })) &&
                        bool(device->createBuffer({.size = readbackBytes, .usage = BufferUsageBits::TransferDestination,
                            .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { readback = std::move(value); })),
                        "Cut test frame resources failed");
                    struct Drain {
                        Queue& queue;
                        RenderFrameContext& frame;
                        ~Drain() { frame.cancel(); (void)queue.waitIdle(); }
                    } drain{queue, frame};
                    MeshletStreamFrameDesc view{.width = 192, .height = 128,
                        .selectedLodLevel = intermediate, .enableGpuLodSelection = false};
                    view.rtDemandMode = compact ? 2u : 1u;
                    view.camera = {.eye = {-.0168404f, .110154f, .22f},
                        .center = {-.0168404f, .110154f, -.00153695f}, .znear = .001f, .zfar = 10.f};
                    uint64_t frameIndex = 0;
                    MeshletStreamGPUBLASHeader blas{};
                    RayTracingGPUInstance tlasInstance{};
                    std::vector<MeshletStreamGPUActiveGroup> active;
                    std::vector<StreamPageTableEntry> pages(asset.pageCount());
                    std::vector<MeshletStreamCLASPageEntry> clas(asset.pageCount());
                    std::vector<uint32_t> masks(asset.groupCount());
                    std::vector<uint32_t> lodState(lodWords), previousLodState(lodWords);
                    MeshletStreamGPUParams params{}, previousParams{};
                    uint64_t peakGeometryBytes = 0, peakCLASBytes = 0;
                    bool cancelledInitialFallback = false;
                    bool cancelledInitialFrame = false;
                    const auto record = [&](bool cancel = false) {
                        runtime.prepareMaintenance();
                        require(runtime.shadingRevision() == initialShadingRevision,
                            "Delayed CLAS publication/readback invalidated CPU shading history a second time");
                        require(bool(frame.begin(++frameIndex)) && bool(pool->reset()) &&
                            bool(commands->begin(frame.submissionContext())) && bool(streamer->beginFrame(frame)), "Cut frame begin failed");
                        require(bool(runtime.cmdBeginFrame(*commands, *streamer, view)) &&
                            bool(streamer->copyStreamedData(*commands)) && bool(runtime.cmdPreTraversal(*commands, view)) &&
                            bool(runtime.cmdPostTraversal(*commands)) && bool(runtime.cmdEndFrame(*commands)), "Cut frame recording failed");
                        std::vector<DebugResourceBinding> bindings;
                        runtime.appendDebugBindings(bindings, "cut.");
                        const auto copy = [&](const char* id, uint64_t offset, uint64_t bytes) {
                            const auto source = std::find_if(bindings.begin(), bindings.end(),
                                [&](const auto& binding) { return binding.id == id; });
                            require(source != bindings.end() && source->buffer->desc().size >= bytes,
                                std::string("Cut telemetry buffer missing or too small: ") + id);
                            BufferBarrierDesc barrier{.buffer = source->buffer,
                                .before = resourceSyncScope(source->state, PipelineStageBits::AllCommands),
                                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                            require(bool(commands->synchronize({.buffers = {&barrier, 1}})), "Cut readback barrier failed");
                            auto src = source->buffer->slice({0, bytes});
                            auto dst = readback->slice({offset, bytes});
                            require(bool(src) && bool(dst) && bool(commands->copyBuffer(*src, *dst)), "Cut readback copy failed");
                            std::swap(barrier.before, barrier.after);
                            require(bool(commands->synchronize({.buffers = {&barrier, 1}})), "Cut readback restore failed");
                        };
                        copy("cut.blasHeader", 0, sizeof(blas));
                        copy("cut.activeHeader", sizeof(blas), sizeof(MeshletStreamGPUActiveHeader));
                        copy("cut.activeGroups", groupsOffset, groupCapacity * sizeof(MeshletStreamGPUActiveGroup));
                        copy("cut.pageTable", pagesOffset, pages.size() * sizeof(StreamPageTableEntry));
                        copy("cut.lodState", lodOffset, lodWords * sizeof(uint32_t));
                        copy("cut.tlasInstances", tlasOffset, sizeof(tlasInstance));
                        // Compact GPU allocation publishes into device memory;
                        // inspect the completed submission through a readback,
                        // rather than assuming the publication table is mappable.
                        if (runtime.clasPool()->usesGPUAllocation()) {
                            auto* clasTable = runtime.clasPool()->pageTableBuffer();
                            BufferBarrierDesc clasBarrier{.buffer = clasTable,
                                .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                            require(bool(commands->synchronize({.buffers = {&clasBarrier, 1}})), "CLAS publication readback barrier failed");
                            auto clasSource = clasTable->slice({0, clas.size() * sizeof(clas[0])});
                            auto clasDestination = readback->slice({clasOffset, clas.size() * sizeof(clas[0])});
                            require(bool(clasSource) && bool(clasDestination) &&
                                bool(commands->copyBuffer(*clasSource, *clasDestination)), "CLAS publication readback copy failed");
                            std::swap(clasBarrier.before, clasBarrier.after);
                            require(bool(commands->synchronize({.buffers = {&clasBarrier, 1}})), "CLAS publication readback restore failed");
                        }
                        require(bool(commands->end()), "Cut frame end failed");
                        if (!cancelledInitialFrame) {
                            cancelledInitialFrame = true;
                            cancel = true;
                        }
                        const auto readiness = runtime.debugSnapshot(false);
                        if (!cancelledInitialFallback &&
                            readiness.at("fallbackBlasRecorded") > readiness.at("fallbackBlasSubmitted")) {
                            require(!runtime.sceneReady(), "Unsubmitted fallback build made the scene ready");
                            cancelledInitialFallback = true;
                            frame.cancel();
                            streamer->endFrame();
                            require(!runtime.sceneReady(), "Cancelled fallback build poisoned readiness");
                            return false;
                        }
                        if (cancel) {
                            frame.cancel();
                            streamer->endFrame();
                            return false;
                        }
                        CommandBuffer* list[] = {commands.get()};
                        require(bool(tracker.submit({.commandBuffers = {list, 1}}, frame)) &&
                            bool(frame.wait(5000000000ull)), "Cut frame submission failed");
                        streamer->endFrame();
                        readback->invalidate();
                        const auto* data = static_cast<const uint8_t*>(readback->map());
                        require(data != nullptr, "Cut readback map failed");
                        std::memcpy(&blas, data, sizeof(blas));
                        MeshletStreamGPUActiveHeader header;
                        std::memcpy(&header, data + sizeof(blas), sizeof(header));
                        require(header.activeGroupCount <= groupCapacity && header.overflowCount == 0, "Cut capacity overflow");
                        active.resize(header.activeGroupCount);
                        std::memcpy(active.data(), data + groupsOffset, active.size() * sizeof(active[0]));
                        std::memcpy(pages.data(), data + pagesOffset, pages.size() * sizeof(pages[0]));
                        previousLodState = lodState;
                        std::memcpy(lodState.data(), data + lodOffset, lodState.size() * sizeof(uint32_t));
                        std::memcpy(&tlasInstance, data + tlasOffset, sizeof(tlasInstance));
                        if (runtime.clasPool()->usesGPUAllocation()) {
                            std::memcpy(clas.data(), data + clasOffset, clas.size() * sizeof(clas[0]));
                        }
                        readback->unmap();
                        previousParams = params;
                        auto* paramsBuffer = runtime.deferredGpuResources().paramsBuffer;
                        paramsBuffer->invalidate();
                        const auto* paramsData = paramsBuffer->map();
                        require(paramsData != nullptr, "Stream parameters map failed");
                        std::memcpy(&params, paramsData, sizeof(params));
                        paramsBuffer->unmap();
                        if (!runtime.clasPool()->usesGPUAllocation()) {
                            auto* table = runtime.clasPool()->pageTableBuffer();
                            table->invalidate();
                            const auto* tableData = table->map();
                            require(tableData != nullptr, "Host CLAS publication table map failed");
                            std::memcpy(clas.data(), tableData, clas.size() * sizeof(clas[0]));
                            table->unmap();
                        }
                        std::fill(masks.begin(), masks.end(), 0);
                        for (const auto& group : active) {
                            require(group.pageIndex < clas.size() &&
                                meshletStreamClasPageState(clas[group.pageIndex]) == MeshletStreamCLASPageState::Active,
                                "Published RT cut references geometry whose CLAS is not ready");
                            const auto state = streamPageTableState(pages[group.pageIndex]);
                            require(state == MeshletStreamPageResidencyState::Resident || state == MeshletStreamPageResidencyState::LockedFallback,
                                "Published RT cut lost its geometry page");
                            masks[asset.pages()[group.pageIndex].lodGroupIndex] |= group.clusterSelectionMask;
                        }
                        const auto geometry = runtime.residency().stats();
                        const auto acceleration = runtime.clasPool()->stats();
                        peakGeometryBytes = std::max(peakGeometryBytes, geometry.usedResidentBytes);
                        peakCLASBytes = std::max(peakCLASBytes, acceleration.storageBytes);
                        require(geometry.usedResidentBytes <= kGeometryBudget && geometry.usedSlotCount <= desc.maxResidentPages &&
                            acceleration.storageBytes <= kCLASBudget && acceleration.usedStorageBytes <= acceleration.storageBytes,
                            "Retained cut escaped geometry or CLAS budgets");
                        return true;
                    };
                    uint32_t stable = 0;
                    std::vector<uint32_t> previousMasks;
                    for (uint32_t attempt = 0; attempt < 400 && stable < 8; ++attempt) {
                        if (!record()) { continue; }
                        const bool intermediateReady = runtime.tlasReady() && !active.empty() &&
                            std::any_of(active.begin(), active.end(), [&](const auto& group) { return group.lodLevel == intermediate; });
                        stable = intermediateReady && masks == previousMasks ? stable + 1 : 0;
                        previousMasks = masks;
                    }
                    require(stable == 8, "Intermediate cut never settled");
                    require(blas.semanticChangeCount == 0, "Settled intermediate cut repeatedly invalidated shading history");
                    require(cancelledInitialFallback && runtime.sceneReady(),
                        "Cancelled initial fallback did not retry and restore scene readiness");
                    const auto warmMasks = masks;
                    const uint32_t warmReferences = blas.liveClusterReferences;
                    require(warmReferences != 0, "Intermediate cut used whole-instance fallback");
                    uint32_t pendingFrames = 0, retainedFrames = 0, fineFrames = 0;
                    view.selectedLodLevel = 0;
                    for (uint32_t attempt = 0; attempt < 600 && fineFrames < 8; ++attempt) {
                        require(record(), "Unexpected fallback cancellation after scene became ready");
                        require(blas.missingClasInstances == 0 && blas.referenceBudgetRejected == 0 &&
                            blas.buildBudgetRejected == 0 && blas.storageRejected == 0 && blas.liveClusterReferences != 0,
                            "Incomplete finer CLAS forced a whole-instance BLAS fallback");
                        // Each previously selected cluster must still be present or be
                        // replaced by a complete recursively covered child group.
                        std::vector<int8_t> covered(asset.groupCount(), -1);
                        std::function<bool(uint32_t)> groupCovered;
                        const auto clusterCovered = [&](uint32_t group, uint32_t cluster) {
                            if ((masks[group] & (1u << cluster)) != 0) { return true; }
                            const auto child = asset.refinedGroups()[asset.groups()[group].clusterRefinedOffset + cluster];
                            return child != UINT32_MAX && child < asset.groupCount() && groupCovered(child);
                        };
                        groupCovered = [&](uint32_t group) {
                            if (covered[group] >= 0) { return covered[group] != 0; }
                            covered[group] = 0;
                            for (uint32_t cluster = 0; cluster < asset.groups()[group].clusterCount; ++cluster) {
                                if (!clusterCovered(group, cluster)) { return false; }
                            }
                            covered[group] = 1;
                            return true;
                        };
                        bool retained = false;
                        for (uint32_t group = 0; group < warmMasks.size(); ++group) {
                            for (uint32_t mask = warmMasks[group]; mask != 0; mask &= mask - 1) {
                                require(clusterCovered(group, std::countr_zero(mask)), "Refinement regressed or lost the previous legal cut");
                            }
                            retained |= (warmMasks[group] & masks[group]) != 0;
                        }
                        bool pendingFine = false;
                        for (uint32_t page = 0; page < pages.size(); ++page) {
                            const auto state = streamPageTableState(pages[page]);
                            pendingFine |= asset.pages()[page].lodLevel < intermediate &&
                                (state == MeshletStreamPageResidencyState::Resident || state == MeshletStreamPageResidencyState::LockedFallback) &&
                                meshletStreamClasPageState(clas[page]) != MeshletStreamCLASPageState::Active;
                        }
                        pendingFrames += pendingFine;
                        retainedFrames += pendingFine && retained;
                        const bool fine = !active.empty() && std::all_of(active.begin(), active.end(),
                            [](const auto& group) { return group.lodLevel == 0; });
                        fineFrames = fine ? fineFrames + 1 : 0;
                    }
                    require(fineFrames == 8, "Finer CLAS never replaced the retained cut");
                    require(blas.semanticChangeCount == 0, "Settled fine cut repeatedly invalidated shading history");
                    require(pendingFrames != 0 && retainedFrames != 0,
                        "Fixture did not expose loaded finer geometry waiting for CLAS while retaining an ancestor");

                    const auto cpuView = [&] {
                        MeshletLODView result;
                        std::copy_n(params.eye, 3, result.eye.begin());
                        result.eye[3] = params.clipOrtho[0];
                        float distanceSquared = 0;
                        for (uint32_t axis = 0; axis < 3; ++axis) {
                            result.forward[axis] = params.center[axis] - params.eye[axis];
                            distanceSquared += result.forward[axis] * result.forward[axis];
                        }
                        const float distance = std::sqrt(distanceSquared);
                        for (uint32_t axis = 0; axis < 3; ++axis) { result.forward[axis] /= distance; }
                        result.forward[3] = params.upProjection[3];
                        result.projection = {params.viewport[2], std::tan(params.viewport[3] * .5f),
                            params.clipOrtho[2], params.lodPixelError};
                        return result;
                    };
                    const auto wantedGroups = [&] {
                        std::vector<uint8_t> wanted(asset.groupCount());
                        for (uint32_t group = 0; group < wanted.size(); ++group) {
                            const uint32_t address = historyOffset + 3u + group * historyStride;
                            wanted[group] = (metrics[group].flags & kMeshletLODTerminalGroup) != 0 ||
                                (lodState[address] == lodState[historyOffset + 1u] && (packedHistory || lodState[address + 1u] != 0));
                        }
                        return wanted;
                    };
                    uint32_t verifiedPolicyFrames = 0;
                    std::string policyStage = "hysteresis roundtrip";
                    const auto verifyPolicy = [&] {
                        const bool continuous = params.enableGpuLodSelection != 0 &&
                            previousParams.rtLodPolicyRevision == params.rtLodPolicyRevision;
                        const uint32_t expectedPrevious = continuous ? previousLodState[historyOffset + 1u] : 0u;
                        require(lodState[historyOffset] == expectedPrevious &&
                            lodState[historyOffset + 1u] == params.rtLodFrameToken && params.rtLodFrameToken != 0u &&
                            lodState[historyOffset + 2u] == params.rtLodPolicyRevision &&
                            ((params.rtLodFlags & 16u) != 0u) == packedHistory,
                            "RT GPU history token/revision did not follow the last submitted frame");
                        const auto metricView = cpuView();
                        const auto wanted = wantedGroups();
                        for (uint32_t group = 0; group < metrics.size(); ++group) {
                            if ((metrics[group].flags & kMeshletLODTerminalGroup) != 0) { continue; }
                            const uint32_t address = historyOffset + 3u + group * historyStride;
                            const bool wasWanted = continuous && expectedPrevious != 0 &&
                                previousLodState[address] == expectedPrevious && (packedHistory || previousLodState[address + 1u] != 0);
                            const bool visible = meshletLodBoundsVisible(refinementBounds[group], metricInstance, metricView,
                                {params.upProjection[0], params.upProjection[1], params.upProjection[2]},
                                params.viewport[0], params.clipOrtho[1]);
                            float error = meshletLodRayPixelError(metrics[group], metricInstance, metricView);
                            if (visible && (params.rtLodFlags & 2u) != 0) {
                                error = std::max(error, meshletLodPixelError(metrics[group], metricInstance, metricView));
                            }
                            const float threshold = params.lodPixelError * (visible ? 1.f : params.rtOffscreenErrorScale);
                            const bool expectedWanted = meshletLodRayNeedsFine(error, threshold, params.rtLodHysteresis, wasWanted);
                            const float boundary = threshold * (wasWanted ? 1.f - params.rtLodHysteresis : 1.f);
                            // Floating-point evaluation may land on opposite sides of
                            // an exact threshold; all directed scenarios stay clear of it.
                            if (std::abs(error - boundary) > std::max(1e-5f, boundary * 1e-4f)) {
                                require(bool(wanted[group]) == expectedWanted,
                                    "CPU/GPU RT demand mismatch at group " + std::to_string(group) +
                                    " frame " + std::to_string(params.frameIndex));
                            }
                        }
                        const bool publishedCoverage = runtime.tlasReady() && tlasInstance.accelerationStructureReference != 0 &&
                            (tlasInstance.customIndexAndMask >> 24u) != 0;
                        require(blas.missingClasInstances == 0 && blas.liveClusterReferences != 0 && publishedCoverage,
                            "RT policy transition lost ready CLAS coverage: stage=" + policyStage +
                            ", frame=" + std::to_string(params.frameIndex) +
                            ", verified=" + std::to_string(verifiedPolicyFrames) +
                            ", compact=" + std::to_string(compact) + ", distributed=" + std::to_string(distributed) +
                            ", missing=" + std::to_string(blas.missingClasInstances) +
                            ", live=" + std::to_string(blas.liveClusterReferences) +
                            ", builds=" + std::to_string(blas.blasBuildCount) +
                            ", reused=" + std::to_string(blas.reusedInstances) +
                            ", tlasVisible=" + std::to_string(publishedCoverage) +
                            ", referenceRejected=" + std::to_string(blas.referenceBudgetRejected) +
                            ", storageRejected=" + std::to_string(blas.storageRejected));
                        ++verifiedPolicyFrames;
                    };
                    const auto policyFrame = [&] {
                        require(record(), "Unexpected RT policy cancellation");
                        verifyPolicy();
                    };
                    const auto& target = metrics[targetGroup];
                    const auto& world = metricInstance.worldMatrix;
                    const float3 targetCenter(
                        world[0] * target.sphere[0] + world[4] * target.sphere[1] + world[8] * target.sphere[2] + world[12],
                        world[1] * target.sphere[0] + world[5] * target.sphere[1] + world[9] * target.sphere[2] + world[13],
                        world[2] * target.sphere[0] + world[6] * target.sphere[1] + world[10] * target.sphere[2] + world[14]);
                    MeshletLODView unitView;
                    unitView.forward[3] = 1;
                    unitView.projection = {1, 1, 1, 1};
                    const float worldError = meshletLodRayPixelError(target, metricInstance, unitView);
                    const float worldRadius = target.sphere[3] * worldError / target.error;
                    const auto placeAtErrorRatio = [&](float ratio) {
                        const float focal = float(view.displayHeight) /
                            (2.f * std::tan(view.camera.fovDegrees * 3.14159265359f / 360.f));
                        const float distance = worldRadius + worldError + worldError * focal / (view.lodPixelError * ratio);
                        view.camera.eye = targetCenter + float3(0, 0, distance);
                        view.camera.center = targetCenter;
                        view.camera.zfar = std::max(10.f, distance * 4.f);
                    };
                    view.enableGpuLodSelection = true;
                    view.displayHeight = 128;
                    view.lodPixelError = 1.5f;
                    view.rtOffscreenErrorScale = 1.f;
                    view.rtLodHysteresis = .2f;
                    struct ThresholdStep { float ratio; bool wanted; };
                    uint32_t thresholdStep = 0;
                    for (const auto step : {ThresholdStep{.75f, false}, {.9f, false}, {1.05f, true}, {.9f, true}, {.75f, false}}) {
                        placeAtErrorRatio(step.ratio);
                        if (thresholdStep++ == 3) {
                            require(!record(true), "Cancelled RT history frame was submitted");
                        }
                        policyFrame();
                        require(bool(wantedGroups()[targetGroup]) == step.wanted,
                            "RT dual-threshold round trip changed detail inside the hysteresis band");
                    }
                    // Returning to the same threshold with a quality revision must
                    // discard wanted history; changing internal resolution must not.
                    policyStage = "quality/manual history revision";
                    placeAtErrorRatio(1.05f);
                    policyFrame();
                    const uint32_t wantedRevision = params.rtLodPolicyRevision;
                    placeAtErrorRatio(.9f);
                    view.rtLodHysteresis = .3f;
                    policyFrame();
                    require(params.rtLodPolicyRevision != wantedRevision && !wantedGroups()[targetGroup],
                        "Quality change reused the previous policy's fine demand");
                    view.rtLodHysteresis = .2f;
                    placeAtErrorRatio(1.05f);
                    policyFrame();
                    view.enableGpuLodSelection = false;
                    view.selectedLodLevel = 0;
                    require(record(), "Manual policy recording failed");
                    const uint32_t manualRevision = params.rtLodPolicyRevision;
                    view.enableGpuLodSelection = true;
                    placeAtErrorRatio(.9f);
                    policyFrame();
                    require(params.rtLodPolicyRevision != manualRevision && !wantedGroups()[targetGroup],
                        "Manual-to-auto transition reused stale fine demand");

                    view.rtLodHysteresis = 0.f;
                    policyStage = "rotation with unit offscreen scale";
                    placeAtErrorRatio(1.5f);
                    policyFrame();
                    const auto facingMasks = masks;
                    const auto facingDemand = wantedGroups();
                    const float3 eye = view.camera.eye;
                    view.camera.center = eye + float3(0, 0, 1);
                    policyFrame();
                    require(masks == facingMasks && wantedGroups() == facingDemand,
                        "RT distance cut changed after a 180-degree rotation with unit offscreen scale");
                    view.camera.center = eye + float3(1, 0, 0);
                    policyFrame();
                    require(masks == facingMasks && wantedGroups() == facingDemand,
                        "RT distance cut changed after a 90-degree rotation with unit offscreen scale");
                    uint32_t reducedGroups = 0;
                    policyStage = "offscreen scale";
                    float offscreenIntermediateRatio = 0.f;
                    bool offscreenIntermediate = false;
                    bool offscreenSemanticChange = false;
                    for (const float ratio : {1.5f, 2.5f, 4.f, 8.f, 16.f, 32.f}) {
                        view.rtOffscreenErrorScale = 1.f;
                        placeAtErrorRatio(ratio);
                        policyFrame();
                        const auto onScreen = wantedGroups();
                        const auto onScreenMasks = masks;
                        view.camera.center = view.camera.eye + float3(0, 0, 1);
                        view.rtOffscreenErrorScale = 2.f;
                        policyFrame();
                        if (masks != onScreenMasks) {
                            require(blas.semanticChangeCount == 1,
                                "Offscreen cut change omitted the global same-frame shading invalidation");
                            offscreenSemanticChange = true;
                        }
                        const auto offscreenDemand = wantedGroups();
                        reducedGroups = 0;
                        for (uint32_t group = 0; group < onScreen.size(); ++group) {
                            require(offscreenDemand[group] <= onScreen[group], "Offscreen multiplier increased RT detail demand");
                            reducedGroups += onScreen[group] && !offscreenDemand[group];
                        }
                        offscreenIntermediate = std::any_of(active.begin(), active.end(), [&](const auto& group) {
                            return group.lodLevel < asset.primitives()[0].lodLevelCount - 1u;
                        });
                        if (reducedGroups != 0 && offscreenIntermediate) { offscreenIntermediateRatio = ratio; break; }
                    }
                    require(reducedGroups != 0 && offscreenIntermediate,
                        "Offscreen policy failed to reduce demand while preserving non-root geometry");
                    require(offscreenSemanticChange, "Offscreen fixture did not exercise a published semantic cut change");
                    policyFrame();
                    require(blas.semanticChangeCount == 0, "Offscreen semantic change invalidated another stable frame");
                    view.camera.center = targetCenter;
                    policyStage = "internal resolution invariance";
                    view.rtOffscreenErrorScale = 1.f;
                    policyFrame();
                    const auto nativeMasks = masks;
                    const auto nativeDemand = wantedGroups();
                    const uint32_t nativeRevision = params.rtLodPolicyRevision;
                    for (const auto [width, height] : {std::pair{96u, 64u}, {384u, 256u}, {192u, 128u}}) {
                        view.width = width;
                        view.height = height;
                        policyFrame();
                        require(params.rtLodPolicyRevision == nativeRevision && masks == nativeMasks && wantedGroups() == nativeDemand,
                            "Internal render scaling reset RT history or changed display-pixel quality");
                    }
                    // A visible shared producer may refine more than pure RT;
                    // outside the primary frustum it follows the same RT policy.
                    policyStage = "raster preservation";
                    placeAtErrorRatio(.999f);
                    // Keep the target inside the horizontal frustum but away
                    // from its axis, where projection error exceeds distance
                    // error even for a very small authored replacement sphere.
                    constexpr float kObliqueAngle = 25.f * 3.14159265359f / 180.f;
                    view.camera.center = view.camera.eye + float3(std::sin(kObliqueAngle), 0, -std::cos(kObliqueAngle));
                    policyFrame();
                    const auto pureDemand = wantedGroups();
                    view.preserveRasterLod = true;
                    policyFrame();
                    const auto sharedDemand = wantedGroups();
                    uint32_t rasterAddedGroups = 0;
                    for (uint32_t group = 0; group < pureDemand.size(); ++group) {
                        require(sharedDemand[group] >= pureDemand[group], "Shared raster policy weakened visible RT quality");
                        rasterAddedGroups += sharedDemand[group] && !pureDemand[group];
                    }
                    require(rasterAddedGroups != 0, "Shared raster fixture failed to expose stronger projection quality");
                    view.camera.center = view.camera.eye + float3(0, 0, 1);
                    policyFrame();
                    const auto sharedOffscreen = wantedGroups();
                    view.preserveRasterLod = false;
                    policyFrame();
                    require(wantedGroups() == sharedOffscreen, "Offscreen shared raster demand differed from pure RT");

                    // Reuse the proven intermediate offscreen pose so this
                    // sequence tests dynamic cache hits, not terminal fallback.
                    placeAtErrorRatio(offscreenIntermediateRatio);
                    view.camera.center = view.camera.eye + float3(0, 0, 1);
                    view.rtOffscreenErrorScale = 2.f;
                    view.preserveRasterLod = true;
                    policyFrame();
                    const auto stableOffscreenMasks = masks;
                    const auto stableOffscreenDemand = wantedGroups();
                    policyStage = "32 stable offscreen frames";
                    for (uint32_t frame = 0; frame < 32; ++frame) {
                        if (frame == 8) { require(!record(true), "Cancelled stable offscreen frame was submitted"); }
                        policyFrame();
                        require(masks == stableOffscreenMasks && wantedGroups() == stableOffscreenDemand,
                            "C1 changed stable offscreen demand/cut across consecutive frames");
                        require(blas.blasBuildCount == 0 && blas.reusedInstances != 0 &&
                            blas.cacheDependencyInvalidations == 0 && blas.semanticChangeCount == 0,
                            "Stable raster-preserving offscreen demand lost cached BLAS reuse or triggered a rebuild");
                    }

                    view.rtOffscreenErrorScale = std::numeric_limits<float>::quiet_NaN();
                    policyStage = "nonfinite policy normalization";
                    view.rtLodHysteresis = std::numeric_limits<float>::quiet_NaN();
                    policyFrame();
                    require(params.rtOffscreenErrorScale == 2.f && params.rtLodHysteresis == .2f,
                        "Non-finite RT policy did not use finite defaults");
                    view.rtOffscreenErrorScale = -3.f;
                    view.rtLodHysteresis = 2.f;
                    policyFrame();
                    require(params.rtOffscreenErrorScale == 1.f && params.rtLodHysteresis == .9f,
                        "RT policy values escaped their safe bounds");
                    report.push_back({{"compact", compact}, {"distributed", distributed}, {"packedHistory", packedHistory},
                        {"demandMode", view.rtDemandMode}, {"frames", frameIndex},
                        {"cancelledInitialFallback", cancelledInitialFallback},
                        {"cancelledInitialFrame", cancelledInitialFrame}, {"verifiedPolicyFrames", verifiedPolicyFrames},
                        {"targetGroup", targetGroup}, {"offscreenReducedGroups", reducedGroups},
                        {"rasterAddedGroups", rasterAddedGroups},
                        {"intermediateLod", intermediate}, {"warmReferences", warmReferences},
                        {"pendingFrames", pendingFrames}, {"retainedFrames", retainedFrames},
                        {"peakGeometryBytes", peakGeometryBytes}, {"peakCLASBytes", peakCLASBytes}});
                }
            }
            { std::ofstream output(context.outputDirectory / "CLASCutReadiness.json"); output << report.dump(2); }
            require(validationErrors.load() == 0, "Vulkan validation errors in CLAS cut transition");
            return RHITestResult::pass("CLAS readiness, budgeted ancestor retention and CPU-matched RT distance/hysteresis policy hold in compact/direct and distributed/ordered traversal");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamCLASCutReadinessTest);

} // namespace
} // namespace metallic::tests

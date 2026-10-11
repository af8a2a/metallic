#include "RHITest.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace metallic::tests {
namespace {

using namespace render;

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
            nlohmann::json report = nlohmann::json::array();
            for (bool compact : {false, true}) {
                for (bool distributed : {false, true}) {
                    MeshletStreamRuntime runtime;
                    runtime.setDebugReadbackEnabled(true);
                    const uint32_t groupCapacity = std::min(2048u, asset.groupCount());
                    constexpr uint64_t kGeometryBudget = 32ull << 20;
                    constexpr uint64_t kCLASBudget = 64ull << 20;
                    MeshletStreamRuntimeDesc desc{.sourcePath = sourcePath, .streamAssetPath = assetPath,
                        .maxResidentBytes = kGeometryBudget, .maxResidentPages = 1024,
                        .maxPageUploadsPerFrame = 256, .maxGpuPageRequests = 1024, .maxGpuPageUnloadRequests = 1024,
                        .maxActiveGroups = groupCapacity, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
                        .pageLoadConcurrency = 0, .maxPageLoadsInFlight = 256, .queuedFrameCount = 2,
                        .enableClusterRtx = true, .enableClas = true, .compactClas = compact,
                        .maxClasBytes = kCLASBudget, .growClasBytes = 4ull << 20, .maxClasBuildClusters = maxPageClusters,
                        .maxBlasClusterReferences = 8192, .maxBlasBytes = 16ull << 20, .maxBlasBuilds = 4,
                        .maxFallbackBlasBytes = 4ull << 20, .distributedPageDemand = distributed,
                        .distributedDemandMinGroups = 0, .prefetchPages = false, .predictivePrefetch = false};
                    require(bool(runtime.initialize(*device, desc, log)), log);
                    auto& queue = *device->getQueue(QueueType::Graphics);
                    QueueSubmissionTracker tracker;
                    RenderFrameContext frame;
                    std::unique_ptr<CommandPool> pool;
                    std::unique_ptr<CommandBuffer> commands;
                    std::unique_ptr<Streamer> streamer;
                    std::unique_ptr<Buffer> readback;
                    const uint64_t groupsOffset = sizeof(MeshletStreamGPUBLASHeader) + sizeof(MeshletStreamGPUActiveHeader);
                    const uint64_t pagesOffset = groupsOffset + groupCapacity * sizeof(MeshletStreamGPUActiveGroup);
                    const uint64_t readbackBytes = pagesOffset + asset.pageCount() * sizeof(StreamPageTableEntry);
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
                    view.camera = {.eye = {-.0168404f, .110154f, .22f},
                        .center = {-.0168404f, .110154f, -.00153695f}, .znear = .001f, .zfar = 10.f};
                    uint64_t frameIndex = 0;
                    MeshletStreamGPUBLASHeader blas{};
                    std::vector<MeshletStreamGPUActiveGroup> active;
                    std::vector<StreamPageTableEntry> pages(asset.pageCount());
                    std::vector<MeshletStreamCLASPageEntry> clas(asset.pageCount());
                    std::vector<uint32_t> masks(asset.groupCount());
                    uint64_t peakGeometryBytes = 0, peakCLASBytes = 0;
                    bool cancelledInitialFallback = false;
                    const auto record = [&] {
                        runtime.prepareMaintenance();
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
                        require(bool(commands->end()), "Cut frame end failed");
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
                        readback->unmap();
                        // Both CLAS implementations expose a host-visible publication table;
                        // the submitted frame has finished all compact copy publications.
                        auto* table = runtime.clasPool()->pageTableBuffer();
                        table->invalidate();
                        const auto* tableData = table->map();
                        require(tableData != nullptr, "CLAS publication table map failed");
                        std::memcpy(clas.data(), tableData, clas.size() * sizeof(clas[0]));
                        table->unmap();
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
                    require(pendingFrames != 0 && retainedFrames != 0,
                        "Fixture did not expose loaded finer geometry waiting for CLAS while retaining an ancestor");
                    report.push_back({{"compact", compact}, {"distributed", distributed}, {"frames", frameIndex},
                        {"cancelledInitialFallback", cancelledInitialFallback},
                        {"intermediateLod", intermediate}, {"warmReferences", warmReferences},
                        {"pendingFrames", pendingFrames}, {"retainedFrames", retainedFrames},
                        {"peakGeometryBytes", peakGeometryBytes}, {"peakCLASBytes", peakCLASBytes}});
                }
            }
            { std::ofstream output(context.outputDirectory / "CLASCutReadiness.json"); output << report.dump(2); }
            require(validationErrors.load() == 0, "Vulkan validation errors in CLAS cut transition");
            return RHITestResult::pass("Delayed finer CLAS preserves complete intermediate coverage and budgets in compact/direct and distributed/ordered traversal");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamCLASCutReadinessTest);

} // namespace
} // namespace metallic::tests

#include "RhiTest.h"

#include "Runtime/Render/Streamer/MeshletStreamClas.h"
#include "Runtime/Render/Streamer/MeshletStreamPageLoader.h"
#include "Runtime/Render/Streamer/MeshletStreamResidency.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/GAPI/StreamUploadCompletion.h"
#include "Runtime/Render/Debug/RenderDebug.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetConfig.h"
#include "Runtime/Render/Streamer/StreamingTaskQueue.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"
#include "Runtime/Task/TaskSystem.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace metallic::tests {
namespace {

RhiTestResult createCommandResources(
    render::Device& device,
    render::Queue& queue,
    std::unique_ptr<render::CommandPool>& outCommandPool,
    std::unique_ptr<render::CommandBuffer>& outCommandBuffer,
    std::unique_ptr<render::Fence>& outFence)
{
    render::Result result = device.createCommandPool(queue, outCommandPool);
    if (!result || outCommandPool == nullptr) {
        return RhiTestResult::fail(std::string("createCommandPool returned ") + toString(result));
    }

    result = outCommandPool->createCommandBuffer(outCommandBuffer);
    if (!result || outCommandBuffer == nullptr) {
        return RhiTestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
    }

    result = device.createFence(false, outFence);
    if (!result || outFence == nullptr) {
        return RhiTestResult::fail(std::string("createFence returned ") + toString(result));
    }

    return RhiTestResult::pass();
}

RhiTestResult submitAndWait(
    render::Queue& queue,
    render::CommandBuffer& commandBuffer,
    render::Fence& fence)
{
    render::CommandBuffer* commandBuffers[] = {&commandBuffer};
    render::Result result = queue.submit(render::QueueSubmitDesc{
        .commandBuffers = commandBuffers,
        .commandBufferCount = 1,
        .signalFence = &fence,
    });
    if (!result) {
        return RhiTestResult::fail(std::string("Queue::submit returned ") + toString(result));
    }

    result = fence.wait(5'000'000'000ull);
    if (!result) {
        return RhiTestResult::fail(std::string("Fence::wait returned ") + toString(result));
    }
    return RhiTestResult::pass();
}

bool readBufferBytes(render::Buffer& buffer, void* outData, uint64_t byteSize)
{
    buffer.invalidate(0, byteSize);
    void* mapped = buffer.map();
    if (mapped == nullptr) {
        return false;
    }
    std::memcpy(outData, mapped, static_cast<size_t>(byteSize));
    buffer.unmap();
    return true;
}

render::StreamerDesc makeTestStreamerDesc(uint64_t dynamicSizePerFrame = 1024)
{
    render::StreamerDesc desc;
    desc.constantBufferSize = 4096;
    desc.dynamicBufferSizePerFrame = dynamicSizePerFrame;
    desc.queuedFrameCount = 2;
    desc.dynamicBufferDesc.usage = render::BufferUsageBits::TransferSource;
    return desc;
}

RhiTestResult buildBunnyStreamAssetForTest(
    const std::filesystem::path& outputPath,
    scene::MeshletStreamAsset& outAsset,
    scene::MeshletStreamPayloadCompression compressionMode = scene::MeshletStreamPayloadCompression::None)
{
    const std::filesystem::path sourcePath =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";

    scene::Scene scene;
    if (!scene.load(sourcePath)) {
        return RhiTestResult::fail("Scene::load failed: " + scene.lastLoadResult().error);
    }

    std::string reason;
    if (!scene::buildMeshletStreamAsset(
            scene::MeshletStreamAssetBuildDesc{
                .scene = &scene,
                .sourcePath = sourcePath,
                .outputPath = outputPath,
                .compressionMode = compressionMode,
            },
            reason)) {
        return RhiTestResult::fail("buildMeshletStreamAsset failed: " + reason);
    }

    if (!outAsset.open(outputPath, reason)) {
        return RhiTestResult::fail("MeshletStreamAsset::open failed: " + reason);
    }
    if (outAsset.pageCount() == 0) {
        return RhiTestResult::fail("streamasset has no pages");
    }
    return RhiTestResult::pass();
}

std::vector<uint32_t> fallbackPagesFor(const scene::MeshletStreamAsset& asset)
{
    std::vector<uint32_t> fallbackPages;
    uint64_t fallbackPageCount = 0;
    for (const scene::MeshletStreamPrimitiveInfo& primitive : asset.primitives()) {
        fallbackPageCount += primitive.fallbackPageCount;
    }
    fallbackPages.reserve(static_cast<size_t>(fallbackPageCount));
    for (const scene::MeshletStreamPrimitiveInfo& primitive : asset.primitives()) {
        for (uint32_t localPage = 0; localPage < primitive.fallbackPageCount; ++localPage) {
            fallbackPages.push_back(primitive.fallbackPageOffset + localPage);
        }
    }
    return fallbackPages;
}

std::vector<uint32_t> nonFallbackPagesFor(
    const scene::MeshletStreamAsset& asset,
    std::span<const uint32_t> fallbackPages)
{
    std::vector<uint8_t> isFallback(asset.pageCount(), 0);
    for (uint32_t page : fallbackPages) {
        if (page < isFallback.size()) {
            isFallback[page] = 1;
        }
    }

    std::vector<uint32_t> pages;
    for (uint32_t page = 0; page < asset.pageCount(); ++page) {
        if (isFallback[page] == 0) {
            pages.push_back(page);
        }
    }
    return pages;
}

uint64_t alignStreamStorageBytes(uint64_t value)
{
    const uint64_t alignment = render::kMeshletStreamStorageAlignment;
    return ((value + alignment - 1u) / alignment) * alignment;
}

uint64_t pageStorageBytes(const scene::MeshletStreamAsset& asset, uint32_t pageIndex)
{
    return pageIndex < asset.pages().size()
        ? alignStreamStorageBytes(asset.pages()[pageIndex].uncompressedSize)
        : 0;
}

uint64_t pageStorageBytes(const scene::MeshletStreamAsset& asset, std::span<const uint32_t> pageIndices)
{
    uint64_t total = 0;
    for (uint32_t pageIndex : pageIndices) {
        total += pageStorageBytes(asset, pageIndex);
    }
    return total;
}

class PageLoaderEventSink final : public task::ITaskEventSink {
public:
    void onGraphSubmitted(const task::TaskGraphSnapshot& snapshot) override
    {
        std::lock_guard lock(mutex);
        submitted.push_back(snapshot);
    }

    void onTaskStateChanged(const task::TaskNodeEvent& event) override
    {
        std::lock_guard lock(mutex);
        events.push_back(event);
    }

    void onGraphCompleted(const task::TaskGraphSnapshot& snapshot) override
    {
        std::lock_guard lock(mutex);
        completed.push_back(snapshot);
    }

    std::mutex mutex;
    std::vector<task::TaskGraphSnapshot> submitted;
    std::vector<task::TaskNodeEvent> events;
    std::vector<task::TaskGraphSnapshot> completed;
};

class MeshletStreamPageLoaderTaskGraphTest : public RhiTest {
public:
    MeshletStreamPageLoaderTaskGraphTest()
    {
        type = RhiTestType::Validation;
        name = "meshlet_stream_page_loader_task_graph";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        const std::filesystem::path streamAssetPath =
            context.outputDirectory / "task_graph_page_loader.meshstream.bin";
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            streamAssetPath,
            asset,
            scene::MeshletStreamPayloadCompression::ByteRle);
        if (!build.passed) {
            return build;
        }

        render::MeshletStreamPageLoader loader;
        std::string reason;
        if (!loader.initialize(asset, 2, reason)) {
            return RhiTestResult::fail("MeshletStreamPageLoader::initialize failed: " + reason);
        }

        auto sink = std::make_shared<PageLoaderEventSink>();
        const task::TaskEventSinkToken sinkToken = task::taskSystem().subscribe(sink);
        auto fail = [&](std::string message) {
            loader.reset();
            (void)task::taskSystem().unsubscribe(sinkToken);
            return RhiTestResult::fail(std::move(message));
        };

        const uint32_t validLoadCount = std::min(asset.pageCount(), 8u);
        const uint32_t expectedLoadCount = validLoadCount + 1u;
        for (uint32_t pageIndex = 0; pageIndex < validLoadCount; ++pageIndex) {
            if (!loader.enqueue(pageIndex)) {
                return fail("MeshletStreamPageLoader rejected a valid page");
            }
        }
        const uint32_t invalidPageIndex = asset.pageCount();
        if (!loader.enqueue(invalidPageIndex)) {
            return fail("MeshletStreamPageLoader rejected the failure propagation page");
        }

        uint32_t maxActiveLoads = loader.activeCount();
        uint32_t completedLoads = 0;
        uint32_t successfulLoads = 0;
        uint32_t failedLoads = 0;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (completedLoads < expectedLoadCount && std::chrono::steady_clock::now() < deadline) {
            maxActiveLoads = std::max(maxActiveLoads, loader.activeCount());
            if (loader.activeCount() > 2) {
                return fail("MeshletStreamPageLoader exceeded its local concurrency budget");
            }
            render::MeshletStreamPageLoadResult result;
            if (!loader.tryPop(result)) {
                std::this_thread::yield();
                continue;
            }
            ++completedLoads;
            if (result.success()) {
                ++successfulLoads;
            } else {
                ++failedLoads;
                if (result.pageIndex != invalidPageIndex || result.failureReason.empty()) {
                    return fail("MeshletStreamPageLoader did not preserve decode failure details");
                }
            }
        }
        if (completedLoads != expectedLoadCount ||
            successfulLoads != validLoadCount ||
            failedLoads != 1 ||
            maxActiveLoads > 2 ||
            loader.outstandingCount() != 0) {
            return fail("MeshletStreamPageLoader did not complete its bounded asynchronous queue");
        }

        const auto eventDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < eventDeadline) {
            std::lock_guard lock(sink->mutex);
            if (sink->completed.size() == expectedLoadCount) {
                break;
            }
            std::this_thread::yield();
        }

        uint32_t failedGraphs = 0;
        {
            std::lock_guard lock(sink->mutex);
            if (sink->submitted.size() != expectedLoadCount ||
                sink->completed.size() != expectedLoadCount) {
                return fail("page loader TaskGraph events were incomplete");
            }
            for (const task::TaskGraphSnapshot& snapshot : sink->completed) {
                if (snapshot.name != "MeshletStreamPageLoad" ||
                    snapshot.nodes.size() != 1 ||
                    !snapshot.edges.empty() ||
                    snapshot.nodes.front().desc.name != "MeshletStreamPageLoad" ||
                    snapshot.nodes.front().desc.category != "Streaming" ||
                    snapshot.nodes.front().workerThreadId == 0) {
                    return fail("page loader TaskGraph event metadata was invalid");
                }
                if (snapshot.status == task::TaskGraphStatus::Failed) {
                    ++failedGraphs;
                    if (snapshot.nodes.front().desc.userTag != invalidPageIndex ||
                        snapshot.nodes.front().error.empty()) {
                        return fail("failed page TaskGraph event lost its page tag or error");
                    }
                }
            }
        }
        if (failedGraphs != 1) {
            return fail("page loader did not emit exactly one failed TaskGraph");
        }

        for (uint32_t pageIndex = 0; pageIndex < validLoadCount; ++pageIndex) {
            if (!loader.enqueue(pageIndex)) {
                return fail("MeshletStreamPageLoader rejected a page before reset drain validation");
            }
        }
        loader.reset();
        (void)task::taskSystem().unsubscribe(sinkToken);
        if (loader.ready() || loader.pendingCount() != 0 || loader.activeCount() != 0 ||
            loader.completedCount() != 0 || loader.outstandingCount() != 0) {
            return RhiTestResult::fail("MeshletStreamPageLoader::reset did not drain and clear the loader");
        }
        return RhiTestResult::pass();
    }
};

class MeshletStreamPageLoadConfigurationCompatibilityTest : public RhiTest {
public:
    MeshletStreamPageLoadConfigurationCompatibilityTest()
    {
        type = RhiTestType::Validation;
        name = "meshlet_stream_page_load_configuration_compatibility";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraphProperties legacyOnly{{"pageLoadWorkerCount", 5}};
        render::RenderGraphProperties bothKeys{
            {"pageLoadWorkerCount", 5},
            {"pageLoadConcurrency", 3},
        };
        render::RenderGraphProperties invalidNewKey{
            {"pageLoadWorkerCount", 4},
            {"pageLoadConcurrency", -1},
        };
        render::RenderGraphProperties capped{{"pageLoadConcurrency", 1000}};

        if (render::builtin_pass::pageLoadConcurrencyFromProperties(legacyOnly) != 5 ||
            render::builtin_pass::pageLoadConcurrencyFromProperties(bothKeys) != 3 ||
            render::builtin_pass::pageLoadConcurrencyFromProperties(invalidNewKey) != 4 ||
            render::builtin_pass::pageLoadConcurrencyFromProperties(capped) !=
                render::kMeshletStreamMaxPageLoadConcurrency) {
            return RhiTestResult::fail("page load concurrency configuration compatibility failed");
        }
        return RhiTestResult::pass();
    }
};

class StreamingTaskQueueLifecycleTest : public RhiTest {
public:
    StreamingTaskQueueLifecycleTest()
    {
        type = RhiTestType::Validation;
        name = "streaming_task_queue_lifecycle";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        render::StreamingTaskQueue queue;
        if (queue.availableTaskCount() != render::kStreamingMaxActiveTasks ||
            queue.queuedTaskCount() != 0 ||
            queue.acquiredTaskCount() != 0) {
            return RhiTestResult::fail("new StreamingTaskQueue did not start with all tasks available");
        }

        const uint32_t first = queue.acquireTaskIndex();
        const uint32_t second = queue.acquireTaskIndex();
        const uint32_t third = queue.acquireTaskIndex();
        const uint32_t fourth = queue.acquireTaskIndex();
        if (first != 0 ||
            second != 1 ||
            third != 2 ||
            fourth != render::kInvalidStreamingTaskIndex ||
            queue.availableTaskCount() != 0 ||
            queue.acquiredTaskCount() != render::kStreamingMaxActiveTasks) {
            return RhiTestResult::fail("StreamingTaskQueue did not allocate fixed task indices in order");
        }

        queue.push(first, 5, 17);
        queue.push(second, 7);
        render::StreamingTaskQueue::Stats queueStats = queue.stats();
        if (!queueStats.acquisitionBlocked ||
            queueStats.frontTaskIndex != first ||
            queueStats.frontDependentIndex != 17 ||
            queueStats.frontCompletionFrameIndex != 5 ||
            queue.frontTaskIndex() != first ||
            queue.frontDependentIndex() != 17 ||
            queue.frontCompletionFrameIndex() != 5) {
            return RhiTestResult::fail("StreamingTaskQueue did not expose front task acquisition pressure");
        }
        if (queue.canPop(4, false) ||
            !queue.canPop(5, false) ||
            queue.queuedTaskCount() != 2) {
            return RhiTestResult::fail("StreamingTaskQueue completion frame test failed");
        }

        uint32_t dependent = render::kInvalidStreamingTaskIndex;
        const uint32_t popped = queue.popWithDependent(dependent);
        if (popped != first || dependent != 17 || queue.queuedTaskCount() != 1) {
            return RhiTestResult::fail("StreamingTaskQueue did not pop the first queued task with its dependent index");
        }
        queue.releaseTaskIndex(popped);
        if (queue.availableTaskCount() != 1 || queue.acquiredTaskCount() != 2) {
            return RhiTestResult::fail("StreamingTaskQueue did not release a completed task index");
        }

        const uint32_t recycled = queue.acquireTaskIndex();
        if (recycled != first) {
            return RhiTestResult::fail("StreamingTaskQueue did not recycle the released task index");
        }
        queue.push(recycled, 6);
        if (queue.canPop(6, false)) {
            return RhiTestResult::fail("StreamingTaskQueue did not preserve FIFO completion order");
        }
        if (!queue.canPop(7, true)) {
            return RhiTestResult::fail("StreamingTaskQueue did not report the front task ready at its completion frame");
        }

        queue.releaseTaskIndex(queue.pop());
        queue.releaseTaskIndex(queue.pop());
        queue.releaseTaskIndex(third);
        if (!queue.empty() ||
            queue.availableTaskCount() != render::kStreamingMaxActiveTasks ||
            queue.acquiredTaskCount() != 0) {
            return RhiTestResult::fail("StreamingTaskQueue did not return to an idle state");
        }

        return RhiTestResult::pass();
    }
};

class MeshletStreamStorageAddressLimitTest : public RhiTest {
public:
    MeshletStreamStorageAddressLimitTest()
    {
        type = RhiTestType::Validation;
        name = "meshlet_stream_storage_address_limit";
    }

    RhiTestResult run(RhiTestContext&) override
    {
        constexpr uint64_t kLargeCapacity = 5ull * 1024ull * 1024ull * 1024ull;
        render::MeshletStreamStorage storage;
        std::string reason;
        if (storage.initialize(kLargeCapacity, 256, reason)) {
            return RhiTestResult::fail("page storage accepted a byte budget above its default 32-bit limit");
        }
        if (!storage.initialize(kLargeCapacity, 256, reason, UINT64_MAX) ||
            storage.capacityBytes() != kLargeCapacity) {
            return RhiTestResult::fail("64-bit CLAS storage budget initialization failed: " + reason);
        }

        const render::MeshletStreamStorageAllocation allocation =
            storage.allocate(kLargeCapacity - 256u);
        if (!allocation.valid() ||
            allocation.offset != 0 ||
            storage.usedBytes() != allocation.allocatedSize) {
            return RhiTestResult::fail("64-bit CLAS storage allocation failed");
        }
        storage.release(allocation);
        if (storage.usedBytes() != 0 || storage.freeBytes() != storage.capacityBytes()) {
            return RhiTestResult::fail("64-bit CLAS storage release did not restore capacity");
        }
        return RhiTestResult::pass();
    }
};

class StreamLodPipelineCacheTest : public RhiTest {
public:
    StreamLodPipelineCacheTest() { type = RhiTestType::Resource; name = "stream_lod_pipeline_cache_persistence"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Metallic Stream LOD Cache Test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true}, device);
        if (!created) {
            return hasError(created, Error::Unsupported) ? RhiTestResult::skip("Bindless device unavailable")
                : RhiTestResult::fail("Cannot create LOD cache test device");
        }
        const auto assetPath = context.outputDirectory / "lod_pipeline_cache.meshstream.bin";
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(assetPath, asset);
        if (!built.passed) { return built; }
        const auto cachePath = context.outputDirectory / "lod_pipeline_cache.pso";
        std::error_code fileError;
        std::filesystem::remove(cachePath, fileError);
        if (fileError) { return RhiTestResult::fail("Cannot clear test pipeline cache: " + fileError.message()); }
        const std::string cacheName = cachePath.string();
        const MeshletStreamRuntimeDesc desc{
            .sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf",
            .streamAssetPath = assetPath,
            .maxResidentBytes = 16ull << 20,
            .maxResidentPages = 256,
            .maxGpuPageRequests = 256,
            .maxGpuPageUnloadRequests = 256,
            .maxActiveGroups = 2048,
            .maxTraversalWorkers = 64,
            .maxTraversalWorkItems = 4096,
            .queuedFrameCount = 2,
        };
        std::unique_ptr<PipelineCache> cache;
        MeshletStreamRuntime runtime;
        std::string log;
        // Fresh runtime and cache objects: the warm pass must use serialized
        // driver data and PSO keys, not retained pipeline objects.
        for (uint32_t pass = 0; pass < 2; ++pass) {
            auto result = device->createPipelineCache(
                {.filePath = cacheName.c_str(), .saveOnDestroy = false}, cache);
            if (!result || !cache) { return RhiTestResult::fail("Cannot create LOD test cache"); }
            const auto expectedLoad = pass == 0 ? PipelineCacheLoadStatus::NotFound : PipelineCacheLoadStatus::Loaded;
            if (cache->stats().loadStatus != expectedLoad) { return RhiTestResult::fail("LOD cache load status mismatch"); }
            result = runtime.initialize(*device, desc, log, cache.get());
            if (!result || !runtime.ready()) { return RhiTestResult::fail("LOD initialization failed: " + log); }
            const auto stats = cache->stats();
            // Page-table init/update, traversal, active build, cooperative LOD.
            if (stats.sessionPsoCount != 5 || stats.hitCount != (pass == 0 ? 0 : 5) ||
                stats.missCount != (pass == 0 ? 5 : 0)) {
                return RhiTestResult::fail("An internal streaming/LOD pipeline bypassed the persistent cache");
            }
            result = cache->save();
            if (!result || cache->stats().backendDataSize == 0) {
                return RhiTestResult::fail("LOD cache did not serialize native pipeline data");
            }
            // Initialization does not retain the caller's cache pointer.
            cache.reset();
            runtime.reset();
        }
        const auto uncached = runtime.initialize(*device, desc, log);
        if (!uncached || !runtime.ready()) {
            return RhiTestResult::fail("Optional-cache compatibility failed: " + log);
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(StreamLodPipelineCacheTest);

class StreamClasRuntimeTest : public RhiTest {
public:
    explicit StreamClasRuntimeTest(bool pressure = false) : pressure_(pressure)
    {
        type = RhiTestType::Rendering;
        name = pressure ? "stream_clas_eviction_reupload" : "stream_clas_runtime_lifecycle";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream CLAS lifecycle",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true, .enableClusterAccelerationStructure = true}, device);
        if (!created) {
            return hasError(created, Error::Unsupported) ? RhiTestResult::skip("Requires CLAS and bindless support")
                : RhiTestResult::fail("CLAS device creation failed");
        }
        if (!device->capabilities().clusterAccelerationStructure) { return RhiTestResult::skip("CLAS unavailable"); }
        const auto path = std::filesystem::absolute(context.outputDirectory / "clas_lifecycle.meshstream.bin");
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(path, asset, scene::MeshletStreamPayloadCompression::ByteRle);
        if (!built.passed) { return built; }
        const auto budget = asset.maxPageClusters();
        MeshletStreamRuntime runtime;
        std::string log;
        const auto initialized = runtime.initialize(*device, {
            .sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf",
            .streamAssetPath = path, .maxResidentBytes = 16ull << 20, .maxResidentPages = pressure_ ? 64u : 256u,
            .maxPageUploadsPerFrame = 64, .maxGpuPageRequests = 256, .maxGpuPageUnloadRequests = 256,
            .maxActiveGroups = 2048, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
            .pageLoadConcurrency = pressure_ ? 0u : 1u, .maxPageLoadsInFlight = 64, .queuedFrameCount = 2,
            .enableClas = true, .maxClasBytes = pressure_ ? 64ull << 10 : 16ull << 20, .maxClasBuildClusters = pressure_ ? budget * asset.pageCount() : budget,
            .prefetchPages = false}, log);
        if (!initialized) { return RhiTestResult::fail("CLAS-only initialize: " + log); }
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        const auto setup = createCommandResources(*device, *queue, pool, commands, fence);
        if (!setup.passed) { return setup; }
        std::unique_ptr<Streamer> streamer;
        if (!device->createStreamer(makeTestStreamerDesc(), streamer)) { return RhiTestResult::fail("Cannot create streamer"); }
        MeshletStreamFrameDesc frame{.width = 192, .height = 128, .selectedLodLevel = 0, .enableGpuLodSelection = false};
        frame.camera = {.eye = {-.0168404f, .110154f, .22f}, .center = {-.0168404f, .110154f, -.00153695f},
            .znear = .001f, .zfar = 10.f};
        const auto nearCamera = frame.camera;
        bool sawBuilt = false, sawPending = false;
        uint64_t stableBuildCount = 0;
        uint32_t reloadPage = UINT32_MAX;
        std::ofstream trace(context.outputDirectory / (pressure_ ? "ClasEviction.jsonl" : "ClasLifecycle.jsonl"));
        for (uint32_t f = 0; f < 420; ++f) {
            frame.camera = nearCamera;
            if (f >= 180 && f < 240) { frame.camera.eye = {100.f, 100.f, 100.f}; frame.camera.center = {101.f, 100.f, 100.f}; }
            if (!pool->reset() || !fence->reset() || !commands->begin()) { return RhiTestResult::fail("Frame setup failed"); }
            if (pressure_ && reloadPage != UINT32_MAX && f % 12 == 2) {
                const_cast<MeshletStreamResidencyManager&>(runtime.residency()).requestPage(reloadPage);
            }
            auto result = runtime.cmdBeginFrame(*commands, *streamer, frame);
            if (result) {
                commands->copyStreamedData(*streamer);
                // Keep the old CLAS queue entry across unload completion, then
                // admit a new upload before draining it (e.g. deferred traversal).
                if (!pressure_ || reloadPage == UINT32_MAX || f % 12 != 1) {
                    result = runtime.cmdPreTraversal(*commands, frame);
                }
            }
            if (result) { result = runtime.cmdPostTraversal(*commands); }
            if (result) { result = runtime.cmdEndFrame(*commands); }
            if (!result || !commands->end()) { return RhiTestResult::fail("CLAS frame failed: " + std::string(toString(result))); }
            const auto submitted = submitAndWait(*queue, *commands, *fence);
            streamer->endFrame();
            if (!submitted.passed) { return submitted; }
            const auto stats = runtime.profilingStats();
            trace << nlohmann::json{{"frame", f}, {"runtime", runtime.debugSnapshot(false)},
                {"clas", {{"resident", stats.clasResidentPages}, {"pending", stats.clasPendingPages},
                    {"built", stats.clasBuiltPages}, {"total", stats.clasTotalBuiltPages},
                    {"retiring", stats.clasRetiringPages}, {"rejected", stats.clasRejectedPages}, {"bytes", stats.clasUsedBytes}}}}.dump() << '\n';
            if (!stats.clasEnabled || runtime.tlasReady() || runtime.accelerationStructure() ||
                stats.clasBuiltClusters > budget || stats.clasUsedBytes > stats.clasCapacityBytes) {
                return RhiTestResult::fail("CLAS-only runtime violated build/storage budget or built a TLAS");
            }
            if (pressure_ && f > 40 && f % 12 == 0) {
                // Inject the same unload operation used by GPU feedback after
                // the frame fence. The runtime itself is non-const; this only
                // controls the regression's interleaving, without a test-only API.
                auto& residency = const_cast<MeshletStreamResidencyManager&>(runtime.residency());
                for (uint32_t page : residency.residentPages()) {
                    if (residency.pageState(page) == MeshletStreamPageResidencyState::Resident &&
                        !runtime.clasPool()->pageHasClas(page)) {
                        residency.unloadPage(page);
                        reloadPage = page;
                        break;
                    }
                }
            }
            sawBuilt |= stats.clasBuiltPages > 0;
            sawPending |= stats.clasPendingPages > 0;
            if (f == 150) { stableBuildCount = stats.clasTotalBuiltPages; }
            if (!pressure_ && f >= 151 && f < 180 && (stats.clasTotalBuiltPages != stableBuildCount || stats.clasPendingPages != 0)) {
                return RhiTestResult::fail("Steady resident CLAS rebuilt or backlog failed to converge");
            }
        }
        const auto last = runtime.profilingStats();
        if (pressure_) {
            const auto residency = runtime.residency().stats();
            if (!sawPending || residency.totalCompletedUnloadCount < 10 || last.clasRejectedPages == 0) {
                return RhiTestResult::fail("Pressure fixture did not exercise eviction and exhausted CLAS storage: " +
                    std::to_string(residency.totalCompletedUnloadCount));
            }
            return RhiTestResult::pass("CLAS budget exhaustion with repeated geometry eviction/reupload retains live upload plans");
        }
        if (!sawBuilt || !sawPending || last.clasPendingPages || last.clasResidentPages != last.residentPages ||
            last.clasTotalBuiltPages != stableBuildCount) {
            return RhiTestResult::fail("Lifecycle coverage/convergence: built=" + std::to_string(sawBuilt) +
                " pending=" + std::to_string(sawPending) +
                " final pending=" + std::to_string(last.clasPendingPages) + " resident=" + std::to_string(last.residentPages) +
                " CLAS=" + std::to_string(last.clasResidentPages));
        }
        return RhiTestResult::pass("Compressed uploads, bounded build backlog, camera round-trip reuses cached CLAS, independent CLAS without BLAS/TLAS");
    }
private:
    bool pressure_ = false;
};
METALLIC_REGISTER_RHI_TEST(StreamClasRuntimeTest);
class StreamClasEvictionTest final : public StreamClasRuntimeTest {
public:
    StreamClasEvictionTest() : StreamClasRuntimeTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(StreamClasEvictionTest);

class StreamBlasCacheTest final : public RhiTest {
public:
    StreamBlasCacheTest() { type = RhiTestType::Rendering; name = "stream_blas_cut_cache"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream BLAS reuse regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true, .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true, .enableClusterAccelerationStructure = true}, device);
        if (!created) { return hasError(created, Error::Unsupported) ? RhiTestResult::skip("CLAS unavailable")
            : RhiTestResult::fail("Device creation failed"); }
        const auto path = std::filesystem::absolute(context.outputDirectory / "blas_cache.meshstream.bin");
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(path, asset, scene::MeshletStreamPayloadCompression::ByteRle);
        if (!built.passed) { return built; }
        const auto require = [](bool condition, const std::string& reason) {
            if (!condition) { throw std::runtime_error(reason); }
        };
        try {
            MeshletStreamRuntime runtime;
            runtime.setDebugReadbackEnabled(true);
            std::string log;
            require(bool(runtime.initialize(*device, {.sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf",
                .streamAssetPath = path, .maxResidentBytes = 16ull << 20, .maxResidentPages = 1024,
                .maxPageUploadsPerFrame = 256, .maxGpuPageRequests = 1024, .maxGpuPageUnloadRequests = 1024,
                .maxActiveGroups = 2048, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
                .pageLoadConcurrency = 0, .maxPageLoadsInFlight = 256, .queuedFrameCount = 2,
                .enableClusterRtx = true, .enableClas = true, .compactClas = true,
                .maxClasBytes = 16ull << 20, .maxClasBuildClusters = 4096,
                .maxBlasClusterReferences = 4096, .maxBlasBytes = 16ull << 20, .maxBlasBuilds = 4,
                .maxFallbackBlasBytes = 4ull << 20, .prefetchPages = false}, log)), log);
            auto* queue = device->getQueue(QueueType::Graphics);
            QueueSubmissionTracker tracker;
            RenderFrameContext frame;
            std::unique_ptr<CommandPool> pool;
            std::unique_ptr<CommandBuffer> commands;
            std::unique_ptr<Streamer> streamer;
            std::unique_ptr<Buffer> readback;
            require(bool(tracker.initialize(*device, *queue)) && bool(device->createCommandPool(*queue, pool)) &&
                bool(pool->createCommandBuffer(commands)) && bool(device->createStreamer(makeTestStreamerDesc(), streamer)) &&
                bool(device->createBuffer({.size = sizeof(MeshletStreamGpuBlasHeader), .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}, readback)), "Frame resources failed");
            MeshletStreamFrameDesc view{.width = 192, .height = 128, .selectedLodLevel = 0, .enableGpuLodSelection = false};
            view.camera = {.eye = {-.0168404f, .110154f, .22f}, .center = {-.0168404f, .110154f, -.00153695f}, .znear = .001f, .zfar = 10.f};
            uint64_t frameId = 0;
            bool cancelledInitialFallback = false;
            MeshletStreamGpuBlasHeader header;
            const auto record = [&](bool cancel = false) {
                require(bool(frame.begin(++frameId)) && bool(pool->reset()) && bool(commands->begin(&frame)) &&
                    bool(streamer->beginFrame(frame)), "Frame begin failed");
                require(bool(runtime.cmdBeginFrame(*commands, *streamer, view)), "Stream begin failed");
                commands->copyStreamedData(*streamer);
                require(bool(runtime.cmdPreTraversal(*commands, view)) && bool(runtime.cmdPostTraversal(*commands)) &&
                    bool(runtime.cmdEndFrame(*commands)), "Stream traversal failed");
                std::vector<DebugResourceBinding> bindings;
                runtime.appendDebugBindings(bindings, "test.");
                const auto found = std::find_if(bindings.begin(), bindings.end(), [](const auto& binding) { return binding.id == "test.blasHeader"; });
                require(found != bindings.end(), "BLAS telemetry missing");
                BufferBarrierDesc barrier{.buffer = found->buffer, .before = found->state, .after = ResourceState::TransferSource};
                commands->barrier({.buffers = &barrier, .bufferCount = 1});
                commands->copyBuffer({.source = found->buffer, .destination = readback.get(), .size = sizeof(header)});
                std::swap(barrier.before, barrier.after);
                commands->barrier({.buffers = &barrier, .bufferCount = 1});
                require(bool(commands->end()), "Frame end failed");
                const auto readiness = runtime.debugSnapshot(false);
                if (!cancelledInitialFallback && readiness.at("fallbackBlasRecorded") > readiness.at("fallbackBlasSubmitted")) {
                    require(!runtime.sceneReady(), "Unsubmitted fallback build made the scene ready");
                    cancelledInitialFallback = true;
                    cancel = true;
                }
                if (cancel) {
                    frame.cancel(); streamer->endFrame();
                    if (readiness.at("fallbackBlasRecorded") > readiness.at("fallbackBlasSubmitted")) {
                        require(!runtime.sceneReady(), "Cancelled fallback build poisoned readiness");
                    }
                    return;
                }
                CommandBuffer* list[] = {commands.get()};
                require(bool(tracker.submit({.commandBuffers = list, .commandBufferCount = 1}, frame)) &&
                    bool(frame.wait(5000000000ull)), "Frame submit failed");
                streamer->endFrame();
                readback->invalidate();
                const auto* data = readback->map();
                require(data != nullptr, "BLAS readback failed");
                std::memcpy(&header, data, sizeof(header));
                readback->unmap();
            };
            uint32_t builds = 0, reused = 0;
            for (uint32_t i = 0; i < 100; ++i) {
                record();
                builds += header.blasBuildCount;
                reused += i >= 90 && header.padding0 == 0 && header.blasBuildCount == 0;
            }
            require(builds > 0 && reused == 10 && header.padding1 > 0 && runtime.tlasReady(), "Stable geometry did not reuse a built BLAS");
            require(cancelledInitialFallback && runtime.sceneReady(), "Cancelled initial fallback did not rebuild and become ready");
            const auto scans = runtime.debugSnapshot(false).at("sceneReadinessScans");
            for (uint32_t query = 0; query < 1000; ++query) {
                require(runtime.sceneReady() && runtime.sceneReadiness().ready, "Stable root readiness changed");
            }
            record();
            require(runtime.sceneReady() && runtime.debugSnapshot(false).at("sceneReadinessScans") == scans,
                "Steady queries / frame advance rescanned all roots");
            view.selectedLodLevel = 2;
            record();
            require(header.padding0 != 0, "Changed cut reused stale BLAS");
            view.selectedLodLevel = 0;
            record(); record();
            require(header.padding0 == 0, "Restored cut did not settle");
            view.selectedLodLevel = 2;
            record(true);
            view.selectedLodLevel = 0;
            record();
            require(header.padding0 != 0 && header.blasBuildCount > 0, "Cancelled recording poisoned BLAS cache");
            record();
            require(header.padding0 == 0, "Retry did not restore reuse");
            const auto residents = runtime.residency().residentPages();
            const auto nonRoot = std::find_if(residents.begin(), residents.end(), [&](uint32_t page) {
                return runtime.residency().pageState(page) == MeshletStreamPageResidencyState::Resident;
            });
            require(nonRoot != residents.end(), "Need a non-root CLAS retirement");
            const uint32_t page = *nonRoot;
            runtime.clasPool()->retirePages(std::span(&page, 1));
            require(runtime.sceneReady(), "Non-root retirement invalidated root readiness");
            record();
            require(header.padding0 != 0, "Retired CLAS did not invalidate cached references");
            const auto currentResidents = runtime.residency().residentPages();
            const auto root = std::find_if(currentResidents.begin(), currentResidents.end(), [&](uint32_t id) {
                return runtime.residency().pageState(id) == MeshletStreamPageResidencyState::LockedFallback;
            });
            require(root != currentResidents.end(), "Need root CLAS invalidation");
            runtime.clasPool()->retirePages(std::span(&*root, 1));
            require(!runtime.sceneReady() && runtime.debugSnapshot(false).at("sceneRootsInvalidated").get<bool>(),
                "Explicit root retirement retained stale readiness");
            runtime.reset();
            require(!runtime.sceneReady() && runtime.sceneReadiness().requiredPages == 0,
                "Reset retained the previous scene readiness");
            return RhiTestResult::pass("Stable cut reuse, changed LOD, cancellation and CLAS retirement invalidation");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamBlasCacheTest);


class StreamerBufferUploadTest : public RhiTest {
public:
    StreamerBufferUploadTest()
    {
        type = RhiTestType::Command;
        name = "streamer_buffer_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr std::array<uint32_t, 4> kExpected{
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };
        constexpr uint64_t kByteSize = kExpected.size() * sizeof(uint32_t);

        std::unique_ptr<render::Streamer> streamer;
        render::Result result = context.device.createStreamer(makeTestStreamerDesc(), streamer);
        if (!result || streamer == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = kByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        const render::StreamDataChunk chunks[] = {
            render::StreamDataChunk{
                .data = kExpected.data(),
                .size = 2 * sizeof(uint32_t),
            },
            render::StreamDataChunk{
                .data = kExpected.data() + 2,
                .size = 2 * sizeof(uint32_t),
            },
        };
        render::BufferOffset streamed = streamer->streamBufferData(render::StreamBufferDataDesc{
            .dataChunks = chunks,
            .dataChunkCount = static_cast<uint32_t>(std::size(chunks)),
            .placementAlignment = 4,
            .dstBuffer = readbackBuffer.get(),
            .dstOffset = 0,
        });
        if (!streamed.valid()) {
            return RhiTestResult::fail("streamBufferData returned an invalid source");
        }
        render::StreamerStats streamerStats = streamer->stats();
        if (streamerStats.currentFrameDynamicBytes != kByteSize ||
            streamerStats.currentFrameDynamicRequestCount != 1 ||
            streamerStats.totalDynamicBytes != 0 ||
            streamerStats.pendingCopies.bufferCopyCount != 1 ||
            streamerStats.pendingCopies.bufferCopyBytes != kByteSize) {
            return RhiTestResult::fail("streamBufferData did not update current-frame dynamic streamer stats");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::BufferBarrierDesc toTransfer{
            .buffer = readbackBuffer.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::TransferDestination,
            .offset = 0,
            .size = kByteSize,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = &toTransfer,
            .bufferCount = 1,
        });
        commandBuffer->copyStreamedData(*streamer);
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }
        streamerStats = streamer->stats();
        if (streamerStats.currentFrameDynamicBytes != 0 ||
            streamerStats.lastFrameDynamicBytes != kByteSize ||
            streamerStats.peakFrameDynamicBytes != kByteSize ||
            streamerStats.totalDynamicBytes != kByteSize ||
            streamerStats.lastFrameDynamicRequestCount != 1) {
            return RhiTestResult::fail("Streamer::endFrame did not roll dynamic upload stats");
        }

        std::array<uint32_t, 4> actual{};
        if (!readBufferBytes(*readbackBuffer, actual.data(), kByteSize)) {
            return RhiTestResult::fail("readback buffer did not map");
        }
        if (actual != kExpected) {
            return RhiTestResult::fail("streamed buffer bytes did not match expected pattern");
        }
        return RhiTestResult::pass();
    }
};

class StreamerTextureUploadTest : public RhiTest {
public:
    StreamerTextureUploadTest()
    {
        type = RhiTestType::Command;
        name = "streamer_texture_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint32_t kWidth = 4;
        constexpr uint32_t kHeight = 4;
        constexpr uint32_t kTightRowPitch = kWidth * 4;
        constexpr uint32_t kSourceRowPitch = 32;
        constexpr uint64_t kPixelByteSize = kWidth * kHeight * 4ull;

        std::array<uint8_t, kSourceRowPitch * kHeight> source{};
        std::array<uint8_t, kPixelByteSize> expected{};
        for (uint32_t y = 0; y < kHeight; ++y) {
            for (uint32_t x = 0; x < kWidth; ++x) {
                const uint8_t r = static_cast<uint8_t>(x * 40 + 3);
                const uint8_t g = static_cast<uint8_t>(y * 35 + 7);
                const uint8_t b = static_cast<uint8_t>(x + y * 10 + 11);
                const uint8_t a = 255;
                const uint32_t sourceIndex = y * kSourceRowPitch + x * 4;
                const uint32_t expectedIndex = y * kTightRowPitch + x * 4;
                source[sourceIndex + 0] = r;
                source[sourceIndex + 1] = g;
                source[sourceIndex + 2] = b;
                source[sourceIndex + 3] = a;
                expected[expectedIndex + 0] = r;
                expected[expectedIndex + 1] = g;
                expected[expectedIndex + 2] = b;
                expected[expectedIndex + 3] = a;
            }
        }

        std::unique_ptr<render::Streamer> streamer;
        render::Result result = context.device.createStreamer(makeTestStreamerDesc(), streamer);
        if (!result || streamer == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Texture> texture;
        result = context.device.createTexture(
            render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::TransferDestination | render::TextureUsageBits::TransferSource,
                .format = render::Format::Rgba8Unorm,
                .width = kWidth,
                .height = kHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            },
            texture);
        if (!result || texture == nullptr) {
            return RhiTestResult::fail(std::string("createTexture returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = kPixelByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            readbackBuffer);
        if (!result || readbackBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
        }

        render::BufferOffset streamed = streamer->streamTextureData(render::StreamTextureDataDesc{
            .data = source.data(),
            .dataRowPitch = kSourceRowPitch,
            .dataSlicePitch = kSourceRowPitch * kHeight,
            .dstTexture = texture.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
        });
        if (!streamed.valid()) {
            return RhiTestResult::fail("streamTextureData returned an invalid source");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::TextureBarrierDesc textureToTransfer{
            .texture = texture.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::TransferDestination,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .textures = &textureToTransfer,
            .textureCount = 1,
        });
        commandBuffer->copyStreamedData(*streamer);
        render::TextureBarrierDesc textureToSource{
            .texture = texture.get(),
            .before = render::ResourceState::TransferDestination,
            .after = render::ResourceState::TransferSource,
            .baseMip = 0,
            .mipCount = 1,
            .baseLayer = 0,
            .layerCount = 1,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .textures = &textureToSource,
            .textureCount = 1,
        });
        commandBuffer->copyTextureToBuffer(render::TextureBufferCopyDesc{
            .texture = texture.get(),
            .buffer = readbackBuffer.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
        });
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }

        std::array<uint8_t, kPixelByteSize> actual{};
        if (!readBufferBytes(*readbackBuffer, actual.data(), actual.size())) {
            return RhiTestResult::fail("texture readback buffer did not map");
        }
        if (actual != expected) {
            return RhiTestResult::fail("streamed texture pixels did not match expected pattern");
        }
        return RhiTestResult::pass();
    }
};

class StreamerConstantUploadTest : public RhiTest {
public:
    StreamerConstantUploadTest()
    {
        type = RhiTestType::Resource;
        name = "streamer_constant_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr std::array<uint32_t, 4> kFirst{
            0x01020304u,
            0x11121314u,
            0x21222324u,
            0x31323334u,
        };
        constexpr std::array<uint32_t, 2> kSecond{
            0xA0A1A2A3u,
            0xB0B1B2B3u,
        };

        std::unique_ptr<render::Streamer> streamer;
        render::StreamerDesc desc = makeTestStreamerDesc();
        desc.constantBufferSize = 4096;
        render::Result result = context.device.createStreamer(desc, streamer);
        if (!result || streamer == nullptr || streamer->constantBuffer() == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        const uint64_t firstOffset = streamer->streamConstantData(
            kFirst.data(),
            kFirst.size() * sizeof(uint32_t));
        const uint64_t secondOffset = streamer->streamConstantData(
            kSecond.data(),
            kSecond.size() * sizeof(uint32_t));
        if (firstOffset == std::numeric_limits<uint64_t>::max() ||
            secondOffset == std::numeric_limits<uint64_t>::max()) {
            return RhiTestResult::fail("streamConstantData returned an invalid offset");
        }
        if (firstOffset != 0) {
            return RhiTestResult::fail("first constant upload did not start at offset zero");
        }

        const uint64_t alignment = std::max<uint64_t>(
            context.device.capabilities().constantBufferOffsetAlignment,
            1);
        if (secondOffset % alignment != 0 ||
            secondOffset < kFirst.size() * sizeof(uint32_t)) {
            return RhiTestResult::fail("second constant upload was not aligned after first upload");
        }
        const uint64_t expectedConstantBytes =
            kFirst.size() * sizeof(uint32_t) + kSecond.size() * sizeof(uint32_t);
        render::StreamerStats streamerStats = streamer->stats();
        if (streamerStats.currentFrameConstantBytes != expectedConstantBytes ||
            streamerStats.currentFrameConstantRequestCount != 2 ||
            streamerStats.totalConstantBytes != 0) {
            return RhiTestResult::fail("streamConstantData did not update current-frame constant streamer stats");
        }

        render::Buffer* constantBuffer = streamer->constantBuffer();
        constantBuffer->invalidate(0, desc.constantBufferSize);
        void* mapped = constantBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("constant buffer did not map");
        }

        bool firstMatches = std::memcmp(
            static_cast<uint8_t*>(mapped) + firstOffset,
            kFirst.data(),
            kFirst.size() * sizeof(uint32_t)) == 0;
        bool secondMatches = std::memcmp(
            static_cast<uint8_t*>(mapped) + secondOffset,
            kSecond.data(),
            kSecond.size() * sizeof(uint32_t)) == 0;
        constantBuffer->unmap();
        if (!firstMatches || !secondMatches) {
            return RhiTestResult::fail("constant buffer contents did not match streamed data");
        }
        streamer->endFrame();
        streamerStats = streamer->stats();
        if (streamerStats.currentFrameConstantBytes != 0 ||
            streamerStats.lastFrameConstantBytes != expectedConstantBytes ||
            streamerStats.peakFrameConstantBytes != expectedConstantBytes ||
            streamerStats.totalConstantBytes != expectedConstantBytes ||
            streamerStats.lastFrameConstantRequestCount != 2) {
            return RhiTestResult::fail("Streamer::endFrame did not roll constant upload stats");
        }
        return RhiTestResult::pass();
    }
};

class StreamerGraphUploadPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Streamed graph data")
            .buffer(kExpected.size() * sizeof(uint32_t), sizeof(uint32_t))
            .transferWrite();
        return reflection;
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        render::Streamer* streamer = context.streamer();
        render::BufferHandle output = context.outputBuffer("data");
        if (streamer == nullptr || !output.valid()) {
            return render::makeError(render::Error::InvalidArgument);
        }

        const render::StreamDataChunk chunk{
            .data = kExpected.data(),
            .size = kExpected.size() * sizeof(uint32_t),
        };
        render::BufferOffset streamed = streamer->streamBufferData(render::StreamBufferDataDesc{
            .dataChunks = &chunk,
            .dataChunkCount = 1,
            .placementAlignment = 4,
            .dstBuffer = output.buffer(),
            .dstOffset = 0,
        });
        return streamed.valid() ? render::Result{} : render::makeError(render::Error::Failure);
    }

    static constexpr std::array<uint32_t, 4> kExpected{
        0x11223344u,
        0xAABBCCDDu,
        0xDEADBEEFu,
        0xCAFEBABEu,
    };
};

class StreamerCrossQueueSourcePass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data", "Cross-queue source")
            .buffer(16)
            .storageReadWrite();
        return reflection;
    }

    render::Result execute(render::RenderGraphExecutionContext&) override
    {
        return render::makeError(render::Error::Failure);
    }
};

class StreamerCrossQueueSinkPass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferInput("data", "Cross-queue input")
            .buffer(16)
            .shaderRead();
        reflection.addBufferOutput("copied", "Cross-queue output")
            .buffer(16)
            .storageReadWrite();
        return reflection;
    }

    render::Result execute(render::RenderGraphExecutionContext&) override
    {
        return render::makeError(render::Error::Failure);
    }
};

void registerStreamerGraphPass()
{
    static bool registered = false;
    if (registered) {
        return;
    }
    registered = true;
    render::registerRenderGraphPassType(
        "StreamerGraphUploadPass",
        "Test-only pass that streams into a graph output buffer",
        []() { return std::make_unique<StreamerGraphUploadPass>(); });
    render::registerRenderGraphPassType(
        "StreamerCrossQueueSourcePass",
        "Test-only graphics pass for streaming cross-queue guards",
        []() { return std::make_unique<StreamerCrossQueueSourcePass>(); });
    render::registerRenderGraphPassType(
        "StreamerCrossQueueSinkPass",
        "Test-only compute pass for streaming cross-queue guards",
        []() { return std::make_unique<StreamerCrossQueueSinkPass>(); });
}

class StreamerRenderGraphFlushTest : public RhiTest {
public:
    StreamerRenderGraphFlushTest()
    {
        type = RhiTestType::Command;
        name = "streamer_render_graph_flush";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerStreamerGraphPass();

        render::RenderGraph graph;
        graph.setName("StreamerGraph");
        graph.addNode("StreamerGraphUploadPass", "Upload");
        graph.markOutput("Upload.data");

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result result = executor.compile(context.device, graph, 1, 1, log);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        result = executor.execute(*commandBuffer);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
        }
        const render::RenderGraphStreamingStats& streamingStats = executor.streamingStats();
        const uint64_t expectedBytes = StreamerGraphUploadPass::kExpected.size() * sizeof(uint32_t);
        if (streamingStats.flushCount != 1 ||
            streamingStats.flushesWithWork != 1 ||
            streamingStats.transferCount != 1 ||
            streamingStats.bufferTransferCount != 1 ||
            streamingStats.textureTransferCount != 0 ||
            streamingStats.transferBytes != expectedBytes ||
            streamingStats.bufferTransferBytes != expectedBytes) {
            return RhiTestResult::fail("RenderGraph streaming subsystem stats did not match the streamed pass work");
        }
        if (streamingStats.streamer.pendingCopies.copyCount() != 0 ||
            streamingStats.streamer.frameIndex == 0) {
            return RhiTestResult::fail("RenderGraph streaming subsystem did not end the streamer frame cleanly");
        }
        if (streamingStats.streamer.currentFrameDynamicBytes != 0 ||
            streamingStats.streamer.lastFrameDynamicBytes != expectedBytes ||
            streamingStats.streamer.peakFrameDynamicBytes != expectedBytes ||
            streamingStats.streamer.totalDynamicBytes != expectedBytes ||
            streamingStats.streamer.lastFrameDynamicRequestCount != 1) {
            return RhiTestResult::fail("RenderGraph streaming subsystem did not retain last-frame Streamer stats");
        }
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        if (!submit.passed) {
            return submit;
        }

        render::RenderGraphResource* output = executor.outputResource("Upload.data");
        if (output == nullptr || output->buffer == nullptr) {
            return RhiTestResult::fail("streamer graph output resource is missing");
        }

        std::array<uint32_t, 4> actual{};
        if (!readBufferBytes(
                *output->buffer,
                actual.data(),
                actual.size() * sizeof(uint32_t))) {
            return RhiTestResult::fail("streamer graph output did not map");
        }
        if (actual != StreamerGraphUploadPass::kExpected) {
            return RhiTestResult::fail("streamer graph output bytes did not match expected pattern");
        }
        return RhiTestResult::pass();
    }
};

class StreamerRenderGraphInvalidDoesNotBeginFrameTest : public RhiTest {
public:
    StreamerRenderGraphInvalidDoesNotBeginFrameTest()
    {
        type = RhiTestType::Command;
        name = "streamer_render_graph_invalid_does_not_begin_frame";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerStreamerGraphPass();

        render::Queue* computeQueue = context.device.getQueue(render::QueueType::Compute);
        if (computeQueue == nullptr) {
            return RhiTestResult::skip("device has no compute queue");
        }

        render::RenderGraph graph;
        graph.setName("StreamerInvalidQueues");
        graph.addNode("StreamerCrossQueueSourcePass", "Source");
        graph.addNode("StreamerCrossQueueSinkPass", "Sink");
        graph.addEdge("Source.data", "Sink.data");
        graph.markOutput("Sink.copied");

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result result = executor.compile(context.device, graph, 1, 1, log);
        if (!result) {
            return RhiTestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        const render::RenderGraphStreamingStats before = executor.streamingStats();
        result = executor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = nullptr,
            .computeQueue = computeQueue,
        });
        if (!render::hasError(result, render::Error::InvalidArgument)) {
            return RhiTestResult::fail(
                std::string("expected InvalidArgument for missing graphics queue, got ") +
                toString(result));
        }

        const render::RenderGraphStreamingStats& after = executor.streamingStats();
        if (after.frameIndex != before.frameIndex ||
            after.streamer.frameIndex != before.streamer.frameIndex ||
            after.flushCount != before.flushCount ||
            after.transferCount != before.transferCount) {
            return RhiTestResult::fail("invalid submit started or mutated the RenderGraph streaming frame");
        }
        return RhiTestResult::pass();
    }
};

class StreamerMeshletResidencyUploadTest : public RhiTest {
public:
    StreamerMeshletResidencyUploadTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_residency_upload";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        const std::filesystem::path streamAssetPath = context.outputDirectory / "streamer_residency.meshstream.bin";
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            streamAssetPath,
            asset,
            scene::MeshletStreamPayloadCompression::ByteRle);
        if (!build.passed) {
            return build;
        }

        std::string reason;
        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        const uint64_t fallbackResidentBytes = pageStorageBytes(asset, fallbackPages);
        const uint32_t maxResidentPages = std::max<uint32_t>(
            static_cast<uint32_t>(fallbackPages.size()) + 2u,
            4u);
        const uint64_t maxResidentBytes =
            fallbackResidentBytes + 2ull * alignStreamStorageBytes(asset.maxPagePayloadBytes());

        render::MeshletStreamResidencyManager residency;
        if (!residency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentBytes = maxResidentBytes,
                    .maxResidentPages = maxResidentPages,
                    .queuedFrameCount = 2,
                    .pageLoadConcurrency = 1,
                    .maxPageLoadsInFlight = 2,
                },
                reason)) {
            return RhiTestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        render::MeshletStreamResidencyStats sparseStats = residency.stats();
        if (sparseStats.pageCount != asset.pageCount() ||
            sparseStats.trackedPageCount != 0 ||
            residency.trackedPageCount() != 0) {
            return RhiTestResult::fail("residency initialization eagerly tracked unloaded scene pages");
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RhiTestResult::fail("lockFallbackPages failed: " + reason);
        }
        render::MeshletStreamResidencyStats stats = residency.stats();
        if (residency.activePages().size() != fallbackPages.size() ||
            !residency.residentPages().empty() ||
            !residency.pendingPages().empty() ||
            stats.activePageCount != fallbackPages.size() ||
            stats.storageAllocationCount != fallbackPages.size() ||
            stats.usedResidentBytes != fallbackResidentBytes ||
            stats.freeResidentBytes != maxResidentBytes - fallbackResidentBytes ||
            stats.queuedRequestTaskCount != 0 ||
            stats.availableRequestTaskCount != render::kStreamingMaxActiveTasks ||
            stats.queuedStorageTaskCount != 0 ||
            stats.availableStorageTaskCount != render::kStreamingMaxActiveTasks ||
            stats.queuedUpdateTaskCount != 0 ||
            stats.availableUpdateTaskCount != render::kStreamingMaxActiveTasks ||
            stats.totalQueuedUploadCount != fallbackPages.size()) {
            return RhiTestResult::fail("fallback lock did not populate active/storage residency tables");
        }

        std::unique_ptr<render::Streamer> streamer;
        render::Result result = context.device.createStreamer(makeTestStreamerDesc(), streamer);
        if (!result || streamer == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = residency.pageBufferSize(),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            pageBuffer);
        if (!result || pageBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(pageBuffer) returned ") + toString(result));
        }

        residency.beginFrame();
        if (fallbackPages.empty() || residency.queuedUploadCount() == 0) {
            return RhiTestResult::fail("lockFallbackPages did not queue fallback uploads");
        }
        const uint32_t pageIndex = fallbackPages.front();
        std::vector<render::StreamPageTableEntry> initialTable(asset.pageCount());
        residency.buildInitialPageTable(initialTable);
        if (render::streamPageTableDeviceOffset(initialTable[pageIndex]) !=
                render::kInvalidStreamDeviceOffsetBytes ||
            render::streamPageTableState(initialTable[pageIndex]) !=
                render::MeshletStreamPageResidencyState::Unloaded) {
            return RhiTestResult::fail("initial stream page table entry did not encode missing fallback page");
        }
        if (asset.pages()[pageIndex].compressionMode !=
            static_cast<uint32_t>(scene::MeshletStreamPayloadCompression::ByteRle)) {
            return RhiTestResult::fail("compressed streamasset did not preserve ByteRle page metadata");
        }
        residency.clearPendingPatches();

        const bool alreadyResident = residency.requestPage(pageIndex);
        if (alreadyResident || residency.queuedUploadCount() == 0) {
            return RhiTestResult::fail("fallback page was resident before upload");
        }
        uint32_t uploaded = 0;
        const auto pageLoadDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (uploaded == 0 && std::chrono::steady_clock::now() < pageLoadDeadline) {
            uploaded = residency.processUploads(*streamer, *pageBuffer, 1);
            if (uploaded == 0) {
                std::this_thread::yield();
            }
        }
        if (uploaded != 1) {
            return RhiTestResult::fail("asynchronous page load did not schedule exactly one upload");
        }
        if (residency.pageState(pageIndex) != render::MeshletStreamPageResidencyState::PendingUpload) {
            return RhiTestResult::fail("uploaded page did not enter PendingUpload state");
        }
        stats = residency.stats();
        if (residency.pendingPages().size() != 1 ||
            residency.pendingPages().front() != pageIndex ||
            stats.pendingPageCount != 1 ||
            stats.residentPageCount != 0 ||
            stats.queuedStorageTaskCount != 1 ||
            stats.availableStorageTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            stats.queuedUpdateTaskCount != 0 ||
            stats.availableUpdateTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            stats.frameScheduledUploadCount != 1 ||
            stats.pageLoadConcurrency != 1 ||
            stats.frameScheduledPageLoadCount == 0 ||
            stats.frameCompletedPageLoadCount == 0 ||
            stats.framePageLoadFailureCount != 0 ||
            stats.oldestPendingAge != 0) {
            return RhiTestResult::fail("pending upload did not update pending table or upload stats");
        }
        std::span<const render::StreamPageTablePatch> patches = residency.pendingPatches();
        if (patches.size() != 1 ||
            patches[0].pageId != pageIndex ||
            render::streamPageTablePatchDeviceOffset(patches[0]) ==
                render::kInvalidStreamDeviceOffsetBytes ||
            render::streamPageTablePatchState(patches[0]) !=
                render::MeshletStreamPageResidencyState::PendingUpload) {
            return RhiTestResult::fail("pending upload did not produce expected page table patch");
        }
        residency.clearPendingPatches();

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RhiTestResult setup = createCommandResources(
            context.device,
            context.graphicsQueue,
            commandPool,
            commandBuffer,
            fence);
        if (!setup.passed) {
            return setup;
        }

        result = commandBuffer->begin();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::BufferBarrierDesc toTransfer{
            .buffer = pageBuffer.get(),
            .before = render::ResourceState::Undefined,
            .after = render::ResourceState::TransferDestination,
            .offset = 0,
            .size = pageBuffer->desc().size,
        };
        commandBuffer->barrier(render::BarrierDesc{
            .buffers = &toTransfer,
            .bufferCount = 1,
        });
        commandBuffer->copyStreamedData(*streamer);
        result = commandBuffer->end();
        if (!result) {
            return RhiTestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RhiTestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }

        residency.beginFrame();
        if (residency.pageResident(pageIndex)) {
            return RhiTestResult::fail("page became resident before queued frame delay elapsed");
        }
        if (!residency.newlyResidentPages().empty() || !residency.newlyUnloadedPages().empty()) {
            return RhiTestResult::fail("residency reported a lifecycle transition before upload completion");
        }
        stats = residency.stats();
        if (stats.pendingPageCount != 1 ||
            stats.residentPageCount != 0 ||
            stats.queuedStorageTaskCount != 1 ||
            stats.availableStorageTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            stats.queuedUpdateTaskCount != 0 ||
            stats.availableUpdateTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            stats.oldestPendingAge != 1) {
            return RhiTestResult::fail("pending table age did not advance while upload was delayed");
        }
        if (!residency.pendingPatches().empty()) {
            return RhiTestResult::fail("residency produced a patch before pending upload completed");
        }
        residency.beginFrame();
        if (residency.pageResident(pageIndex)) {
            return RhiTestResult::fail("page became resident before queued update task elapsed");
        }
        if (!residency.newlyResidentPages().empty() || !residency.newlyUnloadedPages().empty()) {
            return RhiTestResult::fail("residency reported a lifecycle transition before the update task completed");
        }
        stats = residency.stats();
        if (stats.pendingPageCount != 1 ||
            stats.residentPageCount != 0 ||
            stats.queuedStorageTaskCount != 0 ||
            stats.availableStorageTaskCount != render::kStreamingMaxActiveTasks ||
            stats.queuedUpdateTaskCount != 1 ||
            stats.availableUpdateTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            stats.frameCompletedStorageTaskCount != 1 ||
            stats.frameScheduledUpdateCount != 1 ||
            stats.oldestPendingAge != 2) {
            return RhiTestResult::fail("storage completion did not queue a resident update task");
        }
        if (!residency.pendingPatches().empty()) {
            return RhiTestResult::fail("storage completion produced a resident patch before update task completed");
        }
        residency.beginFrame();
        if (!residency.pageResident(pageIndex)) {
            return RhiTestResult::fail("page did not become resident after queued update task elapsed");
        }
        if (residency.newlyResidentPages().size() != 1 ||
            residency.newlyResidentPages().front() != pageIndex ||
            !residency.newlyUnloadedPages().empty()) {
            return RhiTestResult::fail("completed upload did not report the newly resident page");
        }
        stats = residency.stats();
        if (residency.pendingPages().size() != 0 ||
            residency.residentPages().size() != 1 ||
            residency.residentPages().front() != pageIndex ||
            stats.pendingPageCount != 0 ||
            stats.residentPageCount != 1 ||
            stats.queuedStorageTaskCount != 0 ||
            stats.availableStorageTaskCount != render::kStreamingMaxActiveTasks ||
            stats.queuedUpdateTaskCount != 0 ||
            stats.availableUpdateTaskCount != render::kStreamingMaxActiveTasks ||
            stats.frameCompletedUpdateCount != 1 ||
            stats.frameCompletedUploadCount != 1 ||
            stats.oldestResidentAge != residency.pageAge(pageIndex)) {
            return RhiTestResult::fail("resident upload did not update resident table or completion stats");
        }
        patches = residency.pendingPatches();
        if (patches.size() != 1 ||
            patches[0].pageId != pageIndex ||
            render::streamPageTablePatchDeviceOffset(patches[0]) ==
                render::kInvalidStreamDeviceOffsetBytes ||
            render::streamPageTablePatchState(patches[0]) !=
                render::MeshletStreamPageResidencyState::LockedFallback) {
            return RhiTestResult::fail("resident fallback did not produce expected page table patch");
        }

        const uint64_t deviceOffset = residency.deviceOffsetForPage(pageIndex);
        if (deviceOffset == UINT64_MAX) {
            return RhiTestResult::fail("resident page has no device offset");
        }

        std::vector<uint8_t> actual(static_cast<size_t>(asset.pages()[pageIndex].uncompressedSize));
        pageBuffer->invalidate(deviceOffset, actual.size());
        void* mapped = pageBuffer->map();
        if (mapped == nullptr) {
            return RhiTestResult::fail("page buffer did not map");
        }
        std::memcpy(
            actual.data(),
            static_cast<uint8_t*>(mapped) + deviceOffset,
            actual.size());
        pageBuffer->unmap();

        std::vector<uint8_t> expectedStorage;
        std::span<const uint8_t> expected;
        std::string decodeReason;
        if (!scene::decodeMeshletStreamPayloadForDevice(
                asset.pages()[pageIndex],
                asset.pagePayload(pageIndex),
                expectedStorage,
                expected,
                decodeReason)) {
            return RhiTestResult::fail("failed to decode compressed expected streamasset payload: " + decodeReason);
        }
        if (actual.size() != expected.size() ||
            std::memcmp(actual.data(), expected.data(), expected.size()) != 0) {
            return RhiTestResult::fail("streamed page payload bytes did not match decoded streamasset payload");
        }
        return RhiTestResult::pass();
    }
};

class MeshletStreamClasPagePlanTest : public RhiTest {
public:
    MeshletStreamClasPagePlanTest()
    {
        type = RhiTestType::Validation;
        name = "meshlet_stream_clas_page_plan";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_clas_page_plan.meshstream.bin",
            asset,
            scene::MeshletStreamPayloadCompression::ByteRle);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> pageClusterOffsets;
        uint32_t clusterCount = 0;
        std::string reason;
        if (!render::buildMeshletStreamPageClusterOffsets(
                asset,
                pageClusterOffsets,
                clusterCount,
                reason)) {
            return RhiTestResult::fail("buildMeshletStreamPageClusterOffsets failed: " + reason);
        }
        if (pageClusterOffsets.size() != static_cast<size_t>(asset.pageCount()) + 1u ||
            pageClusterOffsets.back() != clusterCount ||
            clusterCount == 0) {
            return RhiTestResult::fail("CLAS page cluster offsets did not cover the streamasset");
        }

        const std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        if (fallbackPages.empty()) {
            return RhiTestResult::fail("streamasset has no fallback page for CLAS planning");
        }
        const uint32_t pageIndex = fallbackPages.front();
        render::MeshletStreamClasPagePlan plan;
        if (!render::buildMeshletStreamClasPagePlan(
                asset,
                pageIndex,
                pageClusterOffsets[pageIndex],
                plan,
                reason)) {
            return RhiTestResult::fail("buildMeshletStreamClasPagePlan failed: " + reason);
        }

        const scene::MeshletStreamPageInfo& page = asset.pages()[pageIndex];
        if (plan.pageIndex != pageIndex ||
            plan.firstClusterId != pageClusterOffsets[pageIndex] ||
            plan.primitiveIndex != page.primitiveIndex ||
            plan.lodLevel != page.lodLevel ||
            plan.payloadByteSize != page.uncompressedSize ||
            plan.clusters.size() != page.clusterCount) {
            return RhiTestResult::fail("CLAS page plan did not preserve streamasset page metadata");
        }

        for (uint32_t clusterIndex = 0; clusterIndex < plan.clusters.size(); ++clusterIndex) {
            const render::MeshletStreamClasClusterInput& cluster = plan.clusters[clusterIndex];
            if (cluster.clusterId != pageClusterOffsets[pageIndex] + clusterIndex ||
                cluster.pageIndex != pageIndex ||
                cluster.clusterIndex != clusterIndex ||
                cluster.primitiveIndex != page.primitiveIndex ||
                cluster.vertexCount == 0 ||
                cluster.triangleCount == 0 ||
                cluster.vertexOffsetBytes >= page.uncompressedSize ||
                cluster.triangleOffsetBytes >= page.uncompressedSize) {
                return RhiTestResult::fail("CLAS page plan contains an invalid cluster build input");
            }
        }
        return RhiTestResult::pass();
    }
};

class StreamerMeshletResidencyGpuRequestPatchTest : public RhiTest {
public:
    StreamerMeshletResidencyGpuRequestPatchTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_residency_gpu_request_patches";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_gpu_request.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RhiTestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
        }
        const uint32_t firstPage = streamablePages[0];
        const uint32_t secondPage = streamablePages[1];
        const uint64_t fallbackResidentBytes = pageStorageBytes(asset, fallbackPages);
        const uint64_t secondPageBytes = pageStorageBytes(asset, secondPage);

        render::MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentBytes = fallbackResidentBytes + secondPageBytes,
                    .maxResidentPages = static_cast<uint32_t>(fallbackPages.size()) + 1u,
                    .queuedFrameCount = 2,
                },
                reason)) {
            return RhiTestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RhiTestResult::fail("lockFallbackPages failed: " + reason);
        }
        residency.clearPendingPatches();

        const std::array<uint32_t, 4> gpuRequests = {
            firstPage,
            firstPage,
            secondPage,
            secondPage,
        };
        const uint32_t scheduled = residency.consumeGpuRequests(gpuRequests);
        if (scheduled != 2) {
            return RhiTestResult::fail("consumeGpuRequests did not deduplicate and schedule page ids");
        }
        const std::span<const uint32_t> requestedPages = residency.requestedPages();
        if (requestedPages.size() != 2 ||
            std::find(requestedPages.begin(), requestedPages.end(), firstPage) == requestedPages.end() ||
            std::find(requestedPages.begin(), requestedPages.end(), secondPage) == requestedPages.end()) {
            return RhiTestResult::fail("request table did not preserve unique GPU-requested page ids");
        }
        render::MeshletStreamResidencyStats requestStats = residency.stats();
        if (requestStats.frameGpuRequestCount != gpuRequests.size() ||
            requestStats.frameUniqueGpuRequestCount != 2 ||
            requestStats.frameScheduledRequestTaskCount != 1 ||
            requestStats.frameConsumedGpuRequestCount != 0 ||
            requestStats.queuedRequestTaskCount != 1 ||
            requestStats.availableRequestTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            requestStats.trackedPageCount != fallbackPages.size() ||
            requestStats.activePageCount != fallbackPages.size() ||
            requestStats.freeResidentBytes != secondPageBytes) {
            return RhiTestResult::fail("GPU request readback did not queue an isolated request task");
        }
        if (residency.pageAllocated(firstPage) ||
            residency.pageAllocated(secondPage) ||
            !residency.pendingPatches().empty()) {
            return RhiTestResult::fail("queued GPU request task modified residency before beginFrame consumed it");
        }

        residency.beginFrame();
        const std::span<const uint32_t> consumedRequestPages = residency.requestedPages();
        if (consumedRequestPages.size() != 2 ||
            std::find(consumedRequestPages.begin(), consumedRequestPages.end(), firstPage) == consumedRequestPages.end() ||
            std::find(consumedRequestPages.begin(), consumedRequestPages.end(), secondPage) == consumedRequestPages.end()) {
            return RhiTestResult::fail("request task did not preserve unique page ids when consumed");
        }
        if (residency.pageAllocated(firstPage)) {
            return RhiTestResult::fail("older requested page received storage despite latest-page pressure");
        }
        if (!residency.pageAllocated(secondPage)) {
            return RhiTestResult::fail("latest requested page did not receive the single streamable allocation");
        }
        const std::span<const uint32_t> activePages = residency.activePages();
        requestStats = residency.stats();
        if (std::find(activePages.begin(), activePages.end(), firstPage) != activePages.end() ||
            std::find(activePages.begin(), activePages.end(), secondPage) == activePages.end() ||
            requestStats.frameCompletedRequestTaskCount != 1 ||
            requestStats.frameConsumedGpuRequestCount != 2 ||
            requestStats.queuedRequestTaskCount != 0 ||
            requestStats.availableRequestTaskCount != render::kStreamingMaxActiveTasks ||
            requestStats.frameEvictedPageCount != 0 ||
            requestStats.frameResidentBudgetFailureCount != 1 ||
            requestStats.frameAllocationFailureCount != 1 ||
            requestStats.trackedPageCount != fallbackPages.size() + 1u ||
            requestStats.activePageCount != fallbackPages.size() + 1u ||
            requestStats.freeResidentBytes != 0) {
            return RhiTestResult::fail("active/request/storage stats did not track GPU request pressure");
        }

        if (!residency.pendingPatches().empty()) {
            return RhiTestResult::fail("budget-limited request emitted an unexpected eviction patch");
        }
        return RhiTestResult::pass();
    }
};

class StreamerMeshletResidencyLatestGpuRequestTest : public RhiTest {
public:
    StreamerMeshletResidencyLatestGpuRequestTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_residency_latest_gpu_request";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_latest_gpu_request.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RhiTestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
        }
        const uint32_t stalePage = streamablePages[0];
        const uint32_t latestPage = streamablePages[1];
        const uint64_t maxResidentBytes =
            pageStorageBytes(asset, fallbackPages) + pageStorageBytes(asset, latestPage);

        render::MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentBytes = maxResidentBytes,
                    .maxResidentPages = static_cast<uint32_t>(fallbackPages.size()) + 1u,
                    .queuedFrameCount = 2,
                },
                reason)) {
            return RhiTestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RhiTestResult::fail("lockFallbackPages failed: " + reason);
        }
        residency.clearPendingPatches();

        const uint32_t staleScheduled = residency.consumeGpuRequests(std::span<const uint32_t>(&stalePage, 1));
        const uint32_t latestScheduled = residency.consumeGpuRequests(std::span<const uint32_t>(&latestPage, 1));
        if (staleScheduled != 1 || latestScheduled != 1) {
            return RhiTestResult::fail("consumeGpuRequests did not schedule two request tasks");
        }

        render::MeshletStreamResidencyStats stats = residency.stats();
        if (stats.queuedRequestTaskCount != 2 ||
            stats.availableRequestTaskCount != render::kStreamingMaxActiveTasks - 2u ||
            stats.frameScheduledRequestTaskCount != 2 ||
            stats.frameUniqueGpuRequestCount != 2 ||
            stats.frameConsumedGpuRequestCount != 0) {
            return RhiTestResult::fail("multiple GPU request readbacks were not queued as separate tasks");
        }

        residency.beginFrame();
        const std::span<const uint32_t> requestedPages = residency.requestedPages();
        stats = residency.stats();
        if (requestedPages.size() != 1 ||
            requestedPages.front() != latestPage ||
            residency.pageAllocated(stalePage) ||
            !residency.pageAllocated(latestPage) ||
            stats.frameDroppedRequestTaskCount != 1 ||
            stats.frameCompletedRequestTaskCount != 1 ||
            stats.frameConsumedGpuRequestCount != 1 ||
            stats.queuedRequestTaskCount != 0 ||
            stats.availableRequestTaskCount != render::kStreamingMaxActiveTasks) {
            return RhiTestResult::fail("request queue did not drop stale ready tasks and consume the latest request");
        }

        return RhiTestResult::pass();
    }
};

class StreamerMeshletScreenPriorityTest : public RhiTest {
public:
    StreamerMeshletScreenPriorityTest() { type = RhiTestType::Command; name = "streamer_meshlet_screen_priority"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "screen_priority.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (roots.empty() || pages.size() < 2) { return RhiTestResult::skip("Needs two streamable pages"); }
        std::sort(pages.begin(), pages.end(), [&](uint32_t a, uint32_t b) {
            return asset.pages()[a].uncompressedSize < asset.pages()[b].uncompressedSize;
        });
        const uint32_t small = pages.front(), large = pages.back();
        const uint64_t bytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, large);
        const auto verify = [&](std::span<const uint32_t> ids, std::span<const float> benefits,
                                uint32_t winner) -> std::string {
            MeshletStreamResidencyManager residency;
            std::string reason;
            if (!residency.initialize({.asset = &asset, .maxResidentBytes = bytes,
                    .maxResidentPages = static_cast<uint32_t>(roots.size() + 1), .queuedFrameCount = 2}, reason) ||
                !residency.lockFallbackPages(roots, reason)) { return reason; }
            (void)residency.consumeGpuRequests({.loadPageIds = ids, .loadPriorities = benefits});
            residency.beginFrame();
            if (!residency.pageAllocated(winner) || residency.pageAllocated(winner == small ? large : small)) {
                return "Screen benefit did not choose the sole streamable admission slot";
            }
            for (uint32_t root : roots) {
                if (!residency.pageAllocated(root)) { return "Priority displaced a locked fallback"; }
            }
            return {};
        };
        const uint32_t ids[] = {large, small, large};
        const float duplicateMax[] = {1.f, 10.f, 1000000.f};
        std::string reason = verify(ids, duplicateMax, large);
        if (!reason.empty()) { return RhiTestResult::fail(reason); }
        const uint32_t two[] = {large, small};
        const float equal[] = {100.f, 100.f};
        reason = verify(two, equal, small);
        if (!reason.empty()) { return RhiTestResult::fail("Benefit per byte: " + reason); }
        const float invalid[] = {std::numeric_limits<float>::quiet_NaN(), 100.f};
        reason = verify(two, invalid, small);
        if (!reason.empty()) { return RhiTestResult::fail("Non-finite feedback: " + reason); }
        const float shortScores[] = {100.f};
        reason = verify(two, shortScores, large);
        if (!reason.empty()) { return RhiTestResult::fail("Short feedback: " + reason); }
        return RhiTestResult::pass("Shared-page maximum, benefit per byte, roots, NaN and short feedback");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletScreenPriorityTest);

class StreamerMeshletPrefetchTest final : public RhiTest {
public:
    StreamerMeshletPrefetchTest() { type = RhiTestType::Command; name = "streamer_meshlet_prefetch_admission"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "prefetch.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        const auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 3) { return RhiTestResult::skip("Needs three streamable pages"); }
        const uint64_t bytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages);
        std::string reason;
        MeshletStreamResidencyManager legacy, immediate;
        if (!legacy.initialize({.asset = &asset, .maxResidentBytes = bytes}, reason) ||
            !immediate.initialize({.asset = &asset, .maxResidentBytes = bytes,
                .measurePageLatency = true, .immediateGpuRequests = true}, reason)) { return RhiTestResult::fail(reason); }
        legacy.beginFrame(); immediate.beginFrame();
        const uint32_t current[] = {pages[0]};
        (void)legacy.consumeGpuRequests({.loadPageIds = current, .frameIndex = 1});
        (void)immediate.consumeGpuRequests({.loadPageIds = current, .frameIndex = 1});
        if (legacy.pageAllocated(pages[0]) || !immediate.pageAllocated(pages[0])) {
            return RhiTestResult::fail("Immediate requests did not remove exactly the admission frame");
        }
        legacy.beginFrame();
        if (!legacy.pageAllocated(pages[0])) { return RhiTestResult::fail("Legacy admission changed"); }
        const uint32_t duplicate[] = {pages[1] | kStreamPrefetchPageTag, pages[1]};
        (void)immediate.consumeGpuRequests({.loadPageIds = duplicate, .frameIndex = 1, .taggedPrefetchRequests = true});
        if (!immediate.pageAllocated(pages[1]) || immediate.stats().totalPrefetchAdmitted != 0 ||
            immediate.latencySnapshot().pendingDemand != 2 || immediate.latencySnapshot().pendingPrefetch != 0) {
            return RhiTestResult::fail("Actual demand did not win a tagged duplicate without priorities");
        }
        // One page of speculation fits, but the next cannot evict it or use the
        // reserved quarter. Current demand can use that remaining capacity.
        MeshletStreamResidencyManager bounded;
        const uint64_t largest = std::max(pageStorageBytes(asset, pages[0]), pageStorageBytes(asset, pages[1]));
        if (!bounded.initialize({.asset = &asset, .maxResidentBytes = largest * 4,
                .maxResidentPages = 2, .immediateGpuRequests = true}, reason)) { return RhiTestResult::fail(reason); }
        bounded.beginFrame();
        const uint32_t forecasts[] = {pages[0] | kStreamPrefetchPageTag, pages[1] | kStreamPrefetchPageTag};
        (void)bounded.consumeGpuRequests({.loadPageIds = forecasts, .taggedPrefetchRequests = true});
        if (bounded.stats().totalPrefetchAdmitted != 1 || bounded.stats().totalPrefetchDeferred != 1 ||
            bounded.stats().totalEvictedPageCount != 0) {
            return RhiTestResult::fail("Speculative admission displaced the demand reserve");
        }
        const uint32_t missing[] = {bounded.pageAllocated(pages[0]) ? pages[1] : pages[0]};
        (void)bounded.consumeGpuRequests({.loadPageIds = missing});
        if (!bounded.pageAllocated(pages[0]) || !bounded.pageAllocated(pages[1])) {
            return RhiTestResult::fail("Actual demand could not use reserved capacity");
        }
        MeshletStreamResidencyManager queued;
        if (!queued.initialize({.asset = &asset, .maxResidentBytes = bytes,
                .pageLoadConcurrency = 1, .maxPageLoadsInFlight = 4, .immediateGpuRequests = true}, reason)) {
            return RhiTestResult::fail(reason);
        }
        const uint32_t queuedForecasts[] = {pages[0] | kStreamPrefetchPageTag,
            pages[1] | kStreamPrefetchPageTag, pages[2] | kStreamPrefetchPageTag};
        (void)queued.consumeGpuRequests({.loadPageIds = queuedForecasts, .taggedPrefetchRequests = true});
        if (queued.queuedUploadCount() != 1 || queued.availablePrefetchRequests() != 0 ||
            queued.stats().totalPrefetchDeferred != 2) {
            return RhiTestResult::fail("Unissued speculative I/O occupied the demand queue reserve");
        }
        (void)queued.consumeGpuRequests({.loadPageIds = std::span(pages).first(3)});
        if (queued.queuedUploadCount() != 3 || queued.stats().totalPrefetchUsed != 1) {
            return RhiTestResult::fail("Demand could not promote or bypass queued speculation");
        }
        return RhiTestResult::pass("Immediate/legacy admission, tagged promotion, memory and queue reserves, no speculative eviction");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletPrefetchTest);

class StreamerMeshletLatencyTest final : public RhiTest {
public:
    StreamerMeshletLatencyTest() { type = RhiTestType::Validation; name = "streamer_meshlet_latency"; }
    RhiTestResult run(RhiTestContext&) override
    {
        using namespace render;
        MeshletStreamLatencyHistogram histogram;
        if (histogram.summary().count != 0) { return RhiTestResult::fail("Empty latency histogram"); }
        for (uint32_t i = 0; i < 99; ++i) { histogram.observe(999); }
        histogram.observe(5000123);
        const auto sample = histogram.summary();
        if (sample.count != 100 || sample.p50 != 1 || sample.p99 != 1 || sample.maximum != 5000.123) {
            return RhiTestResult::fail("Latency percentile or overflow bin lost the tail");
        }
        MeshletStreamLatencyTracker tracker;
        tracker.request(1, 2, 3, true);
        tracker.request(1, 4, 5, false);
        tracker.request(1, 6, 7, false);
        tracker.complete(1, 10);
        tracker.complete(1, 10);
        tracker.request(2, 7, 8, true); tracker.abandon(2);
        tracker.request(3, 7, 8, false); tracker.abandon(3);
        tracker.request(4, 7, 8, true);
        const auto snapshot = tracker.snapshot();
        if (snapshot.demandFrames.count != 1 || snapshot.demandFrames.p95 != 6 ||
            snapshot.abandonedDemand != 1 || snapshot.abandonedPrefetch != 1 ||
            snapshot.pendingDemand != 0 || snapshot.pendingPrefetch != 1 ||
            snapshot.milliseconds[size_t(MeshletStreamLatencyStage::Feedback)].count != 4) {
            return RhiTestResult::fail("Latency duplicate, first demand frame, promotion or retirement accounting");
        }
        return RhiTestResult::pass("Histogram tail, source frames, promotion, duplicates and abandoned requests");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletLatencyTest);

class StreamerMeshletResidencyGpuRequestUnloadOverflowTest : public RhiTest {
public:
    StreamerMeshletResidencyGpuRequestUnloadOverflowTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_residency_gpu_request_unload_overflow";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_gpu_request_unload.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RhiTestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
        }
        const uint32_t unloadPage = streamablePages[0];
        const uint32_t loadPage = streamablePages[1];
        const uint64_t streamableBudgetBytes = std::max(
            pageStorageBytes(asset, unloadPage),
            pageStorageBytes(asset, loadPage));
        const uint64_t maxResidentBytes = pageStorageBytes(asset, fallbackPages) + streamableBudgetBytes;

        render::MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentBytes = maxResidentBytes,
                    .maxResidentPages = static_cast<uint32_t>(fallbackPages.size()) + 1u,
                    .queuedFrameCount = 2,
                },
                reason)) {
            return RhiTestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RhiTestResult::fail("lockFallbackPages failed: " + reason);
        }
        residency.clearPendingPatches();

        (void)residency.requestPage(unloadPage);
        if (!residency.pageAllocated(unloadPage)) {
            return RhiTestResult::fail("test setup did not allocate streamable page storage");
        }

        const std::array<uint32_t, 2> loadRequests = {loadPage, loadPage};
        const std::array<uint32_t, 2> unloadRequests = {unloadPage, unloadPage};
        const uint32_t scheduled = residency.consumeGpuRequests(render::StreamGpuRequestBatch{
            .loadPageIds = loadRequests,
            .unloadPageIds = unloadRequests,
            .loadRequestCounter = 3,
            .unloadRequestCounter = 3,
            .loadOverflowCounter = 1,
            .unloadOverflowCounter = 1,
            .invalidPageCounter = 1,
            .frameIndex = 37,
        });
        if (scheduled != 2) {
            return RhiTestResult::fail("load/unload GPU request batch did not schedule unique page ids");
        }

        render::MeshletStreamResidencyStats stats = residency.stats();
        if (stats.frameGpuRequestCount != 3 ||
            stats.frameUniqueGpuRequestCount != 1 ||
            stats.frameGpuUnloadRequestCount != 3 ||
            stats.frameUniqueGpuUnloadRequestCount != 1 ||
            stats.frameGpuRequestOverflowCount != 1 ||
            stats.frameGpuUnloadRequestOverflowCount != 1 ||
            stats.frameGpuInvalidRequestCount != 1 ||
            stats.frameScheduledRequestTaskCount != 1 ||
            stats.queuedRequestTaskCount != 1) {
            return RhiTestResult::fail("GPU request load/unload overflow stats were not tracked");
        }
        if (residency.requestedPages().size() != 1 ||
            residency.requestedPages().front() != loadPage ||
            residency.unloadRequestedPages().size() != 1 ||
            residency.unloadRequestedPages().front() != unloadPage) {
            return RhiTestResult::fail("GPU request batch did not preserve unique load/unload page ids");
        }

        residency.beginFrame();
        stats = residency.stats();
        if (!residency.pageAllocated(unloadPage) ||
            residency.pageState(unloadPage) != render::MeshletStreamPageResidencyState::PendingUnload ||
            residency.pageAllocated(loadPage) ||
            stats.frameCompletedRequestTaskCount != 1 ||
            stats.frameConsumedGpuRequestCount != 1 ||
            stats.frameConsumedGpuUnloadRequestCount != 1 ||
            stats.frameScheduledUnloadCount != 1 ||
            stats.queuedUnloadTaskCount != 1 ||
            stats.frameDelayedFreeCount != 0 ||
            stats.frameResidentBudgetFailureCount != 1 ||
            stats.frameEvictedPageCount != 0) {
            return RhiTestResult::fail("GPU unload request did not enter delayed-free state before consuming loads");
        }
        if (residency.requestedPages().size() != 1 ||
            residency.requestedPages().front() != loadPage ||
            residency.unloadRequestedPages().size() != 1 ||
            residency.unloadRequestedPages().front() != unloadPage) {
            return RhiTestResult::fail("completed request task did not expose consumed load/unload ids");
        }

        std::span<const render::StreamPageTablePatch> patches = residency.pendingPatches();
        if (patches.empty() ||
            std::find_if(
                patches.begin(),
                patches.end(),
                [unloadPage](const render::StreamPageTablePatch& patch) {
                    return patch.pageId == unloadPage &&
                        render::streamPageTablePatchState(patch) ==
                            render::MeshletStreamPageResidencyState::PendingUnload;
                }) == patches.end()) {
            return RhiTestResult::fail("GPU unload request did not emit a pending-unload page table patch");
        }

        residency.clearPendingPatches();
        residency.beginFrame();
        stats = residency.stats();
        if (residency.pageAllocated(unloadPage) ||
            residency.pageState(unloadPage) != render::MeshletStreamPageResidencyState::Unloaded ||
            stats.trackedPageCount != fallbackPages.size() ||
            stats.frameCompletedUnloadCount != 1 ||
            stats.frameDelayedFreeCount != 1 ||
            stats.freeResidentBytes != streamableBudgetBytes) {
            return RhiTestResult::fail("delayed unload task did not free resident page storage");
        }
        if (residency.newlyUnloadedPages().size() != 1 ||
            residency.newlyUnloadedPages().front() != unloadPage ||
            !residency.newlyResidentPages().empty()) {
            return RhiTestResult::fail("completed unload did not report the newly unloaded page");
        }

        (void)residency.requestPage(loadPage);
        if (!residency.pageAllocated(loadPage) ||
            stats.frameCompletedUnloadCount != 1) {
            return RhiTestResult::fail("load request did not acquire storage after delayed free completed");
        }

        patches = residency.pendingPatches();
        if (std::find_if(
                patches.begin(),
                patches.end(),
                [unloadPage](const render::StreamPageTablePatch& patch) {
                    return patch.pageId == unloadPage &&
                        render::streamPageTablePatchDeviceOffset(patch) ==
                            render::kInvalidStreamDeviceOffsetBytes &&
                        render::streamPageTablePatchState(patch) ==
                            render::MeshletStreamPageResidencyState::Unloaded;
                }) == patches.end()) {
            return RhiTestResult::fail("delayed unload completion did not emit an unloaded page table patch");
        }

        return RhiTestResult::pass();
    }
};

class StreamerMeshletResidencyEvictionDelayAgeTest : public RhiTest {
public:
    StreamerMeshletResidencyEvictionDelayAgeTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_residency_eviction_delay_age";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RhiTestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_eviction_delay_age.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RhiTestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
        }
        const uint32_t residentPage = streamablePages[0];
        const uint32_t requestedPage = streamablePages[1];
        const uint64_t streamableBudgetBytes =
            pageStorageBytes(asset, residentPage) +
            pageStorageBytes(asset, requestedPage);
        const uint64_t maxResidentBytes = pageStorageBytes(asset, fallbackPages) + streamableBudgetBytes;

        render::MeshletStreamResidencyManager pageLimitedResidency;
        std::string pageLimitedReason;
        const std::array<uint32_t, 2> overBudgetLockedPages = {fallbackPages.front(), residentPage};
        if (!pageLimitedResidency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentBytes = maxResidentBytes,
                    .maxResidentPages = 1,
                },
                pageLimitedReason) ||
            pageLimitedResidency.lockFallbackPages(overBudgetLockedPages, pageLimitedReason) ||
            !pageLimitedResidency.activePages().empty()) {
            return RhiTestResult::fail("locked fallback pages did not reject the resident page-count budget atomically");
        }

        render::MeshletStreamResidencyManager residency;
        std::string reason;
        constexpr uint32_t kAgeThreshold = 6;
        if (!residency.initialize(
                render::MeshletStreamResidencyDesc{
                    .asset = &asset,
                    .maxResidentBytes = maxResidentBytes,
                    .maxResidentPages = static_cast<uint32_t>(fallbackPages.size()) + 1u,
                    .queuedFrameCount = 1,
                    .unloadDelayFrames = 1,
                    .evictionAgeThresholdFrames = kAgeThreshold,
                },
                reason)) {
            return RhiTestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RhiTestResult::fail("lockFallbackPages failed: " + reason);
        }

        render::StreamerDesc streamerDesc = makeTestStreamerDesc(
            (static_cast<uint64_t>(fallbackPages.size()) + 1ull) * asset.maxPagePayloadBytes() + 4096ull);
        std::unique_ptr<render::Streamer> streamer;
        render::Result result = context.device.createStreamer(streamerDesc, streamer);
        if (!result || streamer == nullptr) {
            return RhiTestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = context.device.createBuffer(
            render::BufferDesc{
                .size = residency.pageBufferSize(),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            },
            pageBuffer);
        if (!result || pageBuffer == nullptr) {
            return RhiTestResult::fail(std::string("createBuffer(pageBuffer) returned ") + toString(result));
        }

        residency.beginFrame();
        (void)residency.requestPage(residentPage);
        const uint32_t uploadBudget = static_cast<uint32_t>(fallbackPages.size()) + 1u;
        if (residency.processUploads(*streamer, *pageBuffer, uploadBudget) != uploadBudget) {
            return RhiTestResult::fail("processUploads did not schedule fallback and streamable uploads");
        }

        residency.beginFrame();
        residency.beginFrame();
        residency.beginFrame();
        if (!residency.pageResident(residentPage)) {
            return RhiTestResult::fail("test setup did not make streamable page resident");
        }

        (void)residency.requestPage(requestedPage);
        render::MeshletStreamResidencyStats stats = residency.stats();
        if (residency.pageAllocated(requestedPage) ||
            residency.pageState(residentPage) != render::MeshletStreamPageResidencyState::Resident ||
            stats.activePageCount != fallbackPages.size() + 1u ||
            stats.freeResidentBytes != pageStorageBytes(asset, requestedPage) ||
            stats.frameEvictionAgeRejectedCount != 1 ||
            stats.frameResidentBudgetFailureCount != 1 ||
            stats.frameScheduledUnloadCount != 0) {
            return RhiTestResult::fail("age filter did not reject eviction of a young resident page");
        }

        for (uint32_t retry = 0; retry < 10000; ++retry) { (void)residency.requestPage(requestedPage); }
        stats = residency.stats();
        if (stats.frameEvictionScanCount != 1 || stats.frameEvictionCandidateTests != stats.residentPageCount ||
            stats.frameAllocationDeferredCount != 10001 || stats.frameAllocationFailureCount != 1 ||
            stats.cpuWork.allocationAttempts != 1 || stats.cpuWork.budgetRetrySuppressed != 10000 ||
            stats.trackedPageCount != fallbackPages.size() + 1 || residency.pageAllocated(requestedPage)) {
            return RhiTestResult::fail("budget pressure repeated an eviction scan or lost age protection");
        }

        while (residency.pageAge(residentPage) < kAgeThreshold) {
            residency.beginFrame();
        }
        (void)residency.requestPage(requestedPage);
        stats = residency.stats();
        if (residency.pageAllocated(requestedPage) ||
            residency.pageState(residentPage) != render::MeshletStreamPageResidencyState::PendingUnload ||
            stats.frameEvictedPageCount != 1 ||
            stats.frameScheduledUnloadCount != 1 ||
            stats.queuedUnloadTaskCount != 1 ||
            stats.frameDelayedFreeCount != 0) {
            return RhiTestResult::fail("eligible eviction did not schedule a delayed unload task");
        }

        residency.beginFrame();
        stats = residency.stats();
        if (residency.pageAllocated(residentPage) ||
            residency.pageState(residentPage) != render::MeshletStreamPageResidencyState::Unloaded ||
            stats.frameCompletedUnloadCount != 1 ||
            stats.frameDelayedFreeCount != 1 ||
            stats.freeResidentBytes != streamableBudgetBytes) {
            return RhiTestResult::fail("delayed eviction did not free its storage on task completion");
        }

        (void)residency.requestPage(requestedPage);
        stats = residency.stats();
        if (!residency.pageAllocated(requestedPage) ||
            stats.activePageCount != fallbackPages.size() + 1u ||
            stats.usedSlotCount > stats.maxResidentPages ||
            stats.freeSlotCount != 0) {
            return RhiTestResult::fail("request did not acquire storage after delayed eviction completed");
        }

        return RhiTestResult::pass();
    }
};

class StreamerMeshletRequestSelectionTest final : public RhiTest {
public:
    StreamerMeshletRequestSelectionTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_request_selection";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "RequestSelection.meshstream.bin", asset);
        if (!built.passed) { return built; }
        auto pages = nonFallbackPagesFor(asset, fallbackPagesFor(asset));
        if (pages.size() < 5) { return RhiTestResult::fail("Need five request candidates"); }
        pages.resize(std::min<size_t>(pages.size(), 64));
        const uint32_t count = static_cast<uint32_t>(pages.size());
        MeshletStreamResidencyManager residency;
        std::string reason;
        const MeshletStreamResidencyDesc desc{.asset = &asset,
            .maxResidentBytes = pageStorageBytes(asset, pages), .maxResidentPages = 3,
            .measurePageLatency = true, .immediateGpuRequests = true};
        // Reinitialize and permute across batches: old lookup slots must not
        // alias a different current scratch vector, or a previous scene lifetime.
        for (uint32_t round = 0; round < 5; ++round) {
            if (!residency.initialize(desc, reason)) { return RhiTestResult::fail(reason); }
            residency.beginFrame();
            std::rotate(pages.begin(), pages.begin() + 1, pages.end());
            std::vector<uint32_t> ids;
            std::vector<float> benefits;
            std::vector<std::pair<double, uint32_t>> expected;
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t page = pages[i];
                const auto bytes = scene::meshletStreamDevicePayloadSize(asset.pages()[page]);
                const float benefit = float((i + 1) * 1000);
                ids.push_back(page | kStreamPrefetchPageTag);
                benefits.push_back(0.0f);
                expected.emplace_back(double(benefit) / double(std::max<uint64_t>(bytes, 1024)), page);
            }
            for (uint32_t i = count; i-- > 0;) {
                ids.push_back(pages[i]);
                benefits.push_back(float((i + 1) * 1000));
            }
            ids.push_back(UINT32_MAX); benefits.push_back(1.0f);
            std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            const auto batch = StreamGpuRequestBatch{.loadPageIds = ids, .frameIndex = 1,
                .loadPriorities = benefits, .taggedPrefetchRequests = true};
            if (residency.consumeGpuRequests(batch) != count) {
                return RhiTestResult::fail("Request selection lost unique candidates");
            }
            for (size_t i = 0; i < expected.size(); ++i) {
                if (residency.pageAllocated(expected[i].second) != (i < 3)) {
                    return RhiTestResult::fail("Heap selection differs from full priority sort");
                }
            }
            auto stats = residency.stats();
            if (stats.cpuWork.admissionCalls != 4 || stats.cpuWork.admissionPriorityPops != 4 ||
                stats.cpuWork.requestDuplicatesMerged != count || stats.frameGpuInvalidRequestCount != 1 ||
                stats.totalPrefetchAdmitted != 0 || residency.latencySnapshot().pendingPrefetch != 0) {
                return RhiTestResult::fail("Demand merge or bounded priority selection did not hold");
            }
            // Repeat while capacity is blocked: allocated work stays alive,
            // missing work remains demanded, neither requires a heap pop/call.
            std::reverse(ids.begin(), ids.end());
            std::reverse(benefits.begin(), benefits.end());
            (void)residency.consumeGpuRequests(batch);
            stats = residency.stats();
            if (stats.cpuWork.admissionCalls != 4 || stats.cpuWork.admissionPriorityPops != 4 ||
                stats.frameUniqueGpuRequestCount != count * 2 || residency.latencySnapshot().pendingDemand != count) {
                return RhiTestResult::fail("Repeated batch re-admitted blocked or queued pages, or lost latency demand");
            }
            residency.beginFrame();
            (void)residency.consumeGpuRequests(batch);
            if (residency.stats().cpuWork.admissionCalls != 1 || residency.queuedUploadCount() != 3) {
                return RhiTestResult::fail("Next frame lost queued demand or failed to refresh capacity eligibility");
            }
        }
        return RhiTestResult::pass("Priority oracle, duplicate promotion, blocked tails, queued keepalive and lookup lifetime");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletRequestSelectionTest);

class StreamerMeshletBudgetAdmissionTest final : public RhiTest {
public:
    StreamerMeshletBudgetAdmissionTest()
    {
        type = RhiTestType::Command;
        name = "streamer_meshlet_budget_admission";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "BudgetAdmission.meshstream.bin", asset);
        if (!built.passed) { return built; }
        auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (roots.empty() || pages.size() < 2) { return RhiTestResult::fail("Need variable-sized pages"); }
        std::sort(pages.begin(), pages.end(), [&](uint32_t a, uint32_t b) {
            return pageStorageBytes(asset, a) < pageStorageBytes(asset, b);
        });
        const uint32_t small = pages.front(), large = pages.back();
        if (pageStorageBytes(asset, small) == pageStorageBytes(asset, large)) {
            return RhiTestResult::fail("Need unequal page allocation sizes");
        }
        MeshletStreamResidencyManager residency;
        std::string reason;
        const MeshletStreamResidencyDesc desc{.asset = &asset,
            .maxResidentBytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, small),
            .immediateGpuRequests = true};
        if (!residency.initialize(desc, reason) || !residency.lockFallbackPages(roots, reason)) {
            return RhiTestResult::fail(reason);
        }
        residency.beginFrame();
        // A higher-priority oversized page must not block a lower-priority fit.
        const std::array<uint32_t, 2> requests{large, small};
        const std::array<float, 2> priorities{1e9f, 1.0f};
        (void)residency.consumeGpuRequests({.loadPageIds = requests, .loadPriorities = priorities});
        for (uint32_t retry = 0; retry < 100; ++retry) { (void)residency.requestPage(large); }
        const auto stats = residency.stats();
        if (!residency.pageAllocated(small) || residency.pageAllocated(large) ||
            stats.cpuWork.allocationAttempts != 2 || stats.cpuWork.budgetRetrySuppressed != 100 ||
            stats.frameAllocationFailureCount != 1 || stats.usedResidentBytes > stats.maxResidentBytes) {
            return RhiTestResult::fail("Budget gate blocked a smaller fit or repeated impossible allocations");
        }
        for (uint32_t root : roots) {
            if (!residency.pageAllocated(root) || residency.unloadPage(root)) {
                return RhiTestResult::fail("Budget gate lost fallback protection");
            }
        }
        residency.beginFrame();
        (void)residency.requestPage(large);
        if (residency.stats().cpuWork.allocationAttempts != 1) {
            return RhiTestResult::fail("New frame did not refresh admission eligibility");
        }
        // Reset/reinitialize must clear both the exhausted gate and its counters.
        if (!residency.initialize({.asset = &asset, .maxResidentBytes = pageStorageBytes(asset, large)}, reason)) {
            return RhiTestResult::fail(reason);
        }
        (void)residency.requestPage(large);
        if (!residency.pageAllocated(large) || residency.stats().cpuWork.budgetRetrySuppressed != 0) {
            return RhiTestResult::fail("Reinitialization retained stale budget pressure");
        }
        return RhiTestResult::pass("Bounded retries, smaller fit, priority order, locked roots and reset");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletBudgetAdmissionTest);

class StreamerMeshletBatchedUnloadTest final : public RhiTest {
public:
    StreamerMeshletBatchedUnloadTest() { type = RhiTestType::Command; name = "streamer_meshlet_batched_unload"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "batched_unload.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 8) { return RhiTestResult::skip("Requires eight streamable pages"); }
        pages.resize(8);
        render::MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize({.asset = &asset,
                .maxResidentBytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages)}, reason) ||
            !residency.lockFallbackPages(roots, reason)) { return RhiTestResult::fail(reason); }
        residency.beginFrame();
        for (uint32_t page : pages) { (void)residency.requestPage(page); }
        for (uint32_t page : pages) {
            if (!residency.pageAllocated(page) || !residency.unloadPage(page)) {
                return RhiTestResult::fail("Batch unload exhausted the task ring");
            }
        }
        if (residency.stats().queuedUnloadTaskCount != 1 || residency.stats().frameScheduledUnloadCount != pages.size()) {
            return RhiTestResult::fail("Same-frame unloads were not batched");
        }
        residency.beginFrame();
        for (uint32_t page : pages) {
            if (residency.pageAllocated(page)) { return RhiTestResult::fail("Batch was not retired after the delayed free"); }
        }
        for (uint32_t root : roots) {
            if (!residency.pageAllocated(root) || residency.unloadPage(root)) { return RhiTestResult::fail("Batch lost a locked root"); }
        }
        return RhiTestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletBatchedUnloadTest);

class StreamerMeshletDemandCacheTest final : public RhiTest {
public:
    StreamerMeshletDemandCacheTest() { type = RhiTestType::Command; name = "streamer_meshlet_demand_cache"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "demand_cache.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 3) { return RhiTestResult::skip("Requires three streamable pages"); }
        pages.resize(3);
        MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize({.asset = &asset,
                .maxResidentBytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages),
                .maxResidentPages = static_cast<uint32_t>(roots.size() + 2), .queuedFrameCount = 1,
                .unloadDelayFrames = 1, .evictionAgeThresholdFrames = 1}, reason) ||
            !residency.lockFallbackPages(roots, reason)) { return RhiTestResult::fail(reason); }
        std::unique_ptr<Streamer> streamer;
        auto result = context.device.createStreamer(makeTestStreamerDesc((roots.size() + 2) * asset.maxPagePayloadBytes() + 4096), streamer);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        std::unique_ptr<Buffer> destination;
        result = context.device.createBuffer({.size = residency.pageBufferSize(),
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}, destination);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        residency.beginFrame();
        (void)residency.requestPage(pages[0]); (void)residency.requestPage(pages[1]);
        const auto uploads = static_cast<uint32_t>(roots.size() + 2);
        if (residency.processUploads(*streamer, *destination, uploads) != uploads) {
            return RhiTestResult::fail("Cannot prepare demand-cache residents");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
        if (!residency.pageResident(pages[0]) || !residency.pageResident(pages[1])) {
            return RhiTestResult::fail("Demand-cache setup is not resident");
        }
        const std::array<uint32_t, 2> firstUnused{pages[0], pages[0]};
        (void)residency.consumeGpuRequests({.unloadPageIds = firstUnused, .unloadRequestCounter = 2,
            .residentDemandFeedback = true});
        if (residency.stats().queuedUnloadTaskCount != 0 || residency.stats().frameCachedUnusedPageCount != 1) {
            return RhiTestResult::fail("Unused feedback eagerly unloaded cached geometry");
        }
        const auto initialWork = residency.stats().cpuWork;
        if (initialWork.demandUnused != 1 || initialWork.demandVisited != roots.size() + 2 ||
            initialWork.demandRefreshed + initialWork.demandUnused != initialWork.demandVisited) {
            return RhiTestResult::fail("Complete feedback counters do not match resident work");
        }
        residency.beginFrame();
        if (residency.stats().cpuWork.demandVisited != 0) {
            return RhiTestResult::fail("CPU work counters did not reset at beginFrame");
        }
        // An empty complete batch means the previously unused page is needed
        // again. Returning to it must neither reload nor leave it evictable.
        (void)residency.consumeGpuRequests(StreamGpuRequestBatch{.residentDemandFeedback = true});
        if (residency.pageAge(pages[0]) != 0 || !residency.requestPage(pages[0]) ||
            residency.stats().totalScheduledUploadCount != uploads || residency.queuedUploadCount() != 0) {
            return RhiTestResult::fail("Returning demand did not reuse its cached payload");
        }
        residency.beginFrame();
        (void)residency.requestPage(pages[2]);
        if (residency.stats().frameEvictedPageCount != 0 || residency.pageAllocated(pages[2])) {
            return RhiTestResult::fail("Budget pressure evicted demanded geometry");
        }
        // Refresh in the SAME frame after exhausting admission: new explicit
        // cold feedback must reopen the gate without waiting for beginFrame.
        // Even when feedback is truncated, only the explicitly cold page may
        // be reclaimed; omission cannot make the returning page a victim.
        const std::array<uint32_t, 1> secondUnused{pages[1]};
        (void)residency.consumeGpuRequests({.unloadPageIds = secondUnused, .unloadRequestCounter = 2,
            .unloadOverflowCounter = 1, .residentDemandFeedback = true});
        const auto incompleteWork = residency.stats().cpuWork;
        if (incompleteWork.demandUnused != 1 || incompleteWork.demandRefreshed != 0 ||
            incompleteWork.demandIncompleteProtected + 1 != incompleteWork.demandVisited) {
            return RhiTestResult::fail("Truncated feedback counters lost protected pages");
        }
        (void)residency.requestPage(pages[2]);
        if (!residency.pageResident(pages[0]) || residency.pageState(pages[1]) != MeshletStreamPageResidencyState::PendingUnload ||
            residency.stats().frameEvictedPageCount != 1 || residency.pageAllocated(pages[2])) {
            return RhiTestResult::fail("Budget victim ignored demand feedback or delayed release");
        }
        residency.beginFrame();
        (void)residency.requestPage(pages[2]);
        if (!residency.pageAllocated(pages[2]) || residency.pageAllocated(pages[1])) {
            return RhiTestResult::fail("Cached victim was not recycled after completion");
        }
        for (uint32_t page : roots) {
            if (!residency.pageResident(page) || residency.unloadPage(page)) { return RhiTestResult::fail("Demand cache lost a root"); }
        }
        if (!residency.unloadPage(pages[0])) { return RhiTestResult::fail("Explicit unload no longer works"); }
        // Exercise duplicate bits, every asset word (including the final partial
        // word), invalid IDs, and clearing/reusing all touched words.
        std::vector<uint32_t> allUnused;
        for (uint32_t id = 0; id < asset.pageCount(); ++id) {
            allUnused.push_back(id); allUnused.push_back(id);
        }
        allUnused.push_back(asset.pageCount()); allUnused.push_back(UINT32_MAX);
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            residency.beginFrame();
            (void)residency.consumeGpuRequests({.unloadPageIds = allUnused,
                .unloadRequestCounter = static_cast<uint32_t>(allUnused.size()), .residentDemandFeedback = true});
            const auto stats = residency.stats();
            if (stats.frameUniqueGpuUnloadRequestCount != asset.pageCount() || stats.frameGpuInvalidRequestCount != 2) {
                return RhiTestResult::fail("Unused page marks lost duplicates, bounds checks or touched-word clearing");
            }
        }
        return RhiTestResult::pass("Cache reuse, empty/truncated demand feedback, hot-page protection and delayed budget eviction");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletDemandCacheTest);

class StreamerJointColdReclaimTest final : public RhiTest {
public:
    StreamerJointColdReclaimTest() { type = RhiTestType::Command; name = "streamer_joint_cold_reclaim"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "joint_cold.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 3) { return RhiTestResult::skip("Requires three streamable pages"); }
        pages.resize(3);
        MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize({.asset = &asset,
                .maxResidentBytes = 4 * (pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages)),
                .maxResidentPages = static_cast<uint32_t>(roots.size() + 3), .queuedFrameCount = 1,
                .unloadDelayFrames = 1, .evictionAgeThresholdFrames = 1}, reason) ||
            !residency.lockFallbackPages(roots, reason)) { return RhiTestResult::fail(reason); }
        std::unique_ptr<Streamer> streamer;
        auto result = context.device.createStreamer(makeTestStreamerDesc((roots.size() + 2) * asset.maxPagePayloadBytes() + 4096), streamer);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        std::unique_ptr<Buffer> destination;
        result = context.device.createBuffer({.size = residency.pageBufferSize(),
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}, destination);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        residency.beginFrame();
        (void)residency.requestPage(pages[0]); (void)residency.requestPage(pages[1]);
        const auto uploads = static_cast<uint32_t>(roots.size() + 2);
        if (residency.processUploads(*streamer, *destination, uploads) != uploads) {
            return RhiTestResult::fail("Cannot prepare demand-cache residents");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
        if (!residency.pageResident(pages[0]) || !residency.pageResident(pages[1])) {
            return RhiTestResult::fail("Demand-cache setup is not resident");
        }

        const std::array<uint32_t, 1> unused{pages[1]};
        MeshletStreamColdPageReclaimDesc reclaim{.clasUsedBytes = 1024, .clasCapacityBytes = 1024,
            .retentionFrames = 120, .pressureAgeFrames = 16,
            .clasPageBytes = [&](uint32_t page) -> uint64_t { return page == pages[0] || page == pages[1] ? 512 : 0; }};
        (void)residency.consumeGpuRequests({.unloadPageIds = unused, .unloadRequestCounter = 1, .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 0) { return RhiTestResult::fail("CLAS pressure evicted a recent page"); }
        for (uint32_t frame = 0; frame < 16; ++frame) { residency.beginFrame(); }
        (void)residency.consumeGpuRequests({.unloadPageIds = unused, .unloadRequestCounter = 1, .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 1 || residency.pageState(pages[1]) != MeshletStreamPageResidencyState::PendingUnload ||
            !residency.pageResident(pages[0]) || residency.stats().usedResidentBytes >= residency.maxResidentBytes() * 70 / 100) {
            return RhiTestResult::fail("CLAS-only pressure did not schedule the shared cold geometry page");
        }
        if (residency.reclaimColdPages(reclaim) != 0 || residency.stats().frameEvictionScanCount != 1) {
            return RhiTestResult::fail("Pending joint frees caused duplicate victims or scans");
        }
        residency.beginFrame();
        if (residency.pageAllocated(pages[1])) { return RhiTestResult::fail("Joint victim geometry was not freed"); }
        reclaim.clasUsedBytes = 512;
        reclaim.clasRetiringBytes = 512;
        for (uint32_t frame = 0; frame < 121; ++frame) { residency.beginFrame(); }
        // Complete feedback still protects the current view even after a long pause.
        (void)residency.consumeGpuRequests(StreamGpuRequestBatch{.residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 0) { return RhiTestResult::fail("Current view was treated as cold"); }
        const std::array<uint32_t, 1> nowUnused{pages[0]};
        for (uint32_t frame = 0; frame < 121; ++frame) { residency.beginFrame(); }
        (void)residency.consumeGpuRequests({.unloadPageIds = nowUnused, .unloadRequestCounter = 1, .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 1) { return RhiTestResult::fail("Old cold page was retained below both budgets"); }
        for (uint32_t root : roots) {
            if (!residency.pageResident(root)) { return RhiTestResult::fail("Joint reclaim lost a fallback page"); }
        }
        // Reinsert erased entries in reverse order. Age wins over page ID,
        // and equal-age pages keep ID order despite resident-table swaps.
        std::sort(pages.begin(), pages.end());
        reclaim.clasUsedBytes = reclaim.clasRetiringBytes = reclaim.clasCapacityBytes = 0;
        reclaim.retentionFrames = reclaim.pressureAgeFrames = reclaim.maxPages = 1;
        for (uint32_t cycle = 0; cycle < 2; ++cycle) {
            residency.beginFrame();
            for (auto it = pages.rbegin(); it != pages.rend(); ++it) {
                (void)residency.requestPage(*it);
            }
            if (residency.processUploads(*streamer, *destination, 3) != 3) {
                return RhiTestResult::fail("Cannot upload cold-sort fixture");
            }
            for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
            (void)residency.requestPage(pages[0]);
            residency.beginFrame();
            (void)residency.consumeGpuRequests({.unloadPageIds = pages, .unloadRequestCounter = 3,
                                               .residentDemandFeedback = true});
            for (uint32_t expected : {pages[1], pages[2], pages[0]}) {
                // Start with a nonempty sorted prefix, then admit its younger
                // suffix on the next call without disturbing age/ID order.
                reclaim.retentionFrames = cycle == 1 && expected == pages[1] ? 4 : 1;
                if (residency.reclaimColdPages(reclaim) != 1 ||
                    residency.pageState(expected) != MeshletStreamPageResidencyState::PendingUnload) {
                    return RhiTestResult::fail("Cold eviction changed age/ID order after erase and reinsertion");
                }
            }
            if (residency.reclaimColdPages(reclaim) != 0 || residency.stats().frameEvictionScanCount != 1) {
                return RhiTestResult::fail("Cached cold candidates scheduled a duplicate victim");
            }
        }
        residency.beginFrame();
        for (uint32_t id : pages) { (void)residency.requestPage(id); }
        if (residency.processUploads(*streamer, *destination, 3) != 3) {
            return RhiTestResult::fail("Cannot upload partial-sort fixture");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
        (void)residency.consumeGpuRequests({.unloadPageIds = pages, .unloadRequestCounter = 3,
                                           .residentDemandFeedback = true});
        uint32_t clasQueries = 0;
        reclaim.retentionFrames = 120;
        reclaim.clasPageBytes = [&](uint32_t) -> uint64_t { ++clasQueries; return 512; };
        if (residency.reclaimColdPages(reclaim) != 0 || clasQueries != 0 || residency.stats().cpuWork.coldVisited != 0) {
            return RhiTestResult::fail("Unexpired retention suffix was visited or queried for CLAS sizes");
        }
        // Expand the same cached candidates when CLAS pressure appears later in
        // the frame. Once one victim is credited, the remaining young suffix
        // must be skipped even though the requested per-call limit is larger.
        reclaim.clasUsedBytes = reclaim.clasCapacityBytes = 1024;
        reclaim.maxPages = 3;
        if (residency.reclaimColdPages(reclaim) != 1 || clasQueries != 1 ||
            residency.pageState(pages[0]) != MeshletStreamPageResidencyState::PendingUnload ||
            residency.stats().frameEvictionScanCount != 1) {
            return RhiTestResult::fail("Partial cold sort missed new pressure, changed ID order or over-evicted after pressure ended");
        }
        // Pending-free credit ends pressure on another call in the same frame.
        if (residency.reclaimColdPages(reclaim) != 0 || clasQueries != 2) {
            return RhiTestResult::fail("Partial cold sort ignored pending free credit");
        }
        return RhiTestResult::pass("CLAS pressure, delayed credit, partial-sort expansion, due cutoff and age/ID order across reloads");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerJointColdReclaimTest);

class MeshletStreamFragmentedStorageTest final : public RhiTest {
public:
    MeshletStreamFragmentedStorageTest() { type = RhiTestType::Validation; name = "streamer_meshlet_fragmented_storage"; }
    RhiTestResult run(RhiTestContext&) override
    {
        render::MeshletStreamStorage storage;
        std::string reason;
        if (!storage.initialize(4096u * 256u, 256, reason)) { return RhiTestResult::fail(reason); }
        std::vector<render::MeshletStreamStorageAllocation> pages;
        for (uint32_t i = 0; i < 4096; ++i) {
            pages.push_back(storage.allocate(256));
            if (!pages.back().valid()) { return RhiTestResult::fail("Cannot fill fragmented storage"); }
        }
        for (uint32_t i = 0; i < pages.size(); i += 2) { storage.release(pages[i]); }
        for (uint32_t retry = 0; retry < 10000; ++retry) {
            if (storage.canAllocate(257) || storage.allocate(257).valid() || storage.largestFreeBlockBytes() != 256) {
                return RhiTestResult::fail("Fragmented free bytes were mistaken for a contiguous allocation");
            }
        }
        storage.release(pages[1]);
        if (!storage.canAllocate(768) || storage.largestFreeBlockBytes() != 768) {
            return RhiTestResult::fail("Coalescing did not invalidate the free-block bound");
        }
        const auto merged = storage.allocate(768);
        if (!merged.valid() || merged.offset != 0 || storage.canAllocate(512) ||
            storage.canAllocate(UINT64_MAX) || storage.allocate(UINT64_MAX).valid()) {
            return RhiTestResult::fail("Allocation left a stale bound or accepted overflowing alignment");
        }
        storage.release(merged);
        if (!storage.canAllocate(768)) { return RhiTestResult::fail("Released range did not become allocatable"); }
        if (!storage.initialize(512, 256, reason) || storage.largestFreeBlockBytes() != 512) {
            return RhiTestResult::fail("Storage reset retained old free-block bounds");
        }
        return RhiTestResult::pass("Fragmentation, repeated misses, coalescing, allocation, overflow and reset");
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletStreamFragmentedStorageTest);

class StreamerMeshletUploadCompletionTest final : public RhiTest {
public:
    StreamerMeshletUploadCompletionTest() { type = RhiTestType::Command; name = "streamer_meshlet_upload_completion"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "completion.meshstream.bin", asset);
        if (!built.passed) { return built; }
        if (asset.pageCount() < 3) { return RhiTestResult::fail("Need three pages for completion batching"); }
        MeshletStreamResidencyManager residency;
        std::unique_ptr<Streamer> streamer;
        std::unique_ptr<Buffer> destination;
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands, prefix;
        std::unique_ptr<Semaphore> gate;
        RenderFrameContext frame;
        QueueSubmissionTracker tracker;
        // Drain before destroying any resources, including on assertion failure.
        struct Drain {
            Queue& queue;
            std::unique_ptr<Semaphore>& gate;
            RenderFrameContext& frame;
            ~Drain()
            {
                if (gate && gate->currentValue() < 1) { (void)gate->signal(1); }
                frame.cancel();
                (void)queue.waitIdle();
            }
        } drain{context.graphicsQueue, gate, frame};
#define UPLOAD_REQUIRE(expression) \
        if (!(expression)) { return RhiTestResult::fail("Upload completion: " #expression); }
        UPLOAD_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        UPLOAD_REQUIRE(context.device.createStreamer(makeTestStreamerDesc(), streamer));
        UPLOAD_REQUIRE(context.device.createCommandPool(context.graphicsQueue, pool));
        UPLOAD_REQUIRE(pool->createCommandBuffer(commands));
        UPLOAD_REQUIRE(pool->createCommandBuffer(prefix));
        UPLOAD_REQUIRE(context.device.createSemaphore(gate));
        const uint64_t capacity = alignStreamStorageBytes(asset.maxPagePayloadBytes()) * 4u;
        UPLOAD_REQUIRE(context.device.createBuffer({.size = capacity,
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}, destination));
        uint64_t frameIndex = 0;
        // Gate completion, drop an unflushed upload, cancel its recording, then
        // cancel only the copy tail of a batch whose prefix has been accepted.
        for (uint32_t scenario = 0; scenario < 6; ++scenario) {
            std::string reason;
            UPLOAD_REQUIRE(residency.initialize({.asset = &asset, .maxResidentBytes = capacity,
                .maxResidentPages = 4, .queuedFrameCount = 8}, reason));
            const uint32_t root = 0;
            if (scenario == 2) { (void)residency.requestPage(root); }
            else { UPLOAD_REQUIRE(residency.lockFallbackPages(std::span(&root, 1), reason)); }
            const uint32_t pageCount = scenario == 4 ? 3u : 1u;
            for (uint32_t page = 1; page < pageCount; ++page) { (void)residency.requestPage(page); }
            const bool cancel = (scenario >= 1 && scenario <= 3) || scenario == 5;
            for (uint32_t attempt = 0; attempt < (cancel ? 2u : 1u); ++attempt) {
                UPLOAD_REQUIRE(frame.begin(++frameIndex));
                UPLOAD_REQUIRE(pool->reset());
                UPLOAD_REQUIRE(commands->begin(scenario == 5 && attempt == 0 ? nullptr : &frame));
                UPLOAD_REQUIRE(streamer->beginFrame(frame));
                residency.beginFrame();
                for (uint32_t page = 0; page < pageCount; ++page) {
                    UPLOAD_REQUIRE(residency.processUploads(*streamer, *destination, 1) == 1);
                }
                const auto receipt = streamer->pendingCopyCompletion();
                UPLOAD_REQUIRE(receipt && !receipt->isComplete() && !receipt->isCancelled());
                std::vector<StreamPageTablePatch> orderedPatches;
                UPLOAD_REQUIRE(!receipt->isRecordedBefore(*commands));
                UPLOAD_REQUIRE(residency.buildOrderedUploadPatches(*commands, orderedPatches) == 0);
                UPLOAD_REQUIRE(residency.residentPageCount() == 0);
                if (scenario != 1 || attempt != 0) {
                    const BufferBarrierDesc barrier{.buffer = destination.get(),
                        .before = ResourceState::Undefined, .after = ResourceState::TransferDestination,
                        .size = capacity};
                    commands->barrier({.buffers = &barrier, .bufferCount = 1});
                    commands->copyStreamedData(*streamer);
                    const bool sameRecording = !(scenario == 5 && attempt == 0);
                    UPLOAD_REQUIRE(receipt->isRecordedBefore(*commands) == sameRecording);
                    UPLOAD_REQUIRE(residency.buildOrderedUploadPatches(*commands, orderedPatches) ==
                        (sameRecording ? pageCount : 0u));
                    if (sameRecording) {
                        for (uint32_t page = 0; page < pageCount; ++page) {
                            const auto matching = std::count_if(orderedPatches.begin(), orderedPatches.end(),
                                [page](const auto& patch) { return patch.pageId == page; });
                            UPLOAD_REQUIRE(matching == 1);
                            const auto patch = std::find_if(orderedPatches.begin(), orderedPatches.end(),
                                [page](const auto& entry) { return entry.pageId == page; });
                            UPLOAD_REQUIRE(streamPageTablePatchState(*patch) == (page == root && scenario != 2
                                ? MeshletStreamPageResidencyState::LockedFallback : MeshletStreamPageResidencyState::Resident));
                        }
                        UPLOAD_REQUIRE(prefix->begin(&frame));
                        UPLOAD_REQUIRE(!receipt->isRecordedBefore(*prefix));
                        UPLOAD_REQUIRE(residency.buildOrderedUploadPatches(*prefix, orderedPatches) == 0);
                        UPLOAD_REQUIRE(prefix->end());
                    }
                    UPLOAD_REQUIRE(residency.residentPageCount() == 0);
                }
                UPLOAD_REQUIRE(commands->end());
                UPLOAD_REQUIRE(!receipt->isRecordedBefore(*commands));
                streamer->endFrame();
                CommandBuffer* buffers[] = {commands.get()};
                if (cancel && attempt == 0) {
                    if (scenario == 3) {
                        UPLOAD_REQUIRE(prefix->begin(&frame));
                        UPLOAD_REQUIRE(prefix->end());
                        CommandBuffer* prefixBuffers[] = {prefix.get()};
                        GpuCompletionPoint prefixCompletion;
                        UPLOAD_REQUIRE(tracker.submitSegment({.commandBuffers = prefixBuffers,
                            .commandBufferCount = 1}, frame, prefixCompletion));
                    }
                    frame.cancel();
                    if (scenario == 3) {
                        UPLOAD_REQUIRE(frame.wait(5'000'000'000ull));
                        UPLOAD_REQUIRE(frame.completion().isSubmitted() && frame.completion().isComplete());
                    }
                    UPLOAD_REQUIRE(receipt->isCancelled() && !receipt->isComplete());
                    residency.beginFrame();
                    UPLOAD_REQUIRE(residency.pageState(root) == MeshletStreamPageResidencyState::Unloaded);
                    UPLOAD_REQUIRE(residency.queuedUploadCount() == 1 && residency.stats().totalCancelledUploads == 1);
                    UPLOAD_REQUIRE(residency.newlyResidentPages().empty());
                    continue;
                }
                const SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = 1};
                UPLOAD_REQUIRE(tracker.submit({.waitSemaphores = scenario == 0 ? &wait : nullptr,
                    .waitSemaphoreCount = scenario == 0 ? 1u : 0u,
                    .commandBuffers = buffers, .commandBufferCount = 1}, frame));
                if (scenario == 0) {
                    // CPU frame age, recording and queue acceptance prove none
                    // of the GPU copy's completion. No blocking wait is allowed.
                    for (uint32_t cpuFrame = 0; cpuFrame < 16; ++cpuFrame) {
                        residency.beginFrame();
                        UPLOAD_REQUIRE(!receipt->isComplete() && residency.residentPageCount() == 0);
                    }
                    UPLOAD_REQUIRE(gate->signal(1));
                }
                UPLOAD_REQUIRE(frame.wait(5'000'000'000ull));
                UPLOAD_REQUIRE(receipt->isComplete() && !receipt->isCancelled());
                residency.beginFrame();
                UPLOAD_REQUIRE(residency.residentPageCount() == pageCount);
                UPLOAD_REQUIRE(residency.newlyResidentPages().size() == pageCount);
                UPLOAD_REQUIRE(residency.pageState(root) == (scenario == 2
                    ? MeshletStreamPageResidencyState::Resident : MeshletStreamPageResidencyState::LockedFallback));
                UPLOAD_REQUIRE(residency.stats().availableStorageTaskCount == kStreamingMaxActiveTasks);
                UPLOAD_REQUIRE(residency.stats().availableUpdateTaskCount == kStreamingMaxActiveTasks);
                UPLOAD_REQUIRE(residency.stats().queuedUpdateTaskCount == 0);
                std::vector<uint8_t> bytes(static_cast<size_t>(capacity));
                UPLOAD_REQUIRE(readBufferBytes(*destination, bytes.data(), capacity));
                for (uint32_t page = 0; page < pageCount; ++page) {
                    const auto payload = asset.pagePayload(page);
                    UPLOAD_REQUIRE(std::memcmp(bytes.data() + residency.deviceOffsetForPage(page),
                        payload.data(), payload.size()) == 0);
                }
            }
        }
        // An owner can reset while the independent receipt is still pending.
        std::string reason;
        UPLOAD_REQUIRE(residency.initialize({.asset = &asset, .maxResidentBytes = capacity}, reason));
        UPLOAD_REQUIRE(frame.begin(++frameIndex));
        UPLOAD_REQUIRE(streamer->beginFrame(frame));
        (void)residency.requestPage(0);
        UPLOAD_REQUIRE(residency.processUploads(*streamer, *destination, 1) == 1);
        const auto abandoned = streamer->pendingCopyCompletion();
        residency.reset();
        streamer.reset();
        UPLOAD_REQUIRE(abandoned->isCancelled() && !abandoned->isComplete());
        frame.cancel();
#undef UPLOAD_REQUIRE
        return RhiTestResult::pass("GPU gate, unflushed/cancelled/partial/mismatched submissions, demand/root retries, batch drain and reset");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletUploadCompletionTest);

class StreamerOrderedPublicationTest final : public RhiTest {
public:
    StreamerOrderedPublicationTest() { type = RhiTestType::Command; name = "streamer_ordered_publication_retry"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        std::atomic_uint validationMessages = 0;
        std::unique_ptr<Device> ownedDevice;
        const auto created = createDevice({.applicationName = "Stream ordered publication",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationMessages}}, ownedDevice);
        if (!created) {
            return hasError(created, Error::Unsupported) ? RhiTestResult::skip("Bindless device unavailable") :
                RhiTestResult::fail("Cannot create ordered publication device");
        }
        auto& device = *ownedDevice;
        auto& queue = *device.getQueue(QueueType::Graphics);
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "ordered.meshstream.bin", asset);
        if (!built.passed) { return built; }
        MeshletStreamRuntime runtime;
        std::unique_ptr<Streamer> streamer;
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Buffer> readback;
        QueueSubmissionTracker tracker;
        RenderFrameContext frame0(0), frame1(1);
        RenderFrameContext* frames[]{&frame0, &frame1};
        struct Drain {
            Queue& queue; RenderFrameContext& a; RenderFrameContext& b;
            ~Drain() { a.cancel(); b.cancel(); (void)queue.waitIdle(); }
        } drain{queue, frame0, frame1};
#define ORDERED_REQUIRE(expression) \
        if (!(expression)) { return RhiTestResult::fail("Ordered publication: " #expression); }
        std::string log;
        runtime.setDebugReadbackEnabled(true);
        ORDERED_REQUIRE(runtime.initialize(device, {
            .sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf",
            .streamAssetPath = asset.path(), .maxResidentPages = 64, .maxLockedFallbackPages = 64,
            .maxPageUploadsPerFrame = 64, .maxGpuPageRequests = 256, .maxGpuPageUnloadRequests = 256,
            .maxActiveGroups = 4096, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
            .pageLoadConcurrency = 0, .queuedFrameCount = 2}, log));
        ORDERED_REQUIRE(!runtime.sceneReadiness().ready);
        ORDERED_REQUIRE(device.createStreamer(makeTestStreamerDesc(), streamer));
        ORDERED_REQUIRE(device.createCommandPool(queue, pool));
        ORDERED_REQUIRE(pool->createCommandBuffer(commands));
        ORDERED_REQUIRE(tracker.initialize(device, queue));
        ORDERED_REQUIRE(device.createBuffer({.size = sizeof(MeshletStreamGpuActiveHeader),
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}, readback));
        MeshletStreamFrameDesc view{.width = 192, .height = 128, .selectedLodLevel = 0, .enableGpuLodSelection = false};
        view.camera = {.eye = {-.0168404f, .110154f, .22f}, .center = {-.0168404f, .110154f, -.00153695f},
            .znear = .001f, .zfar = 10.f};
        for (uint32_t attempt = 0; attempt < 3; ++attempt) {
            auto& frame = *frames[attempt % 2];
            ORDERED_REQUIRE(frame.begin(attempt + 1));
            ORDERED_REQUIRE(pool->reset());
            ORDERED_REQUIRE(commands->begin(&frame));
            ORDERED_REQUIRE(streamer->beginFrame(frame));
            ORDERED_REQUIRE(runtime.cmdBeginFrame(*commands, *streamer, view));
            ORDERED_REQUIRE(runtime.cmdPreTraversal(*commands, view));
            if (attempt < 2) {
                ORDERED_REQUIRE(runtime.residency().residentPageCount() == 0);
                ORDERED_REQUIRE(runtime.debugSnapshot(false).at("orderedUploadPages").get<uint32_t>() != 0);
            } else { ORDERED_REQUIRE(runtime.sceneReadiness().ready); }
            std::vector<DebugResourceBinding> bindings;
            runtime.appendDebugBindings(bindings, "proof.");
            const auto header = std::find_if(bindings.begin(), bindings.end(), [](const auto& b) { return b.id == "proof.activeHeader"; });
            ORDERED_REQUIRE(header != bindings.end());
            BufferBarrierDesc barrier{.buffer = header->buffer, .before = header->state, .after = ResourceState::TransferSource,
                .size = sizeof(MeshletStreamGpuActiveHeader)};
            commands->barrier({.buffers = &barrier, .bufferCount = 1});
            commands->copyBuffer({.source = header->buffer, .destination = readback.get(), .size = sizeof(MeshletStreamGpuActiveHeader)});
            std::swap(barrier.before, barrier.after);
            commands->barrier({.buffers = &barrier, .bufferCount = 1});
            ORDERED_REQUIRE(runtime.cmdEndFrame(*commands));
            ORDERED_REQUIRE(commands->end());
            streamer->endFrame();
            if (attempt == 0) { frame.cancel(); continue; }
            CommandBuffer* buffers[]{commands.get()};
            ORDERED_REQUIRE(tracker.submit({.commandBuffers = buffers, .commandBufferCount = 1}, frame));
            ORDERED_REQUIRE(frame.wait(5'000'000'000ull));
            MeshletStreamGpuActiveHeader result;
            ORDERED_REQUIRE(readBufferBytes(*readback, &result, sizeof(result)));
            ORDERED_REQUIRE(result.activeGroupCount != 0 && result.overflowCount == 0);
        }
        ORDERED_REQUIRE(runtime.sceneReady());
        const auto stableScans = runtime.debugSnapshot(false).at("sceneReadinessScans");
        for (uint32_t query = 0; query < 1000; ++query) { ORDERED_REQUIRE(runtime.sceneReady()); }
        ORDERED_REQUIRE(runtime.debugSnapshot(false).at("sceneReadinessScans") == stableScans);
        runtime.reset();
        ORDERED_REQUIRE(!runtime.sceneReady() && runtime.sceneReadiness().requiredPages == 0);
        ORDERED_REQUIRE(runtime.initialize(device, {
            .sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf",
            .streamAssetPath = asset.path(), .maxResidentPages = 64, .maxLockedFallbackPages = 64,
            .maxPageUploadsPerFrame = 64, .maxGpuPageRequests = 256, .maxGpuPageUnloadRequests = 256,
            .maxActiveGroups = 4096, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
            .pageLoadConcurrency = 0, .queuedFrameCount = 2}, log));
        ORDERED_REQUIRE(!runtime.sceneReady() && runtime.sceneReadiness().requiredPages != 0);
        ORDERED_REQUIRE(runtime.sceneReadiness().completedPages == 0);
        ORDERED_REQUIRE(validationMessages == 0);
#undef ORDERED_REQUIRE
        return RhiTestResult::pass("Cancelled initial publication retries; GPU selects uploaded roots before CPU confirmation across frame slots");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerOrderedPublicationTest);

METALLIC_REGISTER_RHI_TEST(StreamingTaskQueueLifecycleTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamPageLoaderTaskGraphTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamPageLoadConfigurationCompatibilityTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamStorageAddressLimitTest);
METALLIC_REGISTER_RHI_TEST(StreamerBufferUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerTextureUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerConstantUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerRenderGraphFlushTest);
METALLIC_REGISTER_RHI_TEST(StreamerRenderGraphInvalidDoesNotBeginFrameTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyUploadTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamClasPagePlanTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyGpuRequestPatchTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyLatestGpuRequestTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyGpuRequestUnloadOverflowTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyEvictionDelayAgeTest);


class StreamerUploadByteBudgetTest final : public RhiTest {
public:
    StreamerUploadByteBudgetTest() { type = RhiTestType::Validation; name = "streamer_meshlet_upload_byte_budget"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "ByteBudget.meshstream.bin", asset);
        if (!built.passed) { return built; }
        if (asset.pageCount() < 4) { return RhiTestResult::fail("Need four pages for byte budget test"); }
        const uint64_t capacity = uint64_t(asset.maxPagePayloadBytes()) * 8;
        std::unique_ptr<Buffer> destination;
        auto result = context.device.createBuffer({.size = capacity,
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}, destination);
        if (!result) { return RhiTestResult::fail(toString(result)); }
        for (bool asynchronous : {false, true}) {
            for (uint64_t budget : {0ull, 1ull, uint64_t(asset.maxPagePayloadBytes())}) {
                MeshletStreamResidencyManager residency;
                std::string reason;
                if (!residency.initialize({.asset = &asset, .maxResidentBytes = capacity, .queuedFrameCount = 1,
                        .pageLoadConcurrency = asynchronous ? 2u : 0u, .maxPageLoadsInFlight = 4,
                        .completionDrivenUploads = false}, reason)) { return RhiTestResult::fail(reason); }
                std::unique_ptr<Streamer> streamer;
                result = context.device.createStreamer(makeTestStreamerDesc(capacity + 4096), streamer);
                if (!result) { return RhiTestResult::fail(toString(result)); }
                residency.beginFrame();
                for (uint32_t page = 0; page < 4; ++page) { (void)residency.requestPage(page); }
                if (asynchronous) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (residency.stats().preparedPageLoadCount < 4 && std::chrono::steady_clock::now() < deadline) {
                        (void)residency.processUploads(*streamer, *destination, 0);
                        std::this_thread::yield();
                    }
                    if (residency.stats().preparedPageLoadCount != 4) { return RhiTestResult::fail("Async pages not prepared"); }
                }
                std::vector<uint32_t> seen;
                uint64_t observedBytes = 0;
                const auto observer = [&](uint32_t page, std::span<const uint8_t> bytes) {
                    seen.push_back(page); observedBytes += bytes.size();
                };
                for (uint32_t frame = 0; frame < 5 && seen.size() < 4; ++frame) {
                    if (frame != 0) { residency.beginFrame(); }
                    const size_t before = seen.size();
                    (void)residency.processUploads(*streamer, *destination, 4, observer, nullptr, budget);
                    const auto firstBytes = residency.stats().frameUploadBytes;
                    const size_t after = seen.size();
                    // Retrying must share this frame's byte credit.
                    (void)residency.processUploads(*streamer, *destination, 4, observer, nullptr, budget);
                    const auto spent = residency.stats().frameUploadBytes;
                    if (budget != 0 && spent > budget && (seen.size() - before != 1 || spent != firstBytes || seen.size() != after)) {
                        return RhiTestResult::fail("Byte credit was bypassed by a repeat call or oversized page");
                    }
                    if (seen.size() == before || (budget == 1 && seen.size() - before != 1)) {
                        return RhiTestResult::fail("Oversized page starved or multiple pages escaped the budget");
                    }
                }
                std::sort(seen.begin(), seen.end());
                uint64_t expectedBytes = 0;
                for (uint32_t page = 0; page < 4; ++page) { expectedBytes += scene::meshletStreamDevicePayloadSize(asset.pages()[page]); }
                if (seen != std::vector<uint32_t>{0, 1, 2, 3} || observedBytes != expectedBytes ||
                    residency.stats().totalUploadBytes != expectedBytes) {
                    return RhiTestResult::fail("Budget deferral lost/duplicated a page or counted disk bytes");
                }
            }
        }
        return RhiTestResult::pass("Sync/async uploads: shared frame credit, unlimited mode, oversized progress, exact device bytes and deferred retry");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerUploadByteBudgetTest);

} // namespace
} // namespace metallic::tests

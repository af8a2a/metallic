#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include <stdexcept>
#include <string>

#include "RHITest.h"

#include "Runtime/Render/Streamer/MeshletStreamCLAS.h"
#include "Runtime/Render/Streamer/MeshletStreamInitialLoader.h"
#include "Runtime/Render/Streamer/MeshletStreamPageLoader.h"
#include "Runtime/Render/Streamer/MeshletStreamResidency.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Streamer/StreamUploadCompletion.h"
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
#include <cmath>
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
#include <unordered_map>
#include <vector>

namespace metallic::tests {
namespace {

RHITestResult createCommandResources(
    render::Device& device,
    render::Queue& queue,
    std::unique_ptr<render::CommandPool>& outCommandPool,
    std::unique_ptr<render::CommandBuffer>& outCommandBuffer,
    std::unique_ptr<render::Fence>& outFence)
{
    render::Result<> result = device.createCommandPool(queue).transform([&](auto rhiValue) { outCommandPool = std::move(rhiValue); });
    if (!result || outCommandPool == nullptr) {
        return RHITestResult::fail(std::string("createCommandPool returned ") + toString(result));
    }

    result = outCommandPool->createCommandBuffer().transform([&](auto rhiValue) { outCommandBuffer = std::move(rhiValue); });
    if (!result || outCommandBuffer == nullptr) {
        return RHITestResult::fail(std::string("createCommandBuffer returned ") + toString(result));
    }

    result = device.createFence(false).transform([&](auto rhiValue) { outFence = std::move(rhiValue); });
    if (!result || outFence == nullptr) {
        return RHITestResult::fail(std::string("createFence returned ") + toString(result));
    }

    return RHITestResult::pass();
}

RHITestResult submitAndWait(
    render::Queue& queue,
    render::CommandBuffer& commandBuffer,
    render::Fence& fence)
{
    render::CommandBuffer* commandBuffers[] = {&commandBuffer};
    render::Result<> result = queue.submit(render::QueueSubmitDesc{
        .commandBuffers = {commandBuffers, 1},
        .signalFence = &fence,
    });
    if (!result) {
        return RHITestResult::fail(std::string("Queue::submit returned ") + toString(result));
    }

    result = fence.wait(5'000'000'000ull);
    if (!result) {
        return RHITestResult::fail(std::string("Fence::wait returned ") + toString(result));
    }
    return RHITestResult::pass();
}

bool readBufferBytes(render::Buffer& buffer, void* outData, uint64_t byteSize)
{
    buffer.invalidate({0, byteSize});
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

RHITestResult buildBunnyStreamAssetForTest(
    const std::filesystem::path& outputPath,
    scene::MeshletStreamAsset& outAsset,
    scene::MeshletStreamPayloadCompression compressionMode = scene::MeshletStreamPayloadCompression::None)
{
    const std::filesystem::path sourcePath =
        std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";

    scene::Scene scene;
    if (!scene.load(sourcePath)) {
        return RHITestResult::fail("Scene::load failed: " + scene.lastLoadResult().error);
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
        return RHITestResult::fail("buildMeshletStreamAsset failed: " + reason);
    }

    if (!outAsset.open(outputPath, reason)) {
        return RHITestResult::fail("MeshletStreamAsset::open failed: " + reason);
    }
    if (outAsset.pageCount() == 0) {
        return RHITestResult::fail("streamasset has no pages");
    }
    return RHITestResult::pass();
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

uint64_t pageDeviceStorageBytes(const scene::MeshletStreamAsset& asset, uint32_t pageIndex)
{
    return alignStreamStorageBytes(scene::meshletStreamDevicePayloadSize(asset.pages()[pageIndex]));
}

uint64_t pageDeviceStorageBytes(const scene::MeshletStreamAsset& asset, std::span<const uint32_t> pageIndices)
{
    uint64_t total = 0;
    for (uint32_t id : pageIndices) { total += pageDeviceStorageBytes(asset, id); }
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

class MeshletStreamPageLoaderTaskGraphTest : public RHITest {
public:
    MeshletStreamPageLoaderTaskGraphTest()
    {
        type = RHITestType::Validation;
        name = "meshlet_stream_page_loader_task_graph";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const std::filesystem::path streamAssetPath =
            context.outputDirectory / "task_graph_page_loader.meshstream.bin";
        scene::MeshletStreamAsset asset;
        RHITestResult build = buildBunnyStreamAssetForTest(
            streamAssetPath,
            asset,
            scene::MeshletStreamPayloadCompression::ByteRle);
        if (!build.passed) {
            return build;
        }

        render::MeshletStreamPageLoader loader;
        std::string reason;
        if (!loader.initialize(asset, 2, reason)) {
            return RHITestResult::fail("MeshletStreamPageLoader::initialize failed: " + reason);
        }

        auto sink = std::make_shared<PageLoaderEventSink>();
        const task::TaskEventSinkToken sinkToken = task::taskSystem().subscribe(sink);
        auto fail = [&](std::string message) {
            loader.reset();
            (void)task::taskSystem().unsubscribe(sinkToken);
            return RHITestResult::fail(std::move(message));
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
            return RHITestResult::fail("MeshletStreamPageLoader::reset did not drain and clear the loader");
        }
        return RHITestResult::pass();
    }
};

class MeshletStreamPageLoadConfigurationCompatibilityTest : public RHITest {
public:
    MeshletStreamPageLoadConfigurationCompatibilityTest()
    {
        type = RHITestType::Validation;
        name = "meshlet_stream_page_load_configuration_compatibility";
    }

    RHITestResult run(RHITestContext&) override
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
            return RHITestResult::fail("page load concurrency configuration compatibility failed");
        }
        return RHITestResult::pass();
    }
};

class StreamingTaskQueueLifecycleTest : public RHITest {
public:
    StreamingTaskQueueLifecycleTest()
    {
        type = RHITestType::Validation;
        name = "streaming_task_queue_lifecycle";
    }

    RHITestResult run(RHITestContext&) override
    {
        render::StreamingTaskQueue queue;
        if (queue.availableTaskCount() != render::kStreamingMaxActiveTasks ||
            queue.queuedTaskCount() != 0 ||
            queue.acquiredTaskCount() != 0) {
            return RHITestResult::fail("new StreamingTaskQueue did not start with all tasks available");
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
            return RHITestResult::fail("StreamingTaskQueue did not allocate fixed task indices in order");
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
            return RHITestResult::fail("StreamingTaskQueue did not expose front task acquisition pressure");
        }
        if (queue.canPop(4, false) ||
            !queue.canPop(5, false) ||
            queue.queuedTaskCount() != 2) {
            return RHITestResult::fail("StreamingTaskQueue completion frame test failed");
        }

        uint32_t dependent = render::kInvalidStreamingTaskIndex;
        const uint32_t popped = queue.popWithDependent(dependent);
        if (popped != first || dependent != 17 || queue.queuedTaskCount() != 1) {
            return RHITestResult::fail("StreamingTaskQueue did not pop the first queued task with its dependent index");
        }
        queue.releaseTaskIndex(popped);
        if (queue.availableTaskCount() != 1 || queue.acquiredTaskCount() != 2) {
            return RHITestResult::fail("StreamingTaskQueue did not release a completed task index");
        }

        const uint32_t recycled = queue.acquireTaskIndex();
        if (recycled != first) {
            return RHITestResult::fail("StreamingTaskQueue did not recycle the released task index");
        }
        queue.push(recycled, 6);
        if (queue.canPop(6, false)) {
            return RHITestResult::fail("StreamingTaskQueue did not preserve FIFO completion order");
        }
        if (!queue.canPop(7, true)) {
            return RHITestResult::fail("StreamingTaskQueue did not report the front task ready at its completion frame");
        }

        queue.releaseTaskIndex(queue.pop());
        queue.releaseTaskIndex(queue.pop());
        queue.releaseTaskIndex(third);
        if (!queue.empty() ||
            queue.availableTaskCount() != render::kStreamingMaxActiveTasks ||
            queue.acquiredTaskCount() != 0) {
            return RHITestResult::fail("StreamingTaskQueue did not return to an idle state");
        }

        return RHITestResult::pass();
    }
};

class MeshletStreamStorageAddressLimitTest : public RHITest {
public:
    MeshletStreamStorageAddressLimitTest()
    {
        type = RHITestType::Validation;
        name = "meshlet_stream_storage_address_limit";
    }

    RHITestResult run(RHITestContext&) override
    {
        constexpr uint64_t kLargeCapacity = 5ull * 1024ull * 1024ull * 1024ull;
        render::MeshletStreamStorage storage;
        std::string reason;
        if (storage.initialize(kLargeCapacity, 256, reason)) {
            return RHITestResult::fail("page storage accepted a byte budget above its default 32-bit limit");
        }
        if (!storage.initialize(kLargeCapacity, 256, reason, UINT64_MAX) ||
            storage.capacityBytes() != kLargeCapacity) {
            return RHITestResult::fail("64-bit CLAS storage budget initialization failed: " + reason);
        }

        const render::MeshletStreamStorageAllocation allocation =
            storage.allocate(kLargeCapacity - 256u);
        if (!allocation.valid() ||
            allocation.offset != 0 ||
            storage.usedBytes() != allocation.allocatedSize) {
            return RHITestResult::fail("64-bit CLAS storage allocation failed");
        }
        storage.release(allocation);
        if (storage.usedBytes() != 0 || storage.freeBytes() != storage.capacityBytes()) {
            return RHITestResult::fail("64-bit CLAS storage release did not restore capacity");
        }
        return RHITestResult::pass();
    }
};

class StreamLODPipelineCacheTest : public RHITest {
public:
    StreamLODPipelineCacheTest() { type = RHITestType::Resource; name = "stream_lod_pipeline_cache_persistence"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Metallic Stream LOD Cache Test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("Bindless device unavailable")
                : RHITestResult::fail("Cannot create LOD cache test device");
        }
        const auto assetPath = context.outputDirectory / "lod_pipeline_cache.meshstream.bin";
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(assetPath, asset);
        if (!built.passed) { return built; }
        const auto cachePath = context.outputDirectory / "lod_pipeline_cache.pso";
        std::error_code fileError;
        std::filesystem::remove(cachePath, fileError);
        if (fileError) { return RHITestResult::fail("Cannot clear test pipeline cache: " + fileError.message()); }
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
            auto result = device->createPipelineCache({.filePath = cacheName.c_str(), .saveOnDestroy = false}).transform([&](auto rhiValue) { cache = std::move(rhiValue); });
            if (!result || !cache) { return RHITestResult::fail("Cannot create LOD test cache"); }
            const auto expectedLoad = pass == 0 ? PipelineCacheLoadStatus::NotFound : PipelineCacheLoadStatus::Loaded;
            if (cache->stats().loadStatus != expectedLoad) { return RHITestResult::fail("LOD cache load status mismatch"); }
            result = runtime.initialize(*device, desc, log, cache.get());
            if (!result || !runtime.ready()) { return RHITestResult::fail("LOD initialization failed: " + log); }
            const auto stats = cache->stats();
            // Page-table init/update, traversal, active build, cooperative LOD.
            if (stats.sessionPsoCount != 5 || stats.hitCount != (pass == 0 ? 0 : 5) ||
                stats.missCount != (pass == 0 ? 5 : 0)) {
                return RHITestResult::fail("An internal streaming/LOD pipeline bypassed the persistent cache");
            }
            result = cache->save();
            if (!result || cache->stats().backendDataSize == 0) {
                return RHITestResult::fail("LOD cache did not serialize native pipeline data");
            }
            // Initialization does not retain the caller's cache pointer.
            cache.reset();
            runtime.reset();
        }
        const auto uncached = runtime.initialize(*device, desc, log);
        if (!uncached || !runtime.ready()) {
            return RHITestResult::fail("Optional-cache compatibility failed: " + log);
        }
        return RHITestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(StreamLODPipelineCacheTest);

class StreamLODDisplayPixelParamsTest final : public RHITest {
public:
    StreamLODDisplayPixelParamsTest()
    {
        type = RHITestType::Rendering;
        name = "meshlet_lod_stream_display_pixel_params";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream display-pixel LOD",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(created, Error::Unsupported)) { return RHITestResult::skip("Requires bindless compute"); }
        if (!created) { return RHITestResult::fail("Cannot create stream display-pixel device"); }
        const auto assetPath = std::filesystem::absolute(context.outputDirectory / "DisplayPixelLOD.meshstream.bin");
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(assetPath, asset);
        if (!built.passed) { return built; }
        MeshletStreamRuntime runtime;
        std::string log;
        const auto initialized = runtime.initialize(*device, {
            .sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf",
            .streamAssetPath = assetPath, .maxResidentBytes = 16ull << 20, .maxResidentPages = 256,
            .maxPageUploadsPerFrame = 64, .maxGpuPageRequests = 256, .maxGpuPageUnloadRequests = 256,
            .maxActiveGroups = 2048, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
            .pageLoadConcurrency = 0, .queuedFrameCount = 2, .prefetchPages = false}, log);
        if (!initialized) { return RHITestResult::fail("Stream initialize: " + log); }
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        const auto setup = createCommandResources(*device, *queue, pool, commands, fence);
        if (!setup.passed) { return setup; }
        std::unique_ptr<Streamer> streamer;
        if (!createStreamer(*device, makeTestStreamerDesc()).transform([&](auto value) { streamer = std::move(value); })) {
            return RHITestResult::fail("Cannot create display-pixel streamer");
        }
        struct Case {
            uint32_t renderHeight, displayHeight;
            float pixelError, bias, expected;
        };
        const Case cases[] = {
            {1080, 1080, 1.5f, 0.f, 1.5f}, {720, 1080, 1.5f, 0.f, 1.f},
            {540, 1080, 1.5f, 0.f, .75f}, {2160, 1080, 1.5f, 0.f, 3.f},
            {720, 0, 1.5f, 0.f, 1.5f}, {0, 0, 1.5f, 0.f, 1.5f},
            {0, 1080, 1.5f, 0.f, 1.5f / 1080.f},
            {540, 1080, .01f, 0.f, .025f}, {540, 1080, 32.f, 0.f, 8.f},
            {540, 1080, 1.5f, 1.f, 1.5f}};
        for (bool orthographic : {false, true}) {
            for (const auto& test : cases) {
                MeshletStreamFrameDesc frame;
                frame.width = test.renderHeight * 16u / 9u;
                frame.height = test.renderHeight;
                frame.displayHeight = test.displayHeight;
                frame.lodPixelError = test.pixelError;
                frame.lodBias = test.bias;
                frame.camera = {.eye = {-.0168404f, .110154f, .22f},
                    .center = {-.0168404f, .110154f, -.00153695f},
                    .znear = .001f, .zfar = 10.f, .orthographic = orthographic, .orthoHeight = .24f};
                frame.useSeparateRenderCamera = true;
                frame.renderCamera = frame.camera;
                frame.renderCamera.fovDegrees = 75.f;
                if (!pool->reset() || !fence->reset() || !commands->begin()) {
                    return RHITestResult::fail("Display-pixel frame setup failed");
                }
                auto result = runtime.cmdBeginFrame(*commands, *streamer, frame);
                if (result) { result = streamer->copyStreamedData(*commands); }
                if (result) { result = runtime.cmdPreTraversal(*commands, frame); }
                if (result) { result = runtime.cmdPostTraversal(*commands); }
                if (result) { result = runtime.cmdEndFrame(*commands); }
                if (!result || !commands->end()) { return RHITestResult::fail("Display-pixel frame recording failed"); }
                const auto submitted = submitAndWait(*queue, *commands, *fence);
                streamer->endFrame();
                if (!submitted.passed) { return submitted; }
                MeshletStreamGPUParams params;
                auto* buffer = runtime.deferredGpuResources().paramsBuffer;
                if (!buffer || !readBufferBytes(*buffer, &params, sizeof(params))) {
                    return RHITestResult::fail("Cannot read published stream parameters");
                }
                if (std::abs(params.lodPixelError - test.expected) > 1e-6f ||
                    params.viewport[1] != float(std::max(frame.width, 1u)) ||
                    params.viewport[2] != float(std::max(frame.height, 1u)) ||
                    params.renderViewport[1] != params.viewport[1] || params.renderViewport[2] != params.viewport[2] ||
                    params.upProjection[3] != (orthographic ? 1.f : 0.f)) {
                    return RHITestResult::fail("Published LOD threshold or raster viewport used the wrong pixel space");
                }
            }
        }
        return RHITestResult::pass("Stream frame-to-GPU parameters convert display thresholds after clamp/bias and preserve internal raster dimensions, including separate render cameras and missing extents");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamLODDisplayPixelParamsTest);

class StreamCLASRuntimeTest : public RHITest {
public:
    explicit StreamCLASRuntimeTest(bool pressure = false) : pressure_(pressure)
    {
        type = RHITestType::Rendering;
        name = pressure ? "stream_clas_eviction_reupload" : "stream_clas_runtime_lifecycle";
    }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream CLAS lifecycle",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true, .enableClusterAccelerationStructure = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("Requires CLAS and bindless support")
                : RHITestResult::fail("CLAS device creation failed");
        }
        if (!device->capabilities().clusterAccelerationStructure) { return RHITestResult::skip("CLAS unavailable"); }
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
        if (!initialized) { return RHITestResult::fail("CLAS-only initialize: " + log); }
        auto* queue = device->getQueue(QueueType::Graphics);
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        std::unique_ptr<Fence> fence;
        const auto setup = createCommandResources(*device, *queue, pool, commands, fence);
        if (!setup.passed) { return setup; }
        std::unique_ptr<Streamer> streamer;
        if (!createStreamer(*device, makeTestStreamerDesc()).transform([&](auto rhiValue) { streamer = std::move(rhiValue); })) { return RHITestResult::fail("Cannot create streamer"); }
        MeshletStreamFrameDesc frame{.width = 192, .height = 128, .selectedLodLevel = 0, .enableGpuLodSelection = false};
        frame.camera = {.eye = {-.0168404f, .110154f, .22f}, .center = {-.0168404f, .110154f, -.00153695f},
            .znear = .001f, .zfar = 10.f};
        const auto nearCamera = frame.camera;
        bool sawBuilt = false, sawPending = false;
        uint64_t stableBuildCount = 0;
        uint32_t reloadPage = UINT32_MAX;
        std::ofstream trace(context.outputDirectory / (pressure_ ? "CLASEviction.jsonl" : "CLASLifecycle.jsonl"));
        for (uint32_t f = 0; f < 420; ++f) {
            frame.camera = nearCamera;
            if (f >= 180 && f < 240) { frame.camera.eye = {100.f, 100.f, 100.f}; frame.camera.center = {101.f, 100.f, 100.f}; }
            if (!pool->reset() || !fence->reset() || !commands->begin()) { return RHITestResult::fail("Frame setup failed"); }
            if (pressure_ && reloadPage != UINT32_MAX && f % 12 == 2) {
                const_cast<MeshletStreamResidencyManager&>(runtime.residency()).requestPage(reloadPage);
            }
            auto result = runtime.cmdBeginFrame(*commands, *streamer, frame);
            if (result) {
                if (auto commandResult = streamer->copyStreamedData(*commands); !commandResult) { return RHITestResult::fail(std::string("copyStreamedData failed: ") + render::resultToString(commandResult)); }
                // Keep the old CLAS queue entry across unload completion, then
                // admit a new upload before draining it (e.g. deferred traversal).
                if (!pressure_ || reloadPage == UINT32_MAX || f % 12 != 1) {
                    result = runtime.cmdPreTraversal(*commands, frame);
                }
            }
            if (result) { result = runtime.cmdPostTraversal(*commands); }
            if (result) { result = runtime.cmdEndFrame(*commands); }
            if (!result || !commands->end()) { return RHITestResult::fail("CLAS frame failed: " + std::string(toString(result))); }
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
                return RHITestResult::fail("CLAS-only runtime violated build/storage budget or built a TLAS");
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
                return RHITestResult::fail("Steady resident CLAS rebuilt or backlog failed to converge");
            }
        }
        const auto last = runtime.profilingStats();
        if (pressure_) {
            const auto residency = runtime.residency().stats();
            if (!sawPending || residency.totalCompletedUnloadCount < 10 || last.clasRejectedPages == 0) {
                return RHITestResult::fail("Pressure fixture did not exercise eviction and exhausted CLAS storage: " +
                    std::to_string(residency.totalCompletedUnloadCount));
            }
            return RHITestResult::pass("CLAS budget exhaustion with repeated geometry eviction/reupload retains live upload plans");
        }
        if (!sawBuilt || !sawPending || last.clasPendingPages || last.clasResidentPages != last.residentPages ||
            last.clasTotalBuiltPages != stableBuildCount) {
            return RHITestResult::fail("Lifecycle coverage/convergence: built=" + std::to_string(sawBuilt) +
                " pending=" + std::to_string(sawPending) +
                " final pending=" + std::to_string(last.clasPendingPages) + " resident=" + std::to_string(last.residentPages) +
                " CLAS=" + std::to_string(last.clasResidentPages));
        }
        return RHITestResult::pass("Compressed uploads, bounded build backlog, camera round-trip reuses cached CLAS, independent CLAS without BLAS/TLAS");
    }
private:
    bool pressure_ = false;
};
METALLIC_REGISTER_RHI_TEST(StreamCLASRuntimeTest);
class StreamCLASEvictionTest final : public StreamCLASRuntimeTest {
public:
    StreamCLASEvictionTest() : StreamCLASRuntimeTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(StreamCLASEvictionTest);

class StreamBLASCacheTest final : public RHITest {
public:
    StreamBLASCacheTest() { type = RHITestType::Rendering; name = "stream_blas_cut_cache"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream BLAS reuse regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true, .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true, .enableClusterAccelerationStructure = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!created) { return hasError(created, Error::Unsupported) ? RHITestResult::skip("CLAS unavailable")
            : RHITestResult::fail("Device creation failed"); }
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
            require(bool(tracker.initialize(*device, *queue)) && bool(device->createCommandPool(*queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); })) &&
                bool(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); })) && bool(createStreamer(*device, makeTestStreamerDesc()).transform([&](auto rhiValue) { streamer = std::move(rhiValue); })) &&
                bool(device->createBuffer({.size = sizeof(MeshletStreamGPUBLASHeader) + sizeof(MeshletStreamGPUActiveHeader) + 2048 * sizeof(MeshletStreamGPUActiveGroup), .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); })), "Frame resources failed");
            MeshletStreamFrameDesc view{.width = 192, .height = 128, .selectedLodLevel = 0, .enableGpuLodSelection = false};
            view.camera = {.eye = {-.0168404f, .110154f, .22f}, .center = {-.0168404f, .110154f, -.00153695f}, .znear = .001f, .zfar = 10.f};
            uint64_t frameId = 0;
            bool cancelledInitialFallback = false;
            MeshletStreamGPUBLASHeader header;
            std::vector<uint32_t> referencedPages;
            uint64_t lastAcceptedFeedback = UINT64_MAX;
            const auto record = [&](bool cancel = false) {
                const auto recordedFrame = runtime.frameIndex();
                runtime.prepareMaintenance();
                const auto maintained = runtime.residency().stats();
                runtime.prepareMaintenance();
                require(runtime.frameIndex() == recordedFrame &&
                    runtime.residency().stats().frameGpuRequestCount == maintained.frameGpuRequestCount,
                    "Maintenance advanced recording or consumed feedback twice");
                require(runtime.profilingStats().feedbackFrame == lastAcceptedFeedback,
                    "Maintenance consumed cancelled/uncompleted feedback or missed completed feedback");
                require(bool(frame.begin(++frameId)) && bool(pool->reset()) && bool(commands->begin(frame.submissionContext())) &&
                    bool(streamer->beginFrame(frame)), "Frame begin failed");
                require(bool(runtime.cmdBeginFrame(*commands, *streamer, view)), "Stream begin failed");
                if (auto commandResult = streamer->copyStreamedData(*commands); !commandResult) { throw std::runtime_error(std::string("copyStreamedData failed: ") + metallic::render::resultToString(commandResult)); }
                require(bool(runtime.cmdPreTraversal(*commands, view)) && bool(runtime.cmdPostTraversal(*commands)) &&
                    bool(runtime.cmdEndFrame(*commands)), "Stream traversal failed");
                std::vector<DebugResourceBinding> bindings;
                runtime.appendDebugBindings(bindings, "test.");
                const auto found = std::find_if(bindings.begin(), bindings.end(), [](const auto& binding) { return binding.id == "test.blasHeader"; });
                require(found != bindings.end(), "BLAS telemetry missing");
                BufferBarrierDesc barrier{
                    .buffer = found->buffer,
                    .before = metallic::render::resourceSyncScope(found->state, metallic::render::PipelineStageBits::AllCommands),
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                };
                if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
                {
                    auto sourceSlice = found->buffer->slice({0, sizeof(header)});
                    if (!sourceSlice) { throw std::runtime_error(std::string("source slice failed: ") + metallic::render::resultToString(sourceSlice)); }
                    auto destinationSlice = readback.get()->slice({0, sizeof(header)});
                    if (!destinationSlice) { throw std::runtime_error(std::string("destination slice failed: ") + metallic::render::resultToString(destinationSlice)); }
                    if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { throw std::runtime_error(std::string("copyBuffer failed: ") + metallic::render::resultToString(commandResult)); }
                }
                std::swap(barrier.before, barrier.after);
                if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { throw std::runtime_error(std::string("synchronize failed: ") + metallic::render::resultToString(commandResult)); }
                uint64_t readOffset = sizeof(header);
                for (const auto& [name, bytes] : {std::pair{"test.activeHeader", uint64_t(sizeof(MeshletStreamGPUActiveHeader))},
                        std::pair{"test.activeGroups", uint64_t(2048 * sizeof(MeshletStreamGPUActiveGroup))}}) {
                    const auto source = std::find_if(bindings.begin(), bindings.end(), [&](const auto& binding) { return binding.id == name; });
                    require(source != bindings.end(), "Active cut telemetry missing");
                    BufferBarrierDesc copyBarrier{.buffer = source->buffer,
                        .before = resourceSyncScope(source->state, PipelineStageBits::AllCommands),
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                    require(bool(commands->synchronize({.buffers = {&copyBarrier, 1}})), "Cut copy barrier");
                    const uint64_t copyBytes = std::min(bytes, source->buffer->desc().size);
                    auto src = source->buffer->slice({0, copyBytes}); auto dst = readback->slice({readOffset, copyBytes});
                    require(bool(src) && bool(dst) && bool(commands->copyBuffer(*src, *dst)), "Cut copy");
                    std::swap(copyBarrier.before, copyBarrier.after);
                    require(bool(commands->synchronize({.buffers = {&copyBarrier, 1}})), "Cut restore barrier");
                    readOffset += bytes;
                }
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
                require(bool(tracker.submit({.commandBuffers = {list, 1}}, frame)) &&
                    bool(frame.wait(5000000000ull)), "Frame submit failed");
                lastAcceptedFeedback = runtime.frameIndex();
                streamer->endFrame();
                readback->invalidate();
                const auto* data = readback->map();
                require(data != nullptr, "BLAS readback failed");
                std::memcpy(&header, data, sizeof(header));
                MeshletStreamGPUActiveHeader active;
                std::memcpy(&active, static_cast<const uint8_t*>(data) + sizeof(header), sizeof(active));
                referencedPages.clear();
                for (uint32_t group = 0; group < std::min(active.activeGroupCount, 2048u); ++group) {
                    MeshletStreamGPUActiveGroup entry;
                    std::memcpy(&entry, static_cast<const uint8_t*>(data) + sizeof(header) + sizeof(active) + group * sizeof(entry), sizeof(entry));
                    referencedPages.push_back(entry.pageIndex);
                }
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
                return runtime.residency().pageState(page) == MeshletStreamPageResidencyState::Resident &&
                    std::find(referencedPages.begin(), referencedPages.end(), page) != referencedPages.end();
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
            return RHITestResult::pass("Stable cut reuse, changed LOD, cancellation and CLAS retirement invalidation");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamBLASCacheTest);

class StreamInitialLoadingTest final : public RHITest {
public:
    StreamInitialLoadingTest() { type = RHITestType::Rendering; name = "stream_initial_loading"; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto require = [](bool condition, const std::string& reason) {
            if (!condition) { throw std::runtime_error(reason); }
        };
        std::atomic_uint validationMessages = 0;
        std::unique_ptr<Device> device;
        const auto created = createDevice({.applicationName = "Stream initial loading regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableShaderObject = true, .enableRayTracingAccelerationStructure = true,
            .enableRayQuery = true, .enableClusterAccelerationStructure = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationMessages}})
            .transform([&](auto value) { device = std::move(value); });
        if (!created) { return hasError(created, Error::Unsupported) ? RHITestResult::skip("CLAS unavailable")
            : RHITestResult::fail("Initial loading device creation failed"); }
        try {
            {
                // Four independent primitives force a root batch larger than the
                // one-page steady-state budget, while retaining refinement pages.
                const auto directory = std::filesystem::absolute(context.outputDirectory / "InitialLoading");
                std::filesystem::create_directories(directory);
                const auto original = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
                std::ifstream input(original);
                auto source = nlohmann::json::parse(input);
                const auto primitive = source.at("meshes").at(0).at("primitives").at(0);
                for (uint32_t index = 1; index < 4; ++index) {
                    source["meshes"][0]["primitives"].push_back(primitive);
                }
                for (const auto& buffer : source.at("buffers")) {
                    const auto uri = buffer.at("uri").get<std::string>();
                    std::filesystem::copy_file(original.parent_path() / uri, directory / uri,
                        std::filesystem::copy_options::overwrite_existing);
                }
                const auto sourcePath = directory / "Scene.gltf";
                { std::ofstream output(sourcePath); output << source.dump(2); }
                scene::Scene scene;
                require(scene.load(sourcePath), scene.lastLoadResult().error);
                const auto assetPath = directory / "Scene.meshstream.bin";
                std::string log;
                require(scene::buildMeshletStreamAsset({.scene = &scene, .sourcePath = sourcePath,
                    .outputPath = assetPath, .compressionMode = scene::MeshletStreamPayloadCompression::ByteRle}, log), log);

                MeshletStreamRuntime runtime;
                MeshletStreamInitialLoader loader;
                MeshletStreamRuntimeDesc desc{.sourcePath = sourcePath, .streamAssetPath = assetPath,
                    .maxResidentBytes = 32ull << 20, .maxResidentPages = 1024, .maxLockedFallbackPages = 1024,
                    .maxPageUploadsPerFrame = 1, .maxUploadBytesPerFrame = 1,
                    .maxGpuPageRequests = 1024, .maxGpuPageUnloadRequests = 1024,
                    .maxActiveGroups = 2048, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
                    .pageLoadConcurrency = 0, .maxPageLoadsInFlight = 256, .queuedFrameCount = 2,
                    .enableClusterRtx = true, .enableClas = true, .compactClas = true,
                    .maxClasBytes = 32ull << 20, .maxClasBuildClusters = 4096,
                    .maxBlasClusterReferences = 4096, .maxBlasBytes = 16ull << 20, .maxBlasBuilds = 8,
                    .maxFallbackBlasBytes = 4ull << 20, .prefetchPages = false};
                auto& queue = *device->getQueue(QueueType::Graphics);
                QueueSubmissionTracker tracker;
                RenderFrameContext frame;
                std::unique_ptr<CommandPool> pool;
                std::unique_ptr<CommandBuffer> commands;
                std::unique_ptr<Streamer> streamer;
                std::unique_ptr<Semaphore> gate;
                struct Drain {
                    Queue& queue;
                    RenderFrameContext& frame;
                    std::unique_ptr<Semaphore>& gate;
                    ~Drain()
                    {
                        if (gate && gate->currentValue() < 2) { (void)gate->signal(2); }
                        frame.cancel();
                        (void)queue.waitIdle();
                    }
                } drain{queue, frame, gate};
                require(bool(tracker.initialize(*device, queue)) &&
                    bool(device->createCommandPool(queue).transform([&](auto value) { pool = std::move(value); })) &&
                    bool(pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); })) &&
                    bool(createStreamer(*device, makeTestStreamerDesc()).transform([&](auto value) { streamer = std::move(value); })) &&
                    bool(device->createSemaphore().transform([&](auto value) { gate = std::move(value); })),
                    "Initial loading command resources failed");
                uint64_t frameId = 0;
                nlohmann::json report = nlohmann::json::array();
                std::vector<std::array<uint32_t, 6>> deviceMetadataCut;
                for (uint32_t cycle = 0; cycle < 3; ++cycle) {
                    desc.deviceImmutableMetadata = cycle != 1;
                    // Initialization assigns TransferSource usage only when
                    // readback is enabled before these buffers are allocated.
                    runtime.setDebugReadbackEnabled(true);
                    require(bool(runtime.initialize(*device, desc, log)), log);
                    require(bool(loader.initialize(*device, log)), log);
                    require(!runtime.sceneReady() && runtime.sceneReadiness().completedPages == 0,
                        "Initial loading retained readiness across reload");
                    require(runtime.asset().primitiveCount() == 4 &&
                        runtime.debugSnapshot(false).at("terminalPageCount").get<uint32_t>() >= 4,
                        "Initial loading fixture must contain multiple independent roots");
                    if (cycle < 2) {
                        const uint64_t gateValue = cycle + 1;
                        for (uint32_t attempt = 0; attempt < 2; ++attempt) {
                            require(bool(frame.begin(++frameId)) && bool(pool->reset()) &&
                                bool(commands->begin(frame.submissionContext())) && bool(streamer->beginFrame(frame)),
                                "Initial upload begin failed");
                            const bool metadataBatch = desc.deviceImmutableMetadata && !runtime.immutableMetadataReady();
                            if (metadataBatch) {
                                require(runtime.immutableMetadataUploadedBytes() == 0,
                                    "Cancelled metadata recording advanced its upload offset");
                                require(hasError(runtime.cmdPreTraversal(*commands, MeshletStreamFrameDesc{}), Error::InvalidArgument),
                                    "Traversal accepted uninitialized Device metadata");
                            }
                            require(bool(runtime.cmdLoadInitialResources(*commands, *streamer)),
                                "Initial upload recording failed");
                            const auto snapshot = runtime.debugSnapshot(false);
                            if (metadataBatch) {
                                require(!runtime.immutableMetadataReady() && runtime.immutableMetadataUploadedBytes() == 0 &&
                                    snapshot.at("orderedUploadPages") == 0,
                                    "Unsubmitted metadata copy advanced readiness, bytes, or root publication");
                            } else {
                                require(snapshot.at("orderedUploadPages").get<uint32_t>() >= 4 &&
                                    runtime.residency().stats().frameUploadBytes > desc.maxUploadBytesPerFrame,
                                    "Initial root uploads remained limited by normal frame budgets");
                            }
                            require(!runtime.sceneReady() && runtime.residency().residentPageCount() == 0,
                                "Unsubmitted initial uploads made the scene ready");
                            require(bool(commands->end()), "Initial upload end failed");
                            streamer->endFrame();
                            if (attempt == 0) {
                                frame.cancel();
                                require(!runtime.sceneReady(), "Cancelled initial uploads made the scene ready");
                                if (metadataBatch) {
                                    require(!runtime.immutableMetadataReady() && runtime.immutableMetadataUploadedBytes() == 0,
                                        "Cancelled Device metadata became initialized");
                                }
                                continue;
                            }
                            CommandBuffer* list[]{commands.get()};
                            const SemaphoreSubmitDesc wait{.semaphore = gate.get(), .value = gateValue};
                            require(bool(tracker.submit({.waitSemaphores = {&wait, 1},
                                .commandBuffers = {list, 1}}, frame)), "Initial gated upload submit failed");
                            for (uint32_t query = 0; query < 16; ++query) {
                                runtime.prepareMaintenance();
                                require(!frame.completion().isComplete() && !runtime.sceneReady() &&
                                    runtime.residency().residentPageCount() == 0,
                                    "Queue acceptance or CPU polling published an incomplete root copy");
                                if (metadataBatch) {
                                    require(!runtime.immutableMetadataReady() && runtime.immutableMetadataUploadedBytes() > 0,
                                        "Device metadata readiness did not wait for the accepted GPU copy");
                                }
                            }
                            require(bool(gate->signal(gateValue)) && bool(frame.wait(5'000'000'000ull)),
                                "Initial gated upload did not complete");
                        }
                    } else {
                        // Aggregate completion can be valid even though the
                        // metadata tail was cancelled after an accepted prefix.
                        std::unique_ptr<CommandBuffer> prefix;
                        require(bool(pool->createCommandBuffer().transform([&](auto value) { prefix = std::move(value); })),
                            "Metadata prefix command allocation failed");
                        require(bool(frame.begin(++frameId)) && bool(pool->reset()) &&
                            bool(commands->begin(frame.submissionContext())) && bool(streamer->beginFrame(frame)),
                            "Metadata cancelled-tail begin failed");
                        require(bool(runtime.cmdLoadInitialResources(*commands, *streamer)) && bool(commands->end()),
                            "Metadata cancelled-tail recording failed");
                        streamer->endFrame();
                        require(bool(prefix->begin(frame.submissionContext())) && bool(prefix->end()), "Metadata prefix recording failed");
                        CommandBuffer* prefixCommands[]{prefix.get()};
                        require(bool(tracker.submitSegment({.commandBuffers = {prefixCommands, 1}}, frame)),
                            "Metadata prefix submit failed");
                        frame.cancel();
                        require(bool(frame.wait(5'000'000'000ull)) && frame.completion().isSubmitted() && frame.completion().isComplete(),
                            "Metadata cancelled-tail fixture did not retain accepted-prefix completion");
                        require(!runtime.immutableMetadataReady() && runtime.immutableMetadataUploadedBytes() == 0 &&
                            !runtime.sceneReady(), "Cancelled metadata tail trusted aggregate prefix completion");

                        bool metadataComplete = false;
                        for (uint32_t call = 0; call < 64 && !metadataComplete; ++call) {
                            require(bool(loader.pump(runtime, .001, metadataComplete, log, true)), log);
                        }
                        require(metadataComplete && runtime.immutableMetadataReady() && runtime.immutableMetadataUploadedBytes() > 0 &&
                            !runtime.sceneReady() && runtime.residency().residentPageCount() == 0 &&
                            runtime.sceneReadiness().completedPages == 0,
                            "Metadata-only retry published geometry roots or failed to initialize Device tables");
                    }

                    bool complete = false;
                    for (uint32_t call = 0; call < 64 && !complete; ++call) {
                        require(bool(loader.pump(runtime, .001, complete, log)), log);
                        const auto stats = runtime.profilingStats();
                        require(stats.clasBuiltClusters <= desc.maxClasBuildClusters &&
                            stats.clasUsedBytes <= stats.clasCapacityBytes &&
                            stats.geometryUsedBytes <= stats.geometryBudgetBytes && stats.loadFailures == 0,
                            "Initial loading exceeded a resource budget or failed page IO");
                    }
                    require(complete && runtime.sceneReady(), "Initial loading did not converge in bounded submissions");
                    require(runtime.immutableMetadataReady() &&
                        (desc.deviceImmutableMetadata ? runtime.immutableMetadataUploadedBytes() > 0 : runtime.immutableMetadataUploadedBytes() == 0),
                        "Initial loader did not hand off initialized metadata for the selected memory path");
                    require(loader.stats().batches > 0 && loader.stats().batches <= 64,
                        "Initial loading did not use a bounded independent submission sequence");
                    require(!runtime.tlasReady() && !runtime.accelerationStructure() &&
                        runtime.profilingStats().feedbackFrame == UINT64_MAX,
                        "Initial loading unexpectedly required traversal, TLAS, or rendered-frame feedback");
                    const auto snapshot = runtime.debugSnapshot(false);
                    require(snapshot.at("fallbackBlasSubmitted").get<uint32_t>() == 4 &&
                        snapshot.at("maxUploadBytesPerFrame") == desc.maxUploadBytesPerFrame,
                        "Initial loading omitted fallback BLAS or changed the normal byte budget");
                    require(snapshot.at("immutableMetadataDeviceRequested") == desc.deviceImmutableMetadata &&
                        snapshot.at("immutableMetadataStagingPeakBytes").get<uint64_t>() <= (64ull << 20) &&
                        snapshot.at("immutableMetadataStagingBytes") == 0,
                        "Metadata upload did not honor its staging budget or release completed staging");
                    if (desc.deviceImmutableMetadata) {
                        require(snapshot.at("immutableMetadataSubmittedBytes") == snapshot.at("immutableMetadataBytes"),
                            "Initial loading handed off an incomplete immutable metadata copy");
                    }
                    report.push_back({{"cycle", cycle}, {"batches", loader.stats().batches},
                        {"rootPages", snapshot.at("terminalPageCount")}, {"runtime", snapshot}});

                    // Read the real GPU traversal cut after loader handoff. The
                    // Device copy must contain the same groups/topology as the
                    // original HostUpload metadata, including every primitive.
                    MeshletStreamFrameDesc view{.width = 192, .height = 128,
                        .selectedLodLevel = 31, .enableGpuLodSelection = false};
                    view.camera = {.eye = {-.0168404f, .110154f, .22f},
                        .center = {-.0168404f, .110154f, -.00153695f}, .znear = .001f, .zfar = 10.f};
                    require(bool(frame.begin(++frameId)) && bool(pool->reset()) &&
                        bool(commands->begin(frame.submissionContext())) && bool(streamer->beginFrame(frame)),
                        "Metadata traversal begin failed");
                    require(bool(runtime.cmdBeginFrame(*commands, *streamer, view)) &&
                        bool(streamer->copyStreamedData(*commands)) &&
                        bool(runtime.cmdPreTraversal(*commands, view)) &&
                        bool(runtime.cmdPostTraversal(*commands)) && bool(runtime.cmdEndFrame(*commands)),
                        "Metadata traversal recording failed");
                    std::vector<DebugResourceBinding> bindings;
                    runtime.appendDebugBindings(bindings, "metadata.");
                    const uint64_t headerBytes = sizeof(MeshletStreamGPUActiveHeader);
                    const uint64_t groupsBytes = uint64_t(desc.maxActiveGroups) * sizeof(MeshletStreamGPUActiveGroup);
                    std::unique_ptr<Buffer> cutReadback;
                    require(bool(device->createBuffer({.size = headerBytes + groupsBytes,
                        .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
                        .transform([&](auto value) { cutReadback = std::move(value); })), "Metadata cut readback allocation failed");
                    uint64_t readOffset = 0;
                    for (const auto& [id, maximumBytes] : {std::pair{"metadata.activeHeader", headerBytes},
                            std::pair{"metadata.activeGroups", groupsBytes}}) {
                        const auto binding = std::find_if(bindings.begin(), bindings.end(), [&](const auto& value) { return value.id == id; });
                        require(binding != bindings.end(), "Metadata traversal cut binding missing");
                        BufferBarrierDesc barrier{.buffer = binding->buffer,
                            .before = resourceSyncScope(binding->state, PipelineStageBits::AllCommands),
                            .after = {PipelineStageBits::Transfer, AccessBits::TransferRead}};
                        require(bool(commands->synchronize({.buffers = {&barrier, 1}})), "Metadata cut copy barrier failed");
                        const uint64_t copyBytes = std::min(maximumBytes, binding->buffer->desc().size);
                        auto src = binding->buffer->slice({0, copyBytes});
                        auto dst = cutReadback->slice({readOffset, copyBytes});
                        require(bool(src) && bool(dst), "Metadata cut slice failed");
                        const auto copied = commands->copyBuffer(*src, *dst);
                        require(bool(copied), std::string("Metadata cut copy ") + id + ": " + resultToString(copied));
                        std::swap(barrier.before, barrier.after);
                        require(bool(commands->synchronize({.buffers = {&barrier, 1}})), "Metadata cut restore barrier failed");
                        readOffset += maximumBytes;
                    }
                    const BufferBarrierDesc cutReady{.buffer = cutReadback.get(),
                        .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                        .after = {PipelineStageBits::Host, AccessBits::HostRead}};
                    require(bool(commands->synchronize({.buffers = {&cutReady, 1}})) && bool(commands->end()),
                        "Metadata traversal end failed");
                    CommandBuffer* traversalCommands[]{commands.get()};
                    require(bool(tracker.submit({.commandBuffers = {traversalCommands, 1}}, frame)) &&
                        bool(frame.wait(5'000'000'000ull)), "Metadata traversal submit failed");
                    streamer->endFrame();
                    std::vector<uint8_t> cutBytes(headerBytes + groupsBytes);
                    require(readBufferBytes(*cutReadback, cutBytes.data(), cutBytes.size()), "Metadata cut readback failed");
                    MeshletStreamGPUActiveHeader active;
                    std::memcpy(&active, cutBytes.data(), sizeof(active));
                    require(active.activeGroupCount > 0 && active.activeGroupCount <= desc.maxActiveGroups && active.overflowCount == 0,
                        "Initialized metadata produced an empty or overflowing traversal cut");
                    std::vector<std::array<uint32_t, 6>> cut;
                    for (uint32_t group = 0; group < active.activeGroupCount; ++group) {
                        MeshletStreamGPUActiveGroup entry;
                        std::memcpy(&entry, cutBytes.data() + headerBytes + group * sizeof(entry), sizeof(entry));
                        cut.push_back({entry.instanceIndex, entry.primitiveIndex, entry.pageIndex,
                            entry.lodLevel, entry.clusterCount, entry.clusterSelectionMask});
                    }
                    std::sort(cut.begin(), cut.end());
                    if (cycle == 0) { deviceMetadataCut = cut; }
                    else { require(cut == deviceMetadataCut, "Device and Host metadata produced different GPU traversal cuts"); }
                    report.back()["metadataTraversalGroups"] = active.activeGroupCount;

                    auto& residency = const_cast<MeshletStreamResidencyManager&>(runtime.residency());
                    std::vector<uint32_t> refinements;
                    for (uint32_t page = 0; page < runtime.asset().pageCount() && refinements.size() < 3; ++page) {
                        if (residency.pageState(page) == MeshletStreamPageResidencyState::Unloaded) {
                            const auto queued = residency.queuedUploadCount();
                            // requestPage returns true only for an already resident
                            // or pending-upload page; a newly queued request is false.
                            (void)residency.requestPage(page);
                            require(residency.pageAllocated(page) && residency.queuedUploadCount() == queued + 1,
                                "Could not queue a refinement page");
                            refinements.push_back(page);
                        }
                    }
                    require(refinements.size() == 3, "Initial fixture has insufficient refinement pages");
                    require(bool(frame.begin(++frameId)) && bool(pool->reset()) &&
                        bool(commands->begin(frame.submissionContext())) && bool(streamer->beginFrame(frame)), "Normal frame begin failed");
                    require(bool(runtime.cmdBeginFrame(*commands, *streamer, MeshletStreamFrameDesc{})),
                        "Normal frame upload failed after initial loading");
                    require(residency.stats().frameScheduledUploadCount == 1 &&
                        residency.stats().frameUploadBytes > 0,
                        "Normal frame did not restore its one-page upload limit");
                    require(bool(commands->end()), "Normal frame end failed");
                    streamer->endFrame();
                    frame.cancel();
                    require(bool(loader.reset()), "Initial loader reset failed");
                    runtime.reset();
                    require(!runtime.sceneReady() && runtime.sceneReadiness().requiredPages == 0,
                        "Initial loading reset retained scene readiness");
                }
                std::ofstream(directory / "InitialLoading.json") << report.dump(2) << '\n';
            }
            // Include retirement and device destruction in the validation verdict.
            device.reset();
            require(validationMessages.load() == 0, "Initial loading produced Vulkan validation errors");
            return RHITestResult::pass("Device/Host metadata agree on GPU traversal; independent loading, cancellation, GPU-gated readiness, steady-state budgets and reload");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(StreamInitialLoadingTest);


class StreamerBufferUploadTest : public RHITest {
public:
    StreamerBufferUploadTest()
    {
        type = RHITestType::Command;
        name = "streamer_buffer_upload";
    }

    RHITestResult run(RHITestContext& context) override
    {
        constexpr std::array<uint32_t, 4> kExpected{
            0x11223344u,
            0xAABBCCDDu,
            0xDEADBEEFu,
            0xCAFEBABEu,
        };
        constexpr uint64_t kByteSize = kExpected.size() * sizeof(uint32_t);

        std::unique_ptr<render::Streamer> streamer;
        render::Result<> result = createStreamer(context.device, makeTestStreamerDesc()).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
        if (!result || streamer == nullptr) {
            return RHITestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(render::BufferDesc{
                .size = kByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            }).transform([&](auto rhiValue) { readbackBuffer = std::move(rhiValue); });
        if (!result || readbackBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
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
            .dataChunks = {chunks, static_cast<uint32_t>(std::size(chunks))},
            .placementAlignment = 4,
            .dstBuffer = readbackBuffer.get(),
            .dstOffset = 0,
        });
        if (!streamed.valid()) {
            return RHITestResult::fail("streamBufferData returned an invalid source");
        }
        render::StreamerStats streamerStats = streamer->stats();
        if (streamerStats.currentFrameDynamicBytes != kByteSize ||
            streamerStats.currentFrameDynamicRequestCount != 1 ||
            streamerStats.totalDynamicBytes != 0 ||
            streamerStats.pendingCopies.bufferCopyCount != 1 ||
            streamerStats.pendingCopies.bufferCopyBytes != kByteSize) {
            return RHITestResult::fail("streamBufferData did not update current-frame dynamic streamer stats");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RHITestResult setup = createCommandResources(
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
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::BufferBarrierDesc toTransfer{
            .buffer = readbackBuffer.get(),
            .before = {},
            .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
            .range = {.offset = 0, .size = kByteSize},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = {&toTransfer, 1},
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        if (auto commandResult = streamer->copyStreamedData(*commandBuffer); !commandResult) { return RHITestResult::fail(std::string("copyStreamedData failed: ") + render::resultToString(commandResult)); }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RHITestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
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
            return RHITestResult::fail("Streamer::endFrame did not roll dynamic upload stats");
        }

        std::array<uint32_t, 4> actual{};
        if (!readBufferBytes(*readbackBuffer, actual.data(), kByteSize)) {
            return RHITestResult::fail("readback buffer did not map");
        }
        if (actual != kExpected) {
            return RHITestResult::fail("streamed buffer bytes did not match expected pattern");
        }
        return RHITestResult::pass();
    }
};

class StreamerTextureUploadTest : public RHITest {
public:
    StreamerTextureUploadTest()
    {
        type = RHITestType::Command;
        name = "streamer_texture_upload";
    }

    RHITestResult run(RHITestContext& context) override
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
        render::Result<> result = createStreamer(context.device, makeTestStreamerDesc()).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
        if (!result || streamer == nullptr) {
            return RHITestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Texture> texture;
        result = context.device.createTexture(render::TextureDesc{
                .type = render::TextureType::Texture2D,
                .usage = render::TextureUsageBits::TransferDestination | render::TextureUsageBits::TransferSource,
                .format = render::Format::RGBA8Unorm,
                .width = kWidth,
                .height = kHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = render::MemoryLocation::Device,
            }).transform([&](auto rhiValue) { texture = std::move(rhiValue); });
        if (!result || texture == nullptr) {
            return RHITestResult::fail(std::string("createTexture returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> readbackBuffer;
        result = context.device.createBuffer(render::BufferDesc{
                .size = kPixelByteSize,
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            }).transform([&](auto rhiValue) { readbackBuffer = std::move(rhiValue); });
        if (!result || readbackBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer(readback) returned ") + toString(result));
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
            return RHITestResult::fail("streamTextureData returned an invalid source");
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RHITestResult setup = createCommandResources(
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
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::TextureBarrierDesc textureToTransfer{
            .texture = texture.get(),
            .oldLayout = render::TextureLayout::Undefined,
            .newLayout = render::TextureLayout::TransferDestination,
            .before = {},
            .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
            .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .textures = {&textureToTransfer, 1},
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        if (auto commandResult = streamer->copyStreamedData(*commandBuffer); !commandResult) { return RHITestResult::fail(std::string("copyStreamedData failed: ") + render::resultToString(commandResult)); }
        render::TextureBarrierDesc textureToSource{
            .texture = texture.get(),
            .oldLayout = render::TextureLayout::TransferDestination,
            .newLayout = render::TextureLayout::TransferSource,
            .before = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
            .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferRead},
            .range = {.baseMip = 0, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .textures = {&textureToSource, 1},
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        commandBuffer->copyTextureToBuffer(render::TextureBufferCopyDesc{
            .texture = texture.get(),
            .buffer = readbackBuffer.get(),
            .width = kWidth,
            .height = kHeight,
            .depth = 1,
        });
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RHITestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }

        std::array<uint8_t, kPixelByteSize> actual{};
        if (!readBufferBytes(*readbackBuffer, actual.data(), actual.size())) {
            return RHITestResult::fail("texture readback buffer did not map");
        }
        if (actual != expected) {
            return RHITestResult::fail("streamed texture pixels did not match expected pattern");
        }
        return RHITestResult::pass();
    }
};

class StreamerConstantUploadTest : public RHITest {
public:
    StreamerConstantUploadTest()
    {
        type = RHITestType::Resource;
        name = "streamer_constant_upload";
    }

    RHITestResult run(RHITestContext& context) override
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
        render::Result<> result = createStreamer(context.device, desc).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
        if (!result || streamer == nullptr || streamer->constantBuffer() == nullptr) {
            return RHITestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        const uint64_t firstOffset = streamer->streamConstantData(
            kFirst.data(),
            kFirst.size() * sizeof(uint32_t));
        const uint64_t secondOffset = streamer->streamConstantData(
            kSecond.data(),
            kSecond.size() * sizeof(uint32_t));
        if (firstOffset == std::numeric_limits<uint64_t>::max() ||
            secondOffset == std::numeric_limits<uint64_t>::max()) {
            return RHITestResult::fail("streamConstantData returned an invalid offset");
        }
        if (firstOffset != 0) {
            return RHITestResult::fail("first constant upload did not start at offset zero");
        }

        const uint64_t alignment = std::max<uint64_t>(
            context.device.capabilities().constantBufferOffsetAlignment,
            1);
        if (secondOffset % alignment != 0 ||
            secondOffset < kFirst.size() * sizeof(uint32_t)) {
            return RHITestResult::fail("second constant upload was not aligned after first upload");
        }
        const uint64_t expectedConstantBytes =
            kFirst.size() * sizeof(uint32_t) + kSecond.size() * sizeof(uint32_t);
        render::StreamerStats streamerStats = streamer->stats();
        if (streamerStats.currentFrameConstantBytes != expectedConstantBytes ||
            streamerStats.currentFrameConstantRequestCount != 2 ||
            streamerStats.totalConstantBytes != 0) {
            return RHITestResult::fail("streamConstantData did not update current-frame constant streamer stats");
        }

        render::Buffer* constantBuffer = streamer->constantBuffer();
        constantBuffer->invalidate({0, desc.constantBufferSize});
        void* mapped = constantBuffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("constant buffer did not map");
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
            return RHITestResult::fail("constant buffer contents did not match streamed data");
        }
        streamer->endFrame();
        streamerStats = streamer->stats();
        if (streamerStats.currentFrameConstantBytes != 0 ||
            streamerStats.lastFrameConstantBytes != expectedConstantBytes ||
            streamerStats.peakFrameConstantBytes != expectedConstantBytes ||
            streamerStats.totalConstantBytes != expectedConstantBytes ||
            streamerStats.lastFrameConstantRequestCount != 2) {
            return RHITestResult::fail("Streamer::endFrame did not roll constant upload stats");
        }
        return RHITestResult::pass();
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

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
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
            .dataChunks = {&chunk, 1},
            .placementAlignment = 4,
            .dstBuffer = output.buffer(),
            .dstOffset = 0,
        });
        return streamed.valid() ? render::Result<>{} : render::makeError(render::Error::Failure);
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

    render::Result<> execute(render::RenderGraphExecutionContext&) override
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

    render::Result<> execute(render::RenderGraphExecutionContext&) override
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

class StreamerRenderGraphFlushTest : public RHITest {
public:
    StreamerRenderGraphFlushTest()
    {
        type = RHITestType::Command;
        name = "streamer_render_graph_flush";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerStreamerGraphPass();

        render::RenderGraph graph;
        graph.setName("StreamerGraph");
        graph.addNode("StreamerGraphUploadPass", "Upload");
        graph.markOutput("Upload.data");

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result<> result = executor.compile(context.device, graph, 1, 1, log);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RHITestResult setup = createCommandResources(
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
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        result = executor.execute(*commandBuffer);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::execute returned ") + toString(result));
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
            return RHITestResult::fail("RenderGraph streaming subsystem stats did not match the streamed pass work");
        }
        if (streamingStats.streamer.pendingCopies.copyCount() != 0 ||
            streamingStats.streamer.frameIndex == 0) {
            return RHITestResult::fail("RenderGraph streaming subsystem did not end the streamer frame cleanly");
        }
        if (streamingStats.streamer.currentFrameDynamicBytes != 0 ||
            streamingStats.streamer.lastFrameDynamicBytes != expectedBytes ||
            streamingStats.streamer.peakFrameDynamicBytes != expectedBytes ||
            streamingStats.streamer.totalDynamicBytes != expectedBytes ||
            streamingStats.streamer.lastFrameDynamicRequestCount != 1) {
            return RHITestResult::fail("RenderGraph streaming subsystem did not retain last-frame Streamer stats");
        }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RHITestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        if (!submit.passed) {
            return submit;
        }

        render::RenderGraphResource* output = executor.outputResource("Upload.data");
        if (output == nullptr || output->buffer == nullptr) {
            return RHITestResult::fail("streamer graph output resource is missing");
        }

        std::array<uint32_t, 4> actual{};
        if (!readBufferBytes(
                *output->buffer,
                actual.data(),
                actual.size() * sizeof(uint32_t))) {
            return RHITestResult::fail("streamer graph output did not map");
        }
        if (actual != StreamerGraphUploadPass::kExpected) {
            return RHITestResult::fail("streamer graph output bytes did not match expected pattern");
        }
        return RHITestResult::pass();
    }
};

class StreamerRenderGraphInvalidDoesNotBeginFrameTest : public RHITest {
public:
    StreamerRenderGraphInvalidDoesNotBeginFrameTest()
    {
        type = RHITestType::Command;
        name = "streamer_render_graph_invalid_does_not_begin_frame";
    }

    RHITestResult run(RHITestContext& context) override
    {
        registerStreamerGraphPass();

        render::Queue* computeQueue = context.device.getQueue(render::QueueType::Compute);
        if (computeQueue == nullptr) {
            return RHITestResult::skip("device has no compute queue");
        }

        render::RenderGraph graph;
        graph.setName("StreamerInvalidQueues");
        graph.addNode("StreamerCrossQueueSourcePass", "Source");
        graph.addNode("StreamerCrossQueueSinkPass", "Sink");
        graph.addEdge("Source.data", "Sink.data");
        graph.markOutput("Sink.copied");

        render::RenderGraphExecutor executor;
        std::string log;
        render::Result<> result = executor.compile(context.device, graph, 1, 1, log);
        if (!result) {
            return RHITestResult::fail(std::string("RenderGraphExecutor::compile returned ") + toString(result) + ": " + log);
        }

        const render::RenderGraphStreamingStats before = executor.streamingStats();
        result = executor.execute(render::RenderGraphSubmitDesc{
            .graphicsQueue = nullptr,
            .computeQueue = computeQueue,
        });
        if (!render::hasError(result, render::Error::InvalidArgument)) {
            return RHITestResult::fail(
                std::string("expected InvalidArgument for missing graphics queue, got ") +
                toString(result));
        }

        const render::RenderGraphStreamingStats& after = executor.streamingStats();
        if (after.frameIndex != before.frameIndex ||
            after.streamer.frameIndex != before.streamer.frameIndex ||
            after.flushCount != before.flushCount ||
            after.transferCount != before.transferCount) {
            return RHITestResult::fail("invalid submit started or mutated the RenderGraph streaming frame");
        }
        return RHITestResult::pass();
    }
};

class StreamerMeshletResidencyUploadTest : public RHITest {
public:
    StreamerMeshletResidencyUploadTest()
    {
        type = RHITestType::Command;
        name = "streamer_meshlet_residency_upload";
    }

    RHITestResult run(RHITestContext& context) override
    {
        const std::filesystem::path streamAssetPath = context.outputDirectory / "streamer_residency.meshstream.bin";
        scene::MeshletStreamAsset asset;
        RHITestResult build = buildBunnyStreamAssetForTest(
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
            return RHITestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        render::MeshletStreamResidencyStats sparseStats = residency.stats();
        if (sparseStats.pageCount != asset.pageCount() ||
            sparseStats.trackedPageCount != 0 ||
            residency.trackedPageCount() != 0) {
            return RHITestResult::fail("residency initialization eagerly tracked unloaded scene pages");
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RHITestResult::fail("lockFallbackPages failed: " + reason);
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
            return RHITestResult::fail("fallback lock did not populate active/storage residency tables");
        }

        std::unique_ptr<render::Streamer> streamer;
        render::Result<> result = createStreamer(context.device, makeTestStreamerDesc()).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
        if (!result || streamer == nullptr) {
            return RHITestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = context.device.createBuffer(render::BufferDesc{
                .size = residency.pageBufferSize(),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            }).transform([&](auto rhiValue) { pageBuffer = std::move(rhiValue); });
        if (!result || pageBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer(pageBuffer) returned ") + toString(result));
        }

        residency.beginFrame();
        if (fallbackPages.empty() || residency.queuedUploadCount() == 0) {
            return RHITestResult::fail("lockFallbackPages did not queue fallback uploads");
        }
        const uint32_t pageIndex = fallbackPages.front();
        std::vector<render::StreamPageTableEntry> initialTable(asset.pageCount());
        residency.buildInitialPageTable(initialTable);
        if (render::streamPageTableDeviceOffset(initialTable[pageIndex]) !=
                render::kInvalidStreamDeviceOffsetBytes ||
            render::streamPageTableState(initialTable[pageIndex]) !=
                render::MeshletStreamPageResidencyState::Unloaded) {
            return RHITestResult::fail("initial stream page table entry did not encode missing fallback page");
        }
        if (asset.pages()[pageIndex].compressionMode !=
            static_cast<uint32_t>(scene::MeshletStreamPayloadCompression::ByteRle)) {
            return RHITestResult::fail("compressed streamasset did not preserve ByteRle page metadata");
        }
        residency.clearPendingPatches();

        const bool alreadyResident = residency.requestPage(pageIndex);
        if (alreadyResident || residency.queuedUploadCount() == 0) {
            return RHITestResult::fail("fallback page was resident before upload");
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
            return RHITestResult::fail("asynchronous page load did not schedule exactly one upload");
        }
        if (residency.pageState(pageIndex) != render::MeshletStreamPageResidencyState::PendingUpload) {
            return RHITestResult::fail("uploaded page did not enter PendingUpload state");
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
            return RHITestResult::fail("pending upload did not update pending table or upload stats");
        }
        std::span<const render::StreamPageTablePatch> patches = residency.pendingPatches();
        if (patches.size() != 1 ||
            patches[0].pageId != pageIndex ||
            render::streamPageTablePatchDeviceOffset(patches[0]) ==
                render::kInvalidStreamDeviceOffsetBytes ||
            render::streamPageTablePatchState(patches[0]) !=
                render::MeshletStreamPageResidencyState::PendingUpload) {
            return RHITestResult::fail("pending upload did not produce expected page table patch");
        }
        residency.clearPendingPatches();

        std::unique_ptr<render::CommandPool> commandPool;
        std::unique_ptr<render::CommandBuffer> commandBuffer;
        std::unique_ptr<render::Fence> fence;
        RHITestResult setup = createCommandResources(
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
            return RHITestResult::fail(std::string("CommandBuffer::begin returned ") + toString(result));
        }
        render::BufferBarrierDesc toTransfer{
            .buffer = pageBuffer.get(),
            .before = {},
            .after = {render::PipelineStageBits::Transfer, render::AccessBits::TransferWrite},
            .range = {.offset = 0, .size = pageBuffer->desc().size},
        };
        if (auto commandResult = commandBuffer->synchronize(render::BarrierDesc{
            .buffers = {&toTransfer, 1},
        }); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
        if (auto commandResult = streamer->copyStreamedData(*commandBuffer); !commandResult) { return RHITestResult::fail(std::string("copyStreamedData failed: ") + render::resultToString(commandResult)); }
        result = commandBuffer->end();
        if (!result) {
            return RHITestResult::fail(std::string("CommandBuffer::end returned ") + toString(result));
        }

        RHITestResult submit = submitAndWait(context.graphicsQueue, *commandBuffer, *fence);
        streamer->endFrame();
        if (!submit.passed) {
            return submit;
        }

        residency.beginFrame();
        if (residency.pageResident(pageIndex)) {
            return RHITestResult::fail("page became resident before queued frame delay elapsed");
        }
        if (!residency.newlyResidentPages().empty() || !residency.newlyUnloadedPages().empty()) {
            return RHITestResult::fail("residency reported a lifecycle transition before upload completion");
        }
        stats = residency.stats();
        if (stats.pendingPageCount != 1 ||
            stats.residentPageCount != 0 ||
            stats.queuedStorageTaskCount != 1 ||
            stats.availableStorageTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            stats.queuedUpdateTaskCount != 0 ||
            stats.availableUpdateTaskCount != render::kStreamingMaxActiveTasks - 1u ||
            stats.oldestPendingAge != 1) {
            return RHITestResult::fail("pending table age did not advance while upload was delayed");
        }
        if (!residency.pendingPatches().empty()) {
            return RHITestResult::fail("residency produced a patch before pending upload completed");
        }
        residency.beginFrame();
        if (residency.pageResident(pageIndex)) {
            return RHITestResult::fail("page became resident before queued update task elapsed");
        }
        if (!residency.newlyResidentPages().empty() || !residency.newlyUnloadedPages().empty()) {
            return RHITestResult::fail("residency reported a lifecycle transition before the update task completed");
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
            return RHITestResult::fail("storage completion did not queue a resident update task");
        }
        if (!residency.pendingPatches().empty()) {
            return RHITestResult::fail("storage completion produced a resident patch before update task completed");
        }
        residency.beginFrame();
        if (!residency.pageResident(pageIndex)) {
            return RHITestResult::fail("page did not become resident after queued update task elapsed");
        }
        if (residency.newlyResidentPages().size() != 1 ||
            residency.newlyResidentPages().front() != pageIndex ||
            !residency.newlyUnloadedPages().empty()) {
            return RHITestResult::fail("completed upload did not report the newly resident page");
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
            return RHITestResult::fail("resident upload did not update resident table or completion stats");
        }
        patches = residency.pendingPatches();
        if (patches.size() != 1 ||
            patches[0].pageId != pageIndex ||
            render::streamPageTablePatchDeviceOffset(patches[0]) ==
                render::kInvalidStreamDeviceOffsetBytes ||
            render::streamPageTablePatchState(patches[0]) !=
                render::MeshletStreamPageResidencyState::LockedFallback) {
            return RHITestResult::fail("resident fallback did not produce expected page table patch");
        }

        const uint64_t deviceOffset = residency.deviceOffsetForPage(pageIndex);
        if (deviceOffset == UINT64_MAX) {
            return RHITestResult::fail("resident page has no device offset");
        }

        std::vector<uint8_t> actual(static_cast<size_t>(asset.pages()[pageIndex].uncompressedSize));
        pageBuffer->invalidate({deviceOffset, actual.size()});
        void* mapped = pageBuffer->map();
        if (mapped == nullptr) {
            return RHITestResult::fail("page buffer did not map");
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
            return RHITestResult::fail("failed to decode compressed expected streamasset payload: " + decodeReason);
        }
        if (actual.size() != expected.size() ||
            std::memcmp(actual.data(), expected.data(), expected.size()) != 0) {
            return RHITestResult::fail("streamed page payload bytes did not match decoded streamasset payload");
        }
        return RHITestResult::pass();
    }
};

class MeshletStreamCLASPagePlanTest : public RHITest {
public:
    MeshletStreamCLASPagePlanTest()
    {
        type = RHITestType::Validation;
        name = "meshlet_stream_clas_page_plan";
    }

    RHITestResult run(RHITestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RHITestResult build = buildBunnyStreamAssetForTest(
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
            return RHITestResult::fail("buildMeshletStreamPageClusterOffsets failed: " + reason);
        }
        if (pageClusterOffsets.size() != static_cast<size_t>(asset.pageCount()) + 1u ||
            pageClusterOffsets.back() != clusterCount ||
            clusterCount == 0) {
            return RHITestResult::fail("CLAS page cluster offsets did not cover the streamasset");
        }

        const std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        if (fallbackPages.empty()) {
            return RHITestResult::fail("streamasset has no fallback page for CLAS planning");
        }
        const uint32_t pageIndex = fallbackPages.front();
        render::MeshletStreamCLASPagePlan plan;
        if (!render::buildMeshletStreamClasPagePlan(
                asset,
                pageIndex,
                pageClusterOffsets[pageIndex],
                plan,
                reason)) {
            return RHITestResult::fail("buildMeshletStreamClasPagePlan failed: " + reason);
        }

        const scene::MeshletStreamPageInfo& page = asset.pages()[pageIndex];
        if (plan.pageIndex != pageIndex ||
            plan.firstClusterId != pageClusterOffsets[pageIndex] ||
            plan.primitiveIndex != page.primitiveIndex ||
            plan.lodLevel != page.lodLevel ||
            plan.payloadByteSize != page.uncompressedSize ||
            plan.clusters.size() != page.clusterCount) {
            return RHITestResult::fail("CLAS page plan did not preserve streamasset page metadata");
        }

        for (uint32_t clusterIndex = 0; clusterIndex < plan.clusters.size(); ++clusterIndex) {
            const render::MeshletStreamCLASClusterInput& cluster = plan.clusters[clusterIndex];
            if (cluster.clusterId != pageClusterOffsets[pageIndex] + clusterIndex ||
                cluster.pageIndex != pageIndex ||
                cluster.clusterIndex != clusterIndex ||
                cluster.primitiveIndex != page.primitiveIndex ||
                cluster.vertexCount == 0 ||
                cluster.triangleCount == 0 ||
                cluster.vertexOffsetBytes >= page.uncompressedSize ||
                cluster.triangleOffsetBytes >= page.uncompressedSize) {
                return RHITestResult::fail("CLAS page plan contains an invalid cluster build input");
            }
        }
        return RHITestResult::pass();
    }
};

class StreamerMeshletResidencyGPURequestPatchTest : public RHITest {
public:
    StreamerMeshletResidencyGPURequestPatchTest()
    {
        type = RHITestType::Command;
        name = "streamer_meshlet_residency_gpu_request_patches";
    }

    RHITestResult run(RHITestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RHITestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_gpu_request.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RHITestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
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
            return RHITestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RHITestResult::fail("lockFallbackPages failed: " + reason);
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
            return RHITestResult::fail("consumeGpuRequests did not deduplicate and schedule page ids");
        }
        const std::span<const uint32_t> requestedPages = residency.requestedPages();
        if (requestedPages.size() != 2 ||
            std::find(requestedPages.begin(), requestedPages.end(), firstPage) == requestedPages.end() ||
            std::find(requestedPages.begin(), requestedPages.end(), secondPage) == requestedPages.end()) {
            return RHITestResult::fail("request table did not preserve unique GPU-requested page ids");
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
            return RHITestResult::fail("GPU request readback did not queue an isolated request task");
        }
        if (residency.pageAllocated(firstPage) ||
            residency.pageAllocated(secondPage) ||
            !residency.pendingPatches().empty()) {
            return RHITestResult::fail("queued GPU request task modified residency before beginFrame consumed it");
        }

        residency.beginFrame();
        const std::span<const uint32_t> consumedRequestPages = residency.requestedPages();
        if (consumedRequestPages.size() != 2 ||
            std::find(consumedRequestPages.begin(), consumedRequestPages.end(), firstPage) == consumedRequestPages.end() ||
            std::find(consumedRequestPages.begin(), consumedRequestPages.end(), secondPage) == consumedRequestPages.end()) {
            return RHITestResult::fail("request task did not preserve unique page ids when consumed");
        }
        if (residency.pageAllocated(firstPage)) {
            return RHITestResult::fail("older requested page received storage despite latest-page pressure");
        }
        if (!residency.pageAllocated(secondPage)) {
            return RHITestResult::fail("latest requested page did not receive the single streamable allocation");
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
            return RHITestResult::fail("active/request/storage stats did not track GPU request pressure");
        }

        if (!residency.pendingPatches().empty()) {
            return RHITestResult::fail("budget-limited request emitted an unexpected eviction patch");
        }
        return RHITestResult::pass();
    }
};

class StreamerMeshletResidencyLatestGPURequestTest : public RHITest {
public:
    StreamerMeshletResidencyLatestGPURequestTest()
    {
        type = RHITestType::Command;
        name = "streamer_meshlet_residency_latest_gpu_request";
    }

    RHITestResult run(RHITestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RHITestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_latest_gpu_request.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RHITestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
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
            return RHITestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RHITestResult::fail("lockFallbackPages failed: " + reason);
        }
        residency.clearPendingPatches();

        const uint32_t staleScheduled = residency.consumeGpuRequests(std::span<const uint32_t>(&stalePage, 1));
        const uint32_t latestScheduled = residency.consumeGpuRequests(std::span<const uint32_t>(&latestPage, 1));
        if (staleScheduled != 1 || latestScheduled != 1) {
            return RHITestResult::fail("consumeGpuRequests did not schedule two request tasks");
        }

        render::MeshletStreamResidencyStats stats = residency.stats();
        if (stats.queuedRequestTaskCount != 2 ||
            stats.availableRequestTaskCount != render::kStreamingMaxActiveTasks - 2u ||
            stats.frameScheduledRequestTaskCount != 2 ||
            stats.frameUniqueGpuRequestCount != 2 ||
            stats.frameConsumedGpuRequestCount != 0) {
            return RHITestResult::fail("multiple GPU request readbacks were not queued as separate tasks");
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
            return RHITestResult::fail("request queue did not drop stale ready tasks and consume the latest request");
        }

        return RHITestResult::pass();
    }
};

class StreamerMeshletScreenPriorityTest : public RHITest {
public:
    StreamerMeshletScreenPriorityTest() { type = RHITestType::Command; name = "streamer_meshlet_screen_priority"; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "screen_priority.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (roots.empty() || pages.size() < 2) { return RHITestResult::skip("Needs two streamable pages"); }
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
        if (!reason.empty()) { return RHITestResult::fail(reason); }
        const uint32_t two[] = {large, small};
        const float equal[] = {100.f, 100.f};
        reason = verify(two, equal, small);
        if (!reason.empty()) { return RHITestResult::fail("Benefit per byte: " + reason); }
        const float invalid[] = {std::numeric_limits<float>::quiet_NaN(), 100.f};
        reason = verify(two, invalid, small);
        if (!reason.empty()) { return RHITestResult::fail("Non-finite feedback: " + reason); }
        const float shortScores[] = {100.f};
        reason = verify(two, shortScores, large);
        if (!reason.empty()) { return RHITestResult::fail("Short feedback: " + reason); }
        return RHITestResult::pass("Shared-page maximum, benefit per byte, roots, NaN and short feedback");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletScreenPriorityTest);

class StreamerMeshletPrefetchTest final : public RHITest {
public:
    StreamerMeshletPrefetchTest() { type = RHITestType::Command; name = "streamer_meshlet_prefetch_admission"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "prefetch.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        const auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 3) { return RHITestResult::skip("Needs three streamable pages"); }
        const uint64_t bytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages);
        std::string reason;
        MeshletStreamResidencyManager legacy, immediate;
        if (!legacy.initialize({.asset = &asset, .maxResidentBytes = bytes}, reason) ||
            !immediate.initialize({.asset = &asset, .maxResidentBytes = bytes,
                .measurePageLatency = true, .immediateGpuRequests = true}, reason)) { return RHITestResult::fail(reason); }
        legacy.beginFrame(); immediate.beginFrame();
        const uint32_t current[] = {pages[0]};
        (void)legacy.consumeGpuRequests({.loadPageIds = current, .frameIndex = 1});
        (void)immediate.consumeGpuRequests({.loadPageIds = current, .frameIndex = 1});
        if (legacy.pageAllocated(pages[0]) || !immediate.pageAllocated(pages[0])) {
            return RHITestResult::fail("Immediate requests did not remove exactly the admission frame");
        }
        legacy.beginFrame();
        if (!legacy.pageAllocated(pages[0])) { return RHITestResult::fail("Legacy admission changed"); }
        const uint32_t duplicate[] = {pages[1] | kStreamPrefetchPageTag, pages[1]};
        (void)immediate.consumeGpuRequests({.loadPageIds = duplicate, .frameIndex = 1, .taggedPrefetchRequests = true});
        if (!immediate.pageAllocated(pages[1]) || immediate.stats().totalPrefetchAdmitted != 0 ||
            immediate.latencySnapshot().pendingDemand != 2 || immediate.latencySnapshot().pendingPrefetch != 0) {
            return RHITestResult::fail("Actual demand did not win a tagged duplicate without priorities");
        }
        // One page of speculation fits, but the next cannot evict it or use the
        // reserved quarter. Current demand can use that remaining capacity.
        MeshletStreamResidencyManager bounded;
        const uint64_t largest = std::max(pageStorageBytes(asset, pages[0]), pageStorageBytes(asset, pages[1]));
        if (!bounded.initialize({.asset = &asset, .maxResidentBytes = largest * 4,
                .maxResidentPages = 2, .immediateGpuRequests = true}, reason)) { return RHITestResult::fail(reason); }
        bounded.beginFrame();
        const uint32_t forecasts[] = {pages[0] | kStreamPrefetchPageTag, pages[1] | kStreamPrefetchPageTag};
        (void)bounded.consumeGpuRequests({.loadPageIds = forecasts, .taggedPrefetchRequests = true});
        if (bounded.stats().totalPrefetchAdmitted != 1 || bounded.stats().totalPrefetchDeferred != 1 ||
            bounded.stats().totalEvictedPageCount != 0) {
            return RHITestResult::fail("Speculative admission displaced the demand reserve");
        }
        const uint32_t missing[] = {bounded.pageAllocated(pages[0]) ? pages[1] : pages[0]};
        (void)bounded.consumeGpuRequests({.loadPageIds = missing});
        if (!bounded.pageAllocated(pages[0]) || !bounded.pageAllocated(pages[1])) {
            return RHITestResult::fail("Actual demand could not use reserved capacity");
        }
        // Resident speculation must survive old/truncated feedback and count
        // exactly one hit when a complete eligible view first needs it.
        const uint32_t prefetched[] = {missing[0] == pages[0] ? pages[1] : pages[0]};
        std::unique_ptr<Streamer> uploader;
        auto uploadResult = createStreamer(context.device, makeTestStreamerDesc(2 * asset.maxPagePayloadBytes() + 4096))
            .transform([&](auto value) { uploader = std::move(value); });
        if (!uploadResult) { return RHITestResult::fail(toString(uploadResult)); }
        std::unique_ptr<Buffer> destination;
        uploadResult = context.device.createBuffer({.size = bounded.pageBufferSize(),
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
            .transform([&](auto value) { destination = std::move(value); });
        if (!uploadResult) { return RHITestResult::fail(toString(uploadResult)); }
        if (bounded.processUploads(*uploader, *destination, 2) != 2) {
            return RHITestResult::fail("Cannot upload prefetch cohort");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { bounded.beginFrame(); }
        (void)bounded.consumeGpuRequests({.frameIndex = 1, .residentDemandFeedback = true});
        (void)bounded.consumeGpuRequests({.unloadRequestCounter = 1, .unloadOverflowCounter = 1,
            .residentDemandFeedback = true});
        (void)bounded.consumeGpuRequests({.unloadPageIds = prefetched, .unloadRequestCounter = 1,
            .residentDemandFeedback = true});
        if (bounded.stats().totalPrefetchUsed != 0) {
            return RHITestResult::fail("Old, truncated or unused feedback invented a prefetch hit");
        }
        (void)bounded.consumeGpuRequests({.residentDemandFeedback = true});
        (void)bounded.consumeGpuRequests({.residentDemandFeedback = true});
        if (bounded.stats().totalPrefetchUsed != 1) {
            return RHITestResult::fail("Resident prefetch cohort lost or repeated its hit");
        }
        MeshletStreamResidencyManager queued;
        if (!queued.initialize({.asset = &asset, .maxResidentBytes = bytes,
                .pageLoadConcurrency = 1, .maxPageLoadsInFlight = 4, .immediateGpuRequests = true}, reason)) {
            return RHITestResult::fail(reason);
        }
        const uint32_t queuedForecasts[] = {pages[0] | kStreamPrefetchPageTag,
            pages[1] | kStreamPrefetchPageTag, pages[2] | kStreamPrefetchPageTag};
        (void)queued.consumeGpuRequests({.loadPageIds = queuedForecasts, .taggedPrefetchRequests = true});
        if (queued.queuedUploadCount() != 1 || queued.availablePrefetchRequests() != 0 ||
            queued.stats().totalPrefetchDeferred != 2) {
            return RHITestResult::fail("Unissued speculative I/O occupied the demand queue reserve");
        }
        (void)queued.consumeGpuRequests({.loadPageIds = std::span(pages).first(3)});
        if (queued.queuedUploadCount() != 3 || queued.stats().totalPrefetchUsed != 1) {
            return RHITestResult::fail("Demand could not promote or bypass queued speculation");
        }
        return RHITestResult::pass("Immediate/legacy admission, tagged promotion, memory and queue reserves, no speculative eviction");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletPrefetchTest);

class StreamerMeshletLatencyTest final : public RHITest {
public:
    StreamerMeshletLatencyTest() { type = RHITestType::Validation; name = "streamer_meshlet_latency"; }
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        MeshletStreamLatencyHistogram histogram;
        if (histogram.summary().count != 0) { return RHITestResult::fail("Empty latency histogram"); }
        for (uint32_t i = 0; i < 99; ++i) { histogram.observe(999); }
        histogram.observe(5000123);
        const auto sample = histogram.summary();
        if (sample.count != 100 || sample.p50 != 1 || sample.p99 != 1 || sample.maximum != 5000.123) {
            return RHITestResult::fail("Latency percentile or overflow bin lost the tail");
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
            return RHITestResult::fail("Latency duplicate, first demand frame, promotion or retirement accounting");
        }
        return RHITestResult::pass("Histogram tail, source frames, promotion, duplicates and abandoned requests");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletLatencyTest);

// Verify eligibility at feedback time, independent of immediate/deferred admission.
class StreamerMeshletLatencyEligibilityTest final : public RHITest {
public:
    StreamerMeshletLatencyEligibilityTest() { type = RHITestType::Command; name = "streamer_meshlet_latency_eligibility"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "latency_eligibility.meshstream.bin", asset);
        if (!built.passed) { return built; }
        if (asset.pageCount() < 4) { return RHITestResult::fail("Need four latency lifecycle pages"); }
        const uint64_t capacity = alignStreamStorageBytes(asset.maxPagePayloadBytes()) * 4u;
        std::unique_ptr<Streamer> streamer;
        std::unique_ptr<Buffer> destination;
        if (!createStreamer(context.device, makeTestStreamerDesc(capacity + 4096)).transform([&](auto rhiValue) { streamer = std::move(rhiValue); }) ||
            !context.device.createBuffer({.size = capacity, .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { destination = std::move(rhiValue); })) {
            return RHITestResult::fail("Cannot create latency lifecycle upload resources");
        }
        MeshletStreamResidencyManager residency;
        std::string reason;
        for (const bool immediate : {false, true}) {
            const MeshletStreamResidencyDesc desc{.asset = &asset, .maxResidentBytes = capacity,
                .maxResidentPages = 3, .queuedFrameCount = 1, .unloadDelayFrames = 2,
                .measurePageLatency = true, .immediateGpuRequests = immediate, .completionDrivenUploads = false};
            if (!residency.initialize(desc, reason)) { return RHITestResult::fail(reason); }
            const auto counts = [&](uint64_t feedback, uint32_t demand, uint32_t prefetch) {
                const auto value = residency.latencySnapshot();
                return value.enabled && value.milliseconds[size_t(MeshletStreamLatencyStage::Feedback)].count == feedback &&
                    value.pendingDemand == demand && value.pendingPrefetch == prefetch;
            };
            residency.beginFrame();
            const uint32_t root = 0, promotedRoot = 1, detail = 2;
            if (!residency.lockFallbackPages(std::span(&root, 1), reason)) { return RHITestResult::fail(reason); }
            (void)residency.requestPage(promotedRoot);
            const uint32_t forecast = promotedRoot | kStreamPrefetchPageTag;
            (void)residency.consumeGpuRequests({.loadPageIds = std::span(&forecast, 1), .taggedPrefetchRequests = true});
            if (!counts(1, 0, 1)) { return RHITestResult::fail("Queued prefetch not tracked"); }
            if (!residency.lockFallbackPages(std::span(&promotedRoot, 1), reason)) { return RHITestResult::fail(reason); }
            (void)residency.consumeGpuRequests(std::span(&promotedRoot, 1));
            if (!counts(1, 0, 1)) { return RHITestResult::fail("Fallback pin without state change promoted latency demand"); }
            (void)residency.requestPage(detail);
            const uint32_t mixed[] = {root, promotedRoot, detail, 3, detail, UINT32_MAX};
            (void)residency.consumeGpuRequests(mixed);
            if (!counts(3, 2, 1)) { return RHITestResult::fail("Fallback exclusion, duplicate merge or blocked latency changed"); }
            // Use the existing frame-delayed CPU residency protocol here; real GPU
            // submission/cancellation is covered by streamer_meshlet_upload_completion.
            if (residency.processUploads(*streamer, *destination, 3) != 3) {
                return RHITestResult::fail("Cannot prepare latency lifecycle residents");
            }
            for (uint32_t i = 0; i < 4; ++i) { residency.beginFrame(); }
            if (!residency.pageResident(detail) || !counts(3, 1, 0)) {
                return RHITestResult::fail("Completion did not publish residency and retire latency");
            }
            (void)residency.consumeGpuRequests(mixed);
            if (!counts(3, 1, 0)) { return RHITestResult::fail("Resident feedback created a false latency request"); }
            if (!residency.unloadPage(detail)) { return RHITestResult::fail("Cannot schedule latency lifecycle unload"); }
            (void)residency.consumeGpuRequests(std::span(&detail, 1));
            if (!counts(4, 2, 0)) { return RHITestResult::fail("Pending unload remained excluded from demand tracking"); }
            residency.beginFrame(); residency.beginFrame();
            if (residency.latencySnapshot().abandonedDemand != 1) {
                return RHITestResult::fail("Retirement did not abandon pending unload demand exactly once");
            }
            (void)residency.consumeGpuRequests(std::span(&detail, 1));
            if (!counts(5, 2, 0)) { return RHITestResult::fail("Erased/reloaded page retained stale residency eligibility"); }
            if (!residency.initialize(desc, reason)) { return RHITestResult::fail(reason); }
            residency.beginFrame();
            const uint32_t newScene[] = {root, promotedRoot, detail};
            (void)residency.consumeGpuRequests(newScene);
            if (!counts(3, 3, 0)) { return RHITestResult::fail("Reset retained exclusions for reused page IDs"); }
            auto disabled = desc; disabled.measurePageLatency = false;
            if (!residency.initialize(disabled, reason)) { return RHITestResult::fail(reason); }
            residency.beginFrame();
            if (!residency.lockFallbackPages(std::span(&root, 1), reason)) { return RHITestResult::fail(reason); }
            (void)residency.consumeGpuRequests(newScene);
            if (residency.latencySnapshot().enabled) { return RHITestResult::fail("Disabled latency tracking became enabled"); }
        }
        return RHITestResult::pass("Fallback pin, completion, pending unload, blocked demand, erase/reload, reset and deferred batches");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletLatencyEligibilityTest);

// Compare against the former full-scan semantics using deterministic timestamps.
// Exercise sparse IDs, wheel wrap, frame gaps, slot reuse, promotion and in-flight expiry.
class StreamerMeshletLatencyLifecycleTest final : public RHITest {
public:
    StreamerMeshletLatencyLifecycleTest() { type = RHITestType::Validation; name = "streamer_meshlet_latency_lifecycle"; }
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        for (const uint64_t grace : {2u, 8u, 300u}) {
            MeshletStreamLatencyTracker tracker(grace);
            std::unordered_map<uint32_t, MeshletStreamLatencyTracker::Request> reference;
            std::array<MeshletStreamLatencyHistogram, size_t(MeshletStreamLatencyStage::Count)> histograms{};
            MeshletStreamLatencyHistogram demandFrames;
            uint64_t abandonedDemand = 0, abandonedPrefetch = 0;
            uint64_t frame = 0;
            for (uint32_t iteration = 1; iteration <= 1200; ++iteration) {
                frame += iteration % 199 == 0 ? 513 : 1;
                const uint64_t now = (frame + 1000) * 1000;
                tracker.frameTimes[frame % 256] = {frame, now};
                const auto inFlight = [frame](uint32_t id) { return ((frame / 7 + id) % 11) < 3; };
                tracker.expire(frame, inFlight);
                for (auto it = reference.begin(); it != reference.end();) {
                    if (!inFlight(it->first) && frame - it->second.lastSeenFrame > grace) {
                        if (it->second.demandTime) { ++abandonedDemand; } else { ++abandonedPrefetch; }
                        it = reference.erase(it);
                    } else { ++it; }
                }
                for (uint32_t n = 0; n < 80; ++n) {
                    const uint32_t page = ((iteration * 17 + n * 7) % 193) * 1027;
                    const uint64_t source = frame > (n % 5) ? frame - n % 5 : 0;
                    const bool prefetch = (iteration + n) % 3 == 0;
                    const auto& stamp = tracker.frameTimes[source % 256];
                    const uint64_t start = stamp[0] == source && stamp[1] ? stamp[1] : now;
                    auto& expected = reference[page];
                    if (!expected.firstTime) {
                        expected.firstTime = start; expected.feedbackTime = now;
                        histograms[size_t(MeshletStreamLatencyStage::Feedback)].observe(now - start);
                    }
                    expected.lastSeenFrame = frame;
                    if (!prefetch && !expected.demandTime) { expected.demandTime = start; expected.demandFrame = source; }
                    tracker.request(page, source, frame, prefetch, now);
                    if (n % 9 == 0) { tracker.request(page, source, frame, prefetch, now); }
                }
                for (uint32_t n = 0; n < 15; ++n) {
                    const uint32_t page = ((iteration * 13 + n * 11) % 193) * 1027;
                    const auto it = reference.find(page);
                    if (n % 2) {
                        tracker.complete(page, frame, now);
                        if (it != reference.end()) {
                            const auto& request = it->second;
                            if (request.demandTime) {
                                histograms[size_t(MeshletStreamLatencyStage::DemandToDrawable)].observe(now - request.demandTime);
                                demandFrames.observe((frame - request.demandFrame) * 1000);
                            } else { histograms[size_t(MeshletStreamLatencyStage::PrefetchToDrawable)].observe(now - request.firstTime); }
                        }
                    } else {
                        tracker.abandon(page);
                        if (it != reference.end()) {
                            if (it->second.demandTime) { ++abandonedDemand; } else { ++abandonedPrefetch; }
                        }
                    }
                    if (it != reference.end()) { reference.erase(it); }
                }
                uint32_t pendingDemand = 0;
                double oldest = 0;
                for (uint32_t id = 0; id < 193; ++id) {
                    const uint32_t page = id * 1027;
                    const auto found = reference.find(page);
                    const auto actual = tracker.find(page);
                    if ((actual == nullptr) != (found == reference.end())) { return RHITestResult::fail("Expiry/slot reuse differs from full scan"); }
                    if (!actual) { continue; }
                    const auto& expected = found->second;
                    if (actual->firstTime != expected.firstTime || actual->feedbackTime != expected.feedbackTime ||
                        actual->lastSeenFrame != expected.lastSeenFrame || actual->demandTime != expected.demandTime ||
                        actual->demandFrame != expected.demandFrame) { return RHITestResult::fail("Request timestamp/promotion changed"); }
                    if (expected.demandTime) { ++pendingDemand; oldest = std::max(oldest, double(now - expected.demandTime) / 1000); }
                }
                const auto snapshot = tracker.snapshot(now);
                if (snapshot.pendingDemand != pendingDemand || snapshot.pendingPrefetch != reference.size() - pendingDemand ||
                    snapshot.abandonedDemand != abandonedDemand || snapshot.abandonedPrefetch != abandonedPrefetch ||
                    snapshot.oldestPendingDemandMilliseconds != oldest || tracker.demandFrames.bins != demandFrames.bins) {
                    return RHITestResult::fail("Pending, abandoned or latency frame statistics changed");
                }
                for (size_t stage = 0; stage < histograms.size(); ++stage) {
                    const auto& a = tracker.stages[stage]; const auto& b = histograms[stage];
                    if (a.bins != b.bins || a.count != b.count || a.totalMicroseconds != b.totalMicroseconds ||
                        a.maximumMicroseconds != b.maximumMicroseconds) { return RHITestResult::fail("Latency histogram differs from reference"); }
                }
            }
        }
        // Continuously demanded, budget-blocked requests must not probe residency
        // for expiry or be forgotten. Once overdue, every item is checked exactly once.
        MeshletStreamLatencyTracker blocked(8);
        uint32_t queries = 0;
        for (uint64_t frame = 1; frame <= 300; ++frame) {
            blocked.expire(frame, [&](uint32_t) { ++queries; return false; });
            for (uint32_t page = 0; page < 14000; ++page) { blocked.request(page, frame, frame, false, frame * 1000); }
        }
        if (queries || blocked.snapshot(300000).pendingDemand != 14000) { return RHITestResult::fail("Hot blocked demand was scanned or discarded"); }
        blocked.expire(309, [&](uint32_t) { ++queries; return false; });
        if (queries != 14000 || blocked.snapshot(309000).abandonedDemand != 14000) { return RHITestResult::fail("Due requests were not retired exactly once"); }
        return RHITestResult::pass("Full-scan reference equivalence, sparse IDs, expiry wraps/gaps, in-flight protection and bounded expiry work");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletLatencyLifecycleTest);

class StreamerMeshletResidencyGPURequestUnloadOverflowTest : public RHITest {
public:
    StreamerMeshletResidencyGPURequestUnloadOverflowTest()
    {
        type = RHITestType::Command;
        name = "streamer_meshlet_residency_gpu_request_unload_overflow";
    }

    RHITestResult run(RHITestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RHITestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_gpu_request_unload.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RHITestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
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
            return RHITestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RHITestResult::fail("lockFallbackPages failed: " + reason);
        }
        residency.clearPendingPatches();

        (void)residency.requestPage(unloadPage);
        if (!residency.pageAllocated(unloadPage)) {
            return RHITestResult::fail("test setup did not allocate streamable page storage");
        }

        const std::array<uint32_t, 2> loadRequests = {loadPage, loadPage};
        const std::array<uint32_t, 2> unloadRequests = {unloadPage, unloadPage};
        const uint32_t scheduled = residency.consumeGpuRequests(render::StreamGPURequestBatch{
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
            return RHITestResult::fail("load/unload GPU request batch did not schedule unique page ids");
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
            return RHITestResult::fail("GPU request load/unload overflow stats were not tracked");
        }
        if (residency.requestedPages().size() != 1 ||
            residency.requestedPages().front() != loadPage ||
            residency.unloadRequestedPages().size() != 1 ||
            residency.unloadRequestedPages().front() != unloadPage) {
            return RHITestResult::fail("GPU request batch did not preserve unique load/unload page ids");
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
            return RHITestResult::fail("GPU unload request did not enter delayed-free state before consuming loads");
        }
        if (residency.requestedPages().size() != 1 ||
            residency.requestedPages().front() != loadPage ||
            residency.unloadRequestedPages().size() != 1 ||
            residency.unloadRequestedPages().front() != unloadPage) {
            return RHITestResult::fail("completed request task did not expose consumed load/unload ids");
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
            return RHITestResult::fail("GPU unload request did not emit a pending-unload page table patch");
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
            return RHITestResult::fail("delayed unload task did not free resident page storage");
        }
        if (residency.newlyUnloadedPages().size() != 1 ||
            residency.newlyUnloadedPages().front() != unloadPage ||
            !residency.newlyResidentPages().empty()) {
            return RHITestResult::fail("completed unload did not report the newly unloaded page");
        }

        (void)residency.requestPage(loadPage);
        if (!residency.pageAllocated(loadPage) ||
            stats.frameCompletedUnloadCount != 1) {
            return RHITestResult::fail("load request did not acquire storage after delayed free completed");
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
            return RHITestResult::fail("delayed unload completion did not emit an unloaded page table patch");
        }

        return RHITestResult::pass();
    }
};

class StreamerMeshletResidencyEvictionDelayAgeTest : public RHITest {
public:
    StreamerMeshletResidencyEvictionDelayAgeTest()
    {
        type = RHITestType::Command;
        name = "streamer_meshlet_residency_eviction_delay_age";
    }

    RHITestResult run(RHITestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        RHITestResult build = buildBunnyStreamAssetForTest(
            context.outputDirectory / "streamer_residency_eviction_delay_age.meshstream.bin",
            asset);
        if (!build.passed) {
            return build;
        }

        std::vector<uint32_t> fallbackPages = fallbackPagesFor(asset);
        std::vector<uint32_t> streamablePages = nonFallbackPagesFor(asset, fallbackPages);
        if (fallbackPages.empty() || streamablePages.size() < 2) {
            return RHITestResult::skip("streamasset does not contain enough fallback/non-fallback pages");
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
            return RHITestResult::fail("locked fallback pages did not reject the resident page-count budget atomically");
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
            return RHITestResult::fail("MeshletStreamResidencyManager::initialize failed: " + reason);
        }
        if (!residency.lockFallbackPages(fallbackPages, reason)) {
            return RHITestResult::fail("lockFallbackPages failed: " + reason);
        }

        render::StreamerDesc streamerDesc = makeTestStreamerDesc(
            (static_cast<uint64_t>(fallbackPages.size()) + 1ull) * asset.maxPagePayloadBytes() + 4096ull);
        std::unique_ptr<render::Streamer> streamer;
        render::Result<> result = createStreamer(context.device, streamerDesc).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
        if (!result || streamer == nullptr) {
            return RHITestResult::fail(std::string("createStreamer returned ") + toString(result));
        }

        std::unique_ptr<render::Buffer> pageBuffer;
        result = context.device.createBuffer(render::BufferDesc{
                .size = residency.pageBufferSize(),
                .usage = render::BufferUsageBits::TransferDestination,
                .memoryLocation = render::MemoryLocation::HostReadback,
            }).transform([&](auto rhiValue) { pageBuffer = std::move(rhiValue); });
        if (!result || pageBuffer == nullptr) {
            return RHITestResult::fail(std::string("createBuffer(pageBuffer) returned ") + toString(result));
        }

        residency.beginFrame();
        (void)residency.requestPage(residentPage);
        const uint32_t uploadBudget = static_cast<uint32_t>(fallbackPages.size()) + 1u;
        if (residency.processUploads(*streamer, *pageBuffer, uploadBudget) != uploadBudget) {
            return RHITestResult::fail("processUploads did not schedule fallback and streamable uploads");
        }

        residency.beginFrame();
        residency.beginFrame();
        residency.beginFrame();
        if (!residency.pageResident(residentPage)) {
            return RHITestResult::fail("test setup did not make streamable page resident");
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
            return RHITestResult::fail("age filter did not reject eviction of a young resident page");
        }

        for (uint32_t retry = 0; retry < 10000; ++retry) { (void)residency.requestPage(requestedPage); }
        stats = residency.stats();
        if (stats.frameEvictionScanCount != 1 || stats.frameEvictionCandidateTests != stats.residentPageCount ||
            stats.frameAllocationDeferredCount != 10001 || stats.frameAllocationFailureCount != 1 ||
            stats.cpuWork.allocationAttempts != 1 || stats.cpuWork.budgetRetrySuppressed != 10000 ||
            stats.trackedPageCount != fallbackPages.size() + 1 || residency.pageAllocated(requestedPage)) {
            return RHITestResult::fail("budget pressure repeated an eviction scan or lost age protection");
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
            return RHITestResult::fail("eligible eviction did not schedule a delayed unload task");
        }

        residency.beginFrame();
        stats = residency.stats();
        if (residency.pageAllocated(residentPage) ||
            residency.pageState(residentPage) != render::MeshletStreamPageResidencyState::Unloaded ||
            stats.frameCompletedUnloadCount != 1 ||
            stats.frameDelayedFreeCount != 1 ||
            stats.freeResidentBytes != streamableBudgetBytes) {
            return RHITestResult::fail("delayed eviction did not free its storage on task completion");
        }

        (void)residency.requestPage(requestedPage);
        stats = residency.stats();
        if (!residency.pageAllocated(requestedPage) ||
            stats.activePageCount != fallbackPages.size() + 1u ||
            stats.usedSlotCount > stats.maxResidentPages ||
            stats.freeSlotCount != 0) {
            return RHITestResult::fail("request did not acquire storage after delayed eviction completed");
        }

        return RHITestResult::pass();
    }
};

class StreamerMeshletRequestSelectionTest final : public RHITest {
public:
    StreamerMeshletRequestSelectionTest()
    {
        type = RHITestType::Command;
        name = "streamer_meshlet_request_selection";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "RequestSelection.meshstream.bin", asset);
        if (!built.passed) { return built; }
        auto pages = nonFallbackPagesFor(asset, fallbackPagesFor(asset));
        if (pages.size() < 5) { return RHITestResult::fail("Need five request candidates"); }
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
            if (!residency.initialize(desc, reason)) { return RHITestResult::fail(reason); }
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
            const auto batch = StreamGPURequestBatch{.loadPageIds = ids, .frameIndex = 1,
                .loadPriorities = benefits, .taggedPrefetchRequests = true};
            if (residency.consumeGpuRequests(batch) != count) {
                return RHITestResult::fail("Request selection lost unique candidates");
            }
            for (size_t i = 0; i < expected.size(); ++i) {
                if (residency.pageAllocated(expected[i].second) != (i < 3)) {
                    return RHITestResult::fail("Heap selection differs from full priority sort");
                }
            }
            auto stats = residency.stats();
            if (stats.cpuWork.admissionCalls != 4 || stats.cpuWork.admissionPriorityPops != 4 ||
                stats.cpuWork.requestDuplicatesMerged != count || stats.frameGpuInvalidRequestCount != 1 ||
                stats.totalPrefetchAdmitted != 0 || residency.latencySnapshot().pendingPrefetch != 0) {
                return RHITestResult::fail("Demand merge or bounded priority selection did not hold");
            }
            // Repeat while capacity is blocked: allocated work stays alive,
            // missing work remains demanded, neither requires a heap pop/call.
            std::reverse(ids.begin(), ids.end());
            std::reverse(benefits.begin(), benefits.end());
            (void)residency.consumeGpuRequests(batch);
            stats = residency.stats();
            if (stats.cpuWork.admissionCalls != 4 || stats.cpuWork.admissionPriorityPops != 4 ||
                stats.frameUniqueGpuRequestCount != count * 2 || residency.latencySnapshot().pendingDemand != count) {
                return RHITestResult::fail("Repeated batch re-admitted blocked or queued pages, or lost latency demand");
            }
            residency.beginFrame();
            (void)residency.consumeGpuRequests(batch);
            if (residency.stats().cpuWork.admissionCalls != 1 || residency.queuedUploadCount() != 3) {
                return RHITestResult::fail("Next frame lost queued demand or failed to refresh capacity eligibility");
            }
            if (residency.stats().cpuWork.priorityRecomputed != 0 || residency.stats().cpuWork.priorityReused == 0) {
                return RHITestResult::fail("Unchanged eligible priorities were not reused across feedback batches");
            }
            // Change one surviving candidate's benefit without changing its ID;
            // only that candidate must invalidate, including merged prefetch input.
            const auto changedPage = expected.back().second;
            for (size_t i = 0; i < ids.size(); ++i) {
                if ((ids[i] & ~kStreamPrefetchPageTag) == changedPage) { benefits[i] = 900000.0f; }
            }
            residency.beginFrame();
            (void)residency.consumeGpuRequests(batch);
            if (residency.stats().cpuWork.priorityRecomputed != 1 || residency.stats().cpuWork.priorityReused == 0) {
                return RHITestResult::fail("Benefit change did not incrementally invalidate exactly one cached priority");
            }
        }
        return RHITestResult::pass("Priority oracle, duplicate promotion, blocked tails, queued keepalive and lookup lifetime");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletRequestSelectionTest);

class StreamerMeshletBudgetAdmissionTest final : public RHITest {
public:
    StreamerMeshletBudgetAdmissionTest()
    {
        type = RHITestType::Command;
        name = "streamer_meshlet_budget_admission";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "BudgetAdmission.meshstream.bin", asset);
        if (!built.passed) { return built; }
        auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (roots.empty() || pages.size() < 2) { return RHITestResult::fail("Need variable-sized pages"); }
        std::sort(pages.begin(), pages.end(), [&](uint32_t a, uint32_t b) {
            return pageStorageBytes(asset, a) < pageStorageBytes(asset, b);
        });
        const uint32_t small = pages.front(), large = pages.back();
        if (pageStorageBytes(asset, small) == pageStorageBytes(asset, large)) {
            return RHITestResult::fail("Need unequal page allocation sizes");
        }
        MeshletStreamResidencyManager residency;
        std::string reason;
        const MeshletStreamResidencyDesc desc{.asset = &asset,
            .maxResidentBytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, small),
            .immediateGpuRequests = true};
        if (!residency.initialize(desc, reason) || !residency.lockFallbackPages(roots, reason)) {
            return RHITestResult::fail(reason);
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
            return RHITestResult::fail("Budget gate blocked a smaller fit or repeated impossible allocations");
        }
        for (uint32_t root : roots) {
            if (!residency.pageAllocated(root) || residency.unloadPage(root)) {
                return RHITestResult::fail("Budget gate lost fallback protection");
            }
        }
        residency.beginFrame();
        (void)residency.requestPage(large);
        if (residency.stats().cpuWork.allocationAttempts != 1) {
            return RHITestResult::fail("New frame did not refresh admission eligibility");
        }
        // Reset/reinitialize must clear both the exhausted gate and its counters.
        if (!residency.initialize({.asset = &asset, .maxResidentBytes = pageStorageBytes(asset, large)}, reason)) {
            return RHITestResult::fail(reason);
        }
        (void)residency.requestPage(large);
        if (!residency.pageAllocated(large) || residency.stats().cpuWork.budgetRetrySuppressed != 0) {
            return RHITestResult::fail("Reinitialization retained stale budget pressure");
        }
        return RHITestResult::pass("Bounded retries, smaller fit, priority order, locked roots and reset");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletBudgetAdmissionTest);

class StreamerMeshletBatchedUnloadTest final : public RHITest {
public:
    StreamerMeshletBatchedUnloadTest() { type = RHITestType::Command; name = "streamer_meshlet_batched_unload"; }
    RHITestResult run(RHITestContext& context) override
    {
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "batched_unload.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 8) { return RHITestResult::skip("Requires eight streamable pages"); }
        pages.resize(8);
        render::MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize({.asset = &asset,
                .maxResidentBytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages)}, reason) ||
            !residency.lockFallbackPages(roots, reason)) { return RHITestResult::fail(reason); }
        residency.beginFrame();
        for (uint32_t page : pages) { (void)residency.requestPage(page); }
        for (uint32_t page : pages) {
            if (!residency.pageAllocated(page) || !residency.unloadPage(page)) {
                return RHITestResult::fail("Batch unload exhausted the task ring");
            }
        }
        if (residency.stats().queuedUnloadTaskCount != 1 || residency.stats().frameScheduledUnloadCount != pages.size()) {
            return RHITestResult::fail("Same-frame unloads were not batched");
        }
        residency.beginFrame();
        for (uint32_t page : pages) {
            if (residency.pageAllocated(page)) { return RHITestResult::fail("Batch was not retired after the delayed free"); }
        }
        for (uint32_t root : roots) {
            if (!residency.pageAllocated(root) || residency.unloadPage(root)) { return RHITestResult::fail("Batch lost a locked root"); }
        }
        return RHITestResult::pass();
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletBatchedUnloadTest);

class StreamerMeshletDemandCacheTest final : public RHITest {
public:
    StreamerMeshletDemandCacheTest() { type = RHITestType::Command; name = "streamer_meshlet_demand_cache"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "demand_cache.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 3) { return RHITestResult::skip("Requires three streamable pages"); }
        pages.resize(3);
        MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize({.asset = &asset,
                .maxResidentBytes = pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages),
                .maxResidentPages = static_cast<uint32_t>(roots.size() + 2), .queuedFrameCount = 1,
                .unloadDelayFrames = 1, .evictionAgeThresholdFrames = 1}, reason) ||
            !residency.lockFallbackPages(roots, reason)) { return RHITestResult::fail(reason); }
        std::unique_ptr<Streamer> streamer;
        auto result = createStreamer(context.device, makeTestStreamerDesc((roots.size() + 2) * asset.maxPagePayloadBytes() + 4096)).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
        if (!result) { return RHITestResult::fail(toString(result)); }
        std::unique_ptr<Buffer> destination;
        result = context.device.createBuffer({.size = residency.pageBufferSize(),
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { destination = std::move(rhiValue); });
        if (!result) { return RHITestResult::fail(toString(result)); }
        residency.beginFrame();
        (void)residency.requestPage(pages[0]); (void)residency.requestPage(pages[1]);
        const auto uploads = static_cast<uint32_t>(roots.size() + 2);
        if (residency.processUploads(*streamer, *destination, uploads) != uploads) {
            return RHITestResult::fail("Cannot prepare demand-cache residents");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
        if (!residency.pageResident(pages[0]) || !residency.pageResident(pages[1])) {
            return RHITestResult::fail("Demand-cache setup is not resident");
        }
        const std::array<uint32_t, 2> firstUnused{pages[0], pages[0]};
        (void)residency.consumeGpuRequests({.unloadPageIds = firstUnused, .unloadRequestCounter = 2,
            .residentDemandFeedback = true});
        if (residency.stats().queuedUnloadTaskCount != 0 || residency.stats().frameCachedUnusedPageCount != 1) {
            return RHITestResult::fail("Unused feedback eagerly unloaded cached geometry");
        }
        const auto initialWork = residency.stats().cpuWork;
        if (initialWork.demandUnused != 1 || initialWork.demandVisited != 1 || initialWork.demandEpochUpdates != 1 ||
            initialWork.demandTransitions != 1) {
            return RHITestResult::fail("Complete feedback counters do not match resident work");
        }
        residency.beginFrame();
        if (residency.stats().cpuWork.demandVisited != 0) {
            return RHITestResult::fail("CPU work counters did not reset at beginFrame");
        }
        // An empty complete batch means the previously unused page is needed
        // again. Returning to it must neither reload nor leave it evictable.
        (void)residency.consumeGpuRequests(StreamGPURequestBatch{.residentDemandFeedback = true});
        if (residency.pageAge(pages[0]) != 0 || !residency.requestPage(pages[0]) ||
            residency.stats().totalScheduledUploadCount != uploads || residency.queuedUploadCount() != 0) {
            return RHITestResult::fail("Returning demand did not reuse its cached payload");
        }
        // Compare lazy age against the former eager refresh semantics across
        // complete, truncated and duplicate unused feedback, including two
        // batches in one CPU frame. Stable hot cohorts require zero visits.
        std::array<uint64_t, 2> lastUse{residency.stats().frameIndex, residency.stats().frameIndex};
        for (uint32_t step = 0; step < 96; ++step) {
            if ((step % 3) != 0) { residency.beginFrame(); }
            const bool complete = (step % 5) != 0;
            const uint32_t mask = (step * 13u / 7u) % 4u;
            std::vector<uint32_t> unused;
            for (uint32_t i = 0; i < 2; ++i) {
                if (mask & (1u << i)) { unused.push_back(pages[i]); unused.push_back(pages[i]); }
                else if (complete) { lastUse[i] = residency.stats().frameIndex; }
            }
            (void)residency.consumeGpuRequests({.unloadPageIds = unused,
                .unloadRequestCounter = static_cast<uint32_t>(unused.size()) + !complete,
                .unloadOverflowCounter = complete ? 0u : 1u, .residentDemandFeedback = true});
            for (uint32_t i = 0; i < 2; ++i) {
                if (residency.pageAge(pages[i]) != residency.stats().frameIndex - lastUse[i]) {
                    return RHITestResult::fail("Incremental demand age differs from eager reference");
                }
            }
        }
        (void)residency.consumeGpuRequests(StreamGPURequestBatch{.residentDemandFeedback = true});
        residency.beginFrame();
        (void)residency.consumeGpuRequests(StreamGPURequestBatch{.residentDemandFeedback = true});
        if (residency.stats().cpuWork.demandVisited != 0 || residency.pageAge(pages[0]) != 0 ||
            residency.pageAge(pages[1]) != 0 || residency.stats().cpuWork.demandEpochUpdates != 1) {
            return RHITestResult::fail("Stable hot feedback did not use constant-time epoch refresh");
        }
        // Monotonic producer frames: old feedback cannot turn a newly hot page cold.
        const auto sourceFrame = static_cast<uint32_t>(residency.stats().frameIndex);
        (void)residency.consumeGpuRequests({.frameIndex = sourceFrame, .residentDemandFeedback = true});
        (void)residency.consumeGpuRequests({.unloadPageIds = firstUnused, .unloadRequestCounter = 2,
            .frameIndex = sourceFrame - 1, .residentDemandFeedback = true});
        if (residency.stats().cpuWork.demandStaleBatches != 1 || residency.pageAge(pages[0]) != 0) {
            return RHITestResult::fail("Stale producer feedback changed a newer demand epoch");
        }
        residency.beginFrame();
        (void)residency.requestPage(pages[2]);
        if (residency.stats().frameEvictedPageCount != 0 || residency.pageAllocated(pages[2])) {
            return RHITestResult::fail("Budget pressure evicted demanded geometry");
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
            incompleteWork.demandVisited != 1 || incompleteWork.demandEpochUpdates != 0) {
            return RHITestResult::fail("Truncated feedback counters lost protected pages");
        }
        (void)residency.requestPage(pages[2]);
        if (!residency.pageResident(pages[0]) || residency.pageState(pages[1]) != MeshletStreamPageResidencyState::PendingUnload ||
            residency.stats().frameEvictedPageCount != 1 || residency.pageAllocated(pages[2])) {
            return RHITestResult::fail("Budget victim ignored demand feedback or delayed release");
        }
        residency.beginFrame();
        (void)residency.requestPage(pages[2]);
        if (!residency.pageAllocated(pages[2]) || residency.pageAllocated(pages[1])) {
            return RHITestResult::fail("Cached victim was not recycled after completion");
        }
        for (uint32_t page : roots) {
            if (!residency.pageResident(page) || residency.unloadPage(page)) { return RHITestResult::fail("Demand cache lost a root"); }
        }
        if (!residency.unloadPage(pages[0])) { return RHITestResult::fail("Explicit unload no longer works"); }
        const auto beforeReupload = static_cast<uint32_t>(residency.stats().frameIndex);
        residency.beginFrame();
        (void)residency.requestPage(pages[1]);
        if (residency.processUploads(*streamer, *destination, 2) != 2) {
            return RHITestResult::fail("Could not reupload the previously cold page");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
        const auto ageBeforeOldView = residency.pageAge(pages[1]);
        (void)residency.consumeGpuRequests({.unloadPageIds = secondUnused, .unloadRequestCounter = 1,
            .frameIndex = beforeReupload, .residentDemandFeedback = true});
        if (!residency.pageResident(pages[1]) || residency.pageAge(pages[1]) != ageBeforeOldView ||
            residency.stats().cpuWork.demandNewerThanFeedback == 0) {
            return RHITestResult::fail("Old view altered a new residency of the same page");
        }
        (void)residency.consumeGpuRequests({.frameIndex = beforeReupload, .residentDemandFeedback = true});
        if (residency.pageAge(pages[1]) != ageBeforeOldView) {
            return RHITestResult::fail("Lazy complete epoch refreshed a page absent from the producer view");
        }
        (void)residency.consumeGpuRequests({.frameIndex = static_cast<uint32_t>(residency.stats().frameIndex),
            .residentDemandFeedback = true});
        if (residency.pageAge(pages[1]) != 0) {
            return RHITestResult::fail("New residency did not join an eligible complete epoch");
        }
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
                return RHITestResult::fail("Unused page marks lost duplicates, bounds checks or touched-word clearing");
            }
        }
        return RHITestResult::pass("Cache reuse, empty/truncated demand feedback, hot-page protection and delayed budget eviction");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletDemandCacheTest);

class StreamerJointColdReclaimTest final : public RHITest {
public:
    StreamerJointColdReclaimTest() { type = RHITestType::Command; name = "streamer_joint_cold_reclaim"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "joint_cold.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 3) { return RHITestResult::skip("Requires three streamable pages"); }
        pages.resize(3);
        MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize({.asset = &asset,
                .maxResidentBytes = 4 * (pageStorageBytes(asset, roots) + pageStorageBytes(asset, pages)),
                .maxResidentPages = static_cast<uint32_t>(roots.size() + 3), .queuedFrameCount = 1,
                .unloadDelayFrames = 1, .evictionAgeThresholdFrames = 1}, reason) ||
            !residency.lockFallbackPages(roots, reason)) { return RHITestResult::fail(reason); }
        std::unique_ptr<Streamer> streamer;
        auto result = createStreamer(context.device, makeTestStreamerDesc((roots.size() + 2) * asset.maxPagePayloadBytes() + 4096)).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
        if (!result) { return RHITestResult::fail(toString(result)); }
        std::unique_ptr<Buffer> destination;
        result = context.device.createBuffer({.size = residency.pageBufferSize(),
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { destination = std::move(rhiValue); });
        if (!result) { return RHITestResult::fail(toString(result)); }
        residency.beginFrame();
        (void)residency.requestPage(pages[0]); (void)residency.requestPage(pages[1]);
        const auto uploads = static_cast<uint32_t>(roots.size() + 2);
        if (residency.processUploads(*streamer, *destination, uploads) != uploads) {
            return RHITestResult::fail("Cannot prepare demand-cache residents");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
        if (!residency.pageResident(pages[0]) || !residency.pageResident(pages[1])) {
            return RHITestResult::fail("Demand-cache setup is not resident");
        }

        const std::array<uint32_t, 1> unused{pages[1]};
        MeshletStreamColdPageReclaimDesc reclaim{.clasUsedBytes = 1024, .clasCapacityBytes = 1024,
            .retentionFrames = 120, .pressureAgeFrames = 16,
            .clasPageBytes = [&](uint32_t page) -> uint64_t { return page == pages[0] || page == pages[1] ? 512 : 0; }};
        (void)residency.consumeGpuRequests({.unloadPageIds = unused, .unloadRequestCounter = 1, .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 0) { return RHITestResult::fail("CLAS pressure evicted a recent page"); }
        for (uint32_t frame = 0; frame < 16; ++frame) { residency.beginFrame(); }
        (void)residency.consumeGpuRequests({.unloadPageIds = unused, .unloadRequestCounter = 1, .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 1 || residency.pageState(pages[1]) != MeshletStreamPageResidencyState::PendingUnload ||
            !residency.pageResident(pages[0]) || residency.stats().usedResidentBytes >= residency.maxResidentBytes() * 70 / 100) {
            return RHITestResult::fail("CLAS-only pressure did not schedule the shared cold geometry page");
        }
        if (residency.reclaimColdPages(reclaim) != 0 || residency.stats().frameEvictionScanCount != 1) {
            return RHITestResult::fail("Pending joint frees caused duplicate victims or scans");
        }
        residency.beginFrame();
        if (residency.pageAllocated(pages[1])) { return RHITestResult::fail("Joint victim geometry was not freed"); }
        reclaim.clasUsedBytes = 512;
        reclaim.clasRetiringBytes = 512;
        for (uint32_t frame = 0; frame < 121; ++frame) { residency.beginFrame(); }
        // Complete feedback still protects the current view even after a long pause.
        (void)residency.consumeGpuRequests(StreamGPURequestBatch{.residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 0) { return RHITestResult::fail("Current view was treated as cold"); }
        const std::array<uint32_t, 1> nowUnused{pages[0]};
        for (uint32_t frame = 0; frame < 121; ++frame) { residency.beginFrame(); }
        (void)residency.consumeGpuRequests({.unloadPageIds = nowUnused, .unloadRequestCounter = 1, .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 1) { return RHITestResult::fail("Old cold page was retained below both budgets"); }
        for (uint32_t root : roots) {
            if (!residency.pageResident(root)) { return RHITestResult::fail("Joint reclaim lost a fallback page"); }
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
                return RHITestResult::fail("Cannot upload cold-sort fixture");
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
                    return RHITestResult::fail("Cold eviction changed age/ID order after erase and reinsertion");
                }
            }
            if (residency.reclaimColdPages(reclaim) != 0 || residency.stats().frameEvictionScanCount != 1) {
                return RHITestResult::fail("Cached cold candidates scheduled a duplicate victim");
            }
        }
        residency.beginFrame();
        for (uint32_t id : pages) { (void)residency.requestPage(id); }
        if (residency.processUploads(*streamer, *destination, 3) != 3) {
            return RHITestResult::fail("Cannot upload partial-sort fixture");
        }
        for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
        (void)residency.consumeGpuRequests({.unloadPageIds = pages, .unloadRequestCounter = 3,
                                           .residentDemandFeedback = true});
        uint32_t clasQueries = 0;
        reclaim.retentionFrames = 120;
        reclaim.clasPageBytes = [&](uint32_t) -> uint64_t { ++clasQueries; return 512; };
        if (residency.reclaimColdPages(reclaim) != 0 || clasQueries != 0 || residency.stats().cpuWork.coldVisited != 0) {
            return RHITestResult::fail("Unexpired retention suffix was visited or queried for CLAS sizes");
        }
        // Expand the same cached candidates when CLAS pressure appears later in
        // the frame. Once one victim is credited, the remaining young suffix
        // must be skipped even though the requested per-call limit is larger.
        reclaim.clasUsedBytes = reclaim.clasCapacityBytes = 1024;
        reclaim.maxPages = 3;
        if (residency.reclaimColdPages(reclaim) != 1 || clasQueries != 1 ||
            residency.pageState(pages[0]) != MeshletStreamPageResidencyState::PendingUnload ||
            residency.stats().frameEvictionScanCount != 1) {
            return RHITestResult::fail("Partial cold sort missed new pressure, changed ID order or over-evicted after pressure ended");
        }
        // Pending-free credit ends pressure on another call in the same frame.
        if (residency.reclaimColdPages(reclaim) != 0 || clasQueries != 2) {
            return RHITestResult::fail("Partial cold sort ignored pending free credit");
        }
        return RHITestResult::pass("CLAS pressure, delayed credit, partial-sort expansion, due cutoff and age/ID order across reloads");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerJointColdReclaimTest);

class StreamerMeshletPrefetchByteReserveTest final : public RHITest {
public:
    StreamerMeshletPrefetchByteReserveTest() { type = RHITestType::Command; name = "streamer_meshlet_prefetch_byte_reserve"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "prefetch_byte_reserve.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 9) { return RHITestResult::skip("Requires nine streamable pages"); }
        std::sort(pages.begin(), pages.end(), [&](uint32_t a, uint32_t b) {
            return pageDeviceStorageBytes(asset, a) > pageDeviceStorageBytes(asset, b);
        });
        const uint32_t speculative = pages.back(); pages.pop_back();
        const uint32_t demanded = pages.back(); pages.pop_back();
        const uint64_t reserve = pageDeviceStorageBytes(asset, demanded);
        const uint64_t speculativeBytes = pageDeviceStorageBytes(asset, speculative);
        uint64_t occupied = pageDeviceStorageBytes(asset, roots);
        size_t residentCount = 0;
        while (occupied <= 3 * (reserve + speculativeBytes) && residentCount < pages.size()) {
            occupied += pageDeviceStorageBytes(asset, pages[residentCount++]);
        }
        if (occupied <= 3 * (reserve + speculativeBytes)) { return RHITestResult::skip("Cannot fill more than three quarters of the test budget"); }
        pages.resize(residentCount);
        MeshletStreamResidencyDesc desc{.asset = &asset, .maxResidentBytes = occupied + reserve + speculativeBytes,
            .immediateGpuRequests = true};
        desc.prefetchReserveBytes = reserve;
        MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize(desc, reason) || !residency.lockFallbackPages(roots, reason)) {
            return RHITestResult::fail(reason);
        }
        residency.beginFrame();
        for (uint32_t id : pages) {
            (void)residency.requestPage(id);
            if (!residency.pageAllocated(id)) { return RHITestResult::fail("Cannot allocate the occupied reserve fixture"); }
        }
        if (residency.storage().usedBytes() <= residency.maxResidentBytes() * 3 / 4 ||
            !residency.canPrefetchPage(scene::meshletStreamDevicePayloadSize(asset.pages()[speculative]))) {
            return RHITestResult::fail("Byte reserve still applies the legacy 75 percent watermark");
        }
        const uint32_t forecast[] = {speculative | kStreamPrefetchPageTag};
        (void)residency.consumeGpuRequests({.loadPageIds = forecast, .taggedPrefetchRequests = true});
        if (!residency.pageAllocated(speculative) || residency.stats().totalPrefetchAdmitted != 1 ||
            residency.canPrefetchPage(scene::meshletStreamDevicePayloadSize(asset.pages()[demanded]))) {
            return RHITestResult::fail("Prefetch did not leave exactly the requested byte reserve");
        }
        const uint32_t blockedForecast[] = {demanded | kStreamPrefetchPageTag};
        (void)residency.consumeGpuRequests({.loadPageIds = blockedForecast, .taggedPrefetchRequests = true});
        if (residency.pageAllocated(demanded) || residency.stats().totalPrefetchDeferred != 1 ||
            residency.stats().totalEvictedPageCount != 0) {
            return RHITestResult::fail("Speculative admission consumed the demand reserve or evicted geometry");
        }
        const uint32_t demand[] = {demanded};
        (void)residency.consumeGpuRequests({.loadPageIds = demand});
        if (!residency.pageAllocated(demanded) || residency.storage().usedBytes() != residency.maxResidentBytes() ||
            residency.stats().totalEvictedPageCount != 0) {
            return RHITestResult::fail("Current demand could not consume reserved bytes without eviction");
        }
        return RHITestResult::pass("Prefetch above 75 percent, exact demand byte reserve and no speculative eviction");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletPrefetchByteReserveTest);

class StreamerMeshletDemandRetentionHeadroomTest final : public RHITest {
public:
    StreamerMeshletDemandRetentionHeadroomTest() { type = RHITestType::Command; name = "streamer_meshlet_demand_retention_headroom"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "demand_retention_headroom.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 3) { return RHITestResult::skip("Requires three streamable pages"); }
        pages.resize(3);
        std::sort(pages.begin(), pages.end());
        const uint64_t bytes = pageDeviceStorageBytes(asset, roots) + pageDeviceStorageBytes(asset, pages);
        const auto upload = [&](MeshletStreamResidencyManager& residency, uint32_t count) -> RHITestResult {
            std::unique_ptr<Streamer> streamer;
            auto result = createStreamer(context.device, makeTestStreamerDesc(count * asset.maxPagePayloadBytes() + 4096))
                .transform([&](auto value) { streamer = std::move(value); });
            if (!result) { return RHITestResult::fail(toString(result)); }
            std::unique_ptr<Buffer> destination;
            result = context.device.createBuffer({.size = residency.pageBufferSize(),
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
                .transform([&](auto value) { destination = std::move(value); });
            if (!result) { return RHITestResult::fail(toString(result)); }
            if (residency.processUploads(*streamer, *destination, count) != count) {
                return RHITestResult::fail("Cannot upload the retention fixture");
            }
            for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
            return RHITestResult::pass();
        };
        MeshletStreamResidencyDesc desc{.asset = &asset, .maxResidentBytes = bytes,
            .queuedFrameCount = 1, .unloadDelayFrames = 1, .evictionAgeThresholdFrames = 1,
            .immediateGpuRequests = true};
        desc.prefetchReserveBytes = kMeshletStreamStorageAlignment;
        MeshletStreamResidencyManager residency;
        std::string reason;
        if (!residency.initialize(desc, reason) || !residency.lockFallbackPages(roots, reason)) { return RHITestResult::fail(reason); }
        residency.beginFrame();
        for (uint32_t id : pages) { (void)residency.requestPage(id); }
        const uint32_t uploads = static_cast<uint32_t>(roots.size() + pages.size());
        const auto uploaded = upload(residency, uploads);
        if (!uploaded.passed) { return uploaded; }
        MeshletStreamColdPageReclaimDesc reclaim{.retentionFrames = 2, .pressureAgeFrames = 1};
        reclaim.geometryReserveBytes = pageDeviceStorageBytes(asset, pages[0]);
        reclaim.geometryPressureReserveBytes = reclaim.geometryReserveBytes / 2;
        reclaim.retainDemandCache = true;
        (void)residency.consumeGpuRequests({.residentDemandFeedback = true});
        if (residency.storage().usedBytes() != residency.maxResidentBytes() || residency.reclaimColdPages(reclaim) != 0) {
            return RHITestResult::fail("Headroom policy evicted the fully hot working set");
        }
        // Two old demand pages become unused. Only the first is needed to meet
        // the byte target; the other should survive for a later return.
        for (uint32_t frame = 0; frame < 3; ++frame) { residency.beginFrame(); }
        const auto unused = std::span(pages).first(2);
        (void)residency.consumeGpuRequests({.unloadPageIds = unused, .unloadRequestCounter = 2,
            .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 1 ||
            residency.pageState(pages[0]) != MeshletStreamPageResidencyState::PendingUnload ||
            !residency.pageResident(pages[1]) || !residency.pageResident(pages[2])) {
            return RHITestResult::fail("Headroom did not stop after enough confirmed cold bytes");
        }
        if (residency.reclaimColdPages(reclaim) != 0 ||
            residency.canPrefetchPage(scene::meshletStreamDevicePayloadSize(asset.pages()[pages[0]])) ||
            residency.stats().frameEvictedPageCount != 1 ||
            residency.stats().framePendingFreeBytes != reclaim.geometryReserveBytes) {
            return RHITestResult::fail("Pending free was double-evicted or counted as allocatable prefetch space");
        }
        residency.beginFrame();
        for (uint32_t frame = 0; frame < 120; ++frame) { residency.beginFrame(); }
        const uint32_t stillUnused[] = {pages[1]};
        (void)residency.consumeGpuRequests({.unloadPageIds = stillUnused, .unloadRequestCounter = 1,
            .residentDemandFeedback = true});
        if (residency.reclaimColdPages(reclaim) != 0 || !residency.pageResident(pages[1])) {
            return RHITestResult::fail("Unpressured demand cache expired instead of retaining a returnable page");
        }
        (void)residency.consumeGpuRequests({.residentDemandFeedback = true});
        if (!residency.requestPage(pages[1]) || residency.queuedUploadCount() != 0 ||
            residency.stats().totalScheduledUploadCount != uploads) {
            return RHITestResult::fail("Returning demand reloaded a retained page");
        }
        for (uint32_t root : roots) {
            if (!residency.pageResident(root)) { return RHITestResult::fail("Headroom reclaim lost a locked root"); }
        }
        // Useful demand cache and unused speculation have different retention:
        // with ample memory, only the unused speculative payload expires.
        MeshletStreamResidencyManager speculative;
        desc.maxResidentBytes = bytes * 4;
        if (!speculative.initialize(desc, reason)) { return RHITestResult::fail(reason); }
        speculative.beginFrame();
        (void)speculative.requestPage(pages[0]);
        const uint32_t forecast[] = {pages[1] | kStreamPrefetchPageTag};
        (void)speculative.consumeGpuRequests({.loadPageIds = forecast, .taggedPrefetchRequests = true});
        const auto speculativeUploaded = upload(speculative, 2);
        if (!speculativeUploaded.passed) { return speculativeUploaded; }
        (void)speculative.consumeGpuRequests({.unloadPageIds = unused, .unloadRequestCounter = 2,
            .residentDemandFeedback = true});
        if (speculative.reclaimColdPages(reclaim) != 1 || !speculative.pageResident(pages[0]) ||
            speculative.pageState(pages[1]) != MeshletStreamPageResidencyState::PendingUnload ||
            speculative.stats().totalPrefetchUsed != 0) {
            return RHITestResult::fail("Unused speculation did not expire independently of useful demand cache");
        }
        return RHITestResult::pass("Hot/root safety, bounded headroom, pending-free credit, return reuse and unused speculation expiry");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletDemandRetentionHeadroomTest);

class StreamerMeshletEvictionByteBudgetTest final : public RHITest {
public:
    StreamerMeshletEvictionByteBudgetTest() { type = RHITestType::Command; name = "streamer_meshlet_eviction_byte_budget"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "eviction_byte_budget.meshstream.bin", asset);
        if (!built.passed) { return built; }
        const auto roots = fallbackPagesFor(asset);
        auto pages = nonFallbackPagesFor(asset, roots);
        if (pages.size() < 4) { return RHITestResult::skip("Requires four streamable pages"); }
        std::sort(pages.begin(), pages.end(), [&](uint32_t a, uint32_t b) {
            return pageDeviceStorageBytes(asset, a) > pageDeviceStorageBytes(asset, b);
        });
        const uint32_t demandProbe = pages[3];
        pages.resize(3);
        const uint64_t largest = pageDeviceStorageBytes(asset, pages[0]);
        if (pageDeviceStorageBytes(asset, pages[2]) <= largest / 2) { return RHITestResult::skip("Requires three similarly sized pages"); }
        const uint64_t bytes = pageDeviceStorageBytes(asset, roots) + pageDeviceStorageBytes(asset, pages);
        for (bool constrainBytes : {true, false}) {
            MeshletStreamResidencyDesc desc{.asset = &asset, .maxResidentBytes = bytes * 4,
                .queuedFrameCount = 1, .unloadDelayFrames = 1, .evictionAgeThresholdFrames = 1};
            desc.maxResidentPages = static_cast<uint32_t>(roots.size() + pages.size());
            desc.maxPageEvictionsPerFrame = constrainBytes ? 3 : 1;
            desc.maxEvictionBytesPerFrame = constrainBytes ? largest : largest * 3;
            MeshletStreamResidencyManager residency;
            std::string reason;
            if (!residency.initialize(desc, reason) || !residency.lockFallbackPages(roots, reason)) { return RHITestResult::fail(reason); }
            residency.beginFrame();
            for (uint32_t id : pages) { (void)residency.requestPage(id); }
            const uint32_t count = static_cast<uint32_t>(roots.size() + pages.size());
            std::unique_ptr<Streamer> streamer;
            auto result = createStreamer(context.device, makeTestStreamerDesc(count * asset.maxPagePayloadBytes() + 4096))
                .transform([&](auto value) { streamer = std::move(value); });
            if (!result) { return RHITestResult::fail(toString(result)); }
            std::unique_ptr<Buffer> destination;
            result = context.device.createBuffer({.size = residency.pageBufferSize(),
                .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback})
                .transform([&](auto value) { destination = std::move(value); });
            if (!result) { return RHITestResult::fail(toString(result)); }
            if (residency.processUploads(*streamer, *destination, count) != count) {
                return RHITestResult::fail("Cannot upload the eviction-budget fixture");
            }
            for (uint32_t frame = 0; frame < 4; ++frame) { residency.beginFrame(); }
            (void)residency.consumeGpuRequests({.unloadPageIds = pages, .unloadRequestCounter = 3,
                .residentDemandFeedback = true});
            MeshletStreamColdPageReclaimDesc reclaim{.retentionFrames = 1, .pressureAgeFrames = 1};
            for (uint32_t frame = 0; frame < 2; ++frame) {
                uint32_t evicted = 0;
                for (uint32_t repeat = 0; repeat < 4; ++repeat) { evicted += residency.reclaimColdPages(reclaim); }
                if (frame == 0) {
                    (void)residency.requestPage(demandProbe);
                    if (residency.pageAllocated(demandProbe)) {
                        return RHITestResult::fail("Admission reused a page slot before its delayed free completed");
                    }
                }
                uint64_t pendingBytes = 0;
                for (uint32_t id : pages) {
                    if (residency.pageState(id) == MeshletStreamPageResidencyState::PendingUnload) {
                        pendingBytes += pageDeviceStorageBytes(asset, id);
                    }
                }
                if (evicted != 1 || residency.stats().frameEvictedPageCount != 1 ||
                    pendingBytes > desc.maxEvictionBytesPerFrame || residency.stats().frameEvictedGeometryBytes != pendingBytes ||
                    residency.stats().totalEvictedPageCount != frame + 1) {
                    return RHITestResult::fail(constrainBytes
                        ? "Repeated reclaim exceeded or failed to reset the shared byte cap"
                        : "Repeated reclaim exceeded or failed to reset the shared page cap");
                }
                residency.beginFrame();
            }
            for (uint32_t root : roots) {
                if (!residency.pageResident(root)) { return RHITestResult::fail("Capped reclaim lost a locked root"); }
            }
        }
        return RHITestResult::pass("Byte/page limits shared by reclaim and admission reset only at beginFrame");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletEvictionByteBudgetTest);

class MeshletStreamFragmentedStorageTest final : public RHITest {
public:
    MeshletStreamFragmentedStorageTest() { type = RHITestType::Validation; name = "streamer_meshlet_fragmented_storage"; }
    RHITestResult run(RHITestContext&) override
    {
        render::MeshletStreamStorage storage;
        std::string reason;
        if (!storage.initialize(4096, 256, reason)) { return RHITestResult::fail(reason); }
        const auto large = storage.allocate(1024);
        const auto separator = storage.allocate(256);
        const auto small = storage.allocate(512);
        const auto tail = storage.allocate(2304);
        if (!large.valid() || !separator.valid() || !small.valid() || !tail.valid()) {
            return RHITestResult::fail("Best-fit setup failed");
        }
        storage.release(large); storage.release(small);
        const auto fitted = storage.allocate(257, true);
        const auto preserved = storage.allocate(1024, true);
        if (fitted.offset != small.offset || preserved.offset != large.offset || storage.allocate(UINT64_MAX, true).valid()) {
            return RHITestResult::fail("Best-fit consumed a larger hole or accepted overflow");
        }
        storage.release(fitted); storage.release(preserved); storage.release(separator); storage.release(tail);
        if (storage.largestFreeBlockBytes() != 4096 || storage.freeBlockCount() != 1) {
            return RHITestResult::fail("Best-fit releases failed to coalesce");
        }
        if (!storage.initialize(4096u * 256u, 256, reason)) { return RHITestResult::fail(reason); }
        std::vector<render::MeshletStreamStorageAllocation> pages;
        for (uint32_t i = 0; i < 4096; ++i) {
            pages.push_back(storage.allocate(256));
            if (!pages.back().valid()) { return RHITestResult::fail("Cannot fill fragmented storage"); }
        }
        for (uint32_t i = 0; i < pages.size(); i += 2) { storage.release(pages[i]); }
        for (uint32_t retry = 0; retry < 10000; ++retry) {
            if (storage.canAllocate(257) || storage.allocate(257).valid() || storage.largestFreeBlockBytes() != 256) {
                return RHITestResult::fail("Fragmented free bytes were mistaken for a contiguous allocation");
            }
        }
        storage.release(pages[1]);
        if (!storage.canAllocate(768) || storage.largestFreeBlockBytes() != 768) {
            return RHITestResult::fail("Coalescing did not invalidate the free-block bound");
        }
        const auto merged = storage.allocate(768);
        if (!merged.valid() || merged.offset != 0 || storage.canAllocate(512) ||
            storage.canAllocate(UINT64_MAX) || storage.allocate(UINT64_MAX).valid()) {
            return RHITestResult::fail("Allocation left a stale bound or accepted overflowing alignment");
        }
        storage.release(merged);
        if (!storage.canAllocate(768)) { return RHITestResult::fail("Released range did not become allocatable"); }
        if (!storage.initialize(512, 256, reason) || storage.largestFreeBlockBytes() != 512) {
            return RHITestResult::fail("Storage reset retained old free-block bounds");
        }
        return RHITestResult::pass("Fragmentation, repeated misses, coalescing, allocation, overflow and reset");
    }
};
METALLIC_REGISTER_RHI_TEST(MeshletStreamFragmentedStorageTest);

class StreamerMeshletUploadCompletionTest final : public RHITest {
public:
    StreamerMeshletUploadCompletionTest() { type = RHITestType::Command; name = "streamer_meshlet_upload_completion"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "completion.meshstream.bin", asset);
        if (!built.passed) { return built; }
        if (asset.pageCount() < 3) { return RHITestResult::fail("Need three pages for completion batching"); }
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
        if (!(expression)) { return RHITestResult::fail("Upload completion: " #expression); }
        UPLOAD_REQUIRE(tracker.initialize(context.device, context.graphicsQueue));
        UPLOAD_REQUIRE(createStreamer(context.device, makeTestStreamerDesc()).transform([&](auto rhiValue) { streamer = std::move(rhiValue); }));
        UPLOAD_REQUIRE(context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        UPLOAD_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        UPLOAD_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { prefix = std::move(rhiValue); }));
        UPLOAD_REQUIRE(context.device.createSemaphore().transform([&](auto rhiValue) { gate = std::move(rhiValue); }));
        const uint64_t capacity = alignStreamStorageBytes(asset.maxPagePayloadBytes()) * 4u;
        UPLOAD_REQUIRE(context.device.createBuffer({.size = capacity,
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { destination = std::move(rhiValue); }));
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
                UPLOAD_REQUIRE(commands->begin(scenario == 5 && attempt == 0 ? nullptr : frame.submissionContext()));
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
                    const BufferBarrierDesc barrier{
                        .buffer = destination.get(),
                        .before = {},
                        .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                        .range = {.size = capacity},
                    };
                    if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
                    const auto copyResult = streamer->copyStreamedData(*commands);
                    const bool sameRecording = !(scenario == 5 && attempt == 0);
                    if (sameRecording) { UPLOAD_REQUIRE(copyResult); }
                    else { UPLOAD_REQUIRE(hasError(copyResult, Error::InvalidArgument)); }
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
                        UPLOAD_REQUIRE(prefix->begin(frame.submissionContext()));
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
                        UPLOAD_REQUIRE(prefix->begin(frame.submissionContext()));
                        UPLOAD_REQUIRE(prefix->end());
                        CommandBuffer* prefixBuffers[] = {prefix.get()};
                        GPUCompletionPoint prefixCompletion;
                        UPLOAD_REQUIRE(tracker.submitSegment({
                            .commandBuffers = {prefixBuffers, 1},
                        }, frame).transform([&](auto value) { prefixCompletion = std::move(value); }));
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
                UPLOAD_REQUIRE(tracker.submit({
                    .waitSemaphores = {scenario == 0 ? &wait : nullptr, scenario == 0 ? 1u : 0u},
                    .commandBuffers = {buffers, 1},
                }, frame));
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
        return RHITestResult::pass("GPU gate, unflushed/cancelled/partial/mismatched submissions, demand/root retries, batch drain and reset");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerMeshletUploadCompletionTest);

class StreamerOrderedPublicationTest final : public RHITest {
public:
    StreamerOrderedPublicationTest() { type = RHITestType::Command; name = "streamer_ordered_publication_retry"; }
    RHITestResult run(RHITestContext& context) override
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
            }, .context = &validationMessages}}).transform([&](auto rhiValue) { ownedDevice = std::move(rhiValue); });
        if (!created) {
            return hasError(created, Error::Unsupported) ? RHITestResult::skip("Bindless device unavailable") :
                RHITestResult::fail("Cannot create ordered publication device");
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
        if (!(expression)) { return RHITestResult::fail("Ordered publication: " #expression); }
        std::string log;
        runtime.setDebugReadbackEnabled(true);
        ORDERED_REQUIRE(runtime.initialize(device, {
            .sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf",
            .streamAssetPath = asset.path(), .maxResidentPages = 64, .maxLockedFallbackPages = 64,
            .maxPageUploadsPerFrame = 64, .maxGpuPageRequests = 256, .maxGpuPageUnloadRequests = 256,
            .maxActiveGroups = 4096, .maxTraversalWorkers = 64, .maxTraversalWorkItems = 4096,
            .pageLoadConcurrency = 0, .queuedFrameCount = 2}, log));
        ORDERED_REQUIRE(!runtime.sceneReadiness().ready);
        ORDERED_REQUIRE(createStreamer(device, makeTestStreamerDesc()).transform([&](auto rhiValue) { streamer = std::move(rhiValue); }));
        ORDERED_REQUIRE(device.createCommandPool(queue).transform([&](auto rhiValue) { pool = std::move(rhiValue); }));
        ORDERED_REQUIRE(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); }));
        ORDERED_REQUIRE(tracker.initialize(device, queue));
        ORDERED_REQUIRE(device.createBuffer({.size = sizeof(MeshletStreamGPUActiveHeader),
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); }));
        MeshletStreamFrameDesc view{.width = 192, .height = 128, .selectedLodLevel = 0, .enableGpuLodSelection = false};
        view.camera = {.eye = {-.0168404f, .110154f, .22f}, .center = {-.0168404f, .110154f, -.00153695f},
            .znear = .001f, .zfar = 10.f};
        for (uint32_t attempt = 0; attempt < 3; ++attempt) {
            auto& frame = *frames[attempt % 2];
            ORDERED_REQUIRE(frame.begin(attempt + 1));
            ORDERED_REQUIRE(pool->reset());
            ORDERED_REQUIRE(commands->begin(frame.submissionContext()));
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
            BufferBarrierDesc barrier{
                .buffer = header->buffer,
                .before = metallic::render::resourceSyncScope(header->state, metallic::render::PipelineStageBits::AllCommands),
                .after = {PipelineStageBits::Transfer, AccessBits::TransferRead},
                .range = {.size = sizeof(MeshletStreamGPUActiveHeader)},
            };
            if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            {
                auto sourceSlice = header->buffer->slice({0, sizeof(MeshletStreamGPUActiveHeader)});
                if (!sourceSlice) { return RHITestResult::fail(std::string("source slice failed: ") + render::resultToString(sourceSlice)); }
                auto destinationSlice = readback.get()->slice({0, sizeof(MeshletStreamGPUActiveHeader)});
                if (!destinationSlice) { return RHITestResult::fail(std::string("destination slice failed: ") + render::resultToString(destinationSlice)); }
                if (auto commandResult = commands->copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return RHITestResult::fail(std::string("copyBuffer failed: ") + render::resultToString(commandResult)); }
            }
            std::swap(barrier.before, barrier.after);
            if (auto commandResult = commands->synchronize({.buffers = {&barrier, 1}}); !commandResult) { return RHITestResult::fail(std::string("synchronize failed: ") + render::resultToString(commandResult)); }
            ORDERED_REQUIRE(runtime.cmdEndFrame(*commands));
            ORDERED_REQUIRE(commands->end());
            streamer->endFrame();
            if (attempt == 0) { frame.cancel(); continue; }
            CommandBuffer* buffers[]{commands.get()};
            ORDERED_REQUIRE(tracker.submit({.commandBuffers = {buffers, 1}}, frame));
            ORDERED_REQUIRE(frame.wait(5'000'000'000ull));
            MeshletStreamGPUActiveHeader result;
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
        return RHITestResult::pass("Cancelled initial publication retries; GPU selects uploaded roots before CPU confirmation across frame slots");
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
METALLIC_REGISTER_RHI_TEST(MeshletStreamCLASPagePlanTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyGPURequestPatchTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyLatestGPURequestTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyGPURequestUnloadOverflowTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyEvictionDelayAgeTest);


class StreamerUploadByteBudgetTest final : public RHITest {
public:
    StreamerUploadByteBudgetTest() { type = RHITestType::Validation; name = "streamer_meshlet_upload_byte_budget"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        scene::MeshletStreamAsset asset;
        const auto built = buildBunnyStreamAssetForTest(context.outputDirectory / "ByteBudget.meshstream.bin", asset);
        if (!built.passed) { return built; }
        if (asset.pageCount() < 4) { return RHITestResult::fail("Need four pages for byte budget test"); }
        const uint64_t capacity = uint64_t(asset.maxPagePayloadBytes()) * 8;
        std::unique_ptr<Buffer> destination;
        auto result = context.device.createBuffer({.size = capacity,
            .usage = BufferUsageBits::TransferDestination, .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { destination = std::move(rhiValue); });
        if (!result) { return RHITestResult::fail(toString(result)); }
        for (bool asynchronous : {false, true}) {
            for (uint64_t budget : {0ull, 1ull, uint64_t(asset.maxPagePayloadBytes())}) {
                MeshletStreamResidencyManager residency;
                std::string reason;
                if (!residency.initialize({.asset = &asset, .maxResidentBytes = capacity, .queuedFrameCount = 1,
                        .pageLoadConcurrency = asynchronous ? 2u : 0u, .maxPageLoadsInFlight = 4,
                        .completionDrivenUploads = false}, reason)) { return RHITestResult::fail(reason); }
                std::unique_ptr<Streamer> streamer;
                result = createStreamer(context.device, makeTestStreamerDesc(capacity + 4096)).transform([&](auto rhiValue) { streamer = std::move(rhiValue); });
                if (!result) { return RHITestResult::fail(toString(result)); }
                residency.beginFrame();
                for (uint32_t page = 0; page < 4; ++page) { (void)residency.requestPage(page); }
                if (asynchronous) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (residency.stats().preparedPageLoadCount < 4 && std::chrono::steady_clock::now() < deadline) {
                        (void)residency.processUploads(*streamer, *destination, 0);
                        std::this_thread::yield();
                    }
                    if (residency.stats().preparedPageLoadCount != 4) { return RHITestResult::fail("Async pages not prepared"); }
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
                        return RHITestResult::fail("Byte credit was bypassed by a repeat call or oversized page");
                    }
                    if (seen.size() == before || (budget == 1 && seen.size() - before != 1)) {
                        return RHITestResult::fail("Oversized page starved or multiple pages escaped the budget");
                    }
                }
                std::sort(seen.begin(), seen.end());
                uint64_t expectedBytes = 0;
                for (uint32_t page = 0; page < 4; ++page) { expectedBytes += scene::meshletStreamDevicePayloadSize(asset.pages()[page]); }
                if (seen != std::vector<uint32_t>{0, 1, 2, 3} || observedBytes != expectedBytes ||
                    residency.stats().totalUploadBytes != expectedBytes) {
                    return RHITestResult::fail("Budget deferral lost/duplicated a page or counted disk bytes");
                }
            }
        }
        return RHITestResult::pass("Sync/async uploads: shared frame credit, unlimited mode, oversized progress, exact device bytes and deferred retry");
    }
};
METALLIC_REGISTER_RHI_TEST(StreamerUploadByteBudgetTest);

} // namespace
} // namespace metallic::tests

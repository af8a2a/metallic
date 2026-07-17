#include "RhiTest.h"

#include "Runtime/Render/MeshletStreamClas.h"
#include "Runtime/Render/MeshletStreamPageLoader.h"
#include "Runtime/Render/MeshletStreamResidency.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetConfig.h"
#include "Runtime/Render/StreamingTaskQueue.h"
#include "Runtime/Scene/MeshletStreamAsset.h"
#include "Runtime/Scene/Scene.h"
#include "Runtime/Task/task_system.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
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

class StreamerRenderGraphUnsupportedDoesNotBeginFrameTest : public RhiTest {
public:
    StreamerRenderGraphUnsupportedDoesNotBeginFrameTest()
    {
        type = RhiTestType::Command;
        name = "streamer_render_graph_unsupported_does_not_begin_frame";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        registerStreamerGraphPass();

        render::Queue* computeQueue = context.device.getQueue(render::QueueType::Compute);
        if (computeQueue == nullptr) {
            return RhiTestResult::skip("device has no compute queue");
        }

        render::RenderGraph graph;
        graph.setName("StreamerUnsupportedCrossQueue");
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
            .graphicsQueue = &context.graphicsQueue,
            .computeQueue = computeQueue,
        });
        if (!render::hasError(result, render::Error::Unsupported)) {
            return RhiTestResult::fail(
                std::string("expected Unsupported for cross-queue resource edge, got ") +
                toString(result));
        }

        const render::RenderGraphStreamingStats& after = executor.streamingStats();
        if (after.frameIndex != before.frameIndex ||
            after.streamer.frameIndex != before.streamer.frameIndex ||
            after.flushCount != before.flushCount ||
            after.transferCount != before.transferCount) {
            return RhiTestResult::fail("unsupported submit started or mutated the RenderGraph streaming frame");
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

METALLIC_REGISTER_RHI_TEST(StreamingTaskQueueLifecycleTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamPageLoaderTaskGraphTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamPageLoadConfigurationCompatibilityTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamStorageAddressLimitTest);
METALLIC_REGISTER_RHI_TEST(StreamerBufferUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerTextureUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerConstantUploadTest);
METALLIC_REGISTER_RHI_TEST(StreamerRenderGraphFlushTest);
METALLIC_REGISTER_RHI_TEST(StreamerRenderGraphUnsupportedDoesNotBeginFrameTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyUploadTest);
METALLIC_REGISTER_RHI_TEST(MeshletStreamClasPagePlanTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyGpuRequestPatchTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyLatestGpuRequestTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyGpuRequestUnloadOverflowTest);
METALLIC_REGISTER_RHI_TEST(StreamerMeshletResidencyEvictionDelayAgeTest);

} // namespace
} // namespace metallic::tests

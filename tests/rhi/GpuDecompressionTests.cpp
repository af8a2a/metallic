#include "RhiTest.h"
#include "GpuPageCodecChecks.h"
#include "Runtime/Render/GAPI/StreamUploadCompletion.h"
#include "Runtime/Render/Streamer/MeshletStreamClas.h"
#include "Runtime/Render/Streamer/MeshletStreamResidency.h"
#include "Runtime/Scene/MeshletStreamGpuCodec.h"
#include "Runtime/Scene/Scene.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace metallic::tests {
namespace {

class GpuPageCodecTest : public RhiTest {
public:
    GpuPageCodecTest() { name = "streamer_gpu_page_codec"; type = RhiTestType::Resource; }
    RhiTestResult run(RhiTestContext& context) override
    {
        try {
            checkGpuPageCodec(context.outputDirectory);
            return RhiTestResult::pass("Raw/GDeflate round trip, topology preservation, CLAS sideband, corruption and overwrite rejection");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};

class GpuPageDecompressionTest final : public RhiTest {
public:
    GpuPageDecompressionTest() { name = "streamer_gpu_decompression"; type = RhiTestType::Command; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        if (!context.device.capabilities().memoryDecompression) { return RhiTestResult::skip("EXT GDeflate unavailable"); }
        std::unique_ptr<Streamer> streamer;
        std::unique_ptr<Buffer> destination, readback;
        std::unique_ptr<CommandPool> pool;
        std::unique_ptr<CommandBuffer> commands;
        RenderFrameContext frame;
        QueueSubmissionTracker tracker;
        struct Drain {
            Queue& queue; RenderFrameContext& frame;
            ~Drain() { frame.cancel(); (void)queue.waitIdle(); }
        } drain{context.graphicsQueue, frame};
        try {
            std::string reason;
            const auto decoded = makeMixedTileGpuPagePayload();
            std::vector<uint8_t> stored;
            requireGpuPage(scene::encodeMeshletStreamGpuPage(decoded, true, stored, reason), reason);
            scene::MeshletStreamPageInfo page;
            page.clusterCount = 1; page.vertexCount = 3; page.triangleIndexCount = 3;
            page.attributeFlags = scene::kMeshletStreamPayloadAttributePosition;
            page.payloadFlags |= scene::kMeshletStreamPayloadCompactPositions;
            page.payloadSize = stored.size(); page.uncompressedSize = decoded.size();
            page.compressionMode = uint32_t(scene::MeshletStreamPayloadCompression::GpuTiles);
            scene::MeshletStreamGpuPage metadata;
            requireGpuPage(scene::inspectMeshletStreamGpuPage(page, stored, metadata, reason), reason);
            std::vector<uint8_t> cpu;
            requireGpuPage(scene::decodeMeshletStreamGpuPage(page, stored, cpu, reason) && cpu == decoded, "Multi-tile CPU decode mismatch");
            requireGpuPage(metadata.tiles.size() == 3 && metadata.tiles[0].codec == 1 && metadata.tiles[1].codec == 0, "Fixture must mix Raw and GDeflate");
            std::vector<StreamDecompressionTile> tiles;
            for (const auto& tile : metadata.tiles) { tiles.push_back({tile.sourceOffset, tile.destinationOffset, tile.storedBytes, tile.decodedBytes, tile.codec != 0}); }
            requireGpuPage(bool(tracker.initialize(context.device, context.graphicsQueue)), "Tracker initialization");
            requireGpuPage(bool(context.device.createStreamer({.dynamicBufferSizePerFrame = 1024, .queuedFrameCount = 2}).transform([&](auto rhiValue) { streamer = std::move(rhiValue); })), "Streamer creation");
            requireGpuPage(bool(context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); })) && bool(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); })), "Commands creation");
            requireGpuPage(bool(context.device.createBuffer({.size = decoded.size(),
                .usage = BufferUsageBits::TransferDestination | BufferUsageBits::TransferSource | BufferUsageBits::MemoryDecompression,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { destination = std::move(rhiValue); })), "GPU destination creation");
            requireGpuPage(bool(context.device.createBuffer({.size = decoded.size(), .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); })), "Readback creation");
            // Include cancelled recording and repeated slot reuse.
            for (uint32_t iteration = 0; iteration < 7; ++iteration) {
                requireGpuPage(bool(frame.begin(iteration)) && bool(pool->reset()) && bool(commands->begin(&frame)) &&
                    bool(streamer->beginFrame(frame)), "Frame begin");
                requireGpuPage(!streamer->streamDecompressedBufferData(stored, tiles, *destination, 1), "Misaligned destination accepted");
                requireGpuPage(streamer->streamDecompressedBufferData(stored, tiles, *destination, 0), "GPU page staging");
                auto receipt = streamer->pendingCopyCompletion();
                requireGpuPage(receipt && !receipt->isRecordedBefore(*commands) && !receipt->isComplete(), "Premature upload completion");
                commands->copyStreamedData(*streamer);
                requireGpuPage(receipt->isRecordedBefore(*commands), "Decode not covered by upload receipt");
                const BufferBarrierDesc barrier{.buffer = destination.get(), .before = ResourceState::General,
                    .after = ResourceState::TransferSource, .size = decoded.size()};
                commands->barrier({.buffers = &barrier, .bufferCount = 1});
                commands->copyBuffer({.source = destination.get(), .destination = readback.get(), .size = decoded.size()});
                requireGpuPage(bool(commands->end()), "Command end");
                streamer->endFrame();
                if (iteration == 0) {
                    frame.cancel();
                    requireGpuPage(receipt->isCancelled() && !receipt->isComplete(), "Cancelled decode published");
                    continue;
                }
                CommandBuffer* buffers[] = {commands.get()};
                requireGpuPage(bool(tracker.submit({.commandBuffers = buffers, .commandBufferCount = 1}, frame)) &&
                    bool(frame.wait(5'000'000'000ull)) && receipt->isComplete(), "GPU decode submission/completion");
                readback->invalidate(0, decoded.size());
                auto* data = readback->map();
                requireGpuPage(data != nullptr, "Readback mapping");
                bool equal = std::memcmp(data, decoded.data(), decoded.size()) == 0;
                readback->unmap();
                requireGpuPage(equal, "GPU GDeflate output differs from CPU oracle");
            }
            return RhiTestResult::pass("EXT GPU/CPU byte oracle, mixed 64 KiB tiles + tail, cancellation and slot reuse");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};

METALLIC_REGISTER_RHI_TEST(GpuPageCodecTest);
METALLIC_REGISTER_RHI_TEST(GpuPageDecompressionTest);

class GpuPageAdaptiveTest final : public RhiTest {
public:
    GpuPageAdaptiveTest() { name = "streamer_gpu_adaptive_batch"; type = RhiTestType::Command; }
    RhiTestResult run(RhiTestContext& context) override
    {
        using namespace render;
        if (!context.device.capabilities().memoryDecompression) { return RhiTestResult::skip("EXT GDeflate unavailable"); }
        try {
            checkGpuPageCodec(context.outputDirectory);
            scene::MeshletStreamAsset asset;
            std::string reason;
            requireGpuPage(asset.open(context.outputDirectory / "gpu_page_compressed.meshstream.bin", reason), reason);
            std::vector<uint8_t> reference;
            requireGpuPage(scene::decodeMeshletStreamGpuPage(asset.pages()[0], asset.pagePayload(0), reference, reason), reason);
            for (uint64_t threshold : {uint64_t(0), uint64_t(reference.size()), uint64_t(reference.size() + 1)}) {
                const bool expectGpu = threshold <= reference.size();
                MeshletStreamResidencyManager residency;
                requireGpuPage(residency.initialize({.asset = &asset, .maxResidentBytes = 4 * 1024 * 1024,
                    .queuedFrameCount = 2, .pageLoadConcurrency = 2, .maxPageLoadsInFlight = 4,
                    .gpuDecompression = true, .gpuDecompressionMinBatchBytes = threshold}, reason), reason);
                std::unique_ptr<Streamer> streamer;
                std::unique_ptr<Buffer> destination, readback;
                std::unique_ptr<CommandPool> pool;
                std::unique_ptr<CommandBuffer> commands;
                RenderFrameContext frame;
                QueueSubmissionTracker tracker;
                struct Drain { Queue& queue; RenderFrameContext& frame; ~Drain() { frame.cancel(); (void)queue.waitIdle(); } } drain{context.graphicsQueue, frame};
                requireGpuPage(bool(tracker.initialize(context.device, context.graphicsQueue)), "Tracker");
                requireGpuPage(bool(context.device.createStreamer({.dynamicBufferSizePerFrame = 1024, .queuedFrameCount = 2}).transform([&](auto rhiValue) { streamer = std::move(rhiValue); })), "Streamer");
                requireGpuPage(bool(context.device.createCommandPool(context.graphicsQueue).transform([&](auto rhiValue) { pool = std::move(rhiValue); })) && bool(pool->createCommandBuffer().transform([&](auto rhiValue) { commands = std::move(rhiValue); })), "Commands");
                requireGpuPage(bool(context.device.createBuffer({.size = residency.pageBufferSize(),
                    .usage = BufferUsageBits::TransferDestination | BufferUsageBits::TransferSource | BufferUsageBits::MemoryDecompression}).transform([&](auto rhiValue) { destination = std::move(rhiValue); })), "Destination");
                requireGpuPage(bool(context.device.createBuffer({.size = reference.size(), .usage = BufferUsageBits::TransferDestination,
                    .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto rhiValue) { readback = std::move(rhiValue); })), "Readback");
                (void)residency.requestPage(0); // Return value means already resident.
                requireGpuPage(residency.pageAllocated(0) && residency.queuedUploadCount() == 1, "Request admission");
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                uint32_t frameIndex = 0;
                while (!residency.pageResident(0) && std::chrono::steady_clock::now() < deadline) {
                    requireGpuPage(bool(frame.begin(frameIndex++)) && bool(pool->reset()) && bool(commands->begin(&frame)) && bool(streamer->beginFrame(frame)), "Begin");
                    residency.beginFrame();
                    const uint32_t uploads = residency.processUploads(*streamer, *destination, 1);
                    commands->copyStreamedData(*streamer);
                    if (uploads) {
                        requireGpuPage(residency.throughputSnapshot().totals.geometryReadyPages == 0, "Premature ready counter");
                        const BufferBarrierDesc barrier{.buffer = destination.get(),
                            .before = expectGpu ? ResourceState::General : ResourceState::TransferDestination,
                            .after = ResourceState::TransferSource, .size = destination->desc().size};
                        commands->barrier({.buffers = &barrier, .bufferCount = 1});
                        commands->copyBuffer({.source = destination.get(), .destination = readback.get(),
                            .sourceOffset = residency.deviceOffsetForPage(0), .size = reference.size()});
                    }
                    requireGpuPage(bool(commands->end()), "End");
                    streamer->endFrame();
                    CommandBuffer* buffers[] = {commands.get()};
                    requireGpuPage(bool(tracker.submit({.commandBuffers = buffers, .commandBufferCount = 1}, frame)) && bool(frame.wait(5'000'000'000ull)), "Submit");
                    std::this_thread::yield();
                }
                requireGpuPage(residency.pageResident(0), "Page did not complete");
                const auto traffic = residency.throughputSnapshot().totals;
                requireGpuPage(traffic.loadedPages == 1 && traffic.geometryReadyPages == 1 &&
                    traffic.geometryReadyBytes == reference.size() && traffic.loadedStoredBytes == asset.pages()[0].payloadSize,
                    "Load/ready accounting mismatch");
                requireGpuPage(traffic.smallBatchCpuPages == (expectGpu ? 0 : 1) &&
                    residency.stats(false).totalGpuDecompressedPages == (expectGpu ? 1 : 0), "Adaptive policy mismatch");
                readback->invalidate();
                const auto* data = readback->map();
                requireGpuPage(data != nullptr, "Readback map");
                const bool equal = std::memcmp(data, reference.data(), reference.size()) == 0;
                readback->unmap();
                requireGpuPage(equal, "Adaptive CPU/GPU geometry differs");
            }
            return RhiTestResult::pass("Worker CPU / GPU threshold boundary, equal installed bytes and completion-only speed counters");
        } catch (const std::exception& error) { return RhiTestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(GpuPageAdaptiveTest);
} // namespace
} // namespace metallic::tests

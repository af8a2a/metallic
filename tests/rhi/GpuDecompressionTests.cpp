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
            requireGpuPage(bool(context.device.createStreamer({.dynamicBufferSizePerFrame = 1024, .queuedFrameCount = 2}, streamer)), "Streamer creation");
            requireGpuPage(bool(context.device.createCommandPool(context.graphicsQueue, pool)) && bool(pool->createCommandBuffer(commands)), "Commands creation");
            requireGpuPage(bool(context.device.createBuffer({.size = decoded.size(),
                .usage = BufferUsageBits::TransferDestination | BufferUsageBits::TransferSource | BufferUsageBits::MemoryDecompression,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}, destination)), "GPU destination creation");
            requireGpuPage(bool(context.device.createBuffer({.size = decoded.size(), .usage = BufferUsageBits::TransferDestination,
                .memoryLocation = MemoryLocation::HostReadback}, readback)), "Readback creation");
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
} // namespace
} // namespace metallic::tests

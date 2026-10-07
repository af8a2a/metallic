#pragma once

#include "Runtime/Render/GAPI/RHI.h"

namespace metallic::render {
class RenderFrameContext;
class StreamUploadCompletion;
namespace detail { struct StreamerImpl; }

struct StreamDecompressionTile {
    uint32_t sourceOffset = 0;
    uint32_t destinationOffset = 0;
    uint32_t storedBytes = 0;
    uint32_t decodedBytes = 0;
    bool compressed = false;
};

// Called at recording boundaries, outside the Streamer lock. Callbacks may query
// or stage other batches, but must not change frame/command-buffer lifecycle.
using StreamUploadPhaseCallback = std::function<void(const char*)>;

// Opaque, single-use identity. Zero selects the compatibility/default batch in
// upload/copy APIs; beginCopyBatch() always returns a nonzero identity.
struct StreamerCopyBatch {
    uint64_t id = 0;
    bool valid() const { return id != 0; }
};

struct BufferOffset {
    class Buffer* buffer = nullptr;
    uint64_t offset = 0;

    bool valid() const { return buffer != nullptr; }
};

struct StreamDataChunk {
    const void* data = nullptr;
    uint64_t size = 0;
};

struct StreamerDesc {
    uint64_t constantBufferSize = 0;
    MemoryLocation constantBufferMemoryLocation = MemoryLocation::HostUpload;
    MemoryLocation dynamicBufferMemoryLocation = MemoryLocation::HostUpload;
    BufferDesc dynamicBufferDesc{
        .size = 0,
        .structureStride = 0,
        .usage = BufferUsageBits::TransferSource,
        .memoryLocation = MemoryLocation::HostUpload,
    };
    uint64_t dynamicBufferSizePerFrame = 1024ull * 1024ull;
    uint32_t queuedFrameCount = 2;
    QueueAccessBits constantBufferQueueAccess = QueueAccessBits::Graphics;
};

struct StreamerPendingCopyStats {
    uint32_t bufferCopyCount = 0;
    uint32_t textureCopyCount = 0;
    uint64_t bufferCopyBytes = 0;
    uint64_t textureCopyBytes = 0;

    uint32_t copyCount() const { return bufferCopyCount + textureCopyCount; }
    uint64_t copyBytes() const { return bufferCopyBytes + textureCopyBytes; }
};

struct StreamerStats {
    uint64_t frameIndex = 0;
    uint32_t frameSlot = 0;
    uint32_t queuedFrameCount = 0;
    uint64_t dynamicBufferSizePerFrame = 0;
    uint64_t dynamicBufferOffset = 0;
    uint64_t constantBufferOffset = 0;
    uint64_t currentFrameDynamicBytes = 0;
    uint64_t lastFrameDynamicBytes = 0;
    uint64_t peakFrameDynamicBytes = 0;
    uint64_t totalDynamicBytes = 0;
    uint64_t currentFrameConstantBytes = 0;
    uint64_t lastFrameConstantBytes = 0;
    uint64_t peakFrameConstantBytes = 0;
    uint64_t totalConstantBytes = 0;
    uint32_t currentFrameDynamicRequestCount = 0;
    uint32_t lastFrameDynamicRequestCount = 0;
    uint32_t currentFrameConstantRequestCount = 0;
    uint32_t lastFrameConstantRequestCount = 0;
    uint32_t garbageBufferCount = 0;
    StreamerPendingCopyStats pendingCopies;
};

struct StreamBufferDataDesc {
    std::span<const StreamDataChunk> dataChunks;
    uint32_t placementAlignment = 1;
    class Buffer* dstBuffer = nullptr;
    uint64_t dstOffset = 0;
    StreamerCopyBatch copyBatch;
};

struct StreamTextureDataDesc {
    const void* data = nullptr;
    uint32_t dataRowPitch = 0;
    uint32_t dataSlicePitch = 0;
    class Texture* dstTexture = nullptr;
    uint32_t dstMipLevel = 0;
    uint32_t dstBaseLayer = 0;
    uint32_t dstLayerCount = 1;
    int32_t dstOffsetX = 0;
    int32_t dstOffsetY = 0;
    int32_t dstOffsetZ = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    StreamerCopyBatch copyBatch;
};

// Shared staging and distinct copy batches are thread-safe. Populate a batch
// before recording it, use a separate command pool/context per recording worker,
// and externally order operations on the same batch. Default destination copies
// belong to the coordinator; their flush never consumes explicit batches. Data-
// only and constant uploads may run on workers. Destination resources
// must survive recording, and callers still supply GPU barriers/queue ordering.
// Join all users before beginFrame/endFrame, frame lifecycle, move or destruction.
// beginFrame(frame) retains arenas until GPU completion. Without it, callers must
// also prevent ring reuse/resource destruction while the GPU reads staged data.
class Streamer {
    METALLIC_RHI_HANDLE(Streamer, unique_ptr,
        friend Result<std::unique_ptr<Streamer>> createStreamer(Device&, const StreamerDesc&);
    )

    const StreamerDesc& desc() const;
    StreamerStats stats() const;
    // stats().pendingCopies aggregates all batches; this query covers only the
    // selected batch. Invalid/consumed identities return empty counts.
    StreamerPendingCopyStats pendingCopyStats(StreamerCopyBatch batch = {}) const;
    Buffer* constantBuffer() const;
    BufferOffset streamBufferData(const StreamBufferDataDesc& desc);
    bool streamDecompressedBufferData(std::span<const uint8_t> stored,
        std::span<const StreamDecompressionTile> tiles, Buffer& destination, uint64_t destinationOffset,
        StreamerCopyBatch batch = {});
    BufferOffset streamTextureData(const StreamTextureDataDesc& desc);
    uint64_t streamConstantData(const void* data, uint64_t byteSize);
    Result<> beginFrame(RenderFrameContext& frame);
    [[nodiscard]] Result<StreamerCopyBatch> beginCopyBatch();
    // Cancels only pending explicit batches; zero/consumed/foreign IDs fail.
    [[nodiscard]] Result<> cancelCopyBatch(StreamerCopyBatch batch);
    // Covers copies queued in this batch. Returns null without
    // beginFrame(frame), or when no copies are pending. See StreamUploadCompletion.h.
    std::shared_ptr<StreamUploadCompletion> pendingCopyCompletion(StreamerCopyBatch batch = {});
    [[nodiscard]] Result<> copyStreamedData(CommandBuffer& commandBuffer, const StreamUploadPhaseCallback& phase = {});
    // Consumes the batch, including on failure. Pending stats then exclude it.
    [[nodiscard]] Result<> copyStreamedData(CommandBuffer& commandBuffer, StreamerCopyBatch batch,
        const StreamUploadPhaseCallback& phase = {});
    void endFrame();

};

[[nodiscard]] Result<std::unique_ptr<Streamer>> createStreamer(Device& device, const StreamerDesc& desc);

} // namespace metallic::render

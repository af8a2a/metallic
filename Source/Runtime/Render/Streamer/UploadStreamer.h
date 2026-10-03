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

// Called at recording boundaries, under the Streamer lock. Profiling callbacks
// must not reenter the Streamer. Existing callers may omit the callback.
using StreamUploadPhaseCallback = std::function<void(const char*)>;

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
};

class Streamer {
    METALLIC_RHI_HANDLE(Streamer, unique_ptr,
        friend Result<std::unique_ptr<Streamer>> createStreamer(Device&, const StreamerDesc&);
    )

    const StreamerDesc& desc() const;
    StreamerStats stats() const;
    Buffer* constantBuffer() const;
    BufferOffset streamBufferData(const StreamBufferDataDesc& desc);
    bool streamDecompressedBufferData(std::span<const uint8_t> stored,
        std::span<const StreamDecompressionTile> tiles, Buffer& destination, uint64_t destinationOffset);
    BufferOffset streamTextureData(const StreamTextureDataDesc& desc);
    uint64_t streamConstantData(const void* data, uint64_t byteSize);
    Result<> beginFrame(RenderFrameContext& frame);
    // Covers copies currently queued for the next flush. Returns null without
    // beginFrame(frame), or when no copies are pending. See StreamUploadCompletion.h.
    std::shared_ptr<StreamUploadCompletion> pendingCopyCompletion();
    [[nodiscard]] Result<> copyStreamedData(CommandBuffer& commandBuffer, const StreamUploadPhaseCallback& phase = {});
    void endFrame();

};

[[nodiscard]] Result<std::unique_ptr<Streamer>> createStreamer(Device& device, const StreamerDesc& desc);

} // namespace metallic::render

#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/GAPI/TextureFormat.h"
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Streamer/StreamUploadCompletion.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Profiling/NsightEvents.h"
#include "Runtime/Render/Profiling/CPUPhaseTrace.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace metallic::render {
namespace {

std::atomic<uint64_t> nextCopyBatchId{1};

constexpr uint64_t kDynamicBufferChunkSize = 64ull * 1024ull;
constexpr uint64_t kInvalidStreamOffset = std::numeric_limits<uint64_t>::max();

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

uint32_t textureDimensionAtMip(uint32_t value, uint32_t mipLevel)
{
    for (uint32_t index = 0; index < mipLevel; ++index) {
        value = std::max(value / 2u, 1u);
    }
    return value;
}

uint64_t textureCopyByteSize(const BufferTextureRegion& copy)
{
    return static_cast<uint64_t>(copy.bufferSlicePitch) *
        static_cast<uint64_t>(copy.depth) *
        static_cast<uint64_t>(copy.layerCount);
}

} // namespace

namespace detail {

struct StreamerImpl {
    struct BufferCopyRequest {
        Buffer* destination = nullptr;
        uint64_t destinationOffset = 0;
        Buffer* source = nullptr;
        uint64_t sourceOffset = 0;
        uint64_t size = 0;
    };

    struct TextureCopyRequest {
        BufferTextureRegion copy;
    };

    struct BufferGarbage {
        std::shared_ptr<Buffer> buffer;
        uint32_t frameCount = 0;
    };

    struct CopyBatch {
        std::shared_ptr<StreamUploadCompletion> pendingCompletion;
        std::vector<BufferCopyRequest> bufferRequests;
        std::vector<BufferDecompressionDesc> decompressions;
        std::vector<BufferBarrierDesc> decompressionCopyBarriers;
        std::vector<BufferBarrierDesc> decompressionInputBarriers;
        std::vector<BufferBarrierDesc> decompressionOutputBarriers;
        std::vector<TextureCopyRequest> textureRequests;

        void clear()
        {
            pendingCompletion.reset();
            bufferRequests.clear();
            textureRequests.clear();
            decompressions.clear();
            decompressionCopyBarriers.clear();
            decompressionInputBarriers.clear();
            decompressionOutputBarriers.clear();
        }

        void cancel()
        {
            if (pendingCompletion) { pendingCompletion->submission_->cancel(); }
            clear();
        }

        void accumulate(StreamerPendingCopyStats& stats) const
        {
            stats.bufferCopyCount += static_cast<uint32_t>(bufferRequests.size());
            stats.textureCopyCount += static_cast<uint32_t>(textureRequests.size());
            for (const auto& request : bufferRequests) { stats.bufferCopyBytes += request.size; }
            for (const auto& request : textureRequests) { stats.textureCopyBytes += textureCopyByteSize(request.copy); }
        }
    };

    // Retain one holder from the coordinator. Worker staging only changes this
    // holder under mutex; it never mutates RenderFrameContext's resource list.
    struct UploadResources {
        std::vector<std::shared_ptr<Buffer>> buffers;
    };

    struct UploadSlot {
        std::shared_ptr<UploadResources> resources;
        GPUCompletionPoint completion;
        uint64_t dynamicOffset = 0;
        uint64_t constantOffset = 0;
        std::shared_ptr<Buffer> compressed;
        uint64_t compressedOffset = 0;
    };

    explicit StreamerImpl(Device& streamerDevice)
        : device(&streamerDevice)
    {
    }

    ~StreamerImpl()
    {
        defaultBatch.cancel();
        for (auto& [id, batch] : batches) { batch.cancel(); }
    }

    Result<> create(const StreamerDesc& streamerDesc)
    {
        if (device == nullptr || streamerDesc.queuedFrameCount == 0) {
            return makeError(Error::InvalidArgument);
        }

        desc = streamerDesc;
        const uint64_t constantAlignment = std::max<uint64_t>(1, device->capabilities().constantBufferOffsetAlignment);
        if (desc.constantBufferSize > UINT64_MAX - (constantAlignment - 1)) {
            return makeError(Error::InvalidArgument);
        }
        constantBufferStride = alignUp(desc.constantBufferSize, constantAlignment);
        if (constantBufferStride > UINT64_MAX / desc.queuedFrameCount) {
            return makeError(Error::InvalidArgument);
        }
        if (desc.dynamicBufferSizePerFrame == 0) {
            desc.dynamicBufferSizePerFrame = kDynamicBufferChunkSize;
        }
        const uint64_t maxSizePerFrame = (UINT64_MAX / desc.queuedFrameCount / kDynamicBufferChunkSize) * kDynamicBufferChunkSize;
        if (desc.dynamicBufferSizePerFrame > maxSizePerFrame) {
            return makeError(Error::InvalidArgument);
        }
        dynamicBufferSizePerFrame = alignUp(desc.dynamicBufferSizePerFrame, kDynamicBufferChunkSize);

        if (desc.constantBufferSize > 0) {
            BufferDesc bufferDesc{
                .size = constantBufferStride * desc.queuedFrameCount,
                .usage = BufferUsageBits::Constant,
                .memoryLocation = desc.constantBufferMemoryLocation,
                .queueAccess = desc.constantBufferQueueAccess,
            };
            std::unique_ptr<Buffer> buffer;
            Result<> result = device->createBuffer(bufferDesc).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
            if (!result || buffer == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }
            constantHostAlignment = buffer->hostWriteAlignment();
            while (constantBufferStride % constantHostAlignment) {
                if (constantBufferStride > UINT64_MAX - (constantHostAlignment - 1)) { return makeError(Error::InvalidArgument); }
                constantBufferStride = alignUp(constantBufferStride, constantHostAlignment);
                if (constantBufferStride > UINT64_MAX / desc.queuedFrameCount) { return makeError(Error::InvalidArgument); }
                bufferDesc.size = constantBufferStride * desc.queuedFrameCount;
                result = device->createBuffer(bufferDesc).transform([&](auto value) { buffer = std::move(value); });
                if (!result) { return result; }
                constantHostAlignment = buffer->hostWriteAlignment();
            }
            constantBuffer = std::move(buffer);
        }
        return {};
    }

    Result<> beginFrame(RenderFrameContext& frame)
    {
        std::lock_guard lock(mutex);
        if (!frame.recording() || frame.slotIndex() >= desc.queuedFrameCount ||
            !defaultBatch.bufferRequests.empty() || !defaultBatch.textureRequests.empty() || !batches.empty() ||
            (activeFrame != nullptr && activeFrame != &frame)) {
            return makeError(Error::InvalidArgument);
        }
        uploadSlots.resize(desc.queuedFrameCount);
        UploadSlot& slot = uploadSlots[frame.slotIndex()];
        if (!slot.completion.sameSubmission(frame.completion())) {
            if (!slot.completion.isComplete()) {
                return makeError(Error::InvalidArgument);
            }
            slot.resources = std::make_shared<UploadResources>();
            if (constantBuffer) { slot.resources->buffers.push_back(constantBuffer); }
            if (dynamicBuffer) { slot.resources->buffers.push_back(dynamicBuffer); }
            if (slot.compressed) { slot.resources->buffers.push_back(slot.compressed); }
            frame.retain(slot.resources);
            slot.completion = frame.completion();
            slot.dynamicOffset = slot.constantOffset = slot.compressedOffset = 0;
            dynamicBufferOffset = 0;
            constantBufferOffset = 0;
        } else if (activeFrame == nullptr) {
            dynamicBufferOffset = slot.dynamicOffset;
            constantBufferOffset = slot.constantOffset;
        }
        frameIndex = frame.slotIndex();
        activeFrame = &frame;
        completionTracked = true;
        return {};
    }

    Result<> ensureDynamicBuffer(uint64_t requiredSizePerFrame)
    {
        if (device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (dynamicBuffer != nullptr && requiredSizePerFrame <= dynamicBufferSizePerFrame) {
            return {};
        }

        const uint64_t maxSizePerFrame = (UINT64_MAX / desc.queuedFrameCount / kDynamicBufferChunkSize) * kDynamicBufferChunkSize;
        if (requiredSizePerFrame > maxSizePerFrame) {
            return makeError(Error::InvalidArgument);
        }
        uint64_t capacity = std::max(requiredSizePerFrame, dynamicBufferSizePerFrame);
        if (dynamicBuffer != nullptr) {
            // Each pending copy still references its original buffer. Growing
            // one 64 KiB chunk at a time retains a near-full allocation per
            // page, amplifying cold-start memory residency and release costs.
            const uint64_t doubled = dynamicBufferSizePerFrame > maxSizePerFrame / 2
                ? maxSizePerFrame : dynamicBufferSizePerFrame * 2;
            capacity = std::max(capacity, doubled);
        }
        const uint64_t newSizePerFrame = alignUp(capacity, kDynamicBufferChunkSize);
        BufferDesc bufferDesc = desc.dynamicBufferDesc;
        bufferDesc.size = newSizePerFrame * desc.queuedFrameCount;
        bufferDesc.usage = bufferDesc.usage | BufferUsageBits::TransferSource;
        bufferDesc.memoryLocation = desc.dynamicBufferMemoryLocation;

        profiling::CPUPhase allocationPhase("stream.grow", bufferDesc.size);
        std::unique_ptr<Buffer> newBuffer;
        Result<> result = device->createBuffer(bufferDesc).transform([&](auto rhiValue) { newBuffer = std::move(rhiValue); });
        if (!result || newBuffer == nullptr) {
            return result ? makeError(Error::Failure) : result;
        }

        if (dynamicBuffer != nullptr && activeFrame == nullptr) {
            garbage.push_back(BufferGarbage{
                .buffer = dynamicBuffer,
                .frameCount = 0,
            });
        }
        dynamicBuffer = std::move(newBuffer);
        dynamicHostAlignment = dynamicBuffer->hostWriteAlignment();
        if (activeFrame) { uploadSlots[frameIndex].resources->buffers.push_back(dynamicBuffer); }
        dynamicBufferSizePerFrame = newSizePerFrame;
        return {};
    }

    uint64_t reserveDynamicOffset(uint64_t size, uint64_t alignment)
    {
        // Independently submitted batches must not share a non-coherent flush
        // atom. Recheck after growth in case the allocation's memory type changes.
        for (;;) {
            alignment = std::max(alignment, dynamicHostAlignment);
            if (dynamicBufferOffset > UINT64_MAX - (alignment - 1)) { return kInvalidStreamOffset; }
            const uint64_t offset = alignUp(dynamicBufferOffset, alignment);
            if (size > UINT64_MAX - offset || !ensureDynamicBuffer(offset + size)) { return kInvalidStreamOffset; }
            if (dynamicHostAlignment == 1) { return offset; }
            if (dynamicBufferSizePerFrame % dynamicHostAlignment) { return kInvalidStreamOffset; }
            if (offset % dynamicHostAlignment == 0) { return offset; }
        }
    }

    BufferOffset streamBufferData(const StreamBufferDataDesc& streamDesc)
    {
        std::lock_guard lock(mutex);
        return stageBufferData(streamDesc);
    }

    BufferOffset stageBufferData(const StreamBufferDataDesc& streamDesc)
    {
        auto* batch = findBatch(streamDesc.copyBatch);
        if (!batch) { return {}; }
        if (completionTracked && (activeFrame == nullptr || !activeFrame->recording())) {
            return {};
        }
        if (streamDesc.dataChunks.size() == 0 ||
            streamDesc.dataChunks.size() > UINT32_MAX ||
            desc.queuedFrameCount == 0) {
            return {};
        }

        uint64_t dataSize = 0;
        for (uint32_t index = 0; index < streamDesc.dataChunks.size(); ++index) {
            const StreamDataChunk& chunk = streamDesc.dataChunks[index];
            if (chunk.size > 0 && chunk.data == nullptr) {
                return {};
            }
            if (chunk.size > UINT64_MAX - dataSize) { return {}; }
            dataSize += chunk.size;
        }
        if (dataSize == 0 || dataSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return {};
        }

        const profiling::NsightProfileRange uploadMarker(
            profiling::NsightDomain::Render,
            "Buffer Upload",
            profiling::NsightCategory::ResourceUpload,
            dataSize);

        const uint64_t alignment = std::max<uint64_t>(
            std::max(streamDesc.placementAlignment, 1u),
            device != nullptr ? device->capabilities().bufferCopyOffsetAlignment : 1);
        const uint64_t localOffset = reserveDynamicOffset(dataSize, alignment);
        if (localOffset == kInvalidStreamOffset) { return {}; }
        const uint64_t requiredSizePerFrame = localOffset + dataSize;

        const uint64_t bufferOffset =
            static_cast<uint64_t>(frameIndex) * dynamicBufferSizePerFrame + localOffset;
        void* mapped = dynamicBuffer->map();
        if (mapped == nullptr) {
            return {};
        }

        uint8_t* dst = static_cast<uint8_t*>(mapped) + bufferOffset;
        for (uint32_t index = 0; index < streamDesc.dataChunks.size(); ++index) {
            const StreamDataChunk& chunk = streamDesc.dataChunks[index];
            if (chunk.size == 0) {
                continue;
            }
            std::memcpy(dst, chunk.data, static_cast<size_t>(chunk.size));
            dst += chunk.size;
        }
        dynamicBuffer->flush({bufferOffset, dataSize});
        dynamicBuffer->unmap();

        if (streamDesc.dstBuffer != nullptr) {
            batch->bufferRequests.push_back(BufferCopyRequest{
                .destination = streamDesc.dstBuffer,
                .destinationOffset = streamDesc.dstOffset,
                .source = dynamicBuffer.get(),
                .sourceOffset = bufferOffset,
                .size = dataSize,
            });
        }

        dynamicBufferOffset = requiredSizePerFrame;
        currentFrameDynamicBytes += dataSize;
        ++currentFrameDynamicRequestCount;
        return BufferOffset{
            .buffer = dynamicBuffer.get(),
            .offset = bufferOffset,
        };
    }

    BufferOffset streamTextureData(const StreamTextureDataDesc& streamDesc)
    {
        std::lock_guard lock(mutex);
        auto* batch = findBatch(streamDesc.copyBatch);
        if (!batch) { return {}; }
        if (completionTracked && (activeFrame == nullptr || !activeFrame->recording())) {
            return {};
        }
        if (streamDesc.data == nullptr ||
            streamDesc.dstTexture == nullptr ||
            desc.queuedFrameCount == 0) {
            return {};
        }

        const TextureDesc& textureDesc = streamDesc.dstTexture->desc();
        const auto format = formatInfo(textureDesc.format);
        const uint32_t blockExtent = format.blockExtent;
        const uint32_t bytesPerTexel = format.bytesPerBlock;
        if (bytesPerTexel == 0 || streamDesc.dstMipLevel >= textureDesc.mipCount) {
            return {};
        }

        const uint32_t mipWidth = textureDimensionAtMip(textureDesc.width, streamDesc.dstMipLevel);
        const uint32_t mipHeight = textureDimensionAtMip(textureDesc.height, streamDesc.dstMipLevel);
        const uint32_t mipDepth = textureDimensionAtMip(textureDesc.depth, streamDesc.dstMipLevel);
        if (streamDesc.dstOffsetX < 0 ||
            streamDesc.dstOffsetY < 0 ||
            streamDesc.dstOffsetZ < 0 ||
            static_cast<uint32_t>(streamDesc.dstOffsetX) >= mipWidth ||
            static_cast<uint32_t>(streamDesc.dstOffsetY) >= mipHeight ||
            static_cast<uint32_t>(streamDesc.dstOffsetZ) >= mipDepth ||
            streamDesc.dstBaseLayer >= textureDesc.layerCount ||
            streamDesc.dstLayerCount == 0 ||
            streamDesc.dstBaseLayer + streamDesc.dstLayerCount > textureDesc.layerCount) {
            return {};
        }

        const uint32_t width = streamDesc.width == 0
            ? mipWidth - static_cast<uint32_t>(streamDesc.dstOffsetX)
            : streamDesc.width;
        const uint32_t height = streamDesc.height == 0
            ? mipHeight - static_cast<uint32_t>(streamDesc.dstOffsetY)
            : streamDesc.height;
        const uint32_t depth = streamDesc.depth == 0
            ? mipDepth - static_cast<uint32_t>(streamDesc.dstOffsetZ)
            : streamDesc.depth;
        if (width == 0 ||
            height == 0 ||
            depth == 0 ||
            static_cast<uint32_t>(streamDesc.dstOffsetX) + width > mipWidth ||
            static_cast<uint32_t>(streamDesc.dstOffsetY) + height > mipHeight ||
            static_cast<uint32_t>(streamDesc.dstOffsetZ) + depth > mipDepth) {
            return {};
        }

        if (uint32_t(streamDesc.dstOffsetX) % blockExtent || uint32_t(streamDesc.dstOffsetY) % blockExtent ||
            (width % blockExtent && uint32_t(streamDesc.dstOffsetX) + width != mipWidth) ||
            (height % blockExtent && uint32_t(streamDesc.dstOffsetY) + height != mipHeight)) { return {}; }
        const auto sourceFootprint = textureCopyFootprint(textureDesc.format, width, height,
            uint64_t(depth) * streamDesc.dstLayerCount, streamDesc.dataRowPitch, streamDesc.dataSlicePitch);
        if (!sourceFootprint || sourceFootprint->requiredBytes > std::numeric_limits<size_t>::max()) { return {}; }
        const uint64_t rows = sourceFootprint->rows;
        const uint64_t rowSize = sourceFootprint->rowBytes;
        const uint64_t sourceRowPitch = sourceFootprint->rowPitch;
        const uint64_t sourceSlicePitch = sourceFootprint->slicePitch;

        const DeviceCapabilities& capabilities = device->capabilities();
        const uint64_t rowPitch = alignUp(rowSize, std::max<uint64_t>(capabilities.textureUploadRowPitchAlignment, bytesPerTexel));
        const uint64_t slicePitch = alignUp(
            rowPitch * rows,
            capabilities.textureUploadSlicePitchAlignment);
        const uint64_t copySliceCount =
            static_cast<uint64_t>(depth) * static_cast<uint64_t>(streamDesc.dstLayerCount);
        const auto destinationFootprint = textureCopyFootprint(textureDesc.format, width, height,
            copySliceCount, rowPitch, slicePitch);
        if (!destinationFootprint || slicePitch > std::numeric_limits<uint64_t>::max() / copySliceCount) { return {}; }
        const uint64_t dataSize = slicePitch * copySliceCount;
        if (dataSize == 0 ||
            rowPitch > std::numeric_limits<uint32_t>::max() ||
            slicePitch > std::numeric_limits<uint32_t>::max() ||
            dataSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return {};
        }

        const profiling::NsightProfileRange uploadMarker(
            profiling::NsightDomain::Render,
            "Texture Upload",
            profiling::NsightCategory::ResourceUpload,
            dataSize);

        const uint64_t localOffset = reserveDynamicOffset(
            dataSize,
            std::max<uint64_t>(capabilities.textureUploadBufferOffsetAlignment, bytesPerTexel));
        if (localOffset == kInvalidStreamOffset) { return {}; }
        const uint64_t requiredSizePerFrame = localOffset + dataSize;

        const uint64_t bufferOffset =
            static_cast<uint64_t>(frameIndex) * dynamicBufferSizePerFrame + localOffset;
        void* mapped = dynamicBuffer->map();
        if (mapped == nullptr) {
            return {};
        }

        uint8_t* const dstBase = static_cast<uint8_t*>(mapped) + bufferOffset;
        const uint8_t* const srcBase = static_cast<const uint8_t*>(streamDesc.data);
        for (uint64_t sliceIndex = 0; sliceIndex < copySliceCount; ++sliceIndex) {
            for (uint32_t rowIndex = 0; rowIndex < rows; ++rowIndex) {
                uint8_t* dst = dstBase + sliceIndex * slicePitch + rowIndex * rowPitch;
                const uint8_t* src =
                    srcBase + sliceIndex * sourceSlicePitch + rowIndex * sourceRowPitch;
                std::memcpy(dst, src, static_cast<size_t>(rowSize));
            }
        }
        dynamicBuffer->flush({bufferOffset, dataSize});
        dynamicBuffer->unmap();

        auto copySlice = dynamicBuffer->slice({bufferOffset, dataSize});
        if (!copySlice) { return {}; }
        batch->textureRequests.push_back(TextureCopyRequest{
            .copy = BufferTextureRegion{
                .texture = streamDesc.dstTexture,
                .buffer = *copySlice,
                .bufferRowPitch = static_cast<uint32_t>(rowPitch),
                .bufferSlicePitch = static_cast<uint32_t>(slicePitch),
                .textureOffsetX = streamDesc.dstOffsetX,
                .textureOffsetY = streamDesc.dstOffsetY,
                .textureOffsetZ = streamDesc.dstOffsetZ,
                .width = width,
                .height = height,
                .depth = depth,
                .mipLevel = streamDesc.dstMipLevel,
                .baseLayer = streamDesc.dstBaseLayer,
                .layerCount = streamDesc.dstLayerCount,
            },
        });

        dynamicBufferOffset = requiredSizePerFrame;
        currentFrameDynamicBytes += dataSize;
        ++currentFrameDynamicRequestCount;
        return BufferOffset{
            .buffer = dynamicBuffer.get(),
            .offset = bufferOffset,
        };
    }

    uint64_t streamConstantData(const void* data, uint64_t byteSize)
    {
        std::lock_guard lock(mutex);
        if (completionTracked && (activeFrame == nullptr || !activeFrame->recording())) {
            return kInvalidStreamOffset;
        }
        if (constantBuffer == nullptr ||
            (byteSize > 0 && data == nullptr) ||
            byteSize > desc.constantBufferSize ||
            byteSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return kInvalidStreamOffset;
        }

        const profiling::NsightProfileRange uploadMarker(
            profiling::NsightDomain::Render,
            "Constant Upload",
            profiling::NsightCategory::ResourceUpload,
            byteSize);

        const uint64_t alignment = std::max(constantHostAlignment,
            device->capabilities().constantBufferOffsetAlignment);
        if (constantBufferOffset > UINT64_MAX - (alignment - 1)) { return kInvalidStreamOffset; }
        uint64_t offset = alignUp(constantBufferOffset, alignment);
        if (offset > desc.constantBufferSize || byteSize > desc.constantBufferSize - offset) {
            if (completionTracked) {
                return kInvalidStreamOffset;
            }
            offset = 0;
        }
        if (byteSize > desc.constantBufferSize - offset) {
            return kInvalidStreamOffset;
        }

        const uint64_t bufferOffset = offset +
            (completionTracked ? static_cast<uint64_t>(frameIndex) * constantBufferStride : 0);
        if (byteSize > 0) {
            void* mapped = constantBuffer->map();
            if (mapped == nullptr) {
                return kInvalidStreamOffset;
            }
            std::memcpy(
                static_cast<uint8_t*>(mapped) + bufferOffset,
                data,
                static_cast<size_t>(byteSize));
            constantBuffer->flush({bufferOffset, byteSize});
            constantBuffer->unmap();
        }
        constantBufferOffset = offset + byteSize;
        currentFrameConstantBytes += byteSize;
        ++currentFrameConstantRequestCount;
        return bufferOffset;
    }

    bool streamDecompressedBufferData(std::span<const uint8_t> stored,
        std::span<const StreamDecompressionTile> tiles, Buffer& destination, uint64_t destinationOffset, StreamerCopyBatch copyBatch)
    {
        std::lock_guard lock(mutex);
        auto* batch = findBatch(copyBatch);
        if (!batch) { return false; }
        if (!activeFrame || !activeFrame->recording() || !device->capabilities().memoryDecompression ||
            !hasFlag(destination.desc().usage, BufferUsageBits::MemoryDecompression) ||
            !hasFlag(destination.desc().usage, BufferUsageBits::TransferDestination) || tiles.empty() || stored.empty()) { return false; }
        uint64_t decodedEnd = 0;
        for (const auto& tile : tiles) {
            if (!tile.storedBytes || !tile.decodedBytes || tile.decodedBytes > 65536 ||
                tile.sourceOffset % 4 || tile.destinationOffset % 4 || tile.destinationOffset != decodedEnd ||
                tile.sourceOffset > stored.size() || tile.storedBytes > stored.size() - tile.sourceOffset ||
                (!tile.compressed && (tile.storedBytes != tile.decodedBytes || tile.storedBytes % 4))) { return false; }
            decodedEnd += tile.decodedBytes;
        }
        if (destinationOffset % 4 || destinationOffset > destination.desc().size ||
            decodedEnd > destination.desc().size - destinationOffset) { return false; }
        auto& slot = uploadSlots[frameIndex];
        const uint64_t offset = alignUp(slot.compressedOffset, 16);
        // Bound the retained compressed input per frame slot. Admission retries
        // next frame when the slot is full, retaining the safe resident cut.
        constexpr uint64_t kMaxCompressedBytes = 64ull * 1024 * 1024;
        if (stored.size() > kMaxCompressedBytes || offset > kMaxCompressedBytes - stored.size()) { return false; }
        const uint64_t required = offset + stored.size();
        if (!slot.compressed || slot.compressed->desc().size < required) {
            const uint64_t capacity = std::min(kMaxCompressedBytes, std::max(alignUp(required, kDynamicBufferChunkSize),
                slot.compressed ? slot.compressed->desc().size * 2 : 4ull * 1024 * 1024));
            std::unique_ptr<Buffer> buffer;
            if (!device->createBuffer({.size = capacity,
                    .usage = BufferUsageBits::TransferDestination | BufferUsageBits::MemoryDecompression,
                    .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); })) { return false; }
            slot.compressed = std::move(buffer);
            slot.resources->buffers.push_back(slot.compressed);
        }
        std::vector<BufferDecompressionDesc> regions;
        for (const auto& tile : tiles) {
            if (!tile.compressed) { continue; }
            auto source = slot.compressed->slice({offset + tile.sourceOffset, tile.storedBytes});
            auto target = destination.slice({destinationOffset + tile.destinationOffset, tile.decodedBytes});
            if (!source || !target) { return false; }
            regions.push_back({*source, *target});
        }
        const StreamDataChunk chunk{stored.data(), stored.size()};
        const auto staged = stageBufferData({.dataChunks = {&chunk, 1}, .placementAlignment = 16, .copyBatch = copyBatch});
        if (!staged.valid()) { return false; }
        slot.compressedOffset = required;
        for (const auto& tile : tiles) {
            if (tile.compressed) {
                batch->bufferRequests.push_back({slot.compressed.get(), offset + tile.sourceOffset,
                    staged.buffer, staged.offset + tile.sourceOffset, tile.storedBytes});
                batch->decompressionCopyBarriers.push_back({
                    .buffer = slot.compressed.get(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .range = {.offset = offset + tile.sourceOffset, .size = tile.storedBytes},
                });
                batch->decompressionInputBarriers.push_back({
                    .buffer = slot.compressed.get(),
                    .before = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .after = {PipelineStageBits::MemoryDecompression, AccessBits::DecompressionRead},
                    .range = {offset + tile.sourceOffset, tile.storedBytes},
                });
                batch->decompressionInputBarriers.push_back({
                    .buffer = &destination,
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::MemoryDecompression, AccessBits::DecompressionWrite},
                    .range = {destinationOffset + tile.destinationOffset, tile.decodedBytes},
                });
                batch->decompressionOutputBarriers.push_back({
                    .buffer = &destination,
                    .before = {PipelineStageBits::MemoryDecompression, AccessBits::DecompressionWrite},
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .range = {destinationOffset + tile.destinationOffset, tile.decodedBytes},
                });
            } else {
                batch->bufferRequests.push_back({&destination, destinationOffset + tile.destinationOffset,
                    staged.buffer, staged.offset + tile.sourceOffset, tile.decodedBytes});
                batch->decompressionCopyBarriers.push_back({
                    .buffer = &destination,
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::Transfer, AccessBits::TransferWrite},
                    .range = {.offset = destinationOffset + tile.destinationOffset, .size = tile.decodedBytes},
                });
            }
        }
        batch->decompressions.insert(batch->decompressions.end(), regions.begin(), regions.end());
        return true;
    }

    CopyBatch* findBatch(StreamerCopyBatch batch)
    {
        if (!batch.valid()) { return &defaultBatch; }
        auto found = batches.find(batch.id);
        return found == batches.end() ? nullptr : &found->second;
    }

    CopyBatch acquireBatch()
    {
        if (recycledBatches.empty()) { return {}; }
        auto batch = std::move(recycledBatches.back());
        recycledBatches.pop_back();
        return batch;
    }

    Result<StreamerCopyBatch> beginCopyBatch()
    {
        std::lock_guard lock(mutex);
        if (completionTracked && (!activeFrame || !activeFrame->recording())) { return makeError(Error::InvalidArgument); }
        // Saturate instead of wrapping: stale/cross-Streamer IDs are never reused.
        uint64_t id = nextCopyBatchId.load(std::memory_order_relaxed);
        do {
            if (id == UINT64_MAX) { return makeError(Error::OutOfMemory); }
        } while (!nextCopyBatchId.compare_exchange_weak(id, id + 1, std::memory_order_relaxed));
        batches.emplace(id, acquireBatch());
        return StreamerCopyBatch{id};
    }

    Result<> cancelCopyBatch(StreamerCopyBatch batch)
    {
        std::lock_guard lock(mutex);
        auto found = batches.find(batch.id);
        if (found == batches.end()) { return makeError(Error::InvalidArgument); }
        found->second.cancel();
        if (recycledBatches.size() < 32) { recycledBatches.push_back(std::move(found->second)); }
        batches.erase(found);
        return {};
    }

    StreamerPendingCopyStats pendingCopyStats(StreamerCopyBatch batch) const
    {
        std::lock_guard lock(mutex);
        StreamerPendingCopyStats result;
        if (!batch.valid()) { defaultBatch.accumulate(result); }
        else if (auto found = batches.find(batch.id); found != batches.end()) { found->second.accumulate(result); }
        return result;
    }

    std::shared_ptr<StreamUploadCompletion> pendingCopyCompletion(StreamerCopyBatch copyBatch)
    {
        std::lock_guard lock(mutex);
        auto* batch = findBatch(copyBatch);
        if (!batch || !activeFrame || (batch->bufferRequests.empty() && batch->textureRequests.empty())) { return {}; }
        if (!batch->pendingCompletion) { batch->pendingCompletion.reset(new StreamUploadCompletion(activeFrame->completion())); }
        return batch->pendingCompletion;
    }

    Result<> copyStreamedData(CommandBuffer& commandBuffer, StreamerCopyBatch copyBatch, const StreamUploadPhaseCallback& phase)
    {
        CopyBatch batch;
        {
            std::lock_guard lock(mutex);
            auto* pending = findBatch(copyBatch);
            if (!pending) { return makeError(Error::InvalidArgument); }
            if (activeFrame && !pending->pendingCompletion &&
                (!pending->bufferRequests.empty() || !pending->textureRequests.empty())) {
                pending->pendingCompletion.reset(new StreamUploadCompletion(activeFrame->completion()));
            }
            batch = std::move(*pending);
            if (copyBatch.valid()) { batches.erase(copyBatch.id); }
            else { defaultBatch = acquireBatch(); }
        }
        const auto installation = batch.pendingCompletion;
        struct CancelOnFailure {
            std::shared_ptr<SubmissionTransaction> transaction;
            ~CancelOnFailure() { if (transaction) { transaction->cancel(); } }
        } cancellation{installation ? installation->submission_ : nullptr};
        const auto result = [&]() -> Result<> {
            // Validate before recording writes; failed recordings cancel their
            // publication transaction so partial uploads cannot be submitted.
            if (!batch.decompressions.empty()) {
                auto validated = commandBuffer.validateDecompressionBuffers(batch.decompressions);
                if (!validated) { return validated; }
            }
            if (installation) {
                if (!metallic::render::RenderFrameContext::from(commandBuffer) ||
                    !installation->completion_.sameSubmission(metallic::render::RenderFrameContext::from(commandBuffer)->completion())) {
                    return makeError(Error::InvalidArgument);
                }
                auto attached = commandBuffer.addSubmissionTransaction(installation->submission_);
                if (!attached) { return attached; }
            }
            const profiling::NsightProfileRange copyMarker(
                profiling::NsightDomain::Render,
                "Upload Copies",
                profiling::NsightCategory::ResourceUpload,
                batch.bufferRequests.size() + batch.textureRequests.size());
            if (phase) { phase("Upload copies"); }
            if (!batch.decompressionCopyBarriers.empty()) {
                if (auto commandResult = commandBuffer.synchronize({.buffers = batch.decompressionCopyBarriers}); !commandResult) { return commandResult; }
            }
            for (const BufferCopyRequest& request : batch.bufferRequests) {
                {
                    auto sourceSlice = request.source->slice({request.sourceOffset, request.size});
                    if (!sourceSlice) { return std::unexpected(sourceSlice.error()); }
                    auto destinationSlice = request.destination->slice({request.destinationOffset, request.size});
                    if (!destinationSlice) { return std::unexpected(destinationSlice.error()); }
                    if (auto commandResult = commandBuffer.copyBuffer(*sourceSlice, *destinationSlice); !commandResult) { return commandResult; }
                }
            }

            for (const TextureCopyRequest& request : batch.textureRequests) {
                if (auto commandResult = commandBuffer.copyBufferToTexture(request.copy); !commandResult) { return commandResult; }
            }
            if (!batch.decompressions.empty()) {
                if (phase) { phase("Decompression input barrier"); }
                if (auto result = commandBuffer.synchronize({.buffers = batch.decompressionInputBarriers}); !result) { return result; }
                if (phase) { phase("GPU decompression"); }
                if (auto result = commandBuffer.decompressBuffers(batch.decompressions); !result) { return result; }
                if (phase) { phase("Decompression publish barrier"); }
                if (auto result = commandBuffer.synchronize({.buffers = batch.decompressionOutputBarriers}); !result) { return result; }
            }
            return {};
        }();
        if (result) { cancellation.transaction.reset(); }
        batch.clear();
        {
            std::lock_guard lock(mutex);
            if (recycledBatches.size() < 32) { recycledBatches.push_back(std::move(batch)); }
        }
        return result;
    }

    void endFrame()
    {
        std::lock_guard lock(mutex);
        defaultBatch.cancel();
        for (auto& [id, batch] : batches) { batch.cancel(); }
        batches.clear();
        if (activeFrame != nullptr) {
            UploadSlot& slot = uploadSlots[frameIndex];
            slot.dynamicOffset = dynamicBufferOffset;
            slot.constantOffset = constantBufferOffset;
        }

        for (size_t index = 0; index < garbage.size();) {
            BufferGarbage& entry = garbage[index];
            ++entry.frameCount;
            if (entry.frameCount > desc.queuedFrameCount) {
                entry = std::move(garbage.back());
                garbage.pop_back();
                continue;
            }
            ++index;
        }

        if (!completionTracked && desc.queuedFrameCount != 0) {
            frameIndex = (frameIndex + 1) % desc.queuedFrameCount;
        }
        ++frameSerial;
        lastFrameDynamicBytes = currentFrameDynamicBytes;
        lastFrameDynamicRequestCount = currentFrameDynamicRequestCount;
        peakFrameDynamicBytes = std::max(peakFrameDynamicBytes, currentFrameDynamicBytes);
        totalDynamicBytes += currentFrameDynamicBytes;
        currentFrameDynamicBytes = 0;
        currentFrameDynamicRequestCount = 0;

        lastFrameConstantBytes = currentFrameConstantBytes;
        lastFrameConstantRequestCount = currentFrameConstantRequestCount;
        peakFrameConstantBytes = std::max(peakFrameConstantBytes, currentFrameConstantBytes);
        totalConstantBytes += currentFrameConstantBytes;
        currentFrameConstantBytes = 0;
        currentFrameConstantRequestCount = 0;
        if (!completionTracked) {
            dynamicBufferOffset = 0;
        }
        activeFrame = nullptr;
    }

    StreamerStats stats() const
    {
        std::lock_guard lock(mutex);

        StreamerPendingCopyStats pendingCopies;
        defaultBatch.accumulate(pendingCopies);
        for (const auto& [id, batch] : batches) { batch.accumulate(pendingCopies); }

        return StreamerStats{
            .frameIndex = frameSerial,
            .frameSlot = frameIndex,
            .queuedFrameCount = desc.queuedFrameCount,
            .dynamicBufferSizePerFrame = dynamicBufferSizePerFrame,
            .dynamicBufferOffset = dynamicBufferOffset,
            .constantBufferOffset = constantBufferOffset,
            .currentFrameDynamicBytes = currentFrameDynamicBytes,
            .lastFrameDynamicBytes = lastFrameDynamicBytes,
            .peakFrameDynamicBytes = peakFrameDynamicBytes,
            .totalDynamicBytes = totalDynamicBytes,
            .currentFrameConstantBytes = currentFrameConstantBytes,
            .lastFrameConstantBytes = lastFrameConstantBytes,
            .peakFrameConstantBytes = peakFrameConstantBytes,
            .totalConstantBytes = totalConstantBytes,
            .currentFrameDynamicRequestCount = currentFrameDynamicRequestCount,
            .lastFrameDynamicRequestCount = lastFrameDynamicRequestCount,
            .currentFrameConstantRequestCount = currentFrameConstantRequestCount,
            .lastFrameConstantRequestCount = lastFrameConstantRequestCount,
            .garbageBufferCount = static_cast<uint32_t>(garbage.size()),
            .pendingCopies = pendingCopies,
        };
    }

    Device* device = nullptr;
    StreamerDesc desc;
    std::shared_ptr<Buffer> dynamicBuffer;
    std::shared_ptr<Buffer> constantBuffer;
    std::vector<UploadSlot> uploadSlots;
    RenderFrameContext* activeFrame = nullptr;
    bool completionTracked = false;
    CopyBatch defaultBatch;
    std::unordered_map<uint64_t, CopyBatch> batches;
    std::vector<CopyBatch> recycledBatches;
    std::vector<BufferGarbage> garbage;
    uint64_t dynamicBufferOffset = 0;
    uint64_t dynamicHostAlignment = 1;
    uint64_t constantHostAlignment = 1;
    uint64_t dynamicBufferSizePerFrame = 0;
    uint64_t constantBufferOffset = 0;
    uint64_t constantBufferStride = 0;
    uint64_t currentFrameDynamicBytes = 0;
    uint64_t lastFrameDynamicBytes = 0;
    uint64_t peakFrameDynamicBytes = 0;
    uint64_t totalDynamicBytes = 0;
    uint64_t currentFrameConstantBytes = 0;
    uint64_t lastFrameConstantBytes = 0;
    uint64_t peakFrameConstantBytes = 0;
    uint64_t totalConstantBytes = 0;
    uint64_t frameSerial = 0;
    uint32_t currentFrameDynamicRequestCount = 0;
    uint32_t lastFrameDynamicRequestCount = 0;
    uint32_t currentFrameConstantRequestCount = 0;
    uint32_t lastFrameConstantRequestCount = 0;
    uint32_t frameIndex = 0;
    mutable std::mutex mutex;
};

} // namespace detail

METALLIC_RHI_HANDLE_DEFINITIONS(Streamer)

const StreamerDesc& Streamer::desc() const
{
    static const StreamerDesc emptyDesc;
    return impl_ != nullptr ? impl_->desc : emptyDesc;
}

StreamerStats Streamer::stats() const
{
    return impl_ != nullptr ? impl_->stats() : StreamerStats{};
}

Buffer* Streamer::constantBuffer() const
{
    return impl_ != nullptr ? impl_->constantBuffer.get() : nullptr;
}

BufferOffset Streamer::streamBufferData(const StreamBufferDataDesc& desc)
{
    return impl_ != nullptr ? impl_->streamBufferData(desc) : BufferOffset{};
}

bool Streamer::streamDecompressedBufferData(std::span<const uint8_t> stored,
    std::span<const StreamDecompressionTile> tiles, Buffer& destination, uint64_t destinationOffset, StreamerCopyBatch batch)
{
    return impl_ && impl_->streamDecompressedBufferData(stored, tiles, destination, destinationOffset, batch);
}

BufferOffset Streamer::streamTextureData(const StreamTextureDataDesc& desc)
{
    return impl_ != nullptr ? impl_->streamTextureData(desc) : BufferOffset{};
}

uint64_t Streamer::streamConstantData(const void* data, uint64_t byteSize)
{
    return impl_ != nullptr
        ? impl_->streamConstantData(data, byteSize)
        : kInvalidStreamOffset;
}

std::shared_ptr<StreamUploadCompletion> Streamer::pendingCopyCompletion(StreamerCopyBatch batch)
{
    return impl_ != nullptr ? impl_->pendingCopyCompletion(batch) : nullptr;
}

Result<> Streamer::copyStreamedData(CommandBuffer& commandBuffer, const StreamUploadPhaseCallback& phase)
{
    return impl_ ? impl_->copyStreamedData(commandBuffer, {}, phase) : makeError(Error::InvalidArgument);
}

Result<StreamerCopyBatch> Streamer::beginCopyBatch()
{
    return impl_ ? impl_->beginCopyBatch() : makeError(Error::InvalidArgument);
}

Result<> Streamer::cancelCopyBatch(StreamerCopyBatch batch)
{
    return impl_ ? impl_->cancelCopyBatch(batch) : makeError(Error::InvalidArgument);
}

StreamerPendingCopyStats Streamer::pendingCopyStats(StreamerCopyBatch batch) const
{
    return impl_ ? impl_->pendingCopyStats(batch) : StreamerPendingCopyStats{};
}

Result<> Streamer::copyStreamedData(CommandBuffer& commandBuffer, StreamerCopyBatch batch,
    const StreamUploadPhaseCallback& phase)
{
    return impl_ ? impl_->copyStreamedData(commandBuffer, batch, phase) : makeError(Error::InvalidArgument);
}

void Streamer::endFrame()
{
    if (impl_ != nullptr) {
        impl_->endFrame();
    }
}

Result<> Streamer::beginFrame(RenderFrameContext& frame)
{
    return impl_ != nullptr ? impl_->beginFrame(frame) : makeError(Error::InvalidArgument);
}

Result<std::unique_ptr<Streamer>> createStreamer(Device& device, const StreamerDesc& desc)
{
    if (!device.identity()) {
        return makeError(Error::InvalidArgument);
    }

    auto streamerImpl = std::make_unique<detail::StreamerImpl>(device);
    Result<> result = streamerImpl->create(desc);
    if (!result) {
        return std::unexpected(result.error());
    }

    return std::unique_ptr<Streamer>(new Streamer(std::move(streamerImpl)));
}

} // namespace metallic::render

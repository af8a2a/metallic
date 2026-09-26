#include "Runtime/Render/GAPI/TextureFormat.h"
#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/GAPI/StreamUploadCompletion.h"
#include "Runtime/Render/RenderFrameContext.h"
#include "Runtime/Render/Profiling/NsightEvents.h"
#include "Runtime/Render/Profiling/CpuPhaseTrace.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

namespace metallic::render {
namespace {

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

uint32_t formatTexelByteSize(Format format)
{
    switch (format) {
    case Format::R8Unorm:
    case Format::R8Snorm:
    case Format::R8Uint:
    case Format::R8Sint:
        return 1;
    case Format::Rg8Unorm:
    case Format::Rg8Snorm:
    case Format::Rg8Uint:
    case Format::Rg8Sint:
    case Format::R16Unorm:
    case Format::R16Snorm:
    case Format::R16Uint:
    case Format::R16Sint:
    case Format::R16Sfloat:
        return 2;
    case Format::Bgra8Unorm:
    case Format::Bgra8Srgb:
    case Format::Rgba8Unorm:
    case Format::Rgba8Snorm:
    case Format::Rgba8Srgb:
    case Format::Rgba8Uint:
    case Format::Rgba8Sint:
    case Format::Rg16Unorm:
    case Format::Rg16Snorm:
    case Format::Rg16Uint:
    case Format::Rg16Sint:
    case Format::Rg16Sfloat:
    case Format::R32Uint:
    case Format::R32Sint:
    case Format::R32Sfloat:
    case Format::A2B10G10R10UnormPack32:
    case Format::A2R10G10B10UintPack32:
    case Format::B10G11R11UfloatPack32:
    case Format::E5B9G9R9UfloatPack32:
    case Format::D32Sfloat:
        return 4;
    case Format::Rgba16Unorm:
    case Format::Rgba16Snorm:
    case Format::Rgba16Uint:
    case Format::Rgba16Sint:
    case Format::Rgba16Sfloat:
    case Format::Rg32Uint:
    case Format::Rg32Sint:
    case Format::Rg32Sfloat:
        return 8;
    case Format::Rgb32Uint:
    case Format::Rgb32Sint:
    case Format::Rgb32Sfloat:
        return 12;
    case Format::Rgba32Uint:
    case Format::Rgba32Sint:
    case Format::Rgba32Sfloat:
        return 16;
    case Format::Unknown:
        break;
    }
    return 0;
}

uint64_t textureCopyByteSize(const BufferTextureCopyDesc& copy)
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
        BufferTextureCopyDesc copy;
    };

    struct BufferGarbage {
        std::shared_ptr<Buffer> buffer;
        uint32_t frameCount = 0;
    };

    struct UploadSlot {
        GpuCompletionPoint completion;
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
        if (pendingCompletion) { pendingCompletion->submission_->cancel(); }
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
            constantBuffer = std::move(buffer);
        }
        return {};
    }

    Result<> beginFrame(RenderFrameContext& frame)
    {
        std::lock_guard lock(mutex);
        if (!frame.recording() || frame.slotIndex() >= desc.queuedFrameCount ||
            !bufferRequests.empty() || !textureRequests.empty() ||
            (activeFrame != nullptr && activeFrame != &frame)) {
            return makeError(Error::InvalidArgument);
        }
        uploadSlots.resize(desc.queuedFrameCount);
        UploadSlot& slot = uploadSlots[frame.slotIndex()];
        if (!slot.completion.sameSubmission(frame.completion())) {
            if (!slot.completion.isComplete()) {
                return makeError(Error::InvalidArgument);
            }
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
        frame.retain(constantBuffer);
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

        profiling::CpuPhase allocationPhase("stream.grow", bufferDesc.size);
        std::unique_ptr<Buffer> newBuffer;
        Result<> result = device->createBuffer(bufferDesc).transform([&](auto rhiValue) { newBuffer = std::move(rhiValue); });
        if (!result || newBuffer == nullptr) {
            return result ? makeError(Error::Failure) : result;
        }

        if (dynamicBuffer != nullptr) {
            if (activeFrame != nullptr) {
                activeFrame->retain(dynamicBuffer);
            } else {
                garbage.push_back(BufferGarbage{
                    .buffer = dynamicBuffer,
                    .frameCount = 0,
                });
            }
        }
        dynamicBuffer = std::move(newBuffer);
        dynamicBufferSizePerFrame = newSizePerFrame;
        return {};
    }

    BufferOffset streamBufferData(const StreamBufferDataDesc& streamDesc)
    {
        std::lock_guard lock(mutex);
        return stageBufferData(streamDesc);
    }

    BufferOffset stageBufferData(const StreamBufferDataDesc& streamDesc)
    {
        if (completionTracked && (activeFrame == nullptr || !activeFrame->recording())) {
            return {};
        }
        if (streamDesc.dataChunkCount == 0 ||
            streamDesc.dataChunks == nullptr ||
            desc.queuedFrameCount == 0) {
            return {};
        }

        uint64_t dataSize = 0;
        for (uint32_t index = 0; index < streamDesc.dataChunkCount; ++index) {
            const StreamDataChunk& chunk = streamDesc.dataChunks[index];
            if (chunk.size > 0 && chunk.data == nullptr) {
                return {};
            }
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
        const uint64_t localOffset = alignUp(dynamicBufferOffset, alignment);
        const uint64_t requiredSizePerFrame = localOffset + dataSize;
        Result<> result = ensureDynamicBuffer(requiredSizePerFrame);
        if (!result || dynamicBuffer == nullptr) {
            return {};
        }

        const uint64_t bufferOffset =
            static_cast<uint64_t>(frameIndex) * dynamicBufferSizePerFrame + localOffset;
        void* mapped = dynamicBuffer->map();
        if (mapped == nullptr) {
            return {};
        }

        uint8_t* dst = static_cast<uint8_t*>(mapped) + bufferOffset;
        for (uint32_t index = 0; index < streamDesc.dataChunkCount; ++index) {
            const StreamDataChunk& chunk = streamDesc.dataChunks[index];
            if (chunk.size == 0) {
                continue;
            }
            std::memcpy(dst, chunk.data, static_cast<size_t>(chunk.size));
            dst += chunk.size;
        }
        dynamicBuffer->flush(bufferOffset, dataSize);
        dynamicBuffer->unmap();

        if (activeFrame != nullptr) {
            activeFrame->retain(dynamicBuffer);
        }

        if (streamDesc.dstBuffer != nullptr) {
            bufferRequests.push_back(BufferCopyRequest{
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
        if (completionTracked && (activeFrame == nullptr || !activeFrame->recording())) {
            return {};
        }
        if (streamDesc.data == nullptr ||
            streamDesc.dstTexture == nullptr ||
            desc.queuedFrameCount == 0) {
            return {};
        }

        const TextureDesc& textureDesc = streamDesc.dstTexture->desc();
        const uint32_t blockBytes = compressedBlockBytes(textureDesc.format);
        const uint32_t blockExtent = blockBytes ? 4 : 1;
        const uint32_t bytesPerTexel = blockBytes ? blockBytes : formatTexelByteSize(textureDesc.format);
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
        const uint32_t rows = (height + blockExtent - 1) / blockExtent;
        const uint64_t rowSize = ((uint64_t(width) + blockExtent - 1) / blockExtent) * bytesPerTexel;
        if (rowSize > std::numeric_limits<uint32_t>::max()) {
            return {};
        }
        const uint32_t sourceRowPitch = streamDesc.dataRowPitch == 0
            ? static_cast<uint32_t>(rowSize)
            : streamDesc.dataRowPitch;
        const uint32_t sourceSlicePitch = streamDesc.dataSlicePitch == 0
            ? sourceRowPitch * rows
            : streamDesc.dataSlicePitch;
        if (sourceRowPitch < rowSize || sourceSlicePitch < static_cast<uint64_t>(sourceRowPitch) * rows) {
            return {};
        }

        const DeviceCapabilities& capabilities = device->capabilities();
        const uint64_t rowPitch = alignUp(rowSize, std::max<uint64_t>(capabilities.textureUploadRowPitchAlignment, bytesPerTexel));
        const uint64_t slicePitch = alignUp(
            rowPitch * rows,
            capabilities.textureUploadSlicePitchAlignment);
        const uint64_t copySliceCount =
            static_cast<uint64_t>(depth) * static_cast<uint64_t>(streamDesc.dstLayerCount);
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

        const uint64_t localOffset = alignUp(
            dynamicBufferOffset,
            std::max<uint64_t>(capabilities.textureUploadBufferOffsetAlignment, bytesPerTexel));
        const uint64_t requiredSizePerFrame = localOffset + dataSize;
        Result<> result = ensureDynamicBuffer(requiredSizePerFrame);
        if (!result || dynamicBuffer == nullptr) {
            return {};
        }

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
        dynamicBuffer->flush(bufferOffset, dataSize);
        dynamicBuffer->unmap();

        if (activeFrame != nullptr) {
            activeFrame->retain(dynamicBuffer);
        }
        textureRequests.push_back(TextureCopyRequest{
            .copy = BufferTextureCopyDesc{
                .buffer = dynamicBuffer.get(),
                .texture = streamDesc.dstTexture,
                .bufferOffset = bufferOffset,
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

        const uint64_t alignment = device != nullptr
            ? device->capabilities().constantBufferOffsetAlignment
            : 1;
        uint64_t offset = alignUp(constantBufferOffset, alignment);
        if (offset + byteSize > desc.constantBufferSize) {
            if (completionTracked) {
                return kInvalidStreamOffset;
            }
            offset = 0;
        }
        if (offset + byteSize > desc.constantBufferSize) {
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
            constantBuffer->flush(bufferOffset, byteSize);
            constantBuffer->unmap();
        }
        constantBufferOffset = offset + byteSize;
        currentFrameConstantBytes += byteSize;
        ++currentFrameConstantRequestCount;
        return bufferOffset;
    }

    bool streamDecompressedBufferData(std::span<const uint8_t> stored,
        std::span<const StreamDecompressionTile> tiles, Buffer& destination, uint64_t destinationOffset)
    {
        std::lock_guard lock(mutex);
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
        }
        const StreamDataChunk chunk{stored.data(), stored.size()};
        const auto staged = stageBufferData({.dataChunks = &chunk, .dataChunkCount = 1, .placementAlignment = 16});
        if (!staged.valid()) { return false; }
        activeFrame->retain(slot.compressed);
        slot.compressedOffset = required;
        for (const auto& tile : tiles) {
            if (tile.compressed) {
                bufferRequests.push_back({slot.compressed.get(), offset + tile.sourceOffset,
                    staged.buffer, staged.offset + tile.sourceOffset, tile.storedBytes});
                decompressionCopyBarriers.push_back({.buffer = slot.compressed.get(), .before = ResourceState::General,
                    .after = ResourceState::TransferDestination, .offset = offset + tile.sourceOffset, .size = tile.storedBytes});
                decompressions.push_back({slot.compressed.get(), &destination, offset + tile.sourceOffset,
                    destinationOffset + tile.destinationOffset, tile.storedBytes, tile.decodedBytes});
            } else {
                bufferRequests.push_back({&destination, destinationOffset + tile.destinationOffset,
                    staged.buffer, staged.offset + tile.sourceOffset, tile.decodedBytes});
                decompressionCopyBarriers.push_back({.buffer = &destination, .before = ResourceState::General,
                    .after = ResourceState::TransferDestination, .offset = destinationOffset + tile.destinationOffset, .size = tile.decodedBytes});
            }
        }
        return true;
    }

    std::shared_ptr<StreamUploadCompletion> pendingCopyCompletion()
    {
        std::lock_guard lock(mutex);
        if (activeFrame == nullptr || (bufferRequests.empty() && textureRequests.empty())) {
            return {};
        }
        if (!pendingCompletion) {
            pendingCompletion.reset(new StreamUploadCompletion(activeFrame->completion()));
        }
        return pendingCompletion;
    }

    void copyStreamedData(CommandBuffer& commandBuffer, const StreamUploadPhaseCallback& phase)
    {
        std::lock_guard lock(mutex);
        const auto installation = pendingCompletion;
        // Preflight before recording ANY writes. Cancelling after partial GPU
        // writes would allow residency to recycle their allocation too early.
        if (!decompressions.empty() && !commandBuffer.validateDecompressionBuffers(decompressions)) {
            if (installation) { installation->submission_->cancel(); }
            pendingCompletion.reset();
            bufferRequests.clear();
            decompressions.clear();
            decompressionCopyBarriers.clear();
            textureRequests.clear();
            return;
        }
        if (pendingCompletion) {
            // Attach to the recording containing the copies, not an earlier pass
            // segment. A failed attachment must not leave untracked GPU writes.
            if (commandBuffer.frameContext() == nullptr ||
                !pendingCompletion->completion_.sameSubmission(commandBuffer.frameContext()->completion()) ||
                !commandBuffer.addSubmissionTransaction(pendingCompletion->submission_)) {
                pendingCompletion->submission_->cancel();
                pendingCompletion.reset();
                bufferRequests.clear();
                decompressions.clear();
                decompressionCopyBarriers.clear();
                textureRequests.clear();
                return;
            }
            pendingCompletion.reset();
        }
        const profiling::NsightProfileRange copyMarker(
            profiling::NsightDomain::Render,
            "Upload Copies",
            profiling::NsightCategory::ResourceUpload,
            bufferRequests.size() + textureRequests.size());
        if (phase) { phase("Upload copies"); }
        if (!decompressionCopyBarriers.empty()) {
            commandBuffer.barrier({.buffers = decompressionCopyBarriers.data(), .bufferCount = uint32_t(decompressionCopyBarriers.size())});
        }
        for (const BufferCopyRequest& request : bufferRequests) {
            commandBuffer.copyBuffer(BufferCopyDesc{
                .source = request.source,
                .destination = request.destination,
                .sourceOffset = request.sourceOffset,
                .destinationOffset = request.destinationOffset,
                .size = request.size,
            });
        }

        for (const TextureCopyRequest& request : textureRequests) {
            commandBuffer.copyBufferToTexture(request.copy);
        }
        if (!decompressions.empty()) {
            if (phase) { phase("Decompression input barrier"); }
            std::vector<BufferBarrierDesc> barriers;
            for (const auto& region : decompressions) {
                barriers.push_back({.buffer = region.source, .before = ResourceState::TransferDestination,
                    .after = ResourceState::DecompressionSource, .offset = region.sourceOffset, .size = region.compressedBytes});
                barriers.push_back({.buffer = region.destination, .before = ResourceState::General,
                    .after = ResourceState::DecompressionDestination, .offset = region.destinationOffset, .size = region.decodedBytes});
            }
            commandBuffer.barrier({.buffers = barriers.data(), .bufferCount = uint32_t(barriers.size())});
            if (phase) { phase("GPU decompression"); }
            if (!commandBuffer.decompressBuffers(decompressions)) {
                if (installation) { installation->submission_->cancel(); }
            } else {
                if (phase) { phase("Decompression publish barrier"); }
                barriers.clear();
                for (const auto& region : decompressions) {
                    // General covers both shader consumers and AS build input
                    // reads. The runtime retains its ordinary copy transitions.
                    barriers.push_back({.buffer = region.destination, .before = ResourceState::DecompressionDestination,
                        .after = ResourceState::General, .offset = region.destinationOffset, .size = region.decodedBytes});
                }
                commandBuffer.barrier({.buffers = barriers.data(), .bufferCount = uint32_t(barriers.size())});
            }
        }
        decompressions.clear();
        decompressionCopyBarriers.clear();

        bufferRequests.clear();
        textureRequests.clear();
    }

    void endFrame()
    {
        std::lock_guard lock(mutex);
        if (pendingCompletion) {
            pendingCompletion->submission_->cancel();
            pendingCompletion.reset();
        }
        if (activeFrame != nullptr) {
            UploadSlot& slot = uploadSlots[frameIndex];
            slot.dynamicOffset = dynamicBufferOffset;
            slot.constantOffset = constantBufferOffset;
        }
        bufferRequests.clear();
        decompressions.clear();
        textureRequests.clear();

        decompressionCopyBarriers.clear();

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
        pendingCopies.bufferCopyCount = static_cast<uint32_t>(bufferRequests.size());
        pendingCopies.textureCopyCount = static_cast<uint32_t>(textureRequests.size());
        for (const BufferCopyRequest& request : bufferRequests) {
            pendingCopies.bufferCopyBytes += request.size;
        }
        for (const TextureCopyRequest& request : textureRequests) {
            pendingCopies.textureCopyBytes += textureCopyByteSize(request.copy);
        }

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
    std::shared_ptr<StreamUploadCompletion> pendingCompletion;
    std::vector<BufferCopyRequest> bufferRequests;
    std::vector<BufferDecompressionDesc> decompressions;
    std::vector<BufferBarrierDesc> decompressionCopyBarriers;
    std::vector<TextureCopyRequest> textureRequests;
    std::vector<BufferGarbage> garbage;
    uint64_t dynamicBufferOffset = 0;
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

Streamer::Streamer(std::unique_ptr<detail::StreamerImpl> impl)
    : impl_(std::move(impl))
{
}

Streamer::~Streamer() = default;
Streamer::Streamer(Streamer&&) noexcept = default;
Streamer& Streamer::operator=(Streamer&&) noexcept = default;

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
    std::span<const StreamDecompressionTile> tiles, Buffer& destination, uint64_t destinationOffset)
{
    return impl_ && impl_->streamDecompressedBufferData(stored, tiles, destination, destinationOffset);
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

std::shared_ptr<StreamUploadCompletion> Streamer::pendingCopyCompletion()
{
    return impl_ != nullptr ? impl_->pendingCopyCompletion() : nullptr;
}

void Streamer::copyStreamedData(CommandBuffer& commandBuffer, const StreamUploadPhaseCallback& phase)
{
    if (impl_ != nullptr) {
        impl_->copyStreamedData(commandBuffer, phase);
    }
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

void CommandBuffer::copyStreamedData(Streamer& streamer)
{
    streamer.copyStreamedData(*this);
}

Result<std::unique_ptr<Streamer>> Device::createStreamer(const StreamerDesc& desc)
{
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    auto streamerImpl = std::make_unique<detail::StreamerImpl>(*this);
    Result<> result = streamerImpl->create(desc);
    if (!result) {
        return std::unexpected(result.error());
    }

    return std::unique_ptr<Streamer>(new Streamer(std::move(streamerImpl)));
}

} // namespace metallic::render

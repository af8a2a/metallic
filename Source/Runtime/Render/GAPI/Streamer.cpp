#include "Runtime/Render/GAPI/Rhi.h"

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
        std::unique_ptr<Buffer> buffer;
        uint32_t frameCount = 0;
    };

    explicit StreamerImpl(Device& streamerDevice)
        : device(&streamerDevice)
    {
    }

    Result create(const StreamerDesc& streamerDesc)
    {
        if (device == nullptr || streamerDesc.queuedFrameCount == 0) {
            return makeError(Error::InvalidArgument);
        }

        desc = streamerDesc;
        if (desc.dynamicBufferSizePerFrame == 0) {
            desc.dynamicBufferSizePerFrame = kDynamicBufferChunkSize;
        }
        dynamicBufferSizePerFrame = alignUp(desc.dynamicBufferSizePerFrame, kDynamicBufferChunkSize);

        if (desc.constantBufferSize > 0) {
            BufferDesc bufferDesc{
                .size = desc.constantBufferSize,
                .usage = BufferUsageBits::Constant,
                .memoryLocation = desc.constantBufferMemoryLocation,
            };
            Result result = device->createBuffer(bufferDesc, constantBuffer);
            if (!result || constantBuffer == nullptr) {
                return result ? makeError(Error::Failure) : result;
            }
        }
        return {};
    }

    Result ensureDynamicBuffer(uint64_t requiredSizePerFrame)
    {
        if (device == nullptr) {
            return makeError(Error::InvalidArgument);
        }
        if (dynamicBuffer != nullptr && requiredSizePerFrame <= dynamicBufferSizePerFrame) {
            return {};
        }

        const uint64_t newSizePerFrame = alignUp(
            std::max(requiredSizePerFrame, dynamicBufferSizePerFrame),
            kDynamicBufferChunkSize);
        BufferDesc bufferDesc = desc.dynamicBufferDesc;
        bufferDesc.size = newSizePerFrame * desc.queuedFrameCount;
        bufferDesc.usage = bufferDesc.usage | BufferUsageBits::TransferSource;
        bufferDesc.memoryLocation = desc.dynamicBufferMemoryLocation;

        std::unique_ptr<Buffer> newBuffer;
        Result result = device->createBuffer(bufferDesc, newBuffer);
        if (!result || newBuffer == nullptr) {
            return result ? makeError(Error::Failure) : result;
        }

        if (dynamicBuffer != nullptr) {
            garbage.push_back(BufferGarbage{
                .buffer = std::move(dynamicBuffer),
                .frameCount = 0,
            });
        }
        dynamicBuffer = std::move(newBuffer);
        dynamicBufferSizePerFrame = newSizePerFrame;
        return {};
    }

    BufferOffset streamBufferData(const StreamBufferDataDesc& streamDesc)
    {
        std::lock_guard lock(mutex);
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

        const uint64_t alignment = std::max<uint64_t>(
            std::max(streamDesc.placementAlignment, 1u),
            device != nullptr ? device->capabilities().bufferCopyOffsetAlignment : 1);
        const uint64_t localOffset = alignUp(dynamicBufferOffset, alignment);
        const uint64_t requiredSizePerFrame = localOffset + dataSize;
        Result result = ensureDynamicBuffer(requiredSizePerFrame);
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
        return BufferOffset{
            .buffer = dynamicBuffer.get(),
            .offset = bufferOffset,
        };
    }

    BufferOffset streamTextureData(const StreamTextureDataDesc& streamDesc)
    {
        std::lock_guard lock(mutex);
        if (streamDesc.data == nullptr ||
            streamDesc.dstTexture == nullptr ||
            desc.queuedFrameCount == 0) {
            return {};
        }

        const TextureDesc& textureDesc = streamDesc.dstTexture->desc();
        const uint32_t bytesPerTexel = formatTexelByteSize(textureDesc.format);
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

        const uint64_t rowSize = static_cast<uint64_t>(width) * bytesPerTexel;
        if (rowSize > std::numeric_limits<uint32_t>::max()) {
            return {};
        }
        const uint32_t sourceRowPitch = streamDesc.dataRowPitch == 0
            ? static_cast<uint32_t>(rowSize)
            : streamDesc.dataRowPitch;
        const uint32_t sourceSlicePitch = streamDesc.dataSlicePitch == 0
            ? sourceRowPitch * height
            : streamDesc.dataSlicePitch;
        if (sourceRowPitch < rowSize || sourceSlicePitch < static_cast<uint64_t>(sourceRowPitch) * height) {
            return {};
        }

        const DeviceCapabilities& capabilities = device->capabilities();
        const uint64_t rowPitch = alignUp(rowSize, capabilities.textureUploadRowPitchAlignment);
        const uint64_t slicePitch = alignUp(
            rowPitch * height,
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

        const uint64_t localOffset = alignUp(
            dynamicBufferOffset,
            capabilities.textureUploadBufferOffsetAlignment);
        const uint64_t requiredSizePerFrame = localOffset + dataSize;
        Result result = ensureDynamicBuffer(requiredSizePerFrame);
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
            for (uint32_t rowIndex = 0; rowIndex < height; ++rowIndex) {
                uint8_t* dst = dstBase + sliceIndex * slicePitch + rowIndex * rowPitch;
                const uint8_t* src =
                    srcBase + sliceIndex * sourceSlicePitch + rowIndex * sourceRowPitch;
                std::memcpy(dst, src, static_cast<size_t>(rowSize));
            }
        }
        dynamicBuffer->flush(bufferOffset, dataSize);
        dynamicBuffer->unmap();

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
        return BufferOffset{
            .buffer = dynamicBuffer.get(),
            .offset = bufferOffset,
        };
    }

    uint64_t streamConstantData(const void* data, uint64_t byteSize)
    {
        std::lock_guard lock(mutex);
        if (constantBuffer == nullptr ||
            (byteSize > 0 && data == nullptr) ||
            byteSize > desc.constantBufferSize ||
            byteSize > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            return kInvalidStreamOffset;
        }

        const uint64_t alignment = device != nullptr
            ? device->capabilities().constantBufferOffsetAlignment
            : 1;
        uint64_t offset = alignUp(constantBufferOffset, alignment);
        if (offset + byteSize > desc.constantBufferSize) {
            offset = 0;
        }
        if (offset + byteSize > desc.constantBufferSize) {
            return kInvalidStreamOffset;
        }

        if (byteSize > 0) {
            void* mapped = constantBuffer->map();
            if (mapped == nullptr) {
                return kInvalidStreamOffset;
            }
            std::memcpy(
                static_cast<uint8_t*>(mapped) + offset,
                data,
                static_cast<size_t>(byteSize));
            constantBuffer->flush(offset, byteSize);
            constantBuffer->unmap();
        }
        constantBufferOffset = offset + byteSize;
        return offset;
    }

    void copyStreamedData(CommandBuffer& commandBuffer)
    {
        std::lock_guard lock(mutex);
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

        bufferRequests.clear();
        textureRequests.clear();
    }

    void endFrame()
    {
        std::lock_guard lock(mutex);
        bufferRequests.clear();
        textureRequests.clear();

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

        if (desc.queuedFrameCount != 0) {
            frameIndex = (frameIndex + 1) % desc.queuedFrameCount;
        }
        dynamicBufferOffset = 0;
    }

    Device* device = nullptr;
    StreamerDesc desc;
    std::unique_ptr<Buffer> dynamicBuffer;
    std::unique_ptr<Buffer> constantBuffer;
    std::vector<BufferCopyRequest> bufferRequests;
    std::vector<TextureCopyRequest> textureRequests;
    std::vector<BufferGarbage> garbage;
    uint64_t dynamicBufferOffset = 0;
    uint64_t dynamicBufferSizePerFrame = 0;
    uint64_t constantBufferOffset = 0;
    uint32_t frameIndex = 0;
    std::mutex mutex;
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

Buffer* Streamer::constantBuffer() const
{
    return impl_ != nullptr ? impl_->constantBuffer.get() : nullptr;
}

BufferOffset Streamer::streamBufferData(const StreamBufferDataDesc& desc)
{
    return impl_ != nullptr ? impl_->streamBufferData(desc) : BufferOffset{};
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

void Streamer::copyStreamedData(CommandBuffer& commandBuffer)
{
    if (impl_ != nullptr) {
        impl_->copyStreamedData(commandBuffer);
    }
}

void Streamer::endFrame()
{
    if (impl_ != nullptr) {
        impl_->endFrame();
    }
}

void CommandBuffer::copyStreamedData(Streamer& streamer)
{
    streamer.copyStreamedData(*this);
}

Result Device::createStreamer(const StreamerDesc& desc, std::unique_ptr<Streamer>& outStreamer)
{
    outStreamer.reset();
    if (impl_ == nullptr) {
        return makeError(Error::InvalidArgument);
    }

    auto streamerImpl = std::make_unique<detail::StreamerImpl>(*this);
    Result result = streamerImpl->create(desc);
    if (!result) {
        return result;
    }

    outStreamer.reset(new Streamer(std::move(streamerImpl)));
    return {};
}

} // namespace metallic::render

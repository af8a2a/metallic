#include "Runtime/Render/ImportanceSampling.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace metallic::render {
namespace {

uint32_t paddedDimension(uint32_t value)
{
    if (value == 0 || value > (1u << 30u)) {
        return 0;
    }
    return std::bit_ceil(value);
}

bool validTexelCount(uint32_t width, uint32_t height, size_t& outCount)
{
    const uint64_t count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    if (count == 0 || count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        outCount = 0;
        return false;
    }
    outCount = static_cast<size_t>(count);
    return true;
}

std::string resultMessage(std::string_view label, Result result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

} // namespace

bool ImportanceMipChain::valid() const
{
    return sourceWidth != 0 &&
        sourceHeight != 0 &&
        textureWidth != 0 &&
        textureHeight != 0 &&
        totalWeight > 0.0f &&
        !levels.empty() &&
        levels.front().width == textureWidth &&
        levels.front().height == textureHeight &&
        levels.back().width == 1 &&
        levels.back().height == 1;
}

float ImportanceMipChain::probability(uint32_t x, uint32_t y) const
{
    if (!valid() || x >= sourceWidth || y >= sourceHeight) {
        return 0.0f;
    }
    const size_t index = static_cast<size_t>(y) * textureWidth + x;
    return levels.front().values[index] / totalWeight;
}

ImportanceMipChain buildImportanceMipChain(
    std::span<const float> sourceWeights,
    uint32_t sourceWidth,
    uint32_t sourceHeight)
{
    ImportanceMipChain chain;
    chain.sourceWidth = sourceWidth;
    chain.sourceHeight = sourceHeight;
    chain.textureWidth = paddedDimension(sourceWidth);
    chain.textureHeight = paddedDimension(sourceHeight);

    size_t sourceCount = 0;
    size_t paddedCount = 0;
    if (!validTexelCount(sourceWidth, sourceHeight, sourceCount) ||
        !validTexelCount(chain.textureWidth, chain.textureHeight, paddedCount) ||
        sourceWeights.size() < sourceCount) {
        return {};
    }

    ImportanceMipLevel base;
    base.width = chain.textureWidth;
    base.height = chain.textureHeight;
    base.values.resize(paddedCount, 0.0f);
    double totalWeight = 0.0;
    for (uint32_t y = 0; y < sourceHeight; ++y) {
        for (uint32_t x = 0; x < sourceWidth; ++x) {
            const size_t sourceIndex = static_cast<size_t>(y) * sourceWidth + x;
            const float sourceWeight = sourceWeights[sourceIndex];
            const float weight = std::isfinite(sourceWeight) ? std::max(sourceWeight, 0.0f) : 0.0f;
            base.values[static_cast<size_t>(y) * chain.textureWidth + x] = weight;
            totalWeight += static_cast<double>(weight);
        }
    }

    if (!(totalWeight > 0.0) || !std::isfinite(totalWeight)) {
        totalWeight = static_cast<double>(sourceCount);
        for (uint32_t y = 0; y < sourceHeight; ++y) {
            for (uint32_t x = 0; x < sourceWidth; ++x) {
                base.values[static_cast<size_t>(y) * chain.textureWidth + x] = 1.0f;
            }
        }
    }
    chain.totalWeight = static_cast<float>(totalWeight);
    chain.levels.push_back(std::move(base));

    while (chain.levels.back().width > 1 || chain.levels.back().height > 1) {
        const ImportanceMipLevel& source = chain.levels.back();
        ImportanceMipLevel target;
        target.width = std::max(source.width / 2u, 1u);
        target.height = std::max(source.height / 2u, 1u);
        size_t targetCount = 0;
        if (!validTexelCount(target.width, target.height, targetCount)) {
            return {};
        }
        target.values.resize(targetCount, 0.0f);

        for (uint32_t y = 0; y < target.height; ++y) {
            for (uint32_t x = 0; x < target.width; ++x) {
                float sum = 0.0f;
                for (uint32_t offsetY = 0; offsetY < 2; ++offsetY) {
                    const uint32_t sourceY = y * 2u + offsetY;
                    if (sourceY >= source.height) {
                        continue;
                    }
                    for (uint32_t offsetX = 0; offsetX < 2; ++offsetX) {
                        const uint32_t sourceX = x * 2u + offsetX;
                        if (sourceX < source.width) {
                            sum += source.values[static_cast<size_t>(sourceY) * source.width + sourceX];
                        }
                    }
                }
                target.values[static_cast<size_t>(y) * target.width + x] = sum * 0.25f;
            }
        }
        chain.levels.push_back(std::move(target));
    }
    return chain;
}

struct ImportancePdfTexture::Impl {
    struct Upload {
        std::unique_ptr<Buffer> buffer;
        uint32_t width = 0;
        uint32_t height = 0;
        uint64_t byteSize = 0;
    };

    ImportanceMipChain chain;
    std::vector<Upload> uploads;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    ResourceState state = ResourceState::Undefined;
    uint64_t byteSize = 0;
    bool uploaded = false;

    void clear()
    {
        chain = {};
        uploads.clear();
        view.reset();
        texture.reset();
        state = ResourceState::Undefined;
        byteSize = 0;
        uploaded = false;
    }
};

ImportancePdfTexture::ImportancePdfTexture()
    : impl_(std::make_unique<Impl>())
{
}

ImportancePdfTexture::~ImportancePdfTexture() = default;
ImportancePdfTexture::ImportancePdfTexture(ImportancePdfTexture&&) noexcept = default;
ImportancePdfTexture& ImportancePdfTexture::operator=(ImportancePdfTexture&&) noexcept = default;

Result ImportancePdfTexture::initialize(
    Device& device,
    std::span<const float> sourceWeights,
    uint32_t sourceWidth,
    uint32_t sourceHeight,
    std::string_view debugName,
    std::string& log)
{
    impl_->clear();
    impl_->chain = buildImportanceMipChain(sourceWeights, sourceWidth, sourceHeight);
    if (!impl_->chain.valid()) {
        log = std::string(debugName) + " importance mip input is invalid";
        return makeError(Error::InvalidArgument);
    }

    impl_->uploads.reserve(impl_->chain.levels.size());
    for (uint32_t mipIndex = 0; mipIndex < impl_->chain.levels.size(); ++mipIndex) {
        const ImportanceMipLevel& mip = impl_->chain.levels[mipIndex];
        Impl::Upload upload;
        upload.width = mip.width;
        upload.height = mip.height;
        upload.byteSize = static_cast<uint64_t>(mip.values.size()) * sizeof(float);
        impl_->byteSize += upload.byteSize;

        Result result = device.createBuffer(
            BufferDesc{
                .size = upload.byteSize,
                .usage = BufferUsageBits::TransferSource,
                .memoryLocation = MemoryLocation::HostUpload,
            },
            upload.buffer);
        if (!result || upload.buffer == nullptr) {
            log = resultMessage(std::string("createBuffer(") + std::string(debugName) + " mip upload)", result);
            return result ? makeError(Error::Failure) : result;
        }
        void* mapped = upload.buffer->map();
        if (mapped == nullptr) {
            log = std::string(debugName) + " failed to map mip upload buffer";
            return makeError(Error::Failure);
        }
        std::memcpy(mapped, mip.values.data(), static_cast<size_t>(upload.byteSize));
        upload.buffer->flush(0, upload.byteSize);
        upload.buffer->unmap();
        impl_->uploads.push_back(std::move(upload));
    }

    Result result = device.createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = Format::R32Sfloat,
            .width = impl_->chain.textureWidth,
            .height = impl_->chain.textureHeight,
            .depth = 1,
            .mipCount = static_cast<uint32_t>(impl_->chain.levels.size()),
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        },
        impl_->texture);
    if (!result || impl_->texture == nullptr) {
        log = resultMessage(std::string("createTexture(") + std::string(debugName) + ")", result);
        return result ? makeError(Error::Failure) : result;
    }

    result = device.createTextureView(
        *impl_->texture,
        TextureViewDesc{
            .format = Format::R32Sfloat,
            .baseMip = 0,
            .mipCount = static_cast<uint32_t>(impl_->chain.levels.size()),
            .baseLayer = 0,
            .layerCount = 1,
        },
        impl_->view);
    if (!result || impl_->view == nullptr) {
        log = resultMessage(std::string("createTextureView(") + std::string(debugName) + ")", result);
        return result ? makeError(Error::Failure) : result;
    }
    return {};
}

Result ImportancePdfTexture::upload(CommandBuffer& commandBuffer)
{
    if (impl_->uploaded) {
        return {};
    }
    if (!valid() || impl_->uploads.size() != impl_->chain.levels.size()) {
        return makeError(Error::InvalidArgument);
    }

    TextureBarrierDesc toTransfer{
        .texture = impl_->texture.get(),
        .before = impl_->state,
        .after = ResourceState::TransferDestination,
        .baseMip = 0,
        .mipCount = mipCount(),
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer.barrier(BarrierDesc{.textures = &toTransfer, .textureCount = 1});
    impl_->state = ResourceState::TransferDestination;

    for (uint32_t mipIndex = 0; mipIndex < impl_->uploads.size(); ++mipIndex) {
        const Impl::Upload& upload = impl_->uploads[mipIndex];
        commandBuffer.copyBufferToTexture(BufferTextureCopyDesc{
            .buffer = upload.buffer.get(),
            .texture = impl_->texture.get(),
            .width = upload.width,
            .height = upload.height,
            .depth = 1,
            .mipLevel = mipIndex,
            .baseLayer = 0,
        });
    }

    TextureBarrierDesc toShaderRead{
        .texture = impl_->texture.get(),
        .before = impl_->state,
        .after = ResourceState::ShaderRead,
        .baseMip = 0,
        .mipCount = mipCount(),
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer.barrier(BarrierDesc{.textures = &toShaderRead, .textureCount = 1});
    impl_->state = ResourceState::ShaderRead;
    impl_->uploaded = true;
    return {};
}

void ImportancePdfTexture::clear()
{
    impl_->clear();
}

bool ImportancePdfTexture::valid() const
{
    return impl_ != nullptr && impl_->chain.valid() && impl_->texture != nullptr && impl_->view != nullptr;
}

TextureView* ImportancePdfTexture::view() const
{
    return impl_ != nullptr ? impl_->view.get() : nullptr;
}

uint32_t ImportancePdfTexture::sourceWidth() const
{
    return impl_ != nullptr ? impl_->chain.sourceWidth : 0;
}

uint32_t ImportancePdfTexture::sourceHeight() const
{
    return impl_ != nullptr ? impl_->chain.sourceHeight : 0;
}

uint32_t ImportancePdfTexture::textureWidth() const
{
    return impl_ != nullptr ? impl_->chain.textureWidth : 0;
}

uint32_t ImportancePdfTexture::textureHeight() const
{
    return impl_ != nullptr ? impl_->chain.textureHeight : 0;
}

uint32_t ImportancePdfTexture::mipCount() const
{
    return impl_ != nullptr ? static_cast<uint32_t>(impl_->chain.levels.size()) : 0;
}

float ImportancePdfTexture::totalWeight() const
{
    return impl_ != nullptr ? impl_->chain.totalWeight : 0.0f;
}

uint64_t ImportancePdfTexture::byteSize() const
{
    return impl_ != nullptr ? impl_->byteSize : 0;
}

} // namespace metallic::render

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/LightingKernelParameters.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <iterator>
#include <utility>
#include <vector>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

uint32_t paddedDimension(uint32_t value)
{
    if (value == 0 || value > (1u << 30u)) {
        return 0;
    }
    return std::bit_ceil(value);
}

std::string resultMessage(std::string_view label, Result<> result)
{
    std::string message(label);
    message += " returned ";
    message += resultToString(result);
    return message;
}

} // namespace

ImportancePdfSize computeImportancePdfTextureSize(uint32_t maxItems)
{
    ImportancePdfSize size;
    const uint32_t itemCount = std::max(maxItems, 1u);
    while (static_cast<uint64_t>(size.width) * size.width < itemCount) {
        size.width *= 2u;
    }
    const uint32_t requiredRows = static_cast<uint32_t>(
        (static_cast<uint64_t>(itemCount) + size.width - 1u) / size.width);
    size.height = std::bit_ceil(std::max(requiredRows, 1u));
    size.mipCount = std::bit_width(std::max(size.width, size.height));
    return size;
}

struct ImportancePdfTexture::Impl {
    uint32_t sourceWidth = 0;
    uint32_t sourceHeight = 0;
    uint32_t textureWidth = 0;
    uint32_t textureHeight = 0;
    uint32_t mipCount = 0;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    std::vector<std::unique_ptr<TextureView>> ownedMipViews;
    ResourceState state = ResourceState::Undefined;
    uint64_t byteSize = 0;

};

ImportancePdfTexture::ImportancePdfTexture()
    : impl_(std::make_shared<Impl>())
{
}

ImportancePdfTexture::~ImportancePdfTexture() = default;
ImportancePdfTexture::ImportancePdfTexture(ImportancePdfTexture&&) noexcept = default;
ImportancePdfTexture& ImportancePdfTexture::operator=(ImportancePdfTexture&&) noexcept = default;

Result<> ImportancePdfTexture::initialize(
    Device& device,
    uint32_t sourceWidth,
    uint32_t sourceHeight,
    std::string_view debugName,
    std::string& log)
{
    // Submitted frames retain the previous allocation if a PDF is resized or
    // its owner is cleared while that frame is still in flight.
    impl_ = std::make_shared<Impl>();
    impl_->sourceWidth = sourceWidth;
    impl_->sourceHeight = sourceHeight;
    impl_->textureWidth = paddedDimension(sourceWidth);
    impl_->textureHeight = paddedDimension(sourceHeight);
    if (impl_->textureWidth == 0 || impl_->textureHeight == 0) {
        log = std::string(debugName) + " importance PDF dimensions are invalid";
        return makeError(Error::InvalidArgument);
    }
    impl_->mipCount = std::bit_width(std::max(impl_->textureWidth, impl_->textureHeight));
    if (impl_->mipCount == 0 || impl_->mipCount > kImportancePdfMaxMipCount) {
        log = std::string(debugName) + " importance PDF mip count exceeds the supported limit";
        return makeError(Error::Unsupported);
    }
    uint32_t mipWidth = impl_->textureWidth;
    uint32_t mipHeight = impl_->textureHeight;
    for (uint32_t mipIndex = 0; mipIndex < impl_->mipCount; ++mipIndex) {
        impl_->byteSize += static_cast<uint64_t>(mipWidth) * mipHeight * sizeof(float);
        mipWidth = std::max(mipWidth / 2u, 1u);
        mipHeight = std::max(mipHeight / 2u, 1u);
    }

    Result<> result = device.createTexture(TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage,
            .format = Format::R32Sfloat,
            .width = impl_->textureWidth,
            .height = impl_->textureHeight,
            .depth = 1,
            .mipCount = impl_->mipCount,
            .layerCount = 1,
            .memoryLocation = MemoryLocation::Device,
        }).transform([&](auto rhiValue) { impl_->texture = std::move(rhiValue); });
    if (!result || impl_->texture == nullptr) {
        log = resultMessage(std::string("createTexture(") + std::string(debugName) + ")", result);
        return result ? makeError(Error::Failure) : result;
    }

    result = device.createTextureView(*impl_->texture,
        TextureViewDesc{
            .format = Format::R32Sfloat,
            .range = {.baseMip = 0, .mipCount = impl_->mipCount, .baseLayer = 0, .layerCount = 1},
        }).transform([&](auto rhiValue) { impl_->view = std::move(rhiValue); });
    if (!result || impl_->view == nullptr) {
        log = resultMessage(std::string("createTextureView(") + std::string(debugName) + ")", result);
        return result ? makeError(Error::Failure) : result;
    }

    impl_->ownedMipViews.reserve(impl_->mipCount);
    for (uint32_t mipIndex = 0; mipIndex < impl_->mipCount; ++mipIndex) {
        std::unique_ptr<TextureView> mipView;
        result = device.createTextureView(*impl_->texture,
            TextureViewDesc{
                .format = Format::R32Sfloat,
                .range = {.baseMip = mipIndex, .mipCount = 1, .baseLayer = 0, .layerCount = 1},
            }).transform([&](auto rhiValue) { mipView = std::move(rhiValue); });
        if (!result || mipView == nullptr) {
            log = resultMessage(std::string("createTextureView(") + std::string(debugName) + " mip)", result);
            return result ? makeError(Error::Failure) : result;
        }
        impl_->ownedMipViews.push_back(std::move(mipView));
    }
    return {};
}

Result<> ImportancePdfTexture::beginGpuBuild(CommandBuffer& commandBuffer)
{
    if (!valid()) {
        return {};
    }
    if (auto* frame = metallic::render::RenderFrameContext::from(commandBuffer)) { frame->retain(impl_); }
    TextureBarrierDesc toGeneral{
        .texture = impl_->texture.get(),
        .oldLayout = textureLayoutForResourceState(impl_->state),
        .newLayout = TextureLayout::General,
        .before = resourceSyncScope(impl_->state, PipelineStageBits::AllCommands),
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .range = {.baseMip = 0, .mipCount = mipCount(), .baseLayer = 0, .layerCount = 1},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{.textures = {&toGeneral, 1}}); !commandResult) { return commandResult; }
    impl_->state = ResourceState::General;
    return {};
}

Result<> ImportancePdfTexture::synchronizeGpuBuild(CommandBuffer& commandBuffer)
{
    if (!valid() || impl_->state != ResourceState::General) {
        return {};
    }
    TextureBarrierDesc synchronize{
        .texture = impl_->texture.get(),
        .oldLayout = TextureLayout::General,
        .newLayout = TextureLayout::General,
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .range = {.baseMip = 0, .mipCount = mipCount(), .baseLayer = 0, .layerCount = 1},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{.textures = {&synchronize, 1}}); !commandResult) { return commandResult; }
    return {};
}

Result<> ImportancePdfTexture::endGpuBuild(CommandBuffer& commandBuffer)
{
    if (!valid() || impl_->state != ResourceState::General) {
        return {};
    }
    TextureBarrierDesc toShaderRead{
        .texture = impl_->texture.get(),
        .oldLayout = textureLayoutForResourceState(impl_->state),
        .newLayout = TextureLayout::ShaderRead,
        .before = resourceSyncScope(impl_->state, PipelineStageBits::AllCommands),
        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
        .range = {.baseMip = 0, .mipCount = mipCount(), .baseLayer = 0, .layerCount = 1},
    };
    if (auto commandResult = commandBuffer.synchronize(BarrierDesc{.textures = {&toShaderRead, 1}}); !commandResult) { return commandResult; }
    impl_->state = ResourceState::ShaderRead;
    return {};
}

void ImportancePdfTexture::clear()
{
    impl_ = std::make_shared<Impl>();
}

bool ImportancePdfTexture::valid() const
{
    return impl_ != nullptr &&
        impl_->sourceWidth != 0 &&
        impl_->sourceHeight != 0 &&
        impl_->textureWidth != 0 &&
        impl_->textureHeight != 0 &&
        impl_->mipCount != 0 &&
        impl_->texture != nullptr &&
        impl_->view != nullptr &&
        impl_->ownedMipViews.size() == impl_->mipCount;
}

TextureView* ImportancePdfTexture::view() const
{
    return impl_ != nullptr ? impl_->view.get() : nullptr;
}

TextureView* ImportancePdfTexture::mipView(uint32_t mipLevel) const
{
    return impl_ != nullptr && mipLevel < impl_->ownedMipViews.size()
        ? impl_->ownedMipViews[mipLevel].get() : nullptr;
}

uint32_t ImportancePdfTexture::sourceWidth() const
{
    return impl_ != nullptr ? impl_->sourceWidth : 0;
}

uint32_t ImportancePdfTexture::sourceHeight() const
{
    return impl_ != nullptr ? impl_->sourceHeight : 0;
}

uint32_t ImportancePdfTexture::textureWidth() const
{
    return impl_ != nullptr ? impl_->textureWidth : 0;
}

uint32_t ImportancePdfTexture::textureHeight() const
{
    return impl_ != nullptr ? impl_->textureHeight : 0;
}

uint32_t ImportancePdfTexture::mipCount() const
{
    return impl_ != nullptr ? impl_->mipCount : 0;
}

uint64_t ImportancePdfTexture::byteSize() const
{
    return impl_ != nullptr ? impl_->byteSize : 0;
}

namespace {

inline constexpr const char* kPrepareLightsPdfShaderModuleName = "Features/Lighting/PrepareLightsPdf";
inline constexpr const char* kPrepareLightsPdfEntryPoint = "prepareLightsPdfMain";
inline constexpr uint32_t kPrepareLocalLightsMode = 0;
inline constexpr uint32_t kPrepareEnvironmentMode = 1;
inline constexpr uint32_t kGenerateLocalMipMode = 2;
inline constexpr uint32_t kGenerateEnvironmentMipMode = 3;

uint32_t dimensionAtMip(uint32_t dimension, uint32_t mipLevel)
{
    return std::max(dimension >> mipLevel, 1u);
}

} // namespace

struct ImportancePdfCompute::Impl {
    ComputeKernel program;
    Device* device = nullptr;
};

ImportancePdfCompute::ImportancePdfCompute()
    : impl_(std::make_unique<Impl>())
{
}

ImportancePdfCompute::~ImportancePdfCompute() = default;
ImportancePdfCompute::ImportancePdfCompute(ImportancePdfCompute&&) noexcept = default;
ImportancePdfCompute& ImportancePdfCompute::operator=(ImportancePdfCompute&&) noexcept = default;

Result<> ImportancePdfCompute::initialize(Device& device, std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->program.valid()) {
        return {};
    }

    ShaderCompileResult compileResult;
    const Result<> compile = compileSlangShaderToSpirv(SlangShaderDesc{
            .moduleName = kPrepareLightsPdfShaderModuleName,
            .entryPointName = kPrepareLightsPdfEntryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
        }, compileResult.diagnostics).transform([&](auto value) { compileResult = std::move(value); });
    if (!compile) {
        log = resultMessage("compileSlangShaderToSpirv(PrepareLightsPdf)", compile);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        return compile;
    }

    impl_->device = &device;
    return impl_->program.initialize(
        device,
        ComputeKernelDesc{
            .spirv = compileResult.spirv,
            .parameters = parameterAbi<PrepareLightsPdfParams>(kPrepareLightsPdfABI, ParameterTransport::InlinePush),
            .debugName = "ImportancePdfCompute",
        },
        log);
}

Result<> ImportancePdfCompute::buildLocalLights(
    CommandBuffer& commandBuffer,
    ImportancePdfTexture& localLightPdf,
    Buffer& punctualLights,
    uint32_t lightCount)
{
    if (!valid() || !localLightPdf.valid() ||
        !hasFlag(punctualLights.desc().usage, BufferUsageBits::Storage) ||
        uint64_t(localLightPdf.textureWidth()) * localLightPdf.textureHeight() < lightCount ||
        punctualLights.desc().size < (uint64_t(lightCount) + 1u) * sizeof(float) * 16u) {
        return makeError(Error::InvalidArgument);
    }
    if (auto* frame = metallic::render::RenderFrameContext::from(commandBuffer); frame != nullptr && !frame->recording()) {
        return makeError(Error::InvalidArgument);
    }
    commandBuffer.hostWriteBarrier();

    auto registry = metallic::render::ResourceRegistry::forDevice(*impl_->device);
    if (!registry) { return makeError(registry.error()); }
    auto dispatch = [&](const PrepareLightsPdfPush& push) -> Result<> {
        ParameterWriter writer(*impl_->device, **registry, metallic::render::RenderFrameContext::from(commandBuffer));
        PrepareLightsPdfParams params{.settings = push};
        const bool reduction = push.mode >= kGenerateLocalMipMode;
        if (reduction) {
            params.sourceMip = writer.storageImage(localLightPdf.mipView(push.sourceMipLevel));
        } else {
            params.lights = writer.bufferSpan(&punctualLights, 64, 16);
        }
        params.destinationMip = writer.storageImage(
            localLightPdf.mipView(reduction ? push.sourceMipLevel + 1u : 0u));
        auto encoded = writer.encode(params, kPrepareLightsPdfABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return impl_->program.dispatch(commandBuffer, *encoded,
            (push.destinationSize[0] + 7u) / 8u, (push.destinationSize[1] + 7u) / 8u);
    };

    if (auto commandResult = localLightPdf.beginGpuBuild(commandBuffer); !commandResult) { return commandResult; }
    PrepareLightsPdfPush push{};
    push.mode = kPrepareLocalLightsMode;
    push.lightCount = lightCount;
    push.sourceSize[0] = localLightPdf.textureWidth();
    push.sourceSize[1] = localLightPdf.textureHeight();
    push.destinationSize[0] = localLightPdf.textureWidth();
    push.destinationSize[1] = localLightPdf.textureHeight();
    Result<> result = dispatch(push);
    if (result) {
        if (auto commandResult = localLightPdf.synchronizeGpuBuild(commandBuffer); !commandResult) { return commandResult; }
    }
    for (uint32_t sourceMip = 0;
         result && sourceMip + 1u < localLightPdf.mipCount();
         ++sourceMip) {
        push.mode = kGenerateLocalMipMode;
        push.sourceMipLevel = sourceMip;
        push.sourceSize[0] = dimensionAtMip(localLightPdf.textureWidth(), sourceMip);
        push.sourceSize[1] = dimensionAtMip(localLightPdf.textureHeight(), sourceMip);
        push.destinationSize[0] = dimensionAtMip(localLightPdf.textureWidth(), sourceMip + 1u);
        push.destinationSize[1] = dimensionAtMip(localLightPdf.textureHeight(), sourceMip + 1u);
        result = dispatch(push);
        if (result) {
            if (auto commandResult = localLightPdf.synchronizeGpuBuild(commandBuffer); !commandResult) { return commandResult; }
        }
    }
    if (auto commandResult = localLightPdf.endGpuBuild(commandBuffer); !commandResult) { return commandResult; }
    return result;
}

Result<> ImportancePdfCompute::buildEnvironment(
    CommandBuffer& commandBuffer,
    TextureView& environmentMap,
    ImportancePdfTexture& environmentPdf)
{
    if (!valid() || !environmentPdf.valid()) {
        return makeError(Error::InvalidArgument);
    }
    if (auto* frame = metallic::render::RenderFrameContext::from(commandBuffer)) {
        if (!frame->recording()) { return makeError(Error::InvalidArgument); }
    }
    commandBuffer.hostWriteBarrier();

    auto registry = metallic::render::ResourceRegistry::forDevice(*impl_->device);
    if (!registry) { return makeError(registry.error()); }
    auto dispatch = [&](const PrepareLightsPdfPush& push) -> Result<> {
        ParameterWriter writer(*impl_->device, **registry, metallic::render::RenderFrameContext::from(commandBuffer));
        PrepareLightsPdfParams params{.settings = push};
        const bool reduction = push.mode >= kGenerateLocalMipMode;
        if (reduction) {
            params.sourceMip = writer.storageImage(environmentPdf.mipView(push.sourceMipLevel));
        } else {
            params.environment = writer.sampledImage(&environmentMap);
        }
        params.destinationMip = writer.storageImage(
            environmentPdf.mipView(reduction ? push.sourceMipLevel + 1u : 0u));
        auto encoded = writer.encode(params, kPrepareLightsPdfABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return impl_->program.dispatch(commandBuffer, *encoded,
            (push.destinationSize[0] + 7u) / 8u, (push.destinationSize[1] + 7u) / 8u);
    };

    if (auto commandResult = environmentPdf.beginGpuBuild(commandBuffer); !commandResult) { return commandResult; }
    PrepareLightsPdfPush push{};
    push.mode = kPrepareEnvironmentMode;
    push.sourceSize[0] = environmentPdf.sourceWidth();
    push.sourceSize[1] = environmentPdf.sourceHeight();
    push.destinationSize[0] = environmentPdf.textureWidth();
    push.destinationSize[1] = environmentPdf.textureHeight();
    Result<> result = dispatch(push);
    if (result) {
        if (auto commandResult = environmentPdf.synchronizeGpuBuild(commandBuffer); !commandResult) { return commandResult; }
    }
    for (uint32_t sourceMip = 0;
         result && sourceMip + 1u < environmentPdf.mipCount();
         ++sourceMip) {
        push.mode = kGenerateEnvironmentMipMode;
        push.sourceMipLevel = sourceMip;
        push.sourceSize[0] = dimensionAtMip(environmentPdf.textureWidth(), sourceMip);
        push.sourceSize[1] = dimensionAtMip(environmentPdf.textureHeight(), sourceMip);
        push.destinationSize[0] = dimensionAtMip(environmentPdf.textureWidth(), sourceMip + 1u);
        push.destinationSize[1] = dimensionAtMip(environmentPdf.textureHeight(), sourceMip + 1u);
        result = dispatch(push);
        if (result) {
            if (auto commandResult = environmentPdf.synchronizeGpuBuild(commandBuffer); !commandResult) { return commandResult; }
        }
    }
    if (auto commandResult = environmentPdf.endGpuBuild(commandBuffer); !commandResult) { return commandResult; }
    return result;
}

void ImportancePdfCompute::clear()
{
    if (impl_ != nullptr) {
        impl_->program.clear();
        impl_->device = nullptr;
    }
}

bool ImportancePdfCompute::valid() const
{
    return impl_ != nullptr && impl_->program.valid() && impl_->device != nullptr;
}

} // namespace metallic::render

#include "Runtime/Render/ImportanceSampling.h"
#include "Runtime/Render/GAPI/SceneRtx.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

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
    uint32_t sourceWidth = 0;
    uint32_t sourceHeight = 0;
    uint32_t textureWidth = 0;
    uint32_t textureHeight = 0;
    uint32_t mipCount = 0;
    std::unique_ptr<Texture> texture;
    std::unique_ptr<TextureView> view;
    std::vector<std::unique_ptr<TextureView>> ownedMipViews;
    std::array<TextureView*, kImportancePdfMaxMipCount> mipViews{};
    ResourceState state = ResourceState::Undefined;
    uint64_t byteSize = 0;

    void clear()
    {
        sourceWidth = 0;
        sourceHeight = 0;
        textureWidth = 0;
        textureHeight = 0;
        mipCount = 0;
        mipViews.fill(nullptr);
        ownedMipViews.clear();
        view.reset();
        texture.reset();
        state = ResourceState::Undefined;
        byteSize = 0;
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
    uint32_t sourceWidth,
    uint32_t sourceHeight,
    std::string_view debugName,
    std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_unique<Impl>();
    }
    impl_->clear();
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
        log = std::string(debugName) + " importance PDF mip count exceeds the shader limit";
        return makeError(Error::Unsupported);
    }
    uint32_t mipWidth = impl_->textureWidth;
    uint32_t mipHeight = impl_->textureHeight;
    for (uint32_t mipIndex = 0; mipIndex < impl_->mipCount; ++mipIndex) {
        impl_->byteSize += static_cast<uint64_t>(mipWidth) * mipHeight * sizeof(float);
        mipWidth = std::max(mipWidth / 2u, 1u);
        mipHeight = std::max(mipHeight / 2u, 1u);
    }

    Result result = device.createTexture(
        TextureDesc{
            .type = TextureType::Texture2D,
            .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage,
            .format = Format::R32Sfloat,
            .width = impl_->textureWidth,
            .height = impl_->textureHeight,
            .depth = 1,
            .mipCount = impl_->mipCount,
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
            .mipCount = impl_->mipCount,
            .baseLayer = 0,
            .layerCount = 1,
        },
        impl_->view);
    if (!result || impl_->view == nullptr) {
        log = resultMessage(std::string("createTextureView(") + std::string(debugName) + ")", result);
        return result ? makeError(Error::Failure) : result;
    }

    impl_->ownedMipViews.reserve(impl_->mipCount);
    for (uint32_t mipIndex = 0; mipIndex < impl_->mipCount; ++mipIndex) {
        std::unique_ptr<TextureView> mipView;
        result = device.createTextureView(
            *impl_->texture,
            TextureViewDesc{
                .format = Format::R32Sfloat,
                .baseMip = mipIndex,
                .mipCount = 1,
                .baseLayer = 0,
                .layerCount = 1,
            },
            mipView);
        if (!result || mipView == nullptr) {
            log = resultMessage(std::string("createTextureView(") + std::string(debugName) + " mip)", result);
            return result ? makeError(Error::Failure) : result;
        }
        impl_->mipViews[mipIndex] = mipView.get();
        impl_->ownedMipViews.push_back(std::move(mipView));
    }
    for (uint32_t mipIndex = impl_->mipCount; mipIndex < kImportancePdfMaxMipCount; ++mipIndex) {
        impl_->mipViews[mipIndex] = impl_->mipViews[impl_->mipCount - 1u];
    }
    return {};
}

void ImportancePdfTexture::beginGpuBuild(CommandBuffer& commandBuffer)
{
    if (!valid()) {
        return;
    }
    TextureBarrierDesc toGeneral{
        .texture = impl_->texture.get(),
        .before = impl_->state,
        .after = ResourceState::General,
        .baseMip = 0,
        .mipCount = mipCount(),
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer.barrier(BarrierDesc{.textures = &toGeneral, .textureCount = 1});
    impl_->state = ResourceState::General;
}

void ImportancePdfTexture::synchronizeGpuBuild(CommandBuffer& commandBuffer)
{
    if (!valid() || impl_->state != ResourceState::General) {
        return;
    }
    TextureBarrierDesc synchronize{
        .texture = impl_->texture.get(),
        .before = ResourceState::General,
        .after = ResourceState::General,
        .baseMip = 0,
        .mipCount = mipCount(),
        .baseLayer = 0,
        .layerCount = 1,
    };
    commandBuffer.barrier(BarrierDesc{.textures = &synchronize, .textureCount = 1});
}

void ImportancePdfTexture::endGpuBuild(CommandBuffer& commandBuffer)
{
    if (!valid() || impl_->state != ResourceState::General) {
        return;
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
}

void ImportancePdfTexture::clear()
{
    if (impl_ != nullptr) {
        impl_->clear();
    }
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

TextureView* const* ImportancePdfTexture::mipViews() const
{
    return impl_ != nullptr ? impl_->mipViews.data() : nullptr;
}

uint32_t ImportancePdfTexture::mipViewCount() const
{
    return valid() ? kImportancePdfMaxMipCount : 0;
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

inline constexpr const char* kPrepareLightsPdfShaderModuleName = "PrepareLightsPdf";
inline constexpr const char* kPrepareLightsPdfEntryPoint = "prepareLightsPdfMain";
inline constexpr uint32_t kPrepareLocalLightsMode = 0;
inline constexpr uint32_t kPrepareEnvironmentMode = 1;
inline constexpr uint32_t kGenerateLocalMipMode = 2;
inline constexpr uint32_t kGenerateEnvironmentMipMode = 3;

struct PrepareLightsPdfPush {
    uint32_t mode = 0;
    uint32_t lightCount = 0;
    uint32_t sourceMipLevel = 0;
    uint32_t padding0 = 0;
    uint32_t sourceSize[2] = {1, 1};
    uint32_t destinationSize[2] = {1, 1};
    float localLightIntensity = 0.0f;
    float sceneRadius = 1.0f;
    uint32_t padding1 = 0;
    uint32_t padding2 = 0;
};

static_assert(sizeof(PrepareLightsPdfPush) == 48);

uint32_t dimensionAtMip(uint32_t dimension, uint32_t mipLevel)
{
    return std::max(dimension >> mipLevel, 1u);
}

} // namespace

struct ImportancePdfCompute::Impl {
    SceneRayQueryProgram program;
};

ImportancePdfCompute::ImportancePdfCompute()
    : impl_(std::make_unique<Impl>())
{
}

ImportancePdfCompute::~ImportancePdfCompute() = default;
ImportancePdfCompute::ImportancePdfCompute(ImportancePdfCompute&&) noexcept = default;
ImportancePdfCompute& ImportancePdfCompute::operator=(ImportancePdfCompute&&) noexcept = default;

Result ImportancePdfCompute::initialize(Device& device, std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_unique<Impl>();
    }
    if (impl_->program.valid()) {
        return {};
    }

    ShaderCompileResult compileResult;
    const Result compile = compileSlangShaderToSpirv(
        SlangShaderDesc{
            .moduleName = kPrepareLightsPdfShaderModuleName,
            .entryPointName = kPrepareLightsPdfEntryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
        },
        compileResult);
    if (!compile) {
        log = resultMessage("compileSlangShaderToSpirv(PrepareLightsPdf)", compile);
        if (!compileResult.diagnostics.empty()) {
            log += ": ";
            log += compileResult.diagnostics;
        }
        return compile;
    }

    const SceneRayQueryBindingDesc bindings[] = {
        {.binding = 0, .kind = SceneRayQueryBindingKind::SampledImage},
        {
            .binding = 1,
            .kind = SceneRayQueryBindingKind::StorageImage,
            .descriptorCount = kImportancePdfMaxMipCount,
        },
        {
            .binding = 2,
            .kind = SceneRayQueryBindingKind::StorageImage,
            .descriptorCount = kImportancePdfMaxMipCount,
        },
    };
    return impl_->program.initialize(
        device,
        SceneRayQueryProgramDesc{
            .spirv = compileResult.spirv.data(),
            .byteSize = static_cast<uint64_t>(compileResult.spirv.size() * sizeof(uint32_t)),
            .pushConstantSize = sizeof(PrepareLightsPdfPush),
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .debugName = "ImportancePdfCompute",
            .descriptorSetCount = kImportancePdfMaxMipCount * 2u,
        },
        log);
}

Result ImportancePdfCompute::build(
    CommandBuffer& commandBuffer,
    TextureView& environmentMap,
    ImportancePdfTexture& localLightPdf,
    uint32_t lightCount,
    float localLightIntensity,
    float sceneRadius,
    ImportancePdfTexture& environmentPdf,
    bool rebuildEnvironment)
{
    if (!valid() || !localLightPdf.valid() || !environmentPdf.valid() || lightCount == 0) {
        return makeError(Error::InvalidArgument);
    }

    TextureView* const environmentViews[] = {&environmentMap};
    const SceneRayQueryDispatchBinding bindings[] = {
        {
            .binding = 0,
            .textureViews = environmentViews,
            .textureViewCount = static_cast<uint32_t>(std::size(environmentViews)),
        },
        {
            .binding = 1,
            .textureViews = localLightPdf.mipViews(),
            .textureViewCount = localLightPdf.mipViewCount(),
        },
        {
            .binding = 2,
            .textureViews = environmentPdf.mipViews(),
            .textureViewCount = environmentPdf.mipViewCount(),
        },
    };
    auto dispatch = [&](const PrepareLightsPdfPush& push, uint32_t descriptorSetIndex) {
        return impl_->program.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (push.destinationSize[0] + 7u) / 8u,
            .groupCountY = (push.destinationSize[1] + 7u) / 8u,
            .groupCountZ = 1,
            .descriptorSetIndex = descriptorSetIndex,
        });
    };

    localLightPdf.beginGpuBuild(commandBuffer);
    environmentPdf.beginGpuBuild(commandBuffer);
    auto finish = [&]() {
        localLightPdf.endGpuBuild(commandBuffer);
        environmentPdf.endGpuBuild(commandBuffer);
    };

    PrepareLightsPdfPush push;
    push.mode = kPrepareLocalLightsMode;
    push.lightCount = lightCount;
    push.sourceSize[0] = localLightPdf.textureWidth();
    push.sourceSize[1] = localLightPdf.textureHeight();
    push.destinationSize[0] = localLightPdf.textureWidth();
    push.destinationSize[1] = localLightPdf.textureHeight();
    push.localLightIntensity = localLightIntensity;
    push.sceneRadius = sceneRadius;
    Result result = dispatch(push, 0u);
    if (!result) {
        finish();
        return result;
    }
    localLightPdf.synchronizeGpuBuild(commandBuffer);

    for (uint32_t sourceMip = 0; sourceMip + 1u < localLightPdf.mipCount(); ++sourceMip) {
        push.mode = kGenerateLocalMipMode;
        push.sourceMipLevel = sourceMip;
        push.sourceSize[0] = dimensionAtMip(localLightPdf.textureWidth(), sourceMip);
        push.sourceSize[1] = dimensionAtMip(localLightPdf.textureHeight(), sourceMip);
        push.destinationSize[0] = dimensionAtMip(localLightPdf.textureWidth(), sourceMip + 1u);
        push.destinationSize[1] = dimensionAtMip(localLightPdf.textureHeight(), sourceMip + 1u);
        result = dispatch(push, sourceMip + 1u);
        if (!result) {
            finish();
            return result;
        }
        localLightPdf.synchronizeGpuBuild(commandBuffer);
    }

    if (rebuildEnvironment) {
        push = PrepareLightsPdfPush{};
        push.mode = kPrepareEnvironmentMode;
        push.sourceSize[0] = environmentPdf.sourceWidth();
        push.sourceSize[1] = environmentPdf.sourceHeight();
        push.destinationSize[0] = environmentPdf.textureWidth();
        push.destinationSize[1] = environmentPdf.textureHeight();
        result = dispatch(push, kImportancePdfMaxMipCount);
        if (!result) {
            finish();
            return result;
        }
        environmentPdf.synchronizeGpuBuild(commandBuffer);

        for (uint32_t sourceMip = 0; sourceMip + 1u < environmentPdf.mipCount(); ++sourceMip) {
            push.mode = kGenerateEnvironmentMipMode;
            push.sourceMipLevel = sourceMip;
            push.sourceSize[0] = dimensionAtMip(environmentPdf.textureWidth(), sourceMip);
            push.sourceSize[1] = dimensionAtMip(environmentPdf.textureHeight(), sourceMip);
            push.destinationSize[0] = dimensionAtMip(environmentPdf.textureWidth(), sourceMip + 1u);
            push.destinationSize[1] = dimensionAtMip(environmentPdf.textureHeight(), sourceMip + 1u);
            result = dispatch(push, kImportancePdfMaxMipCount + sourceMip + 1u);
            if (!result) {
                finish();
                return result;
            }
            environmentPdf.synchronizeGpuBuild(commandBuffer);
        }
    }

    finish();
    return {};
}

Result ImportancePdfCompute::buildLocalLights(
    CommandBuffer& commandBuffer,
    TextureView& environmentMap,
    ImportancePdfTexture& localLightPdf,
    uint32_t lightCount,
    float localLightIntensity,
    float sceneRadius)
{
    if (!valid() || !localLightPdf.valid() || lightCount == 0) {
        return makeError(Error::InvalidArgument);
    }

    TextureView* const environmentViews[] = {&environmentMap};
    const SceneRayQueryDispatchBinding bindings[] = {
        {
            .binding = 0,
            .textureViews = environmentViews,
            .textureViewCount = static_cast<uint32_t>(std::size(environmentViews)),
        },
        {
            .binding = 1,
            .textureViews = localLightPdf.mipViews(),
            .textureViewCount = localLightPdf.mipViewCount(),
        },
        {
            .binding = 2,
            .textureViews = localLightPdf.mipViews(),
            .textureViewCount = localLightPdf.mipViewCount(),
        },
    };
    auto dispatch = [&](const PrepareLightsPdfPush& push, uint32_t descriptorSetIndex) {
        return impl_->program.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (push.destinationSize[0] + 7u) / 8u,
            .groupCountY = (push.destinationSize[1] + 7u) / 8u,
            .groupCountZ = 1,
            .descriptorSetIndex = descriptorSetIndex,
        });
    };

    localLightPdf.beginGpuBuild(commandBuffer);
    PrepareLightsPdfPush push;
    push.mode = kPrepareLocalLightsMode;
    push.lightCount = lightCount;
    push.sourceSize[0] = localLightPdf.textureWidth();
    push.sourceSize[1] = localLightPdf.textureHeight();
    push.destinationSize[0] = localLightPdf.textureWidth();
    push.destinationSize[1] = localLightPdf.textureHeight();
    push.localLightIntensity = localLightIntensity;
    push.sceneRadius = sceneRadius;
    Result result = dispatch(push, 0u);
    if (result) {
        localLightPdf.synchronizeGpuBuild(commandBuffer);
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
        result = dispatch(push, sourceMip + 1u);
        if (result) {
            localLightPdf.synchronizeGpuBuild(commandBuffer);
        }
    }
    localLightPdf.endGpuBuild(commandBuffer);
    return result;
}

Result ImportancePdfCompute::buildEnvironment(
    CommandBuffer& commandBuffer,
    TextureView& environmentMap,
    ImportancePdfTexture& environmentPdf)
{
    if (!valid() || !environmentPdf.valid()) {
        return makeError(Error::InvalidArgument);
    }

    TextureView* const environmentViews[] = {&environmentMap};
    const SceneRayQueryDispatchBinding bindings[] = {
        {
            .binding = 0,
            .textureViews = environmentViews,
            .textureViewCount = static_cast<uint32_t>(std::size(environmentViews)),
        },
        {
            .binding = 1,
            .textureViews = environmentPdf.mipViews(),
            .textureViewCount = environmentPdf.mipViewCount(),
        },
        {
            .binding = 2,
            .textureViews = environmentPdf.mipViews(),
            .textureViewCount = environmentPdf.mipViewCount(),
        },
    };
    auto dispatch = [&](const PrepareLightsPdfPush& push, uint32_t descriptorSetIndex) {
        return impl_->program.dispatch(SceneRayQueryDispatchDesc{
            .commandBuffer = &commandBuffer,
            .bindings = bindings,
            .bindingCount = static_cast<uint32_t>(std::size(bindings)),
            .pushData = &push,
            .pushDataSize = sizeof(push),
            .groupCountX = (push.destinationSize[0] + 7u) / 8u,
            .groupCountY = (push.destinationSize[1] + 7u) / 8u,
            .groupCountZ = 1,
            .descriptorSetIndex = descriptorSetIndex,
        });
    };

    environmentPdf.beginGpuBuild(commandBuffer);
    PrepareLightsPdfPush push;
    push.mode = kPrepareEnvironmentMode;
    push.sourceSize[0] = environmentPdf.sourceWidth();
    push.sourceSize[1] = environmentPdf.sourceHeight();
    push.destinationSize[0] = environmentPdf.textureWidth();
    push.destinationSize[1] = environmentPdf.textureHeight();
    Result result = dispatch(push, kImportancePdfMaxMipCount);
    if (result) {
        environmentPdf.synchronizeGpuBuild(commandBuffer);
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
        result = dispatch(push, kImportancePdfMaxMipCount + sourceMip + 1u);
        if (result) {
            environmentPdf.synchronizeGpuBuild(commandBuffer);
        }
    }
    environmentPdf.endGpuBuild(commandBuffer);
    return result;
}

void ImportancePdfCompute::clear()
{
    if (impl_ != nullptr) {
        impl_->program.clear();
    }
}

bool ImportancePdfCompute::valid() const
{
    return impl_ != nullptr && impl_->program.valid();
}

} // namespace metallic::render

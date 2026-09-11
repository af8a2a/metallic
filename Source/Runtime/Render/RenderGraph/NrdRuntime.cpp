#include "Runtime/Render/RenderGraph/NrdRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/RenderFrameContext.h"
#if METALLIC_HAS_NRD
#include "Runtime/Render/Denoising/NrdPlan.h"
#endif
#include <spdlog/spdlog.h>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace metallic::render {
namespace {
#if METALLIC_HAS_NRD
uint16_t divideRoundUp(uint32_t dividend, uint16_t divisor)
{
    return static_cast<uint16_t>((dividend + divisor - 1u) / divisor);
}

Format formatFromNrd(denoising::Format format)
{
    switch (format) {
    case denoising::Format::R8_UNORM:
        return Format::R8Unorm;
    case denoising::Format::R8_SNORM:
        return Format::R8Snorm;
    case denoising::Format::R8_UINT:
        return Format::R8Uint;
    case denoising::Format::R8_SINT:
        return Format::R8Sint;
    case denoising::Format::RG8_UNORM:
        return Format::Rg8Unorm;
    case denoising::Format::RG8_SNORM:
        return Format::Rg8Snorm;
    case denoising::Format::RG8_UINT:
        return Format::Rg8Uint;
    case denoising::Format::RG8_SINT:
        return Format::Rg8Sint;
    case denoising::Format::RGBA8_UNORM:
        return Format::Rgba8Unorm;
    case denoising::Format::RGBA8_SNORM:
        return Format::Rgba8Snorm;
    case denoising::Format::RGBA8_UINT:
        return Format::Rgba8Uint;
    case denoising::Format::RGBA8_SINT:
        return Format::Rgba8Sint;
    case denoising::Format::RGBA8_SRGB:
        return Format::Rgba8Srgb;
    case denoising::Format::R16_UNORM:
        return Format::R16Unorm;
    case denoising::Format::R16_SNORM:
        return Format::R16Snorm;
    case denoising::Format::R16_UINT:
        return Format::R16Uint;
    case denoising::Format::R16_SINT:
        return Format::R16Sint;
    case denoising::Format::R16_SFLOAT:
        return Format::R16Sfloat;
    case denoising::Format::RG16_UNORM:
        return Format::Rg16Unorm;
    case denoising::Format::RG16_SNORM:
        return Format::Rg16Snorm;
    case denoising::Format::RG16_UINT:
        return Format::Rg16Uint;
    case denoising::Format::RG16_SINT:
        return Format::Rg16Sint;
    case denoising::Format::RG16_SFLOAT:
        return Format::Rg16Sfloat;
    case denoising::Format::RGBA16_UNORM:
        return Format::Rgba16Unorm;
    case denoising::Format::RGBA16_SNORM:
        return Format::Rgba16Snorm;
    case denoising::Format::RGBA16_UINT:
        return Format::Rgba16Uint;
    case denoising::Format::RGBA16_SINT:
        return Format::Rgba16Sint;
    case denoising::Format::RGBA16_SFLOAT:
        return Format::Rgba16Sfloat;
    case denoising::Format::R32_UINT:
        return Format::R32Uint;
    case denoising::Format::R32_SINT:
        return Format::R32Sint;
    case denoising::Format::R32_SFLOAT:
        return Format::R32Sfloat;
    case denoising::Format::RG32_UINT:
        return Format::Rg32Uint;
    case denoising::Format::RG32_SINT:
        return Format::Rg32Sint;
    case denoising::Format::RG32_SFLOAT:
        return Format::Rg32Sfloat;
    case denoising::Format::RGB32_UINT:
        return Format::Rgb32Uint;
    case denoising::Format::RGB32_SINT:
        return Format::Rgb32Sint;
    case denoising::Format::RGB32_SFLOAT:
        return Format::Rgb32Sfloat;
    case denoising::Format::RGBA32_UINT:
        return Format::Rgba32Uint;
    case denoising::Format::RGBA32_SINT:
        return Format::Rgba32Sint;
    case denoising::Format::RGBA32_SFLOAT:
        return Format::Rgba32Sfloat;
    case denoising::Format::R10_G10_B10_A2_UNORM:
        return Format::A2B10G10R10UnormPack32;
    case denoising::Format::R10_G10_B10_A2_UINT:
        return Format::A2R10G10B10UintPack32;
    case denoising::Format::R11_G11_B10_UFLOAT:
        return Format::B10G11R11UfloatPack32;
    case denoising::Format::R9_G9_B9_E5_UFLOAT:
        return Format::E5B9G9R9UfloatPack32;
    case denoising::Format::MAX_NUM:
        return Format::Unknown;
    }
    return Format::Unknown;
}

struct NrdPushData {
    uint64_t constants;
    uint64_t resources;
};
struct NrdResourceIndices {
    uint32_t sampled[32]{};
    uint32_t storage[16]{};
    uint32_t samplers[2]{};
};
static_assert(sizeof(NrdPushData) == 16);
static_assert(sizeof(NrdResourceIndices) == 200);
static_assert(offsetof(NrdResourceIndices, storage) == 128);
static_assert(offsetof(NrdResourceIndices, samplers) == 192);
#endif
} // namespace

Format nrdNormalRoughnessFormat()
{
    // Must match the vendored frontend's NRD_NORMAL_ENCODING = 2.
    return Format::A2B10G10R10UnormPack32;
}

#if METALLIC_HAS_NRD
struct NrdRuntime::Impl {
    explicit Impl(bool sigmaOnly) : plan(sigmaOnly) {}
    struct TextureResource {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
        ResourceState state = ResourceState::Undefined;
    };
    struct Handles {
        BindlessHandle sampled;
        BindlessHandle storage;
    };
    denoising::NrdPlan plan;
    Device* device = nullptr;
    uint16_t width = 0, height = 0;
    std::vector<TextureResource> permanentTextures, transientTextures;
    NrdUserTexturePool userTexturePool{};
    std::unique_ptr<BindlessHeap> descriptorHeap;
    std::array<BindlessHandle, 2> samplers;
    std::vector<BindlessHandle> sampledHandles, storageHandles;
    std::unordered_map<TextureView*, Handles> textureHandles;
    std::vector<std::unique_ptr<ComputePipeline>> pipelines;
    uint32_t sampledCursor = 0, storageCursor = 0;
    bool clearPending = true;
    bool historyInvalid = true;
    bool frameReady = false;
    std::array<bool, 5> scheduled{};

    Result pipeline(uint32_t index)
    {
        if (pipelines[index])
            return {};
        const auto& recipe = plan.pipelines()[index];
        std::vector<SlangMacroDefine> defines;
        for (const auto& define : recipe.defines)
            defines.push_back({define.name, define.value});
        const char* searchPaths[] = {PROJECT_SOURCE_DIR "/External/MathLib"};
        ShaderCompileResult compiled;
        Result result = compileSlangShaderToSpirv(
            {
                .moduleName = recipe.shaderName.c_str(),
                .entryPointName = "main",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders/Libraries/Denoising/NRD",
                .additionalSearchPaths = searchPaths,
                .additionalSearchPathCount = 1,
                .macroDefines = defines.data(),
                .macroDefineCount = static_cast<uint32_t>(defines.size()),
            },
            compiled);
        if (!result) {
            spdlog::error("NRD {}: {}", recipe.shaderName, compiled.diagnostics);
            return result;
        }
        std::unique_ptr<ShaderModule> shader;
        result = device->createShaderModule({.code = compiled.spirv.data(),
                                             .byteSize = compiled.spirv.size() * sizeof(uint32_t),
                                             .debugName = recipe.shaderName.c_str()},
                                            shader);
        if (!result)
            return result;
        // Shader source already accesses the native heap. No binding remap,
        // register shifts, descriptor sets, or NRD shader blobs are involved.
        return device->createComputePipeline({.computeShader = shader.get(),
                                              .computeEntryPoint = "main",
                                              .usesBindlessHeap = true,
                                              .bindlessUserPushDataSize = sizeof(NrdPushData)},
                                             pipelines[index]);
    }
};

NrdRuntime::NrdRuntime() = default;
NrdRuntime::~NrdRuntime() = default;
NrdRuntime::NrdRuntime(NrdRuntime&&) noexcept = default;
NrdRuntime& NrdRuntime::operator=(NrdRuntime&&) noexcept = default;

Result NrdRuntime::initialize(Device& device, uint16_t width, uint16_t height,
                              const NrdUserTexturePool& userTexturePool, std::string& log, bool sigmaOnly)
{
    if (!width || !height)
        return makeError(Error::InvalidArgument);
    if (!device.capabilities().bindlessDescriptorHeap)
        return makeError(Error::Unsupported);
    clear();
    impl_ = std::make_shared<Impl>(sigmaOnly);
    impl_->device = &device;
    impl_->width = width;
    impl_->height = height;
    impl_->userTexturePool = userTexturePool;
    auto createTextureResource = [&](const denoising::TextureDesc& nrdDesc, Impl::TextureResource& resource) {
        const Format format = formatFromNrd(nrdDesc.format);
        if (format == Format::Unknown || nrdDesc.downsampleFactor == 0) {
            log = "NrdRuntime received an unsupported internal texture descriptor";
            return makeError(Error::Unsupported);
        }
        const uint16_t textureWidth = divideRoundUp(width, nrdDesc.downsampleFactor);
        const uint16_t textureHeight = divideRoundUp(height, nrdDesc.downsampleFactor);
        Result result = device.createTexture(
            TextureDesc{
                .type = TextureType::Texture2D,
                .usage = TextureUsageBits::Sampled | TextureUsageBits::Storage | TextureUsageBits::TransferDestination,
                .format = format,
                .width = textureWidth,
                .height = textureHeight,
                .depth = 1,
                .mipCount = 1,
                .layerCount = 1,
                .memoryLocation = MemoryLocation::Device,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
            },
            resource.texture);
        if (!result || resource.texture == nullptr) {
            log = "createTexture(NRD internal) returned ";
            log += resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        result = device.createTextureView(*resource.texture,
                                          TextureViewDesc{
                                              .format = format,
                                              .baseMip = 0,
                                              .mipCount = 1,
                                              .baseLayer = 0,
                                              .layerCount = 1,
                                          },
                                          resource.view);
        if (!result || resource.view == nullptr) {
            log = "createTextureView(NRD internal) returned ";
            log += resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        return Result{};
    };

    auto createPool = [&](const auto& descriptions, auto& textures) -> Result {
        textures.resize(descriptions.size());
        for (size_t i = 0; i < descriptions.size(); ++i) {
            auto result = createTextureResource(descriptions[i], textures[i]);
            if (!result)
                return result;
        }
        return {};
    };
    auto result = createPool(impl_->plan.permanentPool(), impl_->permanentTextures);
    if (result)
        result = createPool(impl_->plan.transientPool(), impl_->transientTextures);
    if (!result) {
        clear();
        return result;
    }
    // One descriptor per distinct view/access, reused by every stage in a frame.
    // The extra user slots allow REFERENCE diffuse/specular to bind distinct
    // signals without overwriting descriptors used by an earlier dispatch.
    const uint32_t capacity = static_cast<uint32_t>(impl_->permanentTextures.size() + impl_->transientTextures.size() +
                                                    userTexturePool.size() * 2);
    result = device.createBindlessHeap({.maxSamplers = 2, .maxSampledImages = capacity, .maxStorageImages = capacity},
                                       impl_->descriptorHeap);
    if (!result) {
        clear();
        return result;
    }
    std::array<BindlessSamplerWrite, 2> samplers;
    for (uint32_t i = 0; i < 2; ++i) {
        result = impl_->descriptorHeap->allocateSampler(impl_->samplers[i]);
        if (!result) {
            clear();
            return result;
        }
        const auto filter = i == 0 ? SamplerFilter::Nearest : SamplerFilter::Linear;
        samplers[i] = {.handle = impl_->samplers[i],
                       .sampler = {.minFilter = filter,
                                   .magFilter = filter,
                                   .mipFilter = SamplerFilter::Nearest,
                                   .addressU = SamplerAddressMode::ClampToEdge,
                                   .addressV = SamplerAddressMode::ClampToEdge,
                                   .addressW = SamplerAddressMode::ClampToEdge}};
    }
    result = impl_->descriptorHeap->writeSamplers(samplers.data(), 2);
    if (!result) {
        clear();
        return result;
    }
    impl_->sampledHandles.resize(capacity);
    impl_->storageHandles.resize(capacity);
    for (uint32_t i = 0; i < capacity; ++i) {
        result = impl_->descriptorHeap->allocateSampledImage(impl_->sampledHandles[i]);
        if (result)
            result = impl_->descriptorHeap->allocateStorageImage(impl_->storageHandles[i]);
        if (!result) {
            clear();
            return result;
        }
    }
    impl_->pipelines.resize(impl_->plan.pipelines().size());
    return {};
}

void NrdRuntime::clear()
{
    impl_.reset();
}
bool NrdRuntime::valid() const
{
    return impl_ && impl_->descriptorHeap;
}
uint16_t NrdRuntime::width() const
{
    return impl_ ? impl_->width : 0;
}
uint16_t NrdRuntime::height() const
{
    return impl_ ? impl_->height : 0;
}

void NrdRuntime::setUserPoolTexture(denoising::ResourceType resource, Texture& texture, TextureView& view)
{
    const auto i = static_cast<size_t>(resource);
    if (impl_ && i < impl_->userTexturePool.size())
        impl_->userTexturePool[i] = {&texture, &view};
}

Result NrdRuntime::setCommonSettings(const denoising::CommonSettings& settings)
{
    if (impl_)
        impl_->frameReady = false;
    if (!valid() || settings.resourceSize[0] != width() || settings.resourceSize[1] != height())
        return makeError(Error::InvalidArgument);
    auto common = settings;
    if (impl_->historyInvalid)
        common.accumulationMode = denoising::AccumulationMode::CLEAR_AND_RESTART;
    if (!impl_->plan.beginFrame(common))
        return makeError(Error::InvalidArgument);
    impl_->clearPending |=
        impl_->plan.commonSettings().accumulationMode == denoising::AccumulationMode::CLEAR_AND_RESTART;
    impl_->textureHandles.clear();
    impl_->sampledCursor = impl_->storageCursor = 0;
    impl_->scheduled.fill(false);
    impl_->frameReady = true;
    return {};
}

Result NrdRuntime::setReblurSettings(const denoising::ReblurSettings& settings)
{
    if (!valid())
        return makeError(Error::InvalidArgument);
    impl_->plan.setReblurSettings(settings);
    return {};
}
Result NrdRuntime::setRelaxSettings(const denoising::RelaxSettings& settings)
{
    if (!valid())
        return makeError(Error::InvalidArgument);
    impl_->plan.setRelaxSettings(settings);
    return {};
}

Result NrdRuntime::setSigmaSettings(const denoising::SigmaSettings& settings)
{
    if (!valid()) { return makeError(Error::InvalidArgument); }
    impl_->plan.setSigmaSettings(settings);
    return {};
}

Result NrdRuntime::denoise(NrdDenoiserMode mode, CommandBuffer& commands, Streamer& streamer)
{
    return record(static_cast<uint32_t>(mode), commands, streamer);
}
Result NrdRuntime::denoiseReference(bool specular, CommandBuffer& commands, Streamer& streamer)
{
    return record(specular ? 3 : 2, commands, streamer);
}

Result NrdRuntime::record(uint32_t index, CommandBuffer& commands, Streamer& streamer)
{
    if (!valid() || !impl_->frameReady || index >= impl_->scheduled.size() || impl_->scheduled[index])
        return makeError(Error::InvalidArgument);
    if (index == 0 && !impl_->device->capabilities().shaderImageGatherExtended)
        return makeError(Error::Unsupported);
    impl_->scheduled[index] = true;
    // A discarded recording must not become valid temporal history. Keep the
    // resources alive with the transaction until submission/cancellation.
    auto state = impl_;
    auto result = commands.addSubmissionTransaction(
        std::make_shared<SubmissionTransaction>([state] { state->historyInvalid = false; },
                                                [state] {
                                                    state->historyInvalid = true;
                                                    state->clearPending = true;
                                                    // A discarded command buffer did not perform its layout
                                                    // transitions. The next frame discards and clears every internal
                                                    // image instead.
                                                    for (auto& texture : state->permanentTextures)
                                                        texture.state = ResourceState::Undefined;
                                                    for (auto& texture : state->transientTextures)
                                                        texture.state = ResourceState::Undefined;
                                                }));
    if (!result)
        return result;
    if (impl_->clearPending) {
        auto clearPool = [&](auto& pool) {
            for (auto& texture : pool) {
                TextureBarrierDesc barrier{.texture = texture.texture.get(),
                                           .before = texture.state,
                                           .after = ResourceState::TransferDestination,
                                           .mipCount = 1,
                                           .layerCount = 1};
                commands.barrier({.textures = &barrier, .textureCount = 1});
                texture.state = ResourceState::TransferDestination;
                commands.clearColorTexture(*texture.texture, texture.state, {0, 0, 0, 0});
            }
        };
        clearPool(impl_->permanentTextures);
        clearPool(impl_->transientTextures);
        impl_->clearPending = false;
    }
    const auto dispatches = impl_->plan.schedule(index);
    commands.bindBindlessHeap(*impl_->descriptorHeap);
    for (const auto& stage : dispatches) {
        result = impl_->pipeline(stage.pipelineIndex);
        if (!result)
            return result;
        commands.beginDebugLabel({.name = stage.name, .color = {0.2f, 0.8f, 0.25f, 1.0f}});
        result = dispatch(commands, streamer, stage);
        commands.endDebugLabel();
        if (!result)
            return result;
    }
    return {};
}

Result NrdRuntime::dispatch(CommandBuffer& commands, Streamer& streamer, const denoising::DispatchDesc& stage)
{
    NrdResourceIndices indices;
    for (uint32_t i = 0; i < 2; ++i)
        indices.samplers[i] = impl_->samplers[i].shaderIndex;
    uint32_t sampled = 0, storage = 0;
    std::vector<BindlessImageWrite> writes;
    std::vector<TextureBarrierDesc> barriers;
    std::unordered_set<Texture*> transitioned;
    for (uint32_t i = 0; i < stage.resourcesNum; ++i) {
        const auto& resource = stage.resources[i];
        NrdTextureRef texture;
        ResourceState* state = nullptr;
        if (resource.type == denoising::ResourceType::PERMANENT_POOL ||
            resource.type == denoising::ResourceType::TRANSIENT_POOL) {
            auto& pool = resource.type == denoising::ResourceType::PERMANENT_POOL ? impl_->permanentTextures
                                                                                  : impl_->transientTextures;
            if (resource.indexInPool >= pool.size())
                return makeError(Error::InvalidArgument);
            auto& internal = pool[resource.indexInPool];
            texture = {internal.texture.get(), internal.view.get()};
            state = &internal.state;
        } else {
            const auto slot = static_cast<size_t>(resource.type);
            if (slot >= impl_->userTexturePool.size())
                return makeError(Error::InvalidArgument);
            texture = impl_->userTexturePool[slot];
        }
        if (!texture.texture || !texture.view)
            return makeError(Error::InvalidArgument);
        const bool output = resource.descriptorType == denoising::DescriptorType::STORAGE_TEXTURE;
        if ((output && storage >= 16) || (!output && sampled >= 32))
            return makeError(Error::InvalidArgument);
        auto& handles = impl_->textureHandles[texture.view];
        auto& handle = output ? handles.storage : handles.sampled;
        if (!handle.valid()) {
            auto& cursor = output ? impl_->storageCursor : impl_->sampledCursor;
            const auto& available = output ? impl_->storageHandles : impl_->sampledHandles;
            if (cursor >= available.size())
                return makeError(Error::OutOfMemory);
            handle = available[cursor++];
            writes.push_back({.handle = handle, .view = texture.view, .state = ResourceState::General});
        }
        if (output)
            indices.storage[storage++] = handle.shaderIndex;
        else
            indices.sampled[sampled++] = handle.shaderIndex;
        if (transitioned.insert(texture.texture).second) {
            barriers.push_back({.texture = texture.texture,
                                .before = state ? *state : ResourceState::General,
                                .after = ResourceState::General,
                                .mipCount = 1,
                                .layerCount = 1});
        }
        if (state)
            *state = ResourceState::General;
    }
    if (!writes.empty()) {
        const auto result = impl_->descriptorHeap->writeImages(writes.data(), static_cast<uint32_t>(writes.size()));
        if (!result)
            return result;
    }
    commands.barrier({.textures = barriers.data(), .textureCount = static_cast<uint32_t>(barriers.size())});
    const auto constantsOffset = streamer.streamConstantData(stage.constantBufferData, stage.constantBufferDataSize);
    const auto resourcesOffset = streamer.streamConstantData(&indices, sizeof(indices));
    auto* buffer = streamer.constantBuffer();
    if (!buffer || constantsOffset == UINT64_MAX || resourcesOffset == UINT64_MAX)
        return makeError(Error::OutOfMemory);
    const auto address = buffer->deviceAddress();
    if (!address)
        return makeError(Error::Failure);
    const NrdPushData push{address + constantsOffset, address + resourcesOffset};
    commands.bindComputePipeline(*impl_->pipelines[stage.pipelineIndex], &push, sizeof(push));
    commands.dispatch(stage.gridWidth, stage.gridHeight, 1);
    return {};
}
#endif
} // namespace metallic::render

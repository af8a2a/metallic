#include "Runtime/Render/RenderGraph/NrdRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/ComputeKernel.h"
#include "Runtime/Render/RenderFrameContext.h"
#if METALLIC_HAS_NRD
#include "Runtime/Render/Denoising/NrdPlan.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#endif
#include <spdlog/spdlog.h>
#include <algorithm>
#include <limits>
#include <unordered_map>
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

constexpr uint64_t kNrdAbi = 0x4e52445000000001ull;
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
    denoising::NrdPlan plan;
    Device* device = nullptr;
    uint16_t width = 0, height = 0;
    std::vector<TextureResource> permanentTextures, transientTextures;
    NrdUserTexturePool userTexturePool{};
    std::shared_ptr<ResourceRegistry> registry;
    std::vector<ComputeKernel> pipelines;
    bool clearPending = true;
    bool historyInvalid = true;
    bool frameReady = false;
    std::array<bool, 5> scheduled{};

    Result<> pipeline(uint32_t index)
    {
        if (pipelines[index].valid())
            return {};
        const auto& recipe = plan.pipelines()[index];
        std::vector<SlangMacroDefine> defines;
        for (const auto& define : recipe.defines)
            defines.push_back({define.name, define.value});
        const char* searchPaths[] = {PROJECT_SOURCE_DIR "/External/MathLib"};
        ShaderCompileResult compiled;
        Result<> result = compileSlangShaderToSpirv(
            {
                .moduleName = recipe.shaderName.c_str(),
                .entryPointName = "main",
                .searchPath = PROJECT_SOURCE_DIR "/Shaders/Interop/Denoising/NRD",
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
        std::string log;
        return pipelines[index].initialize(*device, {.spirv = compiled.spirv,
            .parameters = parameterAbi<NrdPushData>(kNrdAbi), .debugName = recipe.shaderName.c_str()}, log);
    }
};

NrdRuntime::NrdRuntime() = default;
NrdRuntime::~NrdRuntime() = default;
NrdRuntime::NrdRuntime(NrdRuntime&&) noexcept = default;
NrdRuntime& NrdRuntime::operator=(NrdRuntime&&) noexcept = default;

Result<> NrdRuntime::initialize(Device& device, uint16_t width, uint16_t height,
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
    auto createTextureResource = [&](const denoising::TextureDesc& nrdDesc, Impl::TextureResource& resource) -> Result<> {
        const Format format = formatFromNrd(nrdDesc.format);
        if (format == Format::Unknown || nrdDesc.downsampleFactor == 0) {
            log = "NrdRuntime received an unsupported internal texture descriptor";
            return makeError(Error::Unsupported);
        }
        const uint16_t textureWidth = divideRoundUp(width, nrdDesc.downsampleFactor);
        const uint16_t textureHeight = divideRoundUp(height, nrdDesc.downsampleFactor);
        Result<> result = device.createTexture(TextureDesc{
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
            }).transform([&](auto rhiValue) { resource.texture = std::move(rhiValue); });
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
                                          }).transform([&](auto rhiValue) { resource.view = std::move(rhiValue); });
        if (!result || resource.view == nullptr) {
            log = "createTextureView(NRD internal) returned ";
            log += resultToString(result);
            return result ? makeError(Error::Failure) : result;
        }
        return Result<>{};
    };

    auto createPool = [&](const auto& descriptions, auto& textures) -> Result<> {
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
    result = device.resourceRegistry().transform([&](auto rhiValue) { impl_->registry = std::move(rhiValue); });
    if (!result) { clear(); return result; }
    impl_->pipelines.resize(impl_->plan.pipelines().size());
    return {};
}

void NrdRuntime::clear()
{
    impl_.reset();
}
bool NrdRuntime::valid() const
{
    return impl_ && impl_->registry;
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

Result<> NrdRuntime::setCommonSettings(const denoising::CommonSettings& settings)
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
    impl_->scheduled.fill(false);
    impl_->frameReady = true;
    return {};
}

Result<> NrdRuntime::setReblurSettings(const denoising::ReblurSettings& settings)
{
    if (!valid())
        return makeError(Error::InvalidArgument);
    impl_->plan.setReblurSettings(settings);
    return {};
}
Result<> NrdRuntime::setRelaxSettings(const denoising::RelaxSettings& settings)
{
    if (!valid())
        return makeError(Error::InvalidArgument);
    impl_->plan.setRelaxSettings(settings);
    return {};
}

Result<> NrdRuntime::setSigmaSettings(const denoising::SigmaSettings& settings)
{
    if (!valid()) { return makeError(Error::InvalidArgument); }
    impl_->plan.setSigmaSettings(settings);
    return {};
}

Result<> NrdRuntime::denoise(NrdDenoiserMode mode, CommandBuffer& commands)
{
    return record(static_cast<uint32_t>(mode), commands);
}
Result<> NrdRuntime::denoiseReference(bool specular, CommandBuffer& commands)
{
    return record(specular ? 3 : 2, commands);
}

Result<> NrdRuntime::record(uint32_t index, CommandBuffer& commands)
{
    if (!commands.recording() || !commands.frameContext() || !commands.frameContext()->recording() ||
        !valid() || !impl_->frameReady || index >= impl_->scheduled.size() || impl_->scheduled[index])
        return makeError(Error::InvalidArgument);
    if (index == 0 && !impl_->device->capabilities().shaderImageGatherExtended)
        return makeError(Error::Unsupported);
    using namespace detail;
    struct RecordingGuard {
        Impl& state;
        bool complete = false;
        ~RecordingGuard()
        {
            if (!complete) {
                // schedule() rotates history before resource validation and
                // lazy pipeline preparation. Even a failure that records no
                // GPU commands must restart accumulation on the next frame.
                state.historyInvalid = true;
                state.clearPending = true;
            }
        }
    } recordingGuard{*impl_};
    // schedule() rotates NRD history slots. A failed preparation must not let
    // this same frame schedule the denoiser a second time and rotate again.
    impl_->scheduled[index] = true;
    const auto dispatches = impl_->plan.schedule(index);
    const bool clearPending = impl_->clearPending;
    std::vector<GraphAccessResource> resources;
    std::vector<GraphAccessBinding> bindings;
    std::vector<std::shared_ptr<void>> owners;
    std::vector<std::vector<ResourceState*>> trackedStates;
    std::unordered_map<const void*, size_t> identities;
    std::vector<GraphAccessPass> accesses;
    std::vector<std::vector<NrdTextureRef>> dispatchTextures;
    dispatchTextures.reserve(dispatches.size());
    accesses.reserve(dispatches.size() + (clearPending ? 1u : 0u));

    const auto importTexture = [&](NrdTextureRef texture, ResourceState* state) -> Result<size_t> {
        if (!texture.texture || !texture.view ||
            texture.view->deviceIdentity() != commands.deviceIdentity() ||
            texture.texture->desc().mipCount == 0 || texture.texture->desc().layerCount == 0) {
            return makeError(Error::InvalidArgument);
        }
        auto owner = texture.view->retainTexture();
        if (!owner || texture.texture->retainAllocation() != owner) { return makeError(Error::InvalidArgument); }
        const ResourceState initial = state ? *state : ResourceState::General;
        const SyncScope scope = !state
            ? SyncScope{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}
            : initial == ResourceState::TransferDestination
                ? SyncScope{PipelineStageBits::Transfer, AccessBits::TransferWrite}
                : SyncScope{PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite};
        const auto [entry, inserted] = identities.try_emplace(owner.get(), resources.size());
        const size_t resource = entry->second;
        if (inserted) {
            resources.push_back({.type = RenderGraphResourceType::Texture2D, .state = initial, .scope = scope});
            bindings.push_back({.texture = texture.texture, .mipCount = texture.texture->desc().mipCount,
                .layerCount = texture.texture->desc().layerCount});
            owners.push_back(std::move(owner));
            trackedStates.emplace_back();
        } else {
            if (resources[resource].state != initial) { return makeError(Error::InvalidArgument); }
            resources[resource].scope.stages = resources[resource].scope.stages | scope.stages;
            resources[resource].scope.access = resources[resource].scope.access | scope.access;
        }
        if (state && std::find(trackedStates[resource].begin(), trackedStates[resource].end(), state) ==
            trackedStates[resource].end()) {
            trackedStates[resource].push_back(state);
        }
        return resource;
    };

    // Validate every resource and compile the complete sequence before emitting
    // commands. Aliased NRD slots refer to one allocation in the access plan.
    if (clearPending) {
        auto& clear = accesses.emplace_back();
        const auto appendPool = [&](auto& pool) -> Result<> {
            for (auto& texture : pool) {
                auto resource = importTexture({texture.texture.get(), texture.view.get()}, &texture.state);
                if (!resource) { return makeError(resource.error()); }
                clear.uses.push_back({*resource, ResourceState::TransferDestination,
                    {PipelineStageBits::Transfer, AccessBits::TransferWrite}, true});
            }
            return {};
        };
        auto result = appendPool(impl_->permanentTextures);
        if (result) { result = appendPool(impl_->transientTextures); }
        if (!result) { return result; }
    }
    for (const auto& stage : dispatches) {
        if (stage.pipelineIndex >= impl_->pipelines.size() || !stage.name || !stage.gridWidth || !stage.gridHeight ||
            (stage.resourcesNum && !stage.resources) || (stage.constantBufferDataSize && !stage.constantBufferData)) {
            return makeError(Error::InvalidArgument);
        }
        auto& access = accesses.emplace_back();
        auto& textures = dispatchTextures.emplace_back();
        textures.reserve(stage.resourcesNum);
        uint32_t sampled = 0, storage = 0;
        for (uint32_t i = 0; i < stage.resourcesNum; ++i) {
            const auto& resource = stage.resources[i];
            if (resource.descriptorType != denoising::DescriptorType::TEXTURE &&
                resource.descriptorType != denoising::DescriptorType::STORAGE_TEXTURE) {
                return makeError(Error::InvalidArgument);
            }
            const bool output = resource.descriptorType == denoising::DescriptorType::STORAGE_TEXTURE;
            if ((output && ++storage > 16) || (!output && ++sampled > 32)) {
                return makeError(Error::InvalidArgument);
            }
            NrdTextureRef texture;
            ResourceState* state = nullptr;
            if (resource.type == denoising::ResourceType::PERMANENT_POOL ||
                resource.type == denoising::ResourceType::TRANSIENT_POOL) {
                auto& pool = resource.type == denoising::ResourceType::PERMANENT_POOL
                    ? impl_->permanentTextures : impl_->transientTextures;
                if (resource.indexInPool >= pool.size()) { return makeError(Error::InvalidArgument); }
                auto& internal = pool[resource.indexInPool];
                texture = {internal.texture.get(), internal.view.get()};
                state = &internal.state;
            } else {
                const auto slot = static_cast<size_t>(resource.type);
                if (slot >= impl_->userTexturePool.size()) { return makeError(Error::InvalidArgument); }
                texture = impl_->userTexturePool[slot];
            }
            const auto usage = output ? TextureUsageBits::Storage : TextureUsageBits::Sampled;
            if (!texture.texture || !hasFlag(texture.texture->desc().usage, usage)) {
                return makeError(Error::InvalidArgument);
            }
            auto canonical = importTexture(texture, state);
            if (!canonical) { return makeError(canonical.error()); }
            textures.push_back(texture);
            // NRD samples in General. Storage descriptors may both load and
            // store; never reduce them to write-only from descriptor type alone.
            access.uses.push_back({*canonical, ResourceState::General,
                {PipelineStageBits::ComputeShader,
                    output ? AccessBits::ShaderRead | AccessBits::ShaderWrite : AccessBits::ShaderRead}, output});
        }
    }
    auto accessPlan = buildGraphAccessPlan(resources, accesses);
    if (!accessPlan) { return makeError(accessPlan.error()); }
    for (const auto& stage : dispatches) {
        auto result = impl_->pipeline(stage.pipelineIndex);
        if (!result) { return result; }
    }
    for (auto& owner : owners) {
        auto result = commands.retainResource(std::move(owner));
        if (!result) { return result; }
    }
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
    const auto recordBoundary = [&](size_t stage) -> Result<> {
        const auto& planned = accessPlan->passes[stage];
        auto recorded = recordGraphAccessBarriers(commands, planned, bindings);
        if (!recorded) { return recorded; }
        for (const auto& use : planned.uses) {
            for (auto* state : trackedStates[use.resource]) { *state = use.state; }
        }
        return {};
    };
    if (clearPending) {
        result = recordBoundary(0);
        if (!result) { return result; }
        for (const auto& use : accessPlan->passes.front().uses) {
            commands.clearColorTexture(*bindings[use.resource].texture, ResourceState::TransferDestination, {0, 0, 0, 0});
        }
        impl_->clearPending = false;
    }
    result = commands.retainResource(impl_);
    if (!result) { return result; }
    for (size_t stageIndex = 0; stageIndex < dispatches.size(); ++stageIndex) {
        const auto& stage = dispatches[stageIndex];
        result = recordBoundary(stageIndex + (clearPending ? 1u : 0u));
        if (!result) { return result; }
        commands.beginDebugLabel({.name = stage.name, .color = {0.2f, 0.8f, 0.25f, 1.0f}});
        result = dispatch(commands, stage, dispatchTextures[stageIndex]);
        commands.endDebugLabel();
        if (!result)
            return result;
    }
    recordingGuard.complete = true;
    return {};
}

Result<> NrdRuntime::dispatch(CommandBuffer& commands, const denoising::DispatchDesc& stage,
    std::span<const NrdTextureRef> textures)
{
    ParameterWriter writer(*impl_->device, *commands.frameContext(), *impl_->registry);
    NrdResourceIndices indices;
    for (uint32_t i = 0; i < 2; ++i) {
        const auto filter = i == 0 ? SamplerFilter::Nearest : SamplerFilter::Linear;
        indices.samplers[i] = static_cast<uint32_t>(writer.sampler({.minFilter = filter, .magFilter = filter,
            .mipFilter = SamplerFilter::Nearest, .addressU = SamplerAddressMode::ClampToEdge,
            .addressV = SamplerAddressMode::ClampToEdge, .addressW = SamplerAddressMode::ClampToEdge}).value);
    }
    uint32_t sampled = 0, storage = 0;
    for (uint32_t i = 0; i < stage.resourcesNum; ++i) {
        const auto& resource = stage.resources[i];
        const auto& texture = textures[i];
        const bool output = resource.descriptorType == denoising::DescriptorType::STORAGE_TEXTURE;
        if (output)
            indices.storage[storage++] = static_cast<uint32_t>(writer.storageImage(texture.view).value);
        else
            indices.sampled[sampled++] = static_cast<uint32_t>(writer.sampledImage(texture.view, ResourceState::General).value);
    }
    if (!writer.status()) { return writer.status(); }
    const NrdPushData params{stage.constantBufferDataSize ? writer.data(stage.constantBufferData, stage.constantBufferDataSize) : 0,
        writer.data(&indices, sizeof(indices))};
    EncodedParameters encoded;
    auto result = writer.encode(params, kNrdAbi, encoded);
    if (!result) { return result; }
    return impl_->pipelines[stage.pipelineIndex].dispatch(commands, encoded, stage.gridWidth, stage.gridHeight);
}
#endif
} // namespace metallic::render

#include "Runtime/Render/Shadow/VirtualShadowMapTemporal.h"

#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <algorithm>
#include <array>
#include <tuple>

namespace metallic::render {
namespace {
constexpr uint64_t kTemporalABI = 0x56534d54454d5031ull;

bool sameProjection(const ViewConstants& a, const ViewConstants& b)
{
    return a.current.upProjection[3] == b.current.upProjection[3] &&
        std::equal(std::begin(a.current.viewport), std::end(a.current.viewport), std::begin(b.current.viewport)) &&
        std::equal(std::begin(a.current.clipOrtho), std::end(a.current.clipOrtho), std::begin(b.current.clipOrtho)) &&
        a.frame[2] == b.frame[2] && a.frame[3] == b.frame[3];
}

bool sameLight(const VirtualShadowMapLight& a, const VirtualShadowMapLight& b)
{
    return a.type == b.type && a.position == b.position && a.direction == b.direction &&
        a.outerConeAngle == b.outerConeAngle && a.range == b.range && a.nearPlane == b.nearPlane &&
        a.angularRadius == b.angularRadius && a.sourceRadius == b.sourceRadius && a.enabled == b.enabled;
}

bool sameSettings(const VirtualShadowMapSettings& a, const VirtualShadowMapSettings& b)
{
    const auto key = [](const auto& s) {
        return std::tuple{s.pageSize, s.virtualPagesPerAxis, s.clipmapCount, s.physicalPageCount,
            s.maxWorkItems, s.firstClipmapExtent, s.shadowBias, s.filter, s.pcfRadius,
            s.smrtRayCount, s.smrtSamplesPerRay, s.smrtMaxRayLength, s.smrtRayLengthScale,
            s.smrtMaxRayAngle, s.receiverFootprint, s.enabled,
            s.shadowLOD, s.shadowLODSmartBias, s.shadowLODSmartBiasRefine, s.shadowLODScreenRayLength};
    };
    // Cache redraw and sampling phase do not change the underlying visibility.
    return key(a) == key(b);
}
} // namespace

struct VirtualShadowMapTemporal::Impl
{
    struct Resource
    {
        std::unique_ptr<Texture> texture;
        std::unique_ptr<TextureView> view;
        GPUCompletionPoint completion;
        TextureLayout layout = TextureLayout::Undefined;
        uint32_t width = 0, height = 0;
    };
    struct Snapshot
    {
        std::shared_ptr<Resource> resource;
        ViewConstants view;
        VirtualShadowMapLight light;
        VirtualShadowMapSettings settings;
        uint64_t frameIndex = 0, invalidationRevision = 0, contentKey = 0;
        uint32_t maxFrames = 0;
        GPUCompletionPoint completion;
        std::shared_ptr<SubmissionTransaction> submission;
    };
    Device* device = nullptr;
    ComputeKernel kernel;
    std::array<std::shared_ptr<Resource>, 3> pool;
    // No snapshot owns another snapshot: repeated frames cannot grow a chain.
    std::shared_ptr<Snapshot> committed, predecessor, pending;

    Result<std::shared_ptr<Resource>> acquire(uint32_t width, uint32_t height, std::string& log)
    {
        for (auto& resource : pool) {
            if (resource && (resource.use_count() != 1 || !resource->completion.isComplete())) { continue; }
            if (resource && resource->width == width && resource->height == height) { return resource; }
            auto next = std::make_shared<Resource>();
            auto texture = device->createTexture({.usage = TextureUsageBits::Storage,
                .format = Format::RG32Uint, .width = width, .height = height});
            if (!texture) { return std::unexpected(texture.error()); }
            auto view = device->createTextureView(**texture, {.format = Format::RG32Uint});
            if (!view) { return std::unexpected(view.error()); }
            next->texture = std::move(*texture); next->view = std::move(*view);
            next->width = width; next->height = height;
            resource = std::move(next);
            return resource;
        }
        log += "VSM temporal recording requires completed prior uses before reusing its three history slots.\n";
        return makeError(Error::InvalidArgument);
    }
};

VirtualShadowMapTemporal::VirtualShadowMapTemporal() = default;
VirtualShadowMapTemporal::~VirtualShadowMapTemporal() = default;
VirtualShadowMapTemporal::VirtualShadowMapTemporal(VirtualShadowMapTemporal&&) noexcept = default;
VirtualShadowMapTemporal& VirtualShadowMapTemporal::operator=(VirtualShadowMapTemporal&&) noexcept = default;

uint64_t VirtualShadowMapTemporal::memoryBytes() const
{
    uint64_t bytes = 0;
    if (impl_) {
        for (const auto& resource : impl_->pool) {
            if (resource) { bytes += resource->texture->allocationSize(); }
        }
    }
    return bytes;
}

Result<> VirtualShadowMapTemporal::record(Device& device, CommandBuffer& commands, TextureView& depth,
    TextureView& shadows, const ViewConstants& view, uint64_t frameIndex,
    uint64_t invalidationRevision, uint64_t contentKey, const VirtualShadowMapLight& light,
    const VirtualShadowMapSettings& settings, uint32_t maxFrames, uint32_t baseLayer,
    uint32_t faceCount, bool compact, std::string& log)
{
    auto* frame = RenderFrameContext::from(commands);
    if (!frame || !frame->recording() || maxFrames == 0 || maxFrames > 64 ||
        faceCount != (light.type == VirtualShadowMapLightType::PointFace ? 6u : 1u)) {
        log += "VSM temporal history requires a recording frame, 1..64 accumulation frames and the light's face count.\n";
        return makeError(Error::InvalidArgument);
    }
    if (!impl_ || impl_->device != &device) {
        impl_ = std::make_unique<Impl>();
        impl_->device = &device;
    }
    auto& state = *impl_;
    if (state.pending) {
        if (state.pending->submission->cancelled() || state.pending->completion.isCancelled()) {
            state.pending.reset();
        } else if (state.pending->submission->resolved() && state.pending->completion.isSubmitted()) {
            if (!state.committed || state.pending->frameIndex != state.committed->frameIndex) {
                state.predecessor = std::move(state.committed);
            }
            state.committed = std::move(state.pending);
        } else {
            log += "Previous VSM temporal recording must be submitted or cancelled before recording again.\n";
            return makeError(Error::InvalidArgument);
        }
    }
    if (!state.kernel.valid()) {
        auto result = ShaderRegistry::instance().getComputeKernel(device,
            {.moduleName = "Features/VirtualShadowMaps/VirtualShadowMapTemporal",
                .entryPointName = "vsmTemporalMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<VirtualShadowMapTemporalParams>(kTemporalABI, ParameterTransport::InlinePush),
                .debugName = "VSM temporal visibility"}, state.kernel, log);
        if (!result) { return result; }
    }
    auto previous = state.committed && state.committed->frameIndex == frameIndex ? state.predecessor : state.committed;
    const bool hasHistory = previous && previous->frameIndex + 1 == frameIndex &&
        previous->invalidationRevision == invalidationRevision && previous->contentKey == contentKey &&
        previous->maxFrames == maxFrames && sameProjection(previous->view, view) &&
        sameLight(previous->light, light) && sameSettings(previous->settings, settings);
    if (hasHistory) {
        if (auto result = addCommandDependency(commands, previous->completion); !result) { return result; }
    }
    const uint32_t width = uint32_t(view.current.viewport[1]), height = uint32_t(view.current.viewport[2]);
    auto acquired = state.acquire(width, height, log);
    if (!acquired) { return std::unexpected(acquired.error()); }
    const auto current = *acquired;
    const auto old = hasHistory ? previous->resource : current;
    // Retain the wrappers as well as their descriptor allocations. The pool
    // must not reuse a slot while a frame still owns either access to it.
    frame->retain(current);
    if (hasHistory) { frame->retain(old); }
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    ParameterWriter writer(device, *frame, **registry);
    const std::array<ViewConstants, 2> views{view, hasHistory ? previous->view : view};
    VirtualShadowMapTemporalParams params{};
    params.views = writer.dataSpan(views.data(), sizeof(views), sizeof(ViewConstants), alignof(ViewConstants));
    params.depth = writer.sampledImage(&depth);
    params.shadows = writer.storageImage(&shadows);
    params.previousHistory = writer.storageImage(old->view.get());
    params.currentHistory = writer.storageImage(current->view.get());
    params.width = width; params.height = height; params.baseLayer = baseLayer; params.faceCount = faceCount;
    params.compact = compact ? 1u : 0u; params.lightType = uint32_t(light.type);
    params.maxFrames = maxFrames; params.hasHistory = hasHistory && maxFrames > 1 ? 1u : 0u;
    params.lightX = light.position[0]; params.lightY = light.position[1]; params.lightZ = light.position[2];
    auto encoded = writer.encode(params, kTemporalABI, ParameterTransport::InlinePush);
    if (!encoded) { return std::unexpected(encoded.error()); }

    auto next = std::make_shared<Impl::Snapshot>();
    next->resource = current; next->view = view; next->light = light; next->settings = settings;
    next->frameIndex = frameIndex; next->invalidationRevision = invalidationRevision;
    next->contentKey = contentKey; next->maxFrames = maxFrames; next->completion = frame->completion();
    next->submission = std::make_shared<SubmissionTransaction>(nullptr,
        [resource = std::weak_ptr<Impl::Resource>(current)] {
            // Cancellation can include an accepted prefix. Discard this slot's
            // contents on its next reuse; completion still protects GPU users.
            if (auto cancelled = resource.lock()) { cancelled->layout = TextureLayout::Undefined; }
        });
    if (auto result = commands.addSubmissionTransaction(next->submission); !result) { return result; }
    std::array<TextureBarrierDesc, 2> barriers{};
    barriers[0] = {.texture = current->texture.get(), .oldLayout = current->layout,
        .newLayout = TextureLayout::General,
        .before = current->layout == TextureLayout::Undefined ? SyncScope{} :
            SyncScope{PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite},
        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
    uint32_t barrierCount = 1;
    if (hasHistory) {
        barriers[1] = {.texture = old->texture.get(), .oldLayout = TextureLayout::General,
            .newLayout = TextureLayout::General,
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
        barrierCount = 2;
        old->completion = frame->completion();
    }
    if (auto result = commands.synchronize({.textures = {barriers.data(), barrierCount}}); !result) { return result; }
    current->layout = TextureLayout::General;
    current->completion = frame->completion();
    if (auto result = state.kernel.dispatch(commands, *encoded, (width + 7u) / 8u, (height + 7u) / 8u); !result) { return result; }
    // The public whole-array view is shared by successive light resolves.
    const MemoryBarrierDesc ready{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
    if (auto result = commands.synchronize({.memory = {&ready, 1}}); !result) { return result; }
    state.pending = std::move(next);
    return {};
}
} // namespace metallic::render

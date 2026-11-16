#include "Runtime/Render/Shadow/VirtualShadowMapTemporal.h"

#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraphTypes.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

namespace metallic::render {
namespace {
constexpr uint64_t kTemporalABI = 0x56534d54454d5033ull;

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
            s.shadowLOD, s.shadowLODSmartBias, s.shadowLODScreenRayLength};
    };
    // Cache redraw and sampling phase do not change the underlying visibility.
    return key(a) == key(b);
}

VSMTemporalViewBasis makeTemporalViewBasis(const ViewConstants& view)
{
    auto sub3 = [](const float (&a)[4], const float (&b)[4]) -> std::array<float, 3> {
        return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    };
    auto cross3 = [](const std::array<float, 3>& a, const std::array<float, 3>& b) -> std::array<float, 3> {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    };
    auto norm3 = [](const std::array<float, 3>& v) -> std::array<float, 3> {
        const float inv = 1.0f / std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return {v[0] * inv, v[1] * inv, v[2] * inv};
    };
    const std::array<float, 3> forward = norm3(sub3(view.current.center, view.current.eye));
    const std::array<float, 3> right = norm3(cross3(forward, {view.current.upProjection[0], view.current.upProjection[1], view.current.upProjection[2]}));
    const std::array<float, 3> up = cross3(right, forward);
    const float vpW = view.current.viewport[1], vpH = view.current.viewport[2];
    const float aspect = view.current.viewport[0];
    const float tanHalfFov = std::tan(view.current.viewport[3] * 0.5f);
    const bool isOrtho = view.current.upProjection[3] > 0.5f;
    const float halfOrthoHeight = view.current.clipOrtho[2] * 0.5f;
    const float nearPlane = view.current.clipOrtho[0], farPlane = view.current.clipOrtho[1];
    const float farMinusNear = farPlane - nearPlane;
    const bool reversed = view.current.clipOrtho[3] > 0.5f;
    const float jx = view.frame[2] != 0u ? view.jitter[0] : 0.0f;
    const float jy = view.frame[2] != 0u ? view.jitter[1] : 0.0f;
    const float ndcScaleX = 2.0f / vpW, ndcScaleY = -2.0f / vpH;
    const float ndcBiasX = std::fma(0.5f + jx, ndcScaleX, -1.0f);
    const float ndcBiasY = std::fma(0.5f + jy, ndcScaleY, 1.0f);
    VSMTemporalViewBasis b{};
    b.eye = {view.current.eye[0], view.current.eye[1], view.current.eye[2]};
    b.nearTimesFar = nearPlane * farPlane;
    b.forward = forward;
    b.right = right;
    b.up = up;
    b.clearDepth = reversed ? 0.0f : 1.0f;
    b.toleranceScale = isOrtho ? 1e-5f : ((2.0f * tanHalfFov / vpH) * 0.01f + 1e-5f);
    b.orthoTolerance = isOrtho ? (halfOrthoHeight * 2.0f / vpH * 0.01f) : 0.0f;
    b.nearPlane = nearPlane;
    b.viewportExtent = {vpW, vpH};
    b.jitter = {jx, jy};
    b.farPlane = farPlane;
    b.isOrtho = isOrtho ? 1u : 0u;
    const std::array<float, 3> rightAspect{right[0] * aspect, right[1] * aspect, right[2] * aspect};
    if (isOrtho) {
        b.depthParamA = reversed ? -farMinusNear : farMinusNear;
        b.depthParamB = reversed ? farPlane : nearPlane;
        const float invHalfOrtho = 1.0f / halfOrthoHeight;
        b.projScaleX = invHalfOrtho / aspect;
        b.projScaleY = invHalfOrtho;
        for (size_t i = 0; i < 3; ++i) {
            b.originOrRayBase[i] = b.eye[i] + (halfOrthoHeight * ndcBiasX) * rightAspect[i] + (halfOrthoHeight * ndcBiasY) * up[i];
            b.dirX[i] = (halfOrthoHeight * ndcScaleX) * rightAspect[i];
            b.dirY[i] = (halfOrthoHeight * ndcScaleY) * up[i];
        }
    } else {
        b.depthParamA = reversed ? farMinusNear : -farMinusNear;
        b.depthParamB = reversed ? nearPlane : farPlane;
        b.projScaleX = 1.0f / (tanHalfFov * aspect);
        b.projScaleY = 1.0f / tanHalfFov;
        for (size_t i = 0; i < 3; ++i) {
            b.originOrRayBase[i] = forward[i] + (tanHalfFov * ndcBiasX) * rightAspect[i] + (tanHalfFov * ndcBiasY) * up[i];
            b.dirX[i] = (tanHalfFov * ndcScaleX) * rightAspect[i];
            b.dirY[i] = (tanHalfFov * ndcScaleY) * up[i];
        }
    }
    return b;
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
    std::array<ComputeKernel, 4> kernels;
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
    uint32_t faceCount, bool compact, std::string& log, RenderGraphExecutionContext* gpuProfiler)
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

    auto previous = state.committed && state.committed->frameIndex == frameIndex ? state.predecessor : state.committed;
    const bool hasHistory = previous && previous->frameIndex + 1 == frameIndex &&
        previous->invalidationRevision == invalidationRevision && previous->contentKey == contentKey &&
        previous->maxFrames == maxFrames && sameProjection(previous->view, view) &&
        sameLight(previous->light, light) && sameSettings(previous->settings, settings);
    const bool isOrtho = view.current.upProjection[3] > 0.5f;
    const bool useHistory = hasHistory && maxFrames > 1;
    const size_t kernelIndex = (isOrtho ? 2u : 0u) | (useHistory ? 1u : 0u);
    auto& kernel = state.kernels[kernelIndex];
    if (!kernel.valid()) {
        const std::array<ShaderSpecializationConstant, 2> specs{
            ShaderSpecializationConstant{0u, ShaderSpecializationType::Bool32, isOrtho ? 1u : 0u},
            ShaderSpecializationConstant{1u, ShaderSpecializationType::Bool32, useHistory ? 1u : 0u}};
        auto result = ShaderRegistry::instance().getComputeKernel(device,
            {.moduleName = "Features/VirtualShadowMaps/VirtualShadowMapTemporal",
                .entryPointName = "vsmTemporalMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
            {.parameters = parameterAbi<VirtualShadowMapTemporalParams>(kTemporalABI, ParameterTransport::InlinePush),
                .debugName = "VSM temporal visibility", .specializationConstants = specs}, kernel, log);
        if (!result) { return result; }
    }
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
    const std::array<VSMTemporalViewBasis, 2> views{makeTemporalViewBasis(view),
        makeTemporalViewBasis(hasHistory ? previous->view : view)};
    VirtualShadowMapTemporalParams params{};
    params.views = writer.dataSpan(views.data(), sizeof(views), sizeof(VSMTemporalViewBasis), alignof(VSMTemporalViewBasis));
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
    auto gpu = gpuProfiler ? gpuProfiler->profileScope(commands, "VSM temporal prepare") :
        RenderGraphExecutionContext::ProfileScope{};
    if (auto result = commands.synchronize({.textures = {barriers.data(), barrierCount}}); !result) { return result; }
    current->layout = TextureLayout::General;
    current->completion = frame->completion();
    gpu.next("VSM temporal dispatch");
    if (auto result = kernel.dispatch(commands, *encoded, (width + 7u) / 8u, (height + 7u) / 8u); !result) { return result; }
    gpu.next("VSM temporal publish");
    // The public whole-array view is shared by successive light resolves.
    const MemoryBarrierDesc ready{{PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
        {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
    if (auto result = commands.synchronize({.memory = {&ready, 1}}); !result) { return result; }
    gpu.end();
    state.pending = std::move(next);
    return {};
}
} // namespace metallic::render

#include "Runtime/Render/Shadow/VirtualShadowMaps.h"

#include "Runtime/Render/Shadow/VirtualShadowMapCasters.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <vector>

namespace metallic::render {
namespace {

constexpr uint32_t kDrawBatchSize = 65535;

struct ShadowPush
{
    uint32_t geometry, materials, pages, coverageInputs;
    uint32_t workItems, statistics, projections, atlas;
    uint32_t depth, shadow, debug, view;
    uint32_t arguments, triangleCount, clipmapCount, virtualPagesPerAxis;
    uint32_t physicalPagesX, physicalPagesY, physicalPageCount, pageSize;
    uint32_t workCapacity, batchCount, workBase, rasterCoarse;
    uint32_t sampler, enabled, totalPages, reserved;
};
static_assert(sizeof(ShadowPush) == 112);

struct ShadowProjection
{
    std::array<float, 4> rowX, rowY, rowZ, metric;
    std::array<uint32_t, 4> page;
};
static_assert(sizeof(ShadowProjection) == 80);

struct ShadowMaterial
{
    std::array<float, 4> alpha;
    std::array<uint32_t, 4> resources;
    std::array<float, 4> uvX, uvY;
};
static_assert(sizeof(ShadowMaterial) == 64);

using Vector = std::array<float, 3>;
float dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector cross(const Vector& a, const Vector& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Vector normalized(Vector value)
{
    const float length = std::sqrt(dot(value, value));
    for (auto& component : value) { component /= length; }
    return value;
}

std::vector<ShadowProjection> makeProjections(const scene::Bounds& bounds, const ViewConstants& view,
    Vector direction, const VirtualShadowMapSettings& settings)
{
    const auto z = normalized(direction);
    const auto x = normalized(cross(std::abs(z[1]) < 0.95f ? Vector{0, 1, 0} : Vector{1, 0, 0}, z));
    const auto y = cross(z, x);
    Vector low{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    Vector high{-low[0], -low[1], -low[2]};
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const Vector position = bounds.valid ? Vector{corner & 1 ? bounds.max.x : bounds.min.x,
            corner & 2 ? bounds.max.y : bounds.min.y, corner & 4 ? bounds.max.z : bounds.min.z} : Vector{0, 0, 0};
        const Vector light{dot(x, position), dot(y, position), dot(z, position)};
        for (size_t axis = 0; axis < 3; ++axis) {
            low[axis] = std::min(low[axis], light[axis]); high[axis] = std::max(high[axis], light[axis]);
        }
    }
    const float margin = std::max(0.01f, settings.shadowBias * 2);
    low[2] -= margin; high[2] += margin;
    const float depthRange = std::max(high[2] - low[2], 0.01f);
    const Vector eye{view.current.eye[0], view.current.eye[1], view.current.eye[2]};
    const Vector eyeLight{dot(x, eye), dot(y, eye), dot(z, eye)};
    std::vector<ShadowProjection> result;
    for (uint32_t level = 0; level <= settings.clipmapCount; ++level) {
        const bool coarse = level == settings.clipmapCount;
        const uint32_t axis = coarse ? 1 : settings.virtualPagesPerAxis;
        const float extent = coarse ? std::max({(high[0] - low[0]) * .5f, (high[1] - low[1]) * .5f, 0.01f}) *
            (1.0f + 4.0f / settings.pageSize) + margin : std::ldexp(settings.firstClipmapExtent, int(level));
        const float pageWorldSize = 2 * extent / axis;
        const float centerX = coarse ? (low[0] + high[0]) * .5f : std::floor(eyeLight[0] / pageWorldSize) * pageWorldSize;
        const float centerY = coarse ? (low[1] + high[1]) * .5f : std::floor(eyeLight[1] / pageWorldSize) * pageWorldSize;
        result.push_back({{x[0] / extent, x[1] / extent, x[2] / extent, -centerX / extent},
            {y[0] / extent, y[1] / extent, y[2] / extent, -centerY / extent},
            {-z[0] / depthRange, -z[1] / depthRange, -z[2] / depthRange, high[2] / depthRange},
            {extent, pageWorldSize / settings.pageSize, settings.shadowBias / depthRange, 0},
            {level * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis, axis, level, coarse ? 1u : 0u}});
    }
    return result;
}

} // namespace

struct VirtualShadowMaps::Programs
{
    Device* device = nullptr;
    uint64_t coverageKey = 0;
    std::array<std::unique_ptr<ShaderModule>, 8> shaders;
    std::array<std::unique_ptr<ComputePipeline>, 6> compute;
    std::unique_ptr<GraphicsPipeline> raster;
};

struct VirtualShadowMaps::State
{
    ShadowPush push{};
    std::shared_ptr<ResourceRegistry> registry;
    std::shared_ptr<Programs> programs;
    std::vector<ResourceLease> leases;
    // materials, page table, coverage, work, statistics, projections, view, indirect.
    std::array<std::unique_ptr<Buffer>, 8> buffers;
    std::array<std::unique_ptr<Texture>, 3> textures; // atlas, shadow, debug
    std::array<std::unique_ptr<TextureView>, 3> views;
};

VirtualShadowMaps::VirtualShadowMaps() = default;
VirtualShadowMaps::~VirtualShadowMaps() = default;

Result<VirtualShadowMapResult> VirtualShadowMaps::record(Device& device, CommandBuffer& commands,
    TextureView& depth, const ViewConstants& view, const GPUCelestialLight& sun,
    const VirtualShadowMapCasterResources& casters, std::span<const scene::RenderMaterial> materials,
    std::span<TextureView* const> alphaTextures, const VirtualShadowMapSettings& settings, std::string& log)
{
    if (!device.capabilities().meshShader) { return makeError(Error::Unsupported); }
    // Scene geometry uses material slot zero when no material was authored,
    // matching the renderer's ordinary default-material publication.
    static const scene::RenderMaterial defaultMaterial;
    if (materials.empty()) { materials = {&defaultMaterial, 1}; }
    if (settings.pageSize == 0 || settings.pageSize > 128 || (settings.pageSize & (settings.pageSize - 1)) ||
        settings.virtualPagesPerAxis == 0 || settings.virtualPagesPerAxis > 128 || settings.clipmapCount == 0 ||
        settings.clipmapCount > 12 || settings.physicalPageCount == 0 || settings.physicalPageCount > 4096 ||
        settings.maxWorkItems == 0 || settings.maxWorkItems > 4 * 1024 * 1024 ||
        !std::isfinite(settings.firstClipmapExtent) || settings.firstClipmapExtent <= 0 ||
        !std::isfinite(settings.shadowBias) || settings.shadowBias < 0 ||
        (!alphaTextures.empty() && alphaTextures.size() != materials.size()) ||
        view.current.viewport[1] < 1 || view.current.viewport[2] < 1 ||
        (view.current.clipOrtho[3] != 0 && view.current.clipOrtho[3] != 1)) {
        log += "Invalid VSM P1 settings, material bindings or receiver depth convention.\n";
        return makeError(Error::InvalidArgument);
    }
    const bool enabled = settings.enabled && (sun.flags & (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow)) ==
        (kGPUCelestialLightEnabled | kGPUCelestialLightCastsShadow);
    Vector direction = enabled ? Vector{sun.direction[0], sun.direction[1], sun.direction[2]} : Vector{0, -1, 0};
    if (!std::all_of(direction.begin(), direction.end(), [](float v) { return std::isfinite(v); }) || dot(direction, direction) < 1e-12f) {
        return makeError(Error::InvalidArgument);
    }
    for (size_t index = 0; index < materials.size(); ++index) {
        const auto& material = materials[index];
        if ((material.alphaMode != "OPAQUE" && material.alphaMode != "MASK") || material.displacementMagnitude != 0 ||
            (!alphaTextures.empty() && alphaTextures[index] && material.baseColorTexture.texCoord != 0)) {
            log += "VSM P1 supports opaque/MASK, UV0 alpha textures and undisplaced caster geometry.\n";
            return makeError(Error::Unsupported);
        }
    }
    const auto coverage = MaterialValueProgramSet::create(materials, log);
    if (!coverage) { return makeError(Error::InvalidArgument); }
    auto state = std::make_shared<State>();
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    state->registry = *registry;
    auto& push = state->push;
    push.triangleCount = casters.triangleCapacity;
    push.clipmapCount = settings.clipmapCount;
    push.virtualPagesPerAxis = settings.virtualPagesPerAxis;
    push.physicalPagesX = static_cast<uint32_t>(std::ceil(std::sqrt(float(settings.physicalPageCount))));
    push.physicalPagesY = (settings.physicalPageCount + push.physicalPagesX - 1) / push.physicalPagesX;
    push.physicalPageCount = settings.physicalPageCount;
    push.pageSize = settings.pageSize;
    push.workCapacity = settings.maxWorkItems;
    push.batchCount = (settings.maxWorkItems + kDrawBatchSize - 1) / kDrawBatchSize;
    push.enabled = enabled;
    push.totalPages = settings.clipmapCount * settings.virtualPagesPerAxis * settings.virtualPagesPerAxis + 1;
    state->leases = casters.leases;
    auto geometry = state->registry->storageBuffer(*casters.parameters);
    if (!geometry) { return std::unexpected(geometry.error()); }
    push.geometry = geometry->shaderIndex(); state->leases.push_back(*geometry);
    const auto projections = makeProjections(casters.bounds, view, direction, settings);
    std::vector<ShadowMaterial> gpuMaterials;
    for (size_t index = 0; index < materials.size(); ++index) {
        const auto& material = materials[index];
        uint32_t alpha = UINT32_MAX;
        if (!alphaTextures.empty() && alphaTextures[index]) {
            auto lease = state->registry->sampledImage(*alphaTextures[index]);
            if (!lease) { return std::unexpected(lease.error()); }
            alpha = lease->shaderIndex(); state->leases.push_back(*lease);
        }
        const auto& transform = material.baseColorTexture.uvTransform;
        gpuMaterials.push_back({{material.baseColorFactor.w, material.alphaCutoff, material.alphaMode == "MASK" ? 1.f : 0.f, 0},
            {alpha, material.doubleSided ? 1u : 0u, 0, 0},
            {transform[0], transform[1], transform[2], 0}, {transform[3], transform[4], transform[5], 0}});
    }
    const uint64_t sizes[]{gpuMaterials.size() * sizeof(ShadowMaterial), uint64_t(push.totalPages) * 16,
        coverage->inputBytes().size(), uint64_t(push.workCapacity) * 8, 64,
        projections.size() * sizeof(ShadowProjection), sizeof(ViewConstants), uint64_t(push.batchCount) * 12};
    const uint32_t strides[]{64, 16, 4, 8, 4, 80, 208, 4};
    const void* data[]{gpuMaterials.data(), nullptr, coverage->inputBytes().data(), nullptr, nullptr, projections.data(), &view, nullptr};
    uint32_t* indices[]{&push.materials, &push.pages, &push.coverageInputs, &push.workItems,
        &push.statistics, &push.projections, &push.view, &push.arguments};
    for (size_t index = 0; index < state->buffers.size(); ++index) {
        auto buffer = device.createBuffer({.size = sizes[index], .structureStride = strides[index],
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (index == 7 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .memoryLocation = data[index] ? MemoryLocation::HostUpload : MemoryLocation::Device});
        if (!buffer) { return std::unexpected(buffer.error()); }
        state->buffers[index] = std::move(*buffer);
        if (data[index]) {
            void* mapped = state->buffers[index]->map();
            if (!mapped) { return makeError(Error::Failure); }
            std::memcpy(mapped, data[index], sizes[index]);
            state->buffers[index]->flush(); state->buffers[index]->unmap();
        }
        auto lease = state->registry->storageBuffer(*state->buffers[index]);
        if (!lease) { return std::unexpected(lease.error()); }
        *indices[index] = lease->shaderIndex(); state->leases.push_back(*lease);
    }
    const uint32_t width = static_cast<uint32_t>(view.current.viewport[1]), height = static_cast<uint32_t>(view.current.viewport[2]);
    const Format formats[]{Format::D32Sfloat, Format::R8Unorm, Format::RGBA8Unorm};
    uint32_t* textureIndices[]{&push.atlas, &push.shadow, &push.debug};
    for (size_t index = 0; index < 3; ++index) {
        auto texture = device.createTexture({.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferSource |
            (index == 0 ? TextureUsageBits::DepthStencilAttachment : TextureUsageBits::Storage),
            .format = formats[index], .width = index == 0 ? push.physicalPagesX * push.pageSize : width,
            .height = index == 0 ? push.physicalPagesY * push.pageSize : height});
        if (!texture) { return std::unexpected(texture.error()); }
        state->textures[index] = std::move(*texture);
        auto textureView = device.createTextureView(*state->textures[index], {.format = formats[index]});
        if (!textureView) { return std::unexpected(textureView.error()); }
        state->views[index] = std::move(*textureView);
        auto lease = index == 0 ? state->registry->sampledImage(*state->views[index]) : state->registry->storageImage(*state->views[index]);
        if (!lease) { return std::unexpected(lease.error()); }
        *textureIndices[index] = lease->shaderIndex(); state->leases.push_back(*lease);
    }
    auto depthLease = state->registry->sampledImage(depth);
    if (!depthLease) { return std::unexpected(depthLease.error()); }
    push.depth = depthLease->shaderIndex(); state->leases.push_back(*depthLease);
    auto sampler = state->registry->sampler({});
    if (!sampler) { return std::unexpected(sampler.error()); }
    push.sampler = sampler->shaderIndex(); state->leases.push_back(*sampler);
    if (!programs_ || programs_->device != &device || programs_->coverageKey != coverage->coverageKey()) {
        std::filesystem::path generatedDirectory;
        if (!coverage->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", generatedDirectory, log)) {
            return makeError(Error::Failure);
        }
        auto programs = std::make_shared<Programs>();
        programs->device = &device;
        programs->coverageKey = coverage->coverageKey();
        const char* entries[]{"vsmResetMain", "vsmMarkMain", "vsmAllocateMain", "vsmBuildMain", "vsmFinalizeMain", "vsmProjectMain",
            "vsmRasterMeshMain", "vsmDepthFragmentMain"};
        const auto generated = generatedDirectory.string();
        const std::array paths{generated.c_str()};
        const std::array capabilities{"spvMeshShadingEXT"};
        const std::array macros{SlangMacroDefine{"METALLIC_CUSTOM_COVERAGE", coverage->coverageProgramCount() ? "1" : "0"}};
        for (size_t index = 0; index < programs->shaders.size(); ++index) {
            auto shader = ShaderRegistry::instance().getShader({.moduleName = "Features/VirtualShadowMaps/VirtualShadowMaps",
                .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders", .additionalSearchPaths = paths,
                .capabilities = capabilities, .macroDefines = macros}, log);
            if (!shader) { return std::unexpected(shader.error()); }
            auto module = ShaderRegistry::instance().getShaderModule(device, {.spirv = shader->spirv, .debugName = entries[index]});
            if (!module) { return std::unexpected(module.error()); }
            programs->shaders[index] = std::move(*module);
            if (index < 6) {
                auto pipeline = ShaderRegistry::instance().getComputePipeline(device, {.computeShader = {programs->shaders[index].get()},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(ShadowPush)});
                if (!pipeline) { return std::unexpected(pipeline.error()); }
                programs->compute[index] = std::move(*pipeline);
            }
        }
        auto raster = ShaderRegistry::instance().getGraphicsPipeline(device, {.meshShader = {programs->shaders[6].get()},
            .fragmentShader = {programs->shaders[7].get()}, .depthStencilFormat = Format::D32Sfloat,
            .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::GreaterEqual}, .usesBindlessHeap = true});
        if (!raster) { return std::unexpected(raster.error()); }
        programs->raster = std::move(*raster);
        programs_ = std::move(programs);
    }
    // Every submitted frame keeps its exact pipelines alive across a Coverage
    // edit. Only immutable programs are reused; shadow resources remain fresh.
    state->programs = programs_;
    if (auto result = commands.retainResource(state); !result) { return std::unexpected(result.error()); }
    if (auto result = state->registry->bind(commands, state->leases); !result) { return std::unexpected(result.error()); }
    auto dispatch = [&](size_t pipeline, uint32_t x, uint32_t y = 1) -> Result<> {
        if (auto result = commands.bindExecution(state->programs->compute[pipeline]->execution(), &push, sizeof(push)); !result) { return result; }
        return commands.dispatch(x, y);
    };
    auto barrier = [&](std::initializer_list<size_t> buffers, PipelineStageBits stages = PipelineStageBits::ComputeShader,
        AccessBits access = AccessBits::ShaderRead | AccessBits::ShaderWrite) -> Result<> {
        std::vector<BufferBarrierDesc> barriers;
        for (auto index : buffers) {
            barriers.push_back({.buffer = state->buffers[index].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
                .after = {stages, access}});
        }
        return commands.synchronize({.buffers = barriers});
    };
    commands.beginDebugLabel({.name = "VSM P1: demand and allocation"});
    if (auto r = dispatch(0, (std::max({push.totalPages, 16u, push.batchCount * 3}) + 63) / 64); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({1, 4, 7}); !r) { return std::unexpected(r.error()); }
    if (auto r = dispatch(1, (width + 7) / 8, (height + 7) / 8); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({1}); !r) { return std::unexpected(r.error()); }
    if (auto r = dispatch(2, 1); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({1, 4}); !r) { return std::unexpected(r.error()); }
    if (push.triangleCount && enabled) {
        // Compute dispatch limit is 65535 workgroups; split immutable triangle ranges.
        for (uint32_t base = 0; base < push.triangleCount;) {
            push.workBase = base;
            const uint32_t count = std::min(push.triangleCount - base, kDrawBatchSize * 64);
            if (auto r = dispatch(3, (count + 63) / 64); !r) { return std::unexpected(r.error()); }
            base += count;
            if (base < push.triangleCount) {
                if (auto r = barrier({3, 4}); !r) { return std::unexpected(r.error()); }
            }
        }
    }
    if (auto r = barrier({4}); !r) { return std::unexpected(r.error()); }
    if (auto r = dispatch(4, (push.batchCount + 63) / 64); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({3, 1}, PipelineStageBits::PreRasterization, AccessBits::ShaderRead); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({7}, PipelineStageBits::DrawIndirect, AccessBits::IndirectRead); !r) { return std::unexpected(r.error()); }
    if (auto r = barrier({4}); !r) { return std::unexpected(r.error()); }
    commands.endDebugLabel();
    const TextureBarrierDesc depthBegin{.texture = state->textures[0].get(), .oldLayout = TextureLayout::Undefined,
        .newLayout = TextureLayout::DepthStencilAttachment,
        .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite}};
    if (auto r = commands.synchronize({.textures = {&depthBegin, 1}}); !r) { return std::unexpected(r.error()); }
    const Rect area{.width = push.physicalPagesX * push.pageSize, .height = push.physicalPagesY * push.pageSize};
    const RenderingAttachmentDesc attachment{.view = state->views[0].get(), .layout = TextureLayout::DepthStencilAttachment,
        .loadOp = LoadOp::Clear, .storeOp = StoreOp::Store, .clearDepth = 0};
    commands.beginDebugLabel({.name = "VSM P1: coarse and fine depth"});
    if (auto r = commands.beginRendering({.renderArea = area, .depthStencilAttachment = &attachment}); !r) { return std::unexpected(r.error()); }
    if (auto r = commands.setViewport({.width = float(area.width), .height = float(area.height), .maxDepth = 1}); !r) { return std::unexpected(r.error()); }
    commands.setScissor(area);
    if (enabled) {
        push.rasterCoarse = 1;
        for (uint32_t base = 0; base < push.triangleCount; base += std::min(push.triangleCount - base, kDrawBatchSize)) {
            push.workBase = base;
            if (auto r = commands.bindExecution(state->programs->raster->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
            if (auto r = commands.drawMeshTasks(std::min(push.triangleCount - base, kDrawBatchSize)); !r) { return std::unexpected(r.error()); }
        }
        push.rasterCoarse = 0;
        for (uint32_t batch = 0; batch < push.batchCount; ++batch) {
            push.workBase = batch * kDrawBatchSize;
            if (auto r = commands.bindExecution(state->programs->raster->execution(), &push, sizeof(push)); !r) { return std::unexpected(r.error()); }
            auto args = state->buffers[7]->slice({uint64_t(batch) * 12, 12});
            if (!args) { return std::unexpected(args.error()); }
            if (auto r = commands.drawMeshTasksIndirect(*args); !r) { return std::unexpected(r.error()); }
        }
    }
    commands.endRendering(); commands.endDebugLabel();
    const TextureBarrierDesc outputBegin[]{
        {.texture = state->textures[0].get(), .oldLayout = TextureLayout::DepthStencilAttachment, .newLayout = TextureLayout::ShaderRead,
            .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite}, .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}},
        {.texture = state->textures[1].get(), .oldLayout = TextureLayout::Undefined, .newLayout = TextureLayout::General,
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}},
        {.texture = state->textures[2].get(), .oldLayout = TextureLayout::Undefined, .newLayout = TextureLayout::General,
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}}};
    if (auto r = commands.synchronize({.textures = outputBegin}); !r) { return std::unexpected(r.error()); }
    commands.beginDebugLabel({.name = "VSM P1: receiver projection"});
    if (auto r = dispatch(5, (width + 7) / 8, (height + 7) / 8); !r) { return std::unexpected(r.error()); }
    commands.endDebugLabel();
    const TextureBarrierDesc outputEnd[]{
        {.texture = state->textures[1].get(), .oldLayout = TextureLayout::General, .newLayout = TextureLayout::ShaderRead,
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}},
        {.texture = state->textures[2].get(), .oldLayout = TextureLayout::General, .newLayout = TextureLayout::ShaderRead,
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead}}};
    const BufferBarrierDesc statisticsReady{.buffer = state->buffers[4].get(),
        .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}, .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead}};
    if (auto r = commands.synchronize({.textures = outputEnd, .buffers = {&statisticsReady, 1}}); !r) { return std::unexpected(r.error()); }
    state_ = state;
    return VirtualShadowMapResult{state->textures[1].get(), state->views[1].get(), state->textures[2].get(), state->views[2].get(),
        state->buffers[4].get(), state->textures[0].get(), state->views[0].get(), state->buffers[1].get(), state->buffers[5].get()};
}

} // namespace metallic::render

#include "Runtime/Render/Shadow/VirtualShadowMapPrototype.h"

#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

namespace metallic::render {
namespace {

struct PrototypePush
{
    uint32_t triangles, materials, pages, coverageInputs;
    uint32_t workItems, arguments, projection, atlas;
    uint32_t receivers, queryResults, triangleCount, receiverCount;
    uint32_t virtualPagesX, virtualPagesY, physicalPagesX, physicalPagesY;
    uint32_t pageSize, workCapacity, sampler, reserved;
};
static_assert(sizeof(PrototypePush) == 80);

struct PrototypeMaterial
{
    std::array<float, 4> alpha;
    std::array<uint32_t, 4> resources;
};
static_assert(sizeof(PrototypeMaterial) == 32);

Result<> upload(Buffer& buffer, const void* data, size_t size)
{
    void* mapped = buffer.map();
    if (!mapped) { return makeError(Error::Failure); }
    if (size != 0) { std::memcpy(mapped, data, size); }
    buffer.flush();
    buffer.unmap();
    return {};
}

bool validProjection(const VirtualShadowMapPrototypeProjection& projection)
{
    const auto& rows = projection.rows;
    return std::all_of(rows.begin(), rows.end(), [](float value) { return std::isfinite(value); }) &&
        rows[12] == 0 && rows[13] == 0 && rows[14] == 0 && rows[15] == 1;
}

bool validTriangles(std::span<const VirtualShadowMapPrototypeTriangle> triangles, uint32_t materialCount)
{
    for (const auto& triangle : triangles) {
        if (triangle.materialIndex >= materialCount) { return false; }
        for (const auto& vertex : triangle.vertices) {
            if (vertex.position[3] != 1 ||
                !std::all_of(vertex.position.begin(), vertex.position.end(), [](float v) { return std::isfinite(v); }) ||
                !std::all_of(vertex.uv.begin(), vertex.uv.end(), [](float v) { return std::isfinite(v); })) { return false; }
        }
    }
    return true;
}

bool validPages(const VirtualShadowMapPrototypeDesc& desc, std::span<const VirtualShadowMapPrototypePage> pages)
{
    if (pages.size() != uint64_t(desc.virtualPagesX) * desc.virtualPagesY) { return false; }
    std::vector<bool> owned(size_t(desc.physicalPagesX) * desc.physicalPagesY);
    for (const auto& page : pages) {
        if (page.physicalPage == UINT32_MAX) { continue; }
        if (page.physicalPage >= owned.size() || owned[page.physicalPage] || page.dirty > 1) { return false; }
        owned[page.physicalPage] = true;
    }
    return true;
}

} // namespace

struct VirtualShadowMapPrototype::Impl
{
    VirtualShadowMapPrototypeDesc desc;
    uint32_t materialCount = 0;
    PrototypePush push{};
    // triangles, materials, pages, coverage, work items, indirect args,
    // projection rows, receiver positions, query results.
    std::array<std::unique_ptr<Buffer>, 9> buffers;
    std::unique_ptr<Texture> atlas;
    std::unique_ptr<TextureView> atlasView;
    std::shared_ptr<ResourceRegistry> registry;
    std::vector<ResourceLease> leases;
    std::array<std::unique_ptr<ShaderModule>, 7> shaders;
    std::array<std::unique_ptr<ComputePipeline>, 4> compute;
    std::array<std::unique_ptr<GraphicsPipeline>, 2> graphics;
};

VirtualShadowMapPrototype::VirtualShadowMapPrototype() = default;
VirtualShadowMapPrototype::~VirtualShadowMapPrototype() = default;

Result<> VirtualShadowMapPrototype::initialize(Device& device, const VirtualShadowMapPrototypeDesc& desc,
    const VirtualShadowMapPrototypeScene& scene, std::string& log)
{
    if (!device.capabilities().meshShader) { return makeError(Error::Unsupported); }
    // P0 intentionally bounds the brute-force triangle/page candidate producer.
    if (desc.pageSize == 0 || desc.pageSize > 128 || (desc.pageSize & (desc.pageSize - 1)) != 0 ||
        desc.virtualPagesX == 0 || desc.virtualPagesX > 16 || desc.virtualPagesY == 0 || desc.virtualPagesY > 16 ||
        desc.physicalPagesX == 0 || desc.physicalPagesX > 16 || desc.physicalPagesY == 0 || desc.physicalPagesY > 16 ||
        desc.maxWorkItems == 0 || desc.maxWorkItems > 65535 || scene.triangles.size() > 65535 ||
        scene.receivers.size() > 65535u * 64u || scene.materials.empty() || scene.materials.size() > UINT32_MAX ||
        (!scene.alphaTextures.empty() && scene.alphaTextures.size() != scene.materials.size()) ||
        !validPages(desc, scene.pages) || !validProjection(scene.projection) ||
        !validTriangles(scene.triangles, static_cast<uint32_t>(scene.materials.size()))) {
        return makeError(Error::InvalidArgument);
    }
    for (const auto& receiver : scene.receivers) {
        if (receiver.positionBias[3] < 0 || !std::all_of(receiver.positionBias.begin(), receiver.positionBias.end(),
            [](float value) { return std::isfinite(value); })) { return makeError(Error::InvalidArgument); }
    }
    for (const auto& material : scene.materials) {
        if ((material.alphaMode != "OPAQUE" && material.alphaMode != "MASK") || material.displacementMagnitude != 0) {
            log += "VSM P0 accepts opaque/MASK geometry with displacement already applied to input vertices.\n";
            return makeError(Error::Unsupported);
        }
    }
    const auto coverage = MaterialValueProgramSet::create(scene.materials, log);
    if (!coverage) { return makeError(Error::InvalidArgument); }
    std::filesystem::path generatedDirectory;
    if (!coverage->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", generatedDirectory, log)) {
        return makeError(Error::Failure);
    }

    auto state = std::make_unique<Impl>();
    state->desc = desc;
    state->materialCount = static_cast<uint32_t>(scene.materials.size());
    auto registry = ResourceRegistry::forDevice(device);
    if (!registry) { return std::unexpected(registry.error()); }
    state->registry = *registry;
    auto& push = state->push;
    push.triangleCount = static_cast<uint32_t>(scene.triangles.size());
    push.receiverCount = static_cast<uint32_t>(scene.receivers.size());
    push.virtualPagesX = desc.virtualPagesX;
    push.virtualPagesY = desc.virtualPagesY;
    push.physicalPagesX = desc.physicalPagesX;
    push.physicalPagesY = desc.physicalPagesY;
    push.pageSize = desc.pageSize;
    push.workCapacity = desc.maxWorkItems;

    std::vector<PrototypeMaterial> materials;
    for (size_t index = 0; index < scene.materials.size(); ++index) {
        const auto& material = scene.materials[index];
        uint32_t textureIndex = UINT32_MAX;
        if (!scene.alphaTextures.empty() && scene.alphaTextures[index]) {
            auto lease = state->registry->sampledImage(*scene.alphaTextures[index]);
            if (!lease) { return std::unexpected(lease.error()); }
            textureIndex = lease->shaderIndex();
            state->leases.push_back(*lease);
        }
        materials.push_back({{material.baseColorFactor.w, material.alphaCutoff, material.alphaMode == "MASK" ? 1.f : 0.f, 0.f},
            {textureIndex, material.doubleSided ? 1u : 0u, 0, 0}});
    }
    const uint32_t pageCount = desc.virtualPagesX * desc.virtualPagesY;
    const uint64_t sizes[] = {std::max<size_t>(1, scene.triangles.size()) * sizeof(VirtualShadowMapPrototypeTriangle),
        materials.size() * sizeof(PrototypeMaterial), scene.pages.size_bytes(), coverage->inputBytes().size(),
        uint64_t(desc.maxWorkItems + pageCount) * 8, 32, sizeof(VirtualShadowMapPrototypeProjection),
        std::max<size_t>(1, scene.receivers.size()) * sizeof(VirtualShadowMapPrototypeReceiver),
        std::max<size_t>(1, scene.receivers.size()) * sizeof(VirtualShadowMapPrototypeQuery)};
    const uint32_t strides[] = {112, 32, 16, 4, 8, 4, 16, 16, 32};
    const void* data[] = {scene.triangles.data(), materials.data(), scene.pages.data(), coverage->inputBytes().data(),
        nullptr, nullptr, &scene.projection, scene.receivers.data(), nullptr};
    const size_t dataSizes[] = {scene.triangles.size_bytes(), materials.size() * sizeof(PrototypeMaterial),
        scene.pages.size_bytes(), coverage->inputBytes().size(), 0, 0, sizeof(scene.projection), scene.receivers.size_bytes(), 0};
    uint32_t* indices[] = {&push.triangles, &push.materials, &push.pages, &push.coverageInputs,
        &push.workItems, &push.arguments, &push.projection, &push.receivers, &push.queryResults};
    for (size_t index = 0; index < state->buffers.size(); ++index) {
        const bool output = index == 4 || index == 5 || index == 8;
        auto buffer = device.createBuffer({.size = sizes[index], .structureStride = strides[index],
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (index == 5 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .memoryLocation = output ? MemoryLocation::Device : MemoryLocation::HostUpload});
        if (!buffer) { return std::unexpected(buffer.error()); }
        state->buffers[index] = std::move(*buffer);
        if (dataSizes[index] != 0) {
            if (auto result = upload(*state->buffers[index], data[index], dataSizes[index]); !result) { return result; }
        }
        auto lease = state->registry->storageBuffer(*state->buffers[index]);
        if (!lease) { return std::unexpected(lease.error()); }
        *indices[index] = lease->shaderIndex();
        state->leases.push_back(*lease);
    }
    auto texture = device.createTexture({.usage = TextureUsageBits::DepthStencilAttachment | TextureUsageBits::Sampled |
        TextureUsageBits::TransferSource, .format = Format::D32Sfloat,
        .width = desc.physicalPagesX * desc.pageSize, .height = desc.physicalPagesY * desc.pageSize});
    if (!texture) { return std::unexpected(texture.error()); }
    state->atlas = std::move(*texture);
    auto view = device.createTextureView(*state->atlas, {.format = Format::D32Sfloat});
    if (!view) { return std::unexpected(view.error()); }
    state->atlasView = std::move(*view);
    auto depthLease = state->registry->sampledImage(*state->atlasView);
    if (!depthLease) { return std::unexpected(depthLease.error()); }
    push.atlas = depthLease->shaderIndex();
    state->leases.push_back(*depthLease);
    auto sampler = state->registry->sampler({});
    if (!sampler) { return std::unexpected(sampler.error()); }
    push.sampler = sampler->shaderIndex();
    state->leases.push_back(*sampler);

    const char* entries[] = {"vsmPrototypeResetMain", "vsmPrototypeBuildMain", "vsmPrototypeFinalizeMain",
        "vsmPrototypeQueryMain", "vsmPrototypeClearMeshMain", "vsmPrototypeRasterMeshMain", "vsmPrototypeDepthFragmentMain"};
    const auto directory = generatedDirectory.string();
    const std::array paths{directory.c_str()};
    const std::array capabilities{"spvMeshShadingEXT"};
    const std::array macros{SlangMacroDefine{"METALLIC_CUSTOM_COVERAGE", coverage->coverageProgramCount() ? "1" : "0"}};
    for (size_t index = 0; index < state->shaders.size(); ++index) {
        auto compiled = ShaderRegistry::instance().getShader({
            .moduleName = "Features/VirtualShadowMaps/VirtualShadowMapPrototype",
            .entryPointName = entries[index], .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .additionalSearchPaths = paths, .capabilities = capabilities, .macroDefines = macros}, log);
        if (!compiled) { return std::unexpected(compiled.error()); }
        auto shader = ShaderRegistry::instance().getShaderModule(device, {.spirv = compiled->spirv, .debugName = entries[index]});
        if (!shader) { return std::unexpected(shader.error()); }
        state->shaders[index] = std::move(*shader);
        if (index < state->compute.size()) {
            auto pipeline = ShaderRegistry::instance().getComputePipeline(device, {
                .computeShader = {state->shaders[index].get()}, .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(PrototypePush)});
            if (!pipeline) { return std::unexpected(pipeline.error()); }
            state->compute[index] = std::move(*pipeline);
        }
    }
    for (size_t index = 0; index < state->graphics.size(); ++index) {
        auto pipeline = ShaderRegistry::instance().getGraphicsPipeline(device, {
            .meshShader = {state->shaders[4 + index].get()}, .fragmentShader = {state->shaders[6].get()},
            .depthStencilFormat = Format::D32Sfloat, .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true,
                .depthCompareOp = index == 0 ? CompareOp::Always : CompareOp::GreaterEqual}, .usesBindlessHeap = true});
        if (!pipeline) { return std::unexpected(pipeline.error()); }
        state->graphics[index] = std::move(*pipeline);
    }
    impl_ = std::move(state);
    return {};
}

Result<> VirtualShadowMapPrototype::updatePages(std::span<const VirtualShadowMapPrototypePage> pages)
{
    if (!validPages(impl_->desc, pages)) { return makeError(Error::InvalidArgument); }
    return upload(*impl_->buffers[2], pages.data(), pages.size_bytes());
}

Result<> VirtualShadowMapPrototype::updateTriangles(std::span<const VirtualShadowMapPrototypeTriangle> triangles)
{
    if (triangles.size() != impl_->push.triangleCount || !validTriangles(triangles, impl_->materialCount)) {
        return makeError(Error::InvalidArgument);
    }
    return upload(*impl_->buffers[0], triangles.data(), triangles.size_bytes());
}

Result<> VirtualShadowMapPrototype::updateProjection(const VirtualShadowMapPrototypeProjection& projection)
{
    if (!validProjection(projection)) { return makeError(Error::InvalidArgument); }
    return upload(*impl_->buffers[6], &projection, sizeof(projection));
}

Result<> VirtualShadowMapPrototype::record(CommandBuffer& commands, TextureLayout previousAtlasLayout, bool clearAtlas)
{
    if (previousAtlasLayout == TextureLayout::Undefined && !clearAtlas) { return makeError(Error::InvalidArgument); }
    auto& state = *impl_;
    auto push = state.push;
    push.reserved = clearAtlas ? 1u : 0u;
    if (auto result = state.registry->bind(commands, state.leases); !result) { return result; }
    const BufferBarrierDesc prepare[] = {
        {.buffer = state.buffers[4].get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}},
        {.buffer = state.buffers[5].get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}},
        {.buffer = state.buffers[8].get(), .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite}}};
    if (auto result = commands.synchronize({.buffers = prepare}); !result) { return result; }
    auto dispatch = [&](size_t pipeline, uint32_t groups) -> Result<> {
        if (auto result = commands.bindExecution(state.compute[pipeline]->execution(), &push, sizeof(push)); !result) { return result; }
        return commands.dispatch(groups);
    };
    auto argumentBarrier = [&]() -> Result<> {
        const BufferBarrierDesc barrier{.buffer = state.buffers[5].get(),
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
        return commands.synchronize({.buffers = {&barrier, 1}});
    };
    commands.beginDebugLabel({.name = "VSM P0: build page work"});
    if (auto result = dispatch(0, 1); !result) { return result; }
    if (auto result = argumentBarrier(); !result) { return result; }
    if (auto result = dispatch(1, (std::max(push.triangleCount, push.virtualPagesX * push.virtualPagesY) + 63) / 64); !result) { return result; }
    if (auto result = argumentBarrier(); !result) { return result; }
    if (auto result = dispatch(2, 1); !result) { return result; }
    commands.endDebugLabel();
    const BufferBarrierDesc rasterInputs[] = {
        {.buffer = state.buffers[4].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::PreRasterization, AccessBits::ShaderRead}},
        {.buffer = state.buffers[5].get(), .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::DrawIndirect | PipelineStageBits::ComputeShader, AccessBits::IndirectRead | AccessBits::ShaderRead}}};
    const TextureBarrierDesc depthBegin{.texture = state.atlas.get(), .oldLayout = previousAtlasLayout,
        .newLayout = TextureLayout::DepthStencilAttachment,
        .before = previousAtlasLayout == TextureLayout::Undefined ? SyncScope{} :
            SyncScope{PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite}};
    if (auto result = commands.synchronize({.textures = {&depthBegin, 1}, .buffers = rasterInputs}); !result) { return result; }

    const Rect area{.width = push.physicalPagesX * push.pageSize, .height = push.physicalPagesY * push.pageSize};
    commands.beginDebugLabel({.name = "VSM P0: clear and raster dirty pages"});
    for (size_t pass = 0; pass < 2; ++pass) {
        const RenderingAttachmentDesc depth{.view = state.atlasView.get(), .layout = TextureLayout::DepthStencilAttachment,
            .loadOp = clearAtlas && pass == 0 ? LoadOp::Clear : LoadOp::Load, .storeOp = StoreOp::Store,
            .clearDepth = 0.0f};
        if (auto result = commands.beginRendering({.renderArea = area, .depthStencilAttachment = &depth}); !result) { return result; }
        if (auto result = commands.setViewport({.width = float(area.width), .height = float(area.height), .maxDepth = 1.0f}); !result) { return result; }
        commands.setScissor(area);
        if (auto result = commands.bindExecution(state.graphics[pass]->execution(), &push, sizeof(push)); !result) { return result; }
        auto args = state.buffers[5]->slice({pass == 0 ? 12u : 0u, 12u});
        if (!args) { return std::unexpected(args.error()); }
        if (auto result = commands.drawMeshTasksIndirect(*args); !result) { return result; }
        commands.endRendering();
        if (pass == 0) {
            const TextureBarrierDesc cleared{.texture = state.atlas.get(), .oldLayout = TextureLayout::DepthStencilAttachment,
                .newLayout = TextureLayout::DepthStencilAttachment,
                .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite},
                .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite}};
            if (auto result = commands.synchronize({.textures = {&cleared, 1}}); !result) { return result; }
        }
    }
    commands.endDebugLabel();
    const TextureBarrierDesc sampleDepth{.texture = state.atlas.get(), .oldLayout = TextureLayout::DepthStencilAttachment,
        .newLayout = TextureLayout::ShaderRead, .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilWrite},
        .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
    if (auto result = commands.synchronize({.textures = {&sampleDepth, 1}}); !result) { return result; }
    if (push.receiverCount != 0) {
        commands.beginDebugLabel({.name = "VSM P0: query receivers"});
        if (auto result = dispatch(3, (push.receiverCount + 63) / 64); !result) { return result; }
        commands.endDebugLabel();
    }
    return {};
}

Texture& VirtualShadowMapPrototype::atlas() const { return *impl_->atlas; }
TextureView& VirtualShadowMapPrototype::atlasView() const { return *impl_->atlasView; }
Buffer& VirtualShadowMapPrototype::receiverResults() const { return *impl_->buffers[8]; }
Buffer& VirtualShadowMapPrototype::statistics() const { return *impl_->buffers[5]; }
const VirtualShadowMapPrototypeDesc& VirtualShadowMapPrototype::desc() const { return impl_->desc; }

} // namespace metallic::render

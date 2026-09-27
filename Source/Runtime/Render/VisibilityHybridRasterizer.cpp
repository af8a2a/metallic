#include "Runtime/Render/ResourceSynchronization.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/SlangCompiler.h"
#include <algorithm>
#include <cmath>

namespace metallic::render {

Result<> VisibilityHybridRasterizer::initialize(Device& device, uint32_t width, uint32_t height,
    std::string& log, uint32_t capacity, uint32_t clusterCapacity)
{
    if (!device.capabilities().shaderBufferInt64Atomics) { return makeError(Error::Unsupported); }
    if (width == 0 || height == 0 || width > 32768 || height > 32768 || capacity == 0 || clusterCapacity == 0 || clusterCapacity > 0x01ffffffu) {
        return makeError(Error::InvalidArgument);
    }
    initialized_ = false;
    clusterInitialized_ = false;
    push_.clusterCapacity = clusterCapacity;
    push_.width = width;
    push_.height = height;
    push_.capacity = std::min(capacity, 262144u);
    // Bound integer edge products even at the largest supported 32px triangle.
    if (device.capabilities().subPixelPrecisionBits < 1 || device.capabilities().subPixelPrecisionBits > 8) {
        return makeError(Error::Unsupported);
    }
    push_.subpixelBits = device.capabilities().subPixelPrecisionBits;
    auto result = device.createBindlessHeap({.maxBuffers = 5}).transform([&](auto rhiValue) { heap_ = std::move(rhiValue); });
    if (!result) { return result; }
    const uint64_t sizes[] = {32ull + push_.capacity * 64ull, uint64_t(width) * height * 8, 12};
    const uint32_t strides[] = {16, 8, 4};
    for (size_t i = 0; i < buffers_.size(); ++i) {
        result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (i == 2 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffers_[i] = std::move(rhiValue); });
        if (!result) { return result; }
        BindlessHandle handle;
        result = heap_->allocateBuffer().transform([&](auto rhiValue) { handle = std::move(rhiValue); });
        if (result) { result = heap_->writeStorageBuffer(handle, *buffers_[i]); }
        if (!result) { return result; }
        if (i == 0) { push_.queueBuffer = handle.shaderIndex; }
        if (i == 1) { push_.pixelBuffer = handle.shaderIndex; }
        if (i == 2) { push_.argumentsBuffer = handle.shaderIndex; }
    }
    const uint64_t clusterSizes[] = {64ull + uint64_t(clusterCapacity) * 9u * 4u + ((clusterCapacity + 127u) / 128u) * 5ull * 4u, 5u * 12u};
    for (size_t i = 0; i < 2; ++i) {
        auto& buffer = i == 0 ? clusterBuffer_ : clusterArguments_;
        result = device.createBuffer({.size = clusterSizes[i], .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (i == 1 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
        if (!result) { return result; }
        BindlessHandle handle;
        result = heap_->allocateBuffer().transform([&](auto rhiValue) { handle = std::move(rhiValue); });
        if (result) { result = heap_->writeStorageBuffer(handle, *buffer); }
        if (!result) { return result; }
        if (i == 0) { push_.clusterBuffer = handle.shaderIndex; } else { push_.clusterArgumentsBuffer = handle.shaderIndex; }
    }
    const char* clusterEntries[] = {"hybridClusterResetMain", "hybridClusterHistogramMain",
        "hybridClusterArgumentsMain", "hybridClusterScatterMain"};
    // Classification, stable bins, and active-group count/scatter dispatches.
    result = device.createBuffer({.size = 16 * sizeof(uint64_t), .structureStride = 8,
        .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource}).transform([&](auto rhiValue) { workloadBuffer_ = std::move(rhiValue); });
    if (!result) { return result; }
    result = device.createBuffer({.size = 36, .structureStride = 4,
        .usage = BufferUsageBits::Storage | BufferUsageBits::Indirect | BufferUsageBits::TransferSource}).transform([&](auto rhiValue) { candidateArguments_ = std::move(rhiValue); });
    if (!result) { return result; }
    for (size_t i = 0; i < clusterShaders_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityHybridRaster",
            .entryPointName = clusterEntries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log += shader.diagnostics; return result; }
        result = device.createShaderModule({
            .spirv = shader.spirv,
            .debugName = clusterEntries[i],
        }).transform([&](auto rhiValue) { clusterShaders_[i] = std::move(rhiValue); });
        if (result) { result = device.createComputePipeline({
            .computeShader = {clusterShaders_[i].get()},
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(Push),
        }).transform([&](auto rhiValue) { clusterPipelines_[i] = std::move(rhiValue); }); }
        if (!result) { return result; }
    }
    const char* entries[] = {"hybridResetMain", "hybridArgumentsMain", "hybridRasterMain",
        "hybridResolveVertexMain", "hybridResolveFragmentMain"};
    for (size_t i = 0; i < shaders_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityHybridRaster",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log += shader.diagnostics; return result; }
        result = device.createShaderModule({
            .spirv = shader.spirv,
            .debugName = entries[i],
        }).transform([&](auto rhiValue) { shaders_[i] = std::move(rhiValue); });
        if (!result) { return result; }
        if (i < compute_.size()) {
            result = device.createComputePipeline({
                .computeShader = {shaders_[i].get()},
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(Push),
            }).transform([&](auto rhiValue) { compute_[i] = std::move(rhiValue); });
            if (!result) { return result; }
        }
    }
    for (size_t i = 0; i < resolve_.size(); ++i) {
        result = device.createGraphicsPipeline({
            .vertexShader = {shaders_[3].get()},
            .fragmentShader = {shaders_[4].get()},
            .colorFormat = Format::R32Uint,
            .depthStencilFormat = Format::D32Sfloat,
            .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true,
                .depthCompareOp = i == 0 ? CompareOp::LessEqual : CompareOp::GreaterEqual},
            .usesBindlessHeap = true,
        }).transform([&](auto rhiValue) { resolve_[i] = std::move(rhiValue); });
        if (!result) { return result; }
    }
    return {};
}

bool VisibilityHybridRasterizer::supportsRenderExtent(uint32_t width, uint32_t height) const
{
    return width != 0 && height != 0 && width <= 32768 && height <= 32768 && buffers_[1] &&
        uint64_t(width) * height * sizeof(uint64_t) <= buffers_[1]->desc().size;
}

Result<> VisibilityHybridRasterizer::setRenderExtent(uint32_t width, uint32_t height)
{
    if (!supportsRenderExtent(width, height)) { return makeError(Error::InvalidArgument); }
    push_.width = width;
    push_.height = height;
    return {};
}

Result<> VisibilityHybridRasterizer::begin(CommandBuffer& commands, float maxPixels, bool reversedZ)
{
    commands.beginDebugLabel({.name = "Hybrid raster: clear queue and depth"});
    push_.maxPixels = std::clamp(std::isfinite(maxPixels) ? maxPixels : 8.0f, 1.0f, 32.0f);
    push_.reversedZ = reversedZ ? 1u : 0u;
    const ResourceState finals[] = {ResourceState::ShaderRead, ResourceState::ShaderRead, ResourceState::IndirectArgument};
    BufferBarrierDesc barriers[3];
    for (size_t i = 0; i < buffers_.size(); ++i) {
        barriers[i] = {
            .buffer = buffers_[i].get(),
            .before = resourceSyncScope(initialized_ ? finals[i] : ResourceState::Undefined, PipelineStageBits::AllCommands),
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        };
    }
    if (auto commandResult = commands.synchronize({.buffers = {barriers, 3}}); !commandResult) { return commandResult; }
    commands.bindBindlessHeap(*heap_);
    if (auto commandResult = commands.bindExecution((compute_[0])->execution()); !commandResult) { return commandResult; }
    commands.pushBindlessData(&push_, sizeof(push_));
    commands.dispatch((push_.width + 63u) / 64u, push_.height);
    for (auto& barrier : barriers) { barrier.before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}; }
    if (auto commandResult = commands.synchronize({.buffers = {barriers, 3}}); !commandResult) { return commandResult; }
    commands.endDebugLabel();
    return {};
}

Result<> VisibilityHybridRasterizer::resolve(CommandBuffer& commands, Texture& visibilityTexture,
    TextureView& visibility, Texture& depthTexture, TextureView& depth, bool softwareRasterized)
{
    commands.beginDebugLabel({.name = "Hybrid raster: software triangles"});
    BufferBarrierDesc barrier{
        .buffer = buffers_[0].get(),
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
    };
    if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    commands.bindBindlessHeap(*heap_);
    if (!softwareRasterized) {
        if (auto commandResult = commands.bindExecution((compute_[1])->execution()); !commandResult) { return commandResult; }
        commands.pushBindlessData(&push_, sizeof(push_));
        commands.dispatch(1);
        barrier = {
            .buffer = buffers_[2].get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
        };
        if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
        if (auto commandResult = commands.bindExecution((compute_[2])->execution()); !commandResult) { return commandResult; }
        commands.pushBindlessData(&push_, sizeof(push_));
        auto result = commands.dispatchIndirect(*buffers_[2]);
        if (!result) { commands.endDebugLabel(); return result; }
    } else {
        barrier = {
            .buffer = buffers_[2].get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
        };
        if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    }
    barrier = {
        .buffer = buffers_[1].get(),
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead},
    };
    if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    commands.endDebugLabel();
    commands.beginDebugLabel({.name = "Hybrid raster: merge visibility and depth"});
    const TextureBarrierDesc attachments[] = {
        {
            .texture = &visibilityTexture,
            .oldLayout = TextureLayout::ColorAttachment,
            .newLayout = TextureLayout::ColorAttachment,
            .before = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
            .after = {PipelineStageBits::ColorAttachment, AccessBits::ColorRead | AccessBits::ColorWrite},
        },
        {
            .texture = &depthTexture,
            .oldLayout = TextureLayout::DepthStencilAttachment,
            .newLayout = TextureLayout::DepthStencilAttachment,
            .before = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite},
            .after = {PipelineStageBits::DepthStencil, AccessBits::DepthStencilRead | AccessBits::DepthStencilWrite},
        }};
    if (auto commandResult = commands.synchronize({.textures = {attachments, 2}}); !commandResult) { return commandResult; }
    const RenderingAttachmentDesc color{.view = &visibility, .state = ResourceState::ColorAttachment,
        .loadOp = LoadOp::Load, .storeOp = StoreOp::Store};
    const RenderingAttachmentDesc z{.view = &depth, .state = ResourceState::DepthStencilAttachment,
        .loadOp = LoadOp::Load, .storeOp = StoreOp::Store};
    const Rect area{.width = push_.width, .height = push_.height};
    if (auto rendering = commands.beginRendering({
        .renderArea = area,
        .colorAttachments = {&color, 1},
        .depthStencilAttachment = &z,
    }); !rendering) { return rendering; }
    commands.setViewport({.width = float(push_.width), .height = float(push_.height), .maxDepth = 1.0f});
    commands.setScissor(area);
    if (auto commandResult = commands.bindExecution((resolve_[push_.reversedZ])->execution()); !commandResult) { return commandResult; }
    commands.pushBindlessData(&push_, sizeof(push_));
    commands.draw(3);
    commands.endRendering();
    commands.endDebugLabel();
    initialized_ = true;
    return {};
}

Result<> VisibilityHybridRasterizer::beginClusters(CommandBuffer& commands, float maxPixels, bool reversedZ,
    uint32_t producerPixelBuffer, uint32_t inputCount, bool stream, bool compact, bool tessellation)
{
    if (inputCount > push_.clusterCapacity) { return makeError(Error::InvalidArgument); }
    // Full HW never touches the software queue/pixel buffers; retain their
    // last resolved state so switching back to hybrid remains valid.
    if (stream && maxPixels == 0.0f) {
        push_.maxPixels = 0.0f;
        push_.reversedZ = reversedZ ? 1u : 0u;
    } else {
        if (auto result = begin(commands, maxPixels, reversedZ); !result) { return result; }
    }
    commands.bindBindlessHeap(*heap_);
    push_.producerPixelBuffer = producerPixelBuffer;
    push_.inputClusterCount = inputCount;
    push_.streamMode = (stream ? 1u : 0u) | (tessellation ? 2u : 0u);
    compactCandidates_ = compact;
    const BufferBarrierDesc barriers[] = {
        {
            .buffer = clusterBuffer_.get(),
            .before = resourceSyncScope(clusterInitialized_ ? ResourceState::ShaderRead : ResourceState::Undefined, PipelineStageBits::AllCommands),
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        },
        {
            .buffer = clusterArguments_.get(),
            .before = resourceSyncScope(clusterInitialized_ ? ResourceState::IndirectArgument : ResourceState::Undefined, PipelineStageBits::AllCommands),
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        },
        {
            .buffer = candidateArguments_.get(),
            .before = resourceSyncScope(clusterInitialized_ ? ResourceState::IndirectArgument : ResourceState::Undefined, PipelineStageBits::AllCommands),
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        }};
    if (auto commandResult = commands.synchronize({.buffers = {barriers, 3}}); !commandResult) { return commandResult; }
    if (auto commandResult = commands.bindExecution((clusterPipelines_[0])->execution()); !commandResult) { return commandResult; }
    commands.pushBindlessData(&push_, sizeof(push_));
    commands.dispatch(1);
    const BufferBarrierDesc ready{
        .buffer = clusterBuffer_.get(),
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
    };
    if (auto commandResult = commands.synchronize({.buffers = {&ready, 1}}); !commandResult) { return commandResult; }
    return {};
}

Result<> VisibilityHybridRasterizer::prepareClusterCandidates(CommandBuffer& commands)
{
    const BufferBarrierDesc barriers[] = {
        {
            .buffer = clusterBuffer_.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        },
        {
            .buffer = candidateArguments_.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
        }};
    if (auto commandResult = commands.synchronize({.buffers = {barriers, 2}}); !commandResult) { return commandResult; }
    return {};
}

Result<> VisibilityHybridRasterizer::prepareStreamClusterCandidates(CommandBuffer& commands,
    ComputePipeline& pipeline, MeshletStreamUserPush push)
{
    if (auto commandResult = commands.bindExecution((pipeline).execution()); !commandResult) { return commandResult; }
    // The prepare entry uses activeBuildPhase for setup/count/prefix/scatter.
    // Only the small block-count prefix stays on a single workgroup.
    for (uint32_t phase = 0; phase < 4; ++phase) {
        push.activeBuildPhase = phase;
        commands.pushBindlessData(&push, sizeof(push));
        if (phase == 0 || phase == 2) {
            commands.dispatch(1);
            if (auto commandResult = prepareClusterCandidates(commands); !commandResult) { return commandResult; }
        } else {
            const Result<> result = commands.dispatchIndirect(*candidateArguments_, kCandidateBuildArgumentsOffset);
            if (!result) { return result; }
            const BufferBarrierDesc barriers[] = {
                {
                    .buffer = clusterBuffer_.get(),
                    .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                },
                {
                    .buffer = candidateArguments_.get(),
                    .before = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
                    .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
                }};
            // Prefix rewrites arguments after count. Scatter only publishes
            // candidate slots, leaving arguments ready for classification.
            if (auto commandResult = commands.synchronize({.buffers = {barriers, phase == 1 ? 2u : 1u}}); !commandResult) { return commandResult; }
        }
    }
    return {};
}

Result<> VisibilityHybridRasterizer::cullStreamClusters(CommandBuffer& commands,
    ComputePipeline& pipeline, MeshletStreamUserPush push)
{
    if (auto commandResult = commands.bindExecution((pipeline).execution()); !commandResult) { return commandResult; }
    push.activeBuildPhase = 0;
    commands.pushBindlessData(&push, sizeof(push));
    const Result<> result = commands.dispatchIndirect(*candidateArguments_, 12);
    if (!result) { return result; }
    const BufferBarrierDesc barriers[] = {
        {
            .buffer = clusterBuffer_.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        },
        {
            .buffer = candidateArguments_.get(),
            .before = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
            .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        }};
    if (auto commandResult = commands.synchronize({.buffers = {barriers, 2}}); !commandResult) { return commandResult; }
    push.activeBuildPhase = 1;
    commands.pushBindlessData(&push, sizeof(push));
    commands.dispatch(1);
    if (auto commandResult = prepareClusterCandidates(commands); !commandResult) { return commandResult; }
    return {};
}

Result<> VisibilityHybridRasterizer::finishClusterBins(CommandBuffer& commands)
{
    commands.beginDebugLabel({.name = "Hybrid raster: stable cluster bins"});
    BufferBarrierDesc barrier{
        .buffer = clusterBuffer_.get(),
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
    };
    if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    commands.bindBindlessHeap(*heap_);
    const uint32_t blocks = (push_.inputClusterCount + 127u) / 128u;
    for (size_t i = 1; i < clusterPipelines_.size(); ++i) {
        if (auto commandResult = commands.bindExecution((clusterPipelines_[i])->execution()); !commandResult) { return commandResult; }
        commands.pushBindlessData(&push_, sizeof(push_));
        if (i == 2) { commands.dispatch(5); }
        else if (compactCandidates_) {
            const Result<> result = commands.dispatchIndirect(*candidateArguments_, 12);
            if (!result) { commands.endDebugLabel(); return result; }
        }
        else if (blocks != 0) { commands.dispatch(std::min(blocks, kDispatchWidth), (blocks + kDispatchWidth - 1u) / kDispatchWidth); }
        if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    }
    barrier.after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead};
    if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    barrier = {
        .buffer = clusterArguments_.get(),
        .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
        .after = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
    };
    if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    if (!compactCandidates_) {
        barrier = {
            .buffer = candidateArguments_.get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
        };
        if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    }
    clusterInitialized_ = true;
    commands.endDebugLabel();
    return {};
}

} // namespace metallic::render

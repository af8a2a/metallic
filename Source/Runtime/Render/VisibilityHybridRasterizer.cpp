#include "Runtime/Render/Core/HybridResolveParameters.h"
#include "Runtime/Render/Core/HybridRasterParameters.h"
#include "Runtime/Render/Core/HybridBinParameters.h"
#include "Runtime/Render/Core/ResourceSynchronization.h"
#include "Runtime/Render/VisibilityHybridRasterizer.h"
#include "Runtime/Render/Streamer/MeshletStreamRuntime.h"
#include "Runtime/Render/Core/SlangCompiler.h"
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
    device_ = &device;
    initialized_ = false;
    clusterInitialized_ = false;
    settings_.clusterCapacity = clusterCapacity;
    settings_.width = width;
    settings_.height = height;
    settings_.capacity = std::min(capacity, 262144u);
    // Bound integer edge products even at the largest supported 32px triangle.
    if (device.capabilities().subPixelPrecisionBits < 1 || device.capabilities().subPixelPrecisionBits > 8) {
        return makeError(Error::Unsupported);
    }
    settings_.subpixelBits = device.capabilities().subPixelPrecisionBits;
    Result<> result;
    const uint64_t sizes[] = {32ull + settings_.capacity * 64ull, uint64_t(width) * height * 8, 12};
    const uint32_t strides[] = {16, 8, 4};
    for (size_t i = 0; i < buffers_.size(); ++i) {
        result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (i == 2 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffers_[i] = std::move(rhiValue); });
        if (!result) { return result; }

    }
    const uint64_t clusterSizes[] = {64ull + uint64_t(clusterCapacity) * 9u * 4u + ((clusterCapacity + 127u) / 128u) * 5ull * 4u, 5u * 12u};
    for (size_t i = 0; i < 2; ++i) {
        auto& buffer = i == 0 ? clusterBuffer_ : clusterArguments_;
        result = device.createBuffer({.size = clusterSizes[i], .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (i == 1 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { buffer = std::move(rhiValue); });
        if (!result) { return result; }

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
    for (size_t i = 0; i < clusterKernels_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityHybridRaster",
            .entryPointName = clusterEntries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log += shader.diagnostics; return result; }
        result = clusterKernels_[i].initialize(device, {.spirv = shader.spirv,
            .parameters = parameterAbi<HybridBinParameters>(kHybridBinABI, ParameterTransport::InlinePush),
            .debugName = clusterEntries[i]}, log);
        if (!result) { return result; }
    }
    const char* entries[] = {"hybridResetMain", "hybridArgumentsMain", "hybridRasterMain",
        "hybridResolveVertexMain", "hybridResolveFragmentMain"};
    for (size_t i = 0; i < shaders_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityHybridRaster",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log += shader.diagnostics; return result; }
        if (i < rasterKernels_.size()) {
            result = rasterKernels_[i].initialize(device, {.spirv = shader.spirv,
                .parameters = parameterAbi<HybridRasterParameters>(kHybridRasterABI, ParameterTransport::InlinePush),
                .debugName = entries[i]}, log);
        } else {
            result = device.createShaderModule({.spirv = shader.spirv, .debugName = entries[i]})
                .transform([&](auto value) { shaders_[i] = std::move(value); });
        }
        if (!result) { return result; }

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

Result<EncodedParameters> VisibilityHybridRasterizer::encodeRasterParameters(CommandBuffer& commands)
{
    auto registry = device_->resourceRegistry();
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(*device_, **registry, commands.frameContext());
    const HybridRasterParameters params{
        .queue = writer.buffer(buffers_[0].get()), .pixels = writer.buffer(buffers_[1].get()),
        .arguments = writer.dataBuffer(buffers_[2].get(), sizeof(uint32_t), alignof(uint32_t)),
        .width = settings_.width, .height = settings_.height, .capacity = settings_.capacity,
        .maxPixels = settings_.maxPixels, .reversedZ = settings_.reversedZ, .subpixelBits = settings_.subpixelBits,
    };
    return writer.encode(params, kHybridRasterABI, ParameterTransport::InlinePush);
}

Result<EncodedParameters> VisibilityHybridRasterizer::encodeBinParameters(CommandBuffer& commands)
{
    auto registry = device_->resourceRegistry();
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(*device_, **registry, commands.frameContext());
    const HybridBinParameters params{
        .bins = writer.buffer(clusterBuffer_.get()),
        .arguments = writer.dataBuffer(clusterArguments_.get(), sizeof(uint32_t), alignof(uint32_t)),
        .width = settings_.width, .height = settings_.height, .clusterCapacity = settings_.clusterCapacity,
        .maxPixels = settings_.maxPixels, .reversedZ = settings_.reversedZ, .subpixelBits = settings_.subpixelBits,
        .inputClusterCount = settings_.inputClusterCount,
        .streamMode = settings_.streamMode,
    };
    return writer.encode(params, kHybridBinABI, ParameterTransport::InlinePush);
}

bool VisibilityHybridRasterizer::supportsRenderExtent(uint32_t width, uint32_t height) const
{
    return width != 0 && height != 0 && width <= 32768 && height <= 32768 && buffers_[1] &&
        uint64_t(width) * height * sizeof(uint64_t) <= buffers_[1]->desc().size;
}

Result<> VisibilityHybridRasterizer::setRenderExtent(uint32_t width, uint32_t height)
{
    if (!supportsRenderExtent(width, height)) { return makeError(Error::InvalidArgument); }
    settings_.width = width;
    settings_.height = height;
    return {};
}

Result<> VisibilityHybridRasterizer::begin(CommandBuffer& commands, float maxPixels, bool reversedZ)
{
    commands.beginDebugLabel({.name = "Hybrid raster: clear queue and depth"});
    settings_.maxPixels = std::clamp(std::isfinite(maxPixels) ? maxPixels : 8.0f, 1.0f, 32.0f);
    settings_.reversedZ = reversedZ ? 1u : 0u;
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
    auto parameters = encodeRasterParameters(commands);
    if (!parameters) { return makeError(parameters.error()); }
    if (auto result = rasterKernels_[0].dispatch(commands, *parameters, (settings_.width + 63u) / 64u, settings_.height); !result) { return result; }
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
    if (!softwareRasterized) {
        auto parameters = encodeRasterParameters(commands);
        if (!parameters) { return makeError(parameters.error()); }
        if (auto result = rasterKernels_[1].dispatch(commands, *parameters, 1); !result) { return result; }
        barrier = {
            .buffer = buffers_[2].get(),
            .before = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite},
            .after = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
        };
        if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
        auto result = rasterKernels_[2].dispatchIndirect(commands, *parameters, *buffers_[2]);
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
    auto registry = device_->resourceRegistry();
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(*device_, **registry, commands.frameContext());
    const HybridResolveParameters params{
        .pixels = writer.dataBuffer(buffers_[1].get(), sizeof(uint64_t), alignof(uint64_t)),
        .width = settings_.width, .reversedZ = settings_.reversedZ,
    };
    auto encoded = writer.encode(params, kHybridResolveABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    if (auto result = encoded->bindResources(commands); !result) { return result; }
    const auto bytes = encoded->inlineData();
    const Rect area{.width = settings_.width, .height = settings_.height};
    if (auto rendering = commands.beginRendering({
        .renderArea = area,
        .colorAttachments = {&color, 1},
        .depthStencilAttachment = &z,
    }); !rendering) { return rendering; }
    commands.setViewport({.width = float(settings_.width), .height = float(settings_.height), .maxDepth = 1.0f});
    commands.setScissor(area);
    if (auto result = commands.bindExecution(resolve_[settings_.reversedZ]->execution(), bytes.data(), uint32_t(bytes.size())); !result) {
        commands.endRendering();
        return result;
    }
    commands.draw(3);
    commands.endRendering();
    commands.endDebugLabel();
    initialized_ = true;
    return {};
}

Result<> VisibilityHybridRasterizer::beginClusters(CommandBuffer& commands, float maxPixels, bool reversedZ,
    uint32_t inputCount, bool stream, bool compact, bool tessellation)
{
    // Compact stream preparation checks its actual candidate count on GPU and
    // falls back to HW when scratch is exhausted. Record IDs remain unbounded by scratch.
    if (inputCount > settings_.clusterCapacity && !(stream && compact)) { return makeError(Error::InvalidArgument); }
    // Full HW never touches the software queue/pixel buffers; retain their
    // last resolved state so switching back to hybrid remains valid.
    if (stream && maxPixels == 0.0f) {
        settings_.maxPixels = 0.0f;
        settings_.reversedZ = reversedZ ? 1u : 0u;
    } else {
        if (auto result = begin(commands, maxPixels, reversedZ); !result) { return result; }
    }
    settings_.inputClusterCount = inputCount;
    settings_.streamMode = (stream ? 1u : 0u) | (tessellation ? 2u : 0u);
    compactCandidates_ = compact;
    const BufferBarrierDesc barriers[] = {
        {
            .buffer = clusterBuffer_.get(),
            .before = {PipelineStageBits::AllCommands, clusterInitialized_ ? AccessBits::MemoryRead | AccessBits::MemoryWrite : AccessBits::None},
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
    auto parameters = encodeBinParameters(commands);
    if (!parameters) { return makeError(parameters.error()); }
    if (auto result = clusterKernels_[0].dispatch(commands, *parameters, 1); !result) { return result; }
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
    const ComputeKernel& kernel, ParameterWriter& writer, StreamCandidateParameters params)
{
    params.bins = writer.buffer(clusterBuffer_.get());
    params.arguments = writer.dataBuffer(candidateArguments_.get(), 4, 4);
    // Each stage owns its inline setup/count/prefix/scatter parameters.
    // Only the small block-count prefix stays on a single workgroup.
    for (uint32_t phase = 0; phase < 4; ++phase) {
        params.stage = phase;
        auto encoded = writer.encode(params, kStreamCandidateABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        if (phase == 0 || phase == 2) {
            auto result = kernel.dispatch(commands, *encoded, 1);
            if (!result) { return result; }
            if (auto commandResult = prepareClusterCandidates(commands); !commandResult) { return commandResult; }
        } else {
            const Result<> result = kernel.dispatchIndirect(commands, *encoded, *candidateArguments_, kCandidateBuildArgumentsOffset);
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
    const ComputeKernel& kernel, ParameterWriter& writer, StreamClusterCullParameters params)
{
    params.bins = writer.buffer(clusterBuffer_.get());
    params.arguments = writer.buffer(candidateArguments_.get());
    params.stage = 0;
    auto cull = writer.encode(params, kStreamClusterCullABI, ParameterTransport::InlinePush);
    if (!cull) { return makeError(cull.error()); }
    params.stage = 1;
    auto finalize = writer.encode(params, kStreamClusterCullABI, ParameterTransport::InlinePush);
    if (!finalize) { return makeError(finalize.error()); }
    auto result = kernel.dispatchIndirect(commands, *cull, *candidateArguments_, 12);
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
    result = kernel.dispatch(commands, *finalize, 1);
    if (!result) { return result; }
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
    auto parameters = encodeBinParameters(commands);
    if (!parameters) { return makeError(parameters.error()); }
    const uint32_t blocks = (settings_.inputClusterCount + 127u) / 128u;
    for (size_t i = 1; i < clusterKernels_.size(); ++i) {
        Result<> result;
        if (i == 2) { result = clusterKernels_[i].dispatch(commands, *parameters, 5); }
        else if (compactCandidates_) { result = clusterKernels_[i].dispatchIndirect(commands, *parameters, *candidateArguments_, 12); }
        else if (blocks != 0) {
            result = clusterKernels_[i].dispatch(commands, *parameters,
                std::min(blocks, kDispatchWidth), (blocks + kDispatchWidth - 1u) / kDispatchWidth);
        }
        if (!result) { commands.endDebugLabel(); return result; }
        if (auto commandResult = commands.synchronize({.buffers = {&barrier, 1}}); !commandResult) { return commandResult; }
    }
    // Overflow HW raster publishes early-phase retry masks in this buffer.
    barrier.after = {PipelineStageBits::AllCommands, AccessBits::ShaderRead | AccessBits::ShaderWrite};
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

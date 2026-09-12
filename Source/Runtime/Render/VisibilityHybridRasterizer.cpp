#include "Runtime/Render/VisibilityHybridRasterizer.h"
#include "Runtime/Render/SlangCompiler.h"
#include <algorithm>
#include <cmath>

namespace metallic::render {

Result VisibilityHybridRasterizer::initialize(Device& device, uint32_t width, uint32_t height,
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
    auto result = device.createBindlessHeap({.maxBuffers = 5}, heap_);
    if (!result) { return result; }
    const uint64_t sizes[] = {32ull + push_.capacity * 64ull, uint64_t(width) * height * 8, 12};
    const uint32_t strides[] = {16, 8, 4};
    for (size_t i = 0; i < buffers_.size(); ++i) {
        result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (i == 2 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}, buffers_[i]);
        if (!result) { return result; }
        BindlessHandle handle;
        result = heap_->allocateBuffer(handle);
        if (result) { result = heap_->writeStorageBuffer(handle, *buffers_[i]); }
        if (!result) { return result; }
        if (i == 0) { push_.queueBuffer = handle.index; }
        if (i == 1) { push_.pixelBuffer = handle.index; }
        if (i == 2) { push_.argumentsBuffer = handle.index; }
    }
    const uint64_t clusterSizes[] = {64ull + uint64_t(clusterCapacity) * 8u * 4u + ((clusterCapacity + 127u) / 128u) * 5ull * 4u, 5u * 12u};
    for (size_t i = 0; i < 2; ++i) {
        auto& buffer = i == 0 ? clusterBuffer_ : clusterArguments_;
        result = device.createBuffer({.size = clusterSizes[i], .structureStride = 4,
            .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                (i == 1 ? BufferUsageBits::Indirect : BufferUsageBits::None),
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}, buffer);
        if (!result) { return result; }
        BindlessHandle handle;
        result = heap_->allocateBuffer(handle);
        if (result) { result = heap_->writeStorageBuffer(handle, *buffer); }
        if (!result) { return result; }
        if (i == 0) { push_.clusterBuffer = handle.index; } else { push_.clusterArgumentsBuffer = handle.index; }
    }
    const char* clusterEntries[] = {"hybridClusterResetMain", "hybridClusterHistogramMain",
        "hybridClusterArgumentsMain", "hybridClusterScatterMain"};
    for (size_t i = 0; i < clusterShaders_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityHybridRaster",
            .entryPointName = clusterEntries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
        if (!result) { log += shader.diagnostics; return result; }
        result = device.createShaderModule({.code = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
            .debugName = clusterEntries[i]}, clusterShaders_[i]);
        if (result) { result = device.createComputePipeline({.computeShader = clusterShaders_[i].get(),
            .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(Push)}, clusterPipelines_[i]); }
        if (!result) { return result; }
    }
    const char* entries[] = {"hybridResetMain", "hybridArgumentsMain", "hybridRasterMain",
        "hybridResolveVertexMain", "hybridResolveFragmentMain"};
    for (size_t i = 0; i < shaders_.size(); ++i) {
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "Features/VisibilityBuffer/VisibilityHybridRaster",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shader);
        if (!result) { log += shader.diagnostics; return result; }
        result = device.createShaderModule({.code = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .debugName = entries[i]}, shaders_[i]);
        if (!result) { return result; }
        if (i < compute_.size()) {
            result = device.createComputePipeline({.computeShader = shaders_[i].get(),
                .usesBindlessHeap = true, .bindlessUserPushDataSize = sizeof(Push)}, compute_[i]);
            if (!result) { return result; }
        }
    }
    for (size_t i = 0; i < resolve_.size(); ++i) {
        result = device.createGraphicsPipeline({.vertexShader = shaders_[3].get(),
            .fragmentShader = shaders_[4].get(), .colorFormat = Format::R32Uint,
            .depthStencilFormat = Format::D32Sfloat,
            .rasterization = {.cullMode = CullMode::None},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true,
                .depthCompareOp = i == 0 ? CompareOp::LessEqual : CompareOp::GreaterEqual},
            .usesBindlessHeap = true}, resolve_[i]);
        if (!result) { return result; }
    }
    return {};
}

void VisibilityHybridRasterizer::begin(CommandBuffer& commands, float maxPixels, bool reversedZ)
{
    commands.beginDebugLabel({.name = "Hybrid raster: clear queue and depth"});
    push_.maxPixels = std::clamp(std::isfinite(maxPixels) ? maxPixels : 8.0f, 1.0f, 32.0f);
    push_.reversedZ = reversedZ ? 1u : 0u;
    const ResourceState finals[] = {ResourceState::ShaderRead, ResourceState::ShaderRead, ResourceState::IndirectArgument};
    BufferBarrierDesc barriers[3];
    for (size_t i = 0; i < buffers_.size(); ++i) {
        barriers[i] = {.buffer = buffers_[i].get(),
            .before = initialized_ ? finals[i] : ResourceState::Undefined, .after = ResourceState::General};
    }
    commands.barrier({.buffers = barriers, .bufferCount = 3});
    commands.bindBindlessHeap(*heap_);
    commands.bindComputePipeline(*compute_[0]);
    commands.pushBindlessData(&push_, sizeof(push_));
    commands.dispatch((push_.width + 63u) / 64u, push_.height);
    for (auto& barrier : barriers) { barrier.before = ResourceState::General; }
    commands.barrier({.buffers = barriers, .bufferCount = 3});
    commands.endDebugLabel();
}

Result VisibilityHybridRasterizer::resolve(CommandBuffer& commands, Texture& visibilityTexture,
    TextureView& visibility, Texture& depthTexture, TextureView& depth, bool softwareRasterized)
{
    commands.beginDebugLabel({.name = "Hybrid raster: software triangles"});
    BufferBarrierDesc barrier{.buffer = buffers_[0].get(),
        .before = ResourceState::General, .after = ResourceState::ShaderRead};
    commands.barrier({.buffers = &barrier, .bufferCount = 1});
    commands.bindBindlessHeap(*heap_);
    if (!softwareRasterized) {
        commands.bindComputePipeline(*compute_[1]);
        commands.pushBindlessData(&push_, sizeof(push_));
        commands.dispatch(1);
        barrier = {.buffer = buffers_[2].get(), .before = ResourceState::General, .after = ResourceState::IndirectArgument};
        commands.barrier({.buffers = &barrier, .bufferCount = 1});
        commands.bindComputePipeline(*compute_[2]);
        commands.pushBindlessData(&push_, sizeof(push_));
        auto result = commands.dispatchIndirect(*buffers_[2]);
        if (!result) { commands.endDebugLabel(); return result; }
    } else {
        barrier = {.buffer = buffers_[2].get(), .before = ResourceState::General, .after = ResourceState::IndirectArgument};
        commands.barrier({.buffers = &barrier, .bufferCount = 1});
    }
    barrier = {.buffer = buffers_[1].get(), .before = ResourceState::General, .after = ResourceState::ShaderRead};
    commands.barrier({.buffers = &barrier, .bufferCount = 1});
    commands.endDebugLabel();
    commands.beginDebugLabel({.name = "Hybrid raster: merge visibility and depth"});
    const TextureBarrierDesc attachments[] = {
        {.texture = &visibilityTexture, .before = ResourceState::ColorAttachment, .after = ResourceState::ColorAttachment},
        {.texture = &depthTexture, .before = ResourceState::DepthStencilAttachment, .after = ResourceState::DepthStencilAttachment}};
    commands.barrier({.textures = attachments, .textureCount = 2});
    const RenderingAttachmentDesc color{.view = &visibility, .state = ResourceState::ColorAttachment,
        .loadOp = LoadOp::Load, .storeOp = StoreOp::Store};
    const RenderingAttachmentDesc z{.view = &depth, .state = ResourceState::DepthStencilAttachment,
        .loadOp = LoadOp::Load, .storeOp = StoreOp::Store};
    const Rect area{.width = push_.width, .height = push_.height};
    commands.beginRendering({.renderArea = area, .colorAttachments = &color,
        .colorAttachmentCount = 1, .depthStencilAttachment = &z});
    commands.setViewport({.width = float(push_.width), .height = float(push_.height), .maxDepth = 1.0f});
    commands.setScissor(area);
    commands.bindGraphicsPipeline(*resolve_[push_.reversedZ]);
    commands.pushBindlessData(&push_, sizeof(push_));
    commands.draw(3);
    commands.endRendering();
    commands.endDebugLabel();
    initialized_ = true;
    return {};
}

Result VisibilityHybridRasterizer::beginClusters(CommandBuffer& commands, float maxPixels, bool reversedZ,
    uint32_t producerPixelBuffer, uint32_t inputCount, bool stream)
{
    if (inputCount > push_.clusterCapacity) { return makeError(Error::InvalidArgument); }
    begin(commands, maxPixels, reversedZ);
    push_.producerPixelBuffer = producerPixelBuffer;
    push_.inputClusterCount = inputCount;
    push_.streamMode = stream ? 1u : 0u;
    const BufferBarrierDesc barriers[] = {
        {.buffer = clusterBuffer_.get(), .before = clusterInitialized_ ? ResourceState::ShaderRead : ResourceState::Undefined,
            .after = ResourceState::General},
        {.buffer = clusterArguments_.get(), .before = clusterInitialized_ ? ResourceState::IndirectArgument : ResourceState::Undefined,
            .after = ResourceState::General}};
    commands.barrier({.buffers = barriers, .bufferCount = 2});
    commands.bindComputePipeline(*clusterPipelines_[0]);
    commands.pushBindlessData(&push_, sizeof(push_));
    commands.dispatch(1);
    const BufferBarrierDesc ready{.buffer = clusterBuffer_.get(), .before = ResourceState::General, .after = ResourceState::General};
    commands.barrier({.buffers = &ready, .bufferCount = 1});
    return {};
}

void VisibilityHybridRasterizer::finishClusterBins(CommandBuffer& commands)
{
    commands.beginDebugLabel({.name = "Hybrid raster: stable cluster bins"});
    BufferBarrierDesc barrier{.buffer = clusterBuffer_.get(), .before = ResourceState::General, .after = ResourceState::General};
    commands.barrier({.buffers = &barrier, .bufferCount = 1});
    commands.bindBindlessHeap(*heap_);
    const uint32_t blocks = (push_.inputClusterCount + 127u) / 128u;
    for (size_t i = 1; i < clusterPipelines_.size(); ++i) {
        commands.bindComputePipeline(*clusterPipelines_[i]);
        commands.pushBindlessData(&push_, sizeof(push_));
        if (i == 2) { commands.dispatch(5); }
        else if (blocks != 0) { commands.dispatch(std::min(blocks, kDispatchWidth), (blocks + kDispatchWidth - 1u) / kDispatchWidth); }
        commands.barrier({.buffers = &barrier, .bufferCount = 1});
    }
    barrier.after = ResourceState::ShaderRead;
    commands.barrier({.buffers = &barrier, .bufferCount = 1});
    barrier = {.buffer = clusterArguments_.get(), .before = ResourceState::General, .after = ResourceState::IndirectArgument};
    commands.barrier({.buffers = &barrier, .bufferCount = 1});
    clusterInitialized_ = true;
    commands.endDebugLabel();
}

} // namespace metallic::render

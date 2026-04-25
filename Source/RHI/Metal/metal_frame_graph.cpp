#include "metal_frame_graph.h"

#ifdef __APPLE__

#include "imgui_metal_bridge.h"
#include "metal_resource_utils.h"
#include "metal_runtime.h"

#include <cstring>
#include <stdexcept>

namespace {

MTL::Buffer* metalBuffer(const RhiBuffer* buffer) {
    return buffer ? static_cast<MTL::Buffer*>(buffer->nativeHandle()) : nullptr;
}

MTL::Texture* metalTextureHandle(const RhiTexture* texture) {
    return texture ? static_cast<MTL::Texture*>(texture->nativeHandle()) : nullptr;
}

MTL::SamplerState* metalSampler(const RhiSampler* sampler) {
    return sampler ? static_cast<MTL::SamplerState*>(sampler->nativeHandle()) : nullptr;
}

MTL::DepthStencilState* metalDepthStencilState(const RhiDepthStencilState* state) {
    return state ? static_cast<MTL::DepthStencilState*>(state->nativeHandle()) : nullptr;
}

MTL::AccelerationStructure* metalAccelerationStructure(const RhiAccelerationStructure* accelerationStructure) {
    return accelerationStructure
        ? static_cast<MTL::AccelerationStructure*>(accelerationStructure->nativeHandle())
        : nullptr;
}

MTL::RenderPipelineState* metalRenderPipeline(const RhiGraphicsPipeline& pipeline) {
    return static_cast<MTL::RenderPipelineState*>(pipeline.nativeHandle());
}

MTL::ComputePipelineState* metalComputePipeline(const RhiComputePipeline& pipeline) {
    return static_cast<MTL::ComputePipelineState*>(pipeline.nativeHandle());
}

MTL::PrimitiveType metalPrimitiveType(RhiPrimitiveType primitiveType) {
    switch (primitiveType) {
    case RhiPrimitiveType::Triangle:
    default:
        return MTL::PrimitiveTypeTriangle;
    }
}

MTL::IndexType metalIndexType(RhiIndexType indexType) {
    switch (indexType) {
    case RhiIndexType::UInt16: return MTL::IndexTypeUInt16;
    case RhiIndexType::UInt32:
    default:
        return MTL::IndexTypeUInt32;
    }
}

size_t rhiIndexSize(RhiIndexType indexType) {
    switch (indexType) {
    case RhiIndexType::UInt16: return sizeof(uint16_t);
    case RhiIndexType::UInt32:
    default:
        return sizeof(uint32_t);
    }
}

MTL::Winding metalWinding(RhiWinding winding) {
    switch (winding) {
    case RhiWinding::Clockwise: return MTL::WindingClockwise;
    case RhiWinding::CounterClockwise:
    default:
        return MTL::WindingCounterClockwise;
    }
}

MTL::CullMode metalCullMode(RhiCullMode cullMode) {
    switch (cullMode) {
    case RhiCullMode::None: return MTL::CullModeNone;
    case RhiCullMode::Front: return MTL::CullModeFront;
    case RhiCullMode::Back:
    default:
        return MTL::CullModeBack;
    }
}

MTL::Origin metalOrigin(RhiOrigin3D origin) {
    return MTL::Origin(origin.x, origin.y, origin.z);
}

MTL::Size metalSize(RhiSize3D size) {
    return MTL::Size(size.width, size.height, size.depth);
}

MTL::ResourceID nullResourceID() {
    return MTL::ResourceID{0};
}

MTL::GPUAddress bufferAddress(const RhiBuffer& buffer, uint64_t offset) {
    auto* metal = static_cast<MTL::Buffer*>(buffer.nativeHandle());
    return metal ? metal->gpuAddress() + offset : 0;
}

class MetalArgumentTableBinder {
public:
    MetalArgumentTableBinder(void* commandBufferHandle, MetalArgumentTableSlot slot)
        : m_commandBufferHandle(commandBufferHandle),
          m_table(static_cast<MTL4::ArgumentTable*>(
              metalRuntimeArgumentTable(commandBufferHandle, slot))) {}

    MTL4::ArgumentTable* table() const { return m_table; }

    void setBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index, NS::UInteger stride = 0) {
        if (!m_table) {
            return;
        }

        MTL::GPUAddress address = 0;
        if (auto* metal = metalBuffer(buffer)) {
            address = metal->gpuAddress() + offset;
        }

        if (stride > 0) {
            m_table->setAddress(address, stride, index);
        } else {
            m_table->setAddress(address, index);
        }
    }

    void setBytes(const void* data, size_t size, uint32_t index) {
        if (!m_table || !data || size == 0) {
            return;
        }

        MetalUploadAllocation allocation;
        if (metalRuntimeUploadBytes(m_commandBufferHandle, data, size, 256, allocation)) {
            m_table->setAddress(allocation.gpuAddress, index);
        }
    }

    void setTexture(const RhiTexture* texture, uint32_t index) {
        if (!m_table) {
            return;
        }

        auto* metalTexture = metalTextureHandle(texture);
        m_table->setTexture(metalTexture ? metalTexture->gpuResourceID() : nullResourceID(), index);
    }

    void setSampler(const RhiSampler* sampler, uint32_t index) {
        if (!m_table) {
            return;
        }

        auto* metalSamplerState = metalSampler(sampler);
        m_table->setSamplerState(metalSamplerState ? metalSamplerState->gpuResourceID() : nullResourceID(), index);
    }

    void setAccelerationStructure(const RhiAccelerationStructure* accelerationStructure, uint32_t index) {
        if (!m_table) {
            return;
        }

        auto* metalAs = metalAccelerationStructure(accelerationStructure);
        m_table->setResource(metalAs ? metalAs->gpuResourceID() : nullResourceID(), index);
    }

private:
    void* m_commandBufferHandle = nullptr;
    MTL4::ArgumentTable* m_table = nullptr;
};

class MetalOwnedTexture final : public RhiTexture {
public:
    explicit MetalOwnedTexture(MTL::Texture* texture)
        : m_texture(texture) {}

    ~MetalOwnedTexture() override {
        if (m_texture) {
            metalReleaseHandle(m_texture);
        }
    }

    void* nativeHandle() const override { return m_texture; }
    uint32_t width() const override { return m_texture ? static_cast<uint32_t>(m_texture->width()) : 0; }
    uint32_t height() const override { return m_texture ? static_cast<uint32_t>(m_texture->height()) : 0; }

private:
    MTL::Texture* m_texture = nullptr;
};

class MetalOwnedBuffer final : public RhiBuffer {
public:
    MetalOwnedBuffer(MTL::Buffer* buffer, size_t byteSize)
        : m_buffer(buffer), m_size(byteSize) {}

    ~MetalOwnedBuffer() override {
        if (m_buffer) {
            metalReleaseHandle(m_buffer);
        }
    }

    size_t size() const override { return m_size; }
    void* nativeHandle() const override { return m_buffer; }
    void* mappedData() override { return m_buffer ? m_buffer->contents() : nullptr; }

private:
    MTL::Buffer* m_buffer = nullptr;
    size_t m_size = 0;
};

class MetalRenderCommandEncoder final : public RhiRenderCommandEncoder {
public:
    MetalRenderCommandEncoder(void* commandBufferHandle,
                              MTL4::RenderCommandEncoder* encoder,
                              TracyMetalGpuZone zone)
        : m_commandBufferHandle(commandBufferHandle),
          m_encoder(encoder),
          m_vertexTable(commandBufferHandle, MetalArgumentTableSlot::Vertex),
          m_fragmentTable(commandBufferHandle, MetalArgumentTableSlot::Fragment),
          m_meshTable(commandBufferHandle, MetalArgumentTableSlot::Mesh),
          m_zone(zone) {
        bindArgumentTables();
    }

    ~MetalRenderCommandEncoder() override {
        if (m_encoder) {
            m_encoder->endEncoding();
        }
        if (m_zone) {
            tracyMetalZoneEnd(m_zone);
        }
    }

    void* nativeHandle() const override { return m_encoder; }
    void setViewport(float width, float height, bool /*flipY*/ = true) override {
        MTL::Viewport viewport{};
        viewport.originX = 0.0;
        viewport.originY = 0.0;
        viewport.width = width;
        viewport.height = height;
        viewport.znear = 0.0;
        viewport.zfar = 1.0;
        m_encoder->setViewport(viewport);
    }
    void setDepthStencilState(const RhiDepthStencilState* state) override { m_encoder->setDepthStencilState(metalDepthStencilState(state)); }
    void setFrontFacingWinding(RhiWinding winding) override { m_encoder->setFrontFacingWinding(metalWinding(winding)); }
    void setCullMode(RhiCullMode cullMode) override { m_encoder->setCullMode(metalCullMode(cullMode)); }
    void setRenderPipeline(const RhiGraphicsPipeline& pipeline) override { m_encoder->setRenderPipelineState(metalRenderPipeline(pipeline)); }
    void setVertexBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override { m_vertexTable.setBuffer(buffer, offset, index); }
    void setFragmentBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override { m_fragmentTable.setBuffer(buffer, offset, index); }
    void setMeshBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override { m_meshTable.setBuffer(buffer, offset, index); }
    void setVertexBytes(const void* data, size_t size, uint32_t index) override { m_vertexTable.setBytes(data, size, index); }
    void setFragmentBytes(const void* data, size_t size, uint32_t index) override { m_fragmentTable.setBytes(data, size, index); }
    void setMeshBytes(const void* data, size_t size, uint32_t index) override { m_meshTable.setBytes(data, size, index); }
    void setPushConstants(const void* data, size_t size) override {
        m_vertexTable.setBytes(data, size, 0);
        m_fragmentTable.setBytes(data, size, 0);
    }
    void setFragmentTexture(const RhiTexture* texture, uint32_t index) override { m_fragmentTable.setTexture(texture, index); }
    void setFragmentTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) override {
        for (uint32_t index = 0; index < count; ++index) {
            m_fragmentTable.setTexture(textures ? textures[index] : nullptr, startIndex + index);
        }
    }
    void setMeshTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) override {
        for (uint32_t index = 0; index < count; ++index) {
            m_meshTable.setTexture(textures ? textures[index] : nullptr, startIndex + index);
        }
    }
    void setFragmentSampler(const RhiSampler* sampler, uint32_t index) override { m_fragmentTable.setSampler(sampler, index); }
    void setMeshSampler(const RhiSampler* sampler, uint32_t index) override { m_meshTable.setSampler(sampler, index); }
    void drawPrimitives(RhiPrimitiveType primitiveType, uint32_t vertexStart, uint32_t vertexCount) override {
        bindArgumentTables();
        m_encoder->drawPrimitives(metalPrimitiveType(primitiveType), vertexStart, vertexCount);
    }
    void drawIndexedPrimitives(RhiPrimitiveType primitiveType,
                               uint32_t indexCount,
                               RhiIndexType indexType,
                               const RhiBuffer& indexBuffer,
                               uint64_t indexBufferOffset) override {
        bindArgumentTables();
        m_encoder->drawIndexedPrimitives(metalPrimitiveType(primitiveType),
                                         indexCount,
                                         metalIndexType(indexType),
                                         bufferAddress(indexBuffer, indexBufferOffset),
                                         indexCount * rhiIndexSize(indexType));
    }
    void drawMeshThreadgroups(RhiSize3D threadgroupsPerGrid,
                              RhiSize3D threadsPerObjectThreadgroup,
                              RhiSize3D threadsPerMeshThreadgroup) override {
        bindArgumentTables();
        m_encoder->drawMeshThreadgroups(metalSize(threadgroupsPerGrid),
                                        metalSize(threadsPerObjectThreadgroup),
                                        metalSize(threadsPerMeshThreadgroup));
    }
    void drawMeshThreadgroupsIndirect(const RhiBuffer& indirectBuffer,
                                      uint64_t indirectBufferOffset,
                                      RhiSize3D threadsPerObjectThreadgroup,
                                      RhiSize3D threadsPerMeshThreadgroup) override {
        bindArgumentTables();
        m_encoder->drawMeshThreadgroups(bufferAddress(indirectBuffer, indirectBufferOffset),
                                        metalSize(threadsPerObjectThreadgroup),
                                        metalSize(threadsPerMeshThreadgroup));
    }
    void renderImGuiDrawData() override {
        bindArgumentTables();
        imguiRenderDrawData(m_commandBufferHandle, m_encoder);
    }

private:
    void bindArgumentTables() {
        if (m_vertexTable.table()) {
            m_encoder->setArgumentTable(m_vertexTable.table(), MTL::RenderStageVertex);
        }
        if (m_fragmentTable.table()) {
            m_encoder->setArgumentTable(m_fragmentTable.table(), MTL::RenderStageFragment);
        }
        if (m_meshTable.table()) {
            m_encoder->setArgumentTable(m_meshTable.table(), MTL::RenderStageMesh);
        }
    }

    void* m_commandBufferHandle = nullptr;
    MTL4::RenderCommandEncoder* m_encoder = nullptr;
    MetalArgumentTableBinder m_vertexTable;
    MetalArgumentTableBinder m_fragmentTable;
    MetalArgumentTableBinder m_meshTable;
    TracyMetalGpuZone m_zone = nullptr;
};

class MetalComputeCommandEncoder final : public RhiComputeCommandEncoder {
public:
    MetalComputeCommandEncoder(void* commandBufferHandle,
                               MTL4::ComputeCommandEncoder* encoder,
                               TracyMetalGpuZone zone)
        : m_encoder(encoder),
          m_computeTable(commandBufferHandle, MetalArgumentTableSlot::Compute),
          m_zone(zone) {
        bindArgumentTable();
    }

    ~MetalComputeCommandEncoder() override {
        if (m_encoder) {
            m_encoder->endEncoding();
        }
        if (m_zone) {
            tracyMetalZoneEnd(m_zone);
        }
    }

    void* nativeHandle() const override { return m_encoder; }
    void setComputePipeline(const RhiComputePipeline& pipeline) override { m_encoder->setComputePipelineState(metalComputePipeline(pipeline)); }
    void setBuffer(const RhiBuffer* buffer, uint64_t offset, uint32_t index) override { m_computeTable.setBuffer(buffer, offset, index); }
    void setBytes(const void* data, size_t size, uint32_t index) override { m_computeTable.setBytes(data, size, index); }
    void setPushConstants(const void* data, size_t size) override { m_computeTable.setBytes(data, size, 0); }
    void setTexture(const RhiTexture* texture, uint32_t index) override { m_computeTable.setTexture(texture, index); }
    void setStorageTexture(const RhiTexture* texture, uint32_t index) override { setTexture(texture, index); }
    void setTextures(const RhiTexture* const* textures, uint32_t startIndex, uint32_t count) override {
        for (uint32_t index = 0; index < count; ++index) {
            m_computeTable.setTexture(textures ? textures[index] : nullptr, startIndex + index);
        }
    }
    void setSampler(const RhiSampler* sampler, uint32_t index) override { m_computeTable.setSampler(sampler, index); }
    void setAccelerationStructure(const RhiAccelerationStructure* accelerationStructure, uint32_t index) override {
        m_computeTable.setAccelerationStructure(accelerationStructure, index);
    }
    void useResource(const RhiBuffer& /*resource*/, RhiResourceUsage /*usage*/) override {}
    void useResource(const RhiAccelerationStructure& /*resource*/, RhiResourceUsage /*usage*/) override {}
    void memoryBarrier(RhiBarrierScope /*scope*/) override {
        m_encoder->barrierAfterEncoderStages(MTL::StageDispatch,
                                             MTL::StageDispatch,
                                             MTL4::VisibilityOptionDevice);
    }
    void dispatchThreadgroups(RhiSize3D threadgroupsPerGrid, RhiSize3D threadsPerThreadgroup) override {
        bindArgumentTable();
        m_encoder->dispatchThreadgroups(metalSize(threadgroupsPerGrid), metalSize(threadsPerThreadgroup));
    }
    void dispatchThreadgroupsIndirect(const RhiBuffer& indirectBuffer,
                                      uint64_t indirectBufferOffset,
                                      RhiSize3D threadsPerThreadgroup) override {
        bindArgumentTable();
        m_encoder->dispatchThreadgroups(bufferAddress(indirectBuffer, indirectBufferOffset),
                                        metalSize(threadsPerThreadgroup));
    }

private:
    void bindArgumentTable() {
        if (m_computeTable.table()) {
            m_encoder->setArgumentTable(m_computeTable.table());
        }
    }

    MTL4::ComputeCommandEncoder* m_encoder = nullptr;
    MetalArgumentTableBinder m_computeTable;
    TracyMetalGpuZone m_zone = nullptr;
};

class MetalBlitCommandEncoder final : public RhiBlitCommandEncoder {
public:
    MetalBlitCommandEncoder(MTL4::ComputeCommandEncoder* encoder, TracyMetalGpuZone zone)
        : m_encoder(encoder), m_zone(zone) {}

    ~MetalBlitCommandEncoder() override {
        if (m_encoder) {
            m_encoder->endEncoding();
        }
        if (m_zone) {
            tracyMetalZoneEnd(m_zone);
        }
    }

    void* nativeHandle() const override { return m_encoder; }
    void copyTexture(const RhiTexture& source,
                     uint32_t sourceSlice,
                     uint32_t sourceLevel,
                     RhiOrigin3D sourceOrigin,
                     RhiSize3D sourceSize,
                     const RhiTexture& destination,
                     uint32_t destinationSlice,
                     uint32_t destinationLevel,
                     RhiOrigin3D destinationOrigin) override {
        m_encoder->copyFromTexture(static_cast<MTL::Texture*>(source.nativeHandle()),
                                   sourceSlice,
                                   sourceLevel,
                                   metalOrigin(sourceOrigin),
                                   metalSize(sourceSize),
                                   static_cast<MTL::Texture*>(destination.nativeHandle()),
                                   destinationSlice,
                                   destinationLevel,
                                   metalOrigin(destinationOrigin));
    }

private:
    MTL4::ComputeCommandEncoder* m_encoder = nullptr;
    TracyMetalGpuZone m_zone = nullptr;
};

} // namespace

std::unique_ptr<RhiTexture> MetalFrameGraphBackend::createTexture(const RhiTextureDesc& desc) {
    auto* textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
        metalPixelFormat(desc.format), desc.width, desc.height, false);
    textureDesc->setStorageMode(metalStorageMode(desc.storageMode));
    textureDesc->setUsage(metalTextureUsage(desc.usage));
    MTL::Texture* texture = m_device->newTexture(textureDesc);
    metalTrackAllocation(m_device, texture);
    return std::make_unique<MetalOwnedTexture>(texture);
}

std::unique_ptr<RhiBuffer> MetalFrameGraphBackend::createBuffer(const RhiBufferDesc& desc) {
    const MTL::ResourceOptions options = desc.hostVisible
        ? MTL::ResourceStorageModeShared
        : MTL::ResourceStorageModePrivate;

    MTL::Buffer* buffer = nullptr;
    if (desc.initialData && desc.hostVisible) {
        buffer = m_device->newBuffer(desc.initialData, desc.size, options);
    } else {
        buffer = m_device->newBuffer(desc.size, options);
        if (desc.initialData && buffer && buffer->contents()) {
            std::memcpy(buffer->contents(), desc.initialData, desc.size);
        }
    }

    if (buffer && desc.debugName) {
        buffer->setLabel(NS::String::string(desc.debugName, NS::UTF8StringEncoding));
    }
    metalTrackAllocation(m_device, buffer);

    return std::make_unique<MetalOwnedBuffer>(buffer, desc.size);
}

MetalCommandBuffer::MetalCommandBuffer(void* commandBufferHandle, TracyMetalCtxHandle tracyContext)
    : m_commandBuffer(static_cast<MTL4::CommandBuffer*>(commandBufferHandle)),
      m_tracyContext(tracyContext) {}

std::unique_ptr<RhiRenderCommandEncoder> MetalCommandBuffer::beginRenderPass(const RhiRenderPassDesc& desc) {
    auto* renderPassDesc = MTL4::RenderPassDescriptor::alloc()->init();
    uint32_t renderTargetWidth = 0;
    uint32_t renderTargetHeight = 0;

    for (uint32_t index = 0; index < desc.colorAttachmentCount; ++index) {
        const auto& colorAttachment = desc.colorAttachments[index];
        auto* attachment = renderPassDesc->colorAttachments()->object(index);
        auto* texture = metalTexture(colorAttachment.texture);
        attachment->setTexture(texture);
        attachment->setLoadAction(metalLoadAction(colorAttachment.loadAction));
        attachment->setStoreAction(metalStoreAction(colorAttachment.storeAction));
        attachment->setClearColor(metalClearColor(colorAttachment.clearColor));
        if (texture && renderTargetWidth == 0 && renderTargetHeight == 0) {
            renderTargetWidth = static_cast<uint32_t>(texture->width());
            renderTargetHeight = static_cast<uint32_t>(texture->height());
        }
    }

    if (desc.depthAttachment.bound) {
        auto* depthAttachment = renderPassDesc->depthAttachment();
        auto* texture = metalTexture(desc.depthAttachment.texture);
        depthAttachment->setTexture(texture);
        depthAttachment->setLoadAction(metalLoadAction(desc.depthAttachment.loadAction));
        depthAttachment->setStoreAction(metalStoreAction(desc.depthAttachment.storeAction));
        depthAttachment->setClearDepth(desc.depthAttachment.clearDepth);
        if (texture && renderTargetWidth == 0 && renderTargetHeight == 0) {
            renderTargetWidth = static_cast<uint32_t>(texture->width());
            renderTargetHeight = static_cast<uint32_t>(texture->height());
        }
    }

    if (renderTargetWidth > 0 && renderTargetHeight > 0) {
        renderPassDesc->setRenderTargetWidth(renderTargetWidth);
        renderPassDesc->setRenderTargetHeight(renderTargetHeight);
        renderPassDesc->setRenderTargetArrayLength(1);
    }

    const uint32_t slot = m_zoneIndex++ % 128u;
    (void)slot;
    TracyMetalGpuZone zone = nullptr;
    MTL4::RenderCommandEncoder* encoder = m_commandBuffer->renderCommandEncoder(renderPassDesc);
    renderPassDesc->release();
    return std::make_unique<MetalRenderCommandEncoder>(m_commandBuffer, encoder, zone);
}

std::unique_ptr<RhiComputeCommandEncoder> MetalCommandBuffer::beginComputePass(const RhiComputePassDesc& desc) {
    const uint32_t slot = m_zoneIndex++ % 128u;
    (void)slot;
    (void)desc;
    TracyMetalGpuZone zone = nullptr;
    MTL4::ComputeCommandEncoder* encoder = m_commandBuffer->computeCommandEncoder();
    return std::make_unique<MetalComputeCommandEncoder>(m_commandBuffer, encoder, zone);
}

std::unique_ptr<RhiBlitCommandEncoder> MetalCommandBuffer::beginBlitPass(const RhiBlitPassDesc& desc) {
    const uint32_t slot = m_zoneIndex++ % 128u;
    (void)slot;
    (void)desc;
    TracyMetalGpuZone zone = nullptr;
    MTL4::ComputeCommandEncoder* encoder = m_commandBuffer->computeCommandEncoder();
    return std::make_unique<MetalBlitCommandEncoder>(encoder, zone);
}

MTL::Texture* metalTexture(RhiTexture* texture) {
    return texture ? static_cast<MTL::Texture*>(texture->nativeHandle()) : nullptr;
}

const MTL::Texture* metalTexture(const RhiTexture* texture) {
    return texture ? static_cast<const MTL::Texture*>(texture->nativeHandle()) : nullptr;
}

MTL4::RenderCommandEncoder* metalEncoder(RhiRenderCommandEncoder& encoder) {
    return static_cast<MTL4::RenderCommandEncoder*>(encoder.nativeHandle());
}

MTL4::ComputeCommandEncoder* metalEncoder(RhiComputeCommandEncoder& encoder) {
    return static_cast<MTL4::ComputeCommandEncoder*>(encoder.nativeHandle());
}

MTL4::ComputeCommandEncoder* metalEncoder(RhiBlitCommandEncoder& encoder) {
    return static_cast<MTL4::ComputeCommandEncoder*>(encoder.nativeHandle());
}

MTL::PixelFormat metalPixelFormat(RhiFormat format) {
    switch (format) {
    case RhiFormat::R8Unorm: return MTL::PixelFormatR8Unorm;
    case RhiFormat::R16Float: return MTL::PixelFormatR16Float;
    case RhiFormat::R32Float: return MTL::PixelFormatR32Float;
    case RhiFormat::R32Uint: return MTL::PixelFormatR32Uint;
    case RhiFormat::RG8Unorm: return MTL::PixelFormatRG8Unorm;
    case RhiFormat::RG16Float: return MTL::PixelFormatRG16Float;
    case RhiFormat::RG32Float: return MTL::PixelFormatRG32Float;
    case RhiFormat::RGBA8Unorm: return MTL::PixelFormatRGBA8Unorm;
    case RhiFormat::RGBA8Srgb: return MTL::PixelFormatRGBA8Unorm_sRGB;
    case RhiFormat::BGRA8Unorm: return MTL::PixelFormatBGRA8Unorm;
    case RhiFormat::RGBA16Float: return MTL::PixelFormatRGBA16Float;
    case RhiFormat::RGBA32Float: return MTL::PixelFormatRGBA32Float;
    case RhiFormat::D32Float: return MTL::PixelFormatDepth32Float;
    case RhiFormat::D16Unorm: return MTL::PixelFormatDepth16Unorm;
    case RhiFormat::Undefined:
    default: return MTL::PixelFormatInvalid;
    }
}

RhiFormat metalToRhiFormat(MTL::PixelFormat format) {
    switch (format) {
    case MTL::PixelFormatR8Unorm: return RhiFormat::R8Unorm;
    case MTL::PixelFormatR16Float: return RhiFormat::R16Float;
    case MTL::PixelFormatR32Float: return RhiFormat::R32Float;
    case MTL::PixelFormatR32Uint: return RhiFormat::R32Uint;
    case MTL::PixelFormatRG8Unorm: return RhiFormat::RG8Unorm;
    case MTL::PixelFormatRG16Float: return RhiFormat::RG16Float;
    case MTL::PixelFormatRG32Float: return RhiFormat::RG32Float;
    case MTL::PixelFormatRGBA8Unorm: return RhiFormat::RGBA8Unorm;
    case MTL::PixelFormatRGBA8Unorm_sRGB: return RhiFormat::RGBA8Srgb;
    case MTL::PixelFormatBGRA8Unorm: return RhiFormat::BGRA8Unorm;
    case MTL::PixelFormatRGBA16Float: return RhiFormat::RGBA16Float;
    case MTL::PixelFormatRGBA32Float: return RhiFormat::RGBA32Float;
    case MTL::PixelFormatDepth32Float: return RhiFormat::D32Float;
    case MTL::PixelFormatDepth16Unorm: return RhiFormat::D16Unorm;
    default: return RhiFormat::Undefined;
    }
}

MTL::TextureUsage metalTextureUsage(RhiTextureUsage usage) {
    MTL::TextureUsage result = MTL::TextureUsageUnknown;
    if ((usage & RhiTextureUsage::RenderTarget) != RhiTextureUsage::None) {
        result = result | MTL::TextureUsageRenderTarget;
    }
    if ((usage & RhiTextureUsage::ShaderRead) != RhiTextureUsage::None) {
        result = result | MTL::TextureUsageShaderRead;
    }
    if ((usage & RhiTextureUsage::ShaderWrite) != RhiTextureUsage::None) {
        result = result | MTL::TextureUsageShaderWrite;
    }
    return result;
}

MTL::StorageMode metalStorageMode(RhiTextureStorageMode storageMode) {
    switch (storageMode) {
    case RhiTextureStorageMode::Shared: return MTL::StorageModeShared;
    case RhiTextureStorageMode::Private:
    default: return MTL::StorageModePrivate;
    }
}

MTL::LoadAction metalLoadAction(RhiLoadAction action) {
    switch (action) {
    case RhiLoadAction::Load: return MTL::LoadActionLoad;
    case RhiLoadAction::DontCare: return MTL::LoadActionDontCare;
    case RhiLoadAction::Clear:
    default: return MTL::LoadActionClear;
    }
}

MTL::StoreAction metalStoreAction(RhiStoreAction action) {
    switch (action) {
    case RhiStoreAction::DontCare: return MTL::StoreActionDontCare;
    case RhiStoreAction::Store:
    default: return MTL::StoreActionStore;
    }
}

MTL::ClearColor metalClearColor(const RhiClearColor& color) {
    return MTL::ClearColor(color.red, color.green, color.blue, color.alpha);
}

#endif

#include "metal_frame_graph.h"

#ifdef __APPLE__

#include <array>
#include <stdexcept>

namespace {

class MetalOwnedTexture final : public RhiTexture {
public:
    explicit MetalOwnedTexture(MTL::Texture* texture)
        : m_texture(texture) {}

    ~MetalOwnedTexture() override {
        if (m_texture) {
            m_texture->release();
        }
    }

    void* nativeHandle() const override { return m_texture; }

private:
    MTL::Texture* m_texture = nullptr;
};

class MetalRenderCommandEncoder final : public RhiRenderCommandEncoder {
public:
    MetalRenderCommandEncoder(MTL::RenderCommandEncoder* encoder, TracyMetalGpuZone zone)
        : m_encoder(encoder), m_zone(zone) {}

    ~MetalRenderCommandEncoder() override {
        if (m_encoder) {
            m_encoder->endEncoding();
        }
        if (m_zone) {
            tracyMetalZoneEnd(m_zone);
        }
    }

    void* nativeHandle() const override { return m_encoder; }

private:
    MTL::RenderCommandEncoder* m_encoder = nullptr;
    TracyMetalGpuZone m_zone = nullptr;
};

class MetalComputeCommandEncoder final : public RhiComputeCommandEncoder {
public:
    MetalComputeCommandEncoder(MTL::ComputeCommandEncoder* encoder, TracyMetalGpuZone zone)
        : m_encoder(encoder), m_zone(zone) {}

    ~MetalComputeCommandEncoder() override {
        if (m_encoder) {
            m_encoder->endEncoding();
        }
        if (m_zone) {
            tracyMetalZoneEnd(m_zone);
        }
    }

    void* nativeHandle() const override { return m_encoder; }

private:
    MTL::ComputeCommandEncoder* m_encoder = nullptr;
    TracyMetalGpuZone m_zone = nullptr;
};

class MetalBlitCommandEncoder final : public RhiBlitCommandEncoder {
public:
    MetalBlitCommandEncoder(MTL::BlitCommandEncoder* encoder, TracyMetalGpuZone zone)
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

private:
    MTL::BlitCommandEncoder* m_encoder = nullptr;
    TracyMetalGpuZone m_zone = nullptr;
};

TracyMetalGpuZone beginRenderZone(TracyMetalCtxHandle tracyContext,
                                  MTL::RenderPassDescriptor* descriptor,
                                  const char* label,
                                  uint32_t slot) {
    if (!tracyContext) {
        return nullptr;
    }

    static std::array<TracyMetalSrcLoc, 128> srcLocs{};
    srcLocs[slot] = {label, "FrameGraph::execute", __FILE__, __LINE__, 0};
    return tracyMetalZoneBeginRender(tracyContext, descriptor, &srcLocs[slot]);
}

TracyMetalGpuZone beginComputeZone(TracyMetalCtxHandle tracyContext,
                                   MTL::ComputePassDescriptor* descriptor,
                                   const char* label,
                                   uint32_t slot) {
    if (!tracyContext) {
        return nullptr;
    }

    static std::array<TracyMetalSrcLoc, 128> srcLocs{};
    srcLocs[slot] = {label, "FrameGraph::execute", __FILE__, __LINE__, 0};
    return tracyMetalZoneBeginCompute(tracyContext, descriptor, &srcLocs[slot]);
}

TracyMetalGpuZone beginBlitZone(TracyMetalCtxHandle tracyContext,
                                MTL::BlitPassDescriptor* descriptor,
                                const char* label,
                                uint32_t slot) {
    if (!tracyContext) {
        return nullptr;
    }

    static std::array<TracyMetalSrcLoc, 128> srcLocs{};
    srcLocs[slot] = {label, "FrameGraph::execute", __FILE__, __LINE__, 0};
    return tracyMetalZoneBeginBlit(tracyContext, descriptor, &srcLocs[slot]);
}

} // namespace

std::unique_ptr<RhiTexture> MetalFrameGraphBackend::createTexture(const RhiTextureDesc& desc) {
    auto* textureDesc = MTL::TextureDescriptor::texture2DDescriptor(
        metalPixelFormat(desc.format), desc.width, desc.height, false);
    textureDesc->setStorageMode(metalStorageMode(desc.storageMode));
    textureDesc->setUsage(metalTextureUsage(desc.usage));
    MTL::Texture* texture = m_device->newTexture(textureDesc);
    textureDesc->release();
    return std::make_unique<MetalOwnedTexture>(texture);
}

MetalCommandBuffer::MetalCommandBuffer(MTL::CommandBuffer* commandBuffer, TracyMetalCtxHandle tracyContext)
    : m_commandBuffer(commandBuffer), m_tracyContext(tracyContext) {}

std::unique_ptr<RhiRenderCommandEncoder> MetalCommandBuffer::beginRenderPass(const RhiRenderPassDesc& desc) {
    auto* renderPassDesc = MTL::RenderPassDescriptor::alloc()->init();

    for (uint32_t index = 0; index < desc.colorAttachmentCount; ++index) {
        const auto& colorAttachment = desc.colorAttachments[index];
        auto* attachment = renderPassDesc->colorAttachments()->object(index);
        attachment->setTexture(metalTexture(colorAttachment.texture));
        attachment->setLoadAction(metalLoadAction(colorAttachment.loadAction));
        attachment->setStoreAction(metalStoreAction(colorAttachment.storeAction));
        attachment->setClearColor(metalClearColor(colorAttachment.clearColor));
    }

    if (desc.depthAttachment.bound) {
        auto* depthAttachment = renderPassDesc->depthAttachment();
        depthAttachment->setTexture(metalTexture(desc.depthAttachment.texture));
        depthAttachment->setLoadAction(metalLoadAction(desc.depthAttachment.loadAction));
        depthAttachment->setStoreAction(metalStoreAction(desc.depthAttachment.storeAction));
        depthAttachment->setClearDepth(desc.depthAttachment.clearDepth);
    }

    const uint32_t slot = m_zoneIndex++ % 128u;
    TracyMetalGpuZone zone = beginRenderZone(m_tracyContext, renderPassDesc, desc.label ? desc.label : "Render Pass", slot);
    MTL::RenderCommandEncoder* encoder = m_commandBuffer->renderCommandEncoder(renderPassDesc);
    renderPassDesc->release();
    return std::make_unique<MetalRenderCommandEncoder>(encoder, zone);
}

std::unique_ptr<RhiComputeCommandEncoder> MetalCommandBuffer::beginComputePass(const RhiComputePassDesc& desc) {
    auto* computePassDesc = MTL::ComputePassDescriptor::alloc()->init();
    const uint32_t slot = m_zoneIndex++ % 128u;
    TracyMetalGpuZone zone = beginComputeZone(m_tracyContext, computePassDesc, desc.label ? desc.label : "Compute Pass", slot);
    MTL::ComputeCommandEncoder* encoder = m_commandBuffer->computeCommandEncoder(computePassDesc);
    computePassDesc->release();
    return std::make_unique<MetalComputeCommandEncoder>(encoder, zone);
}

std::unique_ptr<RhiBlitCommandEncoder> MetalCommandBuffer::beginBlitPass(const RhiBlitPassDesc& desc) {
    auto* blitPassDesc = MTL::BlitPassDescriptor::alloc()->init();
    const uint32_t slot = m_zoneIndex++ % 128u;
    TracyMetalGpuZone zone = beginBlitZone(m_tracyContext, blitPassDesc, desc.label ? desc.label : "Blit Pass", slot);
    MTL::BlitCommandEncoder* encoder = m_commandBuffer->blitCommandEncoder(blitPassDesc);
    blitPassDesc->release();
    return std::make_unique<MetalBlitCommandEncoder>(encoder, zone);
}

MTL::Texture* metalTexture(RhiTexture* texture) {
    return texture ? static_cast<MTL::Texture*>(texture->nativeHandle()) : nullptr;
}

const MTL::Texture* metalTexture(const RhiTexture* texture) {
    return texture ? static_cast<const MTL::Texture*>(texture->nativeHandle()) : nullptr;
}

MTL::RenderCommandEncoder* metalEncoder(RhiRenderCommandEncoder& encoder) {
    return static_cast<MTL::RenderCommandEncoder*>(encoder.nativeHandle());
}

MTL::ComputeCommandEncoder* metalEncoder(RhiComputeCommandEncoder& encoder) {
    return static_cast<MTL::ComputeCommandEncoder*>(encoder.nativeHandle());
}

MTL::BlitCommandEncoder* metalEncoder(RhiBlitCommandEncoder& encoder) {
    return static_cast<MTL::BlitCommandEncoder*>(encoder.nativeHandle());
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


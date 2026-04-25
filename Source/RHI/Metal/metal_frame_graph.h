#pragma once

#ifdef __APPLE__

#include "rhi_backend.h"
#include "tracy_metal.h"

#include <Metal/Metal.hpp>

class MetalImportedTexture final : public RhiTexture {
public:
    explicit MetalImportedTexture(void* textureHandle = nullptr)
        : m_texture(static_cast<MTL::Texture*>(textureHandle)) {}

    void setTexture(void* textureHandle) { m_texture = static_cast<MTL::Texture*>(textureHandle); }
    MTL::Texture* texture() const { return m_texture; }
    void* nativeHandle() const override { return m_texture; }
    uint32_t width() const override { return m_texture ? static_cast<uint32_t>(m_texture->width()) : 0; }
    uint32_t height() const override { return m_texture ? static_cast<uint32_t>(m_texture->height()) : 0; }

private:
    MTL::Texture* m_texture = nullptr;
};

class MetalFrameGraphBackend final : public RhiFrameGraphBackend {
public:
    explicit MetalFrameGraphBackend(void* deviceHandle)
        : m_device(static_cast<MTL::Device*>(deviceHandle)) {}

    std::unique_ptr<RhiTexture> createTexture(const RhiTextureDesc& desc) override;
    std::unique_ptr<RhiBuffer> createBuffer(const RhiBufferDesc& desc) override;

private:
    MTL::Device* m_device = nullptr;
};

class MetalCommandBuffer final : public RhiCommandBuffer {
public:
    MetalCommandBuffer(void* commandBufferHandle, TracyMetalCtxHandle tracyContext = nullptr);

    std::unique_ptr<RhiRenderCommandEncoder> beginRenderPass(const RhiRenderPassDesc& desc) override;
    std::unique_ptr<RhiComputeCommandEncoder> beginComputePass(const RhiComputePassDesc& desc) override;
    std::unique_ptr<RhiBlitCommandEncoder> beginBlitPass(const RhiBlitPassDesc& desc) override;

private:
    MTL4::CommandBuffer* m_commandBuffer = nullptr;
    TracyMetalCtxHandle m_tracyContext = nullptr;
    uint32_t m_zoneIndex = 0;
};

MTL::Texture* metalTexture(RhiTexture* texture);
const MTL::Texture* metalTexture(const RhiTexture* texture);
MTL4::RenderCommandEncoder* metalEncoder(RhiRenderCommandEncoder& encoder);
MTL4::ComputeCommandEncoder* metalEncoder(RhiComputeCommandEncoder& encoder);
MTL4::ComputeCommandEncoder* metalEncoder(RhiBlitCommandEncoder& encoder);

MTL::PixelFormat metalPixelFormat(RhiFormat format);
RhiFormat metalToRhiFormat(MTL::PixelFormat format);
MTL::TextureUsage metalTextureUsage(RhiTextureUsage usage);
MTL::StorageMode metalStorageMode(RhiTextureStorageMode storageMode);
MTL::LoadAction metalLoadAction(RhiLoadAction action);
MTL::StoreAction metalStoreAction(RhiStoreAction action);
MTL::ClearColor metalClearColor(const RhiClearColor& color);

#endif

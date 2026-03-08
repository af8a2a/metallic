#pragma once

#ifdef __APPLE__

#include "rhi_backend.h"
#include "tracy_metal.h"

#include <Metal/Metal.hpp>

class MetalImportedTexture final : public RhiTexture {
public:
    explicit MetalImportedTexture(MTL::Texture* texture = nullptr)
        : m_texture(texture) {}

    void setTexture(MTL::Texture* texture) { m_texture = texture; }
    MTL::Texture* texture() const { return m_texture; }
    void* nativeHandle() const override { return m_texture; }

private:
    MTL::Texture* m_texture = nullptr;
};

class MetalFrameGraphBackend final : public RhiFrameGraphBackend {
public:
    explicit MetalFrameGraphBackend(MTL::Device* device)
        : m_device(device) {}

    std::unique_ptr<RhiTexture> createTexture(const RhiTextureDesc& desc) override;

private:
    MTL::Device* m_device = nullptr;
};

class MetalCommandBuffer final : public RhiCommandBuffer {
public:
    MetalCommandBuffer(MTL::CommandBuffer* commandBuffer, TracyMetalCtxHandle tracyContext = nullptr);

    std::unique_ptr<RhiRenderCommandEncoder> beginRenderPass(const RhiRenderPassDesc& desc) override;
    std::unique_ptr<RhiComputeCommandEncoder> beginComputePass(const RhiComputePassDesc& desc) override;
    std::unique_ptr<RhiBlitCommandEncoder> beginBlitPass(const RhiBlitPassDesc& desc) override;

private:
    MTL::CommandBuffer* m_commandBuffer = nullptr;
    TracyMetalCtxHandle m_tracyContext = nullptr;
    uint32_t m_zoneIndex = 0;
};

MTL::Texture* metalTexture(RhiTexture* texture);
const MTL::Texture* metalTexture(const RhiTexture* texture);
MTL::RenderCommandEncoder* metalEncoder(RhiRenderCommandEncoder& encoder);
MTL::ComputeCommandEncoder* metalEncoder(RhiComputeCommandEncoder& encoder);
MTL::BlitCommandEncoder* metalEncoder(RhiBlitCommandEncoder& encoder);

MTL::PixelFormat metalPixelFormat(RhiFormat format);
RhiFormat metalToRhiFormat(MTL::PixelFormat format);
MTL::TextureUsage metalTextureUsage(RhiTextureUsage usage);
MTL::StorageMode metalStorageMode(RhiTextureStorageMode storageMode);
MTL::LoadAction metalLoadAction(RhiLoadAction action);
MTL::StoreAction metalStoreAction(RhiStoreAction action);
MTL::ClearColor metalClearColor(const RhiClearColor& color);

#endif


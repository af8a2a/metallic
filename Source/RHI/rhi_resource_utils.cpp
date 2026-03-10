#include "rhi_resource_utils.h"

#include "metal_resource_utils.h"

namespace {

MetalSamplerFilter toMetalFilter(RhiSamplerFilterMode filter) {
    switch (filter) {
    case RhiSamplerFilterMode::Nearest:
        return MetalSamplerFilter::Nearest;
    case RhiSamplerFilterMode::Linear:
    default:
        return MetalSamplerFilter::Linear;
    }
}

MetalSamplerMipFilter toMetalMipFilter(RhiSamplerMipFilterMode filter) {
    switch (filter) {
    case RhiSamplerMipFilterMode::Linear:
        return MetalSamplerMipFilter::Linear;
    case RhiSamplerMipFilterMode::None:
    default:
        return MetalSamplerMipFilter::None;
    }
}

MetalSamplerAddressMode toMetalAddressMode(RhiSamplerAddressMode mode) {
    switch (mode) {
    case RhiSamplerAddressMode::ClampToEdge:
        return MetalSamplerAddressMode::ClampToEdge;
    case RhiSamplerAddressMode::Repeat:
    default:
        return MetalSamplerAddressMode::Repeat;
    }
}

} // namespace

RhiBufferHandle rhiCreateSharedBuffer(const RhiDevice& device,
                                      const void* initialData,
                                      size_t size,
                                      const char* debugName) {
    return RhiBufferHandle(metalCreateSharedBuffer(device.nativeHandle(), initialData, size, debugName), size);
}

void* rhiBufferContents(const RhiBuffer& buffer) {
    return metalBufferContents(buffer.nativeHandle());
}

RhiTextureHandle rhiCreateTexture2D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    RhiFormat format,
                                    bool mipmapped,
                                    uint32_t mipLevelCount,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage) {
    return RhiTextureHandle(
        metalCreateTexture2D(device.nativeHandle(),
                             width,
                             height,
                             format,
                             mipmapped,
                             mipLevelCount,
                             storageMode,
                             usage),
        width,
        height);
}

RhiTextureHandle rhiCreateTexture3D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t depth,
                                    RhiFormat format,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage) {
    return RhiTextureHandle(
        metalCreateTexture3D(device.nativeHandle(),
                             width,
                             height,
                             depth,
                             format,
                             storageMode,
                             usage),
        width,
        height);
}

void rhiUploadTexture2D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        const void* data,
                        size_t bytesPerRow,
                        uint32_t mipLevel) {
    metalUploadTexture2D(texture.nativeHandle(), width, height, data, bytesPerRow, mipLevel);
}

void rhiUploadTexture3D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        uint32_t depth,
                        const void* data,
                        size_t bytesPerRow,
                        size_t bytesPerImage,
                        uint32_t mipLevel) {
    metalUploadTexture3D(texture.nativeHandle(),
                         width,
                         height,
                         depth,
                         data,
                         bytesPerRow,
                         bytesPerImage,
                         mipLevel);
}

void rhiGenerateMipmaps(const RhiCommandQueue& commandQueue, const RhiTexture& texture) {
    metalGenerateMipmaps(commandQueue.nativeHandle(), texture.nativeHandle());
}

RhiSamplerHandle rhiCreateSampler(const RhiDevice& device, const RhiSamplerDesc& desc) {
    MetalSamplerDesc metalDesc;
    metalDesc.minFilter = toMetalFilter(desc.minFilter);
    metalDesc.magFilter = toMetalFilter(desc.magFilter);
    metalDesc.mipFilter = toMetalMipFilter(desc.mipFilter);
    metalDesc.addressModeS = toMetalAddressMode(desc.addressModeS);
    metalDesc.addressModeT = toMetalAddressMode(desc.addressModeT);
    metalDesc.addressModeR = toMetalAddressMode(desc.addressModeR);
    return RhiSamplerHandle(metalCreateSampler(device.nativeHandle(), metalDesc));
}

RhiDepthStencilStateHandle rhiCreateDepthStencilState(const RhiDevice& device,
                                                      bool depthWriteEnabled,
                                                      bool reversedZ) {
    return RhiDepthStencilStateHandle(
        metalCreateDepthStencilState(device.nativeHandle(), depthWriteEnabled, reversedZ));
}

bool rhiSupportsRaytracing(const RhiDevice& device) {
    return metalSupportsRaytracing(device.nativeHandle());
}

RhiTextureHandle rhiRetainTexture(const RhiTexture& texture) {
    return RhiTextureHandle(metalRetainHandle(texture.nativeHandle()),
                            texture.width(),
                            texture.height());
}

void rhiReleaseNativeHandle(void* handle) {
    metalReleaseHandle(handle);
}

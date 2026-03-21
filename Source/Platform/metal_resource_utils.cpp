#include "metal_resource_utils.h"

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

namespace {

MTL::Device* metalDevice(void* handle) {
    return static_cast<MTL::Device*>(handle);
}

MTL::CommandQueue* metalCommandQueue(void* handle) {
    return static_cast<MTL::CommandQueue*>(handle);
}

MTL::Buffer* metalBuffer(void* handle) {
    return static_cast<MTL::Buffer*>(handle);
}

MTL::Texture* metalTexture(void* handle) {
    return static_cast<MTL::Texture*>(handle);
}

NS::Object* metalObject(void* handle) {
    return static_cast<NS::Object*>(handle);
}

MTL::PixelFormat metalPixelFormat(RhiFormat format) {
    switch (format) {
    case RhiFormat::R8Unorm: return MTL::PixelFormatR8Unorm;
    case RhiFormat::RGBA8Unorm: return MTL::PixelFormatRGBA8Unorm;
    case RhiFormat::RGBA8Srgb: return MTL::PixelFormatRGBA8Unorm_sRGB;
    case RhiFormat::BGRA8Unorm: return MTL::PixelFormatBGRA8Unorm;
    case RhiFormat::RGBA16Float: return MTL::PixelFormatRGBA16Float;
    case RhiFormat::RGBA32Float: return MTL::PixelFormatRGBA32Float;
    case RhiFormat::D32Float: return MTL::PixelFormatDepth32Float;
    case RhiFormat::Undefined:
    default:
        return MTL::PixelFormatInvalid;
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
    default:
        return MTL::StorageModePrivate;
    }
}

MTL::SamplerMinMagFilter metalSamplerFilter(MetalSamplerFilter filter) {
    switch (filter) {
    case MetalSamplerFilter::Nearest: return MTL::SamplerMinMagFilterNearest;
    case MetalSamplerFilter::Linear:
    default:
        return MTL::SamplerMinMagFilterLinear;
    }
}

MTL::SamplerMipFilter metalSamplerMipFilter(MetalSamplerMipFilter filter) {
    switch (filter) {
    case MetalSamplerMipFilter::Linear: return MTL::SamplerMipFilterLinear;
    case MetalSamplerMipFilter::None:
    default:
        return MTL::SamplerMipFilterNotMipmapped;
    }
}

MTL::SamplerAddressMode metalSamplerAddressMode(MetalSamplerAddressMode mode) {
    switch (mode) {
    case MetalSamplerAddressMode::ClampToEdge: return MTL::SamplerAddressModeClampToEdge;
    case MetalSamplerAddressMode::Repeat:
    default:
        return MTL::SamplerAddressModeRepeat;
    }
}

} // namespace

void* metalCreateSharedBuffer(void* deviceHandle,
                              const void* initialData,
                              size_t size,
                              const char* debugName) {
    auto* device = metalDevice(deviceHandle);
    if (!device || size == 0) {
        return nullptr;
    }

    auto* buffer = initialData
        ? device->newBuffer(initialData, size, MTL::ResourceStorageModeShared)
        : device->newBuffer(size, MTL::ResourceStorageModeShared);
    if (buffer && debugName) {
        buffer->setLabel(NS::String::string(debugName, NS::UTF8StringEncoding));
    }
    return buffer;
}

void* metalBufferContents(void* bufferHandle) {
    auto* buffer = metalBuffer(bufferHandle);
    return buffer ? buffer->contents() : nullptr;
}

void* metalCreateTexture2D(void* deviceHandle,
                           uint32_t width,
                           uint32_t height,
                           RhiFormat format,
                           bool mipmapped,
                           uint32_t mipLevelCount,
                           RhiTextureStorageMode storageMode,
                           RhiTextureUsage usage) {
    auto* device = metalDevice(deviceHandle);
    if (!device || width == 0 || height == 0) {
        return nullptr;
    }

    auto* desc = MTL::TextureDescriptor::texture2DDescriptor(
        metalPixelFormat(format), width, height, mipmapped);
    desc->setStorageMode(metalStorageMode(storageMode));
    desc->setUsage(metalTextureUsage(usage));
    if (mipLevelCount > 0) {
        desc->setMipmapLevelCount(mipLevelCount);
    }

    auto* texture = device->newTexture(desc);
    return texture;
}

void* metalCreateTexture3D(void* deviceHandle,
                           uint32_t width,
                           uint32_t height,
                           uint32_t depth,
                           RhiFormat format,
                           RhiTextureStorageMode storageMode,
                           RhiTextureUsage usage) {
    auto* device = metalDevice(deviceHandle);
    if (!device || width == 0 || height == 0 || depth == 0) {
        return nullptr;
    }

    auto* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType3D);
    desc->setPixelFormat(metalPixelFormat(format));
    desc->setWidth(width);
    desc->setHeight(height);
    desc->setDepth(depth);
    desc->setMipmapLevelCount(1);
    desc->setStorageMode(metalStorageMode(storageMode));
    desc->setUsage(metalTextureUsage(usage));

    auto* texture = device->newTexture(desc);
    desc->release();
    return texture;
}

void metalUploadTexture2D(void* textureHandle,
                          uint32_t width,
                          uint32_t height,
                          const void* data,
                          size_t bytesPerRow,
                          uint32_t mipLevel) {
    auto* texture = metalTexture(textureHandle);
    if (!texture || !data || width == 0 || height == 0) {
        return;
    }

    texture->replaceRegion(MTL::Region(0, 0, 0, width, height, 1),
                           mipLevel,
                           data,
                           bytesPerRow);
}

void metalUploadTexture3D(void* textureHandle,
                          uint32_t width,
                          uint32_t height,
                          uint32_t depth,
                          const void* data,
                          size_t bytesPerRow,
                          size_t bytesPerImage,
                          uint32_t mipLevel) {
    auto* texture = metalTexture(textureHandle);
    if (!texture || !data || width == 0 || height == 0 || depth == 0) {
        return;
    }

    texture->replaceRegion(MTL::Region(0, 0, 0, width, height, depth),
                           mipLevel,
                           0,
                           data,
                           bytesPerRow,
                           bytesPerImage);
}

void metalGenerateMipmaps(void* commandQueueHandle, void* textureHandle) {
    auto* commandQueue = metalCommandQueue(commandQueueHandle);
    auto* texture = metalTexture(textureHandle);
    if (!commandQueue || !texture) {
        return;
    }

    auto* commandBuffer = commandQueue->commandBuffer();
    auto* blitEncoder = commandBuffer->blitCommandEncoder();
    blitEncoder->generateMipmaps(texture);
    blitEncoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
}

void* metalCreateSampler(void* deviceHandle, const MetalSamplerDesc& desc) {
    auto* device = metalDevice(deviceHandle);
    if (!device) {
        return nullptr;
    }

    auto* samplerDesc = MTL::SamplerDescriptor::alloc()->init();
    samplerDesc->setMinFilter(metalSamplerFilter(desc.minFilter));
    samplerDesc->setMagFilter(metalSamplerFilter(desc.magFilter));
    samplerDesc->setMipFilter(metalSamplerMipFilter(desc.mipFilter));
    samplerDesc->setSAddressMode(metalSamplerAddressMode(desc.addressModeS));
    samplerDesc->setTAddressMode(metalSamplerAddressMode(desc.addressModeT));
    samplerDesc->setRAddressMode(metalSamplerAddressMode(desc.addressModeR));

    auto* sampler = device->newSamplerState(samplerDesc);
    samplerDesc->release();
    return sampler;
}

void* metalCreateDepthStencilState(void* deviceHandle, bool depthWriteEnabled, bool reversedZ) {
    auto* device = metalDevice(deviceHandle);
    if (!device) {
        return nullptr;
    }

    auto* depthDesc = MTL::DepthStencilDescriptor::alloc()->init();
    depthDesc->setDepthCompareFunction(
        reversedZ ? MTL::CompareFunctionGreater : MTL::CompareFunctionLess);
    depthDesc->setDepthWriteEnabled(depthWriteEnabled);

    auto* depthState = device->newDepthStencilState(depthDesc);
    depthDesc->release();
    return depthState;
}

bool metalSupportsRaytracing(void* deviceHandle) {
    auto* device = metalDevice(deviceHandle);
    return device && device->supportsRaytracing();
}

uint32_t metalTextureWidth(void* textureHandle) {
    auto* texture = metalTexture(textureHandle);
    return texture ? static_cast<uint32_t>(texture->width()) : 0;
}

uint32_t metalTextureHeight(void* textureHandle) {
    auto* texture = metalTexture(textureHandle);
    return texture ? static_cast<uint32_t>(texture->height()) : 0;
}

void* metalRetainHandle(void* handle) {
    auto* object = metalObject(handle);
    return object ? object->retain() : nullptr;
}

void metalReleaseHandle(void* handle) {
    auto* object = metalObject(handle);
    if (object) {
        object->release();
    }
}

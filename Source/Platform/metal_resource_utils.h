#pragma once

#include <cstddef>
#include <cstdint>

#include "rhi_backend.h"

enum class MetalSamplerFilter {
    Nearest,
    Linear,
};

enum class MetalSamplerMipFilter {
    None,
    Linear,
};

enum class MetalSamplerAddressMode {
    Repeat,
    ClampToEdge,
};

struct MetalSamplerDesc {
    MetalSamplerFilter minFilter = MetalSamplerFilter::Linear;
    MetalSamplerFilter magFilter = MetalSamplerFilter::Linear;
    MetalSamplerMipFilter mipFilter = MetalSamplerMipFilter::None;
    MetalSamplerAddressMode addressModeS = MetalSamplerAddressMode::Repeat;
    MetalSamplerAddressMode addressModeT = MetalSamplerAddressMode::Repeat;
    MetalSamplerAddressMode addressModeR = MetalSamplerAddressMode::Repeat;
};

void* metalCreateSharedBuffer(void* deviceHandle,
                              const void* initialData,
                              size_t size,
                              const char* debugName = nullptr);
void* metalBufferContents(void* bufferHandle);
void metalRegisterResidencySet(void* deviceHandle, void* residencySetHandle);
void metalUnregisterResidencySet(void* deviceHandle);
void metalTrackAllocation(void* deviceHandle, void* allocationHandle);
void metalUntrackAllocation(void* allocationHandle);

void* metalCreateTexture2D(void* deviceHandle,
                           uint32_t width,
                           uint32_t height,
                           RhiFormat format,
                           bool mipmapped,
                           uint32_t mipLevelCount,
                           RhiTextureStorageMode storageMode,
                           RhiTextureUsage usage);
void* metalCreateTexture3D(void* deviceHandle,
                           uint32_t width,
                           uint32_t height,
                           uint32_t depth,
                           RhiFormat format,
                           RhiTextureStorageMode storageMode,
                           RhiTextureUsage usage);
void metalUploadTexture2D(void* textureHandle,
                          uint32_t width,
                          uint32_t height,
                          const void* data,
                          size_t bytesPerRow,
                          uint32_t mipLevel = 0);
void metalUploadTexture3D(void* textureHandle,
                          uint32_t width,
                          uint32_t height,
                          uint32_t depth,
                          const void* data,
                          size_t bytesPerRow,
                          size_t bytesPerImage,
                          uint32_t mipLevel = 0);
void metalGenerateMipmaps(void* commandQueueHandle, void* textureHandle);

void* metalCreateSampler(void* deviceHandle, const MetalSamplerDesc& desc);
void* metalCreateDepthStencilState(void* deviceHandle, bool depthWriteEnabled, bool reversedZ);
bool metalSupportsRaytracing(void* deviceHandle);
uint32_t metalTextureWidth(void* textureHandle);
uint32_t metalTextureHeight(void* textureHandle);

void* metalRetainHandle(void* handle);
void metalReleaseHandle(void* handle);

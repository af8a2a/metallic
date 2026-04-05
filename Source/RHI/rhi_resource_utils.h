#pragma once

#include "rhi_backend.h"

RhiBufferHandle rhiCreateSharedBuffer(const RhiDevice& device,
                                      const void* initialData,
                                      size_t size,
                                      const char* debugName = nullptr);
void* rhiBufferContents(const RhiBuffer& buffer);

RhiTextureHandle rhiCreateTexture2D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    RhiFormat format,
                                    bool mipmapped,
                                    uint32_t mipLevelCount,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage);
RhiTextureHandle rhiCreateTexture3D(const RhiDevice& device,
                                    uint32_t width,
                                    uint32_t height,
                                    uint32_t depth,
                                    RhiFormat format,
                                    RhiTextureStorageMode storageMode,
                                    RhiTextureUsage usage);
void rhiUploadTexture2D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        const void* data,
                        size_t bytesPerRow,
                        uint32_t mipLevel = 0);
void rhiUploadTexture3D(const RhiTexture& texture,
                        uint32_t width,
                        uint32_t height,
                        uint32_t depth,
                        const void* data,
                        size_t bytesPerRow,
                        size_t bytesPerImage,
                        uint32_t mipLevel = 0);
void rhiGenerateMipmaps(const RhiCommandQueue& commandQueue, const RhiTexture& texture);

RhiSamplerHandle rhiCreateSampler(const RhiDevice& device, const RhiSamplerDesc& desc);
RhiDepthStencilStateHandle rhiCreateDepthStencilState(const RhiDevice& device,
                                                      bool depthWriteEnabled,
                                                      bool reversedZ);
bool rhiSupportsRaytracing(const RhiDevice& device);

RhiTextureHandle rhiRetainTexture(const RhiTexture& texture);
void rhiReleaseNativeHandle(void* handle);

template <typename Handle>
inline void rhiReleaseHandle(Handle& handle) {
    rhiReleaseNativeHandle(handle.nativeHandle());
    handle.setNativeHandle(nullptr);
}

#ifdef _WIN32
#include <vulkan/vulkan.h>
#include <type_traits>
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;

template <typename VkHandle>
inline uint64_t vulkanObjectHandle(VkHandle handle) {
    if constexpr (std::is_pointer_v<VkHandle>) {
        return reinterpret_cast<uint64_t>(handle);
    } else {
        return static_cast<uint64_t>(handle);
    }
}

struct VulkanResourceContextInfo {
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VmaAllocator allocator = nullptr;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsQueueFamily = 0;
    bool bufferDeviceAddressEnabled = false;
    bool externalHostMemoryEnabled = false;
    bool rayTracingEnabled = false;
    bool initialized = false;
    bool streamlineHooksEnabled = false;
    bool debugUtilsEnabled = false;
    void* streamlineBeginCommandBufferHook = nullptr;
    void* streamlineCmdBindPipelineHook = nullptr;
    void* streamlineCmdBindDescriptorSetsHook = nullptr;
    PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectName = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabel = nullptr;
};

void vulkanSetResourceContext(VkDevice device,
                              VkPhysicalDevice physicalDevice,
                              VmaAllocator allocator,
                              VkQueue queue,
                              uint32_t queueFamily,
                              bool bufferDeviceAddressEnabled,
                              bool externalHostMemoryEnabled,
                              bool rayTracingEnabled,
                              bool debugUtilsEnabled,
                              void* vkGetDeviceProcAddrProxy = nullptr);
const VulkanResourceContextInfo& vulkanGetResourceContext();
void vulkanClearResourceContext();
void vulkanSetStreamlineHookedCommandsEnabled(bool enabled);
void vulkanSetStreamlineCommandHooks(void* beginCommandBufferHook,
                                     void* cmdBindPipelineHook,
                                     void* cmdBindDescriptorSetsHook);

void vulkanSetObjectDebugName(VkDevice device,
                              VkObjectType objectType,
                              uint64_t handle,
                              const char* name);
void vulkanBeginDebugLabel(VkCommandBuffer commandBuffer, const char* label);
void vulkanEndDebugLabel(VkCommandBuffer commandBuffer);

class VulkanUploadService;
void vulkanSetUploadService(VulkanUploadService* service);

VkResult vulkanBeginCommandBufferHooked(VkCommandBuffer commandBuffer,
                                        const VkCommandBufferBeginInfo* beginInfo);
void vulkanCmdBindPipelineHooked(VkCommandBuffer commandBuffer,
                                 VkPipelineBindPoint pipelineBindPoint,
                                 VkPipeline pipeline);
void vulkanCmdBindDescriptorSetsHooked(VkCommandBuffer commandBuffer,
                                       VkPipelineBindPoint pipelineBindPoint,
                                       VkPipelineLayout layout,
                                       uint32_t firstSet,
                                       uint32_t descriptorSetCount,
                                       const VkDescriptorSet* descriptorSets,
                                       uint32_t dynamicOffsetCount,
                                       const uint32_t* dynamicOffsets);
#endif

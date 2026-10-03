#pragma once
#include "Runtime/Render/GAPI/RHI.h"
#include <volk.h>

namespace metallic::render::vulkan {

struct VulkanSyncScope {
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
};
struct SyncSupport {
    VkQueueFlags queues = 0;
    bool rayTracingAS = false;
    bool decompression = false;
    bool rayTracingPipeline = false;
    bool bindless = false;
};
VkImageLayout imageLayout(TextureLayout usage, bool unified);
VkPipelineStageFlags2 toVkPipelineStages(PipelineStageBits stages);
VkAccessFlags2 accessFlags(AccessBits access);
VulkanSyncScope scopeInfo(SyncScope scope);
bool validScope(SyncScope scope, SyncSupport support);
// Semaphore and timestamp stages use the same rules, excluding host-only stages.
bool validDeviceStages(PipelineStageBits stages, SyncSupport support);

} // namespace metallic::render::vulkan

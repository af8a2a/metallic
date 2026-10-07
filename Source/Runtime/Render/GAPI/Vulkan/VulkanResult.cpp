#include "VulkanResult.h"
#include "VulkanTooling.h"

namespace metallic::render::vulkan {

Result<> resultFromVk(VkResult result)
{
    if (result == VK_ERROR_DEVICE_LOST) { toolingHooks().deviceLost(); }
    return mapVkResult(result);
}

} // namespace metallic::render::vulkan

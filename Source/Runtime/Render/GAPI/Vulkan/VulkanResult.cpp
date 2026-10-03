#include "VulkanResult.h"
#include "Runtime/Render/Profiling/NsightAftermath.h"

namespace metallic::render::vulkan {

Result<> resultFromVk(VkResult result)
{
    if (result == VK_ERROR_DEVICE_LOST) { profiling::handleNsightAftermathDeviceLost(); }
    return mapVkResult(result);
}

} // namespace metallic::render::vulkan

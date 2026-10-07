#include "VulkanTooling.h"
#include "VulkanAftermath.h"
#include "VulkanNvPerf.h"
#include "VulkanNvPerfExtensions.h"
#include "VulkanNsightCapture.h"

namespace metallic::render::vulkan {
const ToolingHooks& toolingHooks()
{
    static const ToolingHooks hooks{
        nsightInjectionActive, nvPerfPassActive,
        [](std::vector<const char*>& extensions, uint32_t apiVersion, bool validation, std::string& error) {
            if (nvPerfRequested() && validation) { error = "Validation incompatible with NvPerf"; return false; }
            return nvPerfInstanceExtensions(extensions, apiVersion, error);
        },
        nvPerfDeviceExtensions,
        [](const char* name) { if (nsightAftermathSdkAvailable()) { initializeNsightAftermath(name); } },
        nsightAftermathInitialized, registerNsightAftermathShaderBinary, handleNsightAftermathDeviceLost};
    return hooks;
}
} // namespace metallic::render::vulkan

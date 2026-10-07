#include "Runtime/Render/GAPI/ShaderTarget.h"
#include "NativeDescriptorHeapSPIRV.h"

namespace metallic::render {
namespace {
bool finalizeVulkanSpirv(std::vector<uint32_t>& code, std::string& diagnostics)
{
    std::vector<uint32_t> normalized;
    if (!vulkan::normalizeNativeDescriptorHeapSpirv(code, normalized, diagnostics)) {
        return false;
    }
    code = std::move(normalized);
    return true;
}
} // namespace

const ShaderTargetPostprocess& vulkanSpirvTarget()
{
    static constexpr ShaderTargetPostprocess target{"vulkan-spirv", 1, finalizeVulkanSpirv};
    return target;
}
} // namespace metallic::render

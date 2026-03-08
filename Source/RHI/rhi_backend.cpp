#include "rhi_backend.h"

#ifdef _WIN32
#include "Vulkan/vulkan_backend.h"
#endif

std::unique_ptr<RhiContext> createRhiContext(RhiBackendType backend,
                                             const RhiCreateInfo& createInfo,
                                             std::string& errorMessage) {
    switch (backend) {
    case RhiBackendType::Vulkan:
#ifdef _WIN32
        return createVulkanContext(createInfo, errorMessage);
#else
        errorMessage = "Vulkan backend is only wired up on Windows in this revision.";
        return {};
#endif
    case RhiBackendType::Metal:
    default:
        errorMessage = "Requested RHI backend is not available in this build.";
        return {};
    }
}


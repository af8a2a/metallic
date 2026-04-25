#pragma once

#include <span>
#include <vector>

#include <vulkan/vulkan.h>

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
#include <map>
#include <memory>
#include "aftermath_gpu_crash_tracker.h"
#endif

class AftermathTracker {
public:
    static AftermathTracker& getInstance();

    void initialize();
    void addShaderBinary(const std::span<const uint32_t>& data);
    void errorCallback(VkResult result);

    bool isAvailable() const;

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    AftermathGpuCrashTracker& gpuCrashTracker();
    const char* diagnosticsConfigExtensionName() const;
    VkPhysicalDeviceDiagnosticsConfigFeaturesNV* diagnosticsConfigFeatures();
    VkDeviceDiagnosticsConfigCreateInfoNV* diagnosticsConfigCreateInfo();
#endif

private:
    AftermathTracker();

#if defined(METALLIC_HAS_AFTERMATH) && METALLIC_HAS_AFTERMATH
    AftermathGpuCrashTracker::MarkerMap m_markers;
    std::unique_ptr<AftermathGpuCrashTracker> m_tracker;

    VkPhysicalDeviceDiagnosticsConfigFeaturesNV m_diagnosticsConfigFeatures{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV,
        .pNext = nullptr,
        .diagnosticsConfig = VK_TRUE};
    VkDeviceDiagnosticsConfigCreateInfoNV m_diagnosticsConfigInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
        .pNext = nullptr,
        .flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV
               | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV
               | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV};
#endif
};

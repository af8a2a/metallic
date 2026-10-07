#pragma once

// Internal, deterministic negotiation model. Driver calls and SDK/environment
// policy stay in VulkanRHI.cpp so unsupported combinations can be tested on CPU.
#include "VulkanDeviceExtensions.h"
#include <volk.h>
#include <algorithm>
#include <cstring>
#include <vector>
#include <utility>

namespace metallic::render::vulkan::negotiation {

template <typename T>
void appendPNext(void**& tail, T& value)
{
    value.pNext = nullptr;
    *tail = &value;
    tail = &value.pNext;
}

struct VulkanExtensionSet {
    std::vector<VkExtensionProperties> properties;
    bool opacityMicromap = false;
    bool opacityMicromapExt = false;
#define MT_VK_EXTENSION(id, name, enabled) bool id = false;
#include "VulkanDeviceFeatureCatalog.inl"

    static VulkanExtensionSet from(std::vector<VkExtensionProperties> properties, bool nsightInjection)
    {
        VulkanExtensionSet result;
        result.properties = std::move(properties);
#define MT_VK_EXTENSION(id, name, enabled) result.id = result.has(name);
#include "VulkanDeviceFeatureCatalog.inl"
        // TODO(Nsight KHR OMM): retain the EXT route until injected build, capture
        // and replay pass; see NsightKhrOpacityMicromapInvestigation.md.
        result.opacityMicromapExt = nsightInjection;
        result.opacityMicromap = nsightInjection ? result.extOpacityMicromap
            : result.khrOpacityMicromap && result.deviceAddressCommands;
        return result;
    }

    bool has(const char* name) const
    {
        return std::ranges::any_of(properties, [name](const auto& property) {
            return std::strcmp(property.extensionName, name) == 0;
        });
    }
};

struct VulkanDeviceFeatureRequest {
#define MT_VK_FEATURE(id, requested, preferred, support, score, enable, publish) bool id = false;
#include "VulkanDeviceFeatureCatalog.inl"
    uint32_t preferredTaskSubgroupSize = 0;

    static VulkanDeviceFeatureRequest from(const DeviceDesc& desc,
        const VulkanDeviceExtensions& vulkanOptions, bool aftermathInitialized)
    {
        VulkanDeviceFeatureRequest result;
#define MT_VK_FEATURE(id, requested, preferred, support, score, enable, publish) result.id = (requested);
#include "VulkanDeviceFeatureCatalog.inl"
        result.preferredTaskSubgroupSize = desc.preferredTaskSubgroupSize;
        return result;
    }
};

struct VulkanFeatureStorage {
#define MT_VK_NODE(type, slot, stype, available, enabled) type slot{.sType = stype};
#include "VulkanDeviceFeatureCatalog.inl"
    VkPhysicalDeviceFeatures2 features{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

    VulkanFeatureStorage() = default;
    // The pNext links point into this object; copying or moving would dangle them.
    VulkanFeatureStorage(const VulkanFeatureStorage&) = delete;
    VulkanFeatureStorage& operator=(const VulkanFeatureStorage&) = delete;
};

struct VulkanDeviceFeatureProbe : VulkanFeatureStorage {
    VkPhysicalDeviceMemoryDecompressionPropertiesEXT decompressionProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_PROPERTIES_EXT,
    };
    VkPhysicalDeviceSubgroupProperties subgroupProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupSizeControlProperties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    };
    void buildChain(const VulkanExtensionSet& extensions)
    {
        features.pNext = nullptr;
        void** featureTail = &features.pNext;
#define MT_VK_NODE(type, slot, stype, available, enabled) if (available) { appendPNext(featureTail, slot); }
#include "VulkanDeviceFeatureCatalog.inl"
        properties.pNext = &subgroupProperties;
        subgroupProperties.pNext = &subgroupSizeControlProperties;
        subgroupSizeControlProperties.pNext = extensions.memoryDecompression ? &decompressionProperties : nullptr;
    }

    bool supportsRequiredCoreFeatures() const
    {
        return vulkan11Features.shaderDrawParameters == VK_TRUE &&
            vulkan12Features.timelineSemaphore == VK_TRUE &&
            vulkan12Features.hostQueryReset == VK_TRUE &&
            vulkan13Features.dynamicRendering == VK_TRUE &&
            vulkan13Features.synchronization2 == VK_TRUE;
    }

    bool supportsAccelerationStructure(const VulkanExtensionSet& extensions) const
    {
        return extensions.accelerationStructure &&
            extensions.deferredHostOperations &&
            accelerationStructureFeatures.accelerationStructure == VK_TRUE &&
            vulkan12Features.bufferDeviceAddress == VK_TRUE;
    }

    bool supportsStreamline(const VulkanExtensionSet& extensions, bool accelerationStructureSupported) const
    {
#if defined(VK_NVX_binary_import) && defined(VK_NVX_image_view_handle)
        return accelerationStructureSupported &&
            vulkan13Features.privateData == VK_TRUE &&
            extensions.rayQuery &&
            rayQueryFeatures.rayQuery == VK_TRUE &&
            extensions.rayTracingPipeline &&
            rayTracingPipelineFeatures.rayTracingPipeline == VK_TRUE &&
            extensions.pipelineLibrary &&
            extensions.pushDescriptor &&
            extensions.streamlineBinaryImport &&
            extensions.streamlineImageViewHandle;
#else
        (void)extensions;
        (void)accelerationStructureSupported;
        return false;
#endif
    }

    bool supportsAftermath(const VulkanExtensionSet& extensions) const
    {
#if defined(VK_NV_device_diagnostic_checkpoints) && defined(VK_NV_device_diagnostics_config)
        return extensions.aftermathDiagnosticCheckpoints &&
            extensions.aftermathDiagnosticsConfig &&
            diagnosticsConfigFeatures.diagnosticsConfig == VK_TRUE;
#else
        (void)extensions;
        return false;
#endif
    }

    bool supportsMeshShader(const VulkanExtensionSet& extensions) const
    {
#ifdef VK_EXT_mesh_shader
        return extensions.meshShader && meshShaderFeatures.meshShader == VK_TRUE;
#else
        (void)extensions;
        return false;
#endif
    }

    bool supportsTaskShader(const VulkanExtensionSet& extensions) const
    {
#ifdef VK_EXT_mesh_shader
        return extensions.meshShader && meshShaderFeatures.taskShader == VK_TRUE;
#else
        (void)extensions;
        return false;
#endif
    }

    bool supportsSubgroupSizeControl() const
    {
        return vulkan13Features.subgroupSizeControl == VK_TRUE &&
            subgroupSizeControlProperties.minSubgroupSize > 0 &&
            subgroupSizeControlProperties.maxSubgroupSize >=
                subgroupSizeControlProperties.minSubgroupSize;
    }

    bool supportsTaskShaderSubgroupBallot() const
    {
#ifdef VK_EXT_mesh_shader
        constexpr VkSubgroupFeatureFlags kRequiredOperations =
            VK_SUBGROUP_FEATURE_BASIC_BIT |
            VK_SUBGROUP_FEATURE_BALLOT_BIT;
        return (subgroupProperties.supportedStages &
                   VK_SHADER_STAGE_TASK_BIT_EXT) != 0 &&
            (subgroupProperties.supportedOperations & kRequiredOperations) ==
                kRequiredOperations;
#else
        return false;
#endif
    }

    bool supportsTaskShaderSubgroupSizeControl() const
    {
#ifdef VK_EXT_mesh_shader
        return supportsSubgroupSizeControl() &&
            (subgroupSizeControlProperties.requiredSubgroupSizeStages &
                VK_SHADER_STAGE_TASK_BIT_EXT) != 0;
#else
        return false;
#endif
    }
};


struct VulkanDeviceFeatureSelection {
#define MT_VK_FEATURE(id, requested, preferred, support, score, enable, publish) bool id = false;
#include "VulkanDeviceFeatureCatalog.inl"
    uint32_t subgroupSize = 0;
    uint32_t minSubgroupSize = 0;
    uint32_t maxSubgroupSize = 0;
    uint32_t maxComputeWorkgroupSubgroups = 0;

    static VulkanDeviceFeatureSelection select(VulkanDeviceFeatureRequest request,
        const VulkanExtensionSet& extensions, const VulkanDeviceFeatureProbe& probe,
        bool descriptorHeapUsable)
    {
        // Resolve catalog-owned request implications before evaluating support.
#define MT_VK_DEPENDENCY(id, prerequisite) request.prerequisite |= request.id;
#include "VulkanDeviceFeatureCatalog.inl"
        const bool accelerationStructureSupported = probe.supportsAccelerationStructure(extensions);
        const bool streamlineSupported = probe.supportsStreamline(extensions, accelerationStructureSupported);
        const bool aftermathSupported = probe.supportsAftermath(extensions);
        const bool meshShaderSupported = probe.supportsMeshShader(extensions);
        const bool taskShaderSupported = probe.supportsTaskShader(extensions);
        const bool subgroupSizeControlSupported = probe.supportsSubgroupSizeControl();
        constexpr VkSubgroupFeatureFlags kMaterialBinningOperations = VK_SUBGROUP_FEATURE_BASIC_BIT |
            VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
        constexpr VkSubgroupFeatureFlags kShuffleOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
        VulkanDeviceFeatureSelection result;
#define MT_VK_FEATURE(id, requested, preferred, support, score, enable, publish) result.id = (support);
#include "VulkanDeviceFeatureCatalog.inl"
        result.subgroupSize = probe.subgroupProperties.subgroupSize;
        result.minSubgroupSize = probe.subgroupSizeControlProperties.minSubgroupSize;
        result.maxSubgroupSize = probe.subgroupSizeControlProperties.maxSubgroupSize;
        result.maxComputeWorkgroupSubgroups = probe.subgroupSizeControlProperties.maxComputeWorkgroupSubgroups;
        return result;
    }

    bool usesBufferDeviceAddress() const { return true; }

    bool matches(const VulkanDeviceFeatureRequest& request) const
    {
#define MT_VK_FEATURE(id, requested, preferred, support, score, enable, publish) \
        if (preferred && request.id && !id) { return false; }
#include "VulkanDeviceFeatureCatalog.inl"
        return request.preferredTaskSubgroupSize == 0 || supportsTaskSubgroupSize(request.preferredTaskSubgroupSize);
    }

    bool supportsTaskSubgroupSize(uint32_t size) const
    {
        return size != 0 &&
            (size & (size - 1u)) == 0 &&
            taskShaderSubgroupSizeControl &&
            minSubgroupSize <= size &&
            maxSubgroupSize >= size;
    }

    int32_t score(const VulkanDeviceFeatureRequest& request) const
    {
        int32_t result = supportsTaskSubgroupSize(request.preferredTaskSubgroupSize) ? 8 : 0;
#define MT_VK_FEATURE(id, requested, preferred, support, weight, enable, publish) result += id ? weight : 0;
#include "VulkanDeviceFeatureCatalog.inl"
        return result;
    }

    void publish(DeviceCapabilities& caps, VulkanDeviceCapabilities& backendCaps) const
    {
#define MT_VK_FEATURE(id, requested, preferred, support, score, enable, publication) publication
#include "VulkanDeviceFeatureCatalog.inl"
        caps.subgroupSize = subgroupSize;
        caps.minSubgroupSize = minSubgroupSize;
        caps.maxSubgroupSize = maxSubgroupSize;
        caps.maxComputeWorkgroupSubgroups = maxComputeWorkgroupSubgroups;
    }
};

struct VulkanEnabledFeatureChain : VulkanFeatureStorage {
#if defined(VK_NV_device_diagnostics_config)
    VkDeviceDiagnosticsConfigCreateInfoNV diagnosticsConfigCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV,
        .flags =
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV |
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
            VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV,
    };
#endif
    explicit VulkanEnabledFeatureChain(const VulkanDeviceFeatureSelection& selection)
    {
        vulkan11Features.shaderDrawParameters = VK_TRUE;
        vulkan12Features.bufferDeviceAddress = VK_TRUE;
        vulkan12Features.timelineSemaphore = VK_TRUE;
        vulkan12Features.hostQueryReset = VK_TRUE;
        vulkan13Features.synchronization2 = VK_TRUE;
        vulkan13Features.dynamicRendering = VK_TRUE;
        deviceAddressCommandsFeatures.deviceAddressCommands = VK_TRUE;
        rayTracingPipelineFeatures.rayTracingPipeline = selection.streamline;
#define MT_VK_FEATURE(id, requested, preferred, support, score, enable, publish) enable
#include "VulkanDeviceFeatureCatalog.inl"
        void** featureTail = &features.pNext;
#define MT_VK_NODE(type, slot, stype, available, enabled) if (enabled) { appendPNext(featureTail, slot); }
#include "VulkanDeviceFeatureCatalog.inl"
#if defined(VK_NV_device_diagnostics_config)
        if (selection.aftermath) { *featureTail = &diagnosticsConfigCreateInfo; }
#endif
    }
};

inline std::vector<const char*> enabledDeviceExtensions(const VulkanDeviceFeatureSelection& selection)
{
    std::vector<const char*> result;
#define MT_VK_EXTENSION(id, name, enabled) if (enabled) { result.push_back(name); }
#include "VulkanDeviceFeatureCatalog.inl"
    return result;
}

} // namespace metallic::render::vulkan::negotiation

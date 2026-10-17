#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceFeatures.h"

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <type_traits>

namespace metallic::render::vulkan::negotiation {
namespace {

static_assert(!std::is_move_constructible_v<VulkanDeviceFeatureProbe>);
static_assert(!std::is_copy_constructible_v<VulkanEnabledFeatureChain>);

std::set<VkStructureType> chainTypes(const void* head)
{
    std::set<VkStructureType> result;
    for (auto* node = static_cast<const VkBaseInStructure*>(head); node; node = node->pNext) {
        if (!result.insert(node->sType).second) {
            ADD_FAILURE() << "Duplicate or cyclic pNext node: " << node->sType;
            break;
        }
    }
    return result;
}

std::set<std::string> extensionNames(const VulkanDeviceFeatureSelection& selection)
{
    const auto names = enabledDeviceExtensions(selection);
    std::set<std::string> result(names.begin(), names.end());
    EXPECT_EQ(result.size(), names.size()) << "Repeated device extension";
    return result;
}

VulkanExtensionSet extensionsFrom(std::initializer_list<const char*> names)
{
    std::vector<VkExtensionProperties> properties;
    for (const char* name : names) {
        VkExtensionProperties property{};
        std::strcpy(property.extensionName, name);
        properties.push_back(property);
    }
    return VulkanExtensionSet::from(std::move(properties));
}

class DeviceFeatures : public testing::Test {
protected:
    DeviceDesc desc;
    VulkanDeviceExtensions options;
    VulkanExtensionSet extensions;
    VulkanDeviceFeatureProbe probe;

    void SetUp() override
    {
        // Default shader objects are mandatory; optional requests remain enabled
        // so tests also exercise their graceful fallback on empty support.
        extensions.shaderObject = true;
        probe.shaderObjectFeatures.shaderObject = VK_TRUE;
    }

    VulkanDeviceFeatureRequest request(bool aftermathInitialized = false) const
    {
        return VulkanDeviceFeatureRequest::from(desc, options, aftermathInitialized);
    }

    VulkanDeviceFeatureSelection select(bool heapUsable = true, bool aftermathInitialized = false) const
    {
        return VulkanDeviceFeatureSelection::select(request(aftermathInitialized), extensions, probe, heapUsable);
    }

    void supportAS()
    {
        extensions.accelerationStructure = true;
        extensions.deferredHostOperations = true;
        probe.accelerationStructureFeatures.accelerationStructure = VK_TRUE;
        probe.vulkan12Features.bufferDeviceAddress = VK_TRUE;
    }
};

TEST_F(DeviceFeatures, RayTracingPipelineIsIndependentAndOptIn)
{
    supportAS();
    extensions.rayTracingPipeline = true;
    probe.rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    EXPECT_FALSE(select().rayTracingPipeline);
    desc.enableRayTracingPipeline = true;
    const auto selected = select();
    EXPECT_TRUE(selected.rayTracingPipeline);
    EXPECT_TRUE(selected.rayTracingAccelerationStructure);
    EXPECT_FALSE(selected.streamline);
    EXPECT_FALSE(selected.rayQuery);
    EXPECT_FALSE(selected.clusterAccelerationStructure);
    EXPECT_TRUE(selected.matches(request()));
    const auto names = extensionNames(selected);
    EXPECT_TRUE(names.contains(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME));
    EXPECT_TRUE(names.contains(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME));
    EXPECT_FALSE(names.contains(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME));
    VulkanEnabledFeatureChain chain(selected);
    EXPECT_EQ(chain.rayTracingPipelineFeatures.rayTracingPipeline, VK_TRUE);
    EXPECT_TRUE(chainTypes(chain.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR));
    DeviceCapabilities caps;
    VulkanDeviceCapabilities backend;
    selected.publish(caps, backend);
    EXPECT_TRUE(caps.rayTracingPipeline);
}

TEST_F(DeviceFeatures, UnavailableRayTracingPipelineLeavesComputeDeviceEligible)
{
    desc.enableRayTracingPipeline = true;
    EXPECT_FALSE(select().rayTracingPipeline);
    EXPECT_TRUE(select().matches(request()));
    supportAS();
    extensions.rayTracingPipeline = true;
    EXPECT_FALSE(select().rayTracingPipeline); // Extension without the feature bit.
    EXPECT_TRUE(select().matches(request()));
    probe.rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    extensions.deferredHostOperations = false;
    EXPECT_FALSE(select().rayTracingPipeline);
    EXPECT_TRUE(select().matches(request()));
}

#ifdef VK_EXT_ray_tracing_invocation_reorder
TEST_F(DeviceFeatures, InvocationReorderIsEXTOnlyAndOptIn)
{
    supportAS();
    extensions.rayTracingPipeline = true;
    probe.rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    extensions.rayTracingInvocationReorder = true;
    probe.rayTracingInvocationReorderFeatures.rayTracingInvocationReorder = VK_TRUE;
    EXPECT_FALSE(select().rayTracingInvocationReorder);
    desc.enableRayTracingInvocationReorder = true;
    const auto selected = select();
    EXPECT_TRUE(selected.rayTracingInvocationReorder);
    EXPECT_TRUE(selected.rayTracingPipeline);
    EXPECT_TRUE(selected.rayTracingAccelerationStructure);
    EXPECT_FALSE(selected.rayQuery);
    EXPECT_FALSE(selected.streamline);
    const auto names = extensionNames(selected);
    EXPECT_TRUE(names.contains(VK_EXT_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME));
    EXPECT_TRUE(names.contains(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME));
    EXPECT_FALSE(names.contains("VK_NV_ray_tracing_invocation_reorder"));
    VulkanEnabledFeatureChain chain(selected);
    EXPECT_EQ(chain.rayTracingInvocationReorderFeatures.rayTracingInvocationReorder, VK_TRUE);
    EXPECT_TRUE(chainTypes(chain.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_EXT));
    DeviceCapabilities caps;
    VulkanDeviceCapabilities backend;
    selected.publish(caps, backend);
    EXPECT_TRUE(caps.rayTracingInvocationReorder);
    EXPECT_FALSE(caps.rayTracingInvocationReorderReorderingHint); // Filled from the property query, not feature support.
}

TEST_F(DeviceFeatures, InvocationReorderRequiresEXTFeatureAndRayPipeline)
{
    supportAS();
    desc.enableRayTracingInvocationReorder = true;
    extensions.rayTracingPipeline = true;
    probe.rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    // An NV-only extension must never satisfy the EXT request.
    extensions = extensionsFrom({"VK_NV_ray_tracing_invocation_reorder",
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_EXT_SHADER_OBJECT_EXTENSION_NAME});
    EXPECT_FALSE(select().rayTracingInvocationReorder);
    EXPECT_TRUE(select().matches(request()));
    extensions.rayTracingInvocationReorder = true;
    EXPECT_FALSE(select().rayTracingInvocationReorder);
    probe.rayTracingInvocationReorderFeatures.rayTracingInvocationReorder = VK_TRUE;
    EXPECT_TRUE(select().rayTracingInvocationReorder);
    probe.rayTracingPipelineFeatures.rayTracingPipeline = VK_FALSE;
    EXPECT_FALSE(select().rayTracingInvocationReorder);
    EXPECT_TRUE(select().matches(request()));
}
#endif

TEST_F(DeviceFeatures, OptionalRequestsDoNotDisqualifyFallback)
{
    desc.enableAftermath = true;
    desc.enableStreamline = true;
    auto selected = select();
    EXPECT_TRUE(selected.matches(request()));
    EXPECT_FALSE(selected.deviceGeneratedCommands);
    EXPECT_FALSE(selected.unifiedImageLayouts);
    EXPECT_FALSE(selected.streamline);
    EXPECT_FALSE(selected.aftermath);
    desc.enableBindlessDescriptorHeap = true;
    EXPECT_FALSE(select().matches(request()));
}

#ifdef VK_NV_cluster_acceleration_structure
TEST_F(DeviceFeatures, RayQueriesRequireClustersWhileRasterOnlyRemainsAvailable)
{
    supportAS();
    extensions.rayQuery = true;
    probe.rayQueryFeatures.rayQuery = VK_TRUE;
    EXPECT_TRUE(select().matches(request()));
    desc.enableRayQuery = true;
    EXPECT_TRUE(request().clusterAccelerationStructure);
    EXPECT_FALSE(select().matches(request()));
    extensions.clusterAccelerationStructure = true;
    probe.clusterAccelerationStructureFeatures.clusterAccelerationStructure = VK_TRUE;
    EXPECT_TRUE(select().matches(request()));
    EXPECT_TRUE(select().clusterAccelerationStructure);
    probe.clusterAccelerationStructureFeatures.clusterAccelerationStructure = VK_FALSE;
    EXPECT_FALSE(select().matches(request()));
}

TEST_F(DeviceFeatures, ClusterRequiresEveryDependencyAndImpliesAS)
{
    desc.enableClusterAccelerationStructure = true;
    supportAS();
    extensions.clusterAccelerationStructure = true;
    probe.clusterAccelerationStructureFeatures.clusterAccelerationStructure = VK_TRUE;
    const auto selected = select();
    EXPECT_TRUE(selected.rayTracingAccelerationStructure);
    EXPECT_TRUE(selected.clusterAccelerationStructure);
    EXPECT_TRUE(selected.matches(request()));
    EXPECT_EQ(selected.score(request()), 8 + 4 + 64);
    DeviceCapabilities caps;
    VulkanDeviceCapabilities backendCaps;
    selected.publish(caps, backendCaps);
    EXPECT_TRUE(caps.clusterAccelerationStructure);
    EXPECT_TRUE(caps.rayTracingAccelerationStructure);
    const auto names = extensionNames(selected);
    EXPECT_TRUE(names.contains(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME));
    EXPECT_TRUE(names.contains(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME));
    EXPECT_TRUE(names.contains(VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME));
    VulkanEnabledFeatureChain chain(selected);
    EXPECT_TRUE(chainTypes(chain.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV));
    EXPECT_EQ(chain.clusterAccelerationStructureFeatures.clusterAccelerationStructure, VK_TRUE);

    for (bool* dependency : {&extensions.clusterAccelerationStructure, &extensions.accelerationStructure,
            &extensions.deferredHostOperations}) {
        *dependency = false;
        EXPECT_FALSE(select().clusterAccelerationStructure);
        EXPECT_FALSE(select().matches(request()));
        *dependency = true;
    }
    for (VkBool32* bit : {&probe.clusterAccelerationStructureFeatures.clusterAccelerationStructure,
            &probe.accelerationStructureFeatures.accelerationStructure, &probe.vulkan12Features.bufferDeviceAddress}) {
        *bit = VK_FALSE;
        EXPECT_FALSE(select().clusterAccelerationStructure);
        *bit = VK_TRUE;
    }
    desc.enableClusterAccelerationStructure = false;
    EXPECT_FALSE(select().clusterAccelerationStructure);
    EXPECT_FALSE(select().rayTracingAccelerationStructure);
}
#endif

#ifdef VK_NV_partitioned_acceleration_structure
TEST_F(DeviceFeatures, PartitionedRequiresASAndPublishesOnlySelectedSupport)
{
    desc.enablePartitionedAccelerationStructure = true;
    supportAS();
    extensions.partitionedAccelerationStructure = true;
    probe.partitionedAccelerationStructureFeatures.partitionedAccelerationStructure = VK_TRUE;
    EXPECT_TRUE(select().partitionedAccelerationStructure);
    EXPECT_EQ(select().score(request()), 8 + 4 + 128);
    probe.vulkan12Features.bufferDeviceAddress = VK_FALSE;
    const auto selected = select();
    EXPECT_FALSE(selected.partitionedAccelerationStructure);
    EXPECT_FALSE(extensionNames(selected).contains(VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME));
    VulkanEnabledFeatureChain chain(selected);
    EXPECT_FALSE(chainTypes(chain.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV));
}
#endif

TEST_F(DeviceFeatures, DescriptorHeapNeedsPropertiesAndAllIndexingBits)
{
    desc.enableBindlessDescriptorHeap = true;
    extensions.descriptorHeap = extensions.shaderUntypedPointers = true;
    probe.descriptorHeapFeatures.descriptorHeap = probe.untypedPointerFeatures.shaderUntypedPointers = VK_TRUE;
    std::array<VkBool32*, 6> bits{&probe.vulkan12Features.descriptorIndexing,
        &probe.vulkan12Features.runtimeDescriptorArray, &probe.vulkan12Features.shaderSampledImageArrayNonUniformIndexing,
        &probe.vulkan12Features.shaderStorageImageArrayNonUniformIndexing,
        &probe.vulkan12Features.shaderStorageBufferArrayNonUniformIndexing, &probe.vulkan12Features.bufferDeviceAddress};
    for (auto* bit : bits) { *bit = VK_TRUE; }
    EXPECT_TRUE(select().bindlessDescriptorHeap);
    EXPECT_TRUE(select().shaderUntypedPointers);
    EXPECT_FALSE(select(false).bindlessDescriptorHeap);
    for (auto* bit : bits) {
        *bit = VK_FALSE;
        EXPECT_FALSE(select().bindlessDescriptorHeap);
        EXPECT_FALSE(select().shaderUntypedPointers);
        *bit = VK_TRUE;
    }
    VulkanEnabledFeatureChain chain(select());
    EXPECT_EQ(chain.vulkan12Features.runtimeDescriptorArray, VK_TRUE);
    EXPECT_EQ(chain.descriptorHeapFeatures.descriptorHeap, VK_TRUE);
    EXPECT_EQ(chain.untypedPointerFeatures.shaderUntypedPointers, VK_TRUE);
}

#ifdef VK_EXT_mesh_shader
TEST_F(DeviceFeatures, TaskAndMeshShareOneNodeAndRespectSubgroupProperties)
{
    desc.enableMeshShader = true;
    desc.enableTaskShaderSubgroupBallot = true;
    desc.preferredTaskSubgroupSize = 32;
    extensions.meshShader = true;
    probe.meshShaderFeatures.meshShader = probe.meshShaderFeatures.taskShader = VK_TRUE;
    probe.vulkan13Features.subgroupSizeControl = VK_TRUE;
    probe.subgroupSizeControlProperties.minSubgroupSize = 32;
    probe.subgroupSizeControlProperties.maxSubgroupSize = 64;
    probe.subgroupSizeControlProperties.requiredSubgroupSizeStages = VK_SHADER_STAGE_TASK_BIT_EXT;
    probe.subgroupProperties.supportedStages = VK_SHADER_STAGE_TASK_BIT_EXT;
    probe.subgroupProperties.supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT;
    auto selected = select();
    EXPECT_TRUE(selected.matches(request()));
    EXPECT_TRUE(selected.supportsTaskSubgroupSize(32));
    EXPECT_FALSE(selected.supportsTaskSubgroupSize(48));
    EXPECT_FALSE(selected.supportsTaskSubgroupSize(128));
    VulkanEnabledFeatureChain chain(selected);
    EXPECT_TRUE(chainTypes(chain.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT));
    EXPECT_EQ(chain.meshShaderFeatures.meshShader, VK_TRUE);
    EXPECT_EQ(chain.meshShaderFeatures.taskShader, VK_TRUE);
    EXPECT_TRUE(extensionNames(selected).contains(VK_EXT_MESH_SHADER_EXTENSION_NAME));
    probe.subgroupProperties.supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT;
    EXPECT_FALSE(select().matches(request()));
    EXPECT_FALSE(select().taskShaderSubgroupBallot);
}
#endif

TEST_F(DeviceFeatures, DGCSecondaryBitDependsOnPrimaryAndRequest)
{
    extensions.deviceGeneratedCommands = true;
    probe.deviceGeneratedCommandsFeatures.dynamicGeneratedPipelineLayout = VK_TRUE;
    EXPECT_FALSE(select().dynamicGeneratedPipelineLayout);
    probe.deviceGeneratedCommandsFeatures.deviceGeneratedCommands = VK_TRUE;
    EXPECT_TRUE(select().dynamicGeneratedPipelineLayout);
    VulkanEnabledFeatureChain chain(select());
    EXPECT_EQ(chain.deviceGeneratedCommandsFeatures.dynamicGeneratedPipelineLayout, VK_TRUE);
    desc.enableDeviceGeneratedCommands = false;
    EXPECT_FALSE(select().deviceGeneratedCommands);
    EXPECT_FALSE(select().dynamicGeneratedPipelineLayout);
}

TEST_F(DeviceFeatures, DecompressionRequiresGDeflateAndDeviceAddress)
{
    extensions.memoryDecompression = true;
    probe.memoryDecompressionFeatures.memoryDecompression = VK_TRUE;
    probe.vulkan12Features.bufferDeviceAddress = VK_TRUE;
    EXPECT_FALSE(select().memoryDecompression);
    probe.decompressionProperties.decompressionMethods = VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT;
    EXPECT_TRUE(select().memoryDecompression);
    probe.vulkan12Features.bufferDeviceAddress = VK_FALSE;
    EXPECT_FALSE(select().memoryDecompression);
}

TEST_F(DeviceFeatures, RebuildingProbeDropsOldNodesAndPropertyLinks)
{
    extensions.memoryDecompression = true;
    extensions.unifiedImageLayouts = true;
    probe.buildChain(extensions);
    EXPECT_TRUE(chainTypes(probe.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR));
    EXPECT_EQ(probe.subgroupSizeControlProperties.pNext, &probe.decompressionProperties);
    probe.buildChain({});
    EXPECT_EQ(chainTypes(probe.features.pNext), (std::set<VkStructureType>{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}));
    EXPECT_EQ(probe.subgroupSizeControlProperties.pNext, nullptr);
}

TEST_F(DeviceFeatures, CoreEnablementAndBackendPublicationKeepUnrelatedCapabilities)
{
    extensions.unifiedImageLayouts = true;
    probe.unifiedImageLayoutsFeatures.unifiedImageLayouts = VK_TRUE;
    probe.vulkan12Features.shaderBufferInt64Atomics = VK_TRUE;
    auto selected = select();
    DeviceCapabilities caps;
    caps.bindlessDescriptorHeap = true; // Published after heap initialization.
    caps.timestampPeriodNanoseconds = 2.5;
    VulkanDeviceCapabilities backendCaps;
    selected.publish(caps, backendCaps);
    EXPECT_TRUE(backendCaps.unifiedImageLayouts);
    EXPECT_TRUE(caps.bindlessDescriptorHeap);
    EXPECT_EQ(caps.timestampPeriodNanoseconds, 2.5);
    EXPECT_FALSE(caps.shaderBufferInt64Atomics);
    probe.features.features.shaderInt64 = VK_TRUE;
    select().publish(caps, backendCaps);
    EXPECT_TRUE(caps.shaderBufferInt64Atomics);
    VulkanEnabledFeatureChain chain(selected);
    EXPECT_EQ(chain.vulkan11Features.shaderDrawParameters, VK_TRUE);
    EXPECT_EQ(chain.vulkan12Features.bufferDeviceAddress, VK_TRUE);
    EXPECT_EQ(chain.vulkan12Features.timelineSemaphore, VK_TRUE);
    EXPECT_EQ(chain.vulkan12Features.hostQueryReset, VK_TRUE);
    EXPECT_EQ(chain.vulkan13Features.dynamicRendering, VK_TRUE);
    EXPECT_EQ(chain.vulkan13Features.synchronization2, VK_TRUE);
    EXPECT_EQ(chain.deviceAddressCommandsFeatures.deviceAddressCommands, VK_TRUE);
    options.preferUnifiedImageLayouts = false;
    select().publish(caps, backendCaps);
    EXPECT_FALSE(backendCaps.unifiedImageLayouts);
}

#if defined(VK_NVX_binary_import) && defined(VK_NVX_image_view_handle)
TEST_F(DeviceFeatures, StreamlineOwnsItsPipelineAndNativeImportExtensions)
{
    desc.enableStreamline = true;
    supportAS();
    extensions.rayQuery = extensions.rayTracingPipeline = extensions.pipelineLibrary = true;
    extensions.pushDescriptor = extensions.streamlineBinaryImport = extensions.streamlineImageViewHandle = true;
    probe.rayQueryFeatures.rayQuery = probe.rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    probe.vulkan13Features.privateData = VK_TRUE;
    auto selected = select();
    EXPECT_TRUE(selected.streamline);
    EXPECT_TRUE(selected.pushDescriptor);
    EXPECT_FALSE(desc.backendExtensions.has_value());
    auto names = extensionNames(selected);
    EXPECT_TRUE(names.contains(VK_NVX_BINARY_IMPORT_EXTENSION_NAME));
    EXPECT_TRUE(names.contains(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME));
    VulkanEnabledFeatureChain chain(selected);
    EXPECT_TRUE(chainTypes(chain.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR));
    EXPECT_EQ(chain.rayTracingPipelineFeatures.rayTracingPipeline, VK_TRUE);
    extensions.streamlineBinaryImport = false;
    selected = select();
    EXPECT_FALSE(selected.streamline);
    EXPECT_FALSE(extensionNames(selected).contains(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME));
    EXPECT_FALSE(extensionNames(selected).contains(VK_NVX_BINARY_IMPORT_EXTENSION_NAME));
    EXPECT_FALSE(extensionNames(selected).contains(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME));
    desc.enableStreamline = false;
    desc.enableRayQuery = true;
    selected = select();
    EXPECT_TRUE(selected.rayQuery);
    EXPECT_FALSE(selected.streamline);
    EXPECT_FALSE(selected.pushDescriptor);
    EXPECT_FALSE(extensionNames(selected).contains(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME));
    EXPECT_FALSE(extensionNames(selected).contains(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME));
    EXPECT_FALSE(extensionNames(selected).contains(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME));
    EXPECT_FALSE(extensionNames(selected).contains(VK_NVX_BINARY_IMPORT_EXTENSION_NAME));
    EXPECT_FALSE(extensionNames(selected).contains(VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME));
    VulkanEnabledFeatureChain queryChain(selected);
    EXPECT_FALSE(chainTypes(queryChain.features.pNext).contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR));
    extensions.pushDescriptor = false;
    EXPECT_TRUE(select().rayQuery);
    desc.enableStreamline = true;
    extensions.streamlineBinaryImport = true;
    EXPECT_FALSE(select().streamline);
}
#endif

#if defined(VK_NV_device_diagnostic_checkpoints) && defined(VK_NV_device_diagnostics_config)
TEST_F(DeviceFeatures, AftermathRequiresInitializationAndBothExtensions)
{
    desc.enableAftermath = true;
    extensions.aftermathDiagnosticCheckpoints = extensions.aftermathDiagnosticsConfig = true;
    probe.diagnosticsConfigFeatures.diagnosticsConfig = VK_TRUE;
    EXPECT_FALSE(select(true, false).aftermath);
    EXPECT_TRUE(select(true, true).aftermath);
    VulkanEnabledFeatureChain chain(select(true, true));
    const auto types = chainTypes(chain.features.pNext);
    EXPECT_TRUE(types.contains(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV));
    EXPECT_TRUE(types.contains(VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV));
    extensions.aftermathDiagnosticCheckpoints = false;
    EXPECT_FALSE(select(true, true).aftermath);
}
#endif

} // namespace
} // namespace metallic::render::vulkan::negotiation

// Intentionally included multiple times. Backend-only feature catalog.
// FEATURE: id, request expression, preferred match, support/dependencies, score,
//          enabled bits, capability publication. Rows are in dependency order.
// SIMPLE also owns its extension name and single-bit pNext structure.
// AS_FEATURE additionally requests acceleration structures when requested itself.
// NODE handles shared/core structures; EXTENSION handles featureless dependencies.
// The public DeviceDesc/Capabilities remain explicit, Vulkan-independent API.
#ifndef MT_VK_FEATURE
#define MT_VK_FEATURE(id, request, preferred, support, score, enable, publish)
#endif
#ifndef MT_VK_EXTENSION
#define MT_VK_EXTENSION(id, name, enabled)
#endif
#ifndef MT_VK_NODE
#define MT_VK_NODE(type, slot, stype, available, enabled)
#endif
#ifndef MT_VK_DEPENDENCY
#define MT_VK_DEPENDENCY(id, prerequisite)
#endif
#define MT_VK_SIMPLE(id, extension, type, stype, member, requested, preferred, dependencies, score, publish) \
    MT_VK_EXTENSION(id, extension, selection.id) \
    MT_VK_NODE(type, id##Features, stype, extensions.id, selection.id) \
    MT_VK_FEATURE(id, requested, preferred, request.id && extensions.id && \
        probe.id##Features.member == VK_TRUE && (dependencies), score, \
        id##Features.member = selection.id;, publish)
#define MT_VK_AS_FEATURE(id, extension, type, stype, member, requested, preferred, dependencies, score, publish) \
    MT_VK_SIMPLE(id, extension, type, stype, member, requested, preferred, dependencies, score, publish) \
    MT_VK_DEPENDENCY(id, rayTracingAccelerationStructure)

MT_VK_SIMPLE(memoryDecompression, VK_EXT_MEMORY_DECOMPRESSION_EXTENSION_NAME,
    VkPhysicalDeviceMemoryDecompressionFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_DECOMPRESSION_FEATURES_EXT,
    memoryDecompression, true, false,
    probe.vulkan12Features.bufferDeviceAddress == VK_TRUE &&
        (probe.decompressionProperties.decompressionMethods & VK_MEMORY_DECOMPRESSION_METHOD_GDEFLATE_1_0_BIT_EXT) != 0, 0, )

MT_VK_FEATURE(deviceGeneratedCommands, desc.enableDeviceGeneratedCommands, false,
    request.deviceGeneratedCommands &&
        extensions.deviceGeneratedCommands &&
        probe.deviceGeneratedCommandsFeatures.deviceGeneratedCommands == VK_TRUE, 0,
    deviceGeneratedCommandsFeatures.deviceGeneratedCommands = selection.deviceGeneratedCommands;, caps.deviceGeneratedCommands = deviceGeneratedCommands;)

MT_VK_FEATURE(dynamicGeneratedPipelineLayout, true, false,
    result.deviceGeneratedCommands && probe.deviceGeneratedCommandsFeatures.dynamicGeneratedPipelineLayout == VK_TRUE, 0,
    deviceGeneratedCommandsFeatures.dynamicGeneratedPipelineLayout = selection.dynamicGeneratedPipelineLayout;, caps.dynamicGeneratedPipelineLayout = dynamicGeneratedPipelineLayout;)

MT_VK_FEATURE(privateData, true, false,
    request.streamline && probe.vulkan13Features.privateData == VK_TRUE, 0,
    vulkan13Features.privateData = selection.privateData;, )

MT_VK_FEATURE(shaderDemoteToHelperInvocation, true, false,
    probe.vulkan13Features.shaderDemoteToHelperInvocation == VK_TRUE, 0,
    vulkan13Features.shaderDemoteToHelperInvocation = selection.shaderDemoteToHelperInvocation;, )

MT_VK_FEATURE(shaderIntegerDotProduct, true, false,
    probe.vulkan13Features.shaderIntegerDotProduct == VK_TRUE, 0,
    vulkan13Features.shaderIntegerDotProduct = selection.shaderIntegerDotProduct;, caps.shaderIntegerDotProduct = shaderIntegerDotProduct;)

#ifdef VK_NV_cooperative_vector
MT_VK_SIMPLE(cooperativeVector, VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME,
    VkPhysicalDeviceCooperativeVectorFeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV,
    cooperativeVector, true, false,
    probe.vulkan12Features.bufferDeviceAddress == VK_TRUE, 16, caps.cooperativeVector = cooperativeVector;)
#else
MT_VK_FEATURE(cooperativeVector, true, false, false, 16, , caps.cooperativeVector = cooperativeVector;)
#endif

MT_VK_FEATURE(bindlessDescriptorHeap, desc.enableBindlessDescriptorHeap, true,
    request.bindlessDescriptorHeap &&
        extensions.descriptorHeap &&
        probe.descriptorHeapFeatures.descriptorHeap == VK_TRUE &&
        probe.vulkan12Features.descriptorIndexing == VK_TRUE &&
        probe.vulkan12Features.runtimeDescriptorArray == VK_TRUE &&
        probe.vulkan12Features.shaderSampledImageArrayNonUniformIndexing == VK_TRUE &&
        probe.vulkan12Features.shaderStorageImageArrayNonUniformIndexing == VK_TRUE &&
        probe.vulkan12Features.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE &&
        probe.vulkan12Features.bufferDeviceAddress == VK_TRUE &&
        descriptorHeapUsable, 16,
    vulkan12Features.descriptorIndexing = selection.bindlessDescriptorHeap;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing = selection.bindlessDescriptorHeap;
    vulkan12Features.shaderStorageImageArrayNonUniformIndexing = selection.bindlessDescriptorHeap;
    vulkan12Features.shaderStorageBufferArrayNonUniformIndexing = selection.bindlessDescriptorHeap;
    vulkan12Features.runtimeDescriptorArray = selection.bindlessDescriptorHeap;
    descriptorHeapFeatures.descriptorHeap = selection.bindlessDescriptorHeap;, )

MT_VK_FEATURE(shaderUntypedPointers, true, false,
    result.bindlessDescriptorHeap && extensions.shaderUntypedPointers && probe.untypedPointerFeatures.shaderUntypedPointers == VK_TRUE, 0,
    untypedPointerFeatures.shaderUntypedPointers = selection.shaderUntypedPointers;, )

MT_VK_SIMPLE(unifiedImageLayouts, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME,
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR,
    unifiedImageLayouts, vulkanOptions.preferUnifiedImageLayouts, false,
    true, 0, backendCaps.unifiedImageLayouts = unifiedImageLayouts;)

MT_VK_SIMPLE(shaderObject, VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
    VkPhysicalDeviceShaderObjectFeaturesEXT, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT,
    shaderObject, desc.enableShaderObject, true,
    true, 8, caps.shaderObject = shaderObject;)

#ifdef VK_EXT_mesh_shader
MT_VK_FEATURE(meshShader, desc.enableMeshShader, true,
    request.meshShader && meshShaderSupported, 8,
    meshShaderFeatures.meshShader = selection.meshShader;, caps.meshShader = meshShader;)
#else
MT_VK_FEATURE(meshShader, desc.enableMeshShader, true, false, 8, , caps.meshShader = meshShader;)
#endif

#ifdef VK_EXT_mesh_shader
MT_VK_FEATURE(taskShader, desc.enableTaskShader || desc.enableTaskShaderSubgroupBallot || desc.preferredTaskSubgroupSize != 0, true,
    request.taskShader && taskShaderSupported, 8,
    meshShaderFeatures.taskShader = selection.taskShader;, caps.taskShader = taskShader;)
#else
MT_VK_FEATURE(taskShader, desc.enableTaskShader || desc.enableTaskShaderSubgroupBallot || desc.preferredTaskSubgroupSize != 0, true, false, 8, , caps.taskShader = taskShader;)
#endif

MT_VK_FEATURE(geometryShader, desc.enableGeometryShader, true,
    request.geometryShader && probe.features.features.geometryShader == VK_TRUE, 0,
    features.features.geometryShader = selection.geometryShader;, caps.geometryShader = geometryShader;)

MT_VK_FEATURE(subgroupSizeControl, desc.enableSubgroupSizeControl || desc.preferredTaskSubgroupSize != 0, true,
    (request.subgroupSizeControl || request.computeFullSubgroups || request.preferredTaskSubgroupSize != 0) && subgroupSizeControlSupported, 8,
    vulkan13Features.subgroupSizeControl = selection.subgroupSizeControl;, caps.subgroupSizeControl = subgroupSizeControl;)

MT_VK_FEATURE(computeFullSubgroups, desc.enableComputeFullSubgroups, true,
    request.computeFullSubgroups && result.subgroupSizeControl && probe.vulkan13Features.computeFullSubgroups == VK_TRUE, 4,
    vulkan13Features.computeFullSubgroups = selection.computeFullSubgroups;, caps.computeFullSubgroups = computeFullSubgroups;)

MT_VK_FEATURE(taskShaderSubgroupBallot, desc.enableTaskShaderSubgroupBallot, true,
    result.taskShader && probe.supportsTaskShaderSubgroupBallot(), 4,
    , caps.taskShaderSubgroupBallot = taskShaderSubgroupBallot;)

MT_VK_FEATURE(taskShaderSubgroupSizeControl, true, false,
    result.taskShader && result.subgroupSizeControl && probe.supportsTaskShaderSubgroupSizeControl(), 4,
    , caps.taskShaderSubgroupSizeControl = taskShaderSubgroupSizeControl;)

MT_VK_FEATURE(computeSubgroupBallotArithmetic, true, false,
    (probe.subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (probe.subgroupProperties.supportedOperations & kMaterialBinningOperations) == kMaterialBinningOperations, 0,
    , caps.computeSubgroupBallotArithmetic = computeSubgroupBallotArithmetic;)

MT_VK_FEATURE(computeSubgroupShuffle, true, false,
    (probe.subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (probe.subgroupProperties.supportedOperations & kShuffleOperations) == kShuffleOperations, 0,
    , caps.computeSubgroupShuffle = computeSubgroupShuffle;)

MT_VK_FEATURE(rayTracingAccelerationStructure, desc.enableRayTracingAccelerationStructure, true,
    (request.rayTracingAccelerationStructure || request.rayQuery || request.streamline) && accelerationStructureSupported, 4,
    accelerationStructureFeatures.accelerationStructure = selection.rayTracingAccelerationStructure;, caps.rayTracingAccelerationStructure = rayTracingAccelerationStructure;)

MT_VK_FEATURE(rayQuery, desc.enableRayQuery, true,
    (request.rayQuery || request.streamline) &&
        accelerationStructureSupported &&
        extensions.rayQuery &&
        probe.rayQueryFeatures.rayQuery == VK_TRUE, 2,
    rayQueryFeatures.rayQuery = selection.rayQuery;, caps.rayQuery = rayQuery;)

// SDK prerequisite, derived by the backend; ordinary ray queries do not need it.
MT_VK_FEATURE(pushDescriptor, false, false,
    request.streamline && extensions.pushDescriptor, 1,
    , backendCaps.pushDescriptor = pushDescriptor;)

MT_VK_FEATURE(opacityMicromap, vulkanOptions.enableOpacityMicromap, false,
    request.opacityMicromap &&
        result.rayTracingAccelerationStructure &&
        extensions.opacityMicromap &&
        (extensions.opacityMicromapExt ? probe.opacityMicromapExtFeatures.micromap == VK_TRUE : probe.opacityMicromapFeatures.micromap == VK_TRUE &&
        probe.deviceAddressCommandsFeatures.deviceAddressCommands == VK_TRUE), 0,
    opacityMicromapFeatures.micromap = selection.opacityMicromap;, backendCaps.opacityMicromap = opacityMicromap;)

MT_VK_FEATURE(opacityMicromapExt, true, false,
    result.opacityMicromap && extensions.opacityMicromapExt, 0,
    opacityMicromapExtFeatures.micromap = selection.opacityMicromapExt;, )

MT_VK_SIMPLE(rayTracingPositionFetch, VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME,
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
    rayTracingPositionFetch, desc.enableRayTracingPositionFetch, false,
    result.rayTracingAccelerationStructure, 0, caps.rayTracingPositionFetch = rayTracingPositionFetch;)

#ifdef VK_NV_cluster_acceleration_structure
MT_VK_AS_FEATURE(clusterAccelerationStructure, VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VkPhysicalDeviceClusterAccelerationStructureFeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV,
    clusterAccelerationStructure, desc.enableClusterAccelerationStructure, true,
    result.rayTracingAccelerationStructure, 64, caps.clusterAccelerationStructure = clusterAccelerationStructure;)
#else
MT_VK_FEATURE(clusterAccelerationStructure, desc.enableClusterAccelerationStructure, true, false, 64, , caps.clusterAccelerationStructure = clusterAccelerationStructure;)
MT_VK_DEPENDENCY(clusterAccelerationStructure, rayTracingAccelerationStructure)
#endif

#ifdef VK_NV_partitioned_acceleration_structure
MT_VK_AS_FEATURE(partitionedAccelerationStructure, VK_NV_PARTITIONED_ACCELERATION_STRUCTURE_EXTENSION_NAME,
    VkPhysicalDevicePartitionedAccelerationStructureFeaturesNV, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PARTITIONED_ACCELERATION_STRUCTURE_FEATURES_NV,
    partitionedAccelerationStructure, desc.enablePartitionedAccelerationStructure, true,
    result.rayTracingAccelerationStructure, 128, caps.partitionedAccelerationStructure = partitionedAccelerationStructure;)
#else
MT_VK_FEATURE(partitionedAccelerationStructure, desc.enablePartitionedAccelerationStructure, true, false, 128, , caps.partitionedAccelerationStructure = partitionedAccelerationStructure;)
MT_VK_DEPENDENCY(partitionedAccelerationStructure, rayTracingAccelerationStructure)
#endif

MT_VK_FEATURE(streamline, desc.enableStreamline, false,
    request.streamline && streamlineSupported && result.rayTracingAccelerationStructure && result.rayQuery && result.pushDescriptor, 32,
    , )

#ifdef VK_NV_device_diagnostics_config
MT_VK_FEATURE(aftermath, desc.enableAftermath && aftermathInitialized, false,
    request.aftermath && aftermathSupported, 0,
    diagnosticsConfigFeatures.diagnosticsConfig = selection.aftermath;, backendCaps.aftermath = aftermath;)
#else
MT_VK_FEATURE(aftermath, desc.enableAftermath && aftermathInitialized, false, false, 0, , backendCaps.aftermath = aftermath;)
#endif

// Opportunistic core bits used by NRC (layouts/fp16/int16) and SHaRC (int64 atomics).
MT_VK_FEATURE(scalarBlockLayout, true, false,
    probe.vulkan12Features.scalarBlockLayout == VK_TRUE, 0,
    vulkan12Features.scalarBlockLayout = selection.scalarBlockLayout;, )

MT_VK_FEATURE(shaderImageGatherExtended, true, false,
    probe.features.features.shaderImageGatherExtended == VK_TRUE, 0,
    features.features.shaderImageGatherExtended = selection.shaderImageGatherExtended;, caps.shaderImageGatherExtended = shaderImageGatherExtended;)

MT_VK_FEATURE(uniformBufferStandardLayout, true, false,
    probe.vulkan12Features.uniformBufferStandardLayout == VK_TRUE, 0,
    vulkan12Features.uniformBufferStandardLayout = selection.uniformBufferStandardLayout;, )

MT_VK_FEATURE(shaderBufferInt64Atomics, true, false,
    probe.vulkan12Features.shaderBufferInt64Atomics == VK_TRUE, 0,
    vulkan12Features.shaderBufferInt64Atomics = selection.shaderBufferInt64Atomics;, caps.shaderBufferInt64Atomics = shaderInt64 &&
        shaderBufferInt64Atomics;)

MT_VK_FEATURE(shaderInt64, true, false,
    probe.features.features.shaderInt64 == VK_TRUE, 0,
    features.features.shaderInt64 = selection.shaderInt64;, )

MT_VK_FEATURE(textureCompressionBC, true, false,
    probe.features.features.textureCompressionBC == VK_TRUE, 0,
    features.features.textureCompressionBC = selection.textureCompressionBC;, )

// NRC barriers include the ray-tracing shader stage even for compute ray queries.
MT_VK_FEATURE(nrcRayTracingPipeline, true, false,
    result.rayQuery && extensions.rayTracingPipeline && probe.rayTracingPipelineFeatures.rayTracingPipeline == VK_TRUE, 0,
    , )

MT_VK_FEATURE(shaderFloat16, true, false,
    probe.vulkan12Features.shaderFloat16 == VK_TRUE, 0,
    vulkan12Features.shaderFloat16 = selection.shaderFloat16;, )

MT_VK_FEATURE(shaderInt16, true, false,
    probe.features.features.shaderInt16 == VK_TRUE, 0,
    features.features.shaderInt16 = selection.shaderInt16;, )

// Core and multi-bit feature structures share one node between probe and enable.
MT_VK_NODE(VkPhysicalDeviceVulkan11Features, vulkan11Features,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, true, true)
MT_VK_NODE(VkPhysicalDeviceVulkan12Features, vulkan12Features,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, true, true)
MT_VK_NODE(VkPhysicalDeviceVulkan13Features, vulkan13Features,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, true, true)
MT_VK_NODE(VkPhysicalDeviceDescriptorHeapFeaturesEXT, descriptorHeapFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT, extensions.descriptorHeap, selection.bindlessDescriptorHeap)
MT_VK_NODE(VkPhysicalDeviceShaderUntypedPointersFeaturesKHR, untypedPointerFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR, extensions.shaderUntypedPointers, selection.shaderUntypedPointers)
MT_VK_NODE(VkPhysicalDeviceAccelerationStructureFeaturesKHR, accelerationStructureFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR, extensions.accelerationStructure, selection.rayTracingAccelerationStructure)
MT_VK_NODE(VkPhysicalDeviceRayQueryFeaturesKHR, rayQueryFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR, extensions.rayQuery, selection.rayQuery)
MT_VK_NODE(VkPhysicalDeviceOpacityMicromapFeaturesKHR, opacityMicromapFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_KHR, extensions.opacityMicromap &&
        !extensions.opacityMicromapExt, selection.opacityMicromap &&
        !selection.opacityMicromapExt)
MT_VK_NODE(VkPhysicalDeviceOpacityMicromapFeaturesEXT, opacityMicromapExtFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT, extensions.opacityMicromap &&
        extensions.opacityMicromapExt, selection.opacityMicromapExt)
MT_VK_NODE(VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT, deviceGeneratedCommandsFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT, extensions.deviceGeneratedCommands, selection.deviceGeneratedCommands)
MT_VK_NODE(VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR, deviceAddressCommandsFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_ADDRESS_COMMANDS_FEATURES_KHR, extensions.deviceAddressCommands, true)
MT_VK_NODE(VkPhysicalDeviceRayTracingPipelineFeaturesKHR, rayTracingPipelineFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR, extensions.rayTracingPipeline, selection.streamline || selection.nrcRayTracingPipeline)
#ifdef VK_NV_device_diagnostics_config
MT_VK_NODE(VkPhysicalDeviceDiagnosticsConfigFeaturesNV, diagnosticsConfigFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DIAGNOSTICS_CONFIG_FEATURES_NV, extensions.aftermathDiagnosticsConfig, selection.aftermath)
#endif
#ifdef VK_EXT_mesh_shader
MT_VK_NODE(VkPhysicalDeviceMeshShaderFeaturesEXT, meshShaderFeatures,
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, extensions.meshShader, selection.meshShader || selection.taskShader)
#endif

// Featureless dependencies and SDK-required pre-promotion extension names.
// NVX imports may be requested by both NRC and Streamline; emit each name once.
MT_VK_EXTENSION(swapchain, VK_KHR_SWAPCHAIN_EXTENSION_NAME, true)
MT_VK_EXTENSION(deviceAddressCommands, VK_KHR_DEVICE_ADDRESS_COMMANDS_EXTENSION_NAME, true)
MT_VK_EXTENSION(deviceGeneratedCommands, VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME, selection.deviceGeneratedCommands)
MT_VK_EXTENSION(descriptorHeap, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, selection.bindlessDescriptorHeap)
MT_VK_EXTENSION(shaderUntypedPointers, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, selection.shaderUntypedPointers)
#ifdef VK_EXT_mesh_shader
MT_VK_EXTENSION(meshShader, VK_EXT_MESH_SHADER_EXTENSION_NAME, selection.meshShader || selection.taskShader)
#endif
MT_VK_EXTENSION(deferredHostOperations, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, selection.rayTracingAccelerationStructure)
MT_VK_EXTENSION(accelerationStructure, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, selection.rayTracingAccelerationStructure)
MT_VK_EXTENSION(rayQuery, VK_KHR_RAY_QUERY_EXTENSION_NAME, selection.rayQuery)
MT_VK_EXTENSION(khrOpacityMicromap, VK_KHR_OPACITY_MICROMAP_EXTENSION_NAME, selection.opacityMicromap && !selection.opacityMicromapExt)
MT_VK_EXTENSION(extOpacityMicromap, VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, selection.opacityMicromapExt)
MT_VK_EXTENSION(pushDescriptor, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, selection.pushDescriptor)
MT_VK_EXTENSION(rayTracingPipeline, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, selection.streamline || selection.nrcRayTracingPipeline)
MT_VK_EXTENSION(pipelineLibrary, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, selection.streamline)
#ifdef VK_NVX_binary_import
MT_VK_EXTENSION(streamlineBinaryImport, VK_NVX_BINARY_IMPORT_EXTENSION_NAME, selection.nvxBinaryImport || selection.streamline)
#endif
#ifdef VK_NVX_image_view_handle
MT_VK_EXTENSION(streamlineImageViewHandle, VK_NVX_IMAGE_VIEW_HANDLE_EXTENSION_NAME, selection.nvxImageViewHandle || selection.streamline)
#endif
#ifdef VK_NV_device_diagnostic_checkpoints
MT_VK_EXTENSION(aftermathDiagnosticCheckpoints, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, selection.aftermath)
#endif
#ifdef VK_NV_device_diagnostics_config
MT_VK_EXTENSION(aftermathDiagnosticsConfig, VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, selection.aftermath)
#endif
MT_VK_EXTENSION(scalarBlockLayout, VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, selection.scalarBlockLayout)
MT_VK_EXTENSION(uniformBufferStandardLayout, VK_KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_EXTENSION_NAME, selection.scalarBlockLayout)
MT_VK_EXTENSION(bufferDeviceAddress, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, selection.usesBufferDeviceAddress())
MT_VK_EXTENSION(memoryBudget, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, selection.usesBufferDeviceAddress())

#undef MT_VK_SIMPLE
#undef MT_VK_AS_FEATURE
#undef MT_VK_DEPENDENCY
#undef MT_VK_FEATURE
#undef MT_VK_EXTENSION
#undef MT_VK_NODE

#include "rhi_shader_utils.h"

#ifdef __APPLE__

#include "metal_shader_utils.h"

RhiVertexDescriptorHandle rhiCreateVertexDescriptor() {
    return RhiVertexDescriptorHandle(metalCreateVertexDescriptor());
}

void rhiVertexDescriptorSetAttribute(const RhiVertexDescriptor& vertexDescriptor,
                                     uint32_t attributeIndex,
                                     RhiVertexFormat format,
                                     uint32_t offset,
                                     uint32_t bufferIndex) {
    metalVertexDescriptorSetAttribute(vertexDescriptor.nativeHandle(),
                                      attributeIndex,
                                      format,
                                      offset,
                                      bufferIndex);
}

void rhiVertexDescriptorSetLayout(const RhiVertexDescriptor& vertexDescriptor,
                                  uint32_t bufferIndex,
                                  uint32_t stride) {
    metalVertexDescriptorSetLayout(vertexDescriptor.nativeHandle(), bufferIndex, stride);
}

RhiShaderLibraryHandle rhiCreateShaderLibraryFromSource(const RhiDevice& device,
                                                        const std::string& source,
                                                        const RhiShaderLibrarySourceDesc& desc,
                                                        std::string& errorMessage) {
    MetalShaderLibraryDesc metalDesc;
    metalDesc.languageVersion = desc.languageVersion;
    return RhiShaderLibraryHandle(
        metalCreateLibraryFromSource(device.nativeHandle(), source, metalDesc, errorMessage));
}

RhiComputePipelineHandle rhiCreateComputePipelineFromLibrary(const RhiDevice& device,
                                                             const RhiShaderLibrary& library,
                                                             const char* entryPoint,
                                                             std::string& errorMessage) {
    return RhiComputePipelineHandle(
        metalCreateComputePipelineFromLibrary(device.nativeHandle(),
                                              library.nativeHandle(),
                                              entryPoint,
                                              errorMessage));
}

RhiGraphicsPipelineHandle rhiCreateRenderPipelineFromSource(const RhiDevice& device,
                                                            const std::string& source,
                                                            const RhiRenderPipelineSourceDesc& desc,
                                                            std::string& errorMessage) {
    MetalRenderPipelineDesc metalDesc;
    metalDesc.vertexEntry = desc.vertexEntry;
    metalDesc.meshEntry = desc.meshEntry;
    metalDesc.fragmentEntry = desc.fragmentEntry;
    metalDesc.colorFormat = desc.colorFormat;
    metalDesc.depthFormat = desc.depthFormat;
    metalDesc.vertexDescriptorHandle = desc.vertexDescriptor ? desc.vertexDescriptor->nativeHandle() : nullptr;
    return RhiGraphicsPipelineHandle(
        metalCreateRenderPipelineFromSource(device.nativeHandle(), source, metalDesc, errorMessage));
}

RhiComputePipelineHandle rhiCreateComputePipelineFromSource(const RhiDevice& device,
                                                            const std::string& source,
                                                            const char* entryPoint,
                                                            std::string& errorMessage) {
    return RhiComputePipelineHandle(
        metalCreateComputePipelineFromSource(device.nativeHandle(), source, entryPoint, errorMessage));
}

#elif defined(_WIN32)

#include "vulkan_backend.h"
#include "vulkan_descriptor_manager.h"
#include "vulkan_frame_graph.h"
#include "vulkan_resource_handles.h"
#include "slang_compiler.h"

#include <array>
#include <cstring>

#include <vulkan/vulkan.h>
#include <spdlog/spdlog.h>

namespace {

// Stored pipeline layout from descriptor manager (set during init)
VkPipelineLayout g_sharedPipelineLayout = VK_NULL_HANDLE;
VkDevice g_shaderDevice = VK_NULL_HANDLE;

VkShaderModule createShaderModuleFromSpirv(VkDevice device, const std::vector<uint32_t>& spirv) {
    VkShaderModuleCreateInfo createInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    VkShaderModule module = VK_NULL_HANDLE;
    VkResult result = vkCreateShaderModule(device, &createInfo, nullptr, &module);
    if (result != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

} // namespace

void vulkanSetShaderContext(VkDevice device, VkPipelineLayout sharedLayout) {
    g_shaderDevice = device;
    g_sharedPipelineLayout = sharedLayout;
}

// Vertex descriptors are not used on Vulkan (vertex input is part of pipeline state)
// but we provide stubs to satisfy the interface
RhiVertexDescriptorHandle rhiCreateVertexDescriptor() {
    return {};
}

void rhiVertexDescriptorSetAttribute(const RhiVertexDescriptor& /*vertexDescriptor*/,
                                     uint32_t /*attributeIndex*/,
                                     RhiVertexFormat /*format*/,
                                     uint32_t /*offset*/,
                                     uint32_t /*bufferIndex*/) {
}

void rhiVertexDescriptorSetLayout(const RhiVertexDescriptor& /*vertexDescriptor*/,
                                  uint32_t /*bufferIndex*/,
                                  uint32_t /*stride*/) {
}

RhiShaderLibraryHandle rhiCreateShaderLibraryFromSource(const RhiDevice& /*device*/,
                                                        const std::string& source,
                                                        const RhiShaderLibrarySourceDesc& /*desc*/,
                                                        std::string& errorMessage) {
    // On Vulkan, the "source" is SPIR-V binary data embedded as a string.
    // This is used by raytracing and native Metal shaders — for Vulkan, those would use
    // a different code path. For now, return empty and log.
    spdlog::warn("rhiCreateShaderLibraryFromSource: Not yet implemented for Vulkan");
    errorMessage = "Shader libraries from source not supported on Vulkan (use SPIR-V pipeline creation)";
    return {};
}

RhiComputePipelineHandle rhiCreateComputePipelineFromLibrary(const RhiDevice& /*device*/,
                                                             const RhiShaderLibrary& /*library*/,
                                                             const char* /*entryPoint*/,
                                                             std::string& errorMessage) {
    spdlog::warn("rhiCreateComputePipelineFromLibrary: Not yet implemented for Vulkan");
    errorMessage = "Compute pipeline from library not supported on Vulkan (use SPIR-V pipeline creation)";
    return {};
}

RhiGraphicsPipelineHandle rhiCreateRenderPipelineFromSource(const RhiDevice& /*device*/,
                                                            const std::string& source,
                                                            const RhiRenderPipelineSourceDesc& desc,
                                                            std::string& errorMessage) {
    if (source.empty()) {
        errorMessage = "Empty shader source";
        return {};
    }

    // The source is SPIR-V binary from Slang — encoded as raw bytes in the string.
    // Reinterpret as uint32_t vector.
    if (source.size() % sizeof(uint32_t) != 0) {
        errorMessage = "SPIR-V source size is not aligned to 4 bytes";
        return {};
    }

    std::vector<uint32_t> spirv(source.size() / sizeof(uint32_t));
    std::memcpy(spirv.data(), source.data(), source.size());

    VkShaderModule shaderModule = createShaderModuleFromSpirv(g_shaderDevice, spirv);
    if (shaderModule == VK_NULL_HANDLE) {
        errorMessage = "Failed to create VkShaderModule from SPIR-V";
        return {};
    }

    // Create pipeline stages
    std::vector<VkPipelineShaderStageCreateInfo> stages;

    bool isMeshShader = (desc.meshEntry != nullptr && desc.meshEntry[0] != '\0');

    if (isMeshShader) {
        VkPipelineShaderStageCreateInfo meshStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        meshStage.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
        meshStage.module = shaderModule;
        meshStage.pName = desc.meshEntry;
        stages.push_back(meshStage);
    } else if (desc.vertexEntry && desc.vertexEntry[0] != '\0') {
        VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStage.module = shaderModule;
        vertStage.pName = desc.vertexEntry;
        stages.push_back(vertStage);
    }

    if (desc.fragmentEntry && desc.fragmentEntry[0] != '\0') {
        VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStage.module = shaderModule;
        fragStage.pName = desc.fragmentEntry;
        stages.push_back(fragStage);
    }

    // Vertex input (empty for mesh shaders and fullscreen passes)
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::array<VkDynamicState, 4> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE
    };
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    bool hasDepth = (desc.depthFormat != RhiFormat::Undefined);
    if (hasDepth) {
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL; // reversed-Z
    }

    // Dynamic rendering
    VkFormat colorFormat = toVkFormat(desc.colorFormat);
    VkFormat depthFormat = hasDepth ? toVkFormat(desc.depthFormat) : VK_FORMAT_UNDEFINED;

    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;
    renderingInfo.depthAttachmentFormat = depthFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = isMeshShader ? nullptr : &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = isMeshShader ? nullptr : &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.pDepthStencilState = hasDepth ? &depthStencil : nullptr;
    pipelineInfo.layout = g_sharedPipelineLayout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = vkCreateGraphicsPipelines(g_shaderDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(g_shaderDevice, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan graphics pipeline (VkResult: " + std::to_string(result) + ")";
        return {};
    }

    auto* res = new VulkanPipelineResource{};
    res->device = g_shaderDevice;
    res->pipeline = pipeline;
    res->layout = g_sharedPipelineLayout;
    res->ownsLayout = false;
    return RhiGraphicsPipelineHandle(res);
}

RhiComputePipelineHandle rhiCreateComputePipelineFromSource(const RhiDevice& /*device*/,
                                                            const std::string& source,
                                                            const char* entryPoint,
                                                            std::string& errorMessage) {
    if (source.empty()) {
        errorMessage = "Empty shader source";
        return {};
    }

    if (source.size() % sizeof(uint32_t) != 0) {
        errorMessage = "SPIR-V source size is not aligned to 4 bytes";
        return {};
    }

    std::vector<uint32_t> spirv(source.size() / sizeof(uint32_t));
    std::memcpy(spirv.data(), source.data(), source.size());

    VkShaderModule shaderModule = createShaderModuleFromSpirv(g_shaderDevice, spirv);
    if (shaderModule == VK_NULL_HANDLE) {
        errorMessage = "Failed to create VkShaderModule from SPIR-V";
        return {};
    }

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = entryPoint ? entryPoint : "computeMain";
    pipelineInfo.layout = g_sharedPipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = vkCreateComputePipelines(g_shaderDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(g_shaderDevice, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan compute pipeline (VkResult: " + std::to_string(result) + ")";
        return {};
    }

    auto* res = new VulkanPipelineResource{};
    res->device = g_shaderDevice;
    res->pipeline = pipeline;
    res->layout = g_sharedPipelineLayout;
    res->ownsLayout = false;
    return RhiComputePipelineHandle(res);
}

#endif

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

#include <algorithm>
#include <array>
#include <cstring>

#include <vulkan/vulkan.h>
#include <spdlog/spdlog.h>

namespace {

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

VkDescriptorType toVkDescriptorType(SlangShaderBindingType type) {
    switch (type) {
    case SlangShaderBindingType::UniformBuffer:
        return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case SlangShaderBindingType::StorageBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case SlangShaderBindingType::SampledTexture:
        return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case SlangShaderBindingType::StorageTexture:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case SlangShaderBindingType::Sampler:
        return VK_DESCRIPTOR_TYPE_SAMPLER;
    case SlangShaderBindingType::AccelerationStructure:
        return VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    default:
        return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }
}

VkFormat toVkVertexFormat(RhiVertexFormat format) {
    switch (format) {
    case RhiVertexFormat::Float2:
        return VK_FORMAT_R32G32_SFLOAT;
    case RhiVertexFormat::Float3:
        return VK_FORMAT_R32G32B32_SFLOAT;
    case RhiVertexFormat::Float4:
        return VK_FORMAT_R32G32B32A32_SFLOAT;
    default:
        return VK_FORMAT_UNDEFINED;
    }
}

void destroyPipelineLayouts(VulkanPipelineResource& resource) {
    for (VkDescriptorSetLayout setLayout : resource.setLayouts) {
        if (setLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(resource.device, setLayout, nullptr);
        }
    }
    resource.setLayouts.clear();

    if (resource.ownsLayout && resource.layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(resource.device, resource.layout, nullptr);
        resource.layout = VK_NULL_HANDLE;
    }
}

bool assignBindingLocation(VulkanDescriptorBindingLocation& location,
                           uint32_t set,
                           uint32_t binding,
                           uint32_t arrayElement,
                           VkDescriptorType descriptorType) {
    location.set = set;
    location.binding = binding;
    location.arrayElement = arrayElement;
    location.descriptorType = descriptorType;
    return true;
}

bool buildPipelineResourceLayout(const void* data,
                                 size_t size,
                                 VulkanPipelineResource& outResource,
                                 std::string& errorMessage) {
    SlangShaderBindingLayout shaderLayout;
    if (!findSlangBindingLayoutForBinary(data, size, shaderLayout)) {
        errorMessage = "Missing cached Slang reflection for SPIR-V shader";
        return false;
    }

    uint32_t maxSetIndex = 0;
    for (const auto& binding : shaderLayout.bindings) {
        maxSetIndex = std::max(maxSetIndex, binding.bindingSpace);
    }

    std::vector<std::vector<VkDescriptorSetLayoutBinding>> perSetBindings(
        shaderLayout.bindings.empty() ? 0 : maxSetIndex + 1);
    std::vector<std::vector<VkDescriptorBindingFlags>> perSetBindingFlags(perSetBindings.size());

    uint32_t logicalBufferIndex = 0;
    uint32_t logicalTextureIndex = 0;
    uint32_t logicalSamplerIndex = 0;

    for (const auto& binding : shaderLayout.bindings) {
        const VkDescriptorType descriptorType = toVkDescriptorType(binding.type);
        if (descriptorType == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
            errorMessage = "Unsupported Slang resource type in Vulkan binding layout";
            destroyPipelineLayouts(outResource);
            return false;
        }

        if (binding.bindingSpace >= perSetBindings.size()) {
            errorMessage = "Invalid descriptor set index in Slang reflection";
            destroyPipelineLayouts(outResource);
            return false;
        }

        auto& setBindings = perSetBindings[binding.bindingSpace];
        auto existingBinding = std::find_if(setBindings.begin(),
                                            setBindings.end(),
                                            [&](const VkDescriptorSetLayoutBinding& vkBinding) {
                                                return vkBinding.binding == binding.bindingIndex;
                                            });
        if (existingBinding == setBindings.end()) {
            VkDescriptorSetLayoutBinding layoutBinding{};
            layoutBinding.binding = binding.bindingIndex;
            layoutBinding.descriptorType = descriptorType;
            layoutBinding.descriptorCount = binding.descriptorCount;
            layoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
            setBindings.push_back(layoutBinding);
            perSetBindingFlags[binding.bindingSpace].push_back(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        }

        auto assignExpandedBinding = [&](auto& locations, uint32_t& logicalIndex) -> bool {
            for (uint32_t arrayElement = 0; arrayElement < binding.descriptorCount; ++arrayElement) {
                if (logicalIndex >= locations.size()) {
                    errorMessage = "Slang reflection exceeded Metallic's Vulkan binding slot limits";
                    return false;
                }
                assignBindingLocation(locations[logicalIndex],
                                      binding.bindingSpace,
                                      binding.bindingIndex,
                                      arrayElement,
                                      descriptorType);
                ++logicalIndex;
            }
            return true;
        };

        bool ok = true;
        switch (binding.type) {
        case SlangShaderBindingType::UniformBuffer:
        case SlangShaderBindingType::StorageBuffer:
            ok = assignExpandedBinding(outResource.bufferBindings, logicalBufferIndex);
            break;
        case SlangShaderBindingType::SampledTexture:
        case SlangShaderBindingType::StorageTexture:
            ok = assignExpandedBinding(outResource.textureBindings, logicalTextureIndex);
            break;
        case SlangShaderBindingType::Sampler:
            ok = assignExpandedBinding(outResource.samplerBindings, logicalSamplerIndex);
            break;
        case SlangShaderBindingType::AccelerationStructure:
            errorMessage = "Acceleration structure reflection is not wired into Vulkan descriptors yet";
            ok = false;
            break;
        }

        if (!ok) {
            destroyPipelineLayouts(outResource);
            return false;
        }
    }

    outResource.setLayouts.resize(perSetBindings.size(), VK_NULL_HANDLE);
    for (size_t setIndex = 0; setIndex < perSetBindings.size(); ++setIndex) {
        auto& bindings = perSetBindings[setIndex];
        if (!bindings.empty()) {
            std::vector<size_t> order(bindings.size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
                return bindings[lhs].binding < bindings[rhs].binding;
            });

            std::vector<VkDescriptorSetLayoutBinding> sortedBindings;
            std::vector<VkDescriptorBindingFlags> sortedFlags;
            sortedBindings.reserve(bindings.size());
            sortedFlags.reserve(bindings.size());
            for (size_t orderedIndex : order) {
                sortedBindings.push_back(bindings[orderedIndex]);
                sortedFlags.push_back(perSetBindingFlags[setIndex][orderedIndex]);
            }
            bindings = std::move(sortedBindings);
            perSetBindingFlags[setIndex] = std::move(sortedFlags);
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
        if (!bindings.empty()) {
            flagsInfo.bindingCount = static_cast<uint32_t>(perSetBindingFlags[setIndex].size());
            flagsInfo.pBindingFlags = perSetBindingFlags[setIndex].data();
            layoutInfo.pNext = &flagsInfo;
            layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
            layoutInfo.pBindings = bindings.data();
        }

        const VkResult result = vkCreateDescriptorSetLayout(g_shaderDevice,
                                                            &layoutInfo,
                                                            nullptr,
                                                            &outResource.setLayouts[setIndex]);
        if (result != VK_SUCCESS) {
            errorMessage = "Failed to create Vulkan descriptor set layout (VkResult: " +
                std::to_string(result) + ")";
            destroyPipelineLayouts(outResource);
            return false;
        }
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(outResource.setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = outResource.setLayouts.empty() ? nullptr : outResource.setLayouts.data();

    const VkResult layoutResult = vkCreatePipelineLayout(g_shaderDevice,
                                                         &pipelineLayoutInfo,
                                                         nullptr,
                                                         &outResource.layout);
    if (layoutResult != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan pipeline layout (VkResult: " +
            std::to_string(layoutResult) + ")";
        destroyPipelineLayouts(outResource);
        return false;
    }

    outResource.ownsLayout = true;
    return true;
}

bool buildVertexInputDescriptions(const RhiVertexDescriptor* vertexDescriptor,
                                  std::vector<VkVertexInputBindingDescription>& outBindings,
                                  std::vector<VkVertexInputAttributeDescription>& outAttributes,
                                  std::string& errorMessage) {
    outBindings.clear();
    outAttributes.clear();

    if (!vertexDescriptor || !vertexDescriptor->nativeHandle()) {
        return true;
    }

    const auto* resource = getVulkanVertexDescriptorResource(*vertexDescriptor);
    if (!resource) {
        errorMessage = "Invalid Vulkan vertex descriptor handle";
        return false;
    }

    outBindings.reserve(resource->bindings.size());
    for (const auto& binding : resource->bindings) {
        if (!binding.valid) {
            continue;
        }

        VkVertexInputBindingDescription vkBinding{};
        vkBinding.binding = binding.binding;
        vkBinding.stride = binding.stride;
        vkBinding.inputRate = binding.inputRate;
        outBindings.push_back(vkBinding);
    }
    std::sort(outBindings.begin(),
              outBindings.end(),
              [](const VkVertexInputBindingDescription& lhs,
                 const VkVertexInputBindingDescription& rhs) {
                  return lhs.binding < rhs.binding;
              });

    outAttributes.reserve(resource->attributes.size());
    for (const auto& attribute : resource->attributes) {
        if (!attribute.valid) {
            continue;
        }

        if (attribute.format == VK_FORMAT_UNDEFINED) {
            errorMessage = "Unsupported Vulkan vertex format";
            return false;
        }

        const bool hasBinding =
            std::any_of(outBindings.begin(),
                        outBindings.end(),
                        [&](const VkVertexInputBindingDescription& binding) {
                            return binding.binding == attribute.binding;
                        });
        if (!hasBinding) {
            errorMessage = "Vertex attribute location " + std::to_string(attribute.location) +
                " references missing vertex layout binding " + std::to_string(attribute.binding);
            return false;
        }

        VkVertexInputAttributeDescription vkAttribute{};
        vkAttribute.location = attribute.location;
        vkAttribute.binding = attribute.binding;
        vkAttribute.format = attribute.format;
        vkAttribute.offset = attribute.offset;
        outAttributes.push_back(vkAttribute);
    }
    std::sort(outAttributes.begin(),
              outAttributes.end(),
              [](const VkVertexInputAttributeDescription& lhs,
                 const VkVertexInputAttributeDescription& rhs) {
                  return lhs.location < rhs.location;
              });

    return true;
}

} // namespace

void vulkanSetShaderContext(VkDevice device) {
    g_shaderDevice = device;
}

RhiVertexDescriptorHandle rhiCreateVertexDescriptor() {
    return RhiVertexDescriptorHandle(new VulkanVertexDescriptorResource{});
}

void rhiVertexDescriptorSetAttribute(const RhiVertexDescriptor& vertexDescriptor,
                                     uint32_t attributeIndex,
                                     RhiVertexFormat format,
                                     uint32_t offset,
                                     uint32_t bufferIndex) {
    auto* resource = getVulkanVertexDescriptorResource(const_cast<RhiVertexDescriptor&>(vertexDescriptor));
    if (!resource) {
        return;
    }

    if (attributeIndex >= resource->attributes.size()) {
        resource->attributes.resize(static_cast<size_t>(attributeIndex) + 1);
    }

    auto& attribute = resource->attributes[attributeIndex];
    attribute.valid = true;
    attribute.location = attributeIndex;
    attribute.binding = bufferIndex;
    attribute.format = toVkVertexFormat(format);
    attribute.offset = offset;
}

void rhiVertexDescriptorSetLayout(const RhiVertexDescriptor& vertexDescriptor,
                                  uint32_t bufferIndex,
                                  uint32_t stride) {
    auto* resource = getVulkanVertexDescriptorResource(const_cast<RhiVertexDescriptor&>(vertexDescriptor));
    if (!resource) {
        return;
    }

    if (bufferIndex >= resource->bindings.size()) {
        resource->bindings.resize(static_cast<size_t>(bufferIndex) + 1);
    }

    auto& binding = resource->bindings[bufferIndex];
    binding.valid = true;
    binding.binding = bufferIndex;
    binding.stride = stride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
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

    auto* res = new VulkanPipelineResource{};
    res->device = g_shaderDevice;
    if (!buildPipelineResourceLayout(source.data(), source.size(), *res, errorMessage)) {
        vkDestroyShaderModule(g_shaderDevice, shaderModule, nullptr);
        delete res;
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

    std::vector<VkVertexInputBindingDescription> vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;
    if (!isMeshShader &&
        !buildVertexInputDescriptions(desc.vertexDescriptor, vertexBindings, vertexAttributes, errorMessage)) {
        vkDestroyShaderModule(g_shaderDevice, shaderModule, nullptr);
        destroyPipelineLayouts(*res);
        delete res;
        return {};
    }

    // Vertex input (empty for mesh shaders and fullscreen passes)
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindings.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindings.empty() ? nullptr : vertexBindings.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributes.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributes.empty() ? nullptr : vertexAttributes.data();
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
    pipelineInfo.layout = res->layout;
    pipelineInfo.renderPass = VK_NULL_HANDLE;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = vkCreateGraphicsPipelines(g_shaderDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(g_shaderDevice, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan graphics pipeline (VkResult: " + std::to_string(result) + ")";
        destroyPipelineLayouts(*res);
        delete res;
        return {};
    }
    res->pipeline = pipeline;
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

    auto* res = new VulkanPipelineResource{};
    res->device = g_shaderDevice;
    if (!buildPipelineResourceLayout(source.data(), source.size(), *res, errorMessage)) {
        vkDestroyShaderModule(g_shaderDevice, shaderModule, nullptr);
        delete res;
        return {};
    }

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = res->layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result = vkCreateComputePipelines(g_shaderDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(g_shaderDevice, shaderModule, nullptr);

    if (result != VK_SUCCESS) {
        errorMessage = "Failed to create Vulkan compute pipeline (VkResult: " + std::to_string(result) + ")";
        destroyPipelineLayouts(*res);
        delete res;
        return {};
    }
    res->pipeline = pipeline;
    return RhiComputePipelineHandle(res);
}

#endif

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

#include "vulkan_resource_handles.h"

#include <vulkan/vulkan.h>
#include <spdlog/spdlog.h>

namespace {

RhiContext* resolveOwningContext(const RhiDevice& device, const char* functionName) {
    RhiContext* context = device.ownerContext();
    if (!context) {
        spdlog::error("{} requires an owning RhiContext on Vulkan", functionName);
        return nullptr;
    }
    if (context->backendType() != RhiBackendType::Vulkan) {
        spdlog::error("{} received a non-Vulkan RhiContext", functionName);
        return nullptr;
    }
    return context;
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

} // namespace

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

RhiShaderLibraryHandle rhiCreateShaderLibraryFromSource(const RhiDevice& device,
                                                        const std::string& source,
                                                        const RhiShaderLibrarySourceDesc& desc,
                                                        std::string& errorMessage) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateShaderLibraryFromSource");
    return context ? context->createShaderLibraryFromSource(source, desc, errorMessage)
                   : RhiShaderLibraryHandle{};
}

RhiComputePipelineHandle rhiCreateComputePipelineFromLibrary(const RhiDevice& device,
                                                             const RhiShaderLibrary& library,
                                                             const char* entryPoint,
                                                             std::string& errorMessage) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateComputePipelineFromLibrary");
    return context ? context->createComputePipelineFromLibrary(library, entryPoint, errorMessage)
                   : RhiComputePipelineHandle{};
}

RhiGraphicsPipelineHandle rhiCreateRenderPipelineFromSource(const RhiDevice& device,
                                                            const std::string& source,
                                                            const RhiRenderPipelineSourceDesc& desc,
                                                            std::string& errorMessage) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateRenderPipelineFromSource");
    return context ? context->createRenderPipelineFromSource(source, desc, errorMessage)
                   : RhiGraphicsPipelineHandle{};
}

RhiComputePipelineHandle rhiCreateComputePipelineFromSource(const RhiDevice& device,
                                                            const std::string& source,
                                                            const char* entryPoint,
                                                            std::string& errorMessage) {
    RhiContext* context = resolveOwningContext(device, "rhiCreateComputePipelineFromSource");
    return context ? context->createComputePipelineFromSource(source, entryPoint, errorMessage)
                   : RhiComputePipelineHandle{};
}

#endif

#include "rhi_shader_utils.h"

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

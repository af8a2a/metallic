#pragma once

#include <cstdint>
#include <string>

#include "rhi_backend.h"

RhiVertexDescriptorHandle rhiCreateVertexDescriptor();
void rhiVertexDescriptorSetAttribute(const RhiVertexDescriptor& vertexDescriptor,
                                     uint32_t attributeIndex,
                                     RhiVertexFormat format,
                                     uint32_t offset,
                                     uint32_t bufferIndex);
void rhiVertexDescriptorSetLayout(const RhiVertexDescriptor& vertexDescriptor,
                                  uint32_t bufferIndex,
                                  uint32_t stride);

struct RhiRenderPipelineSourceDesc {
    const char* vertexEntry = nullptr;
    const char* meshEntry = nullptr;
    const char* fragmentEntry = nullptr;
    RhiFormat colorFormat = RhiFormat::BGRA8Unorm;
    RhiFormat depthFormat = RhiFormat::Undefined;
    const RhiVertexDescriptor* vertexDescriptor = nullptr;
};

struct RhiShaderLibrarySourceDesc {
    uint32_t languageVersion = 0;
};

RhiShaderLibraryHandle rhiCreateShaderLibraryFromSource(const RhiDevice& device,
                                                        const std::string& source,
                                                        const RhiShaderLibrarySourceDesc& desc,
                                                        std::string& errorMessage);
RhiComputePipelineHandle rhiCreateComputePipelineFromLibrary(const RhiDevice& device,
                                                             const RhiShaderLibrary& library,
                                                             const char* entryPoint,
                                                             std::string& errorMessage);
RhiGraphicsPipelineHandle rhiCreateRenderPipelineFromSource(const RhiDevice& device,
                                                            const std::string& source,
                                                            const RhiRenderPipelineSourceDesc& desc,
                                                            std::string& errorMessage);
RhiComputePipelineHandle rhiCreateComputePipelineFromSource(const RhiDevice& device,
                                                            const std::string& source,
                                                            const char* entryPoint,
                                                            std::string& errorMessage);

#ifdef _WIN32
#include <vulkan/vulkan.h>
void vulkanSetShaderContext(VkDevice device);
#endif

#pragma once

#include <cstdint>
#include <string>

#include "rhi_backend.h"

void* metalCreateVertexDescriptor();
void metalVertexDescriptorSetAttribute(void* vertexDescriptorHandle,
                                       uint32_t attributeIndex,
                                       RhiVertexFormat format,
                                       uint32_t offset,
                                       uint32_t bufferIndex);
void metalVertexDescriptorSetLayout(void* vertexDescriptorHandle,
                                    uint32_t bufferIndex,
                                    uint32_t stride);

struct MetalRenderPipelineDesc {
    const char* vertexEntry = nullptr;
    const char* meshEntry = nullptr;
    const char* fragmentEntry = nullptr;
    RhiFormat colorFormat = RhiFormat::BGRA8Unorm;
    RhiFormat depthFormat = RhiFormat::Undefined;
    void* vertexDescriptorHandle = nullptr;
};

struct MetalShaderLibraryDesc {
    uint32_t languageVersion = 0;
};

void* metalCreateLibraryFromSource(void* deviceHandle,
                                   const std::string& source,
                                   const MetalShaderLibraryDesc& desc,
                                   std::string& errorMessage);
void* metalCreateComputePipelineFromLibrary(void* deviceHandle,
                                            void* libraryHandle,
                                            const char* entryPoint,
                                            std::string& errorMessage);
void* metalCreateRenderPipelineFromSource(void* deviceHandle,
                                          const std::string& source,
                                          const MetalRenderPipelineDesc& desc,
                                          std::string& errorMessage);
void* metalCreateComputePipelineFromSource(void* deviceHandle,
                                           const std::string& source,
                                           const char* entryPoint,
                                           std::string& errorMessage);

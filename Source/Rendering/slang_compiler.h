#pragma once

#include "rhi_backend.h"

#include <string>
#include <vector>

// Backend-aware Slang compilation helpers.
// Source output is currently used by the Metal path.
// SPIR-V output is currently used by the Vulkan path.

std::string compileSlangGraphicsSource(RhiBackendType backend,
                                       const char* shaderPath,
                                       const char* searchPath = nullptr);
std::string compileSlangMeshSource(RhiBackendType backend,
                                   const char* shaderPath,
                                   const char* searchPath = nullptr);
std::string compileSlangComputeSource(RhiBackendType backend,
                                      const char* shaderPath,
                                      const char* searchPath = nullptr,
                                      const char* entryPoint = "computeMain");

std::vector<uint32_t> compileSlangGraphicsBinary(RhiBackendType backend,
                                                 const char* shaderPath,
                                                 const char* searchPath = nullptr);
std::vector<uint32_t> compileSlangMeshBinary(RhiBackendType backend,
                                             const char* shaderPath,
                                             const char* searchPath = nullptr);
std::vector<uint32_t> compileSlangComputeBinary(RhiBackendType backend,
                                                const char* shaderPath,
                                                const char* searchPath = nullptr,
                                                const char* entryPoint = "computeMain");

// Backend-specific Slang output workarounds.
std::string patchMeshShaderSource(RhiBackendType backend, const std::string& source);
std::string patchVisibilityShaderSource(RhiBackendType backend, const std::string& source);
std::string patchComputeShaderSource(RhiBackendType backend, const std::string& source);

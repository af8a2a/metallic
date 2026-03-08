#pragma once

#include <string>
#include <vector>

// Slang → Metal source compilation utilities.
// All functions return the generated Metal source string, or empty on failure.

std::string compileSlangToMetal(const char* shaderPath, const char* searchPath = nullptr);
std::string compileSlangMeshShaderToMetal(const char* shaderPath, const char* searchPath = nullptr);
std::string compileSlangComputeShaderToMetal(const char* shaderPath, const char* searchPath = nullptr, const char* entryPoint = "computeMain");

std::vector<uint32_t> compileSlangToSpirv(const char* shaderPath, const char* searchPath = nullptr);
std::vector<uint32_t> compileSlangMeshShaderToSpirv(const char* shaderPath, const char* searchPath = nullptr);
std::vector<uint32_t> compileSlangComputeShaderToSpirv(const char* shaderPath,
                                                       const char* searchPath = nullptr,
                                                       const char* entryPoint = "computeMain");

// Slang bug workarounds — patch generated Metal source before compilation.
std::string patchMeshShaderMetalSource(const std::string& source);
std::string patchVisibilityShaderMetalSource(const std::string& source);
std::string patchComputeShaderMetalSource(const std::string& source);

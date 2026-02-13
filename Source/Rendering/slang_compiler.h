#pragma once

#include <string>

// Slang → Metal source compilation utilities.
// All functions return the generated Metal source string, or empty on failure.

std::string compileSlangToMetal(const char* shaderPath, const char* searchPath = nullptr);
std::string compileSlangMeshShaderToMetal(const char* shaderPath, const char* searchPath = nullptr);
std::string compileSlangComputeShaderToMetal(const char* shaderPath, const char* searchPath = nullptr);

// Slang bug workarounds — patch generated Metal source before compilation.
std::string patchMeshShaderMetalSource(const std::string& source);
std::string patchVisibilityShaderMetalSource(const std::string& source);
std::string patchComputeShaderMetalSource(const std::string& source);

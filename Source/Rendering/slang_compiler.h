#pragma once

#include "rhi_backend.h"

#include <cstddef>
#include <cstdint>
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

struct SlangDiagnosticRecord {
    std::string stage;
    std::string shaderPath;
    std::string message;
};

std::vector<SlangDiagnosticRecord> getRecentSlangDiagnostics();
void clearRecentSlangDiagnostics();

enum class SlangShaderBindingType : uint8_t {
    UniformBuffer,
    StorageBuffer,
    SampledTexture,
    StorageTexture,
    Sampler,
    AccelerationStructure,
    PushConstantBuffer,
};

struct SlangShaderBindingDesc {
    SlangShaderBindingType type = SlangShaderBindingType::StorageBuffer;
    uint32_t bindingIndex = 0;
    uint32_t bindingSpace = 0;
    uint32_t descriptorCount = 1;
    std::string name;
};

struct SlangShaderBindingLayout {
    std::vector<SlangShaderBindingDesc> bindings;
};

bool findSlangBindingLayoutForBinary(const void* data,
                                     size_t size,
                                     SlangShaderBindingLayout& outLayout);

// Backend-specific Slang output workarounds.
std::string patchMeshShaderSource(RhiBackendType backend, const std::string& source);
std::string patchVisibilityShaderSource(RhiBackendType backend, const std::string& source);
std::string patchComputeShaderSource(RhiBackendType backend, const std::string& source);

// Set the directory used to cache compiled SPIR-V binaries between runs.
// Must be called before any compileSlang*Binary calls. Empty string disables the cache.
void setSlangShaderCacheDir(const std::string& dir);

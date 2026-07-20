#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <string>
#include <vector>

namespace metallic::render {

inline constexpr const char* kDefaultSlangProfileName = "spirv_1_6";

struct SlangMacroDefine {
    const char* name = nullptr;
    const char* value = "1";
};

struct SlangShaderDesc {
    const char* moduleName = nullptr;
    const char* entryPointName = nullptr;
    const char* searchPath = nullptr;
    const char* profileName = kDefaultSlangProfileName;
    const char* const* capabilities = nullptr;
    uint32_t capabilityCount = 0;
    const SlangMacroDefine* macroDefines = nullptr;
    uint32_t macroDefineCount = 0;
};

struct ShaderCompileResult {
    std::vector<uint32_t> spirv;
    std::string diagnostics;
};

struct SlangShaderCacheOptions {
    // Null uses the project-local .cache/shaders/spirv directory.
    const char* cacheDirectory = nullptr;
    bool enableDiskCache = true;
    bool* outCacheHit = nullptr;
};

Result compileSlangShaderToSpirv(const SlangShaderDesc& desc, ShaderCompileResult& outResult);
Result compileSlangShaderToSpirv(
    const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& cacheOptions,
    ShaderCompileResult& outResult);

} // namespace metallic::render

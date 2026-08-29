#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <string>
#include <vector>

namespace metallic::render {

inline constexpr const char* kDefaultSlangProfileName = "spirv_1_6";

enum class SlangShaderDebugMode : uint8_t {
    Disabled,
    CaptureSymbols,
    ShaderDebug,
};

struct SlangMacroDefine {
    const char* name = nullptr;
    const char* value = "1";
};

struct SlangShaderDesc {
    const char* moduleName = nullptr;
    const char* entryPointName = nullptr;
    const char* searchPath = nullptr;
    const char* const* additionalSearchPaths = nullptr;
    uint32_t additionalSearchPathCount = 0;
    const char* profileName = kDefaultSlangProfileName;
    const char* const* capabilities = nullptr;
    uint32_t capabilityCount = 0;
    const SlangMacroDefine* macroDefines = nullptr;
    uint32_t macroDefineCount = 0;
};

struct ShaderCompileResult {
    std::vector<uint32_t> spirv;
    std::string diagnostics;
    std::vector<std::string> dependencies;
};

struct SlangShaderCacheOptions {
    // Null uses the project-local .cache/shaders/spirv directory.
    const char* cacheDirectory = nullptr;
    bool enableDiskCache = true;
    bool* outCacheHit = nullptr;
};

// Process-global compilation policy. Configure it before shader compilation begins.
void setSlangShaderDebugMode(SlangShaderDebugMode mode) noexcept;
SlangShaderDebugMode slangShaderDebugMode() noexcept;

// Successful shader compiles automatically register their complete Slang
// dependency list. Poll this from the interactive frame loop to detect stable
// source edits, including edits to included files. Passing zero disables the
// save debounce and is useful for deterministic tests.
std::vector<std::string> pollSlangShaderChanges(
    uint32_t debounceMilliseconds = 150,
    uint32_t retryMilliseconds = 1000);
// Accept the most recently reported source snapshots after every affected
// pipeline has been committed. Failed reloads intentionally stay dirty and are
// reported again after retryMilliseconds, so fixing a newly-added include is
// sufficient to recover without touching an already-known file.
void acknowledgeSlangShaderChanges();
void resetSlangShaderHotReloadTracking();

Result compileSlangShaderToSpirv(const SlangShaderDesc& desc, ShaderCompileResult& outResult);
Result compileSlangShaderToSpirv(
    const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& cacheOptions,
    ShaderCompileResult& outResult);

} // namespace metallic::render

#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <string>
#include <vector>

namespace metallic::render {

struct SlangShaderDesc {
    const char* moduleName = nullptr;
    const char* entryPointName = nullptr;
    const char* searchPath = nullptr;
    const char* profileName = "glsl_450";
    const char* const* capabilities = nullptr;
    uint32_t capabilityCount = 0;
};

struct ShaderCompileResult {
    std::vector<uint32_t> spirv;
    std::string diagnostics;
};

Result compileSlangShaderToSpirv(const SlangShaderDesc& desc, ShaderCompileResult& outResult);

} // namespace metallic::render

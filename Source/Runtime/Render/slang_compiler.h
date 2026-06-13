#pragma once

#include "Runtime/Render/GAPI/rhi.h"

#include <string>
#include <vector>

namespace metallic::render {

struct SlangShaderDesc {
    const char* moduleName = nullptr;
    const char* entryPointName = nullptr;
    const char* searchPath = nullptr;
};

struct ShaderCompileResult {
    std::vector<uint32_t> spirv;
    std::string diagnostics;
};

Result compileSlangShaderToSpirv(const SlangShaderDesc& desc, ShaderCompileResult& outResult);

} // namespace metallic::render

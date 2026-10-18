#pragma once

#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"

namespace metallic::render {

inline uint64_t materialCoverageShaderKey(const MaterialValueProgramSet* programs)
{
    return programs && programs->coverageProgramCount() != 0 ? programs->coverageKey() : 0;
}

inline bool prepareMaterialCoverageShader(const MaterialValueProgramSet* programs,
    SceneShaderOptions& options, std::string& diagnostics)
{
    options.customCoverage = programs && programs->coverageProgramCount() != 0;
    options.coverageInclude.clear();
    if (!options.customCoverage) { return true; }
    std::filesystem::path directory;
    if (!programs->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", directory, diagnostics)) {
        return false;
    }
    options.coverageInclude = directory.string();
    return true;
}

} // namespace metallic::render

#pragma once
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"

namespace metallic::render {

// Compiled built-in target. These resource declarations are consumed by the
// checked ComputeProgram encoder, not merely advisory compiler metadata.
struct MaterialExecutableArtifact
{
    uint64_t key = 0;
    uint64_t parameterABI = kLegacyMaterialABI;
    uint32_t constantsSize = 0;
    std::vector<ComputeProgramBindingDesc> resources;
    ShaderCompileResult shader;
};

// Compilation and pipeline creation are transactional. Prepared dispatches
// retain old executables/parameters through the existing frame completion.
Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ComputeProgramDesc& layout, ComputeProgram& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log);
Result<> initializeMaterialErrorProgram(Device& device, ComputeProgram& program, std::string& log);

} // namespace metallic::render

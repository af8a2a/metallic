#pragma once
#include "Runtime/Render/Core/MaterialErrorParameters.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"

namespace metallic::render {

// Compiled built-in target. Legacy resources are consumed by ComputeProgram;
// typed targets carry their parameter ABI and leave the legacy resource list empty.
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
Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ComputeKernelDesc& layout, ComputeKernel& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log);
Result<> initializeMaterialErrorKernel(Device& device, ComputeKernel& program, std::string& log);
Result<> dispatchMaterialError(Device& device, const ComputeKernel& program, CommandBuffer& commands,
    TextureView& output, uint32_t width, uint32_t height, bool color);

} // namespace metallic::render

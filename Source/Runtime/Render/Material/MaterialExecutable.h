#pragma once
#include "Runtime/Render/Core/MaterialErrorParameters.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/SlangCompiler.h"

namespace metallic::render {

// Compiled built-in target. The complete ABI identifies parameter layout and transport.
struct MaterialExecutableArtifact
{
    uint64_t key = 0;
    ParameterABI parameters;
    ShaderCompileResult shader;
};

// Compilation and pipeline creation are transactional. Prepared dispatches
// retain old executables/parameters through the existing frame completion.
Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ComputeKernelDesc& layout, ComputeKernel& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log);
Result<> initializeMaterialErrorKernel(Device& device, ComputeKernel& program, std::string& log);
Result<> dispatchMaterialError(Device& device, const ComputeKernel& program, CommandBuffer& commands,
    TextureView& output, uint32_t width, uint32_t height, bool color);

} // namespace metallic::render

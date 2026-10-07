#pragma once
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Core/ComputeResourceEncoder.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Core/ShaderRequests.h"

namespace metallic::render {

// CPU compiler choices, not GPU material IDs. Mixed-program scene binning is
// a separate scheduling layer; there is no instance-based shader dispatch.
enum class SurfaceMaterialImplementation { OpenPBR, DebugLambert, DebugMirror };
ShaderRequest specializeSurfaceMaterialProgram(ShaderRequest request, SurfaceMaterialImplementation implementation);

// Compiled built-in target. These resource declarations are consumed by the
// checked resource encoder, not merely advisory compiler metadata.
struct MaterialExecutableArtifact
{
    uint64_t key = 0;
    MaterialProgramKey programKey;
    uint64_t generation = 0;
    uint64_t parameterABI = kLegacyMaterialABI;
    uint32_t constantsSize = 0;
    std::vector<ComputeResourceBindingDesc> resources;
    ShaderCompileResult shader;
    // Immutable once published through shared_ptr<const ...>. This handle and
    // every pass share a single underlying kernel, never one per instance.
    ComputeKernel executable;
    ComputeResourceEncoder encoder;
    std::vector<ComputeResourceField> resourceFields;
    uint32_t resourceParameterSize = 0;
    bool requiresRayQuery = false;
};

struct MaterialProgramCompileOptions
{
    uint64_t definitionHash = 0; // zero derives identity from module + entry
    MaterialDomain domain = MaterialDomain::Surface;
    uint64_t qualityProfile = 0;
    uint64_t parameterABI = kLegacyMaterialABI;
};

struct MaterialProgramCacheStats
{
    uint64_t hits = 0;
    uint64_t pipelineBuilds = 0;
    uint64_t liveEntries = 0;
};
MaterialProgramCacheStats materialProgramCacheStats();

// Compilation and pipeline creation are transactional. Prepared dispatches
// retain old executables/parameters through the existing frame completion.
Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ResourceComputeKernelDesc& layout, ComputeKernel& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log,
    MaterialProgramCompileOptions options = {});
Result<> initializeMaterialErrorProgram(Device& device, ComputeKernel& program, ComputeResourceEncoder& encoder, std::string& log);

} // namespace metallic::render

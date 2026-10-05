#include "Runtime/Render/Core/NamedResourceLayouts.h"
#include "Runtime/Render/Material/MaterialExecutable.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <algorithm>
#include <mutex>
#include <stdexcept>

namespace metallic::render {

ShaderRequest specializeSurfaceMaterialProgram(ShaderRequest request, SurfaceMaterialImplementation implementation)
{
    const char* concreteType = nullptr;
    switch (implementation) {
    case SurfaceMaterialImplementation::OpenPBR: concreteType = "OpenPBRMaterialProgram"; break;
    case SurfaceMaterialImplementation::DebugLambert: concreteType = "DebugLambertMaterialProgram"; break;
    case SurfaceMaterialImplementation::DebugMirror: concreteType = "DebugMirrorMaterialProgram"; break;
    }
    if (!concreteType) { throw std::invalid_argument("Unknown Surface Material implementation"); }
    std::erase_if(request.defines, [](const auto& define) { return define.first == "METALLIC_SURFACE_PROGRAM"; });
    request.defines.emplace_back("METALLIC_SURFACE_PROGRAM", concreteType);
    return request;
}

namespace {
struct ProgramCacheEntry
{
    const void* device;
    MaterialProgramKey key;
    std::weak_ptr<const MaterialExecutableArtifact> artifact;
};
std::mutex programCacheMutex;
std::vector<ProgramCacheEntry> programCache;
uint64_t cacheHits = 0, pipelineBuilds = 0, nextProgramGeneration = 1;
struct ProgramHash
{
    uint64_t value = 14695981039346656037ull;
    void add(uint64_t component)
    {
        for (uint32_t byte = 0; byte < 8; ++byte) {
            value = (value ^ ((component >> (byte * 8)) & 255u)) * 1099511628211ull;
        }
    }
    void add(std::string_view text)
    {
        add(text.size());
        for (unsigned char c : text) { add(c); }
    }
};
} // namespace

MaterialProgramCacheStats materialProgramCacheStats()
{
    std::lock_guard lock(programCacheMutex);
    std::erase_if(programCache, [](const auto& entry) { return entry.artifact.expired(); });
    return {cacheHits, pipelineBuilds, programCache.size()};
}

Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ComputeProgramDesc& layout, ComputeProgram& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log,
    MaterialProgramCompileOptions options)
{
    // Always let Slang validate dependencies (its disk cache does the cheap
    // unchanged-source path). Source edits cannot hit a stale executable here.
    auto candidate = std::make_shared<MaterialExecutableArtifact>();
    candidate->parameterABI = options.parameterABI;
    auto compiled = ShaderRegistry::instance().getShader(source, log);
    if (!compiled) { return makeError(compiled.error()); }
    candidate->shader = std::move(*compiled);
    candidate->resources.assign(layout.bindings.begin(), layout.bindings.end());
    candidate->resourceFields.assign(layout.resourceParameters.fields.begin(), layout.resourceParameters.fields.end());
    candidate->resourceParameterSize = layout.resourceParameters.size;
    candidate->requiresRayQuery = layout.requiresRayQuery;
    candidate->constantsSize = layout.pushConstantSize;
    ProgramHash definition, ir, specialization, capabilities;
    definition.add(source.moduleName ? source.moduleName : "");
    definition.add(source.entryPointName ? source.entryPointName : "");
    definition.add(candidate->parameterABI);
    for (auto word : candidate->shader.spirv) { ir.add(word); }
    specialization.add(definition.value);
    specialization.add(source.profileName ? source.profileName : "");
    specialization.add(uint64_t(slangShaderDebugMode()));
    for (const auto& define : source.macroDefines) {
        specialization.add(define.name ? define.name : "");
        specialization.add(define.value ? define.value : "");
    }
    specialization.add(candidate->constantsSize);
    specialization.add(layout.resourceParameters.size);
    for (const auto& field : layout.resourceParameters.fields) {
        specialization.add(field.binding); specialization.add(uint64_t(field.kind));
        specialization.add(field.offset); specialization.add(uint64_t(field.format));
    }
    for (const auto& binding : layout.bindings) {
        specialization.add(binding.binding); specialization.add(uint64_t(binding.kind));
        specialization.add(binding.descriptorCount); specialization.add(binding.dataStride);
        specialization.add(binding.dataAlignment);
    }
    capabilities.add(layout.requiresRayQuery);
    capabilities.add(uint64_t(source.descriptorHeapMode));
    for (const auto* capability : source.capabilities) { capabilities.add(capability ? capability : ""); }
    candidate->programKey = {.definitionHash = options.definitionHash ? options.definitionHash : definition.value,
        .irHash = ir.value, .specializationSignature = specialization.value, .domain = options.domain,
        .qualityProfile = options.qualityProfile, .targetCapabilities = capabilities.value};
    ProgramHash combined;
    combined.add(candidate->programKey.definitionHash); combined.add(ir.value); combined.add(specialization.value);
    combined.add(uint64_t(options.domain)); combined.add(options.qualityProfile); combined.add(capabilities.value);
    candidate->key = combined.value;

    // Weak ownership prevents cached Vulkan objects from outliving their Device.
    // Serialize lookup/build/publication so parallel identical requests create
    // exactly one pipeline. Failed candidates never replace a published handle.
    std::lock_guard lock(programCacheMutex);
    std::erase_if(programCache, [](const auto& entry) { return entry.artifact.expired(); });
    for (const auto& entry : programCache) {
        if (entry.device != device.identity() || entry.key != candidate->programKey) { continue; }
        const auto cached = entry.artifact.lock();
        // Check full code/layout as well, so a hash collision cannot alias code.
        if (!cached || cached->shader.spirv != candidate->shader.spirv || cached->resources != candidate->resources ||
            cached->resourceFields != candidate->resourceFields || cached->constantsSize != candidate->constantsSize ||
            cached->resourceParameterSize != candidate->resourceParameterSize || cached->requiresRayQuery != candidate->requiresRayQuery) {
            continue;
        }
        program = cached->executable.share();
        artifact = cached;
        ++cacheHits;
        return {};
    }
    auto description = layout;
    description.spirv = candidate->shader.spirv;
    description.bindings = candidate->resources;
    auto result = candidate->executable.initialize(device, description, log);
    if (!result) { return result; }
    candidate->generation = nextProgramGeneration++;
    ++pipelineBuilds;
    program = candidate->executable.share();
    artifact = candidate;
    programCache.push_back({device.identity(), candidate->programKey, candidate});
    return {};
}

Result<> initializeMaterialErrorProgram(Device& device, ComputeProgram& program, std::string& log)
{
    const ComputeProgramBindingDesc output{.binding = 0, .kind = ComputeResourceBindingKind::StorageImage};
    std::shared_ptr<const MaterialExecutableArtifact> artifact;
    return compileMaterialExecutable(device,
        {.moduleName = "Features/Material/MaterialError", .entryPointName = "materialErrorMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.pushConstantSize = 4, .bindings = {&output, 1}, .requiresRayQuery = false, .resourceParameters = kOutputImageResourceLayout},
        program, artifact, log);
}

} // namespace metallic::render

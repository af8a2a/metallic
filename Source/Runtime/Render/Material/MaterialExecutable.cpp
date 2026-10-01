#include "Runtime/Render/Material/MaterialExecutable.h"

namespace metallic::render {

Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ComputeProgramDesc& layout, ComputeProgram& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log)
{
    auto candidate = std::make_shared<MaterialExecutableArtifact>();
    auto compiled = compileSlangShaderToSpirv(source, log);
    if (!compiled) { return makeError(compiled.error()); }
    candidate->shader = std::move(*compiled);
    candidate->resources.assign(layout.bindings.begin(), layout.bindings.end());
    candidate->constantsSize = layout.pushConstantSize;
    auto description = layout;
    description.spirv = candidate->shader.spirv;
    description.bindings = candidate->resources;
    ComputeProgram executable;
    auto result = executable.initialize(device, description, log);
    if (!result) { return result; }
    uint64_t key = 14695981039346656037ull;
    const auto hash = [&](uint64_t value) {
        for (uint32_t byte = 0; byte < 8; ++byte) {
            key = (key ^ ((value >> (byte * 8)) & 255u)) * 1099511628211ull;
        }
    };
    hash(candidate->parameterABI);
    hash(candidate->constantsSize);
    hash(layout.requiresRayQuery);
    for (auto word : candidate->shader.spirv) { hash(word); }
    for (const auto& resource : candidate->resources) {
        hash(resource.binding); hash(uint64_t(resource.kind)); hash(resource.descriptorCount);
        hash(resource.dataStride); hash(resource.dataAlignment);
    }
    candidate->key = key;
    program = std::move(executable);
    artifact = std::move(candidate);
    return {};
}

Result<> initializeMaterialErrorProgram(Device& device, ComputeProgram& program, std::string& log)
{
    const ComputeProgramBindingDesc output{.binding = 0, .kind = ComputeResourceBindingKind::StorageImage};
    std::shared_ptr<const MaterialExecutableArtifact> artifact;
    return compileMaterialExecutable(device,
        {.moduleName = "Features/Material/MaterialError", .entryPointName = "materialErrorMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        {.pushConstantSize = 4, .bindings = {&output, 1}, .requiresRayQuery = false},
        program, artifact, log);
}

} // namespace metallic::render

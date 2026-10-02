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

Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ComputeKernelDesc& layout, ComputeKernel& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log)
{
    auto compiled = compileSlangShaderToSpirv(source, log);
    if (!compiled) { return makeError(compiled.error()); }
    auto candidate = std::make_shared<MaterialExecutableArtifact>();
    candidate->shader = std::move(*compiled);
    candidate->parameterABI = layout.parameters.id;
    candidate->constantsSize = layout.parameters.size;
    auto description = layout;
    description.spirv = candidate->shader.spirv;
    ComputeKernel executable;
    auto result = executable.initialize(device, description, log);
    if (!result) { return result; }
    uint64_t key = 14695981039346656037ull;
    const auto hash = [&](uint64_t value) {
        for (uint32_t byte = 0; byte < 8; ++byte) {
            key = (key ^ ((value >> (byte * 8)) & 255u)) * 1099511628211ull;
        }
    };
    hash(layout.parameters.id); hash(layout.parameters.size); hash(layout.parameters.alignment);
    hash(uint64_t(layout.parameters.transport));
    for (auto word : candidate->shader.spirv) { hash(word); }
    candidate->key = key;
    program = std::move(executable);
    artifact = std::move(candidate);
    return {};
}

Result<> initializeMaterialErrorKernel(Device& device, ComputeKernel& program, std::string& log)
{
    std::shared_ptr<const MaterialExecutableArtifact> artifact;
    return compileMaterialExecutable(device,
        {.moduleName = "Features/Material/MaterialError", .entryPointName = "materialErrorMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"},
        ComputeKernelDesc{.parameters = parameterAbi<MaterialErrorParams>(kMaterialErrorABI, ParameterTransport::InlinePush),
            .debugName = "MaterialError"},
        program, artifact, log);
}

Result<> dispatchMaterialError(Device& device, const ComputeKernel& program, CommandBuffer& commands,
    TextureView& output, uint32_t width, uint32_t height, bool color)
{
    auto registry = device.resourceRegistry();
    if (!registry) { return makeError(registry.error()); }
    ParameterWriter writer(device, **registry, commands.frameContext());
    const MaterialErrorParams params{.output = writer.storageImage(&output), .color = uint32_t(color)};
    auto encoded = writer.encode(params, kMaterialErrorABI, ParameterTransport::InlinePush);
    if (!encoded) { return makeError(encoded.error()); }
    return program.dispatch(commands, *encoded, (width + 7u) / 8u, (height + 7u) / 8u);
}

} // namespace metallic::render

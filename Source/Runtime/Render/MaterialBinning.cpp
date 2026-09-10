#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/SlangCompiler.h"

namespace metallic::render {

struct MaterialBinning::Allocation {
    std::array<std::unique_ptr<Buffer>, 3> buffers; // bins, tile tasks, arguments
    GpuCompletionPoint completion;
    uint32_t tileCount = 0;
    bool initialized = false;
};

void MaterialBinning::clear()
{
    for (auto& program : programs_) { program.clear(); }
    allocations_.clear();
}

Result MaterialBinning::record(Device& device, CommandBuffer& commands,
    const MaterialBinningDesc& desc, MaterialBinningResult& output, std::string& log)
{
    output = {};
    // RHI compute stages do not enable ALLOW_VARYING_SUBGROUP_SIZE. Their native
    // subgroupSize is fixed; one 32-thread workgroup is exactly one NVIDIA warp.
    if (!device.capabilities().computeSubgroupBallotArithmetic || device.capabilities().subgroupSize != 32) {
        log = "Material classification requires native wave32 with subgroup ballot/arithmetic; disable materialBinning on this device";
        return makeError(Error::Unsupported);
    }
    auto* frame = commands.frameContext();
    const uint64_t columns = (uint64_t(desc.width) + kMaterialTileWidth - 1) / kMaterialTileWidth;
    const uint64_t rows = (uint64_t(desc.height) + kMaterialTileHeight - 1) / kMaterialTileHeight;
    const uint64_t tileCount = columns * rows;
    if (frame == nullptr || !frame->recording() || desc.visibility == nullptr ||
        desc.records == nullptr || desc.instances == nullptr || desc.materials == nullptr ||
        desc.shadingMaterials == nullptr || desc.shadingMaterials->desc().size == 0 ||
        tileCount == 0 || tileCount > UINT32_MAX / kMaterialClassCount ||
        uint64_t(desc.width) * desc.height > UINT32_MAX || columns > 65535 || rows > 65535) {
        log = "Material classification requires a recording frame, valid scene materials and bounded non-empty tile dimensions";
        return makeError(Error::InvalidArgument);
    }
    const ComputeProgramBindingDesc layout[] = {
        {.binding = 0, .kind = ComputeResourceBindingKind::SampledImage},
        {.binding = 1}, {.binding = 2}, {.binding = 3}, {.binding = 4},
        {.binding = 5}, {.binding = 6}, {.binding = 7},
    };
    const char* entries[] = {"materialBinningResetMain", "materialBinningClassifyMain", "materialBinningArgumentsMain"};
    const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
    for (size_t i = 0; i < programs_.size(); ++i) {
        if (programs_[i].valid()) { continue; }
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({
            .moduleName = "Features/VisibilityBuffer/VisibilityMaterialBinning",
            .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .capabilities = capabilities, .capabilityCount = 2}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        result = programs_[i].initialize(device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = 16,
            .bindings = layout, .bindingCount = 8, .debugName = entries[i], .requiresRayQuery = false}, log);
        if (!result) { return result; }
    }

    std::shared_ptr<Allocation> allocation;
    for (const auto& candidate : allocations_) {
        if (candidate->completion.isComplete()) { allocation = candidate; break; }
    }
    if (allocation == nullptr) {
        allocation = std::make_shared<Allocation>();
        allocations_.push_back(allocation);
    }
    if (allocation->tileCount != tileCount) {
        allocation->tileCount = 0;
        allocation->initialized = false;
        const uint64_t sizes[] = {kMaterialClassCount * 8, tileCount * kMaterialClassCount * 8, kMaterialClassCount * 12};
        const uint32_t strides[] = {8, 8, 4};
        for (size_t i = 0; i < allocation->buffers.size(); ++i) {
            auto result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                    (i == 2 ? BufferUsageBits::Indirect : BufferUsageBits::None),
                .memoryLocation = MemoryLocation::Device}, allocation->buffers[i]);
            if (!result) { return result; }
        }
        allocation->tileCount = static_cast<uint32_t>(tileCount);
    }
    allocation->completion = frame->completion();
    frame->retain(allocation);
    const auto& buffers = allocation->buffers;
    const ResourceState finalStates[] = {ResourceState::ShaderRead, ResourceState::ShaderRead, ResourceState::IndirectArgument};
    BufferBarrierDesc barriers[3];
    for (size_t i = 0; i < 3; ++i) {
        barriers[i] = {.buffer = buffers[i].get(),
            .before = allocation->initialized ? finalStates[i] : ResourceState::Undefined,
            .after = ResourceState::General};
    }
    commands.barrier({.buffers = barriers, .bufferCount = 3});
    TextureView* visibility = desc.visibility;
    const ComputeDispatchBinding bindings[] = {
        {.binding = 0, .textureViews = &visibility, .textureViewCount = 1},
        {.binding = 1, .buffer = desc.records}, {.binding = 2, .buffer = desc.instances},
        {.binding = 3, .buffer = desc.materials}, {.binding = 4, .buffer = desc.shadingMaterials},
        {.binding = 5, .buffer = buffers[0].get()}, {.binding = 6, .buffer = buffers[1].get()},
        {.binding = 7, .buffer = buffers[2].get()},
    };
    const uint32_t push[] = {desc.width, desc.height, static_cast<uint32_t>(tileCount), 0};
    for (size_t i = 0; i < programs_.size(); ++i) {
        auto result = programs_[i].dispatch({.commandBuffer = &commands, .bindings = bindings,
            .bindingCount = 8, .pushData = push, .pushDataSize = sizeof(push),
            .groupCountX = i == 1 ? static_cast<uint32_t>(columns) : 1,
            .groupCountY = i == 1 ? static_cast<uint32_t>(rows) : 1});
        if (!result) { return result; }
        for (size_t b = 0; b < 3; ++b) {
            barriers[b] = {.buffer = buffers[b].get(), .before = ResourceState::General,
                .after = i == 2 ? finalStates[b] : ResourceState::General};
        }
        commands.barrier({.buffers = barriers, .bufferCount = 3});
    }
    allocation->initialized = true;
    output = {.bins = buffers[0].get(), .tiles = buffers[1].get(),
        .arguments = buffers[2].get(), .binCount = kMaterialClassCount};
    return {};
}

} // namespace metallic::render

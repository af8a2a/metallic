#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/SlangCompiler.h"

namespace metallic::render {

struct MaterialBinning::Allocation {
    std::array<std::unique_ptr<Buffer>, 4> buffers; // counters, bins, pixels, arguments
    GpuCompletionPoint completion;
    uint32_t pixelCount = 0;
    uint32_t binCount = 0;
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
    if (!device.capabilities().computeSubgroupBallotArithmetic) {
        log = "MaterialBinning requires compute subgroup ballot and arithmetic operations; disable materialBinning on this device";
        return makeError(Error::Unsupported);
    }
    auto* frame = commands.frameContext();
    const uint64_t pixelCount = uint64_t(desc.width) * desc.height;
    if (frame == nullptr || !frame->recording() || desc.visibility == nullptr ||
        desc.records == nullptr || desc.instances == nullptr || desc.materials == nullptr ||
        pixelCount == 0 || pixelCount > UINT32_MAX || desc.width > 65535u * 8u ||
        desc.height > 65535u * 8u || desc.materialCount == 0 || desc.materialCount >= 65536u) {
        log = "MaterialBinning requires a recording frame, valid inputs and at most 65535 materials";
        return makeError(Error::InvalidArgument);
    }
    const uint32_t binCount = desc.materialCount + 1u;
    const ComputeProgramBindingDesc layout[] = {
        {.binding = 0, .kind = ComputeResourceBindingKind::SampledImage},
        {.binding = 1}, {.binding = 2}, {.binding = 3}, {.binding = 4},
        {.binding = 5}, {.binding = 6}, {.binding = 7},
    };
    const char* entries[] = {"materialBinningResetMain", "materialBinningCountMain",
        "materialBinningAllocateMain", "materialBinningScatterMain"};
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
    if (allocation->pixelCount != pixelCount || allocation->binCount != binCount) {
        allocation->pixelCount = 0;
        allocation->initialized = false;
        const uint64_t sizes[] = {uint64_t(binCount) * 4, uint64_t(binCount) * 8,
            pixelCount * 4, uint64_t(binCount) * 12};
        const uint32_t strides[] = {4, 8, 4, 4};
        for (size_t i = 0; i < allocation->buffers.size(); ++i) {
            auto result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                    (i == 3 ? BufferUsageBits::Indirect : BufferUsageBits::None),
                .memoryLocation = MemoryLocation::Device}, allocation->buffers[i]);
            if (!result) { return result; }
        }
        allocation->pixelCount = static_cast<uint32_t>(pixelCount);
        allocation->binCount = binCount;
    }
    allocation->completion = frame->completion();
    frame->retain(allocation);
    const auto& buffers = allocation->buffers;
    const ResourceState finalStates[] = {ResourceState::General, ResourceState::ShaderRead,
        ResourceState::ShaderRead, ResourceState::IndirectArgument};
    BufferBarrierDesc barriers[4];
    for (size_t i = 0; i < 4; ++i) {
        barriers[i] = {.buffer = buffers[i].get(),
            .before = allocation->initialized ? finalStates[i] : ResourceState::Undefined,
            .after = ResourceState::General};
    }
    commands.barrier({.buffers = barriers, .bufferCount = 4});
    TextureView* visibility = desc.visibility;
    const ComputeDispatchBinding bindings[] = {
        {.binding = 0, .textureViews = &visibility, .textureViewCount = 1},
        {.binding = 1, .buffer = desc.records}, {.binding = 2, .buffer = desc.instances},
        {.binding = 3, .buffer = desc.materials}, {.binding = 4, .buffer = buffers[0].get()},
        {.binding = 5, .buffer = buffers[1].get()}, {.binding = 6, .buffer = buffers[2].get()},
        {.binding = 7, .buffer = buffers[3].get()},
    };
    const uint32_t push[] = {desc.width, desc.height, binCount, 0};
    for (size_t i = 0; i < programs_.size(); ++i) {
        auto result = programs_[i].dispatch({.commandBuffer = &commands, .bindings = bindings,
            .bindingCount = 8, .pushData = push, .pushDataSize = sizeof(push),
            .groupCountX = i == 0 ? (binCount + 127) / 128 : (i == 2 ? 1 : (desc.width + 7) / 8),
            .groupCountY = i == 0 || i == 2 ? 1 : (desc.height + 7) / 8});
        if (!result) { return result; }
        for (size_t b = 0; b < 4; ++b) {
            barriers[b] = {.buffer = buffers[b].get(), .before = ResourceState::General,
                .after = i == 3 ? finalStates[b] : ResourceState::General};
        }
        // Includes COMPUTE_SHADER -> DRAW_INDIRECT for arguments, and shader
        // writes -> shader reads for bins/pixels before the deferred dispatches.
        commands.barrier({.buffers = barriers, .bufferCount = 4});
    }
    allocation->initialized = true;
    output = {.bins = buffers[1].get(), .pixels = buffers[2].get(),
        .arguments = buffers[3].get(), .binCount = binCount};
    return {};
}

} // namespace metallic::render

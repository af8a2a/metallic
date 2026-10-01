#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/MaterialBinningParams.h"
#include "Runtime/Render/Core/SlangCompiler.h"

namespace metallic::render {

struct MaterialBinning::Allocation {
    std::array<std::unique_ptr<Buffer>, 3> buffers; // bins, tile tasks, arguments
    GPUCompletionPoint completion;
    uint32_t tileCount = 0;
    bool initialized = false;
};

void MaterialBinning::clear()
{
    for (auto& program : programs_) { program.clear(); }
    allocations_.clear();
}

Result<MaterialBinningResult> MaterialBinning::record(
    Device& device,
    CommandBuffer& commands,
    const MaterialBinningDesc& desc,
    std::string& log)
{
    MaterialBinningResult output{};
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
    if (frame == nullptr || !frame->recording() || !commands.recording() || desc.visibility == nullptr ||
        (desc.streamRecords != nullptr && desc.streamGroups == nullptr) ||
        desc.records == nullptr || desc.instances == nullptr || desc.materials == nullptr ||
        desc.shadingMaterials == nullptr || desc.shadingMaterials->desc().size == 0 ||
        tileCount == 0 || tileCount > UINT32_MAX / kMaterialClassCount ||
        uint64_t(desc.width) * desc.height > UINT32_MAX || columns > 65535 || rows > 65535) {
        log = "Material classification requires a recording frame, valid scene materials and bounded non-empty tile dimensions";
        return makeError(Error::InvalidArgument);
    }
    const char* entries[] = {"materialBinningResetMain", "materialBinningClassifyMain", "materialBinningArgumentsMain"};
    const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
    for (size_t i = 0; i < programs_.size(); ++i) {
        if (programs_[i].valid()) { continue; }
        ShaderCompileResult shader;
        auto result = compileSlangShaderToSpirv({
            .moduleName = "Features/VisibilityBuffer/VisibilityMaterialBinning",
            .entryPointName = entries[i],
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .capabilities = {capabilities, 2},
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result.transform([&] { return std::move(output); }); }
        result = programs_[i].initialize(device, {.spirv = shader.spirv,
            .parameters = parameterAbi<MaterialBinningParams>(kMaterialBinningABI), .debugName = entries[i]}, log);
        if (!result) { return makeError(result.error()); }
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
                .memoryLocation = MemoryLocation::Device}).transform([&](auto rhiValue) { allocation->buffers[i] = std::move(rhiValue); });
            if (!result) { return makeError(result.error()); }
        }
        allocation->tileCount = static_cast<uint32_t>(tileCount);
    }
    allocation->completion = frame->completion();
    frame->retain(allocation);
    const auto& buffers = allocation->buffers;
    using namespace detail;
    using Access = RenderGraphResourceAccess;
    constexpr auto compute = RenderGraphPassKind::Compute;
    constexpr auto consumer = RenderGraphPassKind::Unsafe;
    std::array<GraphAccessResource, 3> resources;
    std::array<GraphAccessBinding, 3> bindings;
    for (size_t i = 0; i < buffers.size(); ++i) {
        // Reused private allocations may have prior consumers outside this helper.
        resources[i] = {.type = RenderGraphResourceType::Buffer,
            .state = allocation->initialized ? ResourceState::General : ResourceState::Undefined,
            .scope = {PipelineStageBits::AllCommands, AccessBits::MemoryRead | AccessBits::MemoryWrite}};
        auto slice = buffers[i]->slice();
        if (!slice) { return makeError(slice.error()); }
        bindings[i].buffer = std::move(*slice);
    }
    const GraphAccessPass phases[] = {
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferStorageReadWrite, compute),
            declaredGraphAccess(1, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferStorageRead, compute),
            declaredGraphAccess(2, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferShaderRead, consumer),
            declaredGraphAccess(1, Access::BufferShaderRead, consumer),
            declaredGraphAccess(2, Access::BufferIndirectRead, consumer)}},
    };
    auto plan = buildGraphAccessPlan(resources, phases);
    if (!plan) { return makeError(plan.error()); }
    std::shared_ptr<ResourceRegistry> registry;
    auto result = device.resourceRegistry().transform([&](auto rhiValue) { registry = std::move(rhiValue); });
    if (!result) { return makeError(result.error()); }
    ParameterWriter writer(device, *frame, *registry);
    const MaterialBinningParams params{
        .visibility = writer.sampledImage(desc.visibility),
        .records = writer.dataBuffer(desc.records, 16, 16), .instances = writer.dataBuffer(desc.instances, 160, 16),
        .materials = writer.dataBuffer(desc.materials, 560, 16), .shadingMaterials = writer.dataBuffer(desc.shadingMaterials, 720, 16),
        .bins = writer.dataBuffer(buffers[0].get(), 8, 8), .tiles = writer.dataBuffer(buffers[1].get(), 8, 8),
        .arguments = writer.dataBuffer(buffers[2].get(), 4, 4),
        .streamRecords = desc.streamRecords ? writer.dataBuffer(desc.streamRecords, 4, 4) : ShaderDataSpan{},
        .streamGroups = desc.streamGroups ? writer.dataBuffer(desc.streamGroups, 112, 16) : ShaderDataSpan{},
        .width = desc.width, .height = desc.height, .tileCount = static_cast<uint32_t>(tileCount),
        .residentRecordCount = desc.residentRecordCount,
    };
    EncodedParameters encoded;
    result = writer.encode(params, kMaterialBinningABI).transform([&](auto value) { encoded = std::move(value); });
    if (!result) { return makeError(result.error()); }
    for (size_t i = 0; i < programs_.size(); ++i) {
        result = recordGraphAccessBarriers(commands, plan->passes[i], bindings);
        if (!result) { return makeError(result.error()); }
        result = programs_[i].dispatch(commands, encoded, i == 1 ? static_cast<uint32_t>(columns) : 1,
            i == 1 ? static_cast<uint32_t>(rows) : 1);
        if (!result) { return makeError(result.error()); }
    }
    result = recordGraphAccessBarriers(commands, plan->passes.back(), bindings);
    if (!result) { return makeError(result.error()); }
    allocation->initialized = true;
    output = {.bins = buffers[0].get(), .tiles = buffers[1].get(),
        .arguments = buffers[2].get(), .binCount = kMaterialClassCount};
    return output;
}

} // namespace metallic::render

#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/MaterialBinningParams.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <cstring>

namespace metallic::render {

struct MaterialBinning::Allocation {
    std::array<std::unique_ptr<Buffer>, 3> buffers; // bins, tile tasks, arguments
    GPUCompletionPoint completion;
    uint32_t tileCount = 0;
    uint32_t binCount = 0;
    bool sparse = false;
    std::unique_ptr<Buffer> programBins;
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
    auto* frame = metallic::render::RenderFrameContext::from(commands);
    const uint64_t columns = (uint64_t(desc.width) + kMaterialTileWidth - 1) / kMaterialTileWidth;
    const uint64_t rows = (uint64_t(desc.height) + kMaterialTileHeight - 1) / kMaterialTileHeight;
    const uint64_t tileCount = columns * rows;
    const bool sparse = desc.programBinCount != 0;
    const uint32_t binCount = sparse ? desc.programBinCount : kMaterialClassCount;
    if (sparse && (binCount > kMaxMaterialProgramBins || desc.materialProgramBins.empty() ||
        desc.shadingMaterials == nullptr || desc.materialProgramBins.size() != desc.shadingMaterials->desc().size / 720 ||
        std::any_of(desc.materialProgramBins.begin(), desc.materialProgramBins.end(),
            [binCount](uint32_t bin) { return bin == 0 || bin >= binCount; }))) {
        log = "Program binning requires a complete instance-to-active-program table; slot zero is reserved for background";
        return makeError(Error::InvalidArgument);
    }
    if (frame == nullptr || !frame->recording() || !commands.recording() || desc.visibility == nullptr ||
        (desc.streamRecords != nullptr && desc.streamGroups == nullptr) ||
        desc.records == nullptr || desc.instances == nullptr || desc.materials == nullptr ||
        desc.shadingMaterials == nullptr || desc.shadingMaterials->desc().size == 0 ||
        tileCount == 0 || tileCount > UINT32_MAX / (sparse ? 32u : kMaterialClassCount) ||
        uint64_t(desc.width) * desc.height > UINT32_MAX || columns > 65535 || rows > 65535) {
        log = "Material classification requires a recording frame, valid scene materials and bounded non-empty tile dimensions";
        return makeError(Error::InvalidArgument);
    }
    const char* entries[] = {"materialBinningResetMain", "materialBinningCountMain", "materialBinningAllocateMain",
        "materialBinningClassifyMain", "materialBinningArgumentsMain"};
    const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
    for (size_t i = 0; i < programs_.size(); ++i) {
        if (!sparse && (i == 1 || i == 2)) { continue; }
        if (programs_[i].valid()) { continue; }
        ShaderCompileResult shader;
        auto result = ShaderRegistry::instance().getShader({
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
    if (allocation->tileCount != tileCount || allocation->binCount != binCount || allocation->sparse != sparse) {
        allocation->tileCount = 0;
        allocation->initialized = false;
        // Each nonempty task owns at least one pixel. Capacity is bounded by
        // pixels, independent of ProgramCount; no ProgramCount x TileCount table.
        const uint64_t taskCapacity = sparse ? tileCount * std::min(binCount, 32u)
            : tileCount * kMaterialClassCount;
        const uint64_t sizes[] = {uint64_t(binCount) * 8, taskCapacity * 8, uint64_t(binCount) * 12};
        const uint32_t strides[] = {8, 8, 4};
        for (size_t i = 0; i < allocation->buffers.size(); ++i) {
            auto result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                    (i == 2 ? BufferUsageBits::Indirect : BufferUsageBits::None),
                .memoryLocation = MemoryLocation::Device}).transform([&](auto rhiValue) { allocation->buffers[i] = std::move(rhiValue); });
            if (!result) { return makeError(result.error()); }
        }
        allocation->tileCount = static_cast<uint32_t>(tileCount);
        allocation->binCount = binCount;
        allocation->sparse = sparse;
    }
    if (sparse) {
        const auto bytes = desc.materialProgramBins.size_bytes();
        if (!allocation->programBins || allocation->programBins->desc().size != bytes) {
            auto created = device.createBuffer({.size = bytes, .structureStride = 4,
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload});
            if (!created) { return makeError(created.error()); }
            allocation->programBins = std::move(*created);
        }
        auto* mapped = allocation->programBins->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, desc.materialProgramBins.data(), bytes);
        allocation->programBins->flush();
        allocation->programBins->unmap();
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
    std::vector<GraphAccessPass> phases{
        {.uses = {declaredGraphAccess(0, Access::BufferStorageWrite, compute)}},
    };
    if (sparse) {
        phases.push_back({.uses = {declaredGraphAccess(0, Access::BufferStorageReadWrite, compute)}});
        phases.push_back({.uses = {declaredGraphAccess(0, Access::BufferStorageReadWrite, compute)}});
    }
    const GraphAccessPass consumers[] = {
        {.uses = {declaredGraphAccess(0, Access::BufferStorageReadWrite, compute),
            declaredGraphAccess(1, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferStorageRead, compute),
            declaredGraphAccess(2, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferShaderRead, consumer),
            declaredGraphAccess(1, Access::BufferShaderRead, consumer),
            declaredGraphAccess(2, Access::BufferIndirectRead, consumer)}},
    };
    phases.insert(phases.end(), std::begin(consumers), std::end(consumers));
    auto plan = buildGraphAccessPlan(resources, phases);
    if (!plan) { return makeError(plan.error()); }
    std::shared_ptr<ResourceRegistry> registry;
    auto result = metallic::render::ResourceRegistry::forDevice(device).transform([&](auto rhiValue) { registry = std::move(rhiValue); });
    if (!result) { return makeError(result.error()); }
    ParameterWriter writer(device, *frame, *registry);
    const MaterialBinningParams params{
        .visibility = writer.sampledImage(desc.visibility),
        .records = writer.bufferSpan(desc.records, 16, 16), .instances = writer.bufferSpan(desc.instances, 160, 16),
        .materials = writer.bufferSpan(desc.materials, 560, 16), .shadingMaterials = writer.bufferSpan(desc.shadingMaterials, 720, 16),
        .bins = writer.bufferSpan(buffers[0].get(), 8, 8), .tiles = writer.bufferSpan(buffers[1].get(), 8, 8),
        .arguments = writer.bufferSpan(buffers[2].get(), 4, 4),
        .streamRecords = desc.streamRecords ? writer.bufferSpan(desc.streamRecords, 4, 4) : GPUBufferSpan{},
        .streamGroups = desc.streamGroups ? writer.bufferSpan(desc.streamGroups, 112, 16) : GPUBufferSpan{},
        .width = desc.width, .height = desc.height, .tileCount = static_cast<uint32_t>(tileCount),
        .residentRecordCount = desc.residentRecordCount,
        .materialProgramBins = sparse ? writer.bufferSpan(allocation->programBins.get(), 4, 4) : GPUBufferSpan{},
        .programBinCount = desc.programBinCount,
    };
    EncodedParameters encoded;
    result = writer.encode(params, kMaterialBinningABI).transform([&](auto value) { encoded = std::move(value); });
    if (!result) { return makeError(result.error()); }
    size_t phase = 0;
    for (size_t i = 0; i < programs_.size(); ++i) {
        if (!sparse && (i == 1 || i == 2)) { continue; }
        result = recordGraphAccessBarriers(commands, plan->passes[phase++], bindings);
        if (!result) { return makeError(result.error()); }
        const bool tiles = i == 1 || i == 3;
        result = programs_[i].dispatch(commands, encoded,
            tiles ? static_cast<uint32_t>(columns) : i == 2 ? 1 : (binCount + 31) / 32,
            tiles ? static_cast<uint32_t>(rows) : 1);
        if (!result) { return makeError(result.error()); }
    }
    result = recordGraphAccessBarriers(commands, plan->passes.back(), bindings);
    if (!result) { return makeError(result.error()); }
    allocation->initialized = true;
    output = {.bins = buffers[0].get(), .tiles = buffers[1].get(),
        .arguments = buffers[2].get(), .binCount = binCount};
    return output;
}

} // namespace metallic::render

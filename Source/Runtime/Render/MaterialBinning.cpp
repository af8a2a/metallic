#include "Runtime/Render/Core/ResourceState.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/RenderGraph/RenderGraphAccessPlan.h"
#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/Material/SceneMaterialData.h"
#include "Runtime/Render/MaterialBinningParams.h"
#include "Runtime/Render/Core/SlangCompiler.h"

#include <algorithm>
#include <cstring>

namespace metallic::render {

struct MaterialBinning::Allocation {
    std::array<std::unique_ptr<Buffer>, 5> buffers; // bins, tile tasks, arguments, tile summaries, mixed tile pixel bins
    GPUCompletionPoint completion;
    uint32_t tileCount = 0;
    uint32_t binCount = 0;
    bool sparse = false;
    bool twoPass = false;
    std::unique_ptr<Buffer> programBins;
    bool initialized = false;
};

void MaterialBinning::clear()
{
    resetProgram_.clear();
    countProgram_.clear();
    allocateProgram_.clear();
    for (auto& program : classifyPrograms_) { program.clear(); }
    argumentsProgram_.clear();
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
        desc.shadingMaterials == nullptr || desc.materialProgramBins.size() != desc.shadingMaterials->desc().size / sizeof(SceneMaterialData) ||
        std::any_of(desc.materialProgramBins.begin(), desc.materialProgramBins.end(),
            [binCount](uint32_t bin) { return bin == 0 || bin >= binCount; }))) {
        log = "Program binning requires a complete instance-to-active-program table; slot zero is reserved for background";
        return makeError(Error::InvalidArgument);
    }
    if (frame == nullptr || !frame->recording() || !commands.recording() || desc.visibility == nullptr ||
        (desc.streamRecords == nullptr || desc.streamGroups == nullptr) ||
        desc.instances == nullptr || desc.materials == nullptr ||
        desc.shadingMaterials == nullptr || desc.shadingMaterials->desc().size == 0 ||
        tileCount == 0 || tileCount > UINT32_MAX / (sparse ? 32u : kMaterialClassCount) ||
        uint64_t(desc.width) * desc.height > UINT32_MAX || columns > 65535 || rows > 65535) {
        log = "Material classification requires a recording frame, valid scene materials and bounded non-empty tile dimensions";
        return makeError(Error::InvalidArgument);
    }
    const bool twoPass = sparse && binCount > 2u;
    const size_t classifyMode = !sparse ? 0u : twoPass ? 1u : (desc.trustVisibilityHits ? 3u : 2u);
    const char* capabilities[] = {"spvGroupNonUniformBallot", "spvGroupNonUniformArithmetic"};
    auto ensureKernel = [&](ComputeKernel& kernel, const char* entryPoint, const char* debugName,
                            std::span<const SlangMacroDefine> macroDefines) -> Result<void> {
        if (kernel.valid()) { return {}; }
        ShaderCompileResult shader;
        auto result = ShaderRegistry::instance().getShader({
            .moduleName = "Features/VisibilityBuffer/VisibilityMaterialBinning",
            .entryPointName = entryPoint,
            .searchPath = PROJECT_SOURCE_DIR "/Shaders",
            .capabilities = {capabilities, 2},
            .macroDefines = macroDefines,
        }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        return kernel.initialize(device, {.spirv = shader.spirv,
            .parameters = parameterAbi<MaterialBinningParams>(kMaterialBinningABI), .debugName = debugName}, log);
    };
    const SlangMacroDefine modeOneDefines[] = {{"METALLIC_MATERIAL_BINNING_MODE", "1"}};
    const SlangMacroDefine classifyModeDefines[][1] = {
        {{"METALLIC_MATERIAL_BINNING_MODE", "0"}},
        {{"METALLIC_MATERIAL_BINNING_MODE", "1"}},
        {{"METALLIC_MATERIAL_BINNING_MODE", "2"}},
        {{"METALLIC_MATERIAL_BINNING_MODE", "3"}},
    };
    const char* classifyDebugNames[] = {
        "materialBinningClassifyFixedMain",
        "materialBinningClassifyTwoPassMain",
        "materialBinningClassifySingleForegroundMain",
        "materialBinningClassifySingleForegroundTrustedMain",
    };
    if (auto result = ensureKernel(resetProgram_, "materialBinningResetMain", "materialBinningResetMain", {}); !result) {
        return makeError(result.error());
    }
    if (twoPass) {
        if (auto result = ensureKernel(countProgram_, "materialBinningCountMain", "materialBinningCountMain", modeOneDefines); !result) {
            return makeError(result.error());
        }
        if (auto result = ensureKernel(allocateProgram_, "materialBinningAllocateMain", "materialBinningAllocateMain", {}); !result) {
            return makeError(result.error());
        }
    }
    if (auto result = ensureKernel(classifyPrograms_[classifyMode], "materialBinningClassifyMain",
        classifyDebugNames[classifyMode], classifyModeDefines[classifyMode]); !result) {
        return makeError(result.error());
    }
    if (auto result = ensureKernel(argumentsProgram_, "materialBinningArgumentsMain", "materialBinningArgumentsMain", {}); !result) {
        return makeError(result.error());
    }

    std::shared_ptr<Allocation> allocation;
    for (const auto& candidate : allocations_) {
        if (candidate->completion.isComplete()) { allocation = candidate; break; }
    }
    if (allocation == nullptr) {
        allocation = std::make_shared<Allocation>();
        allocations_.push_back(allocation);
    }
    if (allocation->tileCount != tileCount || allocation->binCount != binCount ||
        allocation->sparse != sparse || allocation->twoPass != twoPass) {
        allocation->tileCount = 0;
        allocation->initialized = false;
        // Each nonempty task owns at least one pixel. Capacity is bounded by
        // pixels, independent of ProgramCount; no ProgramCount x TileCount table.
        const uint64_t taskCapacity = sparse ? tileCount * std::min(binCount, 32u)
            : tileCount * kMaterialClassCount;
        const uint64_t pixelBinStrideBytes = binCount <= 16u ? 16ull : (binCount <= 256u ? 32ull : 64ull);
        const uint64_t sizes[] = {
            uint64_t(binCount) * 8,
            taskCapacity * 8,
            uint64_t(binCount) * 12,
            twoPass ? uint64_t(tileCount) * 4 : 0,
            twoPass ? uint64_t(tileCount) * pixelBinStrideBytes : 0,
        };
        const uint32_t strides[] = {8, 8, 4, 4, 4};
        for (size_t i = 0; i < allocation->buffers.size(); ++i) {
            if (sizes[i] == 0) {
                allocation->buffers[i].reset();
                continue;
            }
            auto result = device.createBuffer({.size = sizes[i], .structureStride = strides[i],
                .usage = BufferUsageBits::Storage | BufferUsageBits::TransferSource |
                    (i == 2 ? BufferUsageBits::Indirect : BufferUsageBits::None),
                .memoryLocation = MemoryLocation::Device}).transform([&](auto rhiValue) { allocation->buffers[i] = std::move(rhiValue); });
            if (!result) { return makeError(result.error()); }
        }
        allocation->tileCount = static_cast<uint32_t>(tileCount);
        allocation->binCount = binCount;
        allocation->sparse = sparse;
        allocation->twoPass = twoPass;
    }
    if (twoPass) {
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
    const size_t resourceCount = twoPass ? 5u : 3u;
    std::array<GraphAccessResource, 5> resources;
    std::array<GraphAccessBinding, 5> bindings;
    for (size_t i = 0; i < resourceCount; ++i) {
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
    if (twoPass) {
        phases.push_back({.uses = {
            declaredGraphAccess(0, Access::BufferStorageReadWrite, compute),
            declaredGraphAccess(3, Access::BufferStorageWrite, compute),
            declaredGraphAccess(4, Access::BufferStorageWrite, compute),
        }});
        phases.push_back({.uses = {
            declaredGraphAccess(0, Access::BufferStorageReadWrite, compute),
            declaredGraphAccess(2, Access::BufferStorageWrite, compute),
        }});
    }
    GraphAccessPass classifyPass{.uses = {
        declaredGraphAccess(0, Access::BufferStorageReadWrite, compute),
        declaredGraphAccess(1, Access::BufferStorageWrite, compute),
    }};
    if (twoPass) {
        classifyPass.uses.push_back(declaredGraphAccess(2, Access::BufferStorageRead, compute));
        classifyPass.uses.push_back(declaredGraphAccess(3, Access::BufferStorageRead, compute));
        classifyPass.uses.push_back(declaredGraphAccess(4, Access::BufferStorageRead, compute));
    }
    const GraphAccessPass consumers[] = {
        std::move(classifyPass),
        {.uses = {declaredGraphAccess(0, Access::BufferStorageRead, compute),
            declaredGraphAccess(2, Access::BufferStorageWrite, compute)}},
        {.uses = {declaredGraphAccess(0, Access::BufferShaderRead, consumer),
            declaredGraphAccess(1, Access::BufferShaderRead, consumer),
            declaredGraphAccess(2, Access::BufferIndirectRead, consumer)}},
    };
    phases.insert(phases.end(), std::begin(consumers), std::end(consumers));
    auto plan = buildGraphAccessPlan(std::span{resources.data(), resourceCount}, phases);
    if (!plan) { return makeError(plan.error()); }
    std::shared_ptr<ResourceRegistry> registry;
    auto result = metallic::render::ResourceRegistry::forDevice(device).transform([&](auto rhiValue) { registry = std::move(rhiValue); });
    if (!result) { return makeError(result.error()); }
    ParameterWriter writer(device, *frame, *registry);
    const MaterialBinningParams params{
        .visibility = writer.sampledImage(desc.visibility),
        .instances = writer.bufferSpan(desc.instances, 160, 16),
        .materials = writer.bufferSpan(desc.materials, 560, 16), .shadingMaterials = writer.bufferSpan(desc.shadingMaterials, sizeof(SceneMaterialData), 16),
        .bins = writer.bufferSpan(buffers[0].get(), 8, 8), .tiles = writer.bufferSpan(buffers[1].get(), 8, 8),
        .arguments = writer.bufferSpan(buffers[2].get(), 4, 4),
        .streamRecords = desc.streamRecords ? writer.bufferSpan(desc.streamRecords, 4, 4) : GPUBufferSpan{},
        .streamGroups = desc.streamGroups ? writer.bufferSpan(desc.streamGroups, 112, 16) : GPUBufferSpan{},
        .width = desc.width, .height = desc.height, .tileCount = static_cast<uint32_t>(tileCount),
        .materialProgramBins = twoPass ? writer.bufferSpan(allocation->programBins.get(), 4, 4) : GPUBufferSpan{},
        .programBinCount = desc.programBinCount,
        .tileSummaries = twoPass ? writer.bufferSpan(buffers[3].get(), 4, 4) : GPUBufferSpan{},
        .tilePixelBins = twoPass ? writer.bufferSpan(buffers[4].get(), 4, 4) : GPUBufferSpan{},
    };
    EncodedParameters encoded;
    result = writer.encode(params, kMaterialBinningABI).transform([&](auto value) { encoded = std::move(value); });
    if (!result) { return makeError(result.error()); }
    size_t phase = 0;
    auto dispatchStage = [&](ComputeKernel& kernel, uint32_t x, uint32_t y) -> Result<void> {
        if (auto barrierResult = recordGraphAccessBarriers(commands, plan->passes[phase++],
            std::span{bindings.data(), resourceCount}); !barrierResult) {
            return makeError(barrierResult.error());
        }
        return kernel.dispatch(commands, encoded, x, y);
    };
    const uint32_t binGroupsX = (binCount + 31u) / 32u;
    const uint32_t ctaCols = (static_cast<uint32_t>(columns) + 3u) / 4u;
    const uint32_t ctaRows = (static_cast<uint32_t>(rows) + 1u) / 2u;
    if (auto step = dispatchStage(resetProgram_, binGroupsX, 1u); !step) { return makeError(step.error()); }
    if (twoPass) {
        if (auto step = dispatchStage(countProgram_, ctaCols, ctaRows); !step) { return makeError(step.error()); }
        if (auto step = dispatchStage(allocateProgram_, 1u, 1u); !step) { return makeError(step.error()); }
    }
    if (auto step = dispatchStage(classifyPrograms_[classifyMode], ctaCols, ctaRows); !step) { return makeError(step.error()); }
    if (auto step = dispatchStage(argumentsProgram_, binGroupsX, 1u); !step) { return makeError(step.error()); }
    result = recordGraphAccessBarriers(commands, plan->passes.back(), std::span{bindings.data(), resourceCount});
    if (!result) { return makeError(result.error()); }
    allocation->initialized = true;
    output = {.bins = buffers[0].get(), .tiles = buffers[1].get(),
        .arguments = buffers[2].get(), .binCount = binCount};
    return output;
}

} // namespace metallic::render

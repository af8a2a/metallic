#include "TestResourceParameters.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "RHITest.h"
#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Material/MaterialExecutable.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <bit>
#include <cmath>
#include <fstream>

namespace metallic::tests {
namespace {

constexpr uint32_t kBinCount = render::kMaterialClassCount;
constexpr uint32_t kProbeHeader = kBinCount * 5 + 2;
constexpr uint64_t kProbeABI = 0x4d42505200000003ull;
struct MaterialProbeParams {
    render::GPUBufferSpan bins, tiles, arguments, output;
    uint32_t width, height, binCount, bin;
};
static_assert(sizeof(MaterialProbeParams) == 64);

class MaterialBinningProbePass final : public render::UnsafePass {
public:
    explicit MaterialBinningProbePass(bool fixture, bool typed = false, uint32_t materialCount = 257)
        : fixture_(fixture), typed_(typed), materialCount_(materialCount) {}

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        if (fixture_) {
            reflection.addTextureOutput("visibility").storageReadWrite().format = render::Format::R32Uint;
            reflection.addBufferOutput("records").buffer((materialCount_ + 3) * 16, 16).storageReadWrite();
            reflection.addBufferOutput("instances").buffer((materialCount_ + 3) * 160, 160).storageReadWrite();
            reflection.addBufferOutput("materials").buffer(materialCount_ * 560, 560).storageReadWrite();
            reflection.addBufferOutput("shadingMaterials").buffer(materialCount_ * 720, 720).storageReadWrite();
        } else {
            reflection.addTextureInput("visibility").sampledRead();
            for (const char* name : {"records", "instances", "materials", "shadingMaterials"}) {
                reflection.addBufferInput(name).shaderRead();
            }
            reflection.addBufferOutput("data").buffer(
                (uint64_t(context.width) * context.height * 3 + kProbeHeader) * 4, 4).storageReadWrite();
        }
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        if (typed_) {
            const char* entries[] = {"materialReadbackResetMain", "materialIndirectProbeMain", "materialIndirectProbeMain"};
            for (uint32_t i = 0; i < kernels_.size(); ++i) {
                const render::SlangMacroDefine defines[] = {{"PROBE_TYPED", "1"}, {"PROBE_ALTERNATE", "1"}};
                render::ShaderCompileResult shader;
                auto result = render::compileSlangShaderToSpirv({
                    .moduleName = "MaterialBinningProbe",
                    .entryPointName = entries[i],
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                    .macroDefines = {defines, i == 2 ? 2u : 1u},
                }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
                if (!result) { log = shader.diagnostics; return result; }
                result = kernels_[i].initialize(*device_, {.spirv = shader.spirv,
                    .parameters = render::parameterAbi<MaterialProbeParams>(kProbeABI)}, log);
                if (!result) { return result; }
            }
            return {};
        }
        const render::ComputeResourceBindingDesc layout[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, visibility), .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, records)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, instances)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, materials)},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, shadingMaterials)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, bins)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, tiles)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, arguments)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, output)}};
        const char* entries[] = {fixture_ ? "materialFixtureMain" : "materialReadbackResetMain",
            "materialIndirectProbeMain", "materialIndirectProbeMain"};
        for (uint32_t i = 0; i < (fixture_ ? 1u : 3u); ++i) {
            const std::string materialCount = std::to_string(materialCount_);
            const render::SlangMacroDefine alternate[] = {{"PROBE_MATERIAL_COUNT", materialCount.c_str()}, {"PROBE_ALTERNATE", "1"}};
            render::ShaderCompileResult shader;
            auto result = render::compileSlangShaderToSpirv({
                .moduleName = "MaterialBinningProbe",
                .entryPointName = entries[i],
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .macroDefines = {alternate, i == 2 ? 2u : 1u},
            }, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
            if (!result) { log = shader.diagnostics; return result; }
            result = programs_[i].initialize(*device_, {
                .spirv = shader.spirv,
                .pushConstantSize = 16,
                .bindings = {fixture_ ? layout : layout + 5, fixture_ ? 5u : 4u},
                .requiresRayQuery = false,
                .resourceParameterSize = sizeof(metallic::tests::MaterialBinningProbeResources),
            }, log);
            if (!result) { return result; }
            if (!fixture_ && i == 0) {
                // Constant layout, rather than obsolete table counts, defines ABI compatibility.
                result = incompatibleProgram_.initialize(*device_, {
                    .spirv = shader.spirv, .pushConstantSize = 20,
                    .bindings = {layout + 5, 4}, .requiresRayQuery = false,
                    .resourceParameterSize = sizeof(metallic::tests::MaterialBinningProbeResources),
                }, log);
                if (!result) { return result; }
            }
        }
        return {};
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto& commands = context.commandBuffer();
        uint32_t push[] = {context.width(), context.height(), kBinCount, static_cast<uint32_t>(context.frameIndex() % 4)};
        const uint32_t pixels = push[0] * push[1];
        const uint32_t fixtureGroups = (std::max(pixels, materialCount_ + 3) + 63) / 64;
        const uint32_t readbackGroups = (std::max(pixels, kBinCount) + 63) / 64;
        if (fixture_) {
            const render::ComputeDispatchBinding bindings[] = {
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, visibility), .textureView = context.outputTexture("visibility").view()},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, records), .buffer = context.outputBuffer("records").buffer()},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, instances), .buffer = context.outputBuffer("instances").buffer()},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, materials), .buffer = context.outputBuffer("materials").buffer()},
                {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, shadingMaterials), .buffer = context.outputBuffer("shadingMaterials").buffer()}};
            return programs_[0].dispatch({
                .commandBuffer = &commands,
                .bindings = {bindings, 5},
                .pushData = push,
                .pushDataSize = sizeof(push),
                .groupCountX = std::min(fixtureGroups, 65535u),
                .groupCountY = (fixtureGroups + 65534) / 65535,
            });
        }
        render::MaterialBinningResult bins;
        std::string log;
        auto result = binning_.record(*device_, commands, {
            .visibility = context.inputTexture("visibility").view(),
            .records = context.inputBuffer("records").buffer(),
            .instances = context.inputBuffer("instances").buffer(),
            .materials = context.inputBuffer("materials").buffer(),
            .shadingMaterials = context.inputBuffer("shadingMaterials").buffer(),
            .width = push[0], .height = push[1]}, log).transform([&](auto value) { bins = std::move(value); });
        if (!result) { return result; }
        if (typed_) { return executeTyped(context, bins, push, readbackGroups); }
        // Invalid inputs must fail before recording vkCmdDispatchIndirect2KHR.
        for (uint64_t offset : {uint64_t(1), bins.arguments->desc().size - 4, UINT64_MAX}) {
            if (!render::hasError((*bins.arguments).slice({offset, 12}).and_then([&](const auto& bufferSlice) { return commands.dispatchIndirect(bufferSlice); }), render::Error::InvalidArgument)) {
                return render::makeError(render::Error::Failure);
            }
        }
        if (!render::hasError((*bins.bins).slice({0, 12}).and_then([&](const auto& bufferSlice) { return commands.dispatchIndirect(bufferSlice); }), render::Error::InvalidArgument)) {
            return render::makeError(render::Error::Failure);
        }
        // Read argument bytes as shader data, then restore their indirect state.
        render::BufferBarrierDesc argumentBarrier{
            .buffer = bins.arguments,
            .before = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead},
        };
        if (auto commandResult = commands.synchronize({.buffers = {&argumentBarrier, 1}}); !commandResult) { return commandResult; }
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, bins), .buffer = bins.bins}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, tiles), .buffer = bins.tiles},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, arguments), .buffer = bins.arguments}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, output), .buffer = context.outputBuffer("data").buffer()}};
        render::ComputeDispatchDesc dispatch{
            .commandBuffer = &commands,
            .bindings = {bindings, 4},
            .pushData = push,
            .pushDataSize = sizeof(push),
            .groupCountX = std::min(readbackGroups, 65535u),
            .groupCountY = (readbackGroups + 65534) / 65535,
        };
        result = programs_[0].dispatch(dispatch);
        if (!result) { return result; }
        std::swap(argumentBarrier.before, argumentBarrier.after);
        if (auto commandResult = commands.synchronize({.buffers = {&argumentBarrier, 1}}); !commandResult) { return commandResult; }
        render::BufferBarrierDesc outputBarrier{
            .buffer = context.outputBuffer("data").buffer(),
            .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
        };
        dispatch.indirectArguments = bins.arguments;
        // Reject an incompatible permutation before descriptor writes or GPU work.
        const render::TestComputeIndirectDispatch incompatible[] = {{.pushData = push, .program = &incompatibleProgram_}};
        if (!render::hasError(programs_[1].dispatchIndirectBatch(dispatch, incompatible), render::Error::InvalidArgument)) {
            return render::makeError(render::Error::Failure);
        }
        if ((context.frameIndex() & 1u) != 0) {
            std::vector<std::array<uint32_t, 4>> pushes(bins.binCount);
            std::vector<render::TestComputeIndirectDispatch> items(bins.binCount);
            for (uint32_t bin = 0; bin < bins.binCount; ++bin) {
                pushes[bin] = {push[0], push[1], push[2], bin};
                items[bin] = {.pushData = pushes[bin].data(), .argumentOffset = uint64_t(bin) * 12,
                    .program = (bin & 1u) != 0 ? &programs_[2] : nullptr};
            }
            if (auto commandResult = commands.synchronize({.buffers = {&outputBarrier, 1}}); !commandResult) { return commandResult; }
            return programs_[1].dispatchIndirectBatch(dispatch, items, {.buffers = {&outputBarrier, 1}});
        }
        for (uint32_t bin = 0; bin < bins.binCount; ++bin) {
            if (auto commandResult = commands.synchronize({.buffers = {&outputBarrier, 1}}); !commandResult) { return commandResult; }
            push[3] = bin;
            dispatch.indirectOffset = uint64_t(bin) * 12;
            result = programs_[1].dispatch(dispatch);
            if (!result) { return result; }
        }
        return {};
    }

private:
    render::Result<> executeTyped(render::RenderGraphExecutionContext& context,
        const render::MaterialBinningResult& bins, const uint32_t* push, uint32_t groups)
    {
        auto& commands = context.commandBuffer();
        std::shared_ptr<render::ResourceRegistry> registry;
        auto result = metallic::render::ResourceRegistry::forDevice(*device_).transform([&](auto rhiValue) { registry = std::move(rhiValue); });
        if (!result) { return result; }
        const auto before = registry->stats();
        render::ParameterWriter writer(*device_, *metallic::render::RenderFrameContext::from(commands), *registry);
        MaterialProbeParams params{writer.bufferSpan(bins.bins, 8, 8), writer.bufferSpan(bins.tiles, 8, 8),
            writer.bufferSpan(bins.arguments, 4, 4), writer.bufferSpan(context.outputBuffer("data").buffer(), 4, 4),
            push[0], push[1], push[2], push[3]};
        // Producer buffers reuse their descriptors; the output may register once.
        const auto after = registry->stats();
        if (after.descriptorWrites - before.descriptorWrites > 1 ||
            after.cacheHits - before.cacheHits < 3) { return render::makeError(render::Error::Failure); }
        render::EncodedParameters encoded;
        result = writer.encode(params, kProbeABI).transform([&](auto value) { encoded = std::move(value); });
        if (!result) { return result; }
        render::BufferBarrierDesc argumentBarrier{
            .buffer = bins.arguments,
            .before = {render::PipelineStageBits::DrawIndirect, render::AccessBits::IndirectRead},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::ShaderRead},
        };
        if (auto commandResult = commands.synchronize({.buffers = {&argumentBarrier, 1}}); !commandResult) { return commandResult; }
        result = kernels_[0].dispatch(commands, encoded, std::min(groups, 65535u), (groups + 65534) / 65535);
        if (!result) { return result; }
        std::swap(argumentBarrier.before, argumentBarrier.after);
        if (auto commandResult = commands.synchronize({.buffers = {&argumentBarrier, 1}}); !commandResult) { return commandResult; }
        for (uint64_t offset : {uint64_t(1), bins.arguments->desc().size - 4, UINT64_MAX}) {
            if (!render::hasError((*bins.arguments).slice({offset, 12}).and_then([&](const auto& bufferSlice) { return kernels_[1].dispatchIndirect(commands, encoded, bufferSlice); }),
                render::Error::InvalidArgument)) { return render::makeError(render::Error::Failure); }
        }
        render::EncodedParameters wrongAbi;
        result = writer.encode(params, kProbeABI + 1).transform([&](auto value) { wrongAbi = std::move(value); });
        if (!result) { return result; }
        if (!render::hasError((*bins.arguments).slice({0, 12}).and_then([&](const auto& bufferSlice) { return kernels_[1].dispatchIndirect(commands, wrongAbi, bufferSlice); }),
            render::Error::InvalidArgument)) { return render::makeError(render::Error::Failure); }
        render::BufferBarrierDesc outputBarrier{
            .buffer = context.outputBuffer("data").buffer(),
            .before = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
            .after = {render::PipelineStageBits::AllCommands, render::AccessBits::MemoryRead | render::AccessBits::MemoryWrite},
        };
        for (uint32_t bin = 0; bin < bins.binCount; ++bin) {
            params.bin = bin;
            result = writer.encode(params, kProbeABI).transform([&](auto value) { encoded = std::move(value); });
            if (!result) { return result; }
            if (auto commandResult = commands.synchronize({.buffers = {&outputBarrier, 1}}); !commandResult) { return commandResult; }
            const size_t permutation = (context.frameIndex() & 1u) && (bin & 1u) ? 2 : 1;
            result = (*bins.arguments).slice({uint64_t(bin) * 12, 12}).and_then([&](const auto& bufferSlice) { return kernels_[permutation].dispatchIndirect(commands, encoded, bufferSlice); });
            if (!result) { return result; }
        }
        return {};
    }
    bool fixture_;
    bool typed_;
    uint32_t materialCount_;
    render::Device* device_ = nullptr;
    std::array<render::ComputeProgram, 3> programs_;
    render::ComputeProgram incompatibleProgram_;
    std::array<render::ComputeKernel, 3> kernels_;
    render::MaterialBinning binning_;
};

// Keep millions of per-pixel atomics in device memory. Marking Probe.data as
// an output makes the graph allocate HostReadback memory (PCIe atomics).
// Copy once after all consumers, preserving the exact coverage assertions.
class MaterialBinningReadbackPass final : public render::ComputePass {
public:
    explicit MaterialBinningReadbackPass(uint32_t binCount = kBinCount) : binCount_(binCount) {}
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        const auto bytes = (uint64_t(context.width) * context.height * 3 + binCount_ * 5 + 2) * 4;
        render::RenderPassReflection reflection;
        reflection.addBufferInput("source").buffer(bytes, 4).transferRead();
        reflection.addBufferOutput("data").buffer(bytes, 4).transferWrite().hostReadback();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        auto source = context.inputBuffer("source").buffer()->slice();
        auto destination = context.outputBuffer("data").buffer()->slice();
        if (!source) { return std::unexpected(source.error()); }
        if (!destination) { return std::unexpected(destination.error()); }
        return context.commandBuffer().copyBuffer(*source, *destination);
    }
private:
    uint32_t binCount_;
};

class MaterialBinningTest : public RHITest {
public:
    explicit MaterialBinningTest(bool typed = false) : typed_(typed)
    {
        type = RHITestType::Rendering;
        name = typed ? "material_binning_typed_indirect_coverage" : "material_binning_indirect_coverage";
    }

    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "Material binning probe",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
        if (!result) { return RHITestResult::fail("Device creation failed"); }
        if (!device->capabilities().computeSubgroupBallotArithmetic || device->capabilities().subgroupSize != 32) {
            return RHITestResult::skip("Requires native wave32 with subgroup ballot and arithmetic");
        }
        render::registerRenderGraphPassType("MaterialBinFixture", "Fixture",
            [] { return std::make_unique<MaterialBinningProbePass>(true); });
        render::registerRenderGraphPassType("MaterialBinProbe", "Probe",
            [typed = typed_] { return std::make_unique<MaterialBinningProbePass>(false, typed); });
        render::registerRenderGraphPassType("MaterialBinReadback", "Readback",
            [] { return std::make_unique<MaterialBinningReadbackPass>(); });
        render::RenderGraph graph;
        graph.addNode("MaterialBinFixture", "Fixture");
        graph.addNode("MaterialBinProbe", "Probe");
        for (const char* field : {"visibility", "records", "instances", "materials", "shadingMaterials"}) {
            graph.addEdge(std::string("Fixture.") + field, std::string("Probe.") + field);
        }
        graph.addNode("MaterialBinReadback", "Readback");
        graph.addEdge("Probe.data", "Readback.source");
        graph.markOutput("Readback.data");
        render::RenderGraphExecutor executor;
        std::string log;
        for (auto extent : {std::array<uint32_t, 2>{63, 37}, {17, 9}, {1, 1}, {8, 4}, {193, 157}, {4097, 1025}}) {
            const auto [width, height] = extent;
            if (!executor.compile(*device, graph, width, height, log)) { return RHITestResult::fail(log); }
            // Repeat without recompiling: pooled scratch must reset counts each frame.
            for (uint32_t frame = 0; frame < 3; ++frame) {
                result = executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)});
                if (!result) { return RHITestResult::fail(std::string("Binning dispatch: ") + toString(result)); }
                result = executor.waitForSubmittedWork(5'000'000'000ull);
                if (!result) {
                    return RHITestResult::fail("Binning wait " + std::to_string(width) + "x" +
                        std::to_string(height) + " frame=" + std::to_string(frame) + ": " + toString(result));
                }
                auto* buffer = executor.outputResource("Readback.data")->buffer;
                buffer->invalidate();
                void* mapped = buffer->map();
                if (mapped == nullptr) { return RHITestResult::fail("Binning readback failed"); }
                std::vector<uint32_t> values(buffer->desc().size / 4);
                std::memcpy(values.data(), mapped, buffer->desc().size);
                buffer->unmap();
                if (values[kProbeHeader - 1] != 0) { return RHITestResult::fail("Invalid tile index, empty task mask or out-of-bounds active lane"); }
                const uint32_t phase = values[kProbeHeader - 2];
                const uint32_t columns = (width + 7) / 8, rows = (height + 3) / 4;
                std::vector<uint32_t> tileClasses(columns * rows, 0);
                for (uint32_t pixel = 0; pixel < width * height; ++pixel) {
                    const uint32_t kind = width > 4096 ? 0 : pixel % 262;
                    const uint32_t source = (((kind * 17) % 257) * 73) % 257;
                    uint32_t expected = kind < 257 && width != 8 ? 1 + (source + phase) % 4 : 0;
                    if (expected != 0 && (source == 1 || source == 2)) { expected = 3; }
                    const uint32_t offset = kProbeHeader + pixel * 3;
                    if (values[offset] != expected || values[offset + 1] != 1 || values[offset + 2] != 32u + (((phase & 1u) != 0 && (expected & 1u) != 0) ? 100u : 0u)) {
                        return RHITestResult::fail("Pixel missing, duplicated, non-wave32 or in wrong feature class: " + std::to_string(pixel) +
                            " expectedClass=" + std::to_string(expected) + " actualClass=" + std::to_string(values[offset]) +
                            " writes=" + std::to_string(values[offset + 1]) + " wave=" + std::to_string(values[offset + 2]));
                    }
                    tileClasses[(pixel / width / 4) * columns + (pixel % width / 8)] |= 1u << expected;
                }
                for (uint32_t bin = 0; bin < kBinCount; ++bin) {
                    uint32_t expectedTasks = 0;
                    for (uint32_t classes : tileClasses) { expectedTasks += (classes & (1u << bin)) != 0; }
                    const uint32_t offset = values[bin * 5], count = values[bin * 5 + 1];
                    if (offset != bin * columns * rows || count != expectedTasks ||
                        values[bin * 5 + 2] != std::min(count, 65535u) ||
                        values[bin * 5 + 3] != (count + 65534) / 65535 || values[bin * 5 + 4] != 1) {
                        return RHITestResult::fail("Invalid tile list, duplicate class tasks or indirect groups");
                    }
                }
            }
        }
        return RHITestResult::pass("257 materials in five feature classes; exact wave32 tile masks, edges, background, texture/NTC conservatism, feature edits, resize/reuse and 2D indirect coverage");
    }
private:
    bool typed_;
};

METALLIC_REGISTER_RHI_TEST(MaterialBinningTest);
class TypedMaterialBinningTest final : public MaterialBinningTest {
public:
    TypedMaterialBinningTest() : MaterialBinningTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(TypedMaterialBinningTest);

struct ProgramBinExecutables
{
    std::vector<render::ComputeProgram> programs;
    render::ComputeProgram reset;
    std::vector<std::shared_ptr<const render::MaterialExecutableArtifact>> artifacts;
};

class ProgramBinningProbePass final : public render::UnsafePass
{
public:
    ProgramBinningProbePass(uint32_t count, bool sparse, std::shared_ptr<ProgramBinExecutables> executables)
        : count_(count), sparse_(sparse), executables_(std::move(executables)) {}
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("visibility").sampledRead();
        for (const char* name : {"records", "instances", "materials", "shadingMaterials"}) {
            reflection.addBufferInput(name).shaderRead();
        }
        reflection.addBufferOutput("data").buffer(
            (uint64_t(context.width) * context.height * 3 + count_ * 5 + 2) * 4, 4).storageReadWrite();
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        device_ = context.device;
        if (executables_->reset.valid()) { return {}; }
        const ComputeResourceBindingDesc bindings[] = {{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, bins)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, tiles)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, arguments)}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, output)}};
        const ResourceComputeKernelDesc layout{.pushConstantSize = 16, .bindings = bindings, .requiresRayQuery = false,
            .resourceParameterSize = sizeof(metallic::tests::MaterialBinningProbeResources)};
        executables_->programs.resize(count_);
        for (uint32_t program = 0; program <= count_; ++program) {
            const std::string binCount = std::to_string(count_), slot = std::to_string(program % count_);
            const SlangMacroDefine defines[] = {{"PROBE_BIN_COUNT", binCount.c_str()}, {"PROBE_STATIC_PROGRAM", slot.c_str()}};
            const SlangShaderDesc source{.moduleName = "ProgramBinningProbe",
                .entryPointName = program == count_ ? "materialReadbackResetMain" : "materialIndirectProbeMain",
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = defines};
            std::shared_ptr<const MaterialExecutableArtifact> artifact;
            auto& target = program == count_ ? executables_->reset : executables_->programs[program];
            auto result = compileMaterialExecutable(*device_, source, layout, target, artifact, log);
            if (!result) { return result; }
            executables_->artifacts.push_back(std::move(artifact));
        }
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto& commands = context.commandBuffer();
        const uint32_t phase = uint32_t(context.frameIndex() % 4);
        // Mapping changes are instance data. They never compile/create a Program.
        std::vector<uint32_t> mapping(context.inputBuffer("shadingMaterials").buffer()->desc().size / 720);
        for (uint32_t i = 0; i < mapping.size(); ++i) {
            mapping[i] = count_ == 5 ? (i == 1 || i == 2 ? 3 : 1 + (i + phase) % 4) : 1 + (i + phase) % (count_ - 1);
        }
        MaterialBinningResult bins;
        std::string log;
        if (sparse_ && context.frameIndex() == 0) {
            auto invalid = mapping;
            invalid[0] = count_;
            auto rejected = binning_.record(*device_, commands, {
                .visibility = context.inputTexture("visibility").view(), .records = context.inputBuffer("records").buffer(),
                .instances = context.inputBuffer("instances").buffer(), .materials = context.inputBuffer("materials").buffer(),
                .shadingMaterials = context.inputBuffer("shadingMaterials").buffer(),
                .width = context.width(), .height = context.height(), .materialProgramBins = invalid, .programBinCount = count_}, log);
            if (!hasError(rejected, Error::InvalidArgument)) { return makeError(Error::Failure); }
        }
        {
            auto timing = context.profileScope("Program classification");
            auto result = binning_.record(*device_, commands, {
                .visibility = context.inputTexture("visibility").view(), .records = context.inputBuffer("records").buffer(),
                .instances = context.inputBuffer("instances").buffer(), .materials = context.inputBuffer("materials").buffer(),
                .shadingMaterials = context.inputBuffer("shadingMaterials").buffer(),
                .width = context.width(), .height = context.height(),
                .materialProgramBins = sparse_ ? std::span<const uint32_t>(mapping) : std::span<const uint32_t>{},
                .programBinCount = sparse_ ? count_ : 0}, log);
            if (!result) { return makeError(result.error()); }
            bins = *result;
        }
        uint32_t push[]{context.width(), context.height(), count_, phase};
        const ComputeDispatchBinding bindings[] = {{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, bins), .buffer = bins.bins}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, tiles), .buffer = bins.tiles},
            {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, arguments), .buffer = bins.arguments}, {.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::MaterialBinningProbeResources, output), .buffer = context.outputBuffer("data").buffer()}};
        BufferBarrierDesc argumentBarrier{.buffer = bins.arguments,
            .before = {PipelineStageBits::DrawIndirect, AccessBits::IndirectRead},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead}};
        auto result = commands.synchronize({.buffers = {&argumentBarrier, 1}});
        if (!result) { return result; }
        const uint32_t resetGroups = (std::max(push[0] * push[1], count_) + 63) / 64;
        result = executables_->reset.dispatch({.commandBuffer = &commands, .bindings = bindings, .pushData = push,
            .pushDataSize = sizeof(push), .groupCountX = std::min(resetGroups, 65535u),
            .groupCountY = (resetGroups + 65534) / 65535});
        if (!result) { return result; }
        std::swap(argumentBarrier.before, argumentBarrier.after);
        result = commands.synchronize({.buffers = {&argumentBarrier, 1}});
        if (!result) { return result; }
        const BufferBarrierDesc outputBarrier{.buffer = context.outputBuffer("data").buffer(),
            .before = {PipelineStageBits::ComputeShader, AccessBits::ShaderWrite},
            .after = {PipelineStageBits::ComputeShader, AccessBits::ShaderRead | AccessBits::ShaderWrite}};
        result = commands.synchronize({.buffers = {&outputBarrier, 1}});
        if (!result) { return result; }
        std::vector<std::array<uint32_t, 4>> constants(count_);
        std::vector<TestComputeIndirectDispatch> dispatches(count_);
        for (uint32_t bin = 0; bin < count_; ++bin) {
            constants[bin] = {push[0], push[1], count_, bin};
            dispatches[bin] = {.pushData = constants[bin].data(), .argumentOffset = uint64_t(bin) * 12,
                .program = &executables_->programs[bin]};
        }
        return executables_->programs[0].dispatchIndirectBatch({.commandBuffer = &commands, .bindings = bindings,
            .pushDataSize = sizeof(push), .indirectArguments = bins.arguments}, dispatches);
    }
private:
    uint32_t count_;
    bool sparse_;
    std::shared_ptr<ProgramBinExecutables> executables_;
    render::Device* device_ = nullptr;
    render::MaterialBinning binning_;
};

class ProgramBinningTest final : public RHITest
{
public:
    ProgramBinningTest() { type = RHITestType::Rendering; name = "material_program_binning_sparse"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        auto created = createDevice({.applicationName = "Sparse Program binning", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true});
        if (!created) { return RHITestResult::fail("Program binning device failed"); }
        auto device = std::move(*created);
        if (!device->capabilities().computeSubgroupBallotArithmetic || device->capabilities().subgroupSize != 32) {
            return RHITestResult::skip("Requires native wave32");
        }
        std::filesystem::create_directories(context.outputDirectory);
        std::ofstream report(context.outputDirectory / "ProgramBinning.txt");
        auto many = std::make_shared<ProgramBinExecutables>(), five = std::make_shared<ProgramBinExecutables>();
        for (uint32_t mode = 0; mode < 3; ++mode) {
            const uint32_t bins = mode == 0 ? 49 : 5, header = bins * 5 + 2;
            const bool sparse = mode != 1;
            auto executables = mode == 0 ? many : five;
            const std::weak_ptr<ProgramBinExecutables> weakExecutables = executables;
            registerRenderGraphPassType("ProgramBinProbe", "Program probe", [=] {
                return std::make_unique<ProgramBinningProbePass>(bins, sparse, weakExecutables.lock()); });
            registerRenderGraphPassType("ProgramBinReadback", "Program readback", [=] {
                return std::make_unique<MaterialBinningReadbackPass>(bins); });
            registerRenderGraphPassType("ProgramBinFixture", "Program fixture", [] {
                return std::make_unique<MaterialBinningProbePass>(true); });
            RenderGraph graph;
            graph.addNode("ProgramBinFixture", "Fixture"); graph.addNode("ProgramBinProbe", "Probe");
            for (const char* field : {"visibility", "records", "instances", "materials", "shadingMaterials"}) {
                graph.addEdge(std::string("Fixture.") + field, std::string("Probe.") + field);
            }
            graph.addNode("ProgramBinReadback", "Readback"); graph.addEdge("Probe.data", "Readback.source");
            graph.markOutput("Readback.data");
            // Identical 5-program workloads are measured on both paths; the
            // 49-program case additionally checks growth, tails and 2D dispatch.
            for (auto extent : {std::array<uint32_t, 3>{1, 1, 257}, {8, 4, 257}, {17, 9, 257},
                    {512, 256, 257}, {513, 257, 4097}, {4097, 1025, 4097}}) {
                const auto [width, height, materials] = extent;
                registerRenderGraphPassType("ProgramBinFixture", "Program fixture", [=] {
                    return std::make_unique<MaterialBinningProbePass>(true, false, materials); });
                RenderGraphExecutor executor;
                std::string log;
                if (!executor.compile(*device, graph, width, height, log)) { return RHITestResult::fail(log); }
                const auto builds = materialProgramCacheStats().pipelineBuilds;
                for (uint32_t frame = 0; frame < 8; ++frame) {
                    auto result = executor.execute({.graphicsQueue = device->getQueue(QueueType::Graphics)});
                    if (!result || !executor.waitForSubmittedWork(5'000'000'000ull)) { return RHITestResult::fail("Program binning execution failed"); }
                    if (materialProgramCacheStats().pipelineBuilds != builds) { return RHITestResult::fail("Instance edit created a pipeline"); }
                    auto* buffer = executor.outputResource("Readback.data")->buffer;
                    buffer->invalidate();
                    const auto* data = static_cast<const uint32_t*>(buffer->map());
                    if (!data) { return RHITestResult::fail("Program readback mapping failed"); }
                    std::vector<uint32_t> values(data, data + buffer->desc().size / 4);
                    buffer->unmap();
                    if (values[header - 1] != 0) { return RHITestResult::fail("Invalid sparse task/mask"); }
                    const uint32_t phase = values[header - 2], columns = (width + 7) / 8, rows = (height + 3) / 4;
                    std::vector<uint64_t> tilePrograms(columns * rows, 0);
                    for (uint32_t pixel = 0; pixel < width * height; ++pixel) {
                        const uint32_t kind = width > 4096 ? 0 : pixel % (materials + 5);
                        const uint32_t source = (((kind * 17) % materials) * 73) % materials;
                        uint32_t expected = kind < materials && width != 8 ? 1 + (source + phase) % (bins - 1) : 0;
                        if (bins == 5 && expected && (source == 1 || source == 2)) { expected = 3; }
                        const uint32_t offset = header + pixel * 3;
                        const float albedo = float(expected) / 64 * (float(1 + pixel % 11) / 11);
                        const float radiance = albedo * (0.25f + (expected % 2 == 0 ? 1 / 3.14159265358979323846f : 0));
                        if (values[offset] != expected || values[offset + 1] != 1 ||
                            !std::isfinite(std::bit_cast<float>(values[offset + 2])) ||
                            std::abs(std::bit_cast<float>(values[offset + 2]) - radiance) > 2e-6f) {
                            return RHITestResult::fail("Program/mask/fused lighting mismatch pixel=" + std::to_string(pixel) +
                                " expected=" + std::to_string(expected) + " actual=" + std::to_string(values[offset]) +
                                " writes=" + std::to_string(values[offset + 1]));
                        }
                        tilePrograms[(pixel / width / 4) * columns + (pixel % width / 8)] |= uint64_t(1) << expected;
                    }
                    uint32_t end = 0;
                    for (uint32_t bin = 0; bin < bins; ++bin) {
                        uint32_t expectedTasks = 0;
                        for (auto mask : tilePrograms) { expectedTasks += (mask & (uint64_t(1) << bin)) != 0; }
                        const uint32_t count = values[bin * 5 + 1];
                        if (values[bin * 5] != (sparse ? end : bin * columns * rows) || count != expectedTasks ||
                            values[bin * 5 + 2] != std::min(count, 65535u) || values[bin * 5 + 3] != (count + 65534) / 65535 ||
                            values[bin * 5 + 4] != 1) { return RHITestResult::fail("Sparse offsets/counts/indirect arguments mismatch"); }
                        end += count;
                    }
                    if (mode == 0 && width == 512 && frame == 0) {
                        std::vector<uint8_t> image(width * height * 4);
                        for (uint32_t pixel = 0; pixel < width * height; ++pixel) {
                            const float value = std::bit_cast<float>(values[header + pixel * 3 + 2]);
                            const uint8_t display = uint8_t(std::lround(255 * std::pow(std::clamp(value, 0.0f, 1.0f), 1 / 2.2f)));
                            for (uint32_t c = 0; c < 3; ++c) { image[pixel * 4 + c] = display; }
                            image[pixel * 4 + 3] = 255;
                        }
                        if (!saveRgba8Png(context.outputDirectory / "ProgramLighting.png", image.data(), width, height, log)) {
                            return RHITestResult::fail(log);
                        }
                    }
                    auto completed = executor.collectCompletedGpuExecutionStats();
                    if (!completed || completed->empty()) { return RHITestResult::fail("GPU timestamps were not collected"); }
                    for (const auto& node : completed->back().nodes) {
                        for (const auto& section : node.sections) {
                            if (section.name == "Program classification" && frame >= 2) {
                                if (!section.gpuTimingAvailable) { return RHITestResult::fail("GPU binning timestamps unavailable"); }
                                report << "bins=" << bins << " sparse=" << sparse << " width=" << width << " height=" << height
                                    << " instances=" << materials << " frame=" << frame << " tasks=" << end
                                    << " gpuMs=" << section.gpuMilliseconds << '\n';
                            }
                        }
                    }
                }
            }
        }
        if (many->artifacts.size() != 50 || five->artifacts.size() != 6) { return RHITestResult::fail("Per-instance Program growth"); }
        return RHITestResult::pass("48 static Surface Programs plus background; sparse mixed tiles shade once; 257/4097 instances share executables; fixed-five A/B timings recorded");
    }
};
METALLIC_REGISTER_RHI_TEST(ProgramBinningTest);

} // namespace
} // namespace metallic::tests

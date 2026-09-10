#include "RhiTest.h"
#include "Runtime/Render/MaterialBinning.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/SlangCompiler.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace metallic::tests {
namespace {

constexpr uint32_t kBinCount = render::kMaterialClassCount;
constexpr uint32_t kProbeHeader = kBinCount * 5 + 2;

class MaterialBinningProbePass final : public render::UnsafePass {
public:
    explicit MaterialBinningProbePass(bool fixture) : fixture_(fixture) {}

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        if (fixture_) {
            reflection.addTextureOutput("visibility").storageReadWrite().format = render::Format::R32Uint;
            reflection.addBufferOutput("records").buffer(260 * 16, 16).storageReadWrite();
            reflection.addBufferOutput("instances").buffer(260 * 160, 160).storageReadWrite();
            reflection.addBufferOutput("materials").buffer(257 * 560, 560).storageReadWrite();
            reflection.addBufferOutput("shadingMaterials").buffer(257 * 608, 608).storageReadWrite();
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

    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        const render::ComputeProgramBindingDesc layout[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = 1}, {.binding = 2}, {.binding = 3},
            {.binding = 4}, {.binding = 5}, {.binding = 6}, {.binding = 7}, {.binding = 8}};
        const char* entries[] = {fixture_ ? "materialFixtureMain" : "materialReadbackResetMain",
            "materialIndirectProbeMain", "materialIndirectProbeMain"};
        for (uint32_t i = 0; i < (fixture_ ? 1u : 3u); ++i) {
            const render::SlangMacroDefine alternate[] = {{"PROBE_ALTERNATE", "1"}};
            render::ShaderCompileResult shader;
            auto result = render::compileSlangShaderToSpirv({.moduleName = "MaterialBinningProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = alternate, .macroDefineCount = i == 2 ? 1u : 0u}, shader);
            if (!result) { log = shader.diagnostics; return result; }
            result = programs_[i].initialize(*device_, {.spirv = shader.spirv.data(),
                .byteSize = shader.spirv.size() * 4, .pushConstantSize = 16,
                .bindings = fixture_ ? layout : layout + 5, .bindingCount = fixture_ ? 5u : 4u, .descriptorSetCount = !fixture_ && i == 0 ? 2u : 1u, .requiresRayQuery = false}, log);
            if (!result) { return result; }
        }
        return {};
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        auto& commands = context.commandBuffer();
        uint32_t push[] = {context.width(), context.height(), kBinCount, static_cast<uint32_t>(context.frameIndex() % 4)};
        const uint32_t pixels = push[0] * push[1];
        const uint32_t fixtureGroups = (std::max(pixels, 260u) + 63) / 64;
        const uint32_t readbackGroups = (std::max(pixels, kBinCount) + 63) / 64;
        if (fixture_) {
            const render::ComputeDispatchBinding bindings[] = {
                {.binding = 0, .textureView = context.outputTexture("visibility").view()},
                {.binding = 1, .buffer = context.outputBuffer("records").buffer()},
                {.binding = 2, .buffer = context.outputBuffer("instances").buffer()},
                {.binding = 3, .buffer = context.outputBuffer("materials").buffer()},
                {.binding = 4, .buffer = context.outputBuffer("shadingMaterials").buffer()}};
            return programs_[0].dispatch({.commandBuffer = &commands, .bindings = bindings, .bindingCount = 5,
                .pushData = push, .pushDataSize = sizeof(push), .groupCountX = std::min(fixtureGroups, 65535u), .groupCountY = (fixtureGroups + 65534) / 65535});
        }
        render::MaterialBinningResult bins;
        std::string log;
        auto result = binning_.record(*device_, commands, {
            .visibility = context.inputTexture("visibility").view(),
            .records = context.inputBuffer("records").buffer(),
            .instances = context.inputBuffer("instances").buffer(),
            .materials = context.inputBuffer("materials").buffer(),
            .shadingMaterials = context.inputBuffer("shadingMaterials").buffer(),
            .width = push[0], .height = push[1]}, bins, log);
        if (!result) { return result; }
        // Invalid inputs must fail before recording vkCmdDispatchIndirect.
        for (uint64_t offset : {uint64_t(1), bins.arguments->desc().size - 4, UINT64_MAX}) {
            if (!render::hasError(commands.dispatchIndirect(*bins.arguments, offset), render::Error::InvalidArgument)) {
                return render::makeError(render::Error::Failure);
            }
        }
        if (!render::hasError(commands.dispatchIndirect(*bins.bins), render::Error::InvalidArgument)) {
            return render::makeError(render::Error::Failure);
        }
        // Read argument bytes as shader data, then restore their indirect state.
        render::BufferBarrierDesc argumentBarrier{.buffer = bins.arguments,
            .before = render::ResourceState::IndirectArgument, .after = render::ResourceState::ShaderRead};
        commands.barrier({.buffers = &argumentBarrier, .bufferCount = 1});
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 5, .buffer = bins.bins}, {.binding = 6, .buffer = bins.tiles},
            {.binding = 7, .buffer = bins.arguments}, {.binding = 8, .buffer = context.outputBuffer("data").buffer()}};
        render::ComputeDispatchDesc dispatch{.commandBuffer = &commands, .bindings = bindings, .bindingCount = 4,
            .pushData = push, .pushDataSize = sizeof(push), .groupCountX = std::min(readbackGroups, 65535u), .groupCountY = (readbackGroups + 65534) / 65535};
        result = programs_[0].dispatch(dispatch);
        if (!result) { return result; }
        std::swap(argumentBarrier.before, argumentBarrier.after);
        commands.barrier({.buffers = &argumentBarrier, .bufferCount = 1});
        render::BufferBarrierDesc outputBarrier{.buffer = context.outputBuffer("data").buffer(),
            .before = render::ResourceState::General, .after = render::ResourceState::General};
        dispatch.indirectArguments = bins.arguments;
        // Reject an incompatible permutation before descriptor writes or GPU work.
        const render::ComputeIndirectDispatch incompatible[] = {{.pushData = push, .program = &programs_[0]}};
        if (!render::hasError(programs_[1].dispatchIndirectBatch(dispatch, incompatible), render::Error::InvalidArgument)) {
            return render::makeError(render::Error::Failure);
        }
        if ((context.frameIndex() & 1u) != 0) {
            std::vector<std::array<uint32_t, 4>> pushes(bins.binCount);
            std::vector<render::ComputeIndirectDispatch> items(bins.binCount);
            for (uint32_t bin = 0; bin < bins.binCount; ++bin) {
                pushes[bin] = {push[0], push[1], push[2], bin};
                items[bin] = {.pushData = pushes[bin].data(), .argumentOffset = uint64_t(bin) * 12,
                    .program = (bin & 1u) != 0 ? &programs_[2] : nullptr};
            }
            commands.barrier({.buffers = &outputBarrier, .bufferCount = 1});
            return programs_[1].dispatchIndirectBatch(dispatch, items, {.buffers = &outputBarrier, .bufferCount = 1});
        }
        for (uint32_t bin = 0; bin < bins.binCount; ++bin) {
            commands.barrier({.buffers = &outputBarrier, .bufferCount = 1});
            push[3] = bin;
            dispatch.indirectOffset = uint64_t(bin) * 12;
            result = programs_[1].dispatch(dispatch);
            if (!result) { return result; }
        }
        return {};
    }

private:
    bool fixture_;
    render::Device* device_ = nullptr;
    std::array<render::ComputeProgram, 3> programs_;
    render::MaterialBinning binning_;
};

class MaterialBinningTest final : public RhiTest {
public:
    MaterialBinningTest() { type = RhiTestType::Rendering; name = "material_binning_indirect_coverage"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "Material binning probe",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("Requires bindless descriptors"); }
        if (!result) { return RhiTestResult::fail("Device creation failed"); }
        if (!device->capabilities().computeSubgroupBallotArithmetic || device->capabilities().subgroupSize != 32) {
            return RhiTestResult::skip("Requires native wave32 with subgroup ballot and arithmetic");
        }
        render::registerRenderGraphPassType("MaterialBinFixture", "Fixture",
            [] { return std::make_unique<MaterialBinningProbePass>(true); });
        render::registerRenderGraphPassType("MaterialBinProbe", "Probe",
            [] { return std::make_unique<MaterialBinningProbePass>(false); });
        render::RenderGraph graph;
        graph.addNode("MaterialBinFixture", "Fixture");
        graph.addNode("MaterialBinProbe", "Probe");
        for (const char* field : {"visibility", "records", "instances", "materials", "shadingMaterials"}) {
            graph.addEdge(std::string("Fixture.") + field, std::string("Probe.") + field);
        }
        graph.markOutput("Probe.data");
        render::RenderGraphExecutor executor;
        std::string log;
        for (auto extent : {std::array<uint32_t, 2>{63, 37}, {17, 9}, {1, 1}, {8, 4}, {193, 157}, {4097, 1025}}) {
            const auto [width, height] = extent;
            if (!executor.compile(*device, graph, width, height, log)) { return RhiTestResult::fail(log); }
            // Repeat without recompiling: pooled scratch must reset counts each frame.
            for (uint32_t frame = 0; frame < 3; ++frame) {
                result = executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)});
                if (!result) { return RhiTestResult::fail(std::string("Binning dispatch: ") + toString(result)); }
                if (!executor.waitForSubmittedWork(5'000'000'000ull)) { return RhiTestResult::fail("Binning wait failed"); }
                auto* buffer = executor.outputResource("Probe.data")->buffer;
                buffer->invalidate();
                void* mapped = buffer->map();
                if (mapped == nullptr) { return RhiTestResult::fail("Binning readback failed"); }
                std::vector<uint32_t> values(buffer->desc().size / 4);
                std::memcpy(values.data(), mapped, buffer->desc().size);
                buffer->unmap();
                if (values[kProbeHeader - 1] != 0) { return RhiTestResult::fail("Invalid tile index, empty task mask or out-of-bounds active lane"); }
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
                        return RhiTestResult::fail("Pixel missing, duplicated, non-wave32 or in wrong feature class: " + std::to_string(pixel));
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
                        return RhiTestResult::fail("Invalid tile list, duplicate class tasks or indirect groups");
                    }
                }
            }
        }
        return RhiTestResult::pass("257 materials in five feature classes; exact wave32 tile masks, edges, background, texture/NTC conservatism, feature edits, resize/reuse and 2D indirect coverage");
    }
};

METALLIC_REGISTER_RHI_TEST(MaterialBinningTest);

} // namespace
} // namespace metallic::tests

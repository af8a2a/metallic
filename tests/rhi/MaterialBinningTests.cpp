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

constexpr uint32_t kBinCount = 258;

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
        } else {
            reflection.addTextureInput("visibility").sampledRead();
            for (const char* name : {"records", "instances", "materials"}) {
                reflection.addBufferInput(name).shaderRead();
            }
            reflection.addBufferOutput("data").buffer(
                (uint64_t(context.width) * context.height * 2 + kBinCount * 5) * 4, 4).storageReadWrite();
        }
        return reflection;
    }

    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        const render::ComputeProgramBindingDesc layout[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = 1}, {.binding = 2}, {.binding = 3},
            {.binding = 4}, {.binding = 5}, {.binding = 6}, {.binding = 7}};
        const char* entries[] = {fixture_ ? "materialFixtureMain" : "materialReadbackResetMain",
            "materialIndirectProbeMain"};
        for (uint32_t i = 0; i < (fixture_ ? 1u : 2u); ++i) {
            render::ShaderCompileResult shader;
            auto result = render::compileSlangShaderToSpirv({.moduleName = "MaterialBinningProbe",
                .entryPointName = entries[i], .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
            if (!result) { log = shader.diagnostics; return result; }
            result = programs_[i].initialize(*device_, {.spirv = shader.spirv.data(),
                .byteSize = shader.spirv.size() * 4, .pushConstantSize = 16,
                .bindings = fixture_ ? layout : layout + 4, .bindingCount = 4, .requiresRayQuery = false}, log);
            if (!result) { return result; }
        }
        return {};
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        auto& commands = context.commandBuffer();
        uint32_t push[] = {context.width(), context.height(), kBinCount, 0};
        const uint32_t pixels = push[0] * push[1];
        if (fixture_) {
            const render::ComputeDispatchBinding bindings[] = {
                {.binding = 0, .textureView = context.outputTexture("visibility").view()},
                {.binding = 1, .buffer = context.outputBuffer("records").buffer()},
                {.binding = 2, .buffer = context.outputBuffer("instances").buffer()},
                {.binding = 3, .buffer = context.outputBuffer("materials").buffer()}};
            return programs_[0].dispatch({.commandBuffer = &commands, .bindings = bindings, .bindingCount = 4,
                .pushData = push, .pushDataSize = sizeof(push), .groupCountX = (std::max(pixels, 260u) + 63) / 64});
        }
        render::MaterialBinningResult bins;
        std::string log;
        auto result = binning_.record(*device_, commands, {
            .visibility = context.inputTexture("visibility").view(),
            .records = context.inputBuffer("records").buffer(),
            .instances = context.inputBuffer("instances").buffer(),
            .materials = context.inputBuffer("materials").buffer(),
            .width = push[0], .height = push[1], .materialCount = kBinCount - 1}, bins, log);
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
            {.binding = 4, .buffer = bins.bins}, {.binding = 5, .buffer = bins.pixels},
            {.binding = 6, .buffer = bins.arguments}, {.binding = 7, .buffer = context.outputBuffer("data").buffer()}};
        render::ComputeDispatchDesc dispatch{.commandBuffer = &commands, .bindings = bindings, .bindingCount = 4,
            .pushData = push, .pushDataSize = sizeof(push), .groupCountX = (std::max(pixels, kBinCount) + 63) / 64};
        result = programs_[0].dispatch(dispatch);
        if (!result) { return result; }
        std::swap(argumentBarrier.before, argumentBarrier.after);
        commands.barrier({.buffers = &argumentBarrier, .bufferCount = 1});
        render::BufferBarrierDesc outputBarrier{.buffer = context.outputBuffer("data").buffer(),
            .before = render::ResourceState::General, .after = render::ResourceState::General};
        dispatch.indirectArguments = bins.arguments;
        if ((context.frameIndex() & 1u) != 0) {
            std::vector<std::array<uint32_t, 4>> pushes(bins.binCount);
            std::vector<render::ComputeIndirectDispatch> items(bins.binCount);
            for (uint32_t bin = 0; bin < bins.binCount; ++bin) {
                pushes[bin] = {push[0], push[1], push[2], bin};
                items[bin] = {.pushData = pushes[bin].data(), .argumentOffset = uint64_t(bin) * 12};
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
    std::array<render::ComputeProgram, 2> programs_;
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
        if (!device->capabilities().computeSubgroupBallotArithmetic) {
            return RhiTestResult::skip("Requires compute subgroup ballot and arithmetic");
        }
        render::registerRenderGraphPassType("MaterialBinFixture", "Fixture",
            [] { return std::make_unique<MaterialBinningProbePass>(true); });
        render::registerRenderGraphPassType("MaterialBinProbe", "Probe",
            [] { return std::make_unique<MaterialBinningProbePass>(false); });
        render::RenderGraph graph;
        graph.addNode("MaterialBinFixture", "Fixture");
        graph.addNode("MaterialBinProbe", "Probe");
        for (const char* field : {"visibility", "records", "instances", "materials"}) {
            graph.addEdge(std::string("Fixture.") + field, std::string("Probe.") + field);
        }
        graph.markOutput("Probe.data");
        render::RenderGraphExecutor executor;
        std::string log;
        for (auto extent : {std::array<uint32_t, 2>{63, 37}, {17, 9}, {1, 1}, {193, 157}, {4097, 1025}}) {
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
                std::array<uint32_t, kBinCount> counts{};
                for (uint32_t pixel = 0; pixel < width * height; ++pixel) {
                    uint32_t kind = width > 4096 ? 0 : pixel % 262;
                    uint32_t expected = kind < 257 ? (((kind * 17) % 257) * 73) % 257 + 1 : 0;
                    ++counts[expected];
                    if (values[kBinCount * 5 + pixel * 2] != expected || values[kBinCount * 5 + pixel * 2 + 1] != 1) {
                        return RhiTestResult::fail("Pixel missing, duplicated or in wrong material bin: " + std::to_string(pixel));
                    }
                }
                std::vector<bool> occupied(width * height, false);
                for (uint32_t bin = 0; bin < kBinCount; ++bin) {
                    uint32_t offset = values[bin * 5], count = values[bin * 5 + 1];
                    uint32_t groups = (count + 63) / 64;
                    if (count != counts[bin] || offset > occupied.size() || count > occupied.size() - offset ||
                        values[bin * 5 + 2] != std::min(groups, 65535u) || values[bin * 5 + 3] != (groups + 65534) / 65535 ||
                        values[bin * 5 + 4] != 1) { return RhiTestResult::fail("Invalid bin metadata/indirect groups"); }
                    for (uint32_t i = offset; i < offset + count; ++i) {
                        if (occupied[i]) { return RhiTestResult::fail("Overlapping material queues"); }
                        occupied[i] = true;
                    }
                }
                if (std::find(occupied.begin(), occupied.end(), false) != occupied.end()) {
                    return RhiTestResult::fail("Material queues do not cover the pixel list");
                }
            }
        }
        return RhiTestResult::pass("258 bins, partial waves, empty bins, invalid IDs, remaps, resize, reuse and 2D indirect dispatch coverage");
    }
};

METALLIC_REGISTER_RHI_TEST(MaterialBinningTest);

} // namespace
} // namespace metallic::tests

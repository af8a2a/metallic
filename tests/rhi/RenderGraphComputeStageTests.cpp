#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <array>
#include <vector>

namespace metallic::tests {
namespace {

enum class StageProbeCase : uint32_t {
    LateUnknownResource,
    LateMissingRecorder,
    HiddenBufferWrite,
    HiddenImageWrite,
    ImageLayoutChange,
    GraphAllocationImport,
    ConflictingImportAliases,
    HiddenImportWrite,
    NonComputeAccess,
    RasterParentScope,
    PrivateAliases,
    RepeatedScope,
    NestedScope,
    ForkInsideScope,
};

struct StageProbe {
    uint32_t callbacks = 0;
    uint32_t unexpectedCallbacks = 0;
    std::vector<uint64_t> barriersAtCallbacks;
};

StageProbe* activeStageProbe = nullptr;

class ComputeStageProbePass final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    render::RenderGraphPassKind kind() const override
    {
        return scenario() == StageProbeCase::RasterParentScope
            ? render::RenderGraphPassKind::Raster : render::RenderGraphPassKind::Compute;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        auto& data = reflection.addBufferOutput("data").buffer(64, 4);
        if (scenario() == StageProbeCase::HiddenBufferWrite) { data.storageRead(); }
        else { data.storageReadWrite(); }
        auto& image = reflection.addTextureOutput("image");
        if (scenario() == StageProbeCase::HiddenImageWrite) { image.sampledRead(); }
        else { image.storageReadWrite(); }
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        auto buffer = context.device->createBuffer({.size = 64, .structureStride = 4,
            .usage = render::BufferUsageBits::Storage, .memoryLocation = render::MemoryLocation::Device,
            .queueAccess = render::QueueAccessBits::Graphics | render::QueueAccessBits::Compute});
        if (!buffer) { return render::makeError(buffer.error()); }
        privateBuffer_ = std::move(*buffer);
        return {};
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using Access = render::RenderGraphResourceAccess;
        using Use = render::RenderGraphStageUse;
        using Stage = render::RenderGraphComputeStage;
        auto& probe = *activeStageProbe;
        const auto mode = scenario();
        std::vector<std::vector<Use>> uses;
        std::vector<render::RenderGraphBufferImport> imports;
        auto privateSlice = privateBuffer_->slice();
        if (!privateSlice) { return render::makeError(privateSlice.error()); }
        switch (mode) {
        case StageProbeCase::LateUnknownResource:
            uses = {{{"data", Access::BufferStorageRead}}, {{"unknown", Access::BufferStorageRead}}};
            break;
        case StageProbeCase::LateMissingRecorder:
            uses = {{{"data", Access::BufferStorageRead}}, {{"data", Access::BufferStorageRead}}};
            break;
        case StageProbeCase::HiddenBufferWrite:
            uses = {{{"data", Access::BufferStorageWrite}}};
            break;
        case StageProbeCase::HiddenImageWrite:
            uses = {{{"image", Access::TextureStorageWrite}}};
            break;
        case StageProbeCase::ImageLayoutChange:
            uses = {{{"image", Access::TextureSampleRead}}};
            break;
        case StageProbeCase::GraphAllocationImport: {
            auto graphSlice = context.outputBuffer("data").buffer()->slice(16, 32);
            if (!graphSlice) { return render::makeError(graphSlice.error()); }
            imports.push_back({"private", *graphSlice, Access::BufferStorageReadWrite});
            uses = {{{"private", Access::BufferStorageWrite}}};
            break;
        }
        case StageProbeCase::ConflictingImportAliases:
            imports.push_back({"privateWrite", *privateSlice, Access::BufferStorageReadWrite});
            imports.push_back({"privateRead", *privateSlice, Access::BufferStorageRead});
            uses = {{{"privateRead", Access::BufferStorageRead}}};
            break;
        case StageProbeCase::HiddenImportWrite:
            imports.push_back({"private", *privateSlice, Access::BufferStorageRead});
            uses = {{{"private", Access::BufferStorageWrite}}};
            break;
        case StageProbeCase::NonComputeAccess:
            uses = {{{"data", Access::BufferTransferWrite}}};
            break;
        case StageProbeCase::PrivateAliases: {
            // Distinct overlapping CPU slices must still resolve to one GPU
            // allocation, rather than independent hazards under the two names.
            auto writeSlice = privateSlice->subslice(0, 48);
            auto readSlice = privateSlice->subslice(16, 48);
            if (!writeSlice || !readSlice) { return render::makeError(render::Error::Failure); }
            imports.push_back({"privateWrite", *writeSlice, Access::BufferStorageReadWrite});
            imports.push_back({"privateRead", *readSlice, Access::BufferStorageReadWrite});
            uses = {{{"privateWrite", Access::BufferStorageWrite}}, {{"privateRead", Access::BufferStorageRead}}};
            break;
        }
        case StageProbeCase::RasterParentScope:
        case StageProbeCase::RepeatedScope:
        case StageProbeCase::NestedScope:
        case StageProbeCase::ForkInsideScope:
            uses = {{{"data", Access::BufferStorageRead}}};
            break;
        }

        const auto record = [&](render::CommandBuffer& commands) -> render::Result<> {
            ++probe.callbacks;
            probe.barriersAtCallbacks.push_back(commands.synchronizationStats().memoryBarriers);
            if (mode == StageProbeCase::NestedScope) {
                const std::array nestedUses{Use{"data", Access::BufferStorageRead}};
                const std::array nestedStages{Stage{"Nested", nestedUses,
                    [&](render::CommandBuffer&) -> render::Result<> { ++probe.unexpectedCallbacks; return {}; }}};
                return context.executeComputeStages(nestedStages);
            }
            if (mode == StageProbeCase::ForkInsideScope) {
                const auto unexpected = [&](render::CommandBuffer&) -> render::Result<> {
                    ++probe.unexpectedCallbacks;
                    return {};
                };
                return context.parallelCompute(unexpected, unexpected);
            }
            return {};
        };
        std::vector<Stage> stages;
        stages.reserve(uses.size());
        for (size_t index = 0; index < uses.size(); ++index) {
            stages.push_back({index == 0 ? "First" : "Second", uses[index], record});
        }
        if (mode == StageProbeCase::LateMissingRecorder) { stages.back().record = {}; }
        auto result = context.executeComputeStages(stages, imports);
        if (result && mode == StageProbeCase::RepeatedScope) {
            return context.executeComputeStages(stages, imports);
        }
        return result;
    }

private:
    StageProbeCase scenario() const
    {
        return static_cast<StageProbeCase>(properties().value("scenario", uint32_t(0)));
    }
    std::unique_ptr<render::Buffer> privateBuffer_;
};

RhiTestResult runStageProbe(RhiTestContext& context, StageProbeCase mode, uint32_t callbacks, bool succeeds = false)
{
    static const bool registered = render::registerRenderGraphPassType("ComputeStageProbePass",
        "Compute-stage declaration validation", [] { return std::make_unique<ComputeStageProbePass>(); });
    (void)registered;
    StageProbe probe;
    struct ProbeScope {
        explicit ProbeScope(StageProbe& value) { activeStageProbe = &value; }
        ~ProbeScope() { activeStageProbe = nullptr; }
    } scope(probe);
    render::RenderGraph graph;
    graph.addNode("ComputeStageProbePass", "Probe", {{"scenario", static_cast<uint32_t>(mode)}});
    graph.markOutput("Probe.data");
    render::RenderGraphExecutor executor;
    std::string log;
    auto result = executor.compile(context.device, graph, 4, 4, log);
    if (!result) { return RhiTestResult::fail("compute-stage probe compile: " + log); }
    result = executor.execute({.graphicsQueue = &context.graphicsQueue, .recordingWorkerLimit = 1,
        .submissionMode = render::FrameSubmissionMode::Joined});
    const std::string label = "compute-stage case " + std::to_string(static_cast<uint32_t>(mode));
    if (succeeds) {
        if (!result) { return RhiTestResult::fail(label + ": " + render::resultToString(result)); }
        result = executor.waitForSubmittedWork(5'000'000'000ull);
        if (!result) { return RhiTestResult::fail(label + ": completion failed"); }
    } else {
        if (result || result.error() != render::Error::InvalidArgument) {
            return RhiTestResult::fail(label + ": expected InvalidArgument");
        }
        if (executor.compiled() || executor.lastSubmittedCompletion().valid()) {
            return RhiTestResult::fail(label + ": failed recording must invalidate the graph without submitting work");
        }
    }
    if (probe.callbacks != callbacks || probe.unexpectedCallbacks != 0) {
        return RhiTestResult::fail(label + ": unexpected callback execution");
    }
    if (succeeds && (probe.barriersAtCallbacks.size() != 2 ||
        probe.barriersAtCallbacks[1] <= probe.barriersAtCallbacks[0])) {
        return RhiTestResult::fail(label + ": aliased private write/read stages lacked their memory dependency");
    }
    return RhiTestResult::pass();
}

class ComputeStageValidationTest final : public RhiTest {
public:
    ComputeStageValidationTest() { type = RhiTestType::Command; name = "render_graph_compute_stages_validate_before_recording"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        for (const auto mode : {StageProbeCase::LateUnknownResource, StageProbeCase::LateMissingRecorder,
                StageProbeCase::HiddenBufferWrite, StageProbeCase::HiddenImageWrite,
                StageProbeCase::ImageLayoutChange, StageProbeCase::GraphAllocationImport,
                StageProbeCase::ConflictingImportAliases, StageProbeCase::HiddenImportWrite,
                StageProbeCase::NonComputeAccess, StageProbeCase::RasterParentScope}) {
            auto result = runStageProbe(context, mode, 0);
            if (!result.passed) { return result; }
        }
        return RhiTestResult::pass("all declarations checked before the first callback");
    }
};

class ComputeStageAliasesTest final : public RhiTest {
public:
    ComputeStageAliasesTest() { type = RhiTestType::Command; name = "render_graph_compute_stages_private_allocation_aliases"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        return runStageProbe(context, StageProbeCase::PrivateAliases, 2, true);
    }
};

class ComputeStageReentryTest final : public RhiTest {
public:
    ComputeStageReentryTest() { type = RhiTestType::Command; name = "render_graph_compute_stages_reject_reentry_and_forks"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        for (const auto mode : {StageProbeCase::RepeatedScope, StageProbeCase::NestedScope, StageProbeCase::ForkInsideScope}) {
            auto result = runStageProbe(context, mode, 1);
            if (!result.passed) { return result; }
        }
        return RhiTestResult::pass();
    }
};

METALLIC_REGISTER_RHI_TEST(ComputeStageValidationTest);
METALLIC_REGISTER_RHI_TEST(ComputeStageAliasesTest);
METALLIC_REGISTER_RHI_TEST(ComputeStageReentryTest);

} // namespace
} // namespace metallic::tests

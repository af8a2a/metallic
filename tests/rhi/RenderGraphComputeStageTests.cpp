#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <array>
#include <atomic>
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
    RepeatedRawReads,
    GenericRepeatedBufferReads,
    GenericRepeatedImageReads,
    RepeatedScope,
    NestedScope,
    ForkInsideScope,
};

struct StageProbe {
    uint32_t callbacks = 0;
    uint32_t unexpectedCallbacks = 0;
    std::vector<uint64_t> barriersAtCallbacks;
    render::SynchronizationStats beforeSequence;
    render::SynchronizationStats afterSequence;
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
        case StageProbeCase::RepeatedRawReads:
        case StageProbeCase::GenericRepeatedBufferReads:
        case StageProbeCase::GenericRepeatedImageReads: {
            // The parent graph boundary covers the initial state. Only the
            // first read of each writer generation needs a RAW barrier here.
            const bool image = mode == StageProbeCase::GenericRepeatedImageReads;
            const auto name = image ? "image" : "data";
            const auto write = image ? Access::TextureStorageWrite : Access::BufferStorageWrite;
            const auto read = image ? Access::TextureStorageRead : Access::BufferStorageRead;
            uses = {{{name, write}}};
            for (uint32_t index = 0; index < 6; ++index) {
                uses.push_back({{name, read}});
            }
            uses.push_back({{name, write}});
            uses.push_back({{name, read}});
            uses.push_back({{name, read}});
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
        const bool generic = mode == StageProbeCase::GenericRepeatedBufferReads ||
            mode == StageProbeCase::GenericRepeatedImageReads;
        probe.beforeSequence = context.commandBuffer().synchronizationStats();
        auto result = generic ? context.executeStages(stages, imports) : context.executeComputeStages(stages, imports);
        probe.afterSequence = context.commandBuffer().synchronizationStats();
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
    if (mode == StageProbeCase::PrivateAliases && (probe.barriersAtCallbacks.size() != 2 ||
        probe.barriersAtCallbacks[1] <= probe.barriersAtCallbacks[0])) {
        return RhiTestResult::fail(label + ": aliased private write/read stages lacked their memory dependency");
    }
    const bool generic = mode == StageProbeCase::GenericRepeatedBufferReads ||
        mode == StageProbeCase::GenericRepeatedImageReads;
    if (mode == StageProbeCase::RepeatedRawReads || generic) {
        // Keep the write-after-read/write dependency, then restart visibility
        // coverage for the new writer. The counters observe native encoding;
        // the callbacks deliberately do not execute a shader workload.
        constexpr std::array<uint64_t, 10> expectedDeltas{0, 1, 1, 1, 1, 1, 1, 2, 3, 3};
        if (probe.barriersAtCallbacks.size() != expectedDeltas.size()) {
            return RhiTestResult::fail(label + ": unexpected synchronization sample count");
        }
        const auto baseline = probe.barriersAtCallbacks.front();
        for (size_t index = 0; index < expectedDeltas.size(); ++index) {
            const auto expected = baseline + expectedDeltas[index];
            if (probe.barriersAtCallbacks[index] != expected) {
                return RhiTestResult::fail(label + ": stage " + std::to_string(index) +
                    " expected " + std::to_string(expected) + " memory barriers, got " +
                    std::to_string(probe.barriersAtCallbacks[index]));
            }
        }
        if (generic && (probe.afterSequence.memoryBarriers != probe.beforeSequence.memoryBarriers + 3 ||
            probe.afterSequence.memoryBarriers != probe.barriersAtCallbacks.back() ||
            probe.afterSequence.imageTransitions != probe.beforeSequence.imageTransitions)) {
            return RhiTestResult::fail(label + ": same-layout generic sequence added an entry/exit barrier or lost an internal dependency");
        }
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

class ComputeStageRawVisibilityTest final : public RhiTest {
public:
    ComputeStageRawVisibilityTest() { type = RhiTestType::Command; name = "render_graph_compute_stages_repeated_raw_encoding"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        return runStageProbe(context, StageProbeCase::RepeatedRawReads, 10, true);
    }
};

class GeneralStageRawVisibilityTest final : public RhiTest {
public:
    GeneralStageRawVisibilityTest() { type = RhiTestType::Command; name = "render_graph_stages_repeated_raw_no_exit_barrier"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        for (const auto mode : {StageProbeCase::GenericRepeatedBufferReads, StageProbeCase::GenericRepeatedImageReads}) {
            auto result = runStageProbe(context, mode, 10, true);
            if (!result.passed) { return result; }
        }
        return RhiTestResult::pass("three internal hazards; no same-layout graph buffer/image exit barrier");
    }
};

enum class GeneralStageCase : uint32_t {
    GraphTransfer,
    UnauthorizedTransfer,
    PrivateTextureAliases,
    WrongTextureView,
    GraphTextureImport,
    QualifiedNames,
    AmbiguousName,
    UnsafeFork,
    DeniedFork,
    InvalidTextureFinalState,
    InvalidTextureInitialState,
    EmptyBufferAccess,
};

struct GeneralStageProbe {
    uint32_t callbacks = 0;
    std::atomic<uint32_t> branches = 0;
};

GeneralStageProbe* activeGeneralStageProbe = nullptr;

class GeneralStageProbePass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        using Access = render::RenderGraphResourceAccess;
        using Kind = render::RenderGraphPassKind;
        render::RenderPassReflection reflection;
        if (scenario() == GeneralStageCase::QualifiedNames || scenario() == GeneralStageCase::AmbiguousName) {
            reflection.addTextureInput("image").storageRead().stageAccess(Access::TextureTransferRead, Kind::Unsafe);
        }
        auto& image = reflection.addTextureOutput("image").storageReadWrite();
        image.format = render::Format::Rgba8Unorm;
        if (scenario() != GeneralStageCase::UnauthorizedTransfer) {
            image.stageAccess(Access::TextureTransferWrite, Kind::Unsafe)
                .stageAccess(Access::TextureTransferRead, Kind::Unsafe);
        }
        reflection.addBufferOutput("data").buffer(4 * 4 * 4).transferWrite().hostReadback();
        return reflection;
    }

    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string&) override
    {
        if (scenario() == GeneralStageCase::EmptyBufferAccess) {
            return context.device->createBuffer({.size = 64, .usage = render::BufferUsageBits::Storage}).transform(
                [&](auto value) { privateBuffer_ = std::move(value); });
        }
        const bool invalidState = scenario() == GeneralStageCase::InvalidTextureFinalState ||
            scenario() == GeneralStageCase::InvalidTextureInitialState;
        if (scenario() != GeneralStageCase::PrivateTextureAliases &&
            scenario() != GeneralStageCase::WrongTextureView && !invalidState) {
            return {};
        }
        const render::TextureDesc desc{.usage = invalidState ? render::TextureUsageBits::Sampled :
            render::TextureUsageBits::Storage | render::TextureUsageBits::TransferSource | render::TextureUsageBits::TransferDestination,
            .format = render::Format::Rgba8Unorm, .width = 4, .height = 4};
        auto result = context.device->createTexture(desc).transform([&](auto value) { privateTexture_ = std::move(value); });
        if (!result) { return result; }
        result = context.device->createTextureView(*privateTexture_, {.format = desc.format}).transform(
            [&](auto value) { privateView_ = std::move(value); });
        if (!result || scenario() != GeneralStageCase::WrongTextureView) { return result; }
        result = context.device->createTexture(desc).transform([&](auto value) { wrongTexture_ = std::move(value); });
        if (!result) { return result; }
        return context.device->createTextureView(*wrongTexture_, {.format = desc.format}).transform(
            [&](auto value) { wrongView_ = std::move(value); });
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using Access = render::RenderGraphResourceAccess;
        using Kind = render::RenderGraphPassKind;
        using Use = render::RenderGraphStageUse;
        using Stage = render::RenderGraphStage;
        auto& probe = *activeGeneralStageProbe;
        const auto mode = scenario();
        if (mode == GeneralStageCase::EmptyBufferAccess) {
            auto slice = privateBuffer_->slice();
            if (!slice) { return render::makeError(slice.error()); }
            const render::RenderGraphBufferImport imports[] = {{"private", *slice, Access::None}};
            const Use uses[] = {{"private", Access::BufferStorageRead}};
            const auto callback = [&](render::CommandBuffer&) -> render::Result<> { ++probe.callbacks; return {}; };
            const Stage stages[] = {{"ValidPrefix", {}, callback}, {"PrivateRead", uses, callback}};
            return context.executeStages(stages, imports);
        }
        if (mode == GeneralStageCase::UnsafeFork || mode == GeneralStageCase::DeniedFork) {
            const Stage stages[] = {{"Fork", {}, [&](render::CommandBuffer&) {
                ++probe.callbacks;
                const auto branch = [&](render::CommandBuffer&) -> render::Result<> {
                    ++probe.branches;
                    return {};
                };
                return context.parallelCompute(branch, branch);
            }, Kind::Unsafe, mode == GeneralStageCase::UnsafeFork}};
            return context.executeStages(stages);
        }

        const auto image = context.outputTexture("image");
        const auto data = context.outputBuffer("data");
        auto* texture = image.texture();
        std::string_view clearName = "image", copyName = "image";
        std::vector<render::RenderGraphTextureImport> textures;
        if (mode == GeneralStageCase::PrivateTextureAliases || mode == GeneralStageCase::WrongTextureView) {
            texture = privateTexture_.get();
            textures = {{"privateWrite", texture, privateView_.get(), privateState_, render::ResourceState::General},
                {"privateRead", texture, mode == GeneralStageCase::WrongTextureView ? wrongView_.get() : privateView_.get(),
                    privateState_, render::ResourceState::General}};
            clearName = "privateWrite";
            copyName = "privateRead";
        } else if (mode == GeneralStageCase::GraphTextureImport) {
            textures.push_back({"privateWrite", texture, image.view(), render::ResourceState::General});
            clearName = "privateWrite";
        } else if (mode == GeneralStageCase::InvalidTextureFinalState || mode == GeneralStageCase::InvalidTextureInitialState) {
            // Neither an unused final layout nor an invalid incoming state may
            // bypass validation simply because all callback uses are graph-owned.
            textures.push_back({"invalidState", privateTexture_.get(), privateView_.get(),
                mode == GeneralStageCase::InvalidTextureInitialState
                    ? render::ResourceState::IndirectArgument : render::ResourceState::Undefined,
                mode == GeneralStageCase::InvalidTextureFinalState
                    ? render::ResourceState::TransferDestination : render::ResourceState::ShaderRead});
        }
        if (mode == GeneralStageCase::QualifiedNames || mode == GeneralStageCase::AmbiguousName) {
            const Use copyUses[] = {{"input.image", Access::TextureTransferRead},
                {mode == GeneralStageCase::AmbiguousName ? "image" : "output.image", Access::TextureTransferWrite}};
            const Use readbackUses[] = {{"output.image", Access::TextureTransferRead}, {"data", Access::BufferTransferWrite}};
            const Stage stages[] = {
                {"CopyInput", copyUses, [&](render::CommandBuffer& commands) -> render::Result<> {
                    ++probe.callbacks;
                    commands.copyTexture({.source = context.inputTexture("image").texture(), .destination = texture,
                        .width = 4, .height = 4, .depth = 1});
                    return {};
                }, Kind::Unsafe},
                {"Readback", readbackUses, [&](render::CommandBuffer& commands) -> render::Result<> {
                    ++probe.callbacks;
                    commands.copyTextureToBuffer({.texture = texture, .buffer = data.buffer(), .width = 4, .height = 4});
                    return {};
                }, Kind::Unsafe},
            };
            return context.executeStages(stages);
        }
        const Use clearUses[] = {{clearName, Access::TextureTransferWrite}};
        const Use copyUses[] = {{copyName, Access::TextureTransferRead}, {"data", Access::BufferTransferWrite}};
        const Stage stages[] = {
            // A valid prefix must remain unrecorded when any later declaration
            // or private view is invalid.
            {"ValidatedPrefix", {}, [&](render::CommandBuffer&) -> render::Result<> { ++probe.callbacks; return {}; }},
            {"Clear", clearUses, [&](render::CommandBuffer& commands) -> render::Result<> {
                ++probe.callbacks;
                commands.clearColorTexture(*texture, render::ResourceState::TransferDestination,
                    render::ColorValue{1.0f, 0.0f, 1.0f, 1.0f});
                return {};
            }, Kind::Unsafe},
            {"Readback", copyUses, [&](render::CommandBuffer& commands) -> render::Result<> {
                ++probe.callbacks;
                commands.copyTextureToBuffer({.texture = texture, .buffer = data.buffer(), .width = 4, .height = 4});
                return {};
            }, Kind::Unsafe},
        };
        auto result = context.executeStages(stages, {}, textures);
        if (result && mode == GeneralStageCase::PrivateTextureAliases) { privateState_ = render::ResourceState::General; }
        return result;
    }

private:
    GeneralStageCase scenario() const
    {
        return static_cast<GeneralStageCase>(properties().value("scenario", uint32_t(0)));
    }
    std::unique_ptr<render::Texture> privateTexture_, wrongTexture_;
    std::unique_ptr<render::TextureView> privateView_, wrongView_;
    std::unique_ptr<render::Buffer> privateBuffer_;
    render::ResourceState privateState_ = render::ResourceState::Undefined;
};

RhiTestResult runGeneralStageProbe(RhiTestContext& context, GeneralStageCase mode, bool succeeds)
{
    static const bool registered = render::registerRenderGraphPassType("GeneralStageProbePass",
        "Resource access stages across compute and transfer", [] { return std::make_unique<GeneralStageProbePass>(); });
    (void)registered;
    GeneralStageProbe probe;
    struct ProbeScope {
        explicit ProbeScope(GeneralStageProbe& value) { activeGeneralStageProbe = &value; }
        ~ProbeScope() { activeGeneralStageProbe = nullptr; }
    } scope(probe);
    render::RenderGraph graph;
    graph.addNode("GeneralStageProbePass", "Probe", {{"scenario", static_cast<uint32_t>(mode)}});
    if (mode == GeneralStageCase::QualifiedNames || mode == GeneralStageCase::AmbiguousName) {
        graph.addNode("ClearColorPass", "Source", {{"color", {1.0, 1.0, 0.0, 1.0}}});
        graph.addEdge("Source.color", "Probe.image");
    }
    graph.markOutput("Probe.data");
    render::RenderGraphExecutor executor;
    std::string log;
    auto result = executor.compile(context.device, graph, 4, 4, log);
    if (!result) { return RhiTestResult::fail("general-stage compile: " + log); }
    const auto label = "general-stage case " + std::to_string(static_cast<uint32_t>(mode));
    const bool fork = mode == GeneralStageCase::UnsafeFork || mode == GeneralStageCase::DeniedFork;
    const uint32_t frameCount = succeeds && !fork ? 2u : 1u;
    for (uint32_t frame = 0; frame < frameCount; ++frame) {
        result = executor.execute({.graphicsQueue = &context.graphicsQueue,
            .computeQueue = context.device.getQueue(render::QueueType::Compute),
            .recordingWorkerLimit = 1, .submissionMode = render::FrameSubmissionMode::Joined});
        if (!succeeds) {
            if (result || result.error() != render::Error::InvalidArgument ||
                probe.callbacks != (mode == GeneralStageCase::DeniedFork ? 1u : 0u) || probe.branches != 0u) {
                return RhiTestResult::fail(label + ": invalid sequence executed callbacks or was accepted");
            }
            return RhiTestResult::pass();
        }
        if (!result || !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RhiTestResult::fail(label + ": stage execution/submission failed");
        }
        if (fork) {
            if (probe.callbacks != 1u || probe.branches != 2u) {
                return RhiTestResult::fail(label + ": declared fork did not join both branches");
            }
            continue;
        }
        const auto* output = executor.outputResource("Probe.data");
        const auto* image = executor.outputResource("Probe.image");
        if (!output || !output->buffer || !image || image->state != render::ResourceState::General) {
            return RhiTestResult::fail(label + ": graph boundary state was not preserved");
        }
        output->buffer->invalidate();
        const auto* bytes = static_cast<const uint8_t*>(output->buffer->map());
        if (!bytes) { return RhiTestResult::fail(label + ": readback map failed"); }
        const std::array<uint8_t, 4> expected = mode == GeneralStageCase::QualifiedNames
            ? std::array<uint8_t, 4>{255, 255, 0, 255} : std::array<uint8_t, 4>{255, 0, 255, 255};
        bool matches = true;
        for (uint32_t pixel = 0; pixel < 16; ++pixel) {
            for (uint32_t channel = 0; channel < 4; ++channel) {
                matches = matches && bytes[pixel * 4u + channel] == expected[channel];
            }
        }
        output->buffer->unmap();
        if (!matches) { return RhiTestResult::fail(label + ": clear/copy pixels differ"); }
    }
    return RhiTestResult::pass();
}

class GeneralStageTransfersTest final : public RhiTest {
public:
    GeneralStageTransfersTest() { type = RhiTestType::Command; name = "render_graph_stages_transfer_pixels_and_private_aliases"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        for (const auto mode : {GeneralStageCase::GraphTransfer, GeneralStageCase::PrivateTextureAliases,
                GeneralStageCase::QualifiedNames}) {
            auto result = runGeneralStageProbe(context, mode, true);
            if (!result.passed) { return result; }
        }
        return RhiTestResult::pass("clear/copy/readback and restored layouts survived two frames");
    }
};

class GeneralStageValidationTest final : public RhiTest {
public:
    GeneralStageValidationTest() { type = RhiTestType::Command; name = "render_graph_stages_validate_transfers_imports_and_names"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        for (const auto mode : {GeneralStageCase::UnauthorizedTransfer, GeneralStageCase::WrongTextureView,
                GeneralStageCase::GraphTextureImport, GeneralStageCase::AmbiguousName,
                GeneralStageCase::InvalidTextureFinalState, GeneralStageCase::InvalidTextureInitialState,
                GeneralStageCase::EmptyBufferAccess}) {
            auto result = runGeneralStageProbe(context, mode, false);
            if (!result.passed) { return result; }
        }
        return RhiTestResult::pass();
    }
};

class GeneralStageForkTest final : public RhiTest {
public:
    GeneralStageForkTest() { type = RhiTestType::Command; name = "render_graph_stages_explicit_unsafe_fork_join"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto result = runGeneralStageProbe(context, GeneralStageCase::UnsafeFork, true);
        if (!result.passed) { return result; }
        return runGeneralStageProbe(context, GeneralStageCase::DeniedFork, false);
    }
};

METALLIC_REGISTER_RHI_TEST(GeneralStageTransfersTest);
METALLIC_REGISTER_RHI_TEST(GeneralStageValidationTest);
METALLIC_REGISTER_RHI_TEST(GeneralStageForkTest);
METALLIC_REGISTER_RHI_TEST(ComputeStageValidationTest);
METALLIC_REGISTER_RHI_TEST(ComputeStageAliasesTest);
METALLIC_REGISTER_RHI_TEST(ComputeStageReentryTest);
METALLIC_REGISTER_RHI_TEST(ComputeStageRawVisibilityTest);
METALLIC_REGISTER_RHI_TEST(GeneralStageRawVisibilityTest);

} // namespace
} // namespace metallic::tests

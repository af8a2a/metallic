#include "RhiTest.h"
#include "RenderGraphViewerTestUi.h"
#include "Editor/EditorRenderGraphViewer.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"
#include "Runtime/Render/SlangCompiler.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>

namespace metallic::tests {
namespace {

class ViewerComputeFixture final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").storageWrite().format = render::Format::Rgba8Unorm;
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "AutoExposureFixture",
            .entryPointName = "autoExposureFixtureMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc binding{.binding = 0, .kind = render::ComputeResourceBindingKind::StorageImage};
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = 16,
            .bindings = &binding, .bindingCount = 1, .requiresRayQuery = false}, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        struct Push { uint32_t width, height; float luminance; uint32_t outliers; };
        const Push push{context.width(), context.height(), 0.25f, 0};
        const render::ComputeDispatchBinding binding{.binding = 0, .textureView = context.outputTexture("color").view()};
        const std::array uses{render::RenderGraphStageUse{"color", render::RenderGraphResourceAccess::TextureStorageWrite}};
        const std::array stages{render::RenderGraphStage{"Fill pixels", uses,
            [&](render::CommandBuffer& commands) {
                return program_.dispatch({.commandBuffer = &commands, .bindings = &binding, .bindingCount = 1,
                    .pushData = &push, .pushDataSize = sizeof(push),
                    .groupCountX = (push.width + 7) / 8, .groupCountY = (push.height + 7) / 8});
            }}};
        return context.executeStages(stages);
    }
private:
    render::ComputeProgram program_;
};

class ViewerReadbackFixture final : public render::UnsafePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    render::QueueType queueType() const override { return render::QueueType::Copy; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("source").transferRead();
        reflection.addBufferOutput("data").buffer(uint64_t(context.width) * context.height * 4).transferWrite().hostReadback();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        context.commandBuffer().copyTextureToBuffer({.texture = context.inputTexture("source").texture(),
            .buffer = context.outputBuffer("data").buffer(), .width = context.width(), .height = context.height()});
        return {};
    }
};

class RenderGraphViewerTest final : public RhiTest {
public:
    RenderGraphViewerTest() { type = RhiTestType::Rendering; name = "render_graph_execution_viewer"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "Render graph viewer regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableAsyncCompute = true, .preferUnifiedImageLayouts = false}).transform([&](auto value) { device = std::move(value); });
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("bindless device unavailable"); }
        if (!result) { return RhiTestResult::fail("viewer device creation failed"); }
        render::registerRenderGraphPassType("ViewerComputeFixture", "Viewer compute fixture",
            [] { return std::make_unique<ViewerComputeFixture>(); });
        render::registerRenderGraphPassType("ViewerReadbackFixture", "Viewer readback fixture",
            [] { return std::make_unique<ViewerReadbackFixture>(); });
        render::RenderGraph graph;
        graph.setName("Execution viewer / real queues");
        graph.addNode("ClearColorPass", "Raster", {{"color", {0.1f, 0.3f, 0.6f, 1.0f}}});
        graph.addNode("ViewerComputeFixture", "Compute");
        graph.addNode("CopyColorPass", "Copy");
        graph.addNode("ViewerReadbackFixture", "Readback");
        graph.addNode("ClearColorPass", "Unused");
        graph.addEdge("Compute.color", "Copy.source");
        graph.addEdge("Copy.color", "Readback.source");
        graph.markOutput("Raster.color");
        graph.markOutput("Readback.data");
        render::RenderGraphExecutor executor;
        if (executor.executionSnapshot()) { return RhiTestResult::fail("disabled capture unexpectedly exposed a snapshot"); }
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        if (!executor.compile(*device, graph, 16, 16, log)) { return RhiTestResult::fail("viewer graph compile: " + log); }
        auto* graphics = device->getQueue(render::QueueType::Graphics);
        auto* compute = device->getQueue(render::QueueType::Compute);
        auto* copy = device->getQueue(render::QueueType::Copy);
        std::shared_ptr<const render::RenderGraphExecutionSnapshot> recorded;
        std::vector<bool> recordedCompletion;
        const auto execute = [&]() {
            if (!executor.execute({.graphicsQueue = graphics, .computeQueue = compute, .copyQueue = copy,
                .recordingWorkerLimit = 1, .recordingBatchWorkload = 1,
                .submissionMode = render::FrameSubmissionMode::Pipelined})) { return false; }
            recorded = executor.executionSnapshot();
            recordedCompletion.clear();
            if (recorded) {
                for (const auto& segment : recorded->segments) { recordedCompletion.push_back(segment.completed); }
            }
            return executor.waitForSubmittedWork(5'000'000'000ull).has_value();
        };
        if (!execute()) { return RhiTestResult::fail("viewer graph execution failed"); }
        const auto first = executor.executionSnapshot();
        if (!first || !first->success || first->passes.size() != 4 || first->resources.size() < 4 ||
            first->status != render::RenderGraphExecutionSnapshotStatus::Submitted) {
            return RhiTestResult::fail("viewer did not receive the actual executed graph and resources");
        }
        if (!recorded || first->segments.empty() || first->segments.size() != recordedCompletion.size()) {
            return RhiTestResult::fail("viewer omitted submitted segments");
        }
        for (size_t index = 0; index < first->segments.size(); ++index) {
            if (!first->segments[index].completionKnown || !first->segments[index].completed ||
                recorded->segments[index].completed != recordedCompletion[index]) {
                return RhiTestResult::fail("viewer completion refresh changed an immutable snapshot or missed GPU completion");
            }
        }
        auto* output = executor.outputResource("Readback.data");
        std::array<uint32_t, 16 * 16> pixels{};
        if (!output || !output->buffer) {
            return RhiTestResult::fail("viewer graph readback failed");
        }
        output->buffer->invalidate();
        const void* mapped = output->buffer->map();
        if (!mapped) { return RhiTestResult::fail("viewer graph readback map failed"); }
        std::memcpy(pixels.data(), mapped, sizeof(pixels));
        output->buffer->unmap();
        for (uint32_t pixel : pixels) {
            if (pixel != 0xff404040u && pixel != 0xff3f3f3fu) {
                return RhiTestResult::fail("viewer fixture did not execute real compute/write/copy/read operations");
            }
        }
        const auto findPass = [&](std::string_view name) -> const render::RenderGraphExecutionPassSnapshot* {
            for (const auto& pass : first->passes) { if (pass.name == name) { return &pass; } }
            return nullptr;
        };
        const auto* computePass = findPass("Compute");
        const auto* rasterPass = findPass("Raster");
        const auto* copyPass = findPass("Copy");
        if (!computePass || !rasterPass || !copyPass || findPass("Unused") || computePass->stages.empty() ||
            computePass->stages.front().name != "Fill pixels" || !computePass->stages.front().recorded) {
            return RhiTestResult::fail("viewer snapshot lost culling, executed passes or internal stage declarations");
        }
        if (compute && !compute->sameQueue(*graphics) && computePass->actualQueueId == rasterPass->actualQueueId) {
            return RhiTestResult::fail("viewer reported the graphics queue for actual async compute work");
        }
        if (copy && !copy->sameQueue(*graphics) && copyPass->actualQueueId == rasterPass->actualQueueId) {
            return RhiTestResult::fail("viewer reported the graphics queue for actual copy work");
        }
        if (copyPass->barriers.empty() || !copyPass->synchronization.calls ||
            !copyPass->synchronization.imageTransitions) {
            return RhiTestResult::fail("viewer lost the planned transfer layout barrier or its native encoding statistics");
        }
        bool alias = false, read = false, write = false;
        std::set<uint64_t> allocationIds;
        for (const auto& resource : first->resources) {
            if (!resource.id || !allocationIds.insert(resource.id).second) {
                return RhiTestResult::fail("viewer resource list duplicated a canonical allocation");
            }
            if (!resource.memory.known || resource.memory.allocationId != resource.id ||
                !resource.memory.memoryBlockId || !resource.memory.sizeBytes) {
                return RhiTestResult::fail("viewer memory data does not describe the actual live allocations");
            }
            if (resource.name == "Compute.color") {
                alias = std::find(resource.aliases.begin(), resource.aliases.end(), "Copy.source") != resource.aliases.end();
                for (const auto& use : computePass->uses) { write |= use.resourceId == resource.id && use.writes; }
                for (const auto& use : copyPass->uses) { read |= use.resourceId == resource.id && use.reads; }
            }
        }
        if (!alias || !read || !write || first->batches.empty()) {
            return RhiTestResult::fail("viewer lost input alias, canonical read/write use or queue batches");
        }
        for (const auto& batch : first->batches) {
            if (!batch.accepted) { return RhiTestResult::fail("completed viewer graph contains an unaccepted batch"); }
        }
        if (compute && !compute->sameQueue(*graphics) &&
            std::none_of(first->batches.begin(), first->batches.end(), [](const auto& batch) {
                return !batch.waitPredecessors.empty();
            })) {
            return RhiTestResult::fail("actual compute-to-copy dependency was missing its cross-queue wait");
        }
        editor::RenderGraphExecutionViewer viewer;
        viewer.setLive(true);
        viewer.update(first);
        viewer.setLive(false);
        if (!execute()) { return RhiTestResult::fail("second viewer graph execution failed"); }
        const auto second = executor.executionSnapshot();
        if (!second || second->executionId == first->executionId) { return RhiTestResult::fail("viewer snapshot did not advance"); }
        viewer.update(second);
        if (viewer.snapshot() != first) { return RhiTestResult::fail("frozen viewer replaced its captured execution"); }
        viewer.requestCapture();
        if (!viewer.wantsCapture()) { return RhiTestResult::fail("viewer capture request was lost"); }
        viewer.update(second);
        if (viewer.snapshot() != second || viewer.wantsCapture()) { return RhiTestResult::fail("viewer did not consume one requested capture"); }

        std::filesystem::create_directories(context.outputDirectory);
        ViewerUiContext ui(true);
        using Tab = editor::RenderGraphExecutionViewer::Tab;
        std::set<std::string> images;
        const std::array tabs{std::pair{Tab::Resources, "resources"}, std::pair{Tab::Queues, "queues"}, std::pair{Tab::Memory, "memory"}};
        for (const auto& [tab, name] : tabs) {
            const auto path = context.outputDirectory / (std::string("render_graph_viewer_") + name + ".png");
            const auto failure = ui.save(viewer, tab, path);
            if (!failure.empty()) { return RhiTestResult::fail(std::string(name) + ": " + failure); }
            std::ifstream file(path, std::ios::binary);
            std::string image((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (image.size() < 1024 || !images.insert(std::move(image)).second) {
                return RhiTestResult::fail("viewer tabs produced missing or identical screenshots");
            }
        }
        executor.setExecutionCaptureEnabled(false);
        if (!execute()) { return RhiTestResult::fail("graph execution with capture disabled failed"); }
        const auto retained = executor.executionSnapshot();
        if (!retained || retained->executionId != second->executionId || viewer.snapshot() != second ||
            first->executionId == second->executionId) {
            return RhiTestResult::fail("disabled capture replaced or invalidated the retained viewer snapshot");
        }
        executor.setExecutionCaptureEnabled(true);
        if (!executor.execute({.graphicsQueue = graphics, .computeQueue = graphics, .copyQueue = graphics,
                .recordingWorkerLimit = 1, .recordingBatchWorkload = 1,
                .submissionMode = render::FrameSubmissionMode::Pipelined}) ||
            !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RhiTestResult::fail("viewer unified-queue execution failed");
        }
        const auto unified = executor.executionSnapshot();
        if (!unified || unified->queues.size() != 1 ||
            std::any_of(unified->passes.begin(), unified->passes.end(), [&](const auto& pass) {
                return pass.actualQueueId != unified->queues.front().id;
            }) || std::any_of(unified->batches.begin(), unified->batches.end(), [](const auto& batch) {
                return !batch.waitPredecessors.empty();
            })) {
            return RhiTestResult::fail("viewer invented queue lanes or cross-queue waits for a unified queue");
        }
        viewer.requestCapture();
        viewer.update(unified);
        auto failure = ui.save(viewer, Tab::Queues, context.outputDirectory / "render_graph_viewer_queues_unified.png");
        if (!failure.empty()) { return RhiTestResult::fail("unified queues: " + failure); }

        // A frozen UI owns only copied diagnostics. Tear down every allocation,
        // submission tracker and the device before asking it to draw again.
        const uint64_t allocation = unified->resources.front().id;
        executor = render::RenderGraphExecutor{};
        device.reset();
        if (viewer.snapshot() != unified || unified->resources.front().id != allocation) {
            return RhiTestResult::fail("graph destruction invalidated the frozen execution snapshot");
        }
        failure = ui.save(viewer, Tab::Memory, context.outputDirectory / "render_graph_viewer_memory_frozen.png");
        if (!failure.empty()) { return RhiTestResult::fail("frozen graph destruction: " + failure); }
        return RhiTestResult::pass("real 16x16 multi/unified queues and frozen teardown; offscreen ImGui PNGs in " + context.outputDirectory.string());
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphViewerTest);

} // namespace
} // namespace metallic::tests

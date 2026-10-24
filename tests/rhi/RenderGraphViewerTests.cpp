#include "Runtime/Render/GAPI/Vulkan/VulkanDeviceExtensions.h"
#include "TestResourceParameters.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "RHITest.h"
#include "RenderGraphViewerTestUI.h"
#include "Editor/EditorRenderGraphViewer.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <cfloat>

namespace metallic::tests {
namespace {

class RenderGraphViewerCanvasTest final : public RHITest {
public:
    RenderGraphViewerCanvasTest() { type = RHITestType::Command; name = "render_graph_viewer_canvas_navigation"; }
    RHITestResult run(RHITestContext& test) override
    {
        auto* previous = ImGui::GetCurrentContext();
        auto* context = ImGui::CreateContext();
        struct Cleanup {
            ImGuiContext* context;
            ImGuiContext* previous;
            ~Cleanup() { ImGui::DestroyContext(context); ImGui::SetCurrentContext(previous); }
        } cleanup{context, previous};
        try {
            const auto require = [](bool condition, const char* message) {
                if (!condition) { throw std::runtime_error(message); }
            };
            auto& io = ImGui::GetIO();
            io.IniFilename = nullptr; io.LogFilename = nullptr;
            io.DisplaySize = ImVec2(1400, 900); io.DeltaTime = 1.f / 60;
            unsigned char* atlas = nullptr; int atlasWidth = 0, atlasHeight = 0;
            io.Fonts->GetTexDataAsRGBA32(&atlas, &atlasWidth, &atlasHeight);
            io.Fonts->SetTexID(ImTextureID(1));
            auto snapshot = std::make_shared<render::RenderGraphExecutionSnapshot>();
            snapshot->graphName = "Canvas navigation";
            snapshot->graphGeneration = 1; snapshot->executionId = 1;
            snapshot->queues.push_back({.id = 0, .type = render::QueueType::Graphics});
            snapshot->passes.push_back({.id = 42, .name = "Canvas pass", .type = "Fixture", .actualQueueId = 0});
            snapshot->segments.push_back({.id = 7, .passId = 42, .queueId = 0, .recorded = true, .accepted = true});
            editor::RenderGraphExecutionViewer viewer;
            viewer.update(snapshot);
            viewer.setTab(editor::RenderGraphExecutionViewer::Tab::Queues);
            const auto draw = [&] {
                ImGui::NewFrame();
                ImGui::SetNextWindowPos(ImVec2(0, 0));
                ImGui::SetNextWindowSize(io.DisplaySize);
                ImGui::Begin("Canvas test", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
                viewer.draw();
                ImGui::End();
                ImGui::Render();
            };
            const auto canvas = [&]() -> ImGuiWindow* {
                for (auto* window : context->Windows) {
                    if (window->Active && std::string_view(window->Name).find("QueueTopology") != std::string_view::npos) {
                        return window;
                    }
                }
                throw std::runtime_error("Missing queue canvas");
            };
            const auto nodeBounds = [&] {
                ImRect result(ImVec2(FLT_MAX, FLT_MAX), ImVec2(-FLT_MAX, -FLT_MAX));
                for (const auto& vertex : canvas()->DrawList->VtxBuffer) {
                    if (vertex.col == IM_COL32(44, 48, 57, 255)) { result.Add(vertex.pos); }
                }
                require(result.Min.x < result.Max.x && result.Min.y < result.Max.y, "Missing canvas node geometry");
                return result;
            };
            const auto close = [](float a, float b) { return std::abs(a - b) < 1.0f; };
            const auto same = [&](const ImRect& a, const ImRect& b) {
                return close(a.Min.x, b.Min.x) && close(a.Min.y, b.Min.y) &&
                    close(a.Max.x, b.Max.x) && close(a.Max.y, b.Max.y);
            };
            const auto move = [&](ImVec2 point) { io.AddMousePosEvent(point.x, point.y); draw(); };
            const auto click = [&](ImVec2 point) {
                move(point);
                io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); draw();
                io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); draw();
            };
            const auto toolbarButton = [&](const char* label) {
                auto* host = canvas()->ParentWindow;
                const auto& style = ImGui::GetStyle();
                const float fitWidth = ImGui::CalcTextSize("Fit topology").x + 2 * style.FramePadding.x;
                const bool fit = std::string_view(label) == "Fit topology";
                const float width = ImGui::CalcTextSize(label).x + 2 * style.FramePadding.x;
                // The toolbar sits directly above the one-line canvas navigation hint.
                const ImVec2 point(host->WorkRect.Min.x + (fit ? 0 : fitWidth + style.ItemSpacing.x) + width * .5f,
                    canvas()->Pos.y - style.ItemSpacing.y * 2 - ImGui::GetTextLineHeight() - ImGui::GetFrameHeight() * .5f);
                click(point);
            };
            for (int i = 0; i < 4; ++i) { draw(); }
            const ImRect fitted = nodeBounds();
            require(!canvas()->ScrollbarX && !canvas()->ScrollbarY, "Canvas still relies on scrollbars");
            const ImVec2 anchor(fitted.Min.x + fitted.GetWidth() * .3f, fitted.Min.y + fitted.GetHeight() * .4f);
            move(anchor);
            io.AddMouseWheelEvent(0, 2); draw();
            const ImRect zoomed = nodeBounds();
            const float ratio = zoomed.GetWidth() / fitted.GetWidth();
            require(ratio > 1.05f, "Wheel did not zoom the canvas");
            require(close(zoomed.Min.x, anchor.x + (fitted.Min.x - anchor.x) * ratio) &&
                close(zoomed.Min.y, anchor.y + (fitted.Min.y - anchor.y) * ratio), "Zoom drifted away from the mouse anchor");
            const auto stable = nodeBounds();
            auto next = std::make_shared<render::RenderGraphExecutionSnapshot>(*snapshot);
            ++next->executionId; viewer.update(next); draw();
            require(same(stable, nodeBounds()), "Live capture update reset the canvas view");
            // Entering the canvas and pressing in one input batch must not apply
            // motion accumulated before the drag was captured.
            move(ImVec2(anchor.x - 20, anchor.y - 10));
            io.AddMousePosEvent(anchor.x, anchor.y);
            io.AddMouseButtonEvent(ImGuiMouseButton_Middle, true); draw();
            require(same(stable, nodeBounds()), "Middle press applied motion from before drag capture");
            move(ImVec2(anchor.x + 55, anchor.y + 25));
            const auto panned = nodeBounds();
            require(close(panned.Min.x - zoomed.Min.x, 55) && close(panned.Min.y - zoomed.Min.y, 25),
                "Middle drag did not translate node geometry");
            const ImVec2 outside(canvas()->InnerRect.Max.x + 20, canvas()->InnerRect.Max.y + 20);
            move(outside);
            const auto outsideDrag = nodeBounds();
            require(!same(outsideDrag, panned), "Canvas drag stopped at its boundary");
            io.AddMouseButtonEvent(ImGuiMouseButton_Middle, false); draw();
            move(anchor);
            require(same(outsideDrag, nodeBounds()), "Released middle drag remained active outside the canvas");
            toolbarButton("Fit topology");
            require(same(fitted, nodeBounds()), "Fit topology did not restore the complete view");
            move(anchor); io.AddMouseWheelEvent(0, 1); draw();
            const auto selectable = nodeBounds();
            click(selectable.GetCenter());
            require(viewer.selectedPassId() == 42, "Zoomed node hit test selected the wrong pass");
            toolbarButton("1:1");
            require(std::abs(nodeBounds().GetWidth() - 132) < 2, "1:1 did not restore native node scale");
            // Wheel events outside the canvas must not change the graph camera.
            const auto native = nodeBounds();
            move(ImVec2(io.DisplaySize.x - 30, 50)); io.AddMouseWheelEvent(0, 2); draw();
            require(same(native, nodeBounds()), "Wheel outside the canvas changed graph zoom");
            std::filesystem::create_directories(test.outputDirectory);
            std::string message;
            if (!saveImGuiTestDrawDataPng(*ImGui::GetDrawData(), atlas, atlasWidth, atlasHeight, 1400, 900,
                    test.outputDirectory / "render_graph_canvas_navigation.png", message)) {
                return RHITestResult::fail(message);
            }
            return RHITestResult::pass("Cursor-anchored wheel zoom, captured middle pan, transformed hit testing, live updates and fit/native reset");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(RenderGraphViewerCanvasTest);

class RenderGraphResourceMemoryCanvasTest final : public RHITest {
public:
    RenderGraphResourceMemoryCanvasTest() { type = RHITestType::Command; name = "render_graph_resource_memory_canvas_navigation"; }
    RHITestResult run(RHITestContext& test) override
    {
        auto* previous = ImGui::GetCurrentContext();
        auto* context = ImGui::CreateContext();
        struct Cleanup {
            ImGuiContext* context;
            ImGuiContext* previous;
            ~Cleanup() { ImGui::DestroyContext(context); ImGui::SetCurrentContext(previous); }
        } cleanup{context, previous};
        try {
            const auto require = [](bool condition, const char* message) {
                if (!condition) { throw std::runtime_error(message); }
            };
            auto& io = ImGui::GetIO();
            io.IniFilename = nullptr; io.LogFilename = nullptr;
            io.DisplaySize = ImVec2(1400, 900); io.DeltaTime = 1.f / 60;
            unsigned char* atlas = nullptr; int atlasWidth = 0, atlasHeight = 0;
            io.Fonts->GetTexDataAsRGBA32(&atlas, &atlasWidth, &atlasHeight);
            io.Fonts->SetTexID(ImTextureID(1));
            auto snapshot = std::make_shared<render::RenderGraphExecutionSnapshot>();
            snapshot->graphName = "Resources and memory canvas";
            snapshot->graphGeneration = 1; snapshot->executionId = 1;
            snapshot->passes.push_back({.id = 42, .name = "Write target", .type = "Fixture"});
            snapshot->passes.back().uses.push_back({.resourceId = 22, .writes = true});
            snapshot->passes.push_back({.id = 57, .name = "Read source", .type = "Fixture"});
            snapshot->passes.back().uses.push_back({.resourceId = 11, .reads = true});
            for (uint32_t i = 0; i < 2; ++i) {
                render::RenderGraphExecutionResourceSnapshot resource;
                resource.id = i == 0 ? 11 : 22;
                resource.name = i == 0 ? "Source buffer" : "Target texture";
                resource.type = i == 0 ? render::RenderGraphResourceType::Buffer : render::RenderGraphResourceType::Texture2D;
                resource.memory.known = true;
                resource.memory.memoryBlockId = 100;
                resource.memory.allocationId = resource.id;
                resource.memory.backingAllocationId = resource.id;
                resource.memory.offsetBytes = i * 512;
                resource.memory.sizeBytes = resource.memory.backingSizeBytes = 256;
                snapshot->resources.push_back(resource);
            }
            editor::RenderGraphExecutionViewer viewer;
            viewer.update(snapshot);
            using Tab = editor::RenderGraphExecutionViewer::Tab;
            const auto draw = [&] {
                ImGui::NewFrame();
                ImGui::SetNextWindowPos(ImVec2(0, 0)); ImGui::SetNextWindowSize(io.DisplaySize);
                ImGui::Begin("Resource canvas test", nullptr, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoResize);
                viewer.draw(); ImGui::End(); ImGui::Render();
            };
            const auto canvas = [&](const char* name) -> ImGuiWindow* {
                for (auto* window : context->Windows) {
                    if (window->Active && std::string_view(window->Name).find(name) != std::string_view::npos) { return window; }
                }
                throw std::runtime_error(std::string("Missing canvas: ") + name);
            };
            const auto bounds = [&](const char* name, ImU32 color) {
                ImRect result(ImVec2(FLT_MAX, FLT_MAX), ImVec2(-FLT_MAX, -FLT_MAX));
                for (const auto& vertex : canvas(name)->DrawList->VtxBuffer) { if (vertex.col == color) { result.Add(vertex.pos); } }
                require(result.GetWidth() > 0 && result.GetHeight() > 0, "Target resource geometry is missing");
                return result;
            };
            const auto close = [](float a, float b) { return std::abs(a - b) < 1.1f; };
            const auto same = [&](const ImRect& a, const ImRect& b) {
                return close(a.Min.x, b.Min.x) && close(a.Min.y, b.Min.y) && close(a.Max.x, b.Max.x) && close(a.Max.y, b.Max.y);
            };
            const auto move = [&](ImVec2 point) { io.AddMousePosEvent(point.x, point.y); draw(); };
            const auto click = [&](ImVec2 point) {
                move(point); io.AddMouseButtonEvent(ImGuiMouseButton_Left, true); draw();
                io.AddMouseButtonEvent(ImGuiMouseButton_Left, false); draw();
            };
            const std::array tabs{Tab::Resources, Tab::Memory};
            const char* names[]{"ResourcePassMatrix", "MemoryBlocks"};
            const char* fitLabels[]{"Fit matrix", "Fit memory"};
            const ImU32 colors[]{IM_COL32(244, 104, 83, 255), IM_COL32(66, 126, 119, 255)};
            std::array<ImRect, 2> retained;
            for (size_t index = 0; index < tabs.size(); ++index) {
                viewer.setTab(tabs[index]);
                io.AddMousePosEvent(-100, -100);
                for (int i = 0; i < 4; ++i) { draw(); }
                const auto fitted = bounds(names[index], colors[index]);
                require(!canvas(names[index])->ScrollbarX && !canvas(names[index])->ScrollbarY, "Resource canvas retained scrollbars");
                const auto anchor = fitted.GetCenter();
                move(anchor); io.AddMouseWheelEvent(0, 2); draw();
                const auto zoomed = bounds(names[index], colors[index]);
                const float ratio = zoomed.GetWidth() / fitted.GetWidth();
                require(ratio > 1.1f, "Resource canvas wheel did not zoom");
                require(close(zoomed.Min.x, anchor.x + (fitted.Min.x - anchor.x) * ratio) &&
                    close(zoomed.Min.y, anchor.y + (fitted.Min.y - anchor.y) * ratio), "Resource zoom lost the mouse anchor");
                io.AddMouseButtonEvent(ImGuiMouseButton_Middle, true); draw();
                move(ImVec2(anchor.x + 27, anchor.y + 19));
                io.AddMouseButtonEvent(ImGuiMouseButton_Middle, false); draw();
                auto panned = bounds(names[index], colors[index]);
                require(close(panned.Min.x - zoomed.Min.x, 27) && close(panned.Min.y - zoomed.Min.y, 19),
                    "Resource middle drag did not translate geometry");
                // Reset selection by selecting the other resource, then click the transformed target.
                if (index == 1) { click(bounds(names[index], IM_COL32(73, 111, 151, 255)).GetCenter()); }
                else { click(bounds(names[index], IM_COL32(157, 203, 73, 255)).GetCenter()); }
                require(viewer.selectedResourceId() == 11, "Source resource selection failed");
                click(panned.GetCenter());
                require(viewer.selectedResourceId() == 22, "Transformed target hit selected the wrong resource");
                if (index == 0) { require(viewer.selectedPassId() == 42, "Transformed matrix cell selected the wrong pass"); }
                auto next = std::make_shared<render::RenderGraphExecutionSnapshot>(*snapshot);
                next->executionId += index + 1; viewer.update(next); draw();
                require(same(panned, bounds(names[index], colors[index])), "Live snapshot reset resource canvas");
                retained[index] = bounds(names[index], colors[index]);
                std::filesystem::create_directories(test.outputDirectory);
                std::string message;
                if (!saveImGuiTestDrawDataPng(*ImGui::GetDrawData(), atlas, atlasWidth, atlasHeight, 1400, 900,
                        test.outputDirectory / (index == 0 ? "render_graph_resource_canvas.png" : "render_graph_memory_canvas.png"), message)) {
                    return RHITestResult::fail(message);
                }
            }
            for (size_t index = 0; index < tabs.size(); ++index) {
                viewer.setTab(tabs[index]); io.AddMousePosEvent(-100, -100); draw(); draw();
                require(same(retained[index], bounds(names[index], colors[index])), "Switching tabs changed resource canvas view");
                auto* child = canvas(names[index]);
                const auto& style = ImGui::GetStyle();
                const ImVec2 fitPoint(child->ParentWindow->WorkRect.Min.x + ImGui::CalcTextSize(fitLabels[index]).x * .5f + style.FramePadding.x,
                    child->Pos.y - style.ItemSpacing.y * 2 - ImGui::GetTextLineHeight() - ImGui::GetFrameHeight() * .5f);
                click(fitPoint);
                require(!same(retained[index], bounds(names[index], colors[index])), "Fit did not reset resource canvas");
            }
            return RHITestResult::pass("Resources and Memory preserve cursor anchors, middle pan, transformed identities, independent tab views and live captures");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(RenderGraphResourceMemoryCanvasTest);

class ViewerComputeFixture final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    bool supportsAsyncQueue() const override { return true; }
    bool supportsPipelinedSubmission() const override { return true; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").storageWrite().format = render::Format::RGBA8Unorm;
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "AutoExposureFixture",
            .entryPointName = "autoExposureFixtureMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc binding{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::AutoExposureFixtureResources, output), .kind = render::ComputeResourceBindingKind::StorageImage};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .pushConstantSize = 16,
            .bindings = {&binding, 1},
            .requiresRayQuery = false,
            .resourceParameterSize = sizeof(metallic::tests::AutoExposureFixtureResources),
        }, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        struct Push { uint32_t width, height; float luminance; uint32_t outliers; };
        const Push push{context.width(), context.height(), 0.25f, 0};
        const render::ComputeDispatchBinding binding{.binding = METALLIC_RESOURCE_MEMBER(metallic::tests::AutoExposureFixtureResources, output), .textureView = context.outputTexture("color").view()};
        const std::array uses{render::RenderGraphStageUse{"color", render::RenderGraphResourceAccess::TextureStorageWrite}};
        const std::array stages{render::RenderGraphStage{"Fill pixels", uses,
            [&](render::CommandBuffer& commands) {
                return program_.dispatch({
                    .commandBuffer = &commands,
                    .bindings = {&binding, 1},
                    .pushData = &push,
                    .pushDataSize = sizeof(push),
                    .groupCountX = (push.width + 7) / 8,
                    .groupCountY = (push.height + 7) / 8,
                });
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
        if (auto commandResult = (context.outputBuffer("data").buffer())->slice().and_then([&](const auto& bufferSlice) { return context.commandBuffer().copyTextureToBuffer({.texture = context.inputTexture("source").texture(),
            .buffer = bufferSlice, .width = context.width(), .height = context.height()}); }); !commandResult) { return commandResult; }
        return {};
    }
};

class RenderGraphViewerTest final : public RHITest {
public:
    RenderGraphViewerTest() { type = RHITestType::Rendering; name = "render_graph_execution_viewer"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "Render graph viewer regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .enableAsyncCompute = true,
            .backendExtensions = metallic::render::vulkan::VulkanDeviceExtensions{
                .preferUnifiedImageLayouts = false,
            },
        }).transform([&](auto value) { device = std::move(value); });
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("bindless device unavailable"); }
        if (!result) { return RHITestResult::fail("viewer device creation failed"); }
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
        if (executor.executionSnapshot()) { return RHITestResult::fail("disabled capture unexpectedly exposed a snapshot"); }
        executor.setExecutionCaptureEnabled(true);
        std::string log;
        if (!executor.compile(*device, graph, 16, 16, log)) { return RHITestResult::fail("viewer graph compile: " + log); }
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
        if (!execute()) { return RHITestResult::fail("viewer graph execution failed"); }
        const auto first = executor.executionSnapshot();
        if (!first || !first->success || first->passes.size() != 4 || first->resources.size() < 4 ||
            first->status != render::RenderGraphExecutionSnapshotStatus::Submitted) {
            return RHITestResult::fail("viewer did not receive the actual executed graph and resources");
        }
        if (!recorded || first->segments.empty() || first->segments.size() != recordedCompletion.size()) {
            return RHITestResult::fail("viewer omitted submitted segments");
        }
        for (size_t index = 0; index < first->segments.size(); ++index) {
            if (!first->segments[index].completionKnown || !first->segments[index].completed ||
                recorded->segments[index].completed != recordedCompletion[index]) {
                return RHITestResult::fail("viewer completion refresh changed an immutable snapshot or missed GPU completion");
            }
        }
        auto* output = executor.outputResource("Readback.data");
        std::array<uint32_t, 16 * 16> pixels{};
        if (!output || !output->buffer) {
            return RHITestResult::fail("viewer graph readback failed");
        }
        output->buffer->invalidate();
        const void* mapped = output->buffer->map();
        if (!mapped) { return RHITestResult::fail("viewer graph readback map failed"); }
        std::memcpy(pixels.data(), mapped, sizeof(pixels));
        output->buffer->unmap();
        for (uint32_t pixel : pixels) {
            if (pixel != 0xff404040u && pixel != 0xff3f3f3fu) {
                return RHITestResult::fail("viewer fixture did not execute real compute/write/copy/read operations");
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
            return RHITestResult::fail("viewer snapshot lost culling, executed passes or internal stage declarations");
        }
        if (compute && !compute->sameQueue(*graphics) && computePass->actualQueueId == rasterPass->actualQueueId) {
            return RHITestResult::fail("viewer reported the graphics queue for actual async compute work");
        }
        if (copy && !copy->sameQueue(*graphics) && copyPass->actualQueueId == rasterPass->actualQueueId) {
            return RHITestResult::fail("viewer reported the graphics queue for actual copy work");
        }
        if (copyPass->barriers.empty() || !copyPass->synchronization.calls ||
            !copyPass->synchronization.imageTransitions) {
            return RHITestResult::fail("viewer lost the planned transfer layout barrier or its native encoding statistics");
        }
        bool alias = false, read = false, write = false;
        std::set<uint64_t> allocationIds;
        for (const auto& resource : first->resources) {
            if (!resource.id || !allocationIds.insert(resource.id).second) {
                return RHITestResult::fail("viewer resource list duplicated a canonical allocation");
            }
            if (!resource.memory.known || resource.memory.allocationId != resource.id ||
                !resource.memory.memoryBlockId || !resource.memory.sizeBytes) {
                return RHITestResult::fail("viewer memory data does not describe the actual live allocations");
            }
            if (resource.name == "Compute.color") {
                alias = std::find(resource.aliases.begin(), resource.aliases.end(), "Copy.source") != resource.aliases.end();
                for (const auto& use : computePass->uses) { write |= use.resourceId == resource.id && use.writes; }
                for (const auto& use : copyPass->uses) { read |= use.resourceId == resource.id && use.reads; }
            }
        }
        if (!alias || !read || !write || first->batches.empty()) {
            return RHITestResult::fail("viewer lost input alias, canonical read/write use or queue batches");
        }
        for (const auto& batch : first->batches) {
            if (!batch.accepted) { return RHITestResult::fail("completed viewer graph contains an unaccepted batch"); }
        }
        if (compute && !compute->sameQueue(*graphics) &&
            std::none_of(first->batches.begin(), first->batches.end(), [](const auto& batch) {
                return !batch.waitPredecessors.empty();
            })) {
            return RHITestResult::fail("actual compute-to-copy dependency was missing its cross-queue wait");
        }
        editor::RenderGraphExecutionViewer viewer;
        viewer.setLive(true);
        viewer.update(first);
        viewer.setLive(false);
        if (!execute()) { return RHITestResult::fail("second viewer graph execution failed"); }
        const auto second = executor.executionSnapshot();
        if (!second || second->executionId == first->executionId) { return RHITestResult::fail("viewer snapshot did not advance"); }
        viewer.update(second);
        if (viewer.snapshot() != first) { return RHITestResult::fail("frozen viewer replaced its captured execution"); }
        viewer.requestCapture();
        if (!viewer.wantsCapture()) { return RHITestResult::fail("viewer capture request was lost"); }
        viewer.update(second);
        if (viewer.snapshot() != second || viewer.wantsCapture()) { return RHITestResult::fail("viewer did not consume one requested capture"); }

        std::filesystem::create_directories(context.outputDirectory);
        ViewerUIContext ui(true);
        using Tab = editor::RenderGraphExecutionViewer::Tab;
        std::set<std::string> images;
        const std::array tabs{std::pair{Tab::Resources, "resources"}, std::pair{Tab::Queues, "queues"}, std::pair{Tab::Memory, "memory"}};
        for (const auto& [tab, name] : tabs) {
            const auto path = context.outputDirectory / (std::string("render_graph_viewer_") + name + ".png");
            const auto failure = ui.save(viewer, tab, path);
            if (!failure.empty()) { return RHITestResult::fail(std::string(name) + ": " + failure); }
            std::ifstream file(path, std::ios::binary);
            std::string image((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            if (image.size() < 1024 || !images.insert(std::move(image)).second) {
                return RHITestResult::fail("viewer tabs produced missing or identical screenshots");
            }
        }
        executor.setExecutionCaptureEnabled(false);
        if (!execute()) { return RHITestResult::fail("graph execution with capture disabled failed"); }
        const auto retained = executor.executionSnapshot();
        if (!retained || retained->executionId != second->executionId || viewer.snapshot() != second ||
            first->executionId == second->executionId) {
            return RHITestResult::fail("disabled capture replaced or invalidated the retained viewer snapshot");
        }
        executor.setExecutionCaptureEnabled(true);
        if (!executor.execute({.graphicsQueue = graphics, .computeQueue = graphics, .copyQueue = graphics,
                .recordingWorkerLimit = 1, .recordingBatchWorkload = 1,
                .submissionMode = render::FrameSubmissionMode::Pipelined}) ||
            !executor.waitForSubmittedWork(5'000'000'000ull)) {
            return RHITestResult::fail("viewer unified-queue execution failed");
        }
        const auto unified = executor.executionSnapshot();
        if (!unified || unified->queues.size() != 1 ||
            std::any_of(unified->passes.begin(), unified->passes.end(), [&](const auto& pass) {
                return pass.actualQueueId != unified->queues.front().id;
            }) || std::any_of(unified->batches.begin(), unified->batches.end(), [](const auto& batch) {
                return !batch.waitPredecessors.empty();
            })) {
            return RHITestResult::fail("viewer invented queue lanes or cross-queue waits for a unified queue");
        }
        viewer.requestCapture();
        viewer.update(unified);
        auto failure = ui.save(viewer, Tab::Queues, context.outputDirectory / "render_graph_viewer_queues_unified.png");
        if (!failure.empty()) { return RHITestResult::fail("unified queues: " + failure); }

        // A frozen UI owns only copied diagnostics. Tear down every allocation,
        // submission tracker and the device before asking it to draw again.
        const uint64_t allocation = unified->resources.front().id;
        executor = render::RenderGraphExecutor{};
        device.reset();
        if (viewer.snapshot() != unified || unified->resources.front().id != allocation) {
            return RHITestResult::fail("graph destruction invalidated the frozen execution snapshot");
        }
        failure = ui.save(viewer, Tab::Memory, context.outputDirectory / "render_graph_viewer_memory_frozen.png");
        if (!failure.empty()) { return RHITestResult::fail("frozen graph destruction: " + failure); }
        return RHITestResult::pass("real 16x16 multi/unified queues and frozen teardown; offscreen ImGui PNGs in " + context.outputDirectory.string());
    }
};

METALLIC_REGISTER_RHI_TEST(RenderGraphViewerTest);

} // namespace
} // namespace metallic::tests

#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/HistoryResources.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderView.h"
#include "Runtime/Render/SlangCompiler.h"

#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

class ViewProbePass final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        auto& output = reflection.addBufferOutput("view").buffer(sizeof(render::ViewConstants), sizeof(render::ViewConstants)).storageReadWrite();
        output.memoryLocation = render::MemoryLocation::HostReadback;
        return reflection;
    }
    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "ViewConstantsProbe",
            .entryPointName = "viewConstantsProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {{.binding = 0}, {.binding = 1}};
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
            .bindings = bindings, .bindingCount = 2, .requiresRayQuery = false}, log);
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        if (context.viewConstants() == nullptr || context.viewConstantsBuffer() == nullptr ||
            context.properties().at("camera").at("eye")[0].get<float>() != context.viewConstants()->current.eye[0]) {
            return render::makeError(render::Error::Failure);
        }
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .buffer = context.outputBuffer("view").buffer()},
            {.binding = 1, .buffer = context.viewConstantsBuffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings, .bindingCount = 2});
    }
private:
    render::ComputeProgram program_;
};

class RenderViewTest final : public RhiTest {
public:
    RenderViewTest() { type = RhiTestType::Rendering; name = "render_view_shared_constants_history"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderView view;
        view.setTemporalJitter(true);
        auto first = view.constants(0, 320, 180, 640, 360);
        auto camera = view.camera();
        camera.center[0] += 0.25f; // Pure rotation: eye is fixed.
        if (!view.setCamera(camera)) { return RhiTestResult::fail("Accept camera rotation"); }
        auto moved = view.constants(1, 320, 180, 640, 360, &first);
        if (!moved.frame[1] || moved.current.eye[0] != first.current.eye[0] ||
            moved.previous.center[0] != first.current.center[0] || moved.current.center[0] == first.current.center[0] ||
            moved.jitter[2] != first.jitter[0] || moved.current.viewport[0] != 320.0f / 180.0f) {
            return RhiTestResult::fail("Rotation retains the previous view and jitter");
        }
        view.cameraCut();
        if (view.constants(2, 320, 180, 640, 360, &moved).frame[1] ||
            view.constants(2, 321, 181, 640, 360, &moved).frame[1]) { return RhiTestResult::fail("Cut/resize invalidates history"); }
        camera.up = {0, 0, 0};
        if (view.setCamera(camera)) { return RhiTestResult::fail("Reject invalid view without replacing it"); }
        render::HistoryResourceManager history;
        auto revision = history.reprojectionInvalidationRevision();
        history.invalidateAll(render::HistoryInvalidationReason::CameraMotion);
        if (history.reprojectionInvalidationRevision() != revision) { return RhiTestResult::fail("Motion retains reprojection history"); }
        history.invalidateAll();
        if (history.reprojectionInvalidationRevision() == revision) { return RhiTestResult::fail("Scene edits reset reprojection history"); }

        render::registerRenderGraphPassType("ViewProbePass", "Shared view probe", [] { return std::make_unique<ViewProbePass>(); });
        render::RenderGraph graph;
        graph.setViewProperties({{"camera", view.cameraProperties()}, {"temporalJitter", true}});
        // Deliberately conflicting legacy node cameras must never win over the view.
        graph.addNode("ViewProbePass", "A", {{"camera", {{"eye", {99, 0, 0}}}}});
        graph.addNode("ViewProbePass", "B", {{"camera", {{"eye", {-99, 0, 0}}}}});
        graph.markOutput("A.view");
        graph.markOutput("B.view");
        std::string log;
        render::RenderGraph restored;
        if (!render::deserializeRenderGraphFromString(render::serializeRenderGraphToString(graph), restored, log) ||
            restored.viewProperties() != graph.viewProperties()) { return RhiTestResult::fail("View serialization: " + log); }
        std::unique_ptr<render::Device> device;
        auto created = render::createDevice({.applicationName = "Shared RenderView test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(created, render::Error::Unsupported)) { return RhiTestResult::skip("Requires bindless descriptors"); }
        if (!created) { return RhiTestResult::fail("Create view test device"); }
        render::RenderGraphExecutor executor, secondExecutor;
        if (!executor.compile(*device, restored, 320, 180, log) ||
            !secondExecutor.compile(*device, restored, 320, 180, log)) { return RhiTestResult::fail(log); }
        const auto execute = [&](render::RenderGraphExecutor& target) {
            return bool(target.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)})) && bool(target.waitForSubmittedWork());
        };
        const auto read = [&](render::RenderGraphExecutor& target, const char* output) {
            auto* buffer = target.outputResource(output)->buffer;
            buffer->invalidate();
            render::ViewConstants result;
            const void* data = buffer->map();
            if (data != nullptr) { std::memcpy(&result, data, sizeof(result)); buffer->unmap(); }
            return result;
        };
        if (!execute(executor) || !execute(secondExecutor)) { return RhiTestResult::fail("Initial view dispatch"); }
        first = read(executor, "A.view");
        camera = executor.renderView()->camera();
        camera.center[0] += 0.2f;
        executor.renderView()->setCamera(camera);
        if (!execute(executor) || !execute(secondExecutor)) { return RhiTestResult::fail("Moving view dispatch"); }
        moved = read(executor, "A.view");
        auto otherPass = read(executor, "B.view");
        auto otherView = read(secondExecutor, "A.view");
        if (std::memcmp(&moved, &otherPass, sizeof(moved)) != 0 || !moved.frame[1] ||
            moved.previous.center[0] != first.current.center[0] || moved.current.center[0] != camera.center[0] ||
            otherView.current.center[0] != first.current.center[0]) {
            return RhiTestResult::fail("GPU ABI, shared pass data, previous frame or independent view isolation");
        }
        executor.renderView()->cameraCut();
        if (!execute(executor) || read(executor, "A.view").frame[1]) { return RhiTestResult::fail("GPU camera cut history"); }
        if (!executor.compile(*device, restored, 321, 181, log) || !execute(executor)) { return RhiTestResult::fail(log); }
        const auto resized = read(executor, "A.view");
        if (resized.frame[1] || resized.current.viewport[1] != 321 || resized.current.center[0] != camera.center[0]) {
            return RhiTestResult::fail("Resize preserves authoring camera and discards temporal history");
        }
        return RhiTestResult::pass("GPU shared view, rotation, camera cut, resize, graph serialization and independent views");
    }
};
METALLIC_REGISTER_RHI_TEST(RenderViewTest);

} // namespace
} // namespace metallic::tests

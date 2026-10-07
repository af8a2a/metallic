#include "TestResourceLayouts.h"
#include "RHITest.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/HistoryResources.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Core/RenderView.h"
#include "Runtime/Render/Core/SlangCompiler.h"

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
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "ViewConstantsProbe",
            .entryPointName = "viewConstantsProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader.diagnostics).transform([&](auto value) { shader = std::move(value); });
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeResourceBindingDesc bindings[] = {{.binding = 0}, {.binding = 1}};
        return program_.initialize(*context.device, {
            .spirv = shader.spirv,
            .bindings = {bindings, 2},
            .requiresRayQuery = false,
            .resourceParameters = metallic::tests::kViewConstantsProbeLayout,
        }, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        if (context.viewConstants() == nullptr || context.viewConstantsBuffer() == nullptr ||
            context.properties().at("camera").at("eye")[0].get<float>() != context.viewConstants()->current.eye[0]) {
            return render::makeError(render::Error::Failure);
        }
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .buffer = context.outputBuffer("view").buffer()},
            {.binding = 1, .buffer = context.viewConstantsBuffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = {bindings, 2}});
    }
private:
    render::ComputeProgram program_;
};

class RenderViewTest final : public RHITest {
public:
    RenderViewTest() { type = RHITestType::Rendering; name = "render_view_shared_constants_history"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::RenderView view;
        view.setTemporalJitter(true);
        auto first = view.constants(0, 320, 180, 640, 360);
        const auto initialRevision = view.revision();
        const auto initialCutSerial = view.cutSerial();
        if (!view.setCameraProperties({{"reversedZ", false}}) || view.revision() != initialRevision ||
            view.cutSerial() != initialCutSerial || view.cameraProperties().contains("reversedZ") ||
            first.current.clipOrtho[3] != 1.0f || first.previous.clipOrtho[3] != 1.0f ||
            !view.constants(1, 320, 180, 640, 360, &first).frame[1]) {
            return RHITestResult::fail("Legacy standard-Z camera option must retain reversed Z and temporal history");
        }
        auto camera = view.camera();
        camera.center[0] += 0.25f; // Pure rotation: eye is fixed.
        if (!view.setCamera(camera)) { return RHITestResult::fail("Accept camera rotation"); }
        auto moved = view.constants(1, 320, 180, 640, 360, &first);
        if (!moved.frame[1] || moved.current.eye[0] != first.current.eye[0] ||
            moved.previous.center[0] != first.current.center[0] || moved.current.center[0] == first.current.center[0] ||
            moved.jitter[2] != first.jitter[0] || moved.current.viewport[0] != 320.0f / 180.0f) {
            return RHITestResult::fail("Rotation retains the previous view and jitter");
        }
        // Reapplying the same camera is a successful setter, not a cut.
        const auto stableRevision = view.revision();
        if (!view.setCamera(view.camera()) || view.revision() != stableRevision ||
            !view.constants(2, 320, 180, 640, 360, &moved).frame[1]) {
            return RHITestResult::fail("Identical camera retains temporal history");
        }
        camera = view.camera();
        camera.eye[0] += 0.5f;
        camera.center[0] += 0.5f;
        if (!view.setCamera(camera) || !view.constants(2, 320, 180, 640, 360, &moved).frame[1]) {
            return RHITestResult::fail("Translation retains reprojection history");
        }
        if (view.constants(3, 320, 180, 640, 360, &moved).frame[1] ||
            view.constants(2, 321, 180, 640, 360, &moved).frame[1] ||
            view.constants(2, 320, 180, 641, 360, &moved).frame[1]) {
            return RHITestResult::fail("Frame gap and render/output resize invalidate history");
        }
        for (uint32_t change = 0; change < 3; ++change) {
            auto before = view.constants(1, 320, 180, 640, 360);
            camera = view.camera();
            if (change == 0) { camera.orthographic = !camera.orthographic; }
            if (change == 1) { camera.nearPlane *= 2.0f; }
            if (change == 2) { camera.farPlane *= 2.0f; }
            if (!view.setCamera(camera) || view.constants(2, 320, 180, 640, 360, &before).frame[1]) {
                return RHITestResult::fail("Projection/clip-plane change invalidates history");
            }
        }
        view.cameraCut();
        if (view.constants(2, 320, 180, 640, 360, &moved).frame[1] ||
            view.constants(2, 321, 181, 640, 360, &moved).frame[1]) { return RHITestResult::fail("Cut/resize invalidates history"); }
        camera.up = {0, 0, 0};
        if (view.setCamera(camera)) { return RHITestResult::fail("Reject invalid view without replacing it"); }
        render::HistoryResourceManager history;
        if (!history.initialize(context.device) || !history.ensureBuffer("accumulation", {
                .size = 64, .usage = render::BufferUsageBits::Storage})) {
            return RHITestResult::fail("Initialize progressive history fixture");
        }
        history.beginFrame(0);
        history.markWritten("accumulation");
        history.beginFrame(1);
        if (!history.hasPrevious("accumulation")) { return RHITestResult::fail("Progressive history fixture is not valid"); }
        auto revision = history.reprojectionInvalidationRevision();
        const auto accumulationRevision = history.invalidationRevision();
        history.invalidateAll(render::HistoryInvalidationReason::CameraMotion);
        if (history.reprojectionInvalidationRevision() != revision ||
            history.invalidationRevision() == accumulationRevision || history.hasPrevious("accumulation")) {
            return RHITestResult::fail("Motion must reset progressive accumulation while retaining reprojection history");
        }
        history.invalidateAll();
        if (history.reprojectionInvalidationRevision() == revision) { return RHITestResult::fail("Scene edits reset reprojection history"); }

        render::registerRenderGraphPassType("ViewProbePass", "Shared view probe", [] { return std::make_unique<ViewProbePass>(); });
        render::RenderGraph graph;
        graph.setViewProperties({{"camera", view.cameraProperties()}, {"temporalJitter", true}});
        auto legacyView = graph.viewProperties();
        legacyView["camera"]["reversedZ"] = false;
        graph.clearDirty();
        graph.setViewProperties(legacyView);
        if (graph.dirty() || graph.viewProperties().at("camera").contains("reversedZ")) {
            return RHITestResult::fail("Graph view ignores obsolete depth convention without rebuilding");
        }
        // Deliberately conflicting legacy node cameras must never win over the view.
        graph.addNode("ViewProbePass", "A", {{"camera", {{"eye", {99, 0, 0}}, {"reversedZ", false}}}});
        graph.addNode("ViewProbePass", "B", {{"camera", {{"eye", {-99, 0, 0}}}}});
        graph.addNode("ViewProbePass", "Asset", {{"sceneBinding", "asset"}, {"viewBinding", "global"},
            {"camera", {{"eye", {199, 0, 0}}}}});
        graph.markOutput("A.view");
        graph.markOutput("B.view");
        graph.markOutput("Asset.view");
        std::string log;
        render::RenderGraph restored;
        const auto viewNodeId = graph.findNode("A")->id;
        if (graph.findNode("A")->properties.at("camera").contains("reversedZ") ||
            !graph.setNodeRuntimeProperty(viewNodeId, "camera.reversedZ", false) ||
            graph.findNode("A")->runtimeProperties.at("camera").contains("reversedZ")) {
            return RHITestResult::fail("Graph node setters omit obsolete depth convention");
        }
        // Direct mutation and saved legacy graphs are also normalized at the file boundary.
        graph.findNode("A")->properties["camera"]["reversedZ"] = false;
        const auto serialized = render::serializeRenderGraphToString(graph);
        if (serialized.find("reversedZ") != std::string::npos) {
            return RHITestResult::fail("Saved graph contains obsolete depth convention");
        }
        auto legacyGraph = nlohmann::json::parse(serialized);
        legacyGraph["view"]["camera"]["reversedZ"] = false;
        legacyGraph["nodes"][0]["properties"]["camera"]["reversedZ"] = false;
        if (!render::deserializeRenderGraphFromString(legacyGraph.dump(), restored, log) ||
            restored.viewProperties() != graph.viewProperties() ||
            restored.findNode("A")->properties.at("camera").contains("reversedZ")) {
            return RHITestResult::fail("Legacy view serialization: " + log);
        }
        std::unique_ptr<render::Device> device;
        auto created = render::createDevice({.applicationName = "Shared RenderView test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(created, render::Error::Unsupported)) { return RHITestResult::skip("Requires bindless descriptors"); }
        if (!created) { return RHITestResult::fail("Create view test device"); }
        render::RenderGraphExecutor executor, secondExecutor;
        if (!executor.compile(*device, restored, 320, 180, log) ||
            !secondExecutor.compile(*device, restored, 320, 180, log)) { return RHITestResult::fail(log); }
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
        if (!execute(executor) || !execute(secondExecutor)) { return RHITestResult::fail("Initial view dispatch"); }
        first = read(executor, "A.view");
        camera = executor.renderView()->camera();
        camera.center[0] += 0.2f;
        executor.renderView()->setCamera(camera);
        if (!execute(executor) || !execute(secondExecutor)) { return RHITestResult::fail("Moving view dispatch"); }
        moved = read(executor, "A.view");
        auto otherPass = read(executor, "B.view");
        auto assetPass = read(executor, "Asset.view");
        auto otherView = read(secondExecutor, "A.view");
        if (std::memcmp(&moved, &otherPass, sizeof(moved)) != 0 ||
            std::memcmp(&moved, &assetPass, sizeof(moved)) != 0 || !moved.frame[1] ||
            moved.current.clipOrtho[3] != 1.0f || moved.previous.clipOrtho[3] != 1.0f ||
            moved.previous.center[0] != first.current.center[0] || moved.current.center[0] != camera.center[0] ||
            otherView.current.center[0] != first.current.center[0]) {
            return RHITestResult::fail("GPU ABI, shared pass data, previous frame or independent view isolation");
        }
        executor.renderView()->setTemporalJitterSuppressed(true);
        if (!execute(executor)) { return RHITestResult::fail("Unjittered preview dispatch"); }
        const auto raw = read(executor, "A.view");
        const auto rawOther = read(executor, "Asset.view");
        if (raw.frame[1] || raw.frame[2] || raw.jitter[0] || raw.jitter[1] ||
            std::memcmp(&raw, &rawOther, sizeof(raw)) != 0 || !executor.renderView()->temporalJitter()) {
            return RHITestResult::fail("Raw preview shares zero jitter, invalidates history and preserves requested sampling");
        }
        camera.center[0] += .1f;
        executor.renderView()->setCamera(camera);
        if (!execute(executor)) { return RHITestResult::fail("Unjittered camera motion dispatch"); }
        const auto rawMoved = read(executor, "A.view");
        if (!rawMoved.frame[1] || rawMoved.jitter[0] || rawMoved.jitter[1] ||
            rawMoved.previous.center[0] != raw.current.center[0] || rawMoved.current.center[0] != camera.center[0]) {
            return RHITestResult::fail("Raw camera motion preserves unjittered previous/current views");
        }
        executor.renderView()->setTemporalJitterSuppressed(false);
        if (!execute(executor)) { return RHITestResult::fail("Restore temporal sampling dispatch"); }
        const auto resumed = read(executor, "A.view");
        if (resumed.frame[1] || resumed.frame[2] != 1 || (resumed.jitter[0] == 0 && resumed.jitter[1] == 0)) {
            return RHITestResult::fail("Returning to reconstruction restores samples with invalidated history");
        }
        executor.renderView()->cameraCut();
        if (!execute(executor) || read(executor, "A.view").frame[1]) { return RHITestResult::fail("GPU camera cut history"); }
        if (!executor.compile(*device, restored, 321, 181, log) || !execute(executor)) { return RHITestResult::fail(log); }
        const auto resized = read(executor, "A.view");
        if (resized.frame[1] || resized.current.viewport[1] != 321 || resized.current.center[0] != camera.center[0]) {
            return RHITestResult::fail("Resize preserves authoring camera and discards temporal history");
        }
        return RHITestResult::pass("GPU shared view, rotation, camera cut, resize, graph serialization and independent views");
    }
};
METALLIC_REGISTER_RHI_TEST(RenderViewTest);

} // namespace
} // namespace metallic::tests

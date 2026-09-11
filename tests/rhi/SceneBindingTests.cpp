#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderPass/RuntimeSceneBinding.h"
#include "Runtime/Scene/SceneDocument.h"

#include <unordered_map>

namespace metallic::tests {
namespace {
struct SceneProbeState {
    uint32_t compiles = 0;
    uint32_t executions = 0;
    bool failConsumer = false;
    std::unordered_map<std::string, uint64_t> identities;
    std::unordered_map<std::string, bool> views;
};
SceneProbeState probe;

class SceneBindingProbePass final : public render::ComputePass {
public:
    explicit SceneBindingProbePass(bool consumer) : consumer_(consumer) {}
    render::RenderGraphSceneDependency sceneDependency() const override
    {
        return consumer_
            ? render::RenderGraphSceneDependency{render::RenderGraphSceneSource::Input, {"first", "second"}}
            : render::RenderGraphSceneDependency{render::RenderGraphSceneSource::World};
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        if (consumer_) {
            reflection.addBufferInput("first").buffer(16, 4).shaderRead();
            reflection.addBufferInput("second").buffer(16, 4).shaderRead();
        }
        reflection.addBufferOutput("value").buffer(16, 4).shaderRead();
        return reflection;
    }
    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        ++probe.compiles;
        if (context.runtimeScene == nullptr ||
            render::runtimeSceneForPath(context.runtimeScene, properties().value("path", "")) == nullptr) {
            log = "Probe received an unresolved scene";
            return render::makeError(render::Error::InvalidArgument);
        }
        identity_ = context.runtimeScene->resourceIdentity();
        materialRevision_ = context.runtimeScene->materialRevision();
        hasView_ = context.renderView != nullptr;
        if (consumer_ && probe.failConsumer) {
            log = "Injected consumer scene preparation failure";
            return render::makeError(render::Error::Failure);
        }
        return {};
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        ++probe.executions;
        if (hasView_ != (context.viewConstants() != nullptr)) { return render::makeError(render::Error::Failure); }
        probe.views[context.passName()] = hasView_;
        if (context.runtimeScene() == nullptr || identity_ != context.runtimeScene()->resourceIdentity() ||
            materialRevision_ != context.runtimeScene()->materialRevision() ||
            render::runtimeSceneForPath(context.runtimeScene(), context.properties().value("path", "")) == nullptr) {
            return render::makeError(render::Error::InvalidArgument);
        }
        probe.identities[context.passName()] = identity_;
        return {};
    }
private:
    bool consumer_;
    uint64_t identity_ = 0;
    uint64_t materialRevision_ = 0;
    bool hasView_ = false;
};

class RenderGraphSceneBindingContractTest final : public RhiTest {
public:
    RenderGraphSceneBindingContractTest() { name = "render_graph_scene_binding_contract"; type = RhiTestType::Rendering; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::registerRenderGraphPassType("SceneBindingProbeRoot", "Scene source probe",
            [] { return std::make_unique<SceneBindingProbePass>(false); });
        render::registerRenderGraphPassType("SceneBindingProbeConsumer", "Inherited scene probe",
            [] { return std::make_unique<SceneBindingProbePass>(true); });
        if (render::renderGraphPassSceneDependency("SceneBindingProbeRoot").source != render::RenderGraphSceneSource::World) {
            return RhiTestResult::fail("New pass scene dependency was not discoverable without an editor whitelist");
        }
        const auto firstPath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf";
        const auto secondPath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/meet_mat.glb";
        scene::SceneDocument first, second;
        if (!first.load(firstPath) || !second.load(secondPath)) { return RhiTestResult::fail("Probe scenes failed to load"); }
        for (bool managedSubmission : {false, true}) {
            probe = {};
            render::RenderGraph graph;
            graph.addNode("SceneBindingProbeRoot", "Root", {{"path", firstPath.string()}});
            graph.addNode("SceneBindingProbeRoot", "Independent", {{"path", secondPath.string()}, {"sceneBinding", "asset"}});
            graph.addNode("SceneBindingProbeConsumer", "Consumer", {{"path", "deliberately-missing-asset.glb"}});
            graph.addEdge("Root.value", "Consumer.first");
            graph.addEdge("Root.value", "Consumer.second");
            graph.markOutput("Consumer.value");
            graph.markOutput("Independent.value");
            render::RenderView view;
            render::RenderGraphExecutor executor;
            executor.bindRenderView(&view);
            executor.bindRuntimeScene(&first);
            std::string log;
            auto result = executor.compile(context.device, graph, 8, 8, log);
            if (!result) { return RhiTestResult::fail("Initial binding: " + log); }
            graph.clearDirty();
            auto* originalOutput = executor.outputResource("Consumer.value");
            const auto* originalBuffer = originalOutput->buffer;
            render::QueueSubmissionTracker submissions;
            render::RenderFrameContext frame;
            std::unique_ptr<render::CommandPool> pool;
            std::unique_ptr<render::CommandBuffer> commands;
            if (!submissions.initialize(context.device, context.graphicsQueue) ||
                !context.device.createCommandPool(context.graphicsQueue, pool) || !pool->createCommandBuffer(commands)) {
                return RhiTestResult::fail("Probe command setup failed");
            }
            uint64_t frameIndex = 0;
            const auto renderFrame = [&]() -> render::Result {
                if (managedSubmission) {
                    auto status = executor.execute(render::RenderGraphSubmitDesc{.graphicsQueue = &context.graphicsQueue});
                    if (status) { status = executor.waitForSubmittedWork(); }
                    return status;
                }
                auto status = frame.begin(frameIndex++);
                if (status) { status = pool->reset(); }
                if (status) { status = commands->begin(&frame); }
                if (status) { status = executor.execute(*commands); }
                if (status) { status = commands->end(); }
                if (status) {
                    render::CommandBuffer* submitted[] = {commands.get()};
                    status = submissions.submit({.commandBuffers = submitted, .commandBufferCount = 1}, frame);
                }
                if (!status) { (void)pool->reset(); frame.cancel(); return status; }
                return frame.wait();
            };
            const auto matches = [&](uint64_t identity) {
                return probe.identities["Root"] == identity && probe.identities["Consumer"] == identity &&
                    probe.identities["Independent"] != identity &&
                    executor.outputResource("Consumer.value") == originalOutput && originalOutput->buffer == originalBuffer;
            };
            if (!renderFrame() || !matches(first.resourceIdentity())) { return RhiTestResult::fail("Initial frame binding mismatch"); }
            if (!probe.views["Root"] || !probe.views["Consumer"] || probe.views["Independent"]) {
                return RhiTestResult::fail("World and independent asset view bindings");
            }
            const uint32_t stableCompiles = probe.compiles;
            if (!renderFrame() || probe.compiles != stableCompiles) { return RhiTestResult::fail("Unchanged scene was recompiled"); }
            // Same object and path, new document identity, no graph dirty or rebind call.
            if (!first.load(firstPath) || !renderFrame() || !matches(first.resourceIdentity())) {
                return RhiTestResult::fail("Same-address scene replacement was not refreshed automatically");
            }
            auto material = first.materials().front();
            material.baseColorFactor.x *= 0.5f;
            if (!first.setMaterialProperties(0, material) || !renderFrame() || !matches(first.resourceIdentity())) {
                return RhiTestResult::fail("Material generation was not refreshed automatically");
            }
            // Fail after the root has prepared the new source. No pass may execute.
            const uint32_t previousExecutions = probe.executions;
            executor.bindRuntimeScene(&second);
            probe.failConsumer = true;
            if (renderFrame() || probe.executions != previousExecutions) {
                return RhiTestResult::fail("Partial scene preparation was exposed to a frame");
            }
            // Return to the previous identity: all partially changed resources must
            // be restored, even though its published generation already matches.
            probe.failConsumer = false;
            executor.bindRuntimeScene(&first);
            if (!renderFrame() || !matches(first.resourceIdentity())) { return RhiTestResult::fail("Preparation rollback did not restore resources"); }
            executor.bindRuntimeScene(&second);
            if (!renderFrame() || !matches(second.resourceIdentity())) { return RhiTestResult::fail("Different-path world replacement failed"); }
            const uint32_t worldCompiles = probe.compiles;
            graph.setNodeRuntimeProperty(graph.findNode("Consumer")->id, "path", "another-missing-asset.glb");
            executor.syncRuntimeProperties(graph);
            if (!renderFrame() || !matches(second.resourceIdentity()) || probe.compiles != worldCompiles) {
                return RhiTestResult::fail("Authored consumer path overrode inherited binding");
            }
            if (graph.dirty()) { return RhiTestResult::fail("Test unexpectedly relied on graph dirty"); }
            // Reject a mixed producer bundle before any pass prepares GPU resources.
            auto invalid = graph;
            for (const auto edge : invalid.edges()) {
                if (edge.dstPass == "Consumer" && edge.dstField == "second") { invalid.removeEdge(edge.id); break; }
            }
            invalid.addEdge("Independent.value", "Consumer.second");
            render::RenderGraphExecutor rejected;
            rejected.bindRuntimeScene(&first);
            const uint32_t beforeInvalid = probe.compiles;
            result = rejected.compile(context.device, invalid, 8, 8, log);
            if (result || probe.compiles != beforeInvalid || log.find("same scene producer") == std::string::npos) {
                return RhiTestResult::fail("Mixed scene input bundle was not rejected before preparation: " + log);
            }
            auto assetView = graph;
            for (const auto edge : graph.edges()) {
                if (edge.dstPass == "Consumer") {
                    assetView.removeEdge(edge.id);
                    assetView.addEdge("Independent.value", "Consumer." + edge.dstField);
                }
            }
            if (!executor.compile(context.device, assetView, 8, 8, log) || !renderFrame() || probe.views["Consumer"]) {
                return RhiTestResult::fail("Scene inputs must inherit the independent asset view: " + log);
            }
        }
        return RhiTestResult::pass("Both execute APIs: generation refresh, independent asset, inherited inputs, stable handles and preparation failure recovery");
    }
};
METALLIC_REGISTER_RHI_TEST(RenderGraphSceneBindingContractTest);
} // namespace
} // namespace metallic::tests

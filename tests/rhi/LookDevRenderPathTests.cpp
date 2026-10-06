#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Render/Subsystem/RenderSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>

namespace metallic::tests {
namespace {

class LookDevRenderPathContractTest final : public RHITest
{
public:
    LookDevRenderPathContractTest() { name = "lookdev_render_path_contract"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext&) override
    {
        using namespace render;
        std::string log;
        uint32_t checked = 0;
        for (const auto& desc : listBuiltInRenderSamples()) {
            if (desc.category != "LookDev" && desc.category != "Painter Validation" && desc.category != "Studio LookDev") { continue; }
            RenderSampleLoadResult sample;
            if (!loadBuiltInRenderSample(desc.id, sample, log)) { return RHITestResult::fail(log); }
            if (!supportsLookDevRenderPaths(sample.graph)) { continue; }
            const auto original = serializeRenderGraphToString(sample.graph);
            for (auto path : {LookDevRenderPath::Comparison, LookDevRenderPath::PathTraceOnly, LookDevRenderPath::DeferredOnly}) {
                RenderGraph graph;
                if (!makeLookDevRenderGraph(sample.graph, path, graph, log)) { return RHITestResult::fail(desc.id + ": " + log); }
                const bool pt = path != LookDevRenderPath::DeferredOnly;
                const bool deferred = path != LookDevRenderPath::PathTraceOnly;
                if (bool(graph.findNode("Reference")) != pt || bool(graph.findNode("Deferred")) != deferred ||
                    bool(graph.findNode("VBuffer")) != deferred ||
                    bool(graph.findNode("Slider")) != (path == LookDevRenderPath::Comparison) ||
                    graph.presentationOutputName() != sample.graph.presentationOutputName() ||
                    graph.viewProperties() != sample.graph.viewProperties()) {
                    return RHITestResult::fail(desc.id + ": wrong topology/view/presentation output");
                }
                for (const auto& node : graph.nodes()) {
                    const auto* source = sample.graph.findNode(node.name);
                    if (!source || source->properties != node.properties || source->runtimeProperties != node.runtimeProperties) {
                        return RHITestResult::fail("Retained pass settings changed");
                    }
                }
                RenderGraph reloaded;
                if (!deserializeRenderGraphFromString(serializeRenderGraphToString(graph), reloaded, log) || !reloaded.validate(log)) {
                    return RHITestResult::fail("Standalone graph roundtrip failed: " + log);
                }
            }
            if (serializeRenderGraphToString(sample.graph) != original) { return RHITestResult::fail("Source graph was mutated"); }
            ++checked;
        }
        RenderSampleLoadResult sample;
        if (!checked || !loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail("No comparison graphs tested"); }
        sample.graph.markOutput("Slider.color");
        if (!makeLookDevRenderGraph(sample.graph, LookDevRenderPath::PathTraceOnly, sample.graph, log) ||
            sample.graph.outputs().size() != 1 || sample.graph.outputs()[0].passName != "Reference") {
            return RHITestResult::fail("Aliased output graph or Slider output remapping failed");
        }
        const auto previous = serializeRenderGraphToString(sample.graph);
        RenderGraph unsupported;
        if (makeLookDevRenderGraph(unsupported, LookDevRenderPath::DeferredOnly, sample.graph, log) ||
            log.empty() || serializeRenderGraphToString(sample.graph) != previous) {
            return RHITestResult::fail("Unsupported graph changed the last valid result");
        }
        return RHITestResult::pass(std::to_string(checked) + " comparison graphs: all modes, roundtrip, unchanged settings and failure atomicity");
    }
};
METALLIC_REGISTER_RHI_TEST(LookDevRenderPathContractTest);

class LookDevRenderPathRenderingTest final : public RHITest
{
public:
    LookDevRenderPathRenderingTest() { name = "lookdev_render_paths"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        constexpr uint32_t kSize = 256;
        std::string log;
        RenderSampleLoadResult sample;
        const char* requestedSample = std::getenv("METALLIC_LOOKDEV_PATH_SAMPLE");
        if (!loadBuiltInRenderSample(requestedSample ? requestedSample : "lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) { return RHITestResult::fail("Scene load failed"); }
        for (auto path : {LookDevRenderPath::PathTraceOnly, LookDevRenderPath::DeferredOnly}) {
            const bool pt = path == LookDevRenderPath::PathTraceOnly;
            const char* output = pt ? "Reference.color" : "Deferred.color";
            std::vector<std::byte> reference;
            for (bool isolated : {false, true}) {
                RenderGraph graph = sample.graph;
                if (isolated && !makeLookDevRenderGraph(sample.graph, path, graph, log)) { return RHITestResult::fail(log); }
                RenderGraphPreviewRenderer preview;
                preview.bindRuntimeScene(&document);
                preview.setEnvironment(document.environment());
                preview.setLighting(document.lighting());
                if (!preview.subsystemHost()->configure<EnvironmentLightingSubsystem>(
                        {.initialDecodeTimeoutMilliseconds = 10000}, log) ||
                    !preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Preview setup failed: " + log); }
                for (uint32_t frame = 0; frame < 32; ++frame) {
                    preview.setRawReadbackEnabled(frame == 31);
                    if (!preview.render(graph, kSize, kSize, output, frame == 31)) { return RHITestResult::fail(preview.lastLog()); }
                    const auto& stats = preview.executionStats();
                    bool sawPT = false, sawDeferred = false, sawVBuffer = false, sawSlider = false;
                    for (const auto& node : stats.nodes) {
                        sawPT |= node.name == "Reference";
                        sawDeferred |= node.name == "Deferred";
                        sawVBuffer |= node.name == "VBuffer";
                        sawSlider |= node.name == "Slider";
                    }
                    if (sawPT != (!isolated || pt) || sawDeferred != (!isolated || !pt) ||
                        sawVBuffer != (!isolated || !pt) || sawSlider != !isolated) {
                        return RHITestResult::fail("Unexpected executed passes in " + graph.name());
                    }
                }
                const auto& bytes = preview.readbackBytes();
                if (bytes.empty()) { return RHITestResult::fail("No HDR readback"); }
                if (preview.readbackFormat() == Format::RGBA32Sfloat) {
                    for (size_t i = 0; i < bytes.size(); i += 4) {
                        float value; std::memcpy(&value, bytes.data() + i, 4);
                        if (!std::isfinite(value)) { return RHITestResult::fail("Nonfinite HDR"); }
                    }
                } else if (preview.readbackFormat() == Format::RGBA16Sfloat) {
                    for (size_t i = 0; i < bytes.size(); i += 2) {
                        uint16_t value; std::memcpy(&value, bytes.data() + i, 2);
                        if ((value & 0x7c00u) == 0x7c00u) { return RHITestResult::fail("Nonfinite HDR"); }
                    }
                } else { return RHITestResult::fail("Expected floating point HDR"); }
                if (!isolated) { reference = bytes; }
                else {
                    if (reference != bytes) { return RHITestResult::fail("Isolated HDR differs from comparison branch"); }
                    const std::string label = pt ? "PathTraceOnly" : "DeferredOnly";
                    if (!saveRenderGraphToFile(graph, context.outputDirectory / (label + ".metallic_graph.json"), log)) { return RHITestResult::fail(log); }
                    if (!preview.render(graph, kSize, kSize, "FinalBlit.color") ||
                        !saveRgba8Png(context.outputDirectory / (label + ".png"),
                            reinterpret_cast<const uint8_t*>(preview.pixels().data()), kSize, kSize, log)) {
                        return RHITestResult::fail(preview.lastLog() + log);
                    }
                    auto executed = RenderGraphProperties::array();
                    for (const auto& node : preview.executionStats().nodes) {
                        executed.push_back({{"name", node.name}, {"type", node.type}});
                    }
                    std::ofstream(context.outputDirectory / (label + ".passes.json")) << executed.dump(2);
                }
            }
        }
        return RHITestResult::pass("32 frames per graph: isolated pass execution, finite HDR byte equality to comparison branches and final display captures");
    }
};
METALLIC_REGISTER_RHI_TEST(LookDevRenderPathRenderingTest);

} // namespace
} // namespace metallic::tests

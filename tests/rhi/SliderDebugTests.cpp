#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

class SliderFixturePass final : public render::ComputePass {
public:
    explicit SliderFixturePass(bool readback = false) : readback_(readback) {}
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        if (readback_) {
            reflection.addTextureInput("source").sampledRead();
            reflection.addBufferOutput("pixels").buffer(uint64_t(context.width) * context.height * 16, 16)
                .storageReadWrite();
        } else {
            auto& color = reflection.addTextureOutput("color").storageReadWrite();
            color.format = properties().value("integer", false) ? render::Format::R32Uint : render::Format::Rgba32Sfloat;
            if (properties().value("wrongExtent", false)) { color.texture2D(7, 5); }
        }
        return reflection;
    }
    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "SliderDebugFixture",
            .entryPointName = readback_ ? "sliderReadbackMain" : "sliderFixtureMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageImage},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::SampledImage},
            {.binding = 2, .kind = render::ComputeResourceBindingKind::StorageBuffer},
        };
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .pushConstantSize = readback_ ? 0u : 4u,
            .bindings = readback_ ? bindings + 1 : bindings, .bindingCount = readback_ ? 2u : 1u,
            .requiresRayQuery = false}, log);
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        if (context.properties().value("integer", false)) { return {}; }
        const uint32_t path = context.properties().value("path", 0u);
        auto* input = context.inputTexture("source").view();
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .textureView = context.outputTexture("color").view()},
            {.binding = 1, .textureViews = &input, .textureViewCount = 1},
            {.binding = 2, .buffer = context.outputBuffer("pixels").buffer()},
        };
        return program_.dispatch({.commandBuffer = &context.commandBuffer(),
            .bindings = readback_ ? bindings + 1 : bindings, .bindingCount = readback_ ? 2u : 1u,
            .pushData = readback_ ? nullptr : &path, .pushDataSize = readback_ ? 0u : 4u,
            .groupCountX = (context.width() + 7) / 8, .groupCountY = (context.height() + 7) / 8});
    }
private:
    bool readback_;
    render::ComputeProgram program_;
};

class SliderDebugPixelsTest final : public RhiTest {
public:
    SliderDebugPixelsTest() { type = RhiTestType::Rendering; name = "slider_debug_hdr_pixels"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::registerRenderGraphPassType("SliderFixture", "Test color", [] { return std::make_unique<SliderFixturePass>(); });
        render::registerRenderGraphPassType("SliderReadback", "Read pixels", [] { return std::make_unique<SliderFixturePass>(true); });
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "SliderDebug GPU test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RhiTestResult::skip("Requires bindless descriptors"); }
        if (!initialized) { return RhiTestResult::fail("Device initialization failed"); }
        render::RenderGraph graph;
        graph.addNode("SliderFixture", "A");
        graph.addNode("SliderFixture", "B", {{"path", 1}});
        const uint32_t slider = graph.addNode("SliderDebugPass", "Slider")->id;
        graph.addNode("SliderReadback", "Readback");
        graph.addEdge("A.color", "Slider.sourceA");
        graph.addEdge("B.color", "Slider.sourceB");
        graph.addEdge("Slider.color", "Readback.source");
        graph.markOutput("Readback.pixels");
        auto pass = render::createRenderGraphPass("SliderDebugPass");
        for (const auto& setting : pass->runtimeSettings()) {
            if (setting.invalidateHistory || setting.rebuildGraph) { return RhiTestResult::fail("Slider must preserve producer histories"); }
        }
        render::RenderGraphExecutor executor;
        std::string log;
        for (const auto extent : {std::array<uint32_t, 2>{63, 37}, {17, 9}, {1, 1}}) {
            const auto [width, height] = extent;
            if (!executor.compile(*device, graph, width, height, log)) { return RhiTestResult::fail(log); }
            graph.clearDirty();
            for (bool horizontal : {false, true}) {
                for (bool swap : {false, true}) {
                    for (float split : {-1.0f, 0.0f, 0.25f, 0.5f, 0.73f, 1.0f, 2.0f}) {
                        graph.setNodeRuntimeProperties(slider, {{"splitPosition", split},
                            {"orientation", horizontal ? "horizontal" : "vertical"}, {"swapSides", swap}});
                        if (graph.dirty()) { return RhiTestResult::fail("Slider change rebuilt the graph"); }
                        executor.syncRuntimeProperties(graph);
                        if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
                            !executor.waitForSubmittedWork()) { return RhiTestResult::fail("Slider dispatch failed"); }
                        auto* buffer = executor.outputResource("Readback.pixels")->buffer;
                        buffer->invalidate();
                        const auto* mapped = static_cast<const std::array<float, 4>*>(buffer->map());
                        if (mapped == nullptr) { return RhiTestResult::fail("Readback mapping failed"); }
                        bool matches = true;
                        for (uint32_t y = 0; y < height; ++y) {
                            for (uint32_t x = 0; x < width; ++x) {
                                bool useA = ((horizontal ? y : x) + 0.5f) < std::clamp(split, 0.0f, 1.0f) * (horizontal ? height : width);
                                if (swap) { useA = !useA; }
                                const std::array<float, 4> expected{float(x) + 0.125f, -float(y) - 0.25f,
                                    useA ? 4.5f : 12.75f, useA ? 0.125f : 0.75f};
                                matches = matches && mapped[y * width + x] == expected;
                            }
                        }
                        buffer->unmap();
                        if (!matches) { return RhiTestResult::fail("Split, pixel alignment, HDR or alpha mismatch"); }
                    }
                }
            }
        }
        graph.setNodeProperties(graph.findNode("B")->id, {{"integer", true}});
        if (!executor.compile(*device, graph, 17, 9, log)) { return RhiTestResult::fail(log); }
        if (executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)})) {
            return RhiTestResult::fail("Integer comparison input was accepted");
        }
        if (!executor.waitForSubmittedWork()) { return RhiTestResult::fail("Failed-input cleanup failed"); }
        graph.setNodeProperties(graph.findNode("B")->id, {{"wrongExtent", true}});
        if (executor.compile(*device, graph, 17, 9, log)) { return RhiTestResult::fail("Mismatched extents were accepted"); }
        graph.removeNode(graph.findNode("B")->id);
        if (executor.compile(*device, graph, 17, 9, log)) { return RhiTestResult::fail("Missing required B input was accepted"); }
        return RhiTestResult::pass("Exact float4 comparison, both axes, swap, endpoints, runtime changes, resize and invalid inputs");
    }
};

class SliderDebugLookDevTest final : public RhiTest {
public:
    SliderDebugLookDevTest() { type = RhiTestType::Rendering; name = "slider_debug_lookdev_capture"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-shading-compare", sample, log)) { return RhiTestResult::fail(log); }
        const auto& a = sample.graph.findNode("OpenPBR")->properties;
        const auto& b = sample.graph.findNode("Standard")->properties;
        for (const char* key : {"path", "camera", "cameraSyncGroup", "samples", "maxDepth", "outputLinear", "accumulate"}) {
            if (a.at(key) != b.at(key)) { return RhiTestResult::fail(std::string("Comparison differs in ") + key); }
        }
        if (a.at("bsdf") != "openpbr" || b.at("bsdf") != "standard") { return RhiTestResult::fail("Wrong comparison BSDFs"); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RhiTestResult::fail(document.lastLoadResult().error);
        }
        if (document.lighting().autoExposure.enabled) { return RhiTestResult::fail("Comparison needs fixed exposure"); }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        preview.setLighting(document.lighting());
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RhiTestResult::skip("Requires ray queries"); }
        if (!initialized) { return RhiTestResult::fail("Preview initialization failed"); }
        constexpr uint32_t kSize = 768;
        for (uint32_t frame = 0; frame < 256; ++frame) {
            if (!preview.render(sample.graph, kSize, kSize)) { return RhiTestResult::fail(preview.lastLog()); }
        }
        const auto output = context.outputDirectory / "LookDevShadingComparison.png";
        if (!saveRgba8Png(output, reinterpret_cast<const uint8_t*>(preview.pixels().data()), kSize, kSize, log)) {
            return RhiTestResult::fail(log);
        }
        // Scene overrides must reach both paths when testing a different material asset.
        if (!render::setRenderSampleScenePath(sample, "Asset/meet_mat.glb", log) ||
            sample.graph.findNode("OpenPBR")->properties["path"] != "Asset/meet_mat.glb" ||
            sample.graph.findNode("Standard")->properties["path"] != "Asset/meet_mat.glb") {
            return RhiTestResult::fail("Scene override did not reach both paths: " + log);
        }
        return RhiTestResult::pass("1024 spp A/B capture: " + output.string());
    }
};

METALLIC_REGISTER_RHI_TEST(SliderDebugPixelsTest);
METALLIC_REGISTER_RHI_TEST(SliderDebugLookDevTest);

} // namespace
} // namespace metallic::tests

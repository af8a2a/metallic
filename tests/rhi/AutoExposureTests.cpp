#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <array>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

class AutoExposureFixturePass final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").storageReadWrite().format = render::Format::Rgba32Sfloat;
        return reflection;
    }
    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
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
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        struct Push { uint32_t width, height; float luminance; uint32_t outliers; };
        const Push push{context.width(), context.height(), context.properties().value("luminance", 0.18f),
            context.properties().value("outliers", 0u)};
        const render::ComputeDispatchBinding binding{.binding = 0, .textureView = context.outputTexture("color").view()};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = &binding, .bindingCount = 1,
            .pushData = &push, .pushDataSize = sizeof(push),
            .groupCountX = (push.width + 7) / 8, .groupCountY = (push.height + 7) / 8});
    }
private:
    render::ComputeProgram program_;
};

class AutoExposureGpuTest final : public RhiTest {
public:
    AutoExposureGpuTest() { type = RhiTestType::Rendering; name = "auto_exposure_histogram_adaptation"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto deviceResult = render::createDevice({.applicationName = "Auto exposure GPU test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(deviceResult, render::Error::Unsupported)) {
            return RhiTestResult::skip("requires bindless descriptors");
        }
        if (!deviceResult) { return RhiTestResult::fail("exposure test device creation failed"); }
        auto* queue = device->getQueue(render::QueueType::Graphics);
        render::registerRenderGraphPassType("AutoExposureFixturePass", "HDR test fixture",
            [] { return std::make_unique<AutoExposureFixturePass>(); });
        render::RenderWorld world;
        scene::LightingSettings lighting;
        auto& settings = lighting.autoExposure;
        settings.lowPercent = 0.0f;
        settings.highPercent = 100.0f;
        world.setLighting(lighting);
        render::RenderGraph graph;
        const uint32_t sourceId = graph.addNode("AutoExposureFixturePass", "Source")->id;
        const uint32_t exposureId = graph.addNode("AutoExposurePass", "Exposure", {{"adaptationDeltaSeconds", 0.1f}})->id;
        graph.addEdge("Source.color", "Exposure.source");
        graph.markOutput("Exposure.exposure");
        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        std::string log;
        if (!executor.compile(*device, graph, 63, 37, log)) { return RhiTestResult::fail(log); }
        std::array<float, 4> values{};
        int serial = 0;
        auto readExposure = [&] {
            auto* buffer = executor.outputResource("Exposure.exposure")->buffer;
            buffer->invalidate();
            void* mapped = buffer->map();
            if (mapped == nullptr) { return false; }
            std::memcpy(values.data(), mapped, sizeof(values));
            buffer->unmap();
            for (float value : values) { if (!std::isfinite(value)) { return false; } }
            return true;
        };
        auto frame = [&](float luminance, bool reset = false, uint32_t outliers = 0u, float dt = 0.1f) {
            if (reset) { ++serial; }
            graph.findNode(sourceId)->runtimeProperties = {{"luminance", luminance}, {"outliers", outliers}};
            graph.findNode(exposureId)->runtimeProperties = {{"resetSerial", serial}, {"adaptationDeltaSeconds", dt}};
            executor.syncRuntimeProperties(graph);
            if (!world.setLighting(lighting) || !executor.execute({.graphicsQueue = queue}) ||
                !executor.waitForSubmittedWork()) { return false; }
            return readExposure();
        };
        auto near = [](float a, float b, float tolerance = 0.02f) { return std::abs(a - b) < tolerance; };
        if (!frame(0.18f) || !near(values[0], 1.0f) || !near(values[1], 0.0f)) {
            return RhiTestResult::fail("18% gray did not meter to EV100=0");
        }
        if (!frame(0.18f * 1024.0f, true) || !near(values[1], 10.0f)) {
            return RhiTestResult::fail("physical luminance scale/first-frame exposure mismatch");
        }
        if (!frame(0.18f, true) || !frame(0.18f * 1024.0f) || !near(values[1], 0.3f)) {
            return RhiTestResult::fail("Speed Up must move 3 stops/s toward a brighter scene");
        }
        if (!frame(0.18f * 1024.0f, true) || !frame(0.18f) || !near(values[1], 9.9f)) {
            return RhiTestResult::fail("Speed Down must move 1 stop/s toward a darker scene");
        }
        if (!frame(0.18f, true) || !frame(0.36f)) { return RhiTestResult::fail("exponential step failed"); }
        const float exponential = values[1];
        if (exponential <= 0.0f || exponential >= 0.3f ||
            !frame(0.18f, true) || !frame(0.36f, false, 0, 0.05f) || !frame(0.36f, false, 0, 0.05f) ||
            !near(values[1], exponential, 0.001f)) {
            return RhiTestResult::fail("exponential adaptation depends on frame rate or overshoots");
        }
        // Cross from the linear region into the exponential region in one step.
        if (!frame(0.18f, true) || !frame(0.18f * 4.0f, false, 0, 0.4f)) {
            return RhiTestResult::fail("transition step failed");
        }
        const float crossing = values[1];
        if (!frame(0.18f, true) || !frame(0.18f * 4.0f, false, 0, 0.2f) ||
            !frame(0.18f * 4.0f, false, 0, 0.2f) || !near(values[1], crossing, 0.001f)) {
            return RhiTestResult::fail("linear/exponential transition depends on frame rate");
        }
        settings.minEV100 = 2.0f;
        settings.maxEV100 = 4.0f;
        if (!frame(0.18f, true) || !near(values[1], 2.0f) ||
            !frame(10000.0f, true) || !near(values[1], 4.0f)) {
            return RhiTestResult::fail("EV100 limits failed");
        }
        settings.minEV100 = settings.maxEV100 = 3.0f;
        if (!frame(0.18f) || !near(values[0], 0.125f, 0.001f)) {
            return RhiTestResult::fail("equal EV100 limits must force fixed exposure");
        }
        settings.minEV100 = -10;
        settings.maxEV100 = 20;
        settings.enabled = false;
        lighting.exposureEV100 = 2;
        settings.compensation = 1;
        if (!frame(10000.0f) || !near(values[0], 0.5f, 0.001f)) {
            return RhiTestResult::fail("manual EV100 or positive exposure compensation failed");
        }
        settings.enabled = true;
        settings.compensation = 0;
        settings.speedUp = settings.speedDown = 0;
        if (!frame(0.18f) || !frame(10000.0f) || !near(values[1], 0.0f)) {
            return RhiTestResult::fail("mode switch reset or zero-speed hold failed");
        }
        settings.lowPercent = 70;
        settings.highPercent = 90;
        if (!frame(1.0f, true) || !near(values[3], 1.0f) ||
            !frame(1.0f, true, 1) || !near(values[3], 1.0f) ||
            !frame(1.0f, true, 3) || !near(values[3], 1.0f)) {
            return RhiTestResult::fail("percentile clipping did not reject bright outliers");
        }
        if (!frame(0.0f, true) || !near(values[2], -10.0f - std::log2(0.18f)) ||
            !frame(-1.0f, true) || !frame(1.0f, true, 2)) {
            return RhiTestResult::fail("black, negative or NaN input produced invalid exposure");
        }
        if (!frame(1.0f) || !near(values[2], values[1])) {
            return RhiTestResult::fail("empty startup meter must not seed adaptation history");
        }
        settings.lowPercent = 0;
        settings.highPercent = 100;
        settings.speedUp = 3;
        settings.speedDown = 1;
        if (!frame(0.18f, true)) { return RhiTestResult::fail("overlap setup failed"); }
        graph.findNode(sourceId)->runtimeProperties = {{"luminance", 0.18f * 1024.0f}};
        executor.syncRuntimeProperties(graph);
        for (int i = 0; i < 4; ++i) {
            if (!executor.execute({.graphicsQueue = queue})) { return RhiTestResult::fail("overlapped frame failed"); }
        }
        if (!executor.waitForSubmittedWork() || !readExposure() || !near(values[1], 1.2f)) {
            return RhiTestResult::fail("overlapping frames did not serialize exposure history on the GPU");
        }
        if (!executor.compile(*device, graph, 1, 1, log) || !frame(0.18f * 1024.0f) || !near(values[1], 10.0f)) {
            return RhiTestResult::fail("resize/single-pixel exposure history reset failed: " + log);
        }
        return RhiTestResult::pass("GPU gray calibration, percentiles, limits, compensation, adaptation, resets and finite output");
    }
};

class AutoExposureSrgbTest final : public RhiTest {
public:
    AutoExposureSrgbTest() { type = RhiTestType::Rendering; name = "auto_exposure_reference_srgb"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::registerRenderGraphPassType("AutoExposureFixturePass", "HDR test fixture",
            [] { return std::make_unique<AutoExposureFixturePass>(); });
        render::RenderGraph graph;
        const uint32_t sourceId = graph.addNode("AutoExposureFixturePass", "Source")->id;
        graph.addNode("AutoExposurePass", "Exposure", {{"toneCurve", "none"}});
        graph.addEdge("Source.color", "Exposure.source");
        graph.markOutput("Exposure.color");
        render::RenderGraphPreviewRenderer preview;
        scene::LightingSettings lighting;
        lighting.autoExposure.enabled = false;
        // Verify that the display option still applies the physical exposure once.
        lighting.exposureEV100 = 2.0f;
        lighting.autoExposure.compensation = 1.0f;
        preview.setLighting(lighting);
        const auto result = preview.initialize(context.enableValidation);
        if (render::hasError(result, render::Error::Unsupported)) {
            return RhiTestResult::skip("sRGB test requires bindless descriptors");
        }
        if (!result) { return RhiTestResult::fail("sRGB renderer initialization failed"); }
        for (float linear : {0.0f, 0.001f, 0.0031308f, 0.18f, 0.8f, 1.0f, 16.0f}) {
            graph.findNode(sourceId)->runtimeProperties = {{"luminance", linear * 2.0f}};
            if (!preview.render(graph, 17, 9, "Exposure.color")) {
                return RhiTestResult::fail(preview.lastLog());
            }
            const float srgb = linear <= 0.0031308f ? linear * 12.92f
                : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
            const int expected = static_cast<int>(std::lround(std::min(srgb, 1.0f) * 255.0f));
            for (uint32_t pixel : preview.pixels()) {
                for (uint32_t shift : {0u, 8u, 16u}) {
                    if (std::abs(static_cast<int>((pixel >> shift) & 255u) - expected) > 1) {
                        return RhiTestResult::fail("sRGB display applied an unexpected tone curve or exposure");
                    }
                }
            }
        }
        return RhiTestResult::pass("sRGB toe, middle gray, display white and clipping with manual exposure");
    }
};

METALLIC_REGISTER_RHI_TEST(AutoExposureGpuTest);
METALLIC_REGISTER_RHI_TEST(AutoExposureSrgbTest);
} // namespace
} // namespace metallic::tests

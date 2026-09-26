#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
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

// Consume every AutoExposure export in another pass. This checks that the
// internal histogram/reduce/apply phases publish their writes to graph users.
class AutoExposureReadbackPass final : public render::ComputePass {
public:
    bool supportsFrameOverlap() const override { return true; }
    render::CpuRecordingPolicy cpuRecordingPolicy() const override
    {
        return render::CpuRecordingPolicy::ParallelJoined;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        const uint64_t histogramBytes = uint64_t((context.width + 15) / 16) *
            ((context.height + 15) / 16) * 64 * sizeof(uint32_t);
        render::RenderPassReflection reflection;
        reflection.addBufferInput("exposure").buffer(16, 16).transferRead();
        reflection.addBufferInput("histogram").buffer(histogramBytes, 4).transferRead();
        reflection.addTextureInput("color").transferRead().format = render::Format::Rgba8Unorm;
        reflection.addBufferOutput("data")
            .buffer(16 + histogramBytes + uint64_t(context.width) * context.height * 4)
            .transferWrite().hostReadback();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const auto exposure = context.inputBuffer("exposure");
        const auto histogram = context.inputBuffer("histogram");
        const auto color = context.inputTexture("color");
        const auto data = context.outputBuffer("data");
        if (!exposure.valid() || !histogram.valid() || !color.valid() || !data.valid()) {
            return render::makeError(render::Error::InvalidArgument);
        }
        auto& commands = context.commandBuffer();
        commands.copyBuffer({.source = exposure.buffer(), .destination = data.buffer(), .size = 16});
        commands.copyBuffer({.source = histogram.buffer(), .destination = data.buffer(),
            .destinationOffset = 16, .size = histogram.desc().size});
        commands.copyTextureToBuffer({.texture = color.texture(), .buffer = data.buffer(),
            .bufferOffset = 16 + histogram.desc().size, .width = context.width(), .height = context.height()});
        return {};
    }
};

class AutoExposureInternalStagesTest final : public RhiTest {
public:
    AutoExposureInternalStagesTest()
    {
        type = RhiTestType::Rendering;
        name = "auto_exposure_internal_stages_exports_and_cancel";
    }
    RhiTestResult run(RhiTestContext& context) override
    {
        constexpr uint32_t kWidth = 63, kHeight = 37;
        constexpr uint32_t kTilesX = (kWidth + 15) / 16, kTilesY = (kHeight + 15) / 16;
        constexpr uint32_t kHistogramCount = kTilesX * kTilesY * 64;
        render::registerRenderGraphPassType("AutoExposureFixturePass", "HDR test fixture",
            [] { return std::make_unique<AutoExposureFixturePass>(); });
        render::registerRenderGraphPassType("AutoExposureReadbackPass", "Auto exposure graph consumer",
            [] { return std::make_unique<AutoExposureReadbackPass>(); });
        for (bool preferUnified : {false, true}) {
            std::atomic_uint validationErrors{0};
            std::unique_ptr<render::Device> device;
            auto result = render::createDevice({.applicationName = "Auto exposure internal stages",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const render::ValidationMessage& message) noexcept {
                    if (message.severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}, .preferUnifiedImageLayouts = preferUnified})
                .transform([&](auto value) { device = std::move(value); });
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip("requires bindless descriptors");
            }
            if (!result) { return RhiTestResult::fail("internal-stage device creation failed"); }
            auto* queue = device->getQueue(render::QueueType::Graphics);
            if (queue == nullptr) { return RhiTestResult::fail("internal-stage device has no graphics queue"); }
            for (uint32_t workers : {1u, 4u}) {
                render::RenderWorld world;
                scene::LightingSettings lighting;
                lighting.autoExposure.lowPercent = 0;
                lighting.autoExposure.highPercent = 100;
                lighting.autoExposure.transitionDistance = 0;
                lighting.autoExposure.speedUp = 3;
                lighting.autoExposure.speedDown = 1;
                if (!world.setLighting(lighting)) { return RhiTestResult::fail("lighting setup failed"); }
                render::RenderGraph graph;
                const auto sourceId = graph.addNode("AutoExposureFixturePass", "Source")->id;
                graph.addNode("AutoExposurePass", "Exposure", {{"adaptationDeltaSeconds", 0.1f}, {"toneCurve", "none"}});
                graph.addNode("AutoExposureReadbackPass", "Readback");
                graph.addEdge("Source.color", "Exposure.source");
                graph.addEdge("Exposure.histogram", "Readback.histogram");
                graph.addEdge("Exposure.exposure", "Readback.exposure");
                graph.addEdge("Exposure.color", "Readback.color");
                graph.markOutput("Readback.data");
                render::RenderGraphExecutor executor;
                executor.bindRenderWorld(&world);
                std::string log;
                if (!executor.compile(*device, graph, kWidth, kHeight, log)) {
                    return RhiTestResult::fail("internal-stage graph compilation failed: " + log);
                }
                std::unique_ptr<render::CommandPool> pool;
                std::unique_ptr<render::CommandBuffer> commands;
                std::unique_ptr<render::Fence> fence;
                if (!device->createCommandPool(*queue).transform([&](auto value) { pool = std::move(value); }) ||
                    !pool->createCommandBuffer().transform([&](auto value) { commands = std::move(value); }) ||
                    !device->createFence(false).transform([&](auto value) { fence = std::move(value); })) {
                    return RhiTestResult::fail("external recording setup failed");
                }
                auto setLuminance = [&](float luminance) {
                    graph.findNode(sourceId)->runtimeProperties = {{"luminance", luminance}};
                    executor.syncRuntimeProperties(graph);
                };
                auto submitExternal = [&](bool cancel) {
                    if (!pool->reset() || !commands->begin() || !executor.execute(*commands) || !commands->end()) {
                        return false;
                    }
                    if (cancel) { return pool->reset().has_value(); }
                    render::CommandBuffer* submitted[] = {commands.get()};
                    return fence->reset() && queue->submit({.commandBuffers = submitted, .commandBufferCount = 1,
                        .signalFence = fence.get()}) && fence->wait(5'000'000'000ull);
                };
                auto checkOutput = [&](float luminance, float expectedEV) -> std::string {
                    auto* output = executor.outputResource("Readback.data");
                    if (output == nullptr || output->buffer == nullptr) { return "missing downstream readback"; }
                    output->buffer->invalidate();
                    const auto* mapped = static_cast<const uint8_t*>(output->buffer->map());
                    if (mapped == nullptr) { return "downstream readback is not mapped"; }
                    std::array<float, 4> exposure{};
                    std::array<uint32_t, kHistogramCount> histogram{};
                    std::array<uint32_t, kWidth * kHeight> pixels{};
                    std::memcpy(exposure.data(), mapped, sizeof(exposure));
                    std::memcpy(histogram.data(), mapped + sizeof(exposure), sizeof(histogram));
                    std::memcpy(pixels.data(), mapped + sizeof(exposure) + sizeof(histogram), sizeof(pixels));
                    output->buffer->unmap();
                    for (float value : exposure) {
                        if (!std::isfinite(value)) { return "downstream exposure is not finite"; }
                    }
                    if (std::abs(exposure[1] - expectedEV) > 0.02f ||
                        std::abs(exposure[0] - std::exp2(-expectedEV)) > 0.02f ||
                        std::abs(exposure[3] - luminance) > luminance * 0.02f) {
                        return "downstream exposure did not observe histogram/reduce or temporal history";
                    }
                    for (uint32_t tileY = 0; tileY < kTilesY; ++tileY) {
                        for (uint32_t tileX = 0; tileX < kTilesX; ++tileX) {
                            uint32_t weight = 0;
                            for (uint32_t bin = 0; bin < 64; ++bin) {
                                weight += histogram[(tileY * kTilesX + tileX) * 64 + bin];
                            }
                            const uint32_t expected = std::min(16u, kWidth - tileX * 16) *
                                std::min(16u, kHeight - tileY * 16) * 256;
                            if (weight != expected) { return "downstream histogram has missing or stale tile writes"; }
                        }
                    }
                    const float linear = std::min(luminance * exposure[0], 1.0f);
                    const float srgb = linear <= 0.0031308f ? linear * 12.92f
                        : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
                    const int expected = static_cast<int>(std::lround(srgb * 255));
                    for (uint32_t pixel : pixels) {
                        if ((pixel >> 24) != 255) { return "downstream color has unwritten pixels"; }
                        for (uint32_t shift : {0u, 8u, 16u}) {
                            if (std::abs(static_cast<int>((pixel >> shift) & 255) - expected) > 1) {
                                return "apply or downstream color copy did not observe the reduced exposure";
                            }
                        }
                    }
                    return {};
                };
                for (uint32_t frame = 0; frame < 4; ++frame) {
                    const float luminance = frame == 0 ? 0.18f : 0.72f;
                    setLuminance(luminance);
                    const bool executed = (frame & 1u) != 0 ? submitExternal(false)
                        : executor.execute({.graphicsQueue = queue, .recordingWorkerLimit = workers}) &&
                            executor.waitForSubmittedWork();
                    if (!executed) { return RhiTestResult::fail("mixed execution entry points failed"); }
                    const std::string failure = checkOutput(luminance, frame * 0.3f);
                    if (!failure.empty()) { return RhiTestResult::fail(failure); }
                }
                // Cancelling a recorded frame must invalidate its adaptation
                // history without committing accesses that never reached the GPU.
                setLuminance(0.18f);
                if (!submitExternal(true)) { return RhiTestResult::fail("cancelled exposure recording failed"); }
                setLuminance(0.72f);
                if (!executor.execute({.graphicsQueue = queue, .recordingWorkerLimit = workers}) ||
                    !executor.waitForSubmittedWork()) {
                    return RhiTestResult::fail("exposure execution after cancellation failed");
                }
                const std::string failure = checkOutput(0.72f, 2.0f);
                if (!failure.empty()) { return RhiTestResult::fail("cancelled history reset: " + failure); }
            }
            if (validationErrors.load() != 0) {
                return RhiTestResult::fail("Vulkan validation rejected AutoExposure internal-stage synchronization");
            }
        }
        return RhiTestResult::pass("histogram/reduce/apply exports, temporal history, external cancellation, 1/4 workers and both layout policies");
    }
};

class AutoExposureGpuTest final : public RhiTest {
public:
    AutoExposureGpuTest() { type = RhiTestType::Rendering; name = "auto_exposure_histogram_adaptation"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto deviceResult = render::createDevice({.applicationName = "Auto exposure GPU test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
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
METALLIC_REGISTER_RHI_TEST(AutoExposureInternalStagesTest);
} // namespace
} // namespace metallic::tests

#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "Runtime/Scene/SceneDocument.h"
#include "stb/stb_image_write.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numbers>
#include <thread>

namespace metallic::tests {
namespace {
constexpr uint32_t kWidth = 256, kHeight = 128, kLayers = 8;
constexpr uint32_t kTexels = kWidth * kHeight * kLayers;

class EnvironmentPrefilterFieldProbe final : public render::ComputePass {
public:
    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array ids{render::EnvironmentLightingSubsystem::kSubsystemId};
        return ids;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("field").buffer((kTexels + 2ull) * 16, 16).storageReadWrite();
        return reflection;
    }
    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "EnvironmentPrefilterFieldProbe",
            .entryPointName = "environmentPrefilterFieldProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {{.binding = 0}, {.binding = 1},
            {.binding = 2, .kind = render::ComputeResourceBindingKind::SampledImage}};
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
            .bindings = bindings, .bindingCount = 3, .requiresRayQuery = false}, log);
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        const auto& environment = context.subsystem<render::EnvironmentLightingSubsystem>()->snapshot();
        auto* source = environment.radianceView;
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .buffer = context.outputBuffer("field").buffer()},
            {.binding = 1, .buffer = environment.prefilteredSpecularBuffer},
            {.binding = 2, .textureViews = &source, .textureViewCount = 1}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings, .bindingCount = 3,
            .groupCountX = (kTexels + 2 + 63) / 64});
    }
private:
    render::ComputeProgram program_;
};

class EnvironmentPrefilterImpulseTest final : public RhiTest {
public:
    EnvironmentPrefilterImpulseTest() { type = RhiTestType::Rendering; name = "environment_prefilter_hdr_impulse"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::filesystem::create_directories(context.outputDirectory);
        std::unique_ptr<render::Device> device;
        auto result = render::createDevice({.applicationName = "HDR prefilter regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("Requires native bindless resources"); }
        if (!result) { return RhiTestResult::fail("Create prefilter device"); }
        render::registerRenderGraphPassType("EnvironmentPrefilterFieldProbe", "HDR prefilter field",
            [] { return std::make_unique<EnvironmentPrefilterFieldProbe>(); });
        std::string failure, log;
        for (uint32_t fixture = 0; fixture < 3; ++fixture) {
            const uint32_t width = fixture == 1 ? 35u : 1024u, height = fixture == 1 ? 17u : 512u;
            const uint32_t brightX = fixture == 1 ? 0u : width / 2, brightY = fixture == 1 ? 0u : height / 2;
            constexpr float bright = 32768.0f;
            auto path = std::filesystem::absolute(context.outputDirectory / (fixture == 0 ? "Needle.hdr" : "Pole.hdr"));
            if (fixture < 2) {
                std::vector<float> pixels(uint64_t(width) * height * 3, 1.0f);
                for (uint32_t channel = 0; channel < 3; ++channel) { pixels[(brightY * width + brightX) * 3 + channel] = bright; }
                if (!stbi_write_hdr(path.string().c_str(), width, height, 3, pixels.data())) { return RhiTestResult::fail("Write HDR fixture"); }
            } else { path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/environment.hdr"; }
            render::RenderWorld world;
            world.setEnvironment({.enabled = true, .path = path});
            render::RenderGraph graph;
            graph.addNode("EnvironmentPrefilterFieldProbe", "Probe");
            graph.markOutput("Probe.field");
            render::RenderGraphExecutor executor;
            executor.bindRenderWorld(&world);
            if (!executor.compile(*device, graph, 1, 1, log)) { return RhiTestResult::fail(log); }
            bool ready = false;
            for (uint32_t frame = 0; frame < 200 && !ready; ++frame) {
                if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
                    !executor.waitForSubmittedWork()) { return RhiTestResult::fail("Prefilter dispatch"); }
                ready = executor.subsystemHost()->get<render::EnvironmentLightingSubsystem>()->snapshot().mapAvailable;
                if (!ready) { std::this_thread::sleep_for(std::chrono::milliseconds(2)); }
            }
            if (!ready) { return RhiTestResult::fail("Environment publication timeout"); }
            auto* buffer = executor.outputResource("Probe.field")->buffer;
            buffer->invalidate();
            const auto* pixels = static_cast<const std::array<float, 4>*>(buffer->map());
            if (pixels == nullptr) { return RhiTestResult::fail("Prefilter readback"); }
            const char* name = fixture == 0 ? "Needle" : fixture == 1 ? "Pole" : "Forest";
            std::ofstream output(context.outputDirectory / (std::string(name) + ".f32"), std::ios::binary);
            output.write(reinterpret_cast<const char*>(pixels), (kTexels + 2ull) * 16);
            for (uint32_t texel = 0; texel < kTexels; ++texel) {
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    if (!std::isfinite(pixels[texel][channel]) || pixels[texel][channel] < 0.0f) {
                        failure = "Prefilter must produce finite nonnegative radiance";
                    }
                }
            }
            if (fixture < 2) {
                constexpr double pi = std::numbers::pi;
                const double solidAngle = (2.0 * pi / width) *
                    (std::cos(double(brightY) * pi / height) - std::cos(double(brightY + 1) * pi / height));
                const double power = (bright - 1.0) * solidAngle;
                const double mean = 1.0 + power / (4.0 * pi);
                spdlog::info("[Prefilter] {} source mean={} expected={} mipLevels={}", name, pixels[kTexels][0], mean, pixels[kTexels][3]);
                if (std::abs(pixels[kTexels][0] - mean) > mean * 0.002) { failure = "Source mip chain must preserve spherical radiance energy"; }
                // The mirror/background source must retain the original HDR peak.
                if (pixels[kTexels + 1][fixture] != bright) { failure = "Base HDR radiance must remain unchanged"; }
                if (fixture == 0) {
                    for (uint32_t layer = 3; layer < kLayers; ++layer) {
                        const double roughness = double(layer) / (kLayers - 1);
                        const double a2 = std::pow(roughness, 4);
                        double weight = 0.0;
                        for (uint32_t sample = 0; sample < 16384; ++sample) {
                            const double nl = (sample + 0.5) / 16384.0;
                            const double denominator = (nl + 1.0) * 0.5 * (a2 - 1.0) + 1.0;
                            weight += 0.5 * a2 / (denominator * denominator) * nl / 16384.0;
                        }
                        // Upper bound of the normalized GGX kernel convolved with one texel.
                        const double expectedPeak = 1.0 + power / (4.0 * pi * a2 * weight);
                        float maximum = 0;
                        for (uint32_t pixel = 0; pixel < kWidth * kHeight; ++pixel) {
                            maximum = std::max(maximum, pixels[layer * kWidth * kHeight + pixel][0]);
                        }
                        spdlog::info("[Prefilter] {} roughness={} peak={} analyticBound={}", name, roughness, maximum, expectedPeak);
                        // Allow finite-sample error on the emitter's contribution,
                        // excluding the unit background from the relative tolerance.
                        if (maximum > 1.0 + (expectedPeak - 1.0) * 1.2 + 0.01) {
                            failure = "HDR impulse creates spurious GGX reflection peaks";
                        }
                    }
                }
            }
            buffer->unmap();
        }
        return failure.empty() ? RhiTestResult::pass("HDR impulse peak bound, NPOT polar mip energy and forest field readback")
            : RhiTestResult::fail(failure);
    }
};

class EnvironmentPrefilterCaptureViewTest final : public RhiTest {
public:
    EnvironmentPrefilterCaptureViewTest() { type = RhiTestType::Rendering; name = "environment_prefilter_capture_view"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("realtime-lighting", sample, log)) { return RhiTestResult::fail(log); }
        // The camera recovered from LookDev_2026_09_11_12_23_50.ngfx-capture.
        auto view = sample.graph.viewProperties();
        view["camera"] = {{"eye", {-0.8574517965, 1.6639534235, 1.8709540367}},
            {"center", {0.6719443798, 2.1819956303, -1.814501524}}, {"up", {0, 1, 0}},
            {"fovDegrees", 45}, {"znear", 0.0016666661}, {"zfar", 166.6666107}, {"reversedZ", true}};
        view["temporalJitter"] = false;
        sample.graph.setViewProperties(view);
        auto* sr = sample.graph.findNode("DlssSr");
        auto* nr = sample.graph.findNode("DlssNr");
        if (sr == nullptr || nr == nullptr) { return RhiTestResult::fail("Missing realtime sample reconstruction nodes"); }
        const auto srId = sr->id, nrId = nr->id;
        sample.graph.removeNode(srId);
        sample.graph.removeNode(nrId);
        sample.graph.addEdge("Deferred.color", "AutoExposure.source");
        sample.graph.addEdge("AutoExposure.color", "FinalBlit.source");
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) { return RhiTestResult::fail(scene.lastLoadResult().error); }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        auto result = preview.initialize(context.enableValidation, true);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("Requires raster mesh shaders and ray queries"); }
        if (!result) { return RhiTestResult::fail("Initialize capture view"); }
        preview.setEnvironment({.enabled = true, .path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/environment.hdr"});
        for (uint32_t frame = 0; frame < 32; ++frame) {
            if (!preview.render(sample.graph, 1198, 438)) { return RhiTestResult::fail(preview.lastLog()); }
        }
        if (!saveRgba8Png(context.outputDirectory / "CaptureView.png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                preview.width(), preview.height(), log)) { return RhiTestResult::fail(log); }
        return RhiTestResult::pass("Captured camera renders through deferred lighting without SR or NR");
    }
};

METALLIC_REGISTER_RHI_TEST(EnvironmentPrefilterImpulseTest);
METALLIC_REGISTER_RHI_TEST(EnvironmentPrefilterCaptureViewTest);
} // namespace
} // namespace metallic::tests

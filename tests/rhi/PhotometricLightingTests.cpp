#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/SceneLightResources.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"
#include "stb/stb_image_write.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace metallic::tests {
namespace {

class PhotometricProbePass final : public render::UnsafePass {
public:
    std::span<const render::RenderSubsystemId> requiredSubsystems() const override
    {
        static constexpr std::array ids{render::EnvironmentLightingSubsystem::kSubsystemId};
        return ids;
    }
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(12 * 16, 16).storageReadWrite();
        return reflection;
    }
    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        device_ = context.device;
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "PhotometricProbe",
            .entryPointName = "photometricProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {
            {.binding = 0, .kind = render::ComputeResourceBindingKind::StorageBuffer},
            {.binding = 1, .kind = render::ComputeResourceBindingKind::StorageBuffer},
            {.binding = 50, .kind = render::ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*device_, {.spirv = shader.spirv.data(),
            .byteSize = shader.spirv.size() * sizeof(uint32_t), .bindings = bindings,
            .bindingCount = 3, .requiresRayQuery = false}, log);
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        auto result = lights_.update(*device_, context.commandBuffer(), *context.subsystems(),
            nullptr, context.world()->lighting());
        if (!result) { return result; }
        const auto& environment = context.subsystem<render::EnvironmentLightingSubsystem>()->snapshot();
        const render::ComputeDispatchBinding bindings[] = {
            {.binding = 0, .buffer = context.outputBuffer("data").buffer()},
            {.binding = 1, .buffer = environment.sphericalHarmonicsBuffer},
            {.binding = 50, .buffer = lights_.buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings, .bindingCount = 3});
    }
private:
    render::Device* device_ = nullptr;
    render::ComputeProgram program_;
    render::SceneLightResources lights_;
};

class PhotometricGpuTest final : public RhiTest {
public:
    PhotometricGpuTest() { type = RhiTestType::Rendering; name = "photometric_gpu_units_falloff_sh"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        auto deviceResult = render::createDevice({.applicationName = "GPU photometry",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(deviceResult, render::Error::Unsupported)) {
            return RhiTestResult::skip("requires bindless descriptors");
        }
        if (!deviceResult) { return RhiTestResult::fail("GPU photometry device creation failed"); }
        auto* queue = device->getQueue(render::QueueType::Graphics);
        const auto hdrPath = std::filesystem::absolute(context.outputDirectory / "constant-photometric.hdr");
        std::filesystem::create_directories(context.outputDirectory);
        std::vector<float> pixels(64 * 32 * 3, 2.0f);
        if (!stbi_write_hdr(hdrPath.string().c_str(), 64, 32, 3, pixels.data())) {
            return RhiTestResult::fail("cannot create constant HDR fixture");
        }
        render::RenderWorld world;
        world.setEnvironment({.enabled = true, .path = hdrPath});
        scene::LightingSettings lighting;
        lighting.exposureEV100 = 2.0f;
        for (const char* type : {"point", "directional", "spot"}) {
            scene::PunctualLight light;
            light.properties.type = type;
            light.properties.intensity = 100.0;
            light.position = float3(0.0f);
            light.direction = float3(0.0f, 0.0f, 1.0f);
            lighting.lights.push_back(light);
        }
        if (!world.setLighting(lighting)) { return RhiTestResult::fail("invalid fixture lighting"); }
        render::registerRenderGraphPassType("PhotometricProbePass", "GPU photometry test",
            [] { return std::make_unique<PhotometricProbePass>(); });
        render::RenderGraph graph;
        graph.addNode("PhotometricProbePass", "Probe");
        graph.markOutput("Probe.data");
        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        std::string log;
        auto result = executor.compile(*device, graph, 1, 1, log);
        if (!result) { return RhiTestResult::fail(log); }
        std::array<float, 48> reference{};
        for (int iteration = 0; iteration < 3; ++iteration) {
            if (iteration == 1) {
                lighting.lights[0].properties.intensityUnit = scene::LightUnit::Lumens;
                lighting.lights[0].properties.intensity = 400.0 * 3.14159265358979323846;
                lighting.lights[1].properties.intensityUnit = scene::LightUnit::EV100;
                lighting.lights[1].properties.intensity = std::log2(40.0);
                lighting.lights[2].properties.intensityUnit = scene::LightUnit::EV100;
                lighting.lights[2].properties.intensity = std::log2(100.0);
                world.setLighting(lighting);
            } else if (iteration == 2) {
                lighting.lights[0].properties.intensity *= 2.0;
                world.setLighting(lighting);
            }
            bool ready = false;
            for (int attempt = 0; attempt < 200; ++attempt) {
                result = executor.execute({.graphicsQueue = queue});
                if (!result) { return RhiTestResult::fail(std::string("probe execution failed: ") + toString(result)); }
                result = executor.waitForSubmittedWork(10'000'000'000ull);
                if (!result) { return RhiTestResult::fail("probe wait failed"); }
                const auto& snapshot = executor.subsystemHost()->get<render::EnvironmentLightingSubsystem>()->snapshot();
                ready = snapshot.mapAvailable && snapshot.status == render::EnvironmentLightingStatus::Ready;
                if (ready) { break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            if (!ready) {
                return RhiTestResult::fail("constant HDR did not finish GPU publication: " +
                    executor.subsystemHost()->get<render::EnvironmentLightingSubsystem>()->snapshot().error);
            }
            auto* buffer = executor.outputResource("Probe.data")->buffer;
            buffer->invalidate();
            void* mapped = buffer->map();
            if (mapped == nullptr) { return RhiTestResult::fail("probe readback failed"); }
            std::array<float, 48> values;
            std::memcpy(values.data(), mapped, sizeof(values));
            buffer->unmap();
            if (iteration == 0) {
                reference = values;
                const std::array<float, 6> expected{100, 25, 100, 100, 100, 0};
                for (size_t i = 0; i < expected.size(); ++i) {
                    if (std::abs(values[4 * i] - expected[i]) > 0.001f) {
                        return RhiTestResult::fail("GPU inverse-square/directional/spot mismatch at " + std::to_string(i));
                    }
                }
                if (values[3] != 0.25f) { return RhiTestResult::fail("GPU exposure metadata mismatch"); }
                for (size_t i = 6; i < 12; ++i) {
                    for (size_t c = 0; c < 3; ++c) {
                        if (std::abs(values[i * 4 + c] - 6.2831853f) > 0.012f) {
                            return RhiTestResult::fail("GPU irradiance SH must evaluate to pi * radiance in every direction");
                        }
                    }
                }
            } else if (iteration == 1) {
                for (size_t i = 0; i < values.size(); ++i) {
                    if (std::abs(values[i] - reference[i]) > 0.001f) {
                        return RhiTestResult::fail("unit conversion changed GPU lighting");
                    }
                }
            } else if (std::abs(values[0] - 200.0f) > 0.001f) {
                return RhiTestResult::fail("light edit did not update GPU snapshot");
            }
        }
        return RhiTestResult::pass("GPU: SI/EV/lumen equivalence, inverse square, spot cutoff, exposure, SH and live edits");
    }
};

class RealtimeLightingRenderTest final : public RhiTest {
public:
    RealtimeLightingRenderTest() { type = RhiTestType::Rendering; name = "photometric_realtime_render"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, true);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("requires ray query"); }
        if (!result) { return RhiTestResult::fail("preview initialization failed"); }
        preview.setEnvironment({.enabled = false});
        render::RenderGraph graph;
        graph.addNode("SceneRealtimeLightingPass", "Lighting", {{"path", "Asset/meet_mat.glb"},
            {"camera", {{"eye", {0.0, 0.25, 3.0}}, {"center", {0.0, 0.15, 0.0}}}}});
        graph.markOutput("Lighting.color");
        result = preview.render(graph, 64, 64);
        if (!result) { return RhiTestResult::fail(preview.lastLog()); }
        const auto dark = preview.pixels();
        scene::LightingSettings settings;
        auto& light = settings.lights.emplace_back();
        light.properties.type = "directional";
        light.properties.intensityUnit = scene::LightUnit::Lux;
        light.properties.intensity = 1000;
        light.direction = float3(0.0f, -0.2f, -1.0f);
        settings.exposureEV100 = 8;
        preview.setLighting(settings);
        result = preview.render(graph, 64, 64);
        if (!result) { return RhiTestResult::fail(preview.lastLog()); }
        const auto lit = preview.pixels();
        if (lit == dark) { return RhiTestResult::fail("directional light did not affect real-time material shading"); }
        settings.lights[0].properties.intensityUnit = scene::LightUnit::EV100;
        settings.lights[0].properties.intensity = std::log2(400.0);
        preview.setLighting(settings);
        result = preview.render(graph, 64, 64);
        if (!result || preview.pixels() != lit) {
            return RhiTestResult::fail("equivalent lux/EV units changed real-time shading");
        }
        settings.lights.clear();
        settings.exposureEV100 = 0;
        preview.setLighting(settings);
        result = preview.render(graph, 64, 64);
        if (!result || preview.pixels() != dark) { return RhiTestResult::fail("deleted light persisted in GPU lighting"); }
        std::string message;
        if (!saveRgba8Png(context.outputDirectory / "physical-realtime.png",
                reinterpret_cast<const uint8_t*>(lit.data()), 64, 64, message)) {
            return RhiTestResult::fail(message);
        }
        render::RenderGraph hdrGraph;
        hdrGraph.addNode("SceneRealtimeLightingPass", "Lighting", {{"path", "Asset/meet_mat.glb"},
            {"outputLinear", true}, {"camera", {{"eye", {0.0, 0.25, 3.0}}, {"center", {0.0, 0.15, 0.0}}}}});
        const uint32_t exposureId = hdrGraph.addNode("AutoExposurePass", "Exposure")->id;
        hdrGraph.addEdge("Lighting.color", "Exposure.source");
        hdrGraph.markOutput("Exposure.color");
        auto& hdrLight = settings.lights.emplace_back();
        hdrLight.properties.type = "directional";
        hdrLight.properties.intensityUnit = scene::LightUnit::Lux;
        hdrLight.properties.intensity = 1000;
        hdrLight.direction = float3(0.0f, -0.2f, -1.0f);
        settings.exposureEV100 = 8;
        settings.autoExposure.enabled = false;
        preview.setLighting(settings);
        result = preview.render(hdrGraph, 64, 64);
        if (!result || preview.pixels() != lit) {
            return RhiTestResult::fail("HDR + manual post exposure differs from inline exposure: " + preview.lastLog());
        }
        settings.autoExposure.enabled = true;
        preview.setLighting(settings);
        result = preview.render(hdrGraph, 64, 64);
        if (!result) { return RhiTestResult::fail(preview.lastLog()); }
        const auto automatic = preview.pixels();
        double subjectBrightness = 0.0;
        size_t subjectPixels = 0;
        for (uint32_t pixel : automatic) {
            const uint32_t value = pixel & 255u;
            if (value > 0) { subjectBrightness += value; ++subjectPixels; }
        }
        if (subjectPixels == 0 || subjectBrightness / subjectPixels > 190.0 ||
            subjectBrightness / subjectPixels < 40.0) {
            return RhiTestResult::fail("black background dominated metering and lost subject detail");
        }
        settings.lights[0].properties.intensity *= 1024;
        preview.setLighting(settings);
        hdrGraph.findNode(exposureId)->runtimeProperties = {{"resetSerial", 1}};
        result = preview.render(hdrGraph, 64, 64);
        if (!result) { return RhiTestResult::fail(preview.lastLog()); }
        double error = 0.0;
        for (size_t i = 0; i < automatic.size(); ++i) {
            for (uint32_t shift : {0u, 8u, 16u}) {
                error += std::abs(int((automatic[i] >> shift) & 255u) -
                    int((preview.pixels()[i] >> shift) & 255u));
            }
        }
        if (error / (automatic.size() * 3) > 8.0) {
            return RhiTestResult::fail("automatic exposure did not compensate a 10-stop physical light increase");
        }
        if (!saveRgba8Png(context.outputDirectory / "auto-exposure-realtime.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), 64, 64, message)) {
            return RhiTestResult::fail(message);
        }
        return RhiTestResult::pass("physical lighting and HDR/manual equivalence; automatic 10-stop compensation");
    }
};

METALLIC_REGISTER_RHI_TEST(PhotometricGpuTest);
METALLIC_REGISTER_RHI_TEST(RealtimeLightingRenderTest);
} // namespace
} // namespace metallic::tests

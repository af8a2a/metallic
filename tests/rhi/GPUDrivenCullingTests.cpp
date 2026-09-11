#include "RhiTest.h"
#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/SlangCompiler.h"
#include "Runtime/Scene/SceneDocument.h"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>

namespace metallic::tests {
namespace {

class GPUDrivenConeProbe final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addBufferOutput("data").buffer(32 * 16, 16).storageReadWrite();
        return reflection;
    }
    render::Result compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        render::ShaderCompileResult shader;
        auto result = render::compileSlangShaderToSpirv({.moduleName = "GPUDrivenConeProbe",
            .entryPointName = "gpuDrivenConeProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"}, shader);
        if (!result) { log = shader.diagnostics; return result; }
        const render::ComputeProgramBindingDesc bindings[] = {{.binding = 0}};
        return program_.initialize(*context.device, {.spirv = shader.spirv.data(), .byteSize = shader.spirv.size() * 4,
            .bindings = bindings, .bindingCount = 1, .requiresRayQuery = false}, log);
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        const render::ComputeDispatchBinding bindings[] = {{.binding = 0, .buffer = context.outputBuffer("data").buffer()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings, .bindingCount = 1});
    }
private:
    render::ComputeProgram program_;
};

class GPUDrivenConeScaleTest final : public RhiTest {
public:
    GPUDrivenConeScaleTest() { type = RhiTestType::Rendering; name = "gpu_driven_cone_scale_invariance"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "Cone scale regression",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RhiTestResult::skip("Requires bindless descriptors"); }
        if (!initialized) { return RhiTestResult::fail("Cone probe device creation failed"); }
        render::registerRenderGraphPassType("GPUDrivenConeProbe", "Normal cone scale probe",
            [] { return std::make_unique<GPUDrivenConeProbe>(); });
        render::RenderGraph graph;
        graph.addNode("GPUDrivenConeProbe", "Probe");
        graph.markOutput("Probe.data");
        render::RenderGraphExecutor executor;
        std::string log;
        if (!executor.compile(*device, graph, 1, 1, log)) { return RhiTestResult::fail(log); }
        if (!executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
            !executor.waitForSubmittedWork()) { return RhiTestResult::fail("Cone probe dispatch failed"); }
        auto* buffer = executor.outputResource("Probe.data")->buffer;
        buffer->invalidate();
        const auto* data = static_cast<const std::array<float, 4>*>(buffer->map());
        if (data == nullptr) { return RhiTestResult::fail("Cone probe readback failed"); }
        bool valid = true;
        for (uint32_t index = 0; index < 16; ++index) {
            const auto& axis = data[index * 2];
            const auto& expected = data[index * 2 + 1];
            for (uint32_t channel = 0; channel < 3; ++channel) {
                valid = valid && std::isfinite(axis[channel]) && std::abs(axis[channel] - expected[channel]) < 1e-5f;
            }
            valid = valid && axis[3] == 1.0f && expected[3] == 0.0f;
            spdlog::info("[Cone] case={} axis=({},{},{}) front={} back={}", index, axis[0], axis[1], axis[2], axis[3], expected[3]);
        }
        buffer->unmap();
        return valid ? RhiTestResult::pass("Normal cone front/back decisions survive uniform scales and rotation")
            : RhiTestResult::fail("Object scale changed the cone axis or rejected a front-facing meshlet");
    }
};

class GPUDrivenSponzaCullingTest final : public RhiTest {
public:
    GPUDrivenSponzaCullingTest() { type = RhiTestType::Rendering; name = "gpu_driven_sponza_culling_equivalence"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/Sponza/glTF/Sponza.gltf")) {
            return RhiTestResult::fail(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders"); }
        if (!initialized) { return RhiTestResult::fail("Preview initialization failed"); }
        preview.bindRuntimeScene(&scene);
        render::RenderGraph graph;
        const uint32_t raster = graph.addNode("VisibilityBufferPass", "VBuffer",
            {{"path", "Asset/Sponza/glTF/Sponza.gltf"}, {"visualization", "meshlet"}})->id;
        graph.markOutput("VBuffer.color");
        const std::array<const char*, 4> flags{
            "instanceFrustumCull", "instanceHzbCull", "meshletFrustumCull", "meshletNormalConeCull"};
        const std::array<std::array<double, 6>, 3> cameras{{
            {-5.646879, 11.323325, -0.051334, 29.722519, -15.616591, -5.377987},
            {-3.325359, 9.545668, -0.429330, 31.658249, -17.904564, -5.697884},
            {-4.960654, 10.867253, -0.183054, 29.629982, -17.087643, -5.392424}}};
        std::string log;
        bool valid = true;
        for (uint32_t cameraIndex = 0; cameraIndex < cameras.size(); ++cameraIndex) {
            const auto& c = cameras[cameraIndex];
            graph.setViewProperties({{"camera", {{"eye", {c[0], c[1], c[2]}}, {"center", {c[3], c[4], c[5]}},
                {"up", {0, 1, 0}}, {"fovDegrees", 45.0}, {"znear", 0.018548}, {"zfar", 1854.789185}, {"reversedZ", true}}},
                {"temporalJitter", false}});
            std::vector<uint32_t> reference;
            // No culling is the oracle. Then exercise all culling and disable one
            // stage at a time, including several frames of temporal HZB reuse.
            for (int configuration = -2; configuration < 4; ++configuration) {
                for (int flag = 0; flag < 4; ++flag) {
                    graph.setNodeRuntimeProperty(raster, flags[flag], configuration != -2 && configuration != flag);
                }
                for (uint32_t frame = 0; frame < 3; ++frame) {
                    if (!preview.render(graph, 799, 292)) { return RhiTestResult::fail(preview.lastLog()); }
                    if (configuration != -2 && preview.pixels() != reference) { valid = false; }
                }
                const auto& pixels = preview.pixels();
                if (configuration == -2) {
                    reference = pixels;
                    const auto covered = std::count_if(pixels.begin(), pixels.end(),
                        [](uint32_t pixel) { return (pixel & 255u) >= 64u; });
                    if (covered < 10000) { return RhiTestResult::fail("Sponza reference contains too little geometry"); }
                }
                size_t changed = 0;
                for (size_t pixel = 0; pixel < pixels.size(); ++pixel) { changed += pixels[pixel] != reference[pixel]; }
                spdlog::info("[Sponza culling] camera={} configuration={} changed={}", cameraIndex, configuration, changed);
                if (!saveRgba8Png(context.outputDirectory / ("Sponza" + std::to_string(cameraIndex) +
                        "Cull" + std::to_string(configuration) + ".png"), reinterpret_cast<const uint8_t*>(pixels.data()),
                        preview.width(), preview.height(), log)) { return RhiTestResult::fail(log); }
            }
        }
        // Review artifact with actual materials/lighting at the failing camera.
        // Native resolution isolates culling from temporal reconstruction.
        render::RenderSampleLoadResult sample;
        if (!render::loadBuiltInRenderSample(render::kDefaultGPUDrivenSampleId, sample, log)) { return RhiTestResult::fail(log); }
        const auto* sr = sample.graph.findNode("DlssSr");
        const auto* nr = sample.graph.findNode("DlssNr");
        if (sr == nullptr || nr == nullptr || !sample.desc.environment.has_value()) {
            return RhiTestResult::fail("Missing realtime sample reconstruction or environment settings");
        }
        const auto srId = sr->id, nrId = nr->id;
        sample.graph.removeNode(srId);
        sample.graph.removeNode(nrId);
        sample.graph.addEdge("Deferred.color", "AutoExposure.source");
        sample.graph.addEdge("AutoExposure.color", "FinalBlit.source");
        const auto& c = cameras[1];
        sample.graph.setViewProperties({{"camera", {{"eye", {c[0], c[1], c[2]}}, {"center", {c[3], c[4], c[5]}},
            {"up", {0, 1, 0}}, {"fovDegrees", 45.0}, {"znear", 0.018548}, {"zfar", 1854.789185}, {"reversedZ", true}}},
            {"temporalJitter", false}});
        preview.setEnvironment({.enabled = true, .path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
        for (uint32_t frame = 0; frame < 32; ++frame) {
            if (!preview.render(sample.graph, 1198, 438)) { return RhiTestResult::fail(preview.lastLog()); }
        }
        if (!saveRgba8Png(context.outputDirectory / "SponzaShaded.png", reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                preview.width(), preview.height(), log)) { return RhiTestResult::fail(log); }
        return valid ? RhiTestResult::pass("Sponza captured cameras match unculled visibility with every culling stage enabled")
            : RhiTestResult::fail("Culling changed visible Sponza geometry");
    }
};

METALLIC_REGISTER_RHI_TEST(GPUDrivenConeScaleTest);
METALLIC_REGISTER_RHI_TEST(GPUDrivenSponzaCullingTest);
} // namespace
} // namespace metallic::tests

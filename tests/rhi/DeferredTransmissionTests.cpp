#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>

namespace metallic::tests {
namespace {

size_t changedPixels(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b, int tolerance)
{
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        for (uint32_t shift : {0u, 8u, 16u}) {
            if (std::abs(int((a[i] >> shift) & 255u) - int((b[i] >> shift) & 255u)) > tolerance) {
                ++count;
                break;
            }
        }
    }
    return count;
}

class DeferredTransmissionTest final : public RHITest {
public:
    DeferredTransmissionTest() { type = RHITestType::Rendering; name = "visibility_buffer_abeautiful_game_transmission"; }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-abeautiful-game", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(scene.lastLoadResult().error);
        }
        if (scene.materials().size() != 15 || scene.materials()[5].transmissionFactor != 1.0f ||
            scene.materials()[7].transmissionFactor != 1.0f) { return RHITestResult::fail("Unexpected ABeautifulGame materials"); }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        auto result = preview.initialize(context.enableValidation, true, false);
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and ray queries"); }
        if (!result) { return RHITestResult::fail("Preview initialization failed"); }
        preview.setEnvironment({.enabled = true,
            .path = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/ABeautifulGame/environment.hdr"});
        auto lighting = scene.lighting();
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 0.0f;
        preview.setLighting(lighting);
        const uint32_t deferred = sample.graph.findNode("Deferred")->id;
        // Keep only the real-time path during scheduling comparisons and timing.
        sample.graph.removeNode(sample.graph.findNode("Reference")->id);
        sample.graph.removeNode(sample.graph.findNode("Slider")->id);
        sample.graph.addEdge("Deferred.color", "AutoExposure.source");
        sample.graph.setNodeRuntimeProperty(deferred, "accumulate", false);
        const auto save = [&](const char* name) {
            return saveRgba8Png(context.outputDirectory / name,
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), preview.width(), preview.height(), log);
        };
        const auto renderFrame = [&](uint32_t width = 385, uint32_t height = 257) {
            if (!preview.render(sample.graph, width, height)) { log = preview.lastLog(); return false; }
            return true;
        };
        sample.graph.setNodeRuntimeProperty(deferred, "debugView", "final");
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
        if (!renderFrame()) { return RHITestResult::fail(log); }
        const auto defaultFlat = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
        if (!renderFrame()) { return RHITestResult::fail(log); }
        const auto defaultBinned = preview.pixels();
        if (changedPixels(defaultFlat, defaultBinned, 2) > defaultFlat.size() / 1000) {
            save("ABeautifulGameDefaultBinningMismatch.png");
            return RHITestResult::fail("Default deferred flat/binned output differs");
        }
        if (!save("ABeautifulGameDeferredDefault.png")) { return RHITestResult::fail(log); }
        // Native FP16 uses the identical raster estimator as its FP32 reference.
        sample.graph.setNodeRuntimeProperty(deferred, "halfPrecision", false);
        if (!renderFrame()) { return RHITestResult::fail(log); }
        const auto fp32 = preview.pixels();
        if (!save("ABeautifulGameFP32.png")) { return RHITestResult::fail(log); }
        if (changedPixels(fp32, defaultBinned, 2) != 0) {
            return RHITestResult::fail("FP16 energy weights exceeded 2/255 display error");
        }
        sample.graph.setNodeRuntimeProperty(deferred, "halfPrecision", true);
        if (!renderFrame() || preview.pixels() != defaultBinned) {
            return RHITestResult::fail("FP16 toggle failed to restore the raster image: " + log);
        }
        // Old assets may contain these fields. They must never re-enable tracing.
        sample.graph.setNodeRuntimeProperty(deferred, "supplementaryPathTracing", true);
        sample.graph.setNodeRuntimeProperty(deferred, "transmissionSamples", 16);
        sample.graph.setNodeRuntimeProperty(deferred, "transmissionDepth", 16);
        sample.graph.setNodeRuntimeProperty(deferred, "lightingMode", "reference");
        if (!renderFrame() || preview.pixels() != defaultBinned) {
            return RHITestResult::fail("Legacy properties changed the raster-only path: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "exportUpscalerGuides", true);
        if (!renderFrame() || changedPixels(preview.pixels(), defaultBinned, 2) != 0) {
            return RHITestResult::fail("Upscaler guides changed raster shading: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "exportUpscalerGuides", false);
        for (const char* debug : {"baseColor", "shadingNormal", "material", "final"}) {
            sample.graph.setNodeRuntimeProperty(deferred, "debugView", debug);
            sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
            if (!renderFrame()) { return RHITestResult::fail(log); }
            const auto flat = preview.pixels();
            sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
            if (!renderFrame()) { return RHITestResult::fail(log); }
            if (changedPixels(flat, preview.pixels(), 2) > flat.size() / 1000) {
                save("ABeautifulGameBinningMismatch.png");
                return RHITestResult::fail(std::string(debug) + " flat/binned output differs");
            }
        }
        const auto glassImage = preview.pixels();
        if (!save("ABeautifulGameDeferred.png")) { return RHITestResult::fail(log); }
        sample.graph.setNodeRuntimeProperty(deferred, "debugDisableTransmission", true);
        if (!renderFrame()) { return RHITestResult::fail(log); }
        const size_t transmissionPixels = changedPixels(glassImage, preview.pixels(), 4);
        if (transmissionPixels < 50) { return RHITestResult::fail("Environment transmission had no visible effect"); }
        if (!save("ABeautifulGameTransmissionDisabled.png")) { return RHITestResult::fail(log); }
        sample.graph.setNodeRuntimeProperty(deferred, "debugDisableTransmission", false);
        if (!renderFrame() || changedPixels(glassImage, preview.pixels(), 1) != 0) {
            return RHITestResult::fail("Transmission toggle did not restore the image: " + log);
        }
        // Volume distance has no meaning without an interior path. Material edits
        // still refresh bindings but must not reintroduce volume integration.
        for (uint32_t index : {5u, 7u}) {
            auto glass = scene.materials()[index];
            glass.attenuationDistance = 0.015f;
            glass.attenuationColor = float3(0.1f, 0.35f, 0.8f);
            if (!scene.setMaterialProperties(index, glass)) { return RHITestResult::fail("Volume edit failed"); }
        }
        if (!renderFrame() || preview.pixels() != glassImage) {
            return RHITestResult::fail("Raster shading unexpectedly integrated an interior volume: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
        // Extent changes recreate scratch allocations. Compare at an odd extent.
        if (!renderFrame(193, 157)) { return RHITestResult::fail(log); }
        const auto resized = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
        if (!renderFrame(193, 157) || changedPixels(resized, preview.pixels(), 2) > resized.size() / 1000) {
            return RHITestResult::fail("Resized material queues changed shading: " + log);
        }
        // Report end-to-end readback timing, not an inferred GPU speedup.
        std::ofstream timings(context.outputDirectory / "ABeautifulGameTiming.txt");
        timings << "Raster OpenPBR; SH + prefiltered IBL; FP16 weights; no accumulation\n"
            << "Median preview.render wall time, including CPU submission and readback; validation=" << context.enableValidation << '\n';
        for (bool binned : {false, true}) {
            sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", binned);
            for (const auto extent : {std::array<uint32_t, 2>{385, 257}, {1280, 720}}) {
                for (int warmup = 0; warmup < 3; ++warmup) {
                    if (!renderFrame(extent[0], extent[1])) { return RHITestResult::fail(log); }
                }
                std::vector<double> elapsed;
                std::vector<double> gpuDeferred;
                std::vector<render::RenderGraphExecutionStats> completed;
                if (auto drained = preview.collectCompletedGpuExecutionStats(); !drained) {
                    return RHITestResult::fail("GPU timestamp readback failed after warmup");
                }
                for (int frame = 0; frame < 12; ++frame) {
                    const auto start = std::chrono::steady_clock::now();
                    if (!renderFrame(extent[0], extent[1])) { return RHITestResult::fail(log); }
                    elapsed.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                    if (!preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) {
                        return RHITestResult::fail("GPU timestamp readback failed");
                    }
                    for (const auto& stats : completed) {
                        for (const auto& node : stats.nodes) {
                            if (node.name == "Deferred" && node.gpuTimingAvailable) { gpuDeferred.push_back(node.gpuMilliseconds); }
                        }
                    }
                }
                std::sort(elapsed.begin(), elapsed.end());
                timings << (binned ? "binned " : "flat ") << extent[0] << 'x' << extent[1] << ": "
                    << (elapsed[5] + elapsed[6]) * 0.5 << " ms\n";
                if (!gpuDeferred.empty()) {
                    std::sort(gpuDeferred.begin(), gpuDeferred.end());
                    timings << "  Deferred GPU timestamp median: " << gpuDeferred[gpuDeferred.size() / 2] << " ms\n";
                } else {
                    timings << "  Deferred GPU timestamps unavailable\n";
                }
            }
        }
        // Keep a reviewable raster/environment approximation versus path-traced reference.
        if (!render::loadBuiltInRenderSample("lookdev-abeautiful-game", sample, log)) { return RHITestResult::fail(log); }
        for (int frame = 0; frame < 32; ++frame) {
            if (!preview.render(sample.graph, 512, 384)) { return RHITestResult::fail(preview.lastLog()); }
        }
        if (!save("ABeautifulGameComparison.png")) { return RHITestResult::fail(log); }
        for (float split : {0.0f, 1.0f}) {
            sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", split);
            if (!preview.render(sample.graph, 512, 384) ||
                !save(split == 0.0f ? "ABeautifulGameDeferredAccumulated.png" : "ABeautifulGameReference.png")) {
                return RHITestResult::fail(preview.lastLog() + log);
            }
        }
        return RHITestResult::pass("15 materials: raster-only legacy settings, FP16/FP32 precision, flat/binned equality, "
            "environment transmission, no volume integration, edits, resize and HDRI comparison");
    }
};

METALLIC_REGISTER_RHI_TEST(DeferredTransmissionTest);

} // namespace
} // namespace metallic::tests

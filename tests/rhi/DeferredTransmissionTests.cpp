#include "RhiTest.h"
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

class DeferredTransmissionTest final : public RhiTest {
public:
    DeferredTransmissionTest() { type = RhiTestType::Rendering; name = "visibility_buffer_abeautiful_game_transmission"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-abeautiful-game", sample, log)) { return RhiTestResult::fail(log); }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RhiTestResult::fail(scene.lastLoadResult().error);
        }
        if (scene.materials().size() != 15 || scene.materials()[5].transmissionFactor != 1.0f ||
            scene.materials()[7].transmissionFactor != 1.0f) { return RhiTestResult::fail("Unexpected ABeautifulGame materials"); }
        const auto whiteGlass = scene.materials()[5], blackGlass = scene.materials()[7];
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        auto result = preview.initialize(context.enableValidation, true, false);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders and ray queries"); }
        if (!result) { return RhiTestResult::fail("Preview initialization failed"); }
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
        sample.graph.setNodeRuntimeProperty(deferred, "transmissionSamples", 4);
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
        if (!renderFrame()) { return RhiTestResult::fail(log); }
        const auto defaultFlat = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
        if (!renderFrame()) { return RhiTestResult::fail(log); }
        const auto defaultBinned = preview.pixels();
        if (changedPixels(defaultFlat, defaultBinned, 2) > defaultFlat.size() / 1000) {
            save("ABeautifulGameDefaultBinningMismatch.png");
            return RhiTestResult::fail("Default deferred flat/binned output differs");
        }
        if (!save("ABeautifulGameDeferredDefault.png")) { return RhiTestResult::fail(log); }
        // The omitted property must select the same compiled variant as false.
        // Continuation settings must have no effect while that variant is off.
        sample.graph.setNodeRuntimeProperty(deferred, "supplementaryPathTracing", false);
        for (bool binned : {false, true}) {
            sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", binned);
            const auto& expected = binned ? defaultBinned : defaultFlat;
            if (!renderFrame() || preview.pixels() != expected) {
                return RhiTestResult::fail("Default supplementary tracing differs from explicit false: " + log);
            }
            for (const auto settings : {std::array<int, 2>{1, 2}, {16, 16}}) {
                sample.graph.setNodeRuntimeProperty(deferred, "transmissionSamples", settings[0]);
                sample.graph.setNodeRuntimeProperty(deferred, "transmissionDepth", settings[1]);
                if (!renderFrame() || preview.pixels() != expected) {
                    return RhiTestResult::fail("Disabled supplementary tracing used continuation settings: " + log);
                }
            }
        }
        sample.graph.setNodeRuntimeProperty(deferred, "transmissionSamples", 4);
        sample.graph.setNodeRuntimeProperty(deferred, "transmissionDepth", 8);
        // Resident realtime lighting must support the same optional continuation.
        sample.graph.setNodeRuntimeProperty(deferred, "lightingMode", "realtime");
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
        if (!renderFrame(193, 157)) { return RhiTestResult::fail(log); }
        const auto realtimeOffFlat = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
        if (!renderFrame(193, 157)) { return RhiTestResult::fail(log); }
        const auto realtimeOffBinned = preview.pixels();
        if (changedPixels(realtimeOffFlat, realtimeOffBinned, 2) > realtimeOffFlat.size() / 1000) {
            return RhiTestResult::fail("Realtime default deferred flat/binned output differs");
        }
        if (!save("ABeautifulGameRealtimeDefault.png")) { return RhiTestResult::fail(log); }
        sample.graph.setNodeRuntimeProperty(deferred, "supplementaryPathTracing", true);
        if (!renderFrame(193, 157)) { return RhiTestResult::fail(log); }
        const auto realtimeOnBinned = preview.pixels();
        if (changedPixels(realtimeOffBinned, realtimeOnBinned, 4) < 20) {
            return RhiTestResult::fail("Realtime supplementary tracing had no visible effect on glass");
        }
        if (!save("ABeautifulGameRealtimeSupplementary.png")) { return RhiTestResult::fail(log); }
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
        if (!renderFrame(193, 157)) { return RhiTestResult::fail(log); }
        const auto realtimeOnFlat = preview.pixels();
        if (changedPixels(realtimeOnFlat, realtimeOnBinned, 2) > realtimeOnFlat.size() / 1000) {
            return RhiTestResult::fail("Realtime supplementary deferred flat/binned output differs");
        }
        sample.graph.setNodeRuntimeProperty(deferred, "supplementaryPathTracing", false);
        if (!renderFrame(193, 157) || preview.pixels() != realtimeOffFlat) {
            return RhiTestResult::fail("Realtime supplementary disable did not restore direct shading: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "supplementaryPathTracing", true);
        if (!renderFrame(193, 157) || preview.pixels() != realtimeOnFlat) {
            return RhiTestResult::fail("Realtime supplementary enable did not restore glass continuation: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "exportUpscalerGuides", true);
        // Guide exports use FP16 color; allow display quantization while enabled.
        if (!renderFrame(193, 157) || changedPixels(preview.pixels(), realtimeOnFlat, 2) > realtimeOnFlat.size() / 1000) {
            return RhiTestResult::fail("Enabling realtime upscaler guides changed supplementary shading: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "exportUpscalerGuides", false);
        if (!renderFrame(193, 157) || preview.pixels() != realtimeOnFlat) {
            return RhiTestResult::fail("Disabling realtime upscaler guides did not restore supplementary shading: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "lightingMode", "reference");
        for (const char* debug : {"baseColor", "shadingNormal", "material", "final"}) {
            sample.graph.setNodeRuntimeProperty(deferred, "debugView", debug);
            sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
            if (!renderFrame()) { return RhiTestResult::fail(log); }
            const auto flat = preview.pixels();
            sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
            if (!renderFrame()) { return RhiTestResult::fail(log); }
            if (changedPixels(flat, preview.pixels(), 2) > flat.size() / 1000) {
                save("ABeautifulGameBinningMismatch.png");
                return RhiTestResult::fail(std::string(debug) + " flat/binned output differs");
            }
        }
        const auto glassImage = preview.pixels();
        if (!save("ABeautifulGameDeferred.png")) { return RhiTestResult::fail(log); }
        sample.graph.setNodeRuntimeProperty(deferred, "supplementaryPathTracing", false);
        if (!renderFrame() || preview.pixels() != defaultBinned) {
            return RhiTestResult::fail("Disabling supplementary tracing did not restore the default variant: " + log);
        }
        sample.graph.setNodeRuntimeProperty(deferred, "supplementaryPathTracing", true);
        if (!renderFrame() || preview.pixels() != glassImage) {
            return RhiTestResult::fail("Re-enabling supplementary tracing did not restore glass continuation: " + log);
        }
        if (changedPixels(defaultBinned, glassImage, 4) < 50) {
            return RhiTestResult::fail("Supplementary tracing had no visible effect on glass");
        }
        sample.graph.setNodeRuntimeProperty(deferred, "debugDisableTransmission", true);
        if (!renderFrame()) { return RhiTestResult::fail(log); }
        const size_t transmissionPixels = changedPixels(glassImage, preview.pixels(), 4);
        if (transmissionPixels < 50) { return RhiTestResult::fail("Glass continuation had no visible effect"); }
        if (!save("ABeautifulGameTransmissionDisabled.png")) { return RhiTestResult::fail(log); }
        sample.graph.setNodeRuntimeProperty(deferred, "debugDisableTransmission", false);
        if (!renderFrame() || changedPixels(glassImage, preview.pixels(), 1) != 0) {
            return RhiTestResult::fail("Transmission toggle did not restore the image: " + log);
        }
        // The asset omits attenuationDistance (infinity). Set a finite distance
        // so this test actually exercises travel inside a participating volume.
        for (uint32_t index : {5u, 7u}) {
            auto glass = scene.materials()[index];
            glass.attenuationDistance = 0.015f;
            glass.attenuationColor = float3(0.1f, 0.35f, 0.8f);
            if (!scene.setMaterialProperties(index, glass)) { return RhiTestResult::fail("Volume edit failed"); }
        }
        if (!renderFrame()) { return RhiTestResult::fail(log); }
        const auto absorbed = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "debugDisableVolumeAttenuation", true);
        if (!renderFrame()) { return RhiTestResult::fail(log); }
        const size_t absorptionPixels = changedPixels(absorbed, preview.pixels(), 4);
        if (absorptionPixels < 20) { return RhiTestResult::fail("Finite volume attenuation had no visible effect"); }
        sample.graph.setNodeRuntimeProperty(deferred, "debugDisableVolumeAttenuation", false);
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
        if (!renderFrame() || changedPixels(absorbed, preview.pixels(), 2) > absorbed.size() / 1000) {
            return RhiTestResult::fail("Edited volume differs between scheduling paths: " + log);
        }
        scene.setMaterialProperties(5, whiteGlass);
        scene.setMaterialProperties(7, blackGlass);
        // Extent changes recreate scratch allocations. Compare at an odd extent.
        if (!renderFrame(193, 157)) { return RhiTestResult::fail(log); }
        const auto resized = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
        if (!renderFrame(193, 157) || changedPixels(resized, preview.pixels(), 2) > resized.size() / 1000) {
            return RhiTestResult::fail("Resized material queues changed shading: " + log);
        }
        // Report end-to-end readback timing, not an inferred GPU speedup.
        std::ofstream timings(context.outputDirectory / "ABeautifulGameTiming.txt");
        timings << "8 environment samples; 4 transmission samples; depth 8; no accumulation\n"
            << "Median preview.render wall time, including CPU submission and readback; validation=" << context.enableValidation << '\n';
        for (bool binned : {false, true}) {
            sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", binned);
            for (const auto extent : {std::array<uint32_t, 2>{385, 257}, {1280, 720}}) {
                for (int warmup = 0; warmup < 3; ++warmup) {
                    if (!renderFrame(extent[0], extent[1])) { return RhiTestResult::fail(log); }
                }
                std::vector<double> elapsed;
                std::vector<double> gpuDeferred;
                std::vector<render::RenderGraphExecutionStats> completed;
                if (auto drained = preview.collectCompletedGpuExecutionStats(); !drained) {
                    return RhiTestResult::fail("GPU timestamp readback failed after warmup");
                }
                for (int frame = 0; frame < 12; ++frame) {
                    const auto start = std::chrono::steady_clock::now();
                    if (!renderFrame(extent[0], extent[1])) { return RhiTestResult::fail(log); }
                    elapsed.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                    if (!preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); })) {
                        return RhiTestResult::fail("GPU timestamp readback failed");
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
        // Explicitly opt in when retaining a reviewable glass continuation/reference comparison.
        if (!render::loadBuiltInRenderSample("lookdev-abeautiful-game", sample, log)) { return RhiTestResult::fail(log); }
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Deferred")->id, "supplementaryPathTracing", true);
        for (int frame = 0; frame < 32; ++frame) {
            if (!preview.render(sample.graph, 512, 384)) { return RhiTestResult::fail(preview.lastLog()); }
        }
        if (!save("ABeautifulGameComparison.png")) { return RhiTestResult::fail(log); }
        for (float split : {0.0f, 1.0f}) {
            sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", split);
            if (!preview.render(sample.graph, 512, 384) ||
                !save(split == 0.0f ? "ABeautifulGameDeferredAccumulated.png" : "ABeautifulGameReference.png")) {
                return RhiTestResult::fail(preview.lastLog() + log);
            }
        }
        return RhiTestResult::pass("15 materials: default-off/explicit false, inactive continuation settings, "
            "reference/realtime variant restoration, flat/binned shading, glass continuation (" + std::to_string(transmissionPixels) +
            " pixels), volume absorption (" + std::to_string(absorptionPixels) + " pixels), edits, resize and HDRI comparison");
    }
};

METALLIC_REGISTER_RHI_TEST(DeferredTransmissionTest);

} // namespace
} // namespace metallic::tests

#include "RhiTest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace metallic::tests {
namespace {

bool savePreview(const render::RenderGraphPreviewRenderer& preview, const std::filesystem::path& path, std::string& log)
{
    return saveRgba8Png(path, reinterpret_cast<const uint8_t*>(preview.pixels().data()),
        preview.width(), preview.height(), log);
}

class VisibilityBufferDeferredTest final : public RhiTest {
public:
    VisibilityBufferDeferredTest() { type = RhiTestType::Rendering; name = "visibility_buffer_deferred_openpbr"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RhiTestResult::fail(log); }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RhiTestResult::fail(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        // Keep normal regression runs uninstrumented; opt in only when collecting a crash dump.
        const char* aftermath = std::getenv("METALLIC_TEST_AFTERMATH");
        const bool enableAftermath = aftermath != nullptr && aftermath[0] == '1' && aftermath[1] == '\0';
        const auto initialized = preview.initialize(context.enableValidation, true, enableAftermath);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RhiTestResult::skip("Requires mesh shaders and ray queries"); }
        if (!initialized) { return RhiTestResult::fail("Preview initialization failed"); }
        preview.setEnvironment({.enabled = false});
        auto lighting = scene.lighting();
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 0;
        preview.setLighting(lighting);
        const uint32_t slider = sample.graph.findNode("Slider")->id;
        const uint32_t raster = sample.graph.findNode("VBuffer")->id;
        const uint32_t deferred = sample.graph.findNode("Deferred")->id;
        // A ray-primary path with identical direct OpenPBR lighting isolates the
        // VBuffer reconstruction from multi-bounce/environment integration differences.
        sample.graph.findNode("Reference")->type = "SceneRealtimeLightingPass";
        sample.graph.markDirty();
        for (bool orthographic : {false, true}) {
            for (const char* nodeName : {"Reference", "VBuffer"}) {
                sample.graph.setNodeRuntimeProperty(sample.graph.findNode(nodeName)->id,
                    "camera.projection", orthographic ? "orthographic" : "perspective");
                sample.graph.setNodeRuntimeProperty(sample.graph.findNode(nodeName)->id, "camera.orthoHeight", 3.0f);
            }
            for (const char* debug : {"baseColor", "shadingNormal", "final"}) {
                sample.graph.setNodeRuntimeProperty(deferred, "debugView", debug);
                sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Reference")->id, "debugView", debug);
                sample.graph.setNodeRuntimeProperty(slider, "splitPosition", 1.0f);
                auto result = preview.render(sample.graph, 193, 157);
                if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip(preview.lastLog()); }
                if (!result) { return RhiTestResult::fail(preview.lastLog()); }
                const auto reference = preview.pixels();
                sample.graph.setNodeRuntimeProperty(slider, "splitPosition", 0.0f);
                if (!preview.render(sample.graph, 193, 157)) { return RhiTestResult::fail(preview.lastLog()); }
                size_t subject = 0, different = 0;
                double error = 0.0;
                for (uint32_t y = 1; y < 156; ++y) {
                    for (uint32_t x = 1; x < 192; ++x) {
                        const size_t i = y * 193 + x;
                        if ((reference[i] & 0xffffffu) == 0 || (preview.pixels()[i] & 0xffffffu) == 0 ||
                            (reference[i - 1] & 0xffffffu) == 0 || (reference[i + 1] & 0xffffffu) == 0 ||
                            (reference[i - 193] & 0xffffffu) == 0 || (reference[i + 193] & 0xffffffu) == 0) { continue; }
                        ++subject;
                        int maximum = 0;
                        for (uint32_t shift : {0u, 8u, 16u}) {
                            int delta = std::abs(int((reference[i] >> shift) & 255u) - int((preview.pixels()[i] >> shift) & 255u));
                            maximum = std::max(maximum, delta);
                            error += delta;
                        }
                        different += maximum > 3;
                    }
                }
                if (subject < 1000 || double(different) / subject > 0.02 || error / (subject * 3) > 0.5) {
                    savePreview(preview, context.outputDirectory / "VBufferMismatch.png", log);
                    return RhiTestResult::fail(std::string(debug) + " raster/ray mismatch: pixels=" + std::to_string(subject) +
                        " outliers=" + std::to_string(different) + " mean error=" + std::to_string(error / std::max<size_t>(subject * 3, 1)));
                }
            }
        }
        // rasterInfo remains authoritative even if an unrelated camera is supplied
        // to the deferred node. Camera freezing must only affect culling.
        const auto before = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "camera.eye", {100, 200, 300});
        sample.graph.setNodeRuntimeProperty(raster, "freezeCullingCamera", true);
        if (!preview.render(sample.graph, 193, 157) || preview.pixels() != before) {
            return RhiTestResult::fail("Deferred shading did not use the actual raster camera");
        }
        lighting.lights.clear();
        preview.setLighting(lighting);
        if (!preview.render(sample.graph, 193, 157)) { return RhiTestResult::fail(preview.lastLog()); }
        for (uint32_t pixel : preview.pixels()) {
            if ((pixel & 0xffffffu) != 0u) { return RhiTestResult::fail("Deferred output retained a removed light"); }
        }
        // Reload the production sample, retaining its full path-traced reference.
        if (!render::loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RhiTestResult::fail(log); }
        preview.setEnvironment(scene.environment());
        preview.setLighting(scene.lighting());
        for (uint32_t frame = 0; frame < 256; ++frame) {
            if (!preview.render(sample.graph, 768, 768)) { return RhiTestResult::fail(preview.lastLog()); }
        }
        if (!savePreview(preview, context.outputDirectory / "LookDevVBufferComparison.png", log)) { return RhiTestResult::fail(log); }
        // Preserve both full views as review artifacts, through the same exposure chain.
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 0.0f);
        if (!preview.render(sample.graph, 768, 768) ||
            !savePreview(preview, context.outputDirectory / "LookDevVBufferDeferred.png", log)) { return RhiTestResult::fail(preview.lastLog() + log); }
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 1.0f);
        if (!preview.render(sample.graph, 768, 768) ||
            !savePreview(preview, context.outputDirectory / "LookDevVBufferReference.png", log)) { return RhiTestResult::fail(preview.lastLog() + log); }
        if (!render::setRenderSampleScenePath(sample, "Asset/meet_mat.glb", log)) { return RhiTestResult::fail(log); }
        for (const char* node : {"Reference", "VBuffer", "Deferred"}) {
            if (sample.graph.findNode(node)->properties["path"] != "Asset/meet_mat.glb") { return RhiTestResult::fail("Scene override missed a path"); }
        }
        // Exercise actual resource replacement and material lookup, not only the
        // sample's path strings. Both paths must resolve the new scene together.
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/meet_mat.glb")) {
            return RhiTestResult::fail(scene.lastLoadResult().error);
        }
        preview.setEnvironment({.enabled = false});
        sample.graph.findNode("Reference")->type = "SceneRealtimeLightingPass";
        sample.graph.markDirty();
        for (const char* node : {"Reference", "VBuffer"}) {
            sample.graph.setNodeRuntimeProperty(sample.graph.findNode(node)->id, "camera",
                {{"eye", {0.0, 0.25, 3.0}}, {"center", {0.0, 0.15, 0.0}}, {"fovDegrees", 50.0}});
        }
        for (const char* node : {"Reference", "Deferred"}) {
            sample.graph.setNodeRuntimeProperty(sample.graph.findNode(node)->id, "debugView", "baseColor");
        }
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 1.0f);
        if (!preview.render(sample.graph, 127, 95)) { return RhiTestResult::fail(preview.lastLog()); }
        const auto materialReference = preview.pixels();
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 0.0f);
        if (!preview.render(sample.graph, 127, 95)) { return RhiTestResult::fail(preview.lastLog()); }
        size_t materialPixels = 0;
        double materialError = 0.0;
        for (size_t i = 0; i < materialReference.size(); ++i) {
            if ((materialReference[i] & 0xffffffu) == 0 || (preview.pixels()[i] & 0xffffffu) == 0) { continue; }
            ++materialPixels;
            for (uint32_t shift : {0u, 8u, 16u}) {
                materialError += std::abs(int((materialReference[i] >> shift) & 255u) -
                    int((preview.pixels()[i] >> shift) & 255u));
            }
        }
        if (materialPixels < 100 || materialError / (materialPixels * 3) > 3.0) {
            return RhiTestResult::fail("Scene replacement/material remap differs from reference");
        }
        return RhiTestResult::pass("VBuffer matches ray-primary OpenPBR lighting; perspective/orthographic, camera metadata, lights and LookDev captures");
    }
};

METALLIC_REGISTER_RHI_TEST(VisibilityBufferDeferredTest);

} // namespace
} // namespace metallic::tests

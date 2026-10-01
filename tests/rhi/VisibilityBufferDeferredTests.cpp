#include "RHITest.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>

namespace metallic::tests {
namespace {

bool savePreview(const render::RenderGraphPreviewRenderer& preview, const std::filesystem::path& path, std::string& log)
{
    return saveRgba8Png(path, reinterpret_cast<const uint8_t*>(preview.pixels().data()),
        preview.width(), preview.height(), log);
}

class VisibilityBufferDeferredTest final : public RHITest {
public:
    VisibilityBufferDeferredTest() { type = RHITestType::Rendering; name = "visibility_buffer_deferred_openpbr"; }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        // Keep normal regression runs uninstrumented; opt in only when collecting a crash dump.
        const char* aftermath = std::getenv("METALLIC_TEST_AFTERMATH");
        const bool enableAftermath = aftermath != nullptr && aftermath[0] == '1' && aftermath[1] == '\0';
        const auto initialized = preview.initialize(context.enableValidation, true, enableAftermath);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and ray queries"); }
        if (!initialized) { return RHITestResult::fail("Preview initialization failed"); }
        preview.setEnvironment({.enabled = false});
        auto lighting = scene.lighting();
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 0;
        preview.setLighting(lighting);
        const uint32_t slider = sample.graph.findNode("Slider")->id;
        // These checks compare attributes against the original ray-traced mesh.
        // Keep both paths on identical geometry; adaptive cuts are tested separately.
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("VBuffer")->id, "autoLod", false);
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("VBuffer")->id, "lodLevel", 0);
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
                if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip(preview.lastLog()); }
                if (!result) { return RHITestResult::fail(preview.lastLog()); }
                const auto reference = preview.pixels();
                sample.graph.setNodeRuntimeProperty(slider, "splitPosition", 0.0f);
                if (!preview.render(sample.graph, 193, 157)) { return RHITestResult::fail(preview.lastLog()); }
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
                    return RHITestResult::fail(std::string(debug) + " raster/ray mismatch: pixels=" + std::to_string(subject) +
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
            return RHITestResult::fail("Deferred shading did not use the actual raster camera");
        }
        lighting.lights.clear();
        preview.setLighting(lighting);
        if (!preview.render(sample.graph, 193, 157)) { return RHITestResult::fail(preview.lastLog()); }
        for (uint32_t pixel : preview.pixels()) {
            if ((pixel & 0xffffffu) != 0u) { return RHITestResult::fail("Deferred output retained a removed light"); }
        }
        // Reload the production sample, retaining its full path-traced reference.
        if (!render::loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        preview.setEnvironment(scene.environment());
        preview.setLighting(scene.lighting());
        for (uint32_t frame = 0; frame < 256; ++frame) {
            if (!preview.render(sample.graph, 768, 768)) { return RHITestResult::fail(preview.lastLog()); }
        }
        if (!savePreview(preview, context.outputDirectory / "LookDevVBufferComparison.png", log)) { return RHITestResult::fail(log); }
        // Preserve both full views as review artifacts, through the same exposure chain.
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 0.0f);
        if (!preview.render(sample.graph, 768, 768) ||
            !savePreview(preview, context.outputDirectory / "LookDevVBufferDeferred.png", log)) { return RHITestResult::fail(preview.lastLog() + log); }
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 1.0f);
        if (!preview.render(sample.graph, 768, 768) ||
            !savePreview(preview, context.outputDirectory / "LookDevVBufferReference.png", log)) { return RHITestResult::fail(preview.lastLog() + log); }
        if (!render::setRenderSampleScenePath(sample, "Asset/meet_mat.glb", log)) { return RHITestResult::fail(log); }
        for (const char* node : {"Reference", "VBuffer", "Deferred"}) {
            if (sample.graph.findNode(node)->properties["path"] != "Asset/meet_mat.glb") { return RHITestResult::fail("Scene override missed a path"); }
        }
        // Exercise actual resource replacement and material lookup, not only the
        // sample's path strings. Both paths must resolve the new scene together.
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/meet_mat.glb")) {
            return RHITestResult::fail(scene.lastLoadResult().error);
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
        if (!preview.render(sample.graph, 127, 95)) { return RHITestResult::fail(preview.lastLog()); }
        const auto materialReference = preview.pixels();
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 0.0f);
        if (!preview.render(sample.graph, 127, 95)) { return RHITestResult::fail(preview.lastLog()); }
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
            return RHITestResult::fail("Scene replacement/material remap differs from reference");
        }
        return RHITestResult::pass("VBuffer matches ray-primary OpenPBR lighting; perspective/orthographic, camera metadata, lights and LookDev captures");
    }
};

METALLIC_REGISTER_RHI_TEST(VisibilityBufferDeferredTest);

class VisibilityBufferDeferredShadowHistoryTest final : public RHITest {
public:
    VisibilityBufferDeferredShadowHistoryTest()
    {
        type = RHITestType::Rendering;
        name = "visibility_buffer_deferred_shadow_history";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(scene.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        const auto initialized = preview.initialize(context.enableValidation, true, false);
        if (render::hasError(initialized, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and ray queries"); }
        if (!initialized) { return RHITestResult::fail("Preview initialization failed"); }
        preview.setEnvironment(scene.environment());
        auto lighting = scene.lighting();
        lighting.autoExposure.enabled = false;
        preview.setLighting(lighting);
        const auto raster = sample.graph.findNode("VBuffer")->id;
        const auto deferred = sample.graph.findNode("Deferred")->id;
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("Slider")->id, "splitPosition", 0.0f);
        const auto renderFrames = [&](uint32_t count) {
            for (uint32_t frame = 0; frame < count; ++frame) {
                if (!preview.render(sample.graph, 512, 512)) { return false; }
            }
            return true;
        };
        // Exercise the production graph with its HDR environment and shadows.
        // Do not override its LOD settings or discard black output pixels.
        if (!renderFrames(32) ||
            !savePreview(preview, context.outputDirectory / "LookDevShadowProduction.png", log)) {
            return RHITestResult::fail(preview.lastLog() + log);
        }
        size_t darkPixels = 0;
        constexpr size_t kSurfacePixels = (345 - 160) * (330 - 165);
        for (uint32_t y = 165; y < 330; ++y) {
            for (uint32_t x = 160; x < 345; ++x) {
                const uint32_t pixel = preview.pixels()[y * 512 + x];
                const uint32_t sum = (pixel & 255u) + ((pixel >> 8u) & 255u) + ((pixel >> 16u) & 255u);
                darkPixels += sum < 48;
            }
        }
        // This fixed camera covers the lit white ball, including its inset.
        // Allow the small legitimate crevice shadows, but reject broad acne.
        if (darkPixels > kSurfacePixels / 100) {
            return RHITestResult::fail("LookDev surface has " + std::to_string(darkPixels) +
                " near-black pixels; raster and shadow geometry must agree");
        }

        // Warm a different raster surface, then change only the producer's LOD.
        // The first restored frame must match an explicitly reset consumer.
        sample.graph.setNodeRuntimeProperty(raster, "autoLod", true);
        if (!renderFrames(32)) { return RHITestResult::fail(preview.lastLog()); }
        sample.graph.setNodeRuntimeProperty(raster, "autoLod", false);
        sample.graph.setNodeRuntimeProperty(raster, "lodLevel", 0);
        if (!renderFrames(1) ||
            !savePreview(preview, context.outputDirectory / "LookDevShadowLODSwitch.png", log)) {
            return RHITestResult::fail(preview.lastLog() + log);
        }
        const auto switched = preview.pixels();
        sample.graph.setNodeRuntimeProperty(deferred, "accumulate", false);
        if (!renderFrames(1)) { return RHITestResult::fail(preview.lastLog()); }
        sample.graph.setNodeRuntimeProperty(deferred, "accumulate", true);
        if (!renderFrames(1) ||
            !savePreview(preview, context.outputDirectory / "LookDevShadowHistoryReset.png", log)) {
            return RHITestResult::fail(preview.lastLog() + log);
        }
        if (switched != preview.pixels()) {
            return RHITestResult::fail("Changing raster LOD retained deferred lighting history");
        }
        return RHITestResult::pass("Production environment/shadows: " + std::to_string(darkPixels) +
            " near-black surface pixels; LOD switch matches a fresh accumulation frame");
    }
};

METALLIC_REGISTER_RHI_TEST(VisibilityBufferDeferredShadowHistoryTest);

class VisibilityBufferMaterialEditTest final : public RHITest {
public:
    VisibilityBufferMaterialEditTest()
    {
        type = RHITestType::Rendering;
        name = "visibility_buffer_material_edit_refresh";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) {
            return RHITestResult::fail(log);
        }
        scene::SceneDocument scene;
        if (!scene.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath) ||
            scene.materials().size() != 1) {
            return RHITestResult::fail("Expected the single-material LookDev shader ball");
        }
        auto baseline = scene.materials().front();
        baseline.baseColorFactor = float4(0.65f, 0.25f, 0.12f, 1.0f);
        baseline.metallicFactor = 0.65f;
        baseline.roughnessFactor = 0.12f;
        if (!scene.setMaterialProperties(0, baseline)) {
            return RHITestResult::fail("Could not configure the material fixture");
        }
        const uint64_t geometryRevision = scene.geometryTransformRevision();
        const uint64_t structuralRevision = scene.sceneGraph().structuralRevision();
        const uint64_t resourceIdentity = scene.resourceIdentity();

        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&scene);
        const char* aftermath = std::getenv("METALLIC_TEST_AFTERMATH");
        const bool enableAftermath = aftermath != nullptr && aftermath[0] == '1' && aftermath[1] == '\0';
        const auto initialized = preview.initialize(context.enableValidation, true, enableAftermath);
        if (render::hasError(initialized, render::Error::Unsupported)) {
            return RHITestResult::skip("Requires mesh shaders and ray queries");
        }
        if (!initialized) { return RHITestResult::fail("Preview initialization failed"); }
        preview.setEnvironment({.enabled = false});
        auto lighting = scene.lighting();
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 0;
        preview.setLighting(lighting);
        const uint32_t reference = sample.graph.findNode("Reference")->id;
        const uint32_t deferred = sample.graph.findNode("Deferred")->id;
        const uint32_t slider = sample.graph.findNode("Slider")->id;
        // These checks compare attributes against the original ray-traced mesh.
        // Keep both paths on identical geometry; adaptive cuts are tested separately.
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("VBuffer")->id, "autoLod", false);
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("VBuffer")->id, "lodLevel", 0);
        sample.graph.findNode("Reference")->type = "SceneRealtimeLightingPass";
        sample.graph.markDirty();
        sample.graph.setNodeRuntimeProperty(deferred, "accumulate", false);

        constexpr uint32_t width = 193;
        constexpr uint32_t height = 157;
        const auto changedPixels = [](const std::vector<uint32_t>& lhs, const std::vector<uint32_t>& rhs) {
            size_t changed = 0;
            for (size_t i = 0; i < lhs.size(); ++i) {
                for (uint32_t shift : {0u, 8u, 16u}) {
                    if (std::abs(int((lhs[i] >> shift) & 255u) - int((rhs[i] >> shift) & 255u)) > 3) {
                        ++changed;
                        break;
                    }
                }
            }
            return changed;
        };
        const auto renderPair = [&](const char* debug, std::vector<uint32_t>& rayImage) {
            sample.graph.setNodeRuntimeProperty(reference, "debugView", debug);
            sample.graph.setNodeRuntimeProperty(deferred, "debugView", debug);
            sample.graph.setNodeRuntimeProperty(slider, "splitPosition", 1.0f);
            if (!preview.render(sample.graph, width, height)) { log = preview.lastLog(); return false; }
            rayImage = preview.pixels();
            sample.graph.setNodeRuntimeProperty(slider, "splitPosition", 0.0f);
            if (!preview.render(sample.graph, width, height)) { log = preview.lastLog(); return false; }
            size_t subject = 0;
            size_t outliers = 0;
            double error = 0.0;
            for (uint32_t y = 1; y + 1 < height; ++y) {
                for (uint32_t x = 1; x + 1 < width; ++x) {
                    const size_t i = y * width + x;
                    if ((rayImage[i] & 0xffffffu) == 0 || (preview.pixels()[i] & 0xffffffu) == 0 ||
                        (rayImage[i - 1] & 0xffffffu) == 0 || (rayImage[i + 1] & 0xffffffu) == 0 ||
                        (rayImage[i - width] & 0xffffffu) == 0 || (rayImage[i + width] & 0xffffffu) == 0) { continue; }
                    ++subject;
                    int maximum = 0;
                    for (uint32_t shift : {0u, 8u, 16u}) {
                        const int delta = std::abs(int((rayImage[i] >> shift) & 255u) -
                            int((preview.pixels()[i] >> shift) & 255u));
                        maximum = std::max(maximum, delta);
                        error += delta;
                    }
                    outliers += maximum > 3;
                }
            }
            if (subject < 1000 || double(outliers) / subject > 0.02 || error / (subject * 3) > 0.5) {
                savePreview(preview, context.outputDirectory / "MaterialEditVBufferMismatch.png", log);
                log = std::string(debug) + " material edit differs between ray and VBuffer paths";
                return false;
            }
            return true;
        };

        // Keep the graph and scene identity unchanged: a setter revision must
        // refresh both material buffers without requiring a scene reload.
        std::vector<uint32_t> baseColorBefore, baseColorAfter, restored;
        if (!renderPair("baseColor", baseColorBefore)) { return RHITestResult::fail(log); }
        auto edited = baseline;
        edited.baseColorFactor = float4(0.08f, 0.55f, 0.3f, 1.0f);
        if (!scene.setMaterialProperties(0, edited) || !renderPair("baseColor", baseColorAfter)) {
            return RHITestResult::fail("Base color edit failed: " + log);
        }
        if (changedPixels(baseColorBefore, baseColorAfter) < 1000) {
            return RHITestResult::fail("Material base color remained stale");
        }
        if (!scene.setMaterialProperties(0, baseline) || !renderPair("baseColor", restored) || restored != baseColorBefore) {
            return RHITestResult::fail("Restoring base color did not restore both shading paths: " + log);
        }
        std::vector<uint32_t> roughnessBefore, roughnessAfter;
        if (!renderPair("final", roughnessBefore)) { return RHITestResult::fail(log); }
        edited = baseline;
        edited.roughnessFactor = 0.85f;
        if (!scene.setMaterialProperties(0, edited) || !renderPair("final", roughnessAfter)) {
            return RHITestResult::fail("Roughness edit failed: " + log);
        }
        if (changedPixels(roughnessBefore, roughnessAfter) < 100) {
            return RHITestResult::fail("Material roughness remained stale");
        }
        if (!scene.setMaterialProperties(0, baseline) || !renderPair("final", restored) || restored != roughnessBefore) {
            return RHITestResult::fail("Restoring roughness did not restore both shading paths: " + log);
        }
        if (scene.resourceIdentity() != resourceIdentity || scene.geometryTransformRevision() != geometryRevision ||
            scene.sceneGraph().structuralRevision() != structuralRevision) {
            return RHITestResult::fail("Material edits changed scene geometry identity or revision");
        }

        // Emission with no incident light gives a deterministic final color.
        // A first frame after editing must equal the next frame, and undo must
        // immediately restore the original color, with no old history mixed in.
        lighting.lights.clear();
        preview.setLighting(lighting);
        sample.graph.findNode("Reference")->type = "ScenePathTracePass";
        sample.graph.markDirty();
        for (uint32_t node : {reference, deferred}) {
            sample.graph.setNodeRuntimeProperty(node, "accumulate", true);
            sample.graph.setNodeRuntimeProperty(node, "cacheMode", "off");
        }
        sample.graph.setNodeRuntimeProperty(reference, "samples", 1);
        sample.graph.setNodeRuntimeProperty(reference, "maxDepth", 1);
        baseline.emissiveFactor = float3(0.2f, 0.04f, 0.01f);
        edited = baseline;
        edited.emissiveFactor = float3(0.01f, 0.25f, 0.07f);
        for (float split : {1.0f, 0.0f}) {
            sample.graph.setNodeRuntimeProperty(slider, "splitPosition", split);
            scene.setMaterialProperties(0, baseline);
            if (!preview.render(sample.graph, width, height)) { return RHITestResult::fail(preview.lastLog()); }
            const auto initial = preview.pixels();
            for (uint32_t frame = 0; frame < 3; ++frame) {
                if (!preview.render(sample.graph, width, height)) { return RHITestResult::fail(preview.lastLog()); }
            }
            if (!scene.setMaterialProperties(0, edited) || !preview.render(sample.graph, width, height)) {
                return RHITestResult::fail("Emissive edit failed: " + preview.lastLog());
            }
            const auto firstEdited = preview.pixels();
            if (!preview.render(sample.graph, width, height) || firstEdited != preview.pixels() ||
                changedPixels(initial, firstEdited) < 1000) {
                return RHITestResult::fail("Material edit retained accumulated radiance");
            }
            if (!scene.setMaterialProperties(0, baseline) || !preview.render(sample.graph, width, height) ||
                preview.pixels() != initial) {
                return RHITestResult::fail("Undo retained accumulated radiance from the edited material");
            }
        }
        // Start with a textured OPAQUE material so the raster path initially
        // has no resident alpha image. Switching to MASK must load the image
        // and rebuild ray geometry opacity; changing cutoff needs neither.
        const std::filesystem::path sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) /
            "Asset/LookDev/OpenPBRDefault/OpenPbrDefault.gltf";
        const std::filesystem::path fixtureDirectory = context.outputDirectory / "material-edit-alpha";
        std::error_code filesystemError;
        std::filesystem::create_directories(fixtureDirectory, filesystemError);
        if (filesystemError) { return RHITestResult::fail(filesystemError.message()); }
        std::ifstream sourceFile(sourcePath);
        auto fixture = render::RenderGraphProperties::parse(sourceFile, nullptr, false);
        if (fixture.is_discarded()) { return RHITestResult::fail("Could not read the LookDev alpha fixture source"); }
        for (const auto& buffer : fixture["buffers"]) {
            const auto uri = buffer["uri"].get<std::string>();
            std::filesystem::copy_file(sourcePath.parent_path() / uri, fixtureDirectory / uri,
                std::filesystem::copy_options::overwrite_existing, filesystemError);
            if (filesystemError) { return RHITestResult::fail(filesystemError.message()); }
        }
        const uint8_t alphaPixel[4] = {255, 255, 255, 64};
        if (!saveRgba8Png(fixtureDirectory / "MaterialEditAlpha.png", alphaPixel, 1, 1, log)) {
            return RHITestResult::fail(log);
        }
        fixture["images"] = render::RenderGraphProperties::array({{{"uri", "MaterialEditAlpha.png"}}});
        fixture["textures"] = render::RenderGraphProperties::array({{{"source", 0}}});
        fixture["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {{"index", 0}};
        const std::filesystem::path fixturePath = fixtureDirectory / "MaterialEdit.gltf";
        {
            std::ofstream fixtureFile(fixturePath, std::ios::binary);
            fixtureFile << fixture.dump(2);
            fixtureFile.flush();
            if (!fixtureFile) { return RHITestResult::fail("Could not write the LookDev alpha fixture"); }
        }
        if (!render::setRenderSampleScenePath(sample, fixturePath.generic_string(), log) || !scene.load(fixturePath)) {
            return RHITestResult::fail("Could not load the LookDev alpha fixture: " + log);
        }
        sample.graph.findNode("Reference")->type = "SceneRealtimeLightingPass";
        sample.graph.markDirty();
        std::vector<uint32_t> opaqueImage;
        if (!renderPair("baseColor", opaqueImage)) { return RHITestResult::fail(log); }
        auto maskedMaterial = scene.materials().front();
        maskedMaterial.alphaMode = "MASK";
        maskedMaterial.alphaCutoff = 0.5f;
        if (!scene.setMaterialProperties(0, maskedMaterial)) { return RHITestResult::fail("MASK edit failed"); }
        const auto expectEmptyMask = [&]() {
            for (float split : {1.0f, 0.0f}) {
                sample.graph.setNodeRuntimeProperty(slider, "splitPosition", split);
                if (!preview.render(sample.graph, width, height)) { log = preview.lastLog(); return false; }
                if (std::any_of(preview.pixels().begin(), preview.pixels().end(),
                        [](uint32_t pixel) { return (pixel & 0xffffffu) != 0; })) {
                    log = split == 1.0f ? "Ray AS/material alpha state remained stale" :
                        "VBuffer alpha texture residency/cutoff remained stale";
                    return false;
                }
            }
            return true;
        };
        if (!expectEmptyMask()) { return RHITestResult::fail(log); }
        maskedMaterial.alphaCutoff = 0.1f;
        if (!scene.setMaterialProperties(0, maskedMaterial) || !renderPair("baseColor", restored) || restored != opaqueImage) {
            return RHITestResult::fail("Lowering alpha cutoff did not restore raster/ray coverage: " + log);
        }
        maskedMaterial.alphaCutoff = 0.5f;
        if (!scene.setMaterialProperties(0, maskedMaterial) || !expectEmptyMask()) {
            return RHITestResult::fail("Restoring alpha cutoff did not restore raster/ray coverage: " + log);
        }
        maskedMaterial.alphaMode = "OPAQUE";
        if (!scene.setMaterialProperties(0, maskedMaterial) || !renderPair("baseColor", restored) || restored != opaqueImage) {
            return RHITestResult::fail("Restoring OPAQUE did not restore raster/ray coverage: " + log);
        }
        return RHITestResult::pass("Material factors, textured alpha masks and undo refresh ray/VBuffer shading and history");
    }
};

METALLIC_REGISTER_RHI_TEST(VisibilityBufferMaterialEditTest);

class VisibilityBufferSceneHandoffTest final : public RHITest {
public:
    VisibilityBufferSceneHandoffTest() { type = RHITestType::Rendering; name = "visibility_buffer_async_scene_handoff"; }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        sample.graph.removeNode(sample.graph.findNode("Reference")->id);
        sample.graph.removeNode(sample.graph.findNode("Slider")->id);
        sample.graph.addEdge("Deferred.color", "AutoExposure.source");
        const uint32_t deferred = sample.graph.findNode("Deferred")->id;
        const uint32_t raster = sample.graph.findNode("VBuffer")->id;
        // An inherited scene must never be independently resolved from this stale path.
        sample.graph.setNodeRuntimeProperty(deferred, "path", "Asset/meet_mat.glb");
        sample.graph.setNodeRuntimeProperty(deferred, "accumulate", false);
        sample.graph.setNodeRuntimeProperty(deferred, "debugView", "baseColor");
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
        // Before the async editor load finishes, the graph resolves its own scene
        // from the sample path. The editor document object already exists but is empty.
        scene::SceneDocument document;
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        auto result = preview.initialize(context.enableValidation, true, false);
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires mesh shaders and ray queries"); }
        if (!result) { return RHITestResult::fail("Preview initialization failed"); }
        preview.setEnvironment({.enabled = false});
        auto lighting = document.lighting();
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 0;
        preview.setLighting(lighting);
        const auto renderFrames = [&]() {
            for (int frame = 0; frame < 3; ++frame) {
                result = preview.render(sample.graph, 193, 157);
                if (!result) { log = std::string(render::resultToString(result)) + ": " + preview.lastLog(); return false; }
            }
            return true;
        };
        if (!renderFrames()) { return RHITestResult::fail("Path-resolved scene: " + log); }
        const auto baseline = preview.pixels();
        if (std::count_if(baseline.begin(), baseline.end(), [](uint32_t pixel) { return (pixel & 0xffffffu) != 0; }) < 1000) {
            return RHITestResult::fail("Path-resolved scene did not render the subject");
        }
        const auto scenePath = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath;
        // Keep the graph topology, scene path and extent unchanged, exactly as the
        // asynchronous editor scene handoff, without asking the caller to mark dirty.
        for (int handoff = 0; handoff < 3; ++handoff) {
            if (handoff < 2) {
                if (!document.load(scenePath)) { return RHITestResult::fail(document.lastLoadResult().error); }
            } else {
                preview.bindRuntimeScene(nullptr); // Return to the path-resolved source.
            }
            if (sample.graph.dirty()) { return RHITestResult::fail("Scene handoff must not rely on graph dirty"); }
            if (!renderFrames()) { return RHITestResult::fail("Scene handoff " + std::to_string(handoff) + ": " + log); }
            if (preview.pixels() != baseline) { return RHITestResult::fail("Scene handoff changed the rendered material"); }
        }
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", false);
        if (!renderFrames() || preview.pixels() != baseline) { return RHITestResult::fail("Flat path differs after scene handoff: " + log); }
        // Change the world to a different file without synchronizing any graph path.
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/meet_mat.glb")) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        preview.bindRuntimeScene(&document);
        if (!renderFrames()) { return RHITestResult::fail("Different-path world binding: " + log); }
        const auto meetPixels = preview.pixels();
        if (meetPixels == baseline) { return RHITestResult::fail("World switch kept rendering the fallback asset"); }
        sample.graph.setNodeRuntimeProperty(deferred, "materialBinning", true);
        if (!renderFrames() || preview.pixels() != meetPixels) {
            return RHITestResult::fail("Classification differs after different-path world switch: " + log);
        }
        // A producer explicitly bound to an asset stays independent from the world;
        // its deferred consumer still follows it despite its own meet_mat path.
        sample.graph.setNodeRuntimeProperty(raster, "sceneBinding", "asset");
        if (!renderFrames() || preview.pixels() != baseline) {
            return RHITestResult::fail("Inherited asset scene binding: " + log);
        }
        sample.graph.setNodeRuntimeProperty(raster, "sceneBinding", "world");
        if (!renderFrames() || preview.pixels() != meetPixels) {
            return RHITestResult::fail("Return to world scene binding: " + log);
        }
        return RHITestResult::pass("Automatic scene generations, stale consumer path, world switch, explicit asset inheritance and classification parity");
    }
};

METALLIC_REGISTER_RHI_TEST(VisibilityBufferSceneHandoffTest);
} // namespace
} // namespace metallic::tests

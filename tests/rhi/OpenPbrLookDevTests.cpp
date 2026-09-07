#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"

#include <cmath>

namespace metallic::tests {
namespace {

class OpenPbrLookDevTest final : public RhiTest {
public:
    OpenPbrLookDevTest()
    {
        type = RhiTestType::Rendering;
        name = "openpbr_lookdev_reference_capture";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("openpbr-lookdev", sample, message)) {
            return RhiTestResult::fail(message);
        }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath) ||
            !document.documentWarning().empty()) {
            return RhiTestResult::fail("LookDev document: " + document.documentWarning() +
                document.lastLoadResult().error);
        }
        const auto& lighting = document.lighting();
        const auto* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr || pathTrace->properties.value("bsdf", "") != "openpbr") {
            return RhiTestResult::fail("LookDev must use the OpenPBR BSDF");
        }
        if (lighting.autoExposure.enabled || lighting.exposureEV100 != 0.0f ||
            lighting.lights.size() != 1 || !document.environment().enabled ||
            !std::filesystem::is_regular_file(document.environment().path) ||
            document.cameras().empty() || document.cameras()[0].fallback) {
            return RhiTestResult::fail("LookDev must load a fixed camera, manual exposure, HDRI and split sun");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        if (!preview.setLighting(lighting)) { return RhiTestResult::fail("Invalid LookDev lighting"); }
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) {
            return RhiTestResult::skip("LookDev capture requires ray queries and bindless descriptors");
        }
        if (!initialized) { return RhiTestResult::fail("LookDev renderer initialization failed"); }
        // Deterministic 1024 spp capture, using the same sample graph and scene as the editor.
        constexpr uint32_t kSize = 768;
        constexpr uint32_t kFrames = 256;
        for (uint32_t frame = 0; frame < kFrames; ++frame) {
            if (!preview.render(sample.graph, kSize, kSize, sample.desc.previewOutput)) {
                return RhiTestResult::fail(preview.lastLog());
            }
        }
        const auto output = context.outputDirectory / "OpenPbrDefault.png";
        if (!saveRgba8Png(output, reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                kSize, kSize, message)) {
            return RhiTestResult::fail(message);
        }
        // The shaderball fills the center; reject missing geometry or a clipped/black subject.
        double subjectLuminance = 0.0;
        uint32_t clipped = 0;
        for (uint32_t y = 250; y < 520; ++y) {
            for (uint32_t x = 250; x < 520; ++x) {
                const uint32_t pixel = preview.pixels()[y * kSize + x];
                const double r = pixel & 255u;
                const double g = (pixel >> 8u) & 255u;
                const double b = (pixel >> 16u) & 255u;
                subjectLuminance += (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0;
                clipped += r == 255.0 && g == 255.0 && b == 255.0;
            }
        }
        subjectLuminance /= 270.0 * 270.0;
        if (subjectLuminance < 0.15 || subjectLuminance > 0.95 || clipped > 7290) {
            return RhiTestResult::fail("LookDev subject is black or overexposed: " + std::to_string(subjectLuminance));
        }
        // Changing only the light sources must make this non-emissive scene black.
        auto darkEnvironment = document.environment();
        darkEnvironment.intensity = 0.0f;
        auto darkLighting = lighting;
        darkLighting.lights.clear();
        preview.setEnvironment(darkEnvironment);
        preview.setLighting(darkLighting);
        sample.graph.setNodeRuntimeProperty(sample.graph.findNode("PathTrace")->id, "accumulate", false);
        if (!preview.render(sample.graph, 32, 32, sample.desc.previewOutput)) {
            return RhiTestResult::fail(preview.lastLog());
        }
        for (uint32_t pixel : preview.pixels()) {
            if ((pixel & 0x00ffffffu) != 0) {
                return RhiTestResult::fail("LookDev contains unexpected emission or retained exposure/history");
            }
        }
        return RhiTestResult::pass("1024 spp reference capture: " + output.string());
    }
};

METALLIC_REGISTER_RHI_TEST(OpenPbrLookDevTest);
} // namespace
} // namespace metallic::tests

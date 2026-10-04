#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"

#include <cmath>
#include <fstream>
#include "Runtime/Render/Core/ColorSpace.h"

namespace metallic::tests {
namespace {

class OpenPBRLookDevTest final : public RHITest {
public:
    OpenPBRLookDevTest()
    {
        type = RHITestType::Rendering;
        name = "openpbr_lookdev_reference_capture";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string message;
        if (!render::loadBuiltInRenderSample("openpbr-lookdev", sample, message)) {
            return RHITestResult::fail(message);
        }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath) ||
            !document.documentWarning().empty()) {
            return RHITestResult::fail("LookDev document: " + document.documentWarning() +
                document.lastLoadResult().error);
        }
        const auto& lighting = document.lighting();
        const auto* pathTrace = sample.graph.findNode("PathTrace");
        if (pathTrace == nullptr || pathTrace->properties.value("bsdf", "") != "openpbr") {
            return RHITestResult::fail("LookDev must use the OpenPBR BSDF");
        }
        if (lighting.autoExposure.enabled || lighting.exposureEV100 != 0.0f ||
            lighting.lights.size() != 1 || !document.environment().enabled ||
            !std::filesystem::is_regular_file(document.environment().path) ||
            document.cameras().empty() || document.cameras()[0].fallback) {
            return RHITestResult::fail("LookDev must load a fixed camera, manual exposure, HDRI and split sun");
        }

        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        if (!preview.setLighting(lighting)) { return RHITestResult::fail("Invalid LookDev lighting"); }
        const auto initialized = preview.initialize(context.enableValidation, true);
        if (render::hasError(initialized, render::Error::Unsupported)) {
            return RHITestResult::skip("LookDev capture requires ray queries and bindless descriptors");
        }
        if (!initialized) { return RHITestResult::fail("LookDev renderer initialization failed"); }
        // Deterministic 1024 spp capture, using the same sample graph and scene as the editor.
        constexpr uint32_t kSize = 768;
        constexpr uint32_t kFrames = 256;
        for (uint32_t frame = 0; frame < kFrames; ++frame) {
            if (!preview.render(sample.graph, kSize, kSize, sample.desc.previewOutput)) {
                return RHITestResult::fail(preview.lastLog());
            }
        }
        const auto output = context.outputDirectory / "OpenPBRDefault.png";
        if (!saveRgba8Png(output, reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                kSize, kSize, message)) {
            return RHITestResult::fail(message);
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
            return RHITestResult::fail("LookDev subject is black or overexposed: " + std::to_string(subjectLuminance));
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
            return RHITestResult::fail(preview.lastLog());
        }
        for (uint32_t pixel : preview.pixels()) {
            if ((pixel & 0x00ffffffu) != 0) {
                return RHITestResult::fail("LookDev contains unexpected emission or retained exposure/history");
            }
        }
        return RHITestResult::pass("1024 spp reference capture: " + output.string());
    }
};

METALLIC_REGISTER_RHI_TEST(OpenPBRLookDevTest);

class ColorGradingLookDevTest final : public RHITest {
public:
    ColorGradingLookDevTest() { type = RHITestType::Rendering; name = "color_grading_lookdev_capture"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("openpbr-lookdev", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) {
            return RHITestResult::fail(document.lastLoadResult().error);
        }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        if (!preview.setLighting(document.lighting())) { return RHITestResult::fail("Invalid LookDev lighting"); }
        auto result = preview.initialize(context.enableValidation, true);
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Requires ray queries and bindless"); }
        if (!result) { return RHITestResult::fail("Preview initialization failed"); }
        const auto display = sample.graph.findNode("ColorGrading")->id;
        for (const char* transform : {"unreal", "aces2"}) {
            sample.graph.setNodeProperties(display, {{"toneCurve", transform}});
            for (uint32_t frame = 0; frame < 64; ++frame) {
                if (!preview.render(sample.graph, 384, 384, "FinalBlit.color")) {
                    return RHITestResult::fail(preview.lastLog());
                }
            }
            if (!saveRgba8Png(context.outputDirectory / (std::string("LookDev-") + transform + ".png"),
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), 384, 384, log)) {
                return RHITestResult::fail(log);
            }
            uint64_t sum = 0;
            for (uint32_t pixel : preview.pixels()) {
                sum += (pixel & 255) + ((pixel >> 8) & 255) + ((pixel >> 16) & 255);
            }
            const double average = double(sum) / (384 * 384 * 3 * 255);
            if (average < 0.02 || average > 0.98) { return RHITestResult::fail("LookDev capture is black or clipped"); }
        }
        return RHITestResult::pass("Captured UE Film and default ACES2 through the graph LUT at 256 spp with fixed LookDev lighting");
    }
};
METALLIC_REGISTER_RHI_TEST(ColorGradingLookDevTest);

class WorkingColorLinearCaptureTest final : public RHITest {
public:
    WorkingColorLinearCaptureTest() { type = RHITestType::Rendering; name = "working_color_lookdev_linear_capture"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("openpbr-lookdev", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) { return RHITestResult::fail(document.lastLoadResult().error); }
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        if (!preview.setLighting(document.lighting())) { return RHITestResult::fail("Invalid capture lighting"); }
        if (!preview.initialize(context.enableValidation, true)) { return RHITestResult::fail("Initialize linear capture"); }
        preview.setRawReadbackEnabled(true);
        sample.graph.findNode("PathTrace")->properties["samples"] = 4;
        sample.graph.findNode("PathTrace")->properties["accumulate"] = true;
        constexpr uint32_t size = 384, frames = 64;
        for (uint32_t frame = 0; frame < frames; ++frame) {
            if (!preview.render(sample.graph, size, size, "PathTrace.color", frame + 1 == frames)) { return RHITestResult::fail(preview.lastLog()); }
        }
        const auto& bytes = preview.readbackBytes();
        if (preview.readbackFormat() != render::Format::RGBA32Sfloat || bytes.size() != size*size*16) { return RHITestResult::fail("Linear capture must retain raw RGBA32F"); }
        float maximum = 0;
        const auto* values = reinterpret_cast<const float*>(bytes.data());
        for (uint32_t pixel = 0; pixel < size*size; ++pixel) {
            for (uint32_t channel = 0; channel < 3; ++channel) {
                const float value = values[pixel*4 + channel];
                if (!std::isfinite(value)) { return RHITestResult::fail("Non-finite scene radiance"); }
                maximum = std::max(maximum, value);
            }
        }
        if (maximum <= 1) { return RHITestResult::fail("Scene capture lost HDR values"); }
        const char* mode = render::sceneWorkingColorSpace() == render::SceneWorkingColorSpace::ACEScg ? "acescg" : "lin_rec709";
        const auto path = context.outputDirectory / (std::string("OpenPBRDefault-") + mode + ".rgba32f");
        std::ofstream raw(path, std::ios::binary);
        raw.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        if (!raw) { return RHITestResult::fail("Write raw scene reference"); }
        nlohmann::json metadata{{"sourceColorSpace", "lin_rec709"}, {"sceneWorkingColorSpace", mode}, {"format", "RGBA32F little-endian"},
            {"width", size}, {"height", size}, {"frames", frames}, {"samplesPerPixel", frames*4},
            {"output", "PathTrace.color"}, {"exposureApplied", false}, {"displayTransformApplied", false},
            {"maximum", maximum}, {"sourceReference", "Asset/LookDev/OpenPbrDefault/Reference.json"}};
        std::ofstream manifest(path.string() + ".json");
        manifest << metadata.dump(2) << '\n';
        return manifest ? RHITestResult::pass(path.string()) : RHITestResult::fail("Write capture metadata");
    }
};
METALLIC_REGISTER_RHI_TEST(WorkingColorLinearCaptureTest);

} // namespace
} // namespace metallic::tests

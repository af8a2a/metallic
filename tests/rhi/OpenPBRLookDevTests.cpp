#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutor.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"

#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

class OpenPBRTypedGuidesTest final : public RHITest {
public:
    OpenPBRTypedGuidesTest() { type = RHITestType::Rendering; name = "openpbr_typed_guides_outputs"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("openpbr-lookdev", sample, log)) { return RHITestResult::fail(log); }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) { return RHITestResult::fail("Load LookDev"); }
        auto* pass = sample.graph.findNode("PathTrace");
        pass->properties["exportDenoiserGuides"] = true;
        pass->properties["samples"] = 1;
        pass->properties["temporalJitter"] = false;
        sample.graph.markDirty();
        render::RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        if (!preview.setLighting(document.lighting())) { return RHITestResult::fail("LookDev lighting"); }
        auto initialized = preview.initialize(context.enableValidation, true);
        if (!initialized) { return RHITestResult::fail(toString(initialized)); }
        preview.setRawReadbackEnabled(true);
        for (uint32_t width : {31u, 63u}) {
            for (const char* field : {"color", "albedo", "specularAlbedo", "normalRoughness", "motionVectors", "linearDepth", "specularHitDistance", "depth"}) {
                const std::string output = std::string("PathTrace.") + field;
                if (!preview.render(sample.graph, width, 39, output)) { return RHITestResult::fail(preview.lastLog()); }
                if (preview.lastLog().find("Initial material compilation failed") != std::string::npos) {
                    return RHITestResult::fail(preview.lastLog());
                }
                const auto format = preview.readbackFormat();
                const bool half = format == render::Format::RGBA16Sfloat || format == render::Format::RG16Sfloat;
                const uint32_t channels = format == render::Format::RGBA16Sfloat ? 4 : format == render::Format::RG16Sfloat ? 2 : 1;
                const auto& bytes = preview.readbackBytes();
                if (bytes.size() != width * 39u * channels * (half ? 2u : 4u)) { return RHITestResult::fail(output + " size"); }
                for (size_t pixel = 0; pixel < width * 39u; ++pixel) {
                    float values[4]{};
                    for (uint32_t channel = 0; channel < channels; ++channel) {
                        const size_t index = pixel * channels + channel;
                        if (half) {
                            uint16_t word; std::memcpy(&word, bytes.data() + index * 2, 2);
                            const uint32_t exponent = (word >> 10) & 31, mantissa = word & 1023;
                            values[channel] = exponent == 0 ? std::ldexp(float(mantissa), -24) :
                                exponent == 31 ? INFINITY : std::ldexp(float(mantissa + 1024), int(exponent) - 25);
                            if (word & 0x8000) { values[channel] = -values[channel]; }
                        } else { std::memcpy(&values[channel], bytes.data() + index * 4, 4); }
                        if (!std::isfinite(values[channel])) { return RHITestResult::fail(output + " nonfinite"); }
                        if ((std::string_view(field) == "albedo" || std::string_view(field) == "specularAlbedo" || std::string_view(field) == "depth") &&
                            (values[channel] < 0 || values[channel] > 1)) { return RHITestResult::fail(output + " range"); }
                    }
                    if (std::string_view(field) == "normalRoughness") {
                        const float lengthSquared = values[0]*values[0] + values[1]*values[1] + values[2]*values[2];
                        if (std::abs(lengthSquared - 1) > 0.003f || values[3] < 0 || values[3] > 1) { return RHITestResult::fail(output + " invalid normal/roughness"); }
                    }
                    if (std::string_view(field) == "motionVectors" && (std::abs(values[0]) > 0.01f || std::abs(values[1]) > 0.01f)) {
                        return RHITestResult::fail(output + " stationary camera motion");
                    }
                }
            }
        }
        return RHITestResult::pass("Eight typed outputs, finite HDR, normalized normals, bounded guides and resize/history");
    }
};
METALLIC_REGISTER_RHI_TEST(OpenPBRTypedGuidesTest);

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
} // namespace
} // namespace metallic::tests

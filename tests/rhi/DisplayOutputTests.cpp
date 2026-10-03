#include "RHITest.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSurfaceFormat.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"
#include "Runtime/Render/Core/ColorGrading.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

class DisplaySurfaceFormatTest final : public RHITest {
public:
    DisplaySurfaceFormatTest() { name = "hdr_surface_format_negotiation"; }
    RHITestResult run(RHITestContext&) override
    {
        using render::DisplayOutputMode;
        const VkSurfaceFormatKHR sdr{VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        const VkSurfaceFormatKHR hdr{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT};
        const VkSurfaceFormatKHR wrongPair{VK_FORMAT_R16G16B16A16_SFLOAT, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        const VkSurfaceFormatKHR pq{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_HDR10_ST2084_EXT};
        VkSurfaceFormatKHR selected{};
        DisplayOutputMode actual{};
        auto choose = [&](std::span<const VkSurfaceFormatKHR> formats, DisplayOutputMode mode, bool fallback) {
            return render::vulkan::selectSurfaceFormat(formats, sdr.format, mode, fallback, selected, actual);
        };
        const std::array full{pq, wrongPair, sdr, hdr};
        if (!choose(full, DisplayOutputMode::HDR10_PQ, false) || actual != DisplayOutputMode::HDR10_PQ ||
            selected.format != pq.format || selected.colorSpace != pq.colorSpace) {
            return RHITestResult::fail("HDR10 must select the exact RGB10A2/BT.2020 PQ pair");
        }
        const std::array noPq{wrongPair, sdr, hdr};
        if (choose(noPq, DisplayOutputMode::HDR10_PQ, false) ||
            !choose(noPq, DisplayOutputMode::HDR10_PQ, true) || actual != DisplayOutputMode::SDR) {
            return RHITestResult::fail("HDR10 must reject missing pairs or explicitly fall back to SDR");
        }
        if (!choose(full, DisplayOutputMode::HDRscRGB, false) || actual != DisplayOutputMode::HDRscRGB ||
            selected.format != hdr.format || selected.colorSpace != hdr.colorSpace) {
            return RHITestResult::fail("scRGB must select the exact FP16/extended-linear pair");
        }
        const std::array fallback{pq, wrongPair, sdr};
        if (choose(fallback, DisplayOutputMode::HDRscRGB, false) ||
            !choose(fallback, DisplayOutputMode::HDRscRGB, true) || actual != DisplayOutputMode::SDR ||
            selected.format != sdr.format || selected.colorSpace != sdr.colorSpace) {
            return RHITestResult::fail("Strict HDR / safe SDR fallback contract failed");
        }
        if (!choose(full, DisplayOutputMode::SDR, true) || actual != DisplayOutputMode::SDR ||
            choose(std::span(&pq, 1), DisplayOutputMode::SDR, true) ||
            choose({}, DisplayOutputMode::HDRscRGB, true)) {
            return RHITestResult::fail("SDR request accepted an unsupported/unknown encoding");
        }
        const VkSurfaceFormatKHR anySdr{VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
        if (choose(std::span(&anySdr, 1), DisplayOutputMode::HDRscRGB, false) ||
            !choose(std::span(&anySdr, 1), DisplayOutputMode::HDRscRGB, true) || selected.format != sdr.format) {
            return RHITestResult::fail("UNDEFINED format must still respect the advertised color space");
        }
        for (const auto mode : {DisplayOutputMode::SDR_sRGB, DisplayOutputMode::HDR_scRGB, DisplayOutputMode::HDR10_PQ}) {
            render::RenderGraphCompileContext compile;
            compile.width = compile.height = 64;
            compile.displayOutput.mode = mode;
            if (!compile.displayOutput.valid()) { return RHITestResult::fail("Invalid formal display profile"); }
            for (const char* passName : {"ScenePathTracePass", "SceneRealtimeLightingPass", "SceneRTXDIPass",
                    "RTXDICompositePass", "AutoExposurePass"}) {
                auto pass = render::createRenderGraphPass(passName);
                pass->setProperties({{"outputLinear", false}});
                const auto reflection = pass->reflect(compile);
                const auto* color = reflection.findField("color", render::RenderGraphFieldVisibility::Output);
                if (!color || (color->format != render::Format::RGBA16Sfloat && color->format != render::Format::RGBA32Sfloat) ||
                    (color->colorEncoding != render::DisplayColorEncoding::SceneLinear &&
                        color->colorEncoding != render::DisplayColorEncoding::ExposedLinear)) {
                    return RHITestResult::fail(std::string(passName) + " changed its linear-HDR contract with the output profile");
                }
            }
        }
        return RHITestResult::pass();
    }
};

class DisplayReadbackPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("color").transferRead().format = render::Format::Unknown;
        reflection.addBufferOutput("pixels").buffer(uint64_t(context.width) * context.height * 8).transferWrite();
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const auto source = context.inputTexture("color");
        const uint32_t bytes = source.desc().format == render::Format::RGBA16Sfloat ? 8 : 4;
        if (auto commandResult = (context.outputBuffer("pixels").buffer())->slice().and_then([&](const auto& bufferSlice) { return context.commandBuffer().copyTextureToBuffer({.texture = source.texture(),
            .buffer = bufferSlice, .bufferRowPitch = context.width() * bytes,
            .bufferSlicePitch = context.width() * context.height() * bytes,
            .width = context.width(), .height = context.height()}); }); !commandResult) { return commandResult; }
        return {};
    }
};

class DisplaySourcePass final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").format = render::Format::RGBA32Sfloat;
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const float level = context.properties().value("level", 1.0f);
        const auto rgb = context.properties().value("rgb", std::array<float, 3>{level, level, level});
        const render::RenderingAttachmentDesc attachment{.view = context.outputTexture("color").view(),
            .layout = render::TextureLayout::ColorAttachment, .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store, .clearColor = {rgb[0], rgb[1], rgb[2], 1.0f}};
        if (auto commandResult = context.commandBuffer().beginRendering({
            .renderArea = {0, 0, context.width(), context.height()},
            .colorAttachments = {&attachment, 1},
        }); !commandResult) { return commandResult; }
        context.commandBuffer().endRendering();
        return {};
    }
};

float halfToFloat(uint16_t value)
{
    const uint32_t exponent = (value >> 10) & 31;
    const uint32_t mantissa = value & 1023;
    const float result = exponent == 0 ? std::ldexp(float(mantissa), -24) :
        (exponent == 31 ? INFINITY : std::ldexp(float(mantissa + 1024), int(exponent) - 25));
    return (value & 0x8000) ? -result : result;
}

class DisplayOutputGPUTest final : public RHITest {
public:
    DisplayOutputGPUTest() { type = RHITestType::Rendering; name = "hdr_display_output_pixels"; }
    RHITestResult run(RHITestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        const auto result = render::createDevice({.applicationName = "HDR output GPU test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (render::hasError(result, render::Error::Unsupported)) { return RHITestResult::skip("Bindless device unavailable"); }
        if (!result) { return RHITestResult::fail(toString(result)); }
        render::registerRenderGraphPassType("DisplayReadback", "HDR readback", [] { return std::make_unique<DisplayReadbackPass>(); });
        render::registerRenderGraphPassType("DisplaySource", "HDR source", [] { return std::make_unique<DisplaySourcePass>(); });
        render::RenderGraph graph;
        graph.addNode("FinalBlitPass", "FinalBlit", {{"calibrationPattern", true}});
        graph.addNode("DisplayReadback", "Readback");
        graph.addEdge("FinalBlit.color", "Readback.color");
        graph.markOutput("Readback.pixels");
        render::RenderGraphExecutor executor;
        render::RenderWorld world;
        scene::LightingSettings lighting;
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 0.0f;
        world.setLighting(lighting);
        executor.bindRenderWorld(&world);
        render::RenderGraphCompileOptions options;
        options.displayOutput.mode = render::DisplayOutputMode::HDRscRGB;
        std::string log;
        std::array<uint16_t, 8 * 8 * 4> pixels{};
        auto frame = [&] {
            if (!executor.compile(*device, graph, 8, 8, options, log) ||
                !executor.execute({.graphicsQueue = device->getQueue(render::QueueType::Graphics)}) ||
                !executor.waitForSubmittedWork()) { return false; }
            auto* readback = executor.outputResource("Readback.pixels")->buffer;
            readback->invalidate();
            auto* mapped = readback->map();
            if (!mapped) { return false; }
            std::memcpy(pixels.data(), mapped, sizeof(pixels));
            readback->unmap();
            return true;
        };
        auto near = [](float a, float b) { return std::isfinite(a) && std::abs(a - b) < 0.015f; };
        if (!frame()) { return RHITestResult::fail("HDR calibration execution: " + log); }
        const float expected[] = {1.0f, 203.0f / 80.0f, 5.0f, 12.5f};
        for (uint32_t x = 0; x < 8; ++x) {
            for (uint32_t c = 0; c < 4; ++c) {
                if (!near(halfToFloat(pixels[x * 4 + c]), c == 3 ? 1.0f : expected[x / 2])) {
                    return RHITestResult::fail("FP16 calibration nits/opaque alpha were clipped or gamma-encoded");
                }
            }
        }
        // Check the neutral gradient and its RGB ramps retain values above 1.
        if (!near(halfToFloat(pixels[(4 * 8 + 7) * 4]), 937.5f / 80.0f) ||
            !near(halfToFloat(pixels[(5 * 8 + 7) * 4 + 1]), 0.0f)) {
            return RHITestResult::fail("HDR calibration gradient/channel isolation failed");
        }
        graph.setNodeProperties(graph.findNode("FinalBlit")->id, {});
        const auto sourceId = graph.addNode("DisplaySource", "Source", {{"level", 1.0f}})->id;
        graph.addNode("AutoExposurePass", "Exposure");
        graph.addEdge("Source.color", "Exposure.source");
        graph.addEdge("Exposure.color", "FinalBlit.source");
        if (!frame() || !near(halfToFloat(pixels[0]), 203.0f / 80.0f) ||
            executor.outputResource("Exposure.color")->desc.format != render::Format::RGBA16Sfloat) {
            return RHITestResult::fail("AutoExposure -> FinalBlit lost scene-linear paper white: " + log);
        }
        graph.setNodeProperties(sourceId, {{"level", 16.0f}});
        if (!frame() || halfToFloat(pixels[0]) <= 5.0f || halfToFloat(pixels[0]) > 12.5f) {
            return RHITestResult::fail("Scene highlights did not survive exposure and peak mapping");
        }
        if (!executor.reloadShaders(log) || !frame() || halfToFloat(pixels[0]) <= 5.0f) {
            return RHITestResult::fail("Shader reload lost HDR display context: " + log);
        }
        graph.setNodeProperties(sourceId, {{"level", 1.0f}});
        options.displayOutput.paperWhiteNits = 400.0f;
        if (!frame() || !near(halfToFloat(pixels[0]), 5.0f)) {
            return RHITestResult::fail("Changing display parameters reused a stale output transform");
        }
        options.displayOutput.mode = render::DisplayOutputMode::SDR;
        if (!frame() || executor.outputResource("FinalBlit.color")->desc.format != render::Format::RGBA8Unorm ||
            executor.outputResource("Exposure.color")->desc.format != render::Format::RGBA16Sfloat) {
            return RHITestResult::fail("SDR output changed the internal linear-HDR contract");
        }
        const auto* sdr = reinterpret_cast<const uint8_t*>(pixels.data());
        if (std::abs(int(sdr[0]) - 188) > 1 || sdr[3] != 255) {
            return RHITestResult::fail("SDR fallback must apply Reinhard and exact sRGB once");
        }
        options.displayOutput.mode = render::DisplayOutputMode::HDRscRGB;
        if (!frame() || !near(halfToFloat(pixels[0]), 5.0f)) {
            return RHITestResult::fail("SDR -> HDR transition failed");
        }
        options.displayOutput.mode = render::DisplayOutputMode::HDR10_PQ;
        if (!frame() || !near(halfToFloat(pixels[0]), 5.0f) ||
            executor.outputResource("Exposure.color")->desc.format != render::Format::RGBA16Sfloat) {
            return RHITestResult::fail("HDR10 must retain the same linear composition image as scRGB");
        }
        return RHITestResult::pass("Verified FP16 nits, gradients, exposure, highlight shoulder, reload and HDR/SDR switching");
    }
};

class ColorGradingGPUTest final : public RHITest {
public:
    ColorGradingGPUTest() { type = RHITestType::Rendering; name = "color_grading_unreal_lut_aces2"; }
    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        std::atomic_uint validationErrors{};
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Color grading reference tests",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
            .validationSink = {.callback = [](void* data, const ValidationMessage& message) noexcept {
                if (message.messageIdName && std::strstr(message.messageIdName, "VUID-")) {
                    ++*static_cast<std::atomic_uint*>(data);
                }
            }, .context = &validationErrors}, .enableAsyncCompute = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Bindless device unavailable"); }
        if (!result) { return RHITestResult::fail(resultToString(result)); }
        registerRenderGraphPassType("DisplayReadback", "HDR readback", [] { return std::make_unique<DisplayReadbackPass>(); });
        registerRenderGraphPassType("DisplaySource", "HDR source", [] { return std::make_unique<DisplaySourcePass>(); });
        RenderGraph graph;
        const auto source = graph.addNode("DisplaySource", "Source", {{"level", 0.18f}})->id;
        const auto display = graph.addNode("ColorGradingLUTPass", "Grading",
            {{"toneCurve", "unreal"}, {"inputEncoding", "linear"}, {"AsyncComputePreferred", true}})->id;
        graph.addNode("FinalBlitPass", "Display", {{"inputEncoding", "linear"}});
        graph.addEdge("Grading.lut", "Display.lut");
        graph.addNode("DisplayReadback", "Readback");
        graph.addEdge("Source.color", "Display.source");
        graph.addEdge("Display.color", "Readback.color");
        graph.markOutput("Readback.pixels");
        RenderGraphExecutor executor;
        RenderGraphCompileOptions options;
        std::string log;
        std::array<uint16_t, 8 * 8 * 4> pixels{};
        const auto frame = [&](bool recompile = true) {
            log.clear();
            if ((recompile && !executor.compile(*device, graph, 8, 8, options, log)) ||
                !executor.execute({.graphicsQueue = device->getQueue(QueueType::Graphics),
                    .computeQueue = device->getQueue(QueueType::Compute)}) ||
                !executor.waitForSubmittedWork()) { return false; }
            auto* buffer = executor.outputResource("Readback.pixels")->buffer;
            buffer->invalidate(); const void* data = buffer->map();
            if (!data) { return false; }
            std::memcpy(pixels.data(), data, sizeof(pixels)); buffer->unmap(); return true;
        };
        const auto channel = [&] { return int(reinterpret_cast<const uint8_t*>(pixels.data())[0]); };
        if (!frame()) { return RHITestResult::fail("UE SDR compile/execute: " + log); }
        const auto* cube = executor.outputResource("Grading.lut");
        if (!cube || cube->desc.type != TextureType::Texture3D || cube->desc.depth != 64 ||
            cube->desc.width != 64 || cube->desc.height != 64 || cube->desc.format != Format::RGBA16Sfloat) {
            return RHITestResult::fail("Grading did not allocate a native 64-cubed FP16 graph texture");
        }
        const int neutral = channel();
        // UE Film's InMatch=OutMatch=0.18 anchor; exact sRGB gives 118/255.
        if (std::abs(neutral - 118) > 2) { return RHITestResult::fail("UE 18% gray Film anchor mismatch"); }
        std::vector<uint8_t> lut(256 * 16 * 4);
        const auto identityPath = context.outputDirectory / "UEIdentityLUT.png";
        const auto inversePath = context.outputDirectory / "UEInverseLUT.png";
        for (uint32_t b = 0; b < 16; ++b) { for (uint32_t g = 0; g < 16; ++g) { for (uint32_t r = 0; r < 16; ++r) {
            const auto i = (g * 256 + b * 16 + r) * 4;
            lut[i] = uint8_t(r * 17); lut[i + 1] = uint8_t(g * 17); lut[i + 2] = uint8_t(b * 17); lut[i + 3] = 255;
        } } }
        if (!saveRgba8Png(identityPath, lut.data(), 256, 16, log)) { return RHITestResult::fail(log); }
        for (size_t i = 0; i < lut.size(); ++i) { if (i % 4 != 3) { lut[i] = 255 - lut[i]; } }
        if (!saveRgba8Png(inversePath, lut.data(), 256, 16, log)) { return RHITestResult::fail(log); }
        RenderGraphProperties properties{{"toneCurve", "unreal"}, {"inputEncoding", "linear"},
            {"AsyncComputePreferred", true}, {"lut1", std::filesystem::absolute(identityPath).string()}};
        graph.setNodeProperties(display, properties);
        if (!frame() || std::abs(channel() - neutral) > 1) { return RHITestResult::fail("Neutral custom LUT changed UE Film output: " + log); }
        properties["lut1"] = std::filesystem::absolute(inversePath).string();
        graph.setNodeProperties(display, properties);
        if (!frame() || std::abs(channel() - (255 - neutral)) > 2) { return RHITestResult::fail("UE unwrapped LUT sampling/transfer mismatch: " + log); }
        graph.setNodeRuntimeProperty(display, "lut1Weight", 0.0f);
        if (!executor.syncRuntimeProperties(graph) || !frame(false) || std::abs(channel() - neutral) > 1) {
            return RHITestResult::fail("Live LUT weight did not update without recompiling the graph");
        }
        graph.setNodeRuntimeProperty(display, "lut1Weight", 1.0f);
        if (!executor.syncRuntimeProperties(graph) || !frame(false) || std::abs(channel() - (255 - neutral)) > 2) {
            return RHITestResult::fail("Live LUT weight restoration failed");
        }
        graph.setNodeRuntimeProperties(display, RenderGraphProperties::object());
        properties["lut1Weight"] = 0.0f; graph.setNodeProperties(display, properties);
        if (!frame() || std::abs(channel() - neutral) > 1) { return RHITestResult::fail("Zero LUT weight changed output"); }
        properties["lut1Weight"] = 1.0f;
        properties["lut2"] = std::filesystem::absolute(identityPath).string(); properties["lut2Weight"] = 1.0f;
        graph.setNodeProperties(display, properties);
        if (!frame() || std::abs(channel() - 128) > 1) { return RHITestResult::fail("LUT blend weights were not normalized"); }
        properties.erase("lut2"); properties.erase("lut2Weight");
        for (const auto rgb : {std::array<float,3>{0.5f,0.05f,0.01f}, {0.01f,0.5f,0.05f}, {0.05f,0.01f,0.5f}}) {
            graph.setNodeProperties(source, {{"rgb", rgb}});
            properties["lut1Weight"] = 0.0f;
            graph.setNodeProperties(display, properties);
            if (!frame()) { return RHITestResult::fail(log); }
            const auto baseline = pixels;
            properties["lut1Weight"] = 1.0f;
            graph.setNodeProperties(display, properties);
            if (!frame()) { return RHITestResult::fail(log); }
            for (size_t c = 0; c < 3; ++c) {
                if (std::abs(int(reinterpret_cast<const uint8_t*>(pixels.data())[c]) +
                    int(reinterpret_cast<const uint8_t*>(baseline.data())[c]) - 255) > 2) {
                    return RHITestResult::fail("Custom LUT RGB axes / blue-slice interpolation mismatch");
                }
            }
        }
        for (const char* transform : {"unreal", "aces2"}) {
            properties["toneCurve"] = transform;
            for (const auto mode : {DisplayOutputMode::SDR_sRGB, DisplayOutputMode::HDR_scRGB, DisplayOutputMode::HDR10_PQ}) {
                options.displayOutput.mode = mode;
                for (float peak : {600.0f, 1000.0f}) {
                    options.displayOutput.peakNits = peak;
                    graph.setNodeProperties(display, properties);
                    graph.setNodeProperties(source, {{"level", 16.0f}});
                    if (!frame()) { return RHITestResult::fail(std::string(transform) + " profile/peak: " + log); }
                    if (isHDROutput(mode)) {
                        const float value = halfToFloat(pixels[0]);
                        if (!std::isfinite(value) || value <= 1 || value > peak / 80 + 0.1f) {
                            return RHITestResult::fail(std::string(transform) + " lost HDR headroom or absolute peak");
                        }
                        const auto withLut = pixels;
                        auto noLut = properties; noLut["lut1Weight"] = 0.0f;
                        graph.setNodeProperties(display, noLut);
                        if (!frame() || pixels != withLut) { return RHITestResult::fail("UE legacy custom LUT affected HDR output"); }
                    }
                }
            }
            properties["lut1Weight"] = 0.0f;
            graph.setNodeProperties(display, properties);
            graph.setNodeProperties(source, {{"level", 0.0f}});
            if (!frame() || !std::isfinite(halfToFloat(pixels[0])) || halfToFloat(pixels[0]) > 0.001f) {
                return RHITestResult::fail(std::string(transform) + " black produced NaN or a lifted floor");
            }
            if (!executor.reloadShaders(log) || !frame()) { return RHITestResult::fail("Grading shader reload: " + log); }
            properties["lut1Weight"] = 1.0f;
        }
        options.displayOutput.mode = DisplayOutputMode::SDR_sRGB;
        graph.setNodeProperties(source, {{"level", 0.18f}});
        graph.setNodeProperties(display, RenderGraphProperties::object());
        if (!frame()) { return RHITestResult::fail("Default ACES2: " + log); }
        const auto defaultPixels = pixels;
        graph.setNodeProperties(display, {{"toneCurve", "aces2"}});
        if (!frame() || pixels != defaultPixels) { return RHITestResult::fail("Default grading is not ACES2"); }
        const int defaultGray = channel();
        graph.setNodeRuntimeProperty(display, "colorGain", {1.0f, 1.0f, 1.0f, 2.0f});
        if (!executor.syncRuntimeProperties(graph) || !frame(false) || channel() <= defaultGray) {
            return RHITestResult::fail("Live grading did not regenerate the graph LUT");
        }
        graph.setNodeRuntimeProperties(display, RenderGraphProperties::object());
        options.displayOutput.exposureEV = 1.0f;
        if (!frame()) { return RHITestResult::fail(log); }
        const auto exposed = pixels;
        options.displayOutput.exposureEV = 0.0f;
        graph.setNodeProperties(source, {{"level", 0.36f}});
        if (!frame() || exposed != pixels) { return RHITestResult::fail("Display exposure was applied twice or in the wrong LUT domain"); }
        properties["lut1"] = "missing-color-grading-lut.png";
        graph.setNodeProperties(display, properties);
        if (executor.compile(*device, graph, 8, 8, options, log) || log.find("256x16") == std::string::npos) {
            return RHITestResult::fail("Invalid LUT did not report a useful error");
        }
        for (const auto& edge : graph.edges()) {
            if (edge.dstPass == "Display" && edge.dstField == "lut") { graph.removeEdge(edge.id); break; }
        }
        graph.addEdge("Source.color", "Display.lut");
        if (graph.validate(log) || log.find("dimension/depth mismatch") == std::string::npos) {
            return RHITestResult::fail("A 2D texture was accepted as a 3D LUT");
        }
        if (validationErrors != 0) { return RHITestResult::fail("Color grading emitted Vulkan VUID diagnostics"); }
        return RHITestResult::pass("UE Film, custom LUT identity/inversion/weights, ACES2 and HDR profiles/peaks");
    }
};

METALLIC_REGISTER_RHI_TEST(DisplaySurfaceFormatTest);
METALLIC_REGISTER_RHI_TEST(DisplayOutputGPUTest);
METALLIC_REGISTER_RHI_TEST(ColorGradingGPUTest);

} // namespace
} // namespace metallic::tests

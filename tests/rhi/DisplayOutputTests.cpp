#include "RHITest.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanSurfaceFormat.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <array>
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
        context.commandBuffer().copyTextureToBuffer({.texture = source.texture(),
            .buffer = context.outputBuffer("pixels").buffer(), .bufferRowPitch = context.width() * bytes,
            .bufferSlicePitch = context.width() * context.height() * bytes,
            .width = context.width(), .height = context.height()});
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
        const render::RenderingAttachmentDesc attachment{.view = context.outputTexture("color").view(),
            .state = render::ResourceState::ColorAttachment, .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store, .clearColor = {level, level, level, 1.0f}};
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
            executor.outputResource("Exposure.color")->desc.format != render::Format::RGBA8Unorm) {
            return RHITestResult::fail("HDR -> SDR transition retained HDR resources");
        }
        const auto* sdr = reinterpret_cast<const uint8_t*>(pixels.data());
        if (std::abs(int(sdr[0]) - 186) > 1 || sdr[3] != 255) {
            return RHITestResult::fail("SDR fallback changed the existing Reinhard/gamma result");
        }
        options.displayOutput.mode = render::DisplayOutputMode::HDRscRGB;
        if (!frame() || !near(halfToFloat(pixels[0]), 5.0f)) {
            return RHITestResult::fail("SDR -> HDR transition failed");
        }
        return RHITestResult::pass("Verified FP16 nits, gradients, exposure, highlight shoulder, reload and HDR/SDR switching");
    }
};

METALLIC_REGISTER_RHI_TEST(DisplaySurfaceFormatTest);
METALLIC_REGISTER_RHI_TEST(DisplayOutputGPUTest);

} // namespace
} // namespace metallic::tests

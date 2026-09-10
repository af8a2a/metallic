#include "RhiTest.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanDlssNr.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Subsystem/RenderWorld.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <spdlog/spdlog.h>

namespace metallic::tests {
namespace {

class DlssNrFixturePass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").transferWrite().format = render::Format::Rgba8Unorm;
        reflection.addTextureOutput("motion").transferWrite().format = render::Format::Rg16Sfloat;
        reflection.addTextureOutput("depth").transferWrite().format = render::Format::R32Sfloat;
        return reflection;
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        auto& command = context.commandBuffer();
        command.clearColorTexture(*context.outputTexture("color").texture(), render::ResourceState::TransferDestination,
            {0.25f, 0.5f, 0.75f, 1.0f});
        command.clearColorTexture(*context.outputTexture("motion").texture(), render::ResourceState::TransferDestination,
            {0.0f, 0.0f, 0.0f, 0.0f});
        command.clearColorTexture(*context.outputTexture("depth").texture(), render::ResourceState::TransferDestination,
            {0.5f, 0.0f, 0.0f, 0.0f});
        return {};
    }
};

class DlssNrReadbackPass final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext& context) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("color").transferRead().format = render::Format::Rgba8Unorm;
        reflection.addBufferOutput("pixels").buffer(uint64_t(context.width) * context.height * 4)
            .transferWrite().hostReadback();
        return reflection;
    }
    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        context.commandBuffer().copyTextureToBuffer({
            .texture = context.inputTexture("color").texture(), .buffer = context.outputBuffer("pixels").buffer(),
            .bufferRowPitch = context.width() * 4, .bufferSlicePitch = context.width() * context.height() * 4,
            .width = context.width(), .height = context.height()});
        return {};
    }
};

render::RenderGraph fixtureGraph(bool enabled, bool fallback)
{
    render::registerRenderGraphPassType("DlssNrFixturePass", "NR test inputs",
        [] { return std::make_unique<DlssNrFixturePass>(); });
    render::RenderGraph graph;
    graph.addNode("DlssNrFixturePass", "Source");
    graph.addNode("DlssNrPass", "Nr", {{"enabled", enabled}, {"fallbackToInput", fallback}});
    graph.addEdge("Source.color", "Nr.inputColor");
    graph.addEdge("Source.motion", "Nr.motionVectors");
    graph.addEdge("Source.depth", "Nr.depth");
    graph.markOutput("Nr.color");
    return graph;
}

class DlssNrContractTest final : public RhiTest {
public:
    DlssNrContractTest() { name = "dlss_nr_contract"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        auto pass = render::createRenderGraphPass("DlssNrPass");
        if (!pass || pass->supportsFrameOverlap() || pass->supportsAsyncQueue() ||
            pass->kind() != render::RenderGraphPassKind::Unsafe || pass->queueType() != render::QueueType::Graphics) {
            return RhiTestResult::fail("NR must serialize feature history on graphics");
        }
        const auto settings = pass->runtimeSettings();
        for (const char* key : {"sliderDebug", "splitPosition", "orientation", "swapSides"}) {
            const auto found = std::find_if(settings.begin(), settings.end(),
                [&](const auto& setting) { return setting.key == key; });
            if (found == settings.end() || found->invalidateHistory || found->rebuildGraph) {
                return RhiTestResult::fail("NR slider controls must preserve history and the compiled graph");
            }
            if (found->key == "sliderDebug" && found->defaultValue != false) {
                return RhiTestResult::fail("NR slider debug must default off");
            }
        }
        std::array<std::unique_ptr<render::Texture>, 4> textures;
        std::array<std::unique_ptr<render::TextureView>, 4> views;
        const std::array formats{render::Format::Rgba16Sfloat, render::Format::Rgba16Sfloat,
            render::Format::Rg16Sfloat, render::Format::R32Sfloat};
        for (size_t i = 0; i < 4; ++i) {
            if (!context.device.createTexture({.usage = render::TextureUsageBits::Storage | render::TextureUsageBits::Sampled, .format = formats[i],
                    .width = 32, .height = 24}, textures[i]) ||
                !context.device.createTextureView(*textures[i], {}, views[i])) {
                return RhiTestResult::fail("NR test texture creation failed");
            }
        }
        render::vulkan::DlssNrDesc desc{
            .inputColor = {textures[0].get(), views[0].get()},
            .outputColor = {textures[1].get(), views[1].get()},
            .motionVectors = {textures[2].get(), views[2].get()},
            .depth = {textures[3].get(), views[3].get()},
        };
        std::string log;
        if (!render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail(log); }
        const auto good = desc;
        desc.outputColor = desc.inputColor;
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("NR accepted aliased output"); }
        desc = good; desc.depth.view = nullptr;
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("NR accepted missing depth view"); }
        desc = good; desc.settings.upscaling = true;
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("NR accepted invalid upscaling ratio"); }
        desc = good; desc.settings.intensity = std::numeric_limits<float>::quiet_NaN();
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("NR accepted NaN intensity"); }
        desc = good; desc.settings.preset = 4;
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("NR accepted invalid preset"); }
        desc = good; desc.depth = desc.motionVectors;
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("NR accepted wrong depth format"); }
        std::unique_ptr<render::Texture> unsampled;
        std::unique_ptr<render::TextureView> unsampledView;
        if (!context.device.createTexture({.usage = render::TextureUsageBits::Storage,
                .format = render::Format::Rgba16Sfloat, .width = 32, .height = 24}, unsampled) ||
            !context.device.createTextureView(*unsampled, {}, unsampledView)) {
            return RhiTestResult::fail("NR invalid-input fixture creation failed");
        }
        desc = good; desc.inputColor = {unsampled.get(), unsampledView.get()};
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("NR accepted a non-sampled input (black output regression)"); }
        std::unique_ptr<render::Texture> upscaled;
        std::unique_ptr<render::TextureView> upscaledView;
        if (!context.device.createTexture({.usage = render::TextureUsageBits::Storage,
                .format = render::Format::Rgba16Sfloat, .width = 64, .height = 48}, upscaled) ||
            !context.device.createTextureView(*upscaled, {}, upscaledView)) {
            return RhiTestResult::fail("NR upscaling fixture creation failed");
        }
        desc = good; desc.outputColor = {upscaled.get(), upscaledView.get()}; desc.settings.upscaling = true;
        if (!render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail(log); }
        desc.settings.upscaling = false;
        if (render::vulkan::validateDlssNrDesc(desc, log)) { return RhiTestResult::fail("Native NR accepted mismatched extents"); }
        render::RenderSampleLoadResult sample;
        if (!render::loadBuiltInRenderSample("pathtracing-sample-dlss-nr", sample, log) ||
            !sample.desc.requiresStreamline || !sample.graph.validate(log)) { return RhiTestResult::fail(log); }
        const auto* rr = sample.graph.findNode("DlssRr");
        const auto* nr = sample.graph.findNode("DlssNr");
        if (!rr || rr->properties.value("mode", "") != "DLAA" || !nr || nr->type != "DlssNrPass") {
            return RhiTestResult::fail("NR sample must use full-resolution RR and guides");
        }
        return RhiTestResult::pass();
    }
};

class DlssNrBypassTest final : public RhiTest {
public:
    DlssNrBypassTest() { type = RhiTestType::Rendering; name = "dlss_nr_bypass"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation);
        if (!result) { return RhiTestResult::skip("Preview device unavailable"); }
        auto graph = fixtureGraph(false, false);
        graph.setNodeRuntimeProperty(graph.findNode("Nr")->id, "sliderDebug", true);
        for (auto extent : {std::array{63u, 37u}, std::array{29u, 19u}}) {
            if (!preview.render(graph, extent[0], extent[1]) || preview.pixels().empty()) {
                return RhiTestResult::fail(preview.lastLog());
            }
            for (uint32_t pixel : preview.pixels()) {
                const auto near = [](uint32_t a, uint32_t b) { return a + 1 >= b && a <= b + 1; };
                if (!near(pixel & 255, 64) || !near((pixel >> 8) & 255, 128) ||
                    !near((pixel >> 16) & 255, 191) || (pixel >> 24) != 255) {
                    return RhiTestResult::fail("NR bypass changed input pixels");
                }
            }
        }
        std::string log;
        if (!saveRgba8Png(context.outputDirectory / "dlss_nr_bypass.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), preview.width(), preview.height(), log)) {
            return RhiTestResult::fail(log);
        }
        // A normal device does not initialize Streamline. Fallback must compile
        // there; strict mode must report Unsupported instead of faking success.
        if (context.device.capabilities().streamline) { return RhiTestResult::pass(); }
        render::RenderGraphExecutor executor;
        graph = fixtureGraph(true, true);
        graph.setNodeRuntimeProperty(graph.findNode("Nr")->id, "sliderDebug", true);
        if (!executor.compile(context.device, graph, 32, 24, log) ||
            !executor.execute({.graphicsQueue = &context.graphicsQueue}) || !executor.waitForSubmittedWork()) {
            return RhiTestResult::fail("Unavailable NR did not pass through: " + log);
        }
        graph = fixtureGraph(true, false);
        if (!render::hasError(executor.compile(context.device, graph, 32, 24, log), render::Error::Unsupported)) {
            return RhiTestResult::fail("Strict unavailable NR did not report Unsupported");
        }
        return RhiTestResult::pass();
    }
};

class DlssNrRuntimeTest final : public RhiTest {
public:
    DlssNrRuntimeTest() { type = RhiTestType::Rendering; name = "dlss_nr_runtime"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!render::vulkan::dlssNrSdkAvailable()) { return RhiTestResult::skip("DLSS-NR build option is off"); }
        // Select Streamline at runner startup so this test uses the same device
        // and Vulkan loader as the test environment.
        if (!context.device.capabilities().streamline) {
            return RhiTestResult::skip("Run separately with --rhi-streamline --filter dlss_nr_runtime");
        }
        auto graph = fixtureGraph(true, false);
        render::registerRenderGraphPassType("DlssNrReadbackPass", "NR GPU readback",
            [] { return std::make_unique<DlssNrReadbackPass>(); });
        graph.addNode("DlssNrReadbackPass", "Readback");
        graph.addEdge("Nr.color", "Readback.color");
        graph.markOutput("Readback.pixels");
        graph.addNode("DlssNrReadbackPass", "SourceReadback");
        graph.addEdge("Source.color", "SourceReadback.color");
        graph.markOutput("SourceReadback.pixels");
        render::RenderGraphExecutor executor;
        std::string log;
        auto result = executor.compile(context.device, graph, 256, 144, log);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip(log); }
        if (!result) { return RhiTestResult::fail(log); }
        for (int frame = 0; frame < 3; ++frame) {
            spdlog::info("[DLSS-NR test] Recording frame {}", frame);
            result = executor.execute({.graphicsQueue = &context.graphicsQueue});
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip("DLL initialized but Vulkan feature 18 is unsupported; see NGX diagnostic");
            }
            if (!result || !executor.waitForSubmittedWork()) { return RhiTestResult::fail("NR dispatch failed; see NGX diagnostic"); }
            spdlog::info("[DLSS-NR test] GPU completed frame {}", frame);
        }
        auto* pixels = executor.outputResource("Readback.pixels")->buffer;
        pixels->invalidate();
        const auto* mapped = static_cast<const uint8_t*>(pixels->map());
        if (!mapped) { return RhiTestResult::fail("NR output readback failed"); }
        std::vector<uint8_t> rgba(mapped, mapped + 256 * 144 * 4);
        float energy = 0.0f;
        for (size_t i = 0; i < rgba.size(); ++i) {
            if (i % 4 == 3) { continue; }
            energy += float(rgba[i]) / 255.0f;
        }
        pixels->unmap();
        spdlog::info("[DLSS-NR test] Output mean absolute RGB: {}", energy / float(256 * 144 * 3));
        if (energy < float(256 * 144 * 3) * 0.1f || energy > float(256 * 144 * 3) * 0.9f) {
            return RhiTestResult::fail("NR output is unexpectedly dark or saturated for the midtone fixture");
        }
        if (!saveRgba8Png(context.outputDirectory / "dlss_nr_runtime.png", rgba.data(), 256, 144, log)) {
            return RhiTestResult::fail(log);
        }
        for (float intensity : {0.5f, 0.0f}) {
            graph.setNodeRuntimeProperty(graph.findNode("Nr")->id, "intensity", intensity);
            executor.syncRuntimeProperties(graph);
            if (!executor.execute({.graphicsQueue = &context.graphicsQueue}) || !executor.waitForSubmittedWork()) {
                return RhiTestResult::fail("NR tuning change or zero-intensity bypass failed");
            }
        }
        // Graph output buffers rotate with frame slots. Read the last submitted
        // slot, not the buffer captured before the tuning-change frames.
        pixels = executor.outputResource("Readback.pixels")->buffer;
        pixels->invalidate();
        mapped = static_cast<const uint8_t*>(pixels->map());
        if (!mapped) { return RhiTestResult::fail("Zero-intensity readback failed"); }
        auto* source = executor.outputResource("SourceReadback.pixels")->buffer;
        source->invalidate();
        const auto* original = static_cast<const uint8_t*>(source->map());
        if (original == nullptr) { pixels->unmap(); return RhiTestResult::fail("Source readback failed"); }
        bool exact = true;
        // Compare GPU bytes directly: the UNORM clear's rounding need not match
        // a CPU conversion at half-way values such as 0.5.
        for (size_t i = 0; i < rgba.size(); ++i) { exact = exact && mapped[i] == original[i]; }
        source->unmap();
        pixels->unmap();
        if (!exact) { return RhiTestResult::fail("Zero NR intensity must preserve every input pixel"); }
        return RhiTestResult::pass("Native Vulkan NR history, feature recreation and zero-intensity bypass");
    }
};

class DlssNrSliderTest final : public RhiTest {
public:
    DlssNrSliderTest() { type = RhiTestType::Rendering; name = "dlss_nr_runtime_slider"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!render::vulkan::dlssNrSdkAvailable() || !context.device.capabilities().streamline ||
            !context.device.capabilities().bindlessDescriptorHeap) {
            return RhiTestResult::skip("Requires the NR build and --rhi-streamline");
        }
        auto graph = fixtureGraph(true, false);
        render::registerRenderGraphPassType("DlssNrReadbackPass", "NR GPU readback",
            [] { return std::make_unique<DlssNrReadbackPass>(); });
        const std::array sources{"Source", "Nr"};
        for (const char* source : sources) {
            const auto name = std::string(source) + "Readback";
            graph.addNode("DlssNrReadbackPass", name);
            graph.addEdge(std::string(source) + ".color", name + ".color");
            graph.markOutput(name + ".pixels");
        }
        render::RenderGraphExecutor executor;
        std::string log;
        constexpr uint32_t width = 255, height = 143;
        // Capture an uninterrupted reference sequence before replaying it with
        // the slider. Separate runs avoid relying on concurrent snippet features.
        constexpr size_t frameCount = 3 * 2 * 2 * 4;
        std::array<std::vector<uint8_t>, frameCount> reference;
        {
            render::RenderGraphExecutor referenceExecutor;
            auto result = referenceExecutor.compile(context.device, graph, width, height, log);
            if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip(log); }
            if (!result) { return RhiTestResult::fail(log); }
            for (auto& image : reference) {
                result = referenceExecutor.execute({.graphicsQueue = &context.graphicsQueue});
                if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("NR evaluation unsupported"); }
                if (!result || !referenceExecutor.waitForSubmittedWork()) { return RhiTestResult::fail("NR reference dispatch failed"); }
                auto* buffer = referenceExecutor.outputResource("NrReadback.pixels")->buffer;
                buffer->invalidate();
                const auto* pixels = static_cast<const uint8_t*>(buffer->map());
                if (!pixels) { return RhiTestResult::fail("NR reference readback failed"); }
                image.assign(pixels, pixels + width * height * 4);
                buffer->unmap();
            }
        }
        auto result = executor.compile(context.device, graph, width, height, log);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip(log); }
        if (!result) { return RhiTestResult::fail(log); }
        graph.clearDirty();
        const uint32_t nr = graph.findNode("Nr")->id;
        size_t frame = 0;
        for (bool enabled : {false, true, false}) {
            for (bool horizontal : {false, true}) {
                for (bool swap : {false, true}) {
                    for (float split : {0.0f, 0.31f, 0.5f, 1.0f}) {
                        graph.setNodeRuntimeProperties(nr, {{"sliderDebug", enabled}, {"splitPosition", split},
                            {"orientation", horizontal ? "horizontal" : "vertical"}, {"swapSides", swap}});
                        if (graph.dirty()) { return RhiTestResult::fail("NR slider rebuilt the graph"); }
                        executor.syncRuntimeProperties(graph);
                        result = executor.execute({.graphicsQueue = &context.graphicsQueue});
                        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip("NR evaluation unsupported"); }
                        if (!result || !executor.waitForSubmittedWork()) { return RhiTestResult::fail("NR slider dispatch failed"); }
                        std::array<std::vector<uint8_t>, 2> images;
                        for (size_t i = 0; i < sources.size(); ++i) {
                            auto* buffer = executor.outputResource(std::string(sources[i]) + "Readback.pixels")->buffer;
                            buffer->invalidate();
                            const auto* pixels = static_cast<const uint8_t*>(buffer->map());
                            if (!pixels) { return RhiTestResult::fail("NR slider readback failed"); }
                            images[i].assign(pixels, pixels + width * height * 4);
                            buffer->unmap();
                        }
                        for (uint32_t y = 0; y < height; ++y) {
                            for (uint32_t x = 0; x < width; ++x) {
                                bool before = ((horizontal ? y : x) + 0.5f) < split * (horizontal ? height : width);
                                if (swap) { before = !before; }
                                const auto& expected = enabled && before ? images[0] : reference[frame];
                                const size_t pixel = (y * width + x) * 4;
                                for (size_t channel = 0; channel < 4; ++channel) {
                                    if (images[1][pixel + channel] != expected[pixel + channel]) {
                                        return RhiTestResult::fail(fmt::format(
                                            "NR slider mismatch enabled={} horizontal={} swap={} split={} pixel=({}, {}) channel={} actual={} expected={}",
                                            enabled, horizontal, swap, split, x, y, channel,
                                            images[1][pixel + channel], expected[pixel + channel]));
                                    }
                                }
                            }
                        }
                        ++frame;
                    }
                }
            }
        }
        return RhiTestResult::pass("NR before/after pixels, axes, swap, odd extents, endpoints and history retention");
    }
};

class DlssNrSceneTest final : public RhiTest {
public:
    DlssNrSceneTest() { type = RhiTestType::Rendering; name = "dlss_nr_runtime_scene"; }
    RhiTestResult run(RhiTestContext& context) override
    {
        if (!render::vulkan::dlssNrSdkAvailable() || !context.device.capabilities().streamline ||
            !context.device.capabilities().rayQuery) {
            return RhiTestResult::skip("Requires the NR build and --rhi-streamline on a ray-query device");
        }
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("pathtracing-sample-dlss-nr", sample, log)) {
            return RhiTestResult::fail(log);
        }
        auto& graph = sample.graph;
        graph.findNode("PathTrace")->properties["maxDepth"] = 4;
        graph.findNode("DlssNr")->properties["fallbackToInput"] = false;
        graph.findNode("AutoExposure")->properties["adaptationDeltaSeconds"] = 1.0f / 60.0f;
        render::registerRenderGraphPassType("DlssNrReadbackPass", "NR GPU readback",
            [] { return std::make_unique<DlssNrReadbackPass>(); });
        graph.addNode("DlssNrReadbackPass", "BeforeNr");
        graph.addEdge("AutoExposure.color", "BeforeNr.color");
        graph.markOutput("BeforeNr.pixels");
        graph.addNode("DlssNrReadbackPass", "AfterNr");
        graph.addEdge("DlssNr.color", "AfterNr.color");
        graph.markOutput("AfterNr.pixels");
        render::RenderWorld world;
        world.setEnvironment({.enabled = true,
            .path = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.environment->path});
        render::RenderGraphExecutor executor;
        executor.bindRenderWorld(&world);
        constexpr uint32_t width = 640, height = 360;
        auto result = executor.compile(context.device, graph, width, height, log);
        if (render::hasError(result, render::Error::Unsupported)) { return RhiTestResult::skip(log); }
        if (!result) { return RhiTestResult::fail(log); }
        for (int frame = 0; frame < 8; ++frame) {
            result = executor.execute({.graphicsQueue = &context.graphicsQueue});
            if (render::hasError(result, render::Error::Unsupported)) {
                return RhiTestResult::skip("Runtime does not support the sample's DLSS features");
            }
            if (!result || !executor.waitForSubmittedWork()) { return RhiTestResult::fail("NR scene dispatch failed"); }
        }
        std::array<std::vector<uint8_t>, 2> images;
        const std::array outputs{"BeforeNr.pixels", "AfterNr.pixels"};
        const std::array names{"dlss_nr_scene_before.png", "dlss_nr_scene_after.png"};
        for (size_t i = 0; i < images.size(); ++i) {
            auto* buffer = executor.outputResource(outputs[i])->buffer;
            buffer->invalidate();
            const auto* pixels = static_cast<const uint8_t*>(buffer->map());
            if (pixels == nullptr) { return RhiTestResult::fail("NR scene readback failed"); }
            images[i].assign(pixels, pixels + width * height * 4);
            buffer->unmap();
            if (!saveRgba8Png(context.outputDirectory / names[i], images[i].data(), width, height, log)) {
                return RhiTestResult::fail(log);
            }
        }
        uint64_t difference = 0, energy = 0;
        for (size_t i = 0; i < images[0].size(); ++i) {
            if (i % 4 == 3) { continue; }
            difference += std::abs(int(images[0][i]) - int(images[1][i]));
            energy += images[1][i];
        }
        if (difference == 0 || energy < uint64_t(width) * height * 3 * 4) {
            return RhiTestResult::fail("NR scene output is unchanged or unexpectedly black");
        }
        graph.setNodeRuntimeProperty(graph.findNode("DlssNr")->id, "sliderDebug", true);
        executor.syncRuntimeProperties(graph);
        if (!executor.execute({.graphicsQueue = &context.graphicsQueue}) || !executor.waitForSubmittedWork()) {
            return RhiTestResult::fail("NR scene slider dispatch failed");
        }
        auto* slider = executor.outputResource("AfterNr.pixels")->buffer;
        slider->invalidate();
        const auto* pixels = static_cast<const uint8_t*>(slider->map());
        if (!pixels) { return RhiTestResult::fail("NR scene slider readback failed"); }
        const bool saved = saveRgba8Png(context.outputDirectory / "dlss_nr_scene_slider.png", pixels, width, height, log);
        slider->unmap();
        if (!saved) { return RhiTestResult::fail(log); }
        return RhiTestResult::pass("Captured RR + exposure before/after native NR over eight scene frames");
    }
};

METALLIC_REGISTER_RHI_TEST(DlssNrContractTest);
METALLIC_REGISTER_RHI_TEST(DlssNrBypassTest);
METALLIC_REGISTER_RHI_TEST(DlssNrRuntimeTest);
METALLIC_REGISTER_RHI_TEST(DlssNrSliderTest);
METALLIC_REGISTER_RHI_TEST(DlssNrSceneTest);

} // namespace
} // namespace metallic::tests

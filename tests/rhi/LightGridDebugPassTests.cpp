#include "RhiTest.h"

#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>

namespace metallic::tests {
namespace {

#define LIGHT_DEBUG_CHECK(condition) \
    do { \
        if (!(condition)) { \
            return RhiTestResult::fail(std::string("LightGridDebug: ") + #condition); \
        } \
    } while (false)

using DebugColor = std::array<float, 3>;
constexpr DebugColor kBlack{0.0f, 0.0f, 0.0f};
constexpr DebugColor kCyan{0.0f, 0.85f, 1.0f};
constexpr DebugColor kGreen{0.15f, 1.0f, 0.25f};
constexpr DebugColor kYellow{1.0f, 0.9f, 0.05f};
constexpr DebugColor kRed{1.0f, 0.08f, 0.02f};
constexpr DebugColor kOverflow{1.0f, 0.0f, 1.0f};

bool pixelMatches(uint32_t pixel, const DebugColor& color)
{
    for (uint32_t channel = 0; channel < color.size(); ++channel) {
        const int expected = static_cast<int>(std::lround(color[channel] * 255.0f));
        const int actual = static_cast<int>((pixel >> (channel * 8u)) & 255u);
        if (std::abs(actual - expected) > 1) { return false; }
    }
    return (pixel >> 24u) == 255u;
}

bool solidColor(const render::RenderGraphPreviewRenderer& preview, const DebugColor& color)
{
    return !preview.pixels().empty() && std::ranges::all_of(preview.pixels(),
        [&](uint32_t pixel) { return pixelMatches(pixel, color); });
}

scene::PunctualLight debugPoint(float3 position = float3(0.0f, 0.0f, -5.0f), double range = 100.0)
{
    scene::PunctualLight light;
    light.name = "Debug test point";
    light.properties.type = "point";
    light.properties.intensity = 100.0;
    light.properties.intensityUnit = scene::LightUnit::Candela;
    light.properties.range = range;
    light.position = position;
    return light;
}

render::RenderGraphProperties debugProperties()
{
    return {{"source", "world"}, {"visualization", "peak"},
        {"showGrid", false}, {"showLegend", false}, {"showCounts", false},
        {"heatmapMaxLights", 4}, {"includeGlobalLights", false},
        {"tileSize", 32}, {"depthSliceCount", 4}, {"maxLightsPerCell", 8},
        {"camera", {{"eye", {0.0, 0.0, 0.0}}, {"center", {0.0, 0.0, -1.0}},
            {"up", {0.0, 1.0, 0.0}}, {"projection", "orthographic"},
            {"orthoHeight", 4.0}, {"znear", 1.0}, {"zfar", 9.0}}}};
}

RhiTestResult initializePreview(render::RenderGraphPreviewRenderer& preview, RhiTestContext& context)
{
    const auto result = preview.initialize(context.enableValidation);
    if (render::hasError(result, render::Error::Unsupported)) {
        return RhiTestResult::skip("LightGrid debug requires bindless compute support");
    }
    if (!result) { return RhiTestResult::fail("LightGrid preview initialization: " + std::string(toString(result))); }
    preview.setEnvironment({.enabled = false});
    return RhiTestResult::pass();
}

class EmptySceneDebugSample final : public render::RenderSample {
public:
    EmptySceneDebugSample(bool loadScene, bool pathTarget) : loadScene_(loadScene), pathTarget_(pathTarget) {}
    std::string_view id() const override { return "test-light-grid-empty-scene"; }
    std::string_view name() const override { return "LightGrid empty scene guard"; }
    std::string_view category() const override { return "Test"; }
    std::string scenePath() const override { return {}; }
    bool loadSceneInEditor() const override { return loadScene_; }
    std::string graphPath() const override { return "Pipelines/Samples/light_grid_debug.metallic_graph.json"; }
    std::vector<std::string> scenePathTargets() const override
    {
        return pathTarget_ ? std::vector<std::string>{"LightGridDebug"} : std::vector<std::string>{};
    }

private:
    bool loadScene_;
    bool pathTarget_;
};

class LightGridDebugContractTest final : public RhiTest {
public:
    LightGridDebugContractTest() { name = "render_graph_light_grid_debug_contract"; }

    RhiTestResult run(RhiTestContext&) override
    {
        render::RenderGraph graph;
        const auto pass = render::createRenderGraphPass("LightGridDebugPass");
        LIGHT_DEBUG_CHECK(pass != nullptr);
        const auto reflection = pass->reflect({.width = 127, .height = 73});
        const auto* color = reflection.findField("color", render::RenderGraphFieldVisibility::Output);
        LIGHT_DEBUG_CHECK(color != nullptr);
        LIGHT_DEBUG_CHECK(color->format == render::Format::Rgba8Unorm);
        const auto settings = pass->runtimeSettings();
        for (const auto& setting : settings) {
            if (setting.type == render::RenderGraphRuntimeSettingType::Int) {
                // ImGui's integer sliders require half-range bounds internally.
                LIGHT_DEBUG_CHECK(setting.minValue.get<int64_t>() >= INT32_MIN / 2);
                LIGHT_DEBUG_CHECK(setting.maxValue.get<int64_t>() <= INT32_MAX / 2);
            }
        }
        for (const char* key : {"source", "visualization", "sliceIndex", "viewDepth", "heatmapMaxLights",
                "showGrid", "showLegend", "showCounts", "includeGlobalLights", "lightCount", "tileSize"}) {
            LIGHT_DEBUG_CHECK(std::ranges::any_of(settings, [&](const auto& setting) { return setting.key == key; }));
        }

        render::RenderSampleLoadResult sample;
        std::string log;
        LIGHT_DEBUG_CHECK(render::loadBuiltInRenderSample("light-grid-debug", sample, log));
        LIGHT_DEBUG_CHECK(sample.desc.scenePath.empty());
        LIGHT_DEBUG_CHECK(!sample.desc.loadSceneInEditor);
        LIGHT_DEBUG_CHECK(sample.desc.scenePathTargets.empty());
        LIGHT_DEBUG_CHECK(sample.desc.environment.has_value() && !sample.desc.environment->enabled);
        LIGHT_DEBUG_CHECK(sample.desc.previewOutput == "FinalBlit.color");
        LIGHT_DEBUG_CHECK(sample.graph.validate(log));
        const auto* node = sample.graph.findNode("LightGridDebug");
        LIGHT_DEBUG_CHECK(node != nullptr && node->type == "LightGridDebugPass");
        LIGHT_DEBUG_CHECK(node->properties.value("source", "") == "bench");
        LIGHT_DEBUG_CHECK(!node->properties.value("animate", true));
        LIGHT_DEBUG_CHECK(sample.graph.firstOutputName() == "FinalBlit.color");
        LIGHT_DEBUG_CHECK(sample.graph.outputs().empty());
        LIGHT_DEBUG_CHECK(render::deserializeRenderGraphFromString(
            render::serializeRenderGraphToString(sample.graph), graph, log));
        LIGHT_DEBUG_CHECK(graph.validate(log));
        LIGHT_DEBUG_CHECK(graph.findNode("LightGridDebug")->properties == node->properties);

        LIGHT_DEBUG_CHECK(render::loadRenderSample(EmptySceneDebugSample(false, false), sample, log));
        for (const auto [loadScene, pathTarget] : {std::pair{true, false}, std::pair{false, true}, std::pair{true, true}}) {
            LIGHT_DEBUG_CHECK(!render::loadRenderSample(EmptySceneDebugSample(loadScene, pathTarget), sample, log));
            LIGHT_DEBUG_CHECK(log.find("scenePath") != std::string::npos);
        }
        return RhiTestResult::pass("Debug pass registration, runtime controls, asset-free sample and scene-path guards");
    }
};

class LightGridDebugPixelsTest final : public RhiTest {
public:
    LightGridDebugPixelsTest() { type = RhiTestType::Rendering; name = "render_graph_light_grid_debug_pixels"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = initializePreview(preview, context);
        if (!initialized.passed) { return initialized; }
        render::RenderGraph graph;
        const uint32_t node = graph.addNode("LightGridDebugPass", "Debug", debugProperties())->id;
        graph.markOutput("Debug.color");
        const auto render = [&]() { return preview.render(graph, 64, 64); };
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));

        scene::LightingSettings lighting;
        for (const auto [count, color] : {std::pair{1u, kCyan}, std::pair{2u, kGreen}, std::pair{4u, kRed}}) {
            lighting.lights.assign(count, debugPoint());
            LIGHT_DEBUG_CHECK(preview.setLighting(lighting));
            if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
            LIGHT_DEBUG_CHECK(solidColor(preview, color));
        }
        // Do not turn a list-capacity overflow into a falsely low stored-count heatmap.
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "maxLightsPerCell", 1));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kOverflow));
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "maxLightsPerCell", 8));

        lighting.lights.assign(1, debugPoint());
        lighting.lights.push_back(debugPoint(float3(0.0f), 0.0));
        auto directional = debugPoint();
        directional.properties.type = "directional";
        directional.properties.intensityUnit = scene::LightUnit::Lux;
        directional.direction = float3(0.0f, 0.0f, -1.0f);
        lighting.lights.push_back(directional);
        LIGHT_DEBUG_CHECK(preview.setLighting(lighting));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kCyan));
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "includeGlobalLights", true));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kYellow));
        // Global lights do not occupy local-list capacity and must not report overflow.
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "maxLightsPerCell", 1));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kYellow));

        lighting.lights.assign(2, debugPoint());
        lighting.lights[0].enabled = false;
        lighting.lights[1].properties.intensity = 0.0;
        LIGHT_DEBUG_CHECK(preview.setLighting(lighting));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));
        lighting.lights.clear();
        LIGHT_DEBUG_CHECK(preview.setLighting(lighting));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));
        return RhiTestResult::pass("GPU heatmap palette, live world-light edits, disabled lights, global lights and overflow");
    }
};

class LightGridDebugSlicesTest final : public RhiTest {
public:
    LightGridDebugSlicesTest() { type = RhiTestType::Rendering; name = "render_graph_light_grid_debug_slices"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = initializePreview(preview, context);
        if (!initialized.passed) { return initialized; }
        render::RenderGraph graph;
        auto properties = debugProperties();
        properties["visualization"] = "slice";
        properties["sliceIndex"] = 0;
        const uint32_t node = graph.addNode("LightGridDebugPass", "Debug", properties)->id;
        graph.markOutput("Debug.color");
        scene::LightingSettings lighting;
        lighting.lights.push_back(debugPoint(float3(-1.0f, 1.0f, -2.0f), 0.2));
        LIGHT_DEBUG_CHECK(preview.setLighting(lighting));
        const auto render = [&]() { return preview.render(graph, 64, 64); };
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels().size() == 64u * 64u);
        for (uint32_t y = 0; y < 64; ++y) {
            for (uint32_t x = 0; x < 64; ++x) {
                LIGHT_DEBUG_CHECK(pixelMatches(preview.pixels()[y * 64 + x], x < 32 && y < 32 ? kCyan : kBlack));
            }
        }
        const auto covered = preview.pixels();
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "sliceIndex", 3));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "visualization", "depth"));
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "viewDepth", 2.0));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == covered);
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "viewDepth", 8.0));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));
        for (double depth : {0.0, 10.0}) {
            LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "viewDepth", depth));
            if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
            LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));
        }
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "visualization", "peak"));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == covered);

        lighting.lights[0].position.z = -8.0f;
        LIGHT_DEBUG_CHECK(preview.setLighting(lighting));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == covered);
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "visualization", "slice"));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == covered);
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "sliceIndex", 255));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == covered);
        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "sliceIndex", 0));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));

        LIGHT_DEBUG_CHECK(graph.setNodeRuntimeProperty(node, "visualization", "peak"));
        if (!preview.render(graph, 127, 73)) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.width() == 127 && preview.height() == 73);
        LIGHT_DEBUG_CHECK(preview.pixels().size() == 127u * 73u);
        const auto litCount = std::ranges::count_if(preview.pixels(), [](uint32_t p) { return !pixelMatches(p, kBlack); });
        LIGHT_DEBUG_CHECK(litCount > 0 && static_cast<size_t>(litCount) < preview.pixels().size());
        LIGHT_DEBUG_CHECK(std::ranges::all_of(preview.pixels(), [](uint32_t p) { return (p >> 24u) == 255u; }));
        const auto resized = preview.pixels();
        if (!preview.render(graph, 127, 73)) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == resized);
        return RhiTestResult::pass("GPU tile coverage, explicit depth and Z slice, peak reduction, partial tiles and resize");
    }
};

class LightGridDebugBenchTest final : public RhiTest {
public:
    LightGridDebugBenchTest() { type = RhiTestType::Rendering; name = "render_graph_light_grid_debug_bench"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraphPreviewRenderer preview;
        const auto initialized = initializePreview(preview, context);
        if (!initialized.passed) { return initialized; }
        render::RenderSampleLoadResult sample;
        std::string log;
        LIGHT_DEBUG_CHECK(render::loadBuiltInRenderSample("light-grid-debug", sample, log));
        const auto render = [&]() { return preview.render(sample.graph, 640, 360); };
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        const auto reference = preview.pixels();
        LIGHT_DEBUG_CHECK(reference.size() == 640u * 360u);
        const std::set<uint32_t> colors(reference.begin(), reference.end());
        LIGHT_DEBUG_CHECK(colors.size() > 4);
        LIGHT_DEBUG_CHECK(saveRgba8Png(context.outputDirectory / "light_grid_debug_heatmap.png",
            reinterpret_cast<const uint8_t*>(reference.data()), 640, 360, log));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == reference);
        // The bench owns its fixtures; unrelated editor/world lights must not leak into it.
        scene::LightingSettings lighting;
        lighting.lights.assign(20, debugPoint());
        LIGHT_DEBUG_CHECK(preview.setLighting(lighting));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == reference);

        const uint32_t node = sample.graph.findNode("LightGridDebug")->id;
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "showCounts", true));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() != reference);
        LIGHT_DEBUG_CHECK(saveRgba8Png(context.outputDirectory / "light_grid_debug_counts.png",
            reinterpret_cast<const uint8_t*>(preview.pixels().data()), 640, 360, log));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "showCounts", false));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "spotFraction", 0.0));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        const auto pointsOnly = preview.pixels();
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "spotFraction", 1.0));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() != pointsOnly);
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "spotFraction", 0.25));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "seed", 17));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() != reference);
        const auto seeded = preview.pixels();
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(preview.pixels() == seeded);

        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "layout", "overlap"));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "lightCount", 64));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "maxLightsPerCell", 4));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "showLegend", false));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "showGrid", false));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "showCounts", false));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(std::ranges::any_of(preview.pixels(), [](uint32_t p) { return pixelMatches(p, kOverflow); }));
        LIGHT_DEBUG_CHECK(saveRgba8Png(context.outputDirectory / "light_grid_debug_overflow.png",
            reinterpret_cast<const uint8_t*>(preview.pixels().data()), 640, 360, log));
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "lightCount", 0));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kBlack));
        // Switching back proves fixture creation/removal did not overwrite world lighting.
        LIGHT_DEBUG_CHECK(sample.graph.setNodeRuntimeProperty(node, "source", "world"));
        if (!render()) { return RhiTestResult::fail(preview.lastLog()); }
        LIGHT_DEBUG_CHECK(solidColor(preview, kOverflow));
        return RhiTestResult::pass("Asset-free sample, deterministic GPU fixtures, seed changes, isolated bench and overflow image");
    }
};

// This test intentionally injects a compile failure into the production producer.
// Only append a unique marker, and restore only if the original bytes and our
// marker still match: never overwrite an unrelated concurrent shader edit.
class ScopedLightGridShaderFailure {
public:
    explicit ScopedLightGridShaderFailure(std::filesystem::path path) : path_(std::move(path)) {}
    ~ScopedLightGridShaderFailure() { (void)restore(); }

    bool inject()
    {
        if (active_ || !readFile(original_) || original_.empty()) { return false; }
        active_ = true;
        std::ofstream stream(path_, std::ios::binary | std::ios::app);
        stream.write(marker_.data(), static_cast<std::streamsize>(marker_.size()));
        stream.flush();
        return stream.good();
    }

    bool restore()
    {
        if (!active_) { return true; }
        std::string current;
        if (!readFile(current) || !current.starts_with(original_) ||
            !marker_.starts_with(current.substr(original_.size()))) {
            return false;
        }
        std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
        stream.write(original_.data(), static_cast<std::streamsize>(original_.size()));
        stream.flush();
        if (!stream.good()) { return false; }
        active_ = false;
        return true;
    }

private:
    bool readFile(std::string& contents) const
    {
        std::ifstream stream(path_, std::ios::binary);
        if (!stream) { return false; }
        contents.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
        return !stream.bad();
    }

    std::filesystem::path path_;
    std::string original_;
    const std::string marker_ = "\n#error Metallic_LightGridDebug_Producer_Reload_Failure_Probe\n";
    bool active_ = false;
};

class LightGridDebugReloadTest final : public RhiTest {
public:
    LightGridDebugReloadTest() { type = RhiTestType::Rendering; name = "render_graph_light_grid_debug_reload"; }

    RhiTestResult run(RhiTestContext& context) override
    {
        std::unique_ptr<render::Device> device;
        const auto initialized = render::createDevice({.applicationName = "LightGrid debug reload test",
            .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, device);
        if (render::hasError(initialized, render::Error::Unsupported)) {
            return RhiTestResult::skip("LightGrid debug reload requires bindless compute support");
        }
        LIGHT_DEBUG_CHECK(initialized);
        auto* queue = device->getQueue(render::QueueType::Graphics);
        LIGHT_DEBUG_CHECK(queue != nullptr);
        std::unique_ptr<render::CommandPool> pool;
        std::unique_ptr<render::CommandBuffer> commands;
        std::unique_ptr<render::Buffer> readback;
        render::RenderGraphExecutor executor;
        struct WaitBeforeDestruction {
            render::Device& device;
            ~WaitBeforeDestruction() { (void)device.waitIdle(); }
        } waitBeforeDestruction{*device};
        LIGHT_DEBUG_CHECK(device->createCommandPool(*queue, pool));
        LIGHT_DEBUG_CHECK(pool->createCommandBuffer(commands));
        constexpr uint32_t kExtent = 32;
        constexpr uint64_t kReadbackBytes = kExtent * kExtent * sizeof(uint32_t);
        LIGHT_DEBUG_CHECK(device->createBuffer({.size = kReadbackBytes,
            .usage = render::BufferUsageBits::TransferDestination,
            .memoryLocation = render::MemoryLocation::HostReadback}, readback));
        render::RenderGraph graph;
        auto properties = debugProperties();
        properties["source"] = "bench";
        properties["lightCount"] = 1;
        properties["lightRange"] = 100;
        properties["spotFraction"] = 0;
        properties["layout"] = "overlap";
        graph.addNode("LightGridDebugPass", "Debug", properties);
        graph.markOutput("Debug.color");
        std::string log;
        auto result = executor.compile(*device, graph, kExtent, kExtent,
            {.enablePreviewOutputAccess = true}, log);
        if (!result) { return RhiTestResult::fail("LightGrid reload graph compile: " + log); }
        const auto* output = executor.outputResource("Debug.color");
        LIGHT_DEBUG_CHECK(output != nullptr && output->texture != nullptr);
        bool firstReadback = true;
        const auto renderFrame = [&]() -> RhiTestResult {
            LIGHT_DEBUG_CHECK(pool->reset());
            LIGHT_DEBUG_CHECK(commands->begin());
            // Exercise legacy, untracked recording as well as the tracked preview tests.
            LIGHT_DEBUG_CHECK(executor.execute(*commands));
            LIGHT_DEBUG_CHECK(executor.transitionOutput(*commands, "Debug.color", render::ResourceState::TransferSource));
            const render::BufferBarrierDesc barrier{.buffer = readback.get(),
                .before = firstReadback ? render::ResourceState::Undefined : render::ResourceState::TransferDestination,
                .after = render::ResourceState::TransferDestination};
            commands->barrier({.buffers = &barrier, .bufferCount = 1});
            commands->copyTextureToBuffer({.texture = output->texture, .buffer = readback.get(),
                .width = kExtent, .height = kExtent, .depth = 1});
            LIGHT_DEBUG_CHECK(commands->end());
            render::CommandBuffer* submission[] = {commands.get()};
            LIGHT_DEBUG_CHECK(queue->submit({.commandBuffers = submission, .commandBufferCount = 1}));
            LIGHT_DEBUG_CHECK(queue->waitIdle());
            readback->invalidate();
            const auto* mapped = static_cast<const uint32_t*>(readback->map());
            LIGHT_DEBUG_CHECK(mapped != nullptr);
            std::array<uint32_t, kExtent * kExtent> pixels;
            std::memcpy(pixels.data(), mapped, sizeof(pixels));
            readback->unmap();
            firstReadback = false;
            LIGHT_DEBUG_CHECK(std::ranges::all_of(pixels, [](uint32_t pixel) { return pixelMatches(pixel, kCyan); }));
            return RhiTestResult::pass();
        };
        auto rendered = renderFrame();
        if (!rendered.passed) { return rendered; }
        result = executor.reloadShaders(log);
        if (!result) { return RhiTestResult::fail("LightGrid successful reload: " + log); }
        LIGHT_DEBUG_CHECK(executor.outputResource("Debug.color") == output);
        rendered = renderFrame();
        if (!rendered.passed) { return rendered; }

        ScopedLightGridShaderFailure shaderFailure(PROJECT_SOURCE_DIR "/Shaders/Features/Lighting/ClusterLightGrid.slang");
        LIGHT_DEBUG_CHECK(shaderFailure.inject());
        result = executor.reloadShaders(log);
        LIGHT_DEBUG_CHECK(!result);
        LIGHT_DEBUG_CHECK(executor.compiled());
        LIGHT_DEBUG_CHECK(executor.outputResource("Debug.color") == output);
        // The producer remains invalid on disk here. Rendering must use the old
        // committed producer bytecode instead of lazily recompiling after reload.
        rendered = renderFrame();
        if (!rendered.passed) { return rendered; }
        LIGHT_DEBUG_CHECK(shaderFailure.restore());
        result = executor.reloadShaders(log);
        if (!result) { return RhiTestResult::fail("LightGrid recovery reload: " + log); }
        rendered = renderFrame();
        if (!rendered.passed) { return rendered; }
        LIGHT_DEBUG_CHECK(pool->reset());
        return RhiTestResult::pass("Raw recording, successful shader reload, failed-producer rollback and recovery preserve GPU pixels");
    }
};

METALLIC_REGISTER_RHI_TEST(LightGridDebugContractTest);
METALLIC_REGISTER_RHI_TEST(LightGridDebugPixelsTest);
METALLIC_REGISTER_RHI_TEST(LightGridDebugSlicesTest);
METALLIC_REGISTER_RHI_TEST(LightGridDebugBenchTest);
METALLIC_REGISTER_RHI_TEST(LightGridDebugReloadTest);

#undef LIGHT_DEBUG_CHECK

} // namespace
} // namespace metallic::tests

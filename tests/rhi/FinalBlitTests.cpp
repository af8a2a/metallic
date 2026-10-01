#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unordered_map>

namespace metallic::tests {
namespace {

class FinalBlitTestSource final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").texture2D(17, 9).format =
            properties().value("integer", false) ? render::Format::R32Uint : render::Format::RGBA16Sfloat;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        const render::TextureHandle color = context.outputTexture("color");
        render::RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = render::ResourceState::ColorAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearColor = render::ColorValue{0.2f, 0.4f, 0.8f, 0.0f},
        };
        if (auto commandResult = context.commandBuffer().beginRendering(render::RenderingDesc{
            .renderArea = render::Rect{0, 0, context.width(), context.height()},
            .colorAttachments = {&attachment, 1},
        }); !commandResult) { return commandResult; }
        context.commandBuffer().endRendering();
        return {};
    }
};

class FinalBlitGraphTest : public RHITest {
public:
    FinalBlitGraphTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_final_blit_contract";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::RenderGraph graph;
        const uint32_t finalId = graph.addNode("FinalBlitPass", "FinalBlit")->id;
        std::string log;
        if (!graph.validate(log) || !graph.outputs().empty() ||
            graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("FinalBlit must work without marked outputs: " + log);
        }
        render::RenderGraph roundTrip;
        if (!render::deserializeRenderGraphFromString(render::serializeRenderGraphToString(graph), roundTrip, log) ||
            roundTrip.firstOutputName() != "FinalBlit.color" || !roundTrip.outputs().empty()) {
            return RHITestResult::fail("FinalBlit presentation did not survive serialization: " + log);
        }
        graph.addNode("ClearColorPass", "Source");
        graph.addEdge("Source.color", "FinalBlit.source");
        // No extraOutputs: the executor itself must retain the presentation root.
        std::unique_ptr<render::Device> device;
        const render::Result<> deviceResult = render::createDevice(render::DeviceDesc{
            .applicationName = "FinalBlit Test",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
        }).transform([&](auto rhiValue) { device = std::move(rhiValue); });
        if (!deviceResult) {
            return render::hasError(deviceResult, render::Error::Unsupported)
                ? RHITestResult::skip(toString(deviceResult))
                : RHITestResult::fail(toString(deviceResult));
        }
        render::RenderGraphExecutor executor;
        if (!executor.compile(*device, graph, 63, 37, log) ||
            executor.outputResource("FinalBlit.color") == nullptr ||
            executor.outputResource("Source.color") == nullptr) {
            return RHITestResult::fail("FinalBlit or its producer was culled: " + log);
        }
        const auto* output = executor.outputResource("FinalBlit.color");
        if (!render::hasFlag(output->desc.usage, render::TextureUsageBits::Sampled) ||
            !render::hasFlag(output->desc.usage, render::TextureUsageBits::TransferSource)) {
            return RHITestResult::fail("Automatic presentation lacks display/readback access");
        }
        graph.markOutput("Source.color");
        if (graph.firstOutputName() != "FinalBlit.color") {
            return RHITestResult::fail("Legacy marked output overrides presentation");
        }
        graph.renameNode(finalId, "Present");
        if (graph.firstOutputName() != "Present.color" || !graph.validate(log)) {
            return RHITestResult::fail("Presentation rename broke graph: " + log);
        }
        const uint32_t duplicateId = graph.addNode("FinalBlitPass", "Duplicate")->id;
        if (graph.validate(log) || log.find("multiple presentation") == std::string::npos) {
            return RHITestResult::fail("Ambiguous presentation outputs were accepted");
        }
        graph.removeNode(duplicateId);
        graph.removeNode(finalId);
        if (!graph.validate(log) || graph.firstOutputName() != "Source.color") {
            return RHITestResult::fail("Removing FinalBlit did not restore legacy output");
        }
        return RHITestResult::pass();
    }
};

// The real scene exercises typed presentation alongside legacy resource-table passes.
// Keep this opt-in and bounded; do not substitute the much larger default scene.
class MiniZorahFinalBlitTest final : public RHITest {
public:
    MiniZorahFinalBlitTest() { type = RHITestType::Rendering; name = "minizorah_typed_final_blit_frames"; }
    RHITestResult run(RHITestContext& context) override
    {
        const char* enabled = std::getenv("METALLIC_TEST_MINIZORAH");
        if (!enabled || std::string_view(enabled) != "1") { return RHITestResult::skip("Set METALLIC_TEST_MINIZORAH=1"); }
        render::RenderSampleLoadResult sample;
        std::string log;
        if (!render::loadBuiltInRenderSample("gpu-driven-minizorah-vbuffer", sample, log)) {
            return RHITestResult::fail(log);
        }
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, false, false);
        if (!result) { return RHITestResult::fail(std::string("Preview initialization: ") + toString(result)); }
        for (uint32_t frame = 0; frame < 12; ++frame) {
            preview.setRecordingWorkerLimit(frame < 6 ? 1 : 4);
            result = preview.render(sample.graph, 640, 360, "FinalBlit.color");
            if (!result) {
                return RHITestResult::fail("MiniZorah frame " + std::to_string(frame) + ": " + toString(result) + ": " + preview.lastLog());
            }
            const auto& pixels = preview.pixels();
            if (pixels.size() != 640 * 360 || std::all_of(pixels.begin(), pixels.end(),
                    [&](uint32_t pixel) { return pixel == pixels.front(); })) {
                return RHITestResult::fail("MiniZorah final display is empty or uniform");
            }
        }
        if (!saveRgba8Png(context.outputDirectory / "minizorah_typed_final_blit.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()), 640, 360, log)) {
            return RHITestResult::fail(log);
        }
        return RHITestResult::pass("12 MiniZorah final-display frames with 1/4 recording workers; sampled-image inline parameters");
    }
};
METALLIC_REGISTER_RHI_TEST(MiniZorahFinalBlitTest);

class FinalBlitPixelsTest : public RHITest {
public:
    FinalBlitPixelsTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_final_blit_pixels";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::registerRenderGraphPassType("FinalBlitTestSource", "Test source",
            [] { return std::make_unique<FinalBlitTestSource>(); });
        render::RenderGraphPreviewRenderer preview;
        render::Result<> result = preview.initialize(context.enableValidation);
        if (!result) {
            return RHITestResult::skip("Preview device unavailable: " + std::string(toString(result)));
        }
        render::RenderGraph graph;
        graph.addNode("FinalBlitPass", "FinalBlit");
        result = preview.render(graph, 63, 37);
        if (!result || !isUv(preview)) {
            return RHITestResult::fail("Unconnected FinalBlit did not display UV: " + preview.lastLog());
        }
        std::string log;
        if (!saveRgba8Png(context.outputDirectory / "final_blit_uv.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                preview.width(), preview.height(), log)) {
            return RHITestResult::fail(log);
        }

        graph.addNode("FinalBlitTestSource", "Source");
        const uint32_t edgeId = graph.addEdge("Source.color", "FinalBlit.source")->id;
        result = preview.render(graph, 63, 37);
        if (!result || preview.width() != 63 || preview.height() != 37 || !isSolid(preview)) {
            return RHITestResult::fail("Float RT blit/resize/opaque alpha failed: " + preview.lastLog());
        }
        if (!saveRgba8Png(context.outputDirectory / "final_blit_connected.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                preview.width(), preview.height(), log)) {
            return RHITestResult::fail(log);
        }
        result = preview.render(graph, 29, 19);
        if (!result || preview.width() != 29 || preview.height() != 19 || !isSolid(preview)) {
            return RHITestResult::fail("Connected FinalBlit viewport resize failed: " + preview.lastLog());
        }
        const uint32_t sourceId = graph.findNode("Source")->id;
        graph.setNodeProperties(sourceId, {{"integer", true}});
        result = preview.render(graph, 29, 19);
        if (!result || !isUv(preview)) {
            return RHITestResult::fail("Integer RT did not use UV fallback: " + preview.lastLog());
        }
        graph.removeEdge(edgeId);
        result = preview.render(graph, 29, 19);
        if (!result || !isUv(preview)) {
            return RHITestResult::fail("Disconnect did not restore UV: " + preview.lastLog());
        }
        graph.removeNode(sourceId);
        result = preview.render(graph, 1, 1);
        if (!result || !isUv(preview)) {
            return RHITestResult::fail("Single-pixel UV fallback failed: " + preview.lastLog());
        }
        return RHITestResult::pass();
    }

private:
    static bool channelNear(uint32_t pixel, uint32_t shift, float expected)
    {
        return std::abs(static_cast<int>((pixel >> shift) & 255u) -
            static_cast<int>(std::lround(expected * 255.0f))) <= 1;
    }

    static bool isUv(const render::RenderGraphPreviewRenderer& preview)
    {
        if (preview.pixels().size() != size_t(preview.width()) * preview.height()) {
            return false;
        }
        for (uint32_t y = 0; y < preview.height(); ++y) {
            for (uint32_t x = 0; x < preview.width(); ++x) {
                const uint32_t pixel = preview.pixels()[size_t(y) * preview.width() + x];
                if (!channelNear(pixel, 0, (x + 0.5f) / preview.width()) ||
                    !channelNear(pixel, 8, (y + 0.5f) / preview.height()) ||
                    (pixel & 0xffff0000u) != 0xff000000u) {
                    return false;
                }
            }
        }
        return true;
    }

    static bool isSolid(const render::RenderGraphPreviewRenderer& preview)
    {
        if (preview.pixels().empty()) {
            return false;
        }
        for (uint32_t pixel : preview.pixels()) {
            if (!channelNear(pixel, 0, 0.2f) || !channelNear(pixel, 8, 0.4f) ||
                !channelNear(pixel, 16, 0.8f) || (pixel >> 24) != 255u) {
                return false;
            }
        }
        return true;
    }
};

class FinalBlitPipelinesTest : public RHITest {
public:
    FinalBlitPipelinesTest()
    {
        type = RHITestType::Rendering;
        name = "render_graph_final_blit_pipelines";
    }

    RHITestResult run(RHITestContext&) override
    {
        const std::unordered_map<std::string, std::string> expectedSources{
            {"default.metallic_graph.json", "PathTrace.color"},
            {"hdr_calibration.metallic_graph.json", ""},
            {"material_shader_object.metallic_graph.json", "MaterialScene.color"},
            {"gpu_driven_sponza.metallic_graph.json", "GPUDriven.color"},
            {"gpu_driven_realtime.metallic_graph.json", "DLSSSR.color"},
            {"gpu_driven_tessellation.metallic_graph.json", "DLSSSR.color"},
            {"gpu_driven_zorah_full.metallic_graph.json", "DLSSSR.color"},
            {"gpu_driven_minizorah.metallic_graph.json", "GPUDriven.color"},
            {"gpu_driven_minizorah_vbuffer.metallic_graph.json", "MaterialResolve.color"},
            {"gpu_driven_sponza_rtas_visualization.metallic_graph.json", "GPUDriven.color"},
            {"gpu_driven_sponza_streamasset.metallic_graph.json", "GPUDriven.color"},
            {"gpu_driven_terrain_p0_streamasset.metallic_graph.json", "GPUDriven.color"},
            {"gpu_driven_terrain_p1_unified.metallic_graph.json", "GPUDriven.color"},
            {"light_grid_debug.metallic_graph.json", "LightGridDebug.color"},
            {"openpbr_lookdev.metallic_graph.json", "PathTrace.color"},
            {"lookdev_shading_compare.metallic_graph.json", "Slider.color"},
            {"lookdev_vbuffer.metallic_graph.json", "Slider.color"},
            {"lookdev_abeautiful_game.metallic_graph.json", "Slider.color"},
            {"material_visualization_abeautiful_game.metallic_graph.json", "MaterialViz.color"},
            {"pathtracing_abeautiful_game_openpbr.metallic_graph.json", "PathTrace.color"},
            {"pathtracing_abeautiful_game_openpbr_dlss_rr.metallic_graph.json", "DLSSRR.color"},
            {"pathtracing_abeautiful_game_openpbr_dlss_nr.metallic_graph.json", "DLSSRR.color"},
            {"pathtracing_abeautiful_game_openpbr_dlss_sr.metallic_graph.json", "DLSSSR.color"},
            {"pathtracing_meet_mat.metallic_graph.json", "PathTrace.color"},
            {"pathtracing_meet_mat_nrc.metallic_graph.json", "PathTrace.color"},
            {"pathtracing_meet_mat_sharc.metallic_graph.json", "PathTrace.color"},
            {"rtxcr_material_showcase.metallic_graph.json", "PathTrace.color"},
            {"realtime_lighting.metallic_graph.json", "DLSSSR.color"},
            {"rtxdi_meet_mat.metallic_graph.json", "Composite.color"},
        };
        std::string log;
        size_t graphCount = 0;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(PROJECT_SOURCE_DIR "/Pipelines")) {
            const std::string filename = entry.path().filename().string();
            if (!entry.is_regular_file() || !filename.ends_with(".metallic_graph.json")) {
                continue;
            }
            render::RenderGraph graph;
            if (!render::loadRenderGraphFromFile(entry.path(), graph, log) || !graph.validate(log)) {
                return RHITestResult::fail(filename + ": " + log);
            }
            const auto expected = expectedSources.find(filename);
            if (expected == expectedSources.end() || !hasFinalOutput(graph, expected->second)) {
                return RHITestResult::fail(filename + " must present its final color through FinalBlit");
            }
            ++graphCount;
        }
        if (graphCount != expectedSources.size()) {
            return RHITestResult::fail("Not all expected pipeline assets were checked");
        }
        for (const render::RenderSampleDesc& desc : render::listBuiltInRenderSamples()) {
            render::RenderSampleLoadResult sample;
            if (!render::loadBuiltInRenderSample(desc.id, sample, log)) {
                return RHITestResult::fail(desc.id + ": " + log);
            }
            const auto expected = expectedSources.find(std::filesystem::path(desc.graphPath).filename().string());
            if (desc.previewOutput != "FinalBlit.color" || sample.desc.previewOutput != "FinalBlit.color" ||
                expected == expectedSources.end() || !hasFinalOutput(sample.graph, expected->second)) {
                return RHITestResult::fail(desc.id + " bypasses FinalBlit presentation");
            }
        }
        return RHITestResult::pass("All pipeline assets and built-in Samples present through FinalBlit");
    }

private:
    static bool hasFinalOutput(const render::RenderGraph& graph, std::string_view sourceOutput)
    {
        const auto* exposure = graph.findNode("AutoExposure");
        bool physical = false;
        for (const auto& node : graph.nodes()) {
            if (node.type == "ScenePathTracePass" || node.type == "SceneRealtimeLightingPass" ||
                node.type == "VisibilityBufferDeferredPass" ||
                node.type == "SceneRTXDIPass" || node.type == "RTXDICompositePass") {
                physical = true;
                if (!node.properties.value("outputLinear", false)) { return false; }
            }
        }
        if (physical != (exposure != nullptr)) { return false; }
        if (exposure != nullptr) {
            if (exposure->type != "AutoExposurePass") { return false; }
            size_t hdrConnections = 0;
            for (const auto& edge : graph.edges()) {
                if (edge.dstPass == exposure->name && edge.dstField == "source") {
                    if (render::makeRenderGraphFieldName(edge.srcPass, edge.srcField) != sourceOutput) { return false; }
                    ++hdrConnections;
                }
            }
            if (hdrConnections != 1) { return false; }
            sourceOutput = "AutoExposure.color";
        }
        if (const auto* nr = graph.findNode("DLSSNR"); nr != nullptr) {
            if (nr->type != "DLSSNRPass") { return false; }
            size_t colorConnections = 0;
            for (const auto& edge : graph.edges()) {
                if (edge.dstPass == nr->name && edge.dstField == "inputColor") {
                    if (render::makeRenderGraphFieldName(edge.srcPass, edge.srcField) != sourceOutput) { return false; }
                    ++colorConnections;
                }
            }
            if (colorConnections != 1) { return false; }
            sourceOutput = "DLSSNR.color";
        }
        const render::RenderGraphNode* final = graph.findNode("FinalBlit");
        if (final == nullptr || final->type != "FinalBlitPass" || !graph.outputs().empty() ||
            graph.firstOutputName() != "FinalBlit.color") {
            return false;
        }
        if (physical) {
            const auto* grading = graph.findNode("ColorGrading");
            if (!grading || grading->type != "ColorGradingLUTPass" ||
                grading->properties.value("toneCurve", "aces2") != "aces2") { return false; }
            if (std::none_of(graph.edges().begin(), graph.edges().end(), [](const auto& edge) {
                return edge.srcPass == "ColorGrading" && edge.srcField == "lut" &&
                    edge.dstPass == "FinalBlit" && edge.dstField == "lut";
            })) { return false; }
        }
        size_t connections = 0;
        for (const render::RenderGraphEdge& edge : graph.edges()) {
            if (edge.dstPass == final->name && edge.dstField == "source") {
                if (render::makeRenderGraphFieldName(edge.srcPass, edge.srcField) != sourceOutput) {
                    return false;
                }
                ++connections;
            }
        }
        return sourceOutput.empty() ? connections == 0 && final->properties.value("calibrationPattern", false)
            : connections == 1;
    }
};

METALLIC_REGISTER_RHI_TEST(FinalBlitPipelinesTest);
METALLIC_REGISTER_RHI_TEST(FinalBlitGraphTest);
METALLIC_REGISTER_RHI_TEST(FinalBlitPixelsTest);

} // namespace
} // namespace metallic::tests

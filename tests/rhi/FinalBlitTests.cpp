#include "RhiTest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <cmath>

namespace metallic::tests {
namespace {

class FinalBlitTestSource final : public render::RasterPass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureOutput("color").texture2D(17, 9).format =
            properties().value("integer", false) ? render::Format::R32Uint : render::Format::Rgba16Sfloat;
        return reflection;
    }

    render::Result execute(render::RenderGraphExecutionContext& context) override
    {
        const render::TextureHandle color = context.outputTexture("color");
        render::RenderingAttachmentDesc attachment{
            .view = color.view(),
            .state = render::ResourceState::ColorAttachment,
            .loadOp = render::LoadOp::Clear,
            .storeOp = render::StoreOp::Store,
            .clearColor = render::ColorValue{0.2f, 0.4f, 0.8f, 0.0f},
        };
        context.commandBuffer().beginRendering(render::RenderingDesc{
            .renderArea = render::Rect{0, 0, context.width(), context.height()},
            .colorAttachments = &attachment,
            .colorAttachmentCount = 1,
        });
        context.commandBuffer().endRendering();
        return {};
    }
};

class FinalBlitGraphTest : public RhiTest {
public:
    FinalBlitGraphTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_final_blit_contract";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::RenderGraph graph;
        const uint32_t finalId = graph.addNode("FinalBlitPass", "FinalBlit")->id;
        std::string log;
        if (!graph.validate(log) || !graph.outputs().empty() ||
            graph.firstOutputName() != "FinalBlit.color") {
            return RhiTestResult::fail("FinalBlit must work without marked outputs: " + log);
        }
        render::RenderGraph roundTrip;
        if (!render::deserializeRenderGraphFromString(render::serializeRenderGraphToString(graph), roundTrip, log) ||
            roundTrip.firstOutputName() != "FinalBlit.color" || !roundTrip.outputs().empty()) {
            return RhiTestResult::fail("FinalBlit presentation did not survive serialization: " + log);
        }
        graph.addNode("ClearColorPass", "Source");
        graph.addEdge("Source.color", "FinalBlit.source");
        // No extraOutputs: the executor itself must retain the presentation root.
        std::unique_ptr<render::Device> device;
        const render::Result deviceResult = render::createDevice(render::DeviceDesc{
            .applicationName = "FinalBlit Test",
            .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true,
        }, device);
        if (!deviceResult) {
            return render::hasError(deviceResult, render::Error::Unsupported)
                ? RhiTestResult::skip(toString(deviceResult))
                : RhiTestResult::fail(toString(deviceResult));
        }
        render::RenderGraphExecutor executor;
        if (!executor.compile(*device, graph, 63, 37, log) ||
            executor.outputResource("FinalBlit.color") == nullptr ||
            executor.outputResource("Source.color") == nullptr) {
            return RhiTestResult::fail("FinalBlit or its producer was culled: " + log);
        }
        const auto* output = executor.outputResource("FinalBlit.color");
        if (!render::hasFlag(output->desc.usage, render::TextureUsageBits::Sampled) ||
            !render::hasFlag(output->desc.usage, render::TextureUsageBits::TransferSource)) {
            return RhiTestResult::fail("Automatic presentation lacks display/readback access");
        }
        graph.markOutput("Source.color");
        if (graph.firstOutputName() != "FinalBlit.color") {
            return RhiTestResult::fail("Legacy marked output overrides presentation");
        }
        graph.renameNode(finalId, "Present");
        if (graph.firstOutputName() != "Present.color" || !graph.validate(log)) {
            return RhiTestResult::fail("Presentation rename broke graph: " + log);
        }
        const uint32_t duplicateId = graph.addNode("FinalBlitPass", "Duplicate")->id;
        if (graph.validate(log) || log.find("multiple presentation") == std::string::npos) {
            return RhiTestResult::fail("Ambiguous presentation outputs were accepted");
        }
        graph.removeNode(duplicateId);
        graph.removeNode(finalId);
        if (!graph.validate(log) || graph.firstOutputName() != "Source.color") {
            return RhiTestResult::fail("Removing FinalBlit did not restore legacy output");
        }
        return RhiTestResult::pass();
    }
};

class FinalBlitPixelsTest : public RhiTest {
public:
    FinalBlitPixelsTest()
    {
        type = RhiTestType::Rendering;
        name = "render_graph_final_blit_pixels";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        render::registerRenderGraphPassType("FinalBlitTestSource", "Test source",
            [] { return std::make_unique<FinalBlitTestSource>(); });
        render::RenderGraphPreviewRenderer preview;
        render::Result result = preview.initialize(context.enableValidation);
        if (!result) {
            return RhiTestResult::skip("Preview device unavailable: " + std::string(toString(result)));
        }
        render::RenderGraph graph;
        graph.addNode("FinalBlitPass", "FinalBlit");
        result = preview.render(graph, 63, 37);
        if (!result || !isUv(preview)) {
            return RhiTestResult::fail("Unconnected FinalBlit did not display UV: " + preview.lastLog());
        }
        std::string log;
        if (!saveRgba8Png(context.outputDirectory / "final_blit_uv.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                preview.width(), preview.height(), log)) {
            return RhiTestResult::fail(log);
        }

        graph.addNode("FinalBlitTestSource", "Source");
        const uint32_t edgeId = graph.addEdge("Source.color", "FinalBlit.source")->id;
        result = preview.render(graph, 63, 37);
        if (!result || preview.width() != 63 || preview.height() != 37 || !isSolid(preview)) {
            return RhiTestResult::fail("Float RT blit/resize/opaque alpha failed: " + preview.lastLog());
        }
        if (!saveRgba8Png(context.outputDirectory / "final_blit_connected.png",
                reinterpret_cast<const uint8_t*>(preview.pixels().data()),
                preview.width(), preview.height(), log)) {
            return RhiTestResult::fail(log);
        }
        result = preview.render(graph, 29, 19);
        if (!result || preview.width() != 29 || preview.height() != 19 || !isSolid(preview)) {
            return RhiTestResult::fail("Connected FinalBlit viewport resize failed: " + preview.lastLog());
        }
        const uint32_t sourceId = graph.findNode("Source")->id;
        graph.setNodeProperties(sourceId, {{"integer", true}});
        result = preview.render(graph, 29, 19);
        if (!result || !isUv(preview)) {
            return RhiTestResult::fail("Integer RT did not use UV fallback: " + preview.lastLog());
        }
        graph.removeEdge(edgeId);
        result = preview.render(graph, 29, 19);
        if (!result || !isUv(preview)) {
            return RhiTestResult::fail("Disconnect did not restore UV: " + preview.lastLog());
        }
        graph.removeNode(sourceId);
        result = preview.render(graph, 1, 1);
        if (!result || !isUv(preview)) {
            return RhiTestResult::fail("Single-pixel UV fallback failed: " + preview.lastLog());
        }
        return RhiTestResult::pass();
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

METALLIC_REGISTER_RHI_TEST(FinalBlitGraphTest);
METALLIC_REGISTER_RHI_TEST(FinalBlitPixelsTest);

} // namespace
} // namespace metallic::tests

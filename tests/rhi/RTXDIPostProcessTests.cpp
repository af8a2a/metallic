#include "RHITest.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace metallic::tests {
namespace {

class RTXDIPostProcessSource final : public render::UnsafePass {
public:
    inline static float signal = 1.0f;
    inline static float inverseScale = 1.0f;

    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        for (const auto* name : {"diffuse", "specular", "motion"}) {
            reflection.addTextureOutput(name).transferWrite().format = render::Format::RGBA16Sfloat;
        }
        reflection.addTextureOutput("emissive").transferWrite().format = render::Format::RGBA32Sfloat;
        reflection.addTextureOutput("base").transferWrite().format = render::Format::RGBA8Unorm;
        return reflection;
    }

    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        for (const auto* name : {"diffuse", "specular", "motion", "emissive", "base"}) {
            const std::string_view field(name);
            const float value = field == "base" ? 1.0f : field == "motion" ? 0.0f
                : field == "emissive" ? 0.25f : signal / inverseScale;
            const float alpha = field == "emissive" ? inverseScale : 0.0f;
            if (auto commandResult = context.commandBuffer().clearColorTexture(*context.outputTexture(name).texture(),
                render::TextureLayout::TransferDestination, {value, value, value, alpha}); !commandResult) { return commandResult; }
        }
        return {};
    }
};

class RTXDIPostProcessPixelsTest final : public RHITest {
public:
    RTXDIPostProcessPixelsTest() { type = RHITestType::Rendering; name = "rtxdi_typed_post_process_pixels_history"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::registerRenderGraphPassType("RTXDIPostProcessSource", "RTXDI test inputs",
            [] { return std::make_unique<RTXDIPostProcessSource>(); });
        render::RenderGraph graph;
        graph.addNode("RTXDIPostProcessSource", "Source");
        const auto confidenceId = graph.addNode("RTXDIConfidencePass", "Confidence",
            {{"confidenceHistoryLength", 0.0}})->id;
        graph.addNode("RTXDICompositePass", "Composite");
        graph.markOutput("Composite.color");
        for (const auto& edge : {std::pair{"Source.diffuse", "Confidence.noisyDiffuse"},
                {"Source.specular", "Confidence.noisySpecular"}, {"Source.base", "Confidence.baseColorMetalness"},
                {"Source.motion", "Confidence.motionVectors"}, {"Source.diffuse", "Composite.denoisedDiffuse"},
                {"Source.specular", "Composite.denoisedSpecular"}, {"Source.base", "Composite.baseColorMetalness"},
                {"Source.emissive", "Composite.emissive"}}) {
            graph.addEdge(edge.first, edge.second);
        }
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, false, false);
        if (!result) { return RHITestResult::fail(toString(result)); }
        preview.setRawReadbackEnabled(true);
        preview.setEnvironment({.enabled = false});
        RTXDIPostProcessSource::signal = 1.0f;
        RTXDIPostProcessSource::inverseScale = 1.0f;
        result = preview.render(graph, 31, 19, "Composite.color");
        if (!result) { return RHITestResult::fail(preview.lastLog()); }
        const auto& color = preview.readbackBytes();
        if (color.size() != 31u * 19u * 16u) { return RHITestResult::fail("Composite readback size"); }
        for (size_t i = 0; i < color.size(); i += 16) {
            float channels[4];
            std::memcpy(channels, color.data() + i, sizeof(channels));
            for (uint32_t channel = 0; channel < 4; ++channel) {
                if (std::abs(channels[channel] - (channel == 3 ? 1.0f : 1.29f)) > 0.0001f) {
                    return RHITestResult::fail("Composite differs from diffuse + F0 * specular + emissive");
                }
            }
        }
        for (const char* output : {"Confidence.diffuseConfidence", "Confidence.specularConfidence"}) {
            for (uint32_t filters : {0u, 1u, 2u, 5u, 6u}) {
                graph.setNodeRuntimeProperty(confidenceId, "gradientFilterPasses", filters);
                graph.setNodeRuntimeProperty(confidenceId, "resetSerial", filters + 1u);
                RTXDIPostProcessSource::signal = 1.0f;
                // Reset, steady history, changed signal, explicit reset.
                for (uint32_t frame = 0; frame < 4; ++frame) {
                    if (frame == 2) { RTXDIPostProcessSource::signal = 4.0f; }
                    if (frame == 3) { graph.setNodeRuntimeProperty(confidenceId, "resetSerial", 100u + filters); }
                    result = preview.render(graph, 31u + filters, 19, output);
                    if (!result) { return RHITestResult::fail(preview.lastLog()); }
                    const auto& bytes = preview.readbackBytes();
                    if (bytes.size() != (31u + filters) * 19u) { return RHITestResult::fail("Confidence readback size"); }
                    size_t pixelIndex = 0;
                    for (const auto value : bytes) {
                        const auto confidence = std::to_integer<uint8_t>(value);
                        // Stable absolute specular luminance is compared against 32F history.
                        if ((frame == 2 && confidence > 2) || (frame == 1 && confidence < 253) ||
                            ((frame == 0 || frame == 3) && confidence != 255)) {
                            return RHITestResult::fail("Confidence history/filter mismatch at frame " + std::to_string(frame)
                                + ", filters " + std::to_string(filters) + ", value " + std::to_string(confidence)
                                + ", pixel " + std::to_string(pixelIndex) + ", output " + output);
                        }
                        ++pixelIndex;
                    }
                }
            }
        }
        return RHITestResult::pass("Exact composite; both confidence outputs, history/reset, 0/odd/even filters and NPOT resize");
    }
};
METALLIC_REGISTER_RHI_TEST(RTXDIPostProcessPixelsTest);

class RTXDIConfidenceExposureDomainTest final : public RHITest {
public:
    RTXDIConfidenceExposureDomainTest()
    {
        type = RHITestType::Rendering;
        name = "rtxdi_confidence_pre_exposure_domain";
    }

    RHITestResult run(RHITestContext& context) override
    {
        render::registerRenderGraphPassType("RTXDIPostProcessSource", "RTXDI test inputs",
            [] { return std::make_unique<RTXDIPostProcessSource>(); });
        std::array<std::vector<std::byte>, 2> changedConfidence;
        for (const char* output : {"Confidence.diffuseConfidence", "Confidence.specularConfidence"}) {
            for (size_t exposureCase = 0; exposureCase < 2; ++exposureCase) {
                render::RenderGraph graph;
                graph.addNode("RTXDIPostProcessSource", "Source");
                graph.addNode("RTXDIConfidencePass", "Confidence",
                    {{"confidenceHistoryLength", 0.0}, {"gradientFilterPasses", 0}});
                graph.markOutput(output);
                for (const auto& edge : {std::pair{"Source.diffuse", "Confidence.noisyDiffuse"},
                        {"Source.specular", "Confidence.noisySpecular"},
                        {"Source.base", "Confidence.baseColorMetalness"},
                        {"Source.motion", "Confidence.motionVectors"},
                        {"Source.emissive", "Confidence.emissive"}}) {
                    graph.addEdge(edge.first, edge.second);
                }
                render::RenderGraphPreviewRenderer preview;
                auto result = preview.initialize(context.enableValidation, false, false);
                if (!result) { return RHITestResult::fail(toString(result)); }
                preview.setRawReadbackEnabled(true);
                RTXDIPostProcessSource::inverseScale = exposureCase == 0 ? 1.0f : 1024.0f;
                // Powers of two make the two 16F input representations exact.
                // The bias/epsilon must still use absolute luminance for dark signals.
                for (uint32_t frame = 0; frame < 4; ++frame) {
                    RTXDIPostProcessSource::signal = frame < 2 ? 0.001953125f : 0.0078125f;
                    if (frame == 3) { RTXDIPostProcessSource::inverseScale = 1.0f; }
                    result = preview.render(graph, 27, 15, output);
                    if (!result) { return RHITestResult::fail(preview.lastLog()); }
                    const auto& bytes = preview.readbackBytes();
                    if (bytes.size() != 27u * 15u) { return RHITestResult::fail("Confidence exposure readback size"); }
                    if (frame == 2) {
                        changedConfidence[exposureCase] = bytes;
                    } else if (std::any_of(bytes.begin(), bytes.end(), [](std::byte value) {
                        return std::to_integer<uint8_t>(value) < 253;
                    })) {
                        return RHITestResult::fail("Unchanged absolute lighting lost confidence after a scale change");
                    }
                }
                RTXDIPostProcessSource::inverseScale = 1024.0f;
                // Restored luminance above the 16F maximum needs unclamped 32F history.
                for (uint32_t frame = 0; frame < 3; ++frame) {
                    RTXDIPostProcessSource::signal = frame < 2 ? 1048576.0f : 4194304.0f;
                    result = preview.render(graph, 27, 15, output);
                    if (!result) { return RHITestResult::fail(preview.lastLog()); }
                    if (frame == 1 || frame == 2) {
                        for (const auto value : preview.readbackBytes()) {
                            const auto confidence = std::to_integer<uint8_t>(value);
                            if ((frame == 1 && confidence < 253) || (frame == 2 && confidence > 2)) {
                                return RHITestResult::fail("Confidence truncated absolute HDR history/gradients to 16F");
                            }
                        }
                    }
                }
            }
            if (changedConfidence[0] != changedConfidence[1]) {
                return RHITestResult::fail("Dark lighting confidence changes under equivalent noisy pre-exposure");
            }
            if (std::all_of(changedConfidence[0].begin(), changedConfidence[0].end(), [](std::byte value) {
                return std::to_integer<uint8_t>(value) > 250;
            })) {
                return RHITestResult::fail("Changed dark lighting incorrectly retained full history confidence");
            }
        }
        RTXDIPostProcessSource::inverseScale = 1.0f;
        return RHITestResult::pass("Diffuse/specular confidence preserves darkness thresholds and HDR history across pre-exposure changes");
    }
};
METALLIC_REGISTER_RHI_TEST(RTXDIConfidenceExposureDomainTest);

} // namespace
} // namespace metallic::tests

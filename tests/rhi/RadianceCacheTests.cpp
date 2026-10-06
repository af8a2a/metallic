#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "RHITest.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/PathTraceStageParameters.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include <cstring>

#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <cmath>
#include <string>
#include <vector>

namespace metallic::tests {
namespace {

// Deterministic inputs exercise the NRC output kernel without requiring the NRC SDK.
class PathTraceTonemapSource final : public render::UnsafePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        for (const auto* name : {"source", "previous"}) {
            reflection.addTextureOutput(name).transferWrite().format = render::Format::RGBA32Sfloat;
        }
        return reflection;
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        if (auto commandResult = context.commandBuffer().clearColorTexture(*context.outputTexture("source").texture(),
            render::TextureLayout::TransferDestination, {4.0f, 2.0f, 1.0f, 0.0f}); !commandResult) { return commandResult; }
        if (auto commandResult = context.commandBuffer().clearColorTexture(*context.outputTexture("previous").texture(),
            render::TextureLayout::TransferDestination, {0.0f, 2.0f, 3.0f, 0.0f}); !commandResult) { return commandResult; }
        return {};
    }
};

class PathTraceTonemapProbe final : public render::ComputePass {
public:
    render::RenderPassReflection reflect(const render::RenderGraphCompileContext&) const override
    {
        render::RenderPassReflection reflection;
        reflection.addTextureInput("source").storageReadWrite().format = render::Format::RGBA32Sfloat;
        reflection.addTextureInput("previous").storageRead().format = render::Format::RGBA32Sfloat;
        reflection.addTextureOutput("color").storageWrite().format = render::Format::RGBA32Sfloat;
        return reflection;
    }
    render::Result<> compile(const render::RenderGraphCompileContext& context, std::string& log) override
    {
        using namespace render;
        device_ = context.device;
        auto shader = compileSlangShaderToSpirv({.moduleName = "Features/PostProcess/ScenePathTraceTonemap",
            .entryPointName = "scenePathTraceTonemapMain", .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, log);
        if (!shader) { return makeError(shader.error()); }
        return kernel_.initialize(*device_, {.spirv = shader->spirv,
            .parameters = parameterAbi<PathTraceTonemapParams>(kPathTraceTonemapABI, ParameterTransport::InlinePush)}, log);
    }
    render::Result<> execute(render::RenderGraphExecutionContext& context) override
    {
        using namespace render;
        auto registry = metallic::render::ResourceRegistry::forDevice(*device_);
        if (!registry) { return makeError(registry.error()); }
        ParameterWriter writer(*device_, **registry, metallic::render::RenderFrameContext::from(context.commandBuffer()));
        const PathTraceTonemapParams params{
            .source = writer.storageImage(context.inputTexture("source").view()),
            .output = writer.storageImage(context.outputTexture("color").view()),
            .historyPrevious = writer.storageImage(context.inputTexture("previous").view()),
            .settings = {context.width(), context.height(), 2.0f,
                context.properties().value("linear", 1u), context.properties().value("history", 0u), 3u},
        };
        auto encoded = writer.encode(params, kPathTraceTonemapABI, ParameterTransport::InlinePush);
        if (!encoded) { return makeError(encoded.error()); }
        return kernel_.dispatch(context.commandBuffer(), *encoded, (context.width() + 7) / 8, (context.height() + 7) / 8);
    }
private:
    render::Device* device_ = nullptr;
    render::ComputeKernel kernel_;
};

class PathTraceTonemapPixelsTest final : public RHITest {
public:
    PathTraceTonemapPixelsTest() { type = RHITestType::Rendering; name = "path_trace_typed_tonemap_pixels_history"; }
    RHITestResult run(RHITestContext& context) override
    {
        render::registerRenderGraphPassType("PathTraceTonemapSource", "Path trace test inputs",
            [] { return std::make_unique<PathTraceTonemapSource>(); });
        render::registerRenderGraphPassType("PathTraceTonemapProbe", "Path trace output probe",
            [] { return std::make_unique<PathTraceTonemapProbe>(); });
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, false, false);
        if (!result) { return RHITestResult::fail(toString(result)); }
        preview.setRawReadbackEnabled(true);
        preview.setEnvironment({.enabled = false});
        for (uint32_t linear : {0u, 1u}) {
            for (uint32_t history : {0u, 1u}) {
                render::RenderGraph graph;
                graph.addNode("PathTraceTonemapSource", "Source");
                graph.addNode("PathTraceTonemapProbe", "Output", {{"linear", linear}, {"history", history}});
                graph.addEdge("Source.source", "Output.source");
                graph.addEdge("Source.previous", "Output.previous");
                graph.markOutput("Output.color");
                for (const char* output : {"Output.color", "Source.source"}) {
                    result = preview.render(graph, 31, 19, output);
                    if (!result) { return RHITestResult::fail(preview.lastLog()); }
                    const auto& bytes = preview.readbackBytes();
                    if (bytes.size() != 31u * 19u * 16u) { return RHITestResult::fail("Tonemap readback size"); }
                    const float expected[] = {history ? 1.0f : 4.0f, 2.0f, history ? 2.5f : 1.0f, 1.0f};
                    for (size_t i = 0; i < bytes.size(); i += 16) {
                        float channels[4];
                        std::memcpy(channels, bytes.data() + i, sizeof(channels));
                        for (uint32_t channel = 0; channel < 4; ++channel) {
                            float value = expected[channel];
                            if (!linear && channel < 3 && std::string_view(output) == "Output.color") {
                                value = std::pow(value * 2.0f / (value * 2.0f + 1.0f), 1.0f / 2.2f);
                            }
                            if (!std::isfinite(channels[channel]) || std::abs(channels[channel] - value) > 0.0001f) {
                                return RHITestResult::fail(std::string(output) + ": tonemap/history pixel mismatch");
                            }
                        }
                    }
                }
            }
        }
        return RHITestResult::pass("NPOT linear/tonemapped output, history blending and linear history writeback");
    }
};
METALLIC_REGISTER_RHI_TEST(PathTraceTonemapPixelsTest);

// With one surface bounce, the caches must preserve the directly visible
// surface's lighting instead of replacing it with a voxel or an NRC prediction.
class RadianceCacheLightingTest : public RHITest {
public:
    RadianceCacheLightingTest()
    {
        type = RHITestType::Rendering;
        name = "radiance_cache_lighting";
    }

    RHITestResult run(RHITestContext& context) override
    {
        // Match the editor: switch graphs while retaining one Vulkan device.
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, true);
        if (!result) {
            return RHITestResult::skip("Ray-query preview is unavailable");
        }
        // Compare cache lighting with identical exposure. Wall-clock adaptation
        // makes the 64-frame result depend on shader/PSO compilation delays.
        scene::LightingSettings lighting;
        lighting.autoExposure.enabled = false;
        lighting.exposureEV100 = 4.0f;
        preview.setLighting(lighting);
        for (uint32_t maxDepth : {1u, 3u}) {
            const auto tested = runDepth(context, preview, maxDepth);
            if (!tested.passed) {
                return tested;
            }
        }
        return RHITestResult::pass();
    }

private:
    RHITestResult runDepth(RHITestContext& context,
        render::RenderGraphPreviewRenderer& preview, uint32_t maxDepth)
    {
        constexpr uint32_t kSize = 256;
        constexpr uint32_t kFrames = 64;
        std::vector<uint32_t> reference;
        std::string failures;
        for (const char* mode : {"off", "sharc", "nrc"}) {
#if !METALLIC_HAS_NRC
            if (std::string_view(mode) == "nrc") {
                continue;
            }
#endif
            preview.setEnvironment(render::EnvironmentSettings{
                .enabled = true,
                .path = PROJECT_SOURCE_DIR "/Asset/ABeautifulGame/environment.hdr",
                .intensity = 1.0f,
                .visible = false,
            });
            render::RenderGraph graph;
            graph.addNode("ScenePathTracePass", "PathTrace", {
                {"path", PROJECT_SOURCE_DIR "/Asset/meet_mat.glb"},
                {"cacheMode", mode},
                {"maxDepth", maxDepth},
                {"samples", 1},
                {"accumulate", true},
                {"outputLinear", true},
                {"camera", {
                    {"eye", {0.0f, 1.2525f, 4.0236f}},
                    {"center", {0.0f, 1.2525f, 0.0f}},
                    {"up", {0.0f, 1.0f, 0.0f}},
                    {"fovDegrees", 45.0f},
                }},
            });
            graph.addNode("AutoExposurePass", "Tonemap");
            graph.addEdge("PathTrace.color", "Tonemap.source");
            graph.markOutput("Tonemap.color");
            for (uint32_t frame = 0; frame < kFrames; ++frame) {
                const auto result = preview.render(graph, kSize, kSize);
                if (!result) {
                    return RHITestResult::fail(std::string(mode) + ": " + preview.lastLog());
                }
            }
            const auto& pixels = preview.pixels();
            std::string message;
            if (!saveRgba8Png(context.outputDirectory /
                    (std::string(name) + "_depth" + std::to_string(maxDepth) + "_" + mode + ".png"),
                    reinterpret_cast<const uint8_t*>(pixels.data()), kSize, kSize, message)) {
                return RHITestResult::fail(message);
            }
            if (reference.empty()) {
                reference = pixels;
                continue;
            }
            double squaredError = 0.0;
            double energy = 0.0;
            double referenceEnergy = 0.0;
            uint32_t channelCount = 0;
            for (size_t i = 0; i < pixels.size(); ++i) {
                // Exclude the dark background so it cannot mask a black object.
                if ((reference[i] & 0x00ffffffu) == 0) {
                    continue;
                }
                for (uint32_t shift : {0u, 8u, 16u}) {
                    const double a = double((pixels[i] >> shift) & 255u) / 255.0;
                    const double b = double((reference[i] >> shift) & 255u) / 255.0;
                    squaredError += (a - b) * (a - b);
                    energy += a;
                    referenceEnergy += b;
                    ++channelCount;
                }
            }
            if (channelCount < 3000 || referenceEnergy < 100.0) {
                return RHITestResult::fail("Reference surface is missing or unlit");
            }
            const double rmse = std::sqrt(squaredError / channelCount);
            const double ratio = energy / referenceEnergy;
            const double tolerance = maxDepth == 1 ? 0.08 : 0.12;
            if (rmse > tolerance || ratio < 0.85 || ratio > 1.15) {
                failures += std::string(mode) + " RMSE=" + std::to_string(rmse) +
                    " energy ratio=" + std::to_string(ratio) + "; ";
            }
        }
        return failures.empty() ? RHITestResult::pass() : RHITestResult::fail(failures);
    }
};

METALLIC_REGISTER_RHI_TEST(RadianceCacheLightingTest);

} // namespace
} // namespace metallic::tests

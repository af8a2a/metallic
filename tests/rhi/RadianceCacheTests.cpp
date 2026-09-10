#include "RhiTest.h"

#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <cmath>
#include <string>
#include <vector>

namespace metallic::tests {
namespace {

// With one surface bounce, the caches must preserve the directly visible
// surface's lighting instead of replacing it with a voxel or an NRC prediction.
class RadianceCacheLightingTest : public RhiTest {
public:
    RadianceCacheLightingTest()
    {
        type = RhiTestType::Rendering;
        name = "radiance_cache_lighting";
    }

    RhiTestResult run(RhiTestContext& context) override
    {
        // Match the editor: switch graphs while retaining one Vulkan device.
        render::RenderGraphPreviewRenderer preview;
        auto result = preview.initialize(context.enableValidation, true);
        if (!result) {
            return RhiTestResult::skip("Ray-query preview is unavailable");
        }
        for (uint32_t maxDepth : {1u, 3u}) {
            const auto tested = runDepth(context, preview, maxDepth);
            if (!tested.passed) {
                return tested;
            }
        }
        return RhiTestResult::pass();
    }

private:
    RhiTestResult runDepth(RhiTestContext& context,
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
                    return RhiTestResult::fail(std::string(mode) + ": " + preview.lastLog());
                }
            }
            const auto& pixels = preview.pixels();
            std::string message;
            if (!saveRgba8Png(context.outputDirectory /
                    (std::string(name) + "_depth" + std::to_string(maxDepth) + "_" + mode + ".png"),
                    reinterpret_cast<const uint8_t*>(pixels.data()), kSize, kSize, message)) {
                return RhiTestResult::fail(message);
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
                return RhiTestResult::fail("Reference surface is missing or unlit");
            }
            const double rmse = std::sqrt(squaredError / channelCount);
            const double ratio = energy / referenceEnergy;
            const double tolerance = maxDepth == 1 ? 0.08 : 0.12;
            if (rmse > tolerance || ratio < 0.85 || ratio > 1.15) {
                failures += std::string(mode) + " RMSE=" + std::to_string(rmse) +
                    " energy ratio=" + std::to_string(ratio) + "; ";
            }
        }
        return failures.empty() ? RhiTestResult::pass() : RhiTestResult::fail(failures);
    }
};

METALLIC_REGISTER_RHI_TEST(RadianceCacheLightingTest);

} // namespace
} // namespace metallic::tests

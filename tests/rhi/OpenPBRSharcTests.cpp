#include "RHITest.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderGraph/RenderGraphExecutionSnapshot.h"
#include "Runtime/Scene/SceneDocument.h"
#include "stb/stb_image_write.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {

using Json = render::RenderGraphProperties;

void requireSharc(bool condition, const std::string& message)
{
    if (!condition) { throw std::runtime_error(message); }
}

// Open-front diffuse room: primary visibility is unchanged by caching, while
// most secondary rays hit other walls. The constant HDRI gives a cheap,
// reproducible indirect-light workload and an exact lights-off oracle.
std::filesystem::path writeSharcRoom(const std::filesystem::path& root)
{
    using V = std::array<float, 3>;
    std::vector<V> vertices;
    const auto quad = [&](V a, V b, V c, V d) {
        for (const V& vertex : {a, b, c, a, c, d}) { vertices.push_back(vertex); }
    };
    quad({-1,-1,-1}, {1,-1,-1}, {1,1,-1}, {-1,1,-1});
    quad({-1,-1,-1}, {-1,-1,1}, {1,-1,1}, {1,-1,-1});
    quad({-1,1,-1}, {1,1,-1}, {1,1,1}, {-1,1,1});
    quad({-1,-1,1}, {-1,-1,-1}, {-1,1,-1}, {-1,1,1});
    quad({1,-1,-1}, {1,-1,1}, {1,1,1}, {1,1,-1});
    const size_t bytes = vertices.size() * sizeof(V);
    std::ofstream binary(root / "Room.bin", std::ios::binary);
    binary.write(reinterpret_cast<const char*>(vertices.data()), std::streamsize(bytes));
    requireSharc(bool(binary), "Write SHaRC room geometry");
    Json gltf{{"asset", {{"version", "2.0"}}}, {"scene", 0},
        {"scenes", {{{"nodes", {0}}}}}, {"nodes", {{{"mesh", 0}}}},
        {"buffers", {{{"uri", "Room.bin"}, {"byteLength", bytes}}}},
        {"bufferViews", {{{"buffer", 0}, {"byteLength", bytes}}}},
        {"accessors", {{{"bufferView", 0}, {"componentType", 5126}, {"count", vertices.size()},
            {"type", "VEC3"}, {"min", {-1,-1,-1}}, {"max", {1,1,1}}}}},
        {"materials", {{{"pbrMetallicRoughness", {{"baseColorFactor", {0.6,0.5,0.4,1}},
            {"metallicFactor", 0}, {"roughnessFactor", 1}}}}}},
        {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}}}, {"material", 0}}}}}}}};
    const auto path = root / "Room.gltf";
    std::ofstream file(path);
    file << gltf.dump(2);
    requireSharc(bool(file), "Write SHaRC room scene");
    return path;
}

class OpenPBRSharcTest final : public RHITest
{
public:
    OpenPBRSharcTest()
    {
        name = "openpbr_sharc_guides_and_history";
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        try {
            const auto root = std::filesystem::absolute(context.outputDirectory / "OpenPBRSharc");
            std::filesystem::create_directories(root);
            const auto scenePath = writeSharcRoom(root);
            std::vector<float> source(16 * 8 * 3, 1.0f);
            const auto environmentPath = root / "Constant.hdr";
            requireSharc(stbi_write_hdr(environmentPath.string().c_str(), 16, 8, 3, source.data()) != 0, "Write HDRI");
            scene::SceneDocument document;
            requireSharc(document.load(scenePath), document.lastLoadResult().error);
            constexpr uint32_t kSize = 96, kFrames = 48;
            std::array<std::vector<double>, 3> averages;
            std::array<std::vector<std::vector<std::byte>>, 3> guides;
            const std::array guideNames{"albedo", "specularAlbedo", "normalRoughness", "linearDepth",
                "motionVectors", "specularHitDistance", "depth"};
            Json report = Json::object();
            for (uint32_t mode = 0; mode < 3; ++mode) {
                const std::string label = std::array{"Off", "NoQuery", "Query"}[mode];
                RenderGraph graph;
                auto* node = graph.addNode("ScenePathTracePass", "PathTrace");
                requireSharc(node != nullptr, "Create path tracer");
                node->properties = {{"path", scenePath.string()}, {"bsdf", "openpbr"},
                    {"accumulate", false}, {"samples", 4}, {"maxDepth", 8}, {"outputLinear", true},
                    {"exportDenoiserGuides", true}, {"cacheMode", mode == 0 ? "off" : "sharc"},
                    {"sharc.entriesLog2", 16}, {"sharc.updateStride", 2}, {"sharc.updateMaxDepth", 8},
                    {"sharc.sceneScale", 16}, {"sharc.queryMinDepth", mode == 1 ? 32 : 1},
                    {"camera", {{"eye", {0,0,3.8}}, {"center", {0,0,-1}}, {"up", {0,1,0}},
                        {"fovDegrees", 40}, {"znear", 0.001}, {"zfar", 100}}}};
                const uint32_t nodeId = node->id;
                graph.markOutput("PathTrace.color");
                for (const char* name : guideNames) { graph.markOutput("PathTrace." + std::string(name)); }
                RenderGraphPreviewRenderer preview;
                preview.bindRuntimeScene(&document);
                auto environment = document.environment();
                environment.enabled = true; environment.visible = true; environment.path = environmentPath;
                environment.intensity = 1.0f; environment.sourceColorSpace = kACEScg;
                environment.sourceColorSpaceExplicit = true;
                preview.setEnvironment(environment);
                auto lighting = document.lighting();
                lighting.lights.clear(); lighting.autoExposure.enabled = false; lighting.exposureEV100 = 0;
                requireSharc(preview.setLighting(lighting), "Set lighting");
                environment::WorldEnvironment world;
                world.source = environment::EnvironmentSource::HDRI;
                world.sun.enabled = false; world.moon.enabled = false;
                requireSharc(preview.setWorldEnvironment(world), "Set world");
                preview.setRawReadbackEnabled(true);
                preview.setExecutionCaptureEnabled(true);
                const auto initialized = preview.initialize(context.enableValidation, true, false);
                if (hasError(initialized, Error::Unsupported)) { return RHITestResult::skip("SHaRC requires ray query and bindless"); }
                requireSharc(bool(initialized), "Initialize preview");
                auto& mean = averages[mode];
                mean.resize(kSize * kSize * 4, 0.0);
                std::vector<double> timings;
                const auto renderColor = [&] {
                    requireSharc(bool(preview.render(graph, kSize, kSize, "PathTrace.color")), preview.lastLog());
                    const auto& bytes = preview.readbackBytes();
                    requireSharc(preview.readbackFormat() == Format::RGBA32Sfloat && bytes.size() == mean.size() * sizeof(float), "HDR layout");
                    std::vector<float> values(mean.size());
                    std::memcpy(values.data(), bytes.data(), bytes.size());
                    for (float value : values) { requireSharc(std::isfinite(value) && value >= 0, "Invalid cache HDR"); }
                    return values;
                };
                const auto hasStage = [&](std::string_view name) {
                    const auto snapshot = preview.executionSnapshot();
                    if (!snapshot) { return false; }
                    for (const auto& pass : snapshot->passes) {
                        for (const auto& stage : pass.stages) { if (stage.name == name) { return true; } }
                    }
                    return false;
                };
                for (uint32_t frame = 0; frame < kFrames; ++frame) {
                    const auto values = renderColor();
                    if (mode != 0) {
                        requireSharc(hasStage("SHaRC update") && hasStage("SHaRC resolve") && hasStage("SHaRC query"), "Missing cache stages: " + preview.lastLog());
                        requireSharc(hasStage("SHaRC clear") == (frame == 0), "Cache unexpectedly cleared while RR accumulation is off");
                    }
                    const auto completed = preview.collectCompletedGpuExecutionStats();
                    requireSharc(bool(completed), "Read GPU timestamps");
                    if (frame >= 16) {
                        for (size_t i = 0; i < values.size(); ++i) { mean[i] += values[i] / double(kFrames - 16); }
                        for (const auto& stats : *completed) {
                            for (const auto& pass : stats.nodes) {
                                if (pass.name == "PathTrace" && pass.gpuTimingAvailable) { timings.push_back(pass.gpuMilliseconds); }
                            }
                        }
                    }
                }
                std::vector<float> meanFloat(mean.size());
                std::transform(mean.begin(), mean.end(), meanFloat.begin(), [](double value) { return static_cast<float>(value); });
                std::ofstream hdr(root / (label + ".rgba32f"), std::ios::binary);
                hdr.write(reinterpret_cast<const char*>(meanFloat.data()), std::streamsize(meanFloat.size() * sizeof(float)));
                std::string log;
                requireSharc(saveRgba8Png(root / (label + ".png"), reinterpret_cast<const uint8_t*>(preview.pixels().data()), kSize, kSize, log), log);
                for (const char* name : guideNames) {
                    requireSharc(bool(preview.render(graph, kSize, kSize, "PathTrace." + std::string(name))), preview.lastLog());
                    guides[mode].push_back(preview.readbackBytes());
                    requireSharc(!guides[mode].back().empty(), "Missing guide bytes");
                }
                if (mode != 0) {
                    // Changing the voxel scale must invalidate even with accumulate=false.
                    requireSharc(graph.setNodeRuntimeProperty(nodeId, "sharc.sceneScale", 24.0f), "Change grid scale");
                    renderColor();
                    requireSharc(hasStage("SHaRC clear"), "Grid change retained stale cache");
                    environment.intensity = 0;
                    preview.setEnvironment(environment);
                    const auto dark = renderColor();
                    requireSharc(hasStage("SHaRC clear"), "Environment change retained stale cache");
                    for (size_t i = 0; i < dark.size(); ++i) {
                        if (i % 4 != 3) { requireSharc(dark[i] < 1e-6f, "Stale lighting after environment disabled"); }
                    }
                    requireSharc(graph.setNodeRuntimeProperty(nodeId, "cacheMode", "off"), "Disable cache at runtime");
                    renderColor();
                    requireSharc(!hasStage("SHaRC query"), "Runtime cache switch did not select reference integrator");
                    requireSharc(graph.setNodeRuntimeProperty(nodeId, "cacheMode", "sharc"), "Enable cache at runtime");
                    renderColor();
                    requireSharc(hasStage("SHaRC clear") && hasStage("SHaRC query"), "Runtime cache switch reused stale history");
                }
                std::sort(timings.begin(), timings.end());
                report[label] = {{"gpuSamples", timings.size()}, {"pathTraceIncludingCacheMedianMs", timings.empty() ? 0.0 : timings[timings.size()/2]}};
            }
            double error = 0, magnitude = 0, referenceEnergy = 0, cacheEnergy = 0;
            for (size_t i = 0; i < averages[0].size(); ++i) {
                requireSharc(averages[0][i] == averages[1][i], "Cache miss path changed reference RNG or transport");
                if (i % 4 == 3) { continue; }
                const double difference = averages[2][i] - averages[0][i];
                error += difference * difference; magnitude += averages[0][i] * averages[0][i];
                referenceEnergy += averages[0][i]; cacheEnergy += averages[2][i];
            }
            for (size_t i = 0; i < guideNames.size(); ++i) {
                requireSharc(guides[0][i] == guides[1][i] && guides[0][i] == guides[2][i],
                    "Cache changed primary RR guide: " + std::string(guideNames[i]));
            }
            const double relativeRMSE = std::sqrt(error / std::max(magnitude, 1e-12));
            const double energyRatio = cacheEnergy / std::max(referenceEnergy, 1e-12);
            report["relativeRMSE"] = relativeRMSE;
            report["energyRatio"] = energyRatio;
            report["frames"] = kFrames; report["warmupFrames"] = 16;
            report["width"] = kSize; report["height"] = kSize;
            report["guidesByteIdentical"] = true; report["cacheMissByteIdentical"] = true;
            std::ofstream(root / "Report.json") << report.dump(2);
            requireSharc(relativeRMSE > 1e-5, "No observable cache queries; test did not exercise early termination");
            requireSharc(relativeRMSE < 0.25 && energyRatio > 0.85 && energyRatio < 1.15, "Cache radiance deviates from reference; see Report.json");
            return RHITestResult::pass("OpenPBR SHaRC GPU transport, cache-miss equivalence, seven exact RR guides, scale/environment invalidation; " + (root / "Report.json").string());
        } catch (const std::exception& exception) {
            return RHITestResult::fail(exception.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(OpenPBRSharcTest);

} // namespace
} // namespace metallic::tests

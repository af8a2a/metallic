#include "RHITest.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/RenderView.h"
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
std::filesystem::path writeSharcRoom(const std::filesystem::path& root, float metalness, float roughness)
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
            {"metallicFactor", metalness}, {"roughnessFactor", roughness}}}}}},
        {"meshes", {{{"primitives", {{{"attributes", {{"POSITION", 0}}}, {"material", 0}}}}}}}};
    const auto path = root / "Room.gltf";
    std::ofstream file(path);
    file << gltf.dump(2);
    requireSharc(bool(file), "Write SHaRC room scene");
    return path;
}

class OpenPBRSharcTest : public RHITest
{
public:
    explicit OpenPBRSharcTest(uint32_t material = 0) : material_(material)
    {
        name = std::array{"openpbr_sharc_guides_and_history", "openpbr_sharc_mixed_components", "openpbr_sharc_directional_metal"}[material];
        type = RHITestType::Rendering;
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        try {
            const auto root = std::filesystem::absolute(context.outputDirectory / name);
            std::filesystem::create_directories(root);
            const auto scenePath = writeSharcRoom(root, material_ == 2 ? 1.0f : 0.0f, material_ == 0 ? 1.0f : 0.6f);
            std::vector<float> source(16 * 8 * 3, 1.0f);
            if (material_ != 0) {
                // Angular color/energy variation exposes inappropriate reuse of
                // specular radiance from a different outgoing direction.
                for (uint32_t y = 0; y < 8; ++y) {
                    for (uint32_t x = 0; x < 16; ++x) {
                        const size_t i = (y * 16 + x) * 3;
                        source[i] = x < 8 ? 3.0f : 0.25f;
                        source[i + 1] = y < 4 ? 0.5f : 1.5f;
                        source[i + 2] = x < 8 ? 0.25f : 3.0f;
                    }
                }
            }
            const auto environmentPath = root / "Constant.hdr";
            requireSharc(stbi_write_hdr(environmentPath.string().c_str(), 16, 8, 3, source.data()) != 0, "Write HDRI");
            scene::SceneDocument document;
            requireSharc(document.load(scenePath), document.lastLoadResult().error);
            constexpr uint32_t kSize = 96, kFrames = 96;
            std::array<std::vector<double>, 5> averages, movingAverages;
            std::array<std::vector<std::vector<std::byte>>, 5> guides;
            const std::array guideNames{"albedo", "specularAlbedo", "normalRoughness", "linearDepth",
                "motionVectors", "specularHitDistance", "depth"};
            Json report = Json::object();
            for (uint32_t mode = 0; mode < averages.size(); ++mode) {
                const std::string label = std::array{"Off", "NoQuery", "Query", "DiffuseOnly", "SpecularOnly"}[mode];
                RenderGraph graph;
                auto* node = graph.addNode("ScenePathTracePass", "PathTrace");
                requireSharc(node != nullptr, "Create path tracer");
                node->properties = {{"path", scenePath.string()}, {"bsdf", "openpbr"},
                    {"accumulate", false}, {"samples", 4}, {"maxDepth", 8}, {"outputLinear", true},
                    {"exportDenoiserGuides", true}, {"cacheMode", mode == 0 ? "off" : "sharc"},
                    {"sharc.entriesLog2", 16}, {"sharc.updateStride", 2}, {"sharc.updateMaxDepth", 8},
                    {"sharc.sceneScale", 12}, {"sharc.queryMinDepth", mode == 1 ? 32 : 1},
                    {"sharc.lobeMask", mode == 3 ? 1 : mode == 4 ? 2 : 3},
                    {"camera", {{"eye", {0,0,3.8}}, {"center", {0,0,-1}}, {"up", {0,1,0}},
                        {"fovDegrees", 40}, {"znear", 0.001}, {"zfar", 100}}}};
                const uint32_t nodeId = node->id;
                graph.markOutput("PathTrace.color");
                for (const char* name : guideNames) { graph.markOutput("PathTrace." + std::string(name)); }
                RenderView view;
                view.setRenderResolution(kSize, kSize);
                view.setCameraProperties(node->properties.at("camera"));
                node->properties.erase("camera");
                RenderGraphPreviewRenderer preview;
                preview.bindRenderView(&view);
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
                    if (frame >= 32) {
                        for (size_t i = 0; i < values.size(); ++i) { mean[i] += values[i] / double(kFrames - 32); }
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
                auto& movingMean = movingAverages[mode];
                movingMean.resize(mean.size(), 0.0);
                for (uint32_t frame = 0; frame < 32; ++frame) {
                    auto camera = view.camera();
                    camera.eye[0] = 0.01f * float(frame + 1);
                    camera.eye[2] = 3.8f - 0.02f * float(frame + 1);
                    view.setCamera(camera);
                    const auto values = renderColor();
                    if (mode != 0) { requireSharc(!hasStage("SHaRC clear"), "Camera motion discarded cache history"); }
                    for (size_t i = 0; i < values.size(); ++i) { movingMean[i] += values[i] / 32.0; }
                }
                requireSharc(saveRgba8Png(root / (label + "Moving.png"), reinterpret_cast<const uint8_t*>(preview.pixels().data()), kSize, kSize, log), log);
                if (mode != 0) {
                    // Switching components invalidates both cache partitions.
                    requireSharc(graph.setNodeRuntimeProperty(nodeId, "sharc.lobeMask", 3), "Change cached components");
                    renderColor();
                    if (mode >= 3) { requireSharc(hasStage("SHaRC clear"), "Component change retained stale cache"); }
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
            for (size_t i = 0; i < averages[0].size(); ++i) {
                requireSharc(averages[0][i] == averages[1][i], "Cache miss path changed reference RNG or transport");
            }
            for (uint32_t mode = 1; mode < averages.size(); ++mode) {
                for (size_t i = 0; i < guideNames.size(); ++i) {
                    requireSharc(guides[0][i] == guides[mode][i], "Cache changed primary RR guide: " + std::string(guideNames[i]));
                }
            }
            bool accurate = true, queried = true;
            for (uint32_t mode = 2; mode < averages.size(); ++mode) {
                double error = 0, magnitude = 0, referenceEnergy = 0, cacheEnergy = 0;
                for (size_t i = 0; i < averages[0].size(); ++i) {
                    if (i % 4 == 3) { continue; }
                    const double difference = averages[mode][i] - averages[0][i];
                    error += difference * difference; magnitude += averages[0][i] * averages[0][i];
                    referenceEnergy += averages[0][i]; cacheEnergy += averages[mode][i];
                }
                const double relativeRMSE = std::sqrt(error / std::max(magnitude, 1e-12));
                const double energyRatio = cacheEnergy / std::max(referenceEnergy, 1e-12);
                const char* label = std::array{"Off", "NoQuery", "Query", "DiffuseOnly", "SpecularOnly"}[mode];
                report[label]["relativeRMSE"] = relativeRMSE;
                report[label]["energyRatio"] = energyRatio;
                double movingError = 0, movingMagnitude = 0;
                for (size_t i = 0; i < movingAverages[0].size(); ++i) {
                    if (i % 4 == 3) { continue; }
                    const double difference = movingAverages[mode][i] - movingAverages[0][i];
                    movingError += difference * difference;
                    movingMagnitude += movingAverages[0][i] * movingAverages[0][i];
                }
                const double movingRMSE = std::sqrt(movingError / std::max(movingMagnitude, 1e-12));
                report[label]["movingRelativeRMSE"] = movingRMSE;
                accurate &= movingRMSE < 0.3;
                accurate &= relativeRMSE < 0.25 && energyRatio > 0.85 && energyRatio < 1.15;
                // Pure metal must never use the diffuse cache. Other cases must
                // actually replace the requested component, not silently miss.
                if (material_ == 2 && mode == 3) { accurate &= error == 0.0; }
                else { queried &= relativeRMSE > 1e-5; }
            }
            report["frames"] = kFrames; report["warmupFrames"] = 32;
            report["width"] = kSize; report["height"] = kSize;
            report["guidesByteIdentical"] = true; report["cacheMissByteIdentical"] = true;
            std::ofstream(root / "Report.json") << report.dump(2);
            requireSharc(queried, "No observable cache queries; test did not exercise early termination");
            requireSharc(accurate, "Cache radiance deviates from reference; see Report.json");
            return RHITestResult::pass("OpenPBR SHaRC GPU transport, cache-miss equivalence, seven exact RR guides, scale/environment invalidation; " + (root / "Report.json").string());
        } catch (const std::exception& exception) {
            return RHITestResult::fail(exception.what());
        }
    }
private:
    uint32_t material_;
};

class OpenPBRMixedSharcTest final : public OpenPBRSharcTest
{
public:
    OpenPBRMixedSharcTest() : OpenPBRSharcTest(1) {}
};
class OpenPBRMetalSharcTest final : public OpenPBRSharcTest
{
public:
    OpenPBRMetalSharcTest() : OpenPBRSharcTest(2) {}
};
METALLIC_REGISTER_RHI_TEST(OpenPBRSharcTest);
METALLIC_REGISTER_RHI_TEST(OpenPBRMixedSharcTest);
METALLIC_REGISTER_RHI_TEST(OpenPBRMetalSharcTest);

} // namespace
} // namespace metallic::tests

#include "RHITest.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include <cmath>
#include <cstring>
#include <fstream>

namespace metallic::tests {
namespace {
using namespace render;
class NativeStrandTest final : public RHITest
{
public:
    NativeStrandTest()
    {
        name = "native_strand_visibility";
        type = RHITestType::Rendering;
    }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            auto require = [](bool ok, const std::string& message) {
                if (!ok) {
                    throw std::runtime_error(message);
                }
            };
            std::string log;
            LegacyMaterialPayload hair;
            hair.textureParams[2] = float(MaterialProgramId::RTXCRChiang);
            hair.rtxcrHairBaseColor[3] = 1;
            auto generation = MaterialGeneration::create(std::span(&hair, 1), 1, log);
            require(generation && generation->supports(MaterialEvaluationTarget::StrandVisibility, log), log);
            require(!generation->supports(MaterialEvaluationTarget::VisibilityBuffer, log),
                    "Opaque VBuffer accepted Fiber");
            const auto fixture = std::filesystem::absolute(context.outputDirectory / "Layers.strands.json");
            nlohmann::json asset{{"version", 1},
                                 {"materialRoot", PROJECT_SOURCE_DIR "/Asset"},
                                 {"materials", {"asset://Materials/Examples/ChestnutFiber.material"}},
                                 {"strands", nlohmann::json::array()}};
            for (int i = 0; i < 3; ++i) {
                asset["strands"].push_back({{"id", i == 2 ? 0x80000030u : uint32_t(10 + i * 10)},
                                            {"material", 0},
                                            {"opacity", 0.5},
                                            {"points",
                                             {{{"position", {-0.7, 0.0, i * 0.1}}, {"radius", 0.04}},
                                              {{"position", {0.7, 0.0, i * 0.1}}, {"radius", 0.04}}}}});
            }
            std::ofstream(fixture) << asset.dump(2);
            RenderSampleLoadResult sample;
            require(loadBuiltInRenderSample("native-strands", sample, log), log);
            auto& vp = sample.graph.findNode("Strands")->properties;
            vp["path"] = fixture.string();
            vp["capacity"] = 4;
            sample.graph.findNode("Fiber")->properties["shadows"] = false;
            sample.graph.markDirty();
            RenderGraphPreviewRenderer preview;
            require(bool(preview.initialize(context.enableValidation, false, false)), "Initialize strand renderer");
            preview.setRawReadbackEnabled(true);
            constexpr uint32_t w = 129, h = 97;
            const size_t center = (size_t(h / 2) * w + w / 2) * 4;
            auto capture = [&](const char* output) {
                require(bool(preview.render(sample.graph, w, h, output)), preview.lastLog());
                require(preview.lastLog().find("error[E") == std::string::npos, preview.lastLog());
                require(preview.readbackBytes().size() == w * h * 16, "Unexpected readback size");
                return preview.readbackBytes();
            };
            auto floats = [&](const char* output) {
                auto data = capture(output);
                std::vector<float> result(data.size() / 4);
                std::memcpy(result.data(), data.data(), data.size());
                for (float v : result) {
                    require(std::isfinite(v), "Nonfinite strand output");
                }
                return result;
            };
            auto counts = capture("Strands.counts");
            std::array<uint32_t, 4> count;
            std::memcpy(count.data(), counts.data() + center * 4, 16);
            require(count[0] == 3 && count[1] == 0 && count[2] == 3, "Multi-layer visibility missing");
            auto image = floats("Fiber.color");
            require(std::abs(image[center + 3] - 0.875f) < 1e-5f, "Front-to-back coverage compositing failed");
            require(image == floats("Fiber.color"), "Static strand coverage is not deterministic");
            auto ids = capture("Fiber.identity");
            std::array<uint32_t, 4> identity;
            std::memcpy(identity.data(), ids.data() + center * 4, 16);
            require(identity[0] == 0x80000030u && identity[1] == 0,
                    "Nearest stable strand identity lost uint32 precision");
            auto staticMotion = floats("Fiber.motion");
            require(std::abs(staticMotion[center]) + std::abs(staticMotion[center + 1]) < 1e-4f,
                    "Static strand motion is nonzero: " + std::to_string(staticMotion[center]) + ", " +
                        std::to_string(staticMotion[center + 1]));
            const auto runtime = [&](const char* node, const char* key, nlohmann::json value) {
                require(sample.graph.setNodeRuntimeProperty(sample.graph.findNode(node)->id, key, std::move(value)),
                        "Set runtime property");
            };
            runtime("Strands", "phase", 0.7);
            auto motion = floats("Fiber.motion");
            require(motion[center] < -0.1f && std::abs(motion[center + 1]) < 1e-4f,
                    "Root-pinned deformation motion missing/wrong direction: " + std::to_string(motion[center]));
            float u = motion[center + 2], depth = motion[center + 3];
            float expected = -std::sin(0.7f) * 0.15f * u * float(h) * 0.5f / (depth * std::tan(45.0f * 0.00872664626f));
            require(std::abs(motion[center] - expected) < 1e-4f, "Material-coordinate reprojection failed");
            auto stopped = floats("Fiber.motion");
            require(std::abs(stopped[center]) < 1e-4f, "Stopped deformation retained motion");
            runtime("Strands", "phase", 0.0);
            runtime("Strands", "density", 0.0);
            auto empty = floats("Fiber.color");
            require(empty[center + 3] == 0, "LOD zero retained coverage");
            runtime("Strands", "density", 1.0);
            auto restored = floats("Fiber.color");
            require(restored == image, "LOD restored a different strand image");
            vp["capacity"] = 2;
            sample.graph.markDirty();
            counts = capture("Strands.counts");
            std::memcpy(count.data(), counts.data() + center * 4, 16);
            require(count[0] == 2 && count[1] == 1, "Overflow capacity/count contract failed");
            auto overflow = floats("Fiber.color");
            require(overflow[center] == 1 && overflow[center + 1] == 0 && overflow[center + 2] == 1,
                    "Overflow was silently discarded");
            runtime("Fiber", "overflowPolicy", "nearest");
            auto nearest = floats("Fiber.color");
            require(std::abs(nearest[center + 3] - 0.75f) < 1e-5f, "Nearest-K compositing failed");

            // A joint still contributes one layer, even when that strand is discarded by K.
            for (auto& strand : asset["strands"]) {
                strand["points"].insert(
                    strand["points"].begin() + 1,
                    nlohmann::json{{"position", {0.0, 0.0, strand["points"][0]["position"][2].get<double>()}},
                                   {"radius", 0.04}});
            }
            std::ofstream(fixture) << asset.dump(2);
            vp["capacity"] = 1;
            sample.graph.markDirty();
            counts = capture("Strands.counts");
            std::memcpy(count.data(), counts.data() + center * 4, 16);
            require(count[0] == 1 && count[1] == 2 && count[2] == 3, "Joint union depended on layer capacity");
            vp["capacity"] = 4;
            sample.graph.markDirty();
            require(std::abs(floats("Fiber.color")[center + 3] - 0.875f) < 1e-5f, "Joint doubled opacity");

            // The editor's shared RenderView must drive native strands and camera motion.
            RenderView view;
            ViewCamera camera;
            camera.eye = {0, 0, 2.4f};
            camera.center = {0, 0, 0};
            camera.fovDegrees = 45;
            require(view.setCamera(camera), "Set strand camera");
            preview.bindRenderView(&view);
            floats("Fiber.motion");
            floats("Fiber.motion");
            camera.eye[0] = camera.center[0] = 0.01f;
            require(view.setCamera(camera), "Move strand camera");
            auto cameraMotion = floats("Fiber.motion");
            const float cameraExpected =
                0.01f * float(h) * 0.5f / (cameraMotion[center + 3] * std::tan(45.0f * 0.00872664626f));
            require(std::abs(cameraMotion[center] - cameraExpected) < 1e-4f,
                    "Shared camera motion/reprojection failed");
            view.cameraCut();
            counts = capture("Strands.counts");
            std::memcpy(count.data(), counts.data() + center * 4, 16);
            require(count[3] == 0, "Camera cut retained stale history");
            camera.eye[0] = camera.center[0] = 0;
            camera.nearPlane = 2.35f;
            require(view.setCamera(camera), "Set near plane");
            counts = capture("Strands.counts");
            std::memcpy(count.data(), counts.data() + center * 4, 16);
            require(count[0] == 1, "Near-plane clipping failed");
            camera.nearPlane = 0.05f;
            camera.orthographic = true;
            require(view.setCamera(camera), "Set orthographic camera");
            require(std::abs(floats("Fiber.color")[center + 3] - 0.875f) < 1e-5f, "Orthographic coverage failed");
            preview.bindRenderView(nullptr);

            // A single thin strand should conserve coverage under subpixel translation.
            auto thin = asset;
            thin["strands"].erase(thin["strands"].begin() + 1, thin["strands"].end());
            for (auto& point : thin["strands"][0]["points"]) {
                point["radius"] = 0.003;
            }
            std::ofstream(fixture) << thin.dump(2);
            sample.graph.markDirty();
            camera.orthographic = false;
            camera.eye = {0, 0, 2.4f};
            camera.center = {0, 0, 0};
            view.setCamera(camera);
            preview.bindRenderView(&view);
            double referenceCoverage = 0;
            for (int step = 0; step < 5; ++step) {
                camera.eye[1] = camera.center[1] = float(step) * 0.005f;
                view.setCamera(camera);
                auto translated = floats("Fiber.color");
                double coverage = 0;
                for (size_t pixel = 0; pixel < translated.size(); pixel += 4) {
                    coverage += translated[pixel + 3];
                }
                if (step == 0) {
                    referenceCoverage = coverage;
                }
                require(referenceCoverage > 1 && std::abs(coverage - referenceCoverage) / referenceCoverage < 0.08,
                        "Subpixel silhouette lost coverage: " + std::to_string(coverage) + " / " +
                            std::to_string(referenceCoverage));
            }
            preview.bindRenderView(nullptr);
            std::ofstream(fixture) << asset.dump(2);
            sample.graph.markDirty();

            // Optional opaque inputs use scene-linear RGB and positive linear view Z.
            auto base = floats("Fiber.color");
            require(sample.graph.addNode("ClearColorPass", "Opaque", {{"color", {0.2, 0.4, 0.6, 1.0}}}) != nullptr,
                    "Add opaque background");
            require(sample.graph.addEdge("Opaque.color", "Fiber.opaqueColor") != nullptr, "Bind opaque color");
            auto composite = floats("Fiber.color");
            const float background[3] = {0.2f, 0.4f, 0.6f};
            // Alpha is unchanged; check the background contribution at a completely uncovered pixel.
            for (size_t c = 0; c < 3; ++c) {
                require(std::abs(composite[c] - background[c]) < 0.005f, "Opaque background composition failed");
            }
            for (size_t c = 0; c < 3; ++c) {
                require(std::abs(composite[center + c] - base[center + c] - (background[c] - 0.025f) * 0.125f) < 0.001f,
                        "Opaque background used incorrect transmittance");
            }
            require(sample.graph.addNode("ClearColorPass", "Depth", {{"color", {1.0, 0.0, 0.0, 1.0}}}) != nullptr,
                    "Add opaque depth");
            auto* depthEdge = sample.graph.addEdge("Depth.color", "Strands.opaqueDepth");
            require(depthEdge != nullptr, "Bind opaque depth");
            const auto depthEdgeId = depthEdge->id;
            require(floats("Fiber.color")[center + 3] == 0, "Opaque surface did not occlude strands");
            sample.graph.removeEdge(depthEdgeId);
            sample.graph.removeNode(sample.graph.findNode("Opaque")->id);
            sample.graph.removeNode(sample.graph.findNode("Depth")->id);

            // Transmissive self shadow: blocker is outside the camera ray, on the key-light ray.
            auto shadowAsset = asset;
            shadowAsset["strands"].erase(shadowAsset["strands"].begin() + 2);
            for (auto& point : shadowAsset["strands"][1]["points"]) {
                point["position"][1] = 0.1;
                point["position"][2] = 0.2;
            }
            std::ofstream(fixture) << shadowAsset.dump(2);
            sample.graph.markDirty();
            runtime("Fiber", "shadows", false);
            auto lit = floats("Fiber.color");
            runtime("Fiber", "shadows", true);
            auto shadowed = floats("Fiber.color");
            float shadowDifference = 0;
            for (size_t c = 0; c < 3; ++c) {
                require(shadowed[center + c] <= lit[center + c] + 1e-6f, "Shadow added energy");
                shadowDifference += lit[center + c] - shadowed[center + c];
            }
            require(shadowDifference > 1e-5f, "Native strand shadow did not attenuate key light");

            // Reload a private material instance through the same asset library/generation path.
            const auto materialRoot = std::filesystem::absolute(context.outputDirectory / "ReloadAssets");
            std::filesystem::create_directories(materialRoot / "Materials/Examples");
            std::filesystem::copy_file(PROJECT_SOURCE_DIR "/Asset/Materials/RTXCRChiang.materialdef",
                                       materialRoot / "Materials/RTXCRChiang.materialdef",
                                       std::filesystem::copy_options::overwrite_existing);
            nlohmann::json materialInstance;
            std::ifstream(PROJECT_SOURCE_DIR "/Asset/Materials/Examples/ChestnutFiber.material") >> materialInstance;
            const auto instancePath = materialRoot / "Materials/Examples/ChestnutFiber.material";
            std::ofstream(instancePath) << materialInstance.dump(2);
            shadowAsset["materialRoot"] = materialRoot.string();
            std::ofstream(fixture) << shadowAsset.dump(2);
            sample.graph.markDirty();
            auto beforeReload = floats("Fiber.color");
            materialInstance["parameters"]["melanin"] = 0.05;
            std::ofstream(instancePath) << materialInstance.dump(2);
            sample.graph.markDirty();
            auto afterReload = floats("Fiber.color");
            float reloadDifference = 0;
            for (size_t c = 0; c < 3; ++c) {
                reloadDifference += std::abs(afterReload[center + c] - beforeReload[center + c]);
            }
            require(reloadDifference > 1e-5f && afterReload[center + 3] == beforeReload[center + 3],
                    "Fiber material reload failed or altered geometric coverage");

            // addNode may relocate node storage: reacquire properties after graph edits.
            sample.graph.findNode("Strands")->properties["path"] =
                PROJECT_SOURCE_DIR "/Asset/Strands/NativeGroom.strands.json";
            sample.graph.findNode("Strands")->properties["capacity"] = 8;
            sample.graph.findNode("Fiber")->properties["shadows"] = true;
            sample.graph.markDirty();
            require(bool(preview.render(sample.graph, 512, 512, "FinalBlit.color")), preview.lastLog());
            require(saveRgba8Png(context.outputDirectory / "NativeGroom.png",
                                 reinterpret_cast<const uint8_t*>(preview.pixels().data()), 512, 512, log),
                    log);
            require(bool(preview.render(sample.graph, 512, 512, "Strands.counts")), preview.lastLog());
            const auto& groomCounts = preview.readbackBytes();
            uint32_t maximumLayers = 0, maximumCandidates = 0, coveredPixels = 0, overflowPixels = 0;
            for (size_t offset = 0; offset < groomCounts.size(); offset += 16) {
                std::array<uint32_t, 4> pixel;
                std::memcpy(pixel.data(), groomCounts.data()+offset, 16);
                overflowPixels += pixel[1] != 0;
                maximumLayers = std::max(maximumLayers, pixel[0]);
                maximumCandidates = std::max(maximumCandidates, pixel[2]);
                coveredPixels += pixel[0] != 0;
            }
            require(coveredPixels > 1000, "Default groom was not visible");
            std::ofstream(context.outputDirectory / "NativeStrandMetrics.json") << nlohmann::json{
                {"width",512},{"height",512},{"capacity",8},{"maximumLayers",maximumLayers},
                {"maximumCandidates",maximumCandidates},{"coveredPixels",coveredPixels},
                {"overflowPixels",overflowPixels},{"shadowRgbDifference",shadowDifference},
                {"materialReloadRgbDifference",reloadDifference},{"subpixelCoverageRelativeTolerance",0.08}}.dump(2);
            require(overflowPixels == 0, "Default sample overflowed its declared capacity: " + std::to_string(maximumCandidates));
            return RHITestResult::pass(
                "Native curves: layers, joints, IDs, deformation/camera motion, cuts, near clip, orthographic view, "
                "subpixel coverage, LOD, overflow, opaque composition, shadows and material reload; Fiber groom "
                "captured");
        }
        catch (const std::exception& e) {
            return RHITestResult::fail(e.what());
        }
    }
};
METALLIC_REGISTER_RHI_TEST(NativeStrandTest);
} // namespace
} // namespace metallic::tests

#include "harness/Fixtures.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Scene/SceneDocument.h"

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;
using Float4 = std::array<float, 4>;
constexpr uint32_t kLanes = 4096, kSlots = 16;
constexpr uint64_t kABI = 0x4649424552000001ull;
struct FiberParams { GPUBufferSpan output; uint32_t lightCount; };
static_assert(sizeof(FiberParams) == 16);

void check(bool result, const std::string& message)
{
    if (!result) { throw std::runtime_error(message); }
}
template<class T> void check(const Result<T>& result, const std::string& message)
{
    check(bool(result), message);
}

class FiberMaterialTest final : public RHITest
{
public:
    FiberMaterialTest() { name = "material_fiber_native_and_stages"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            std::atomic_uint errors{0};
            bench::TestDevice device;
            check(bench::createTestDevice(context, {.applicationName = "Fiber domain",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if ((message.severity == render::ValidationSeverity::Error) &&
                        (render::hasFlag(message.type, render::ValidationCategory::Validation))) { ++*static_cast<std::atomic_uint*>(target); }
                }, &errors}}).transform([&](auto value) { device = std::move(value); }), "Device");
            ResourceRegistry registry;
            check(registry.initialize(*device), "Registry");
            std::unique_ptr<Buffer> output;
            check(device->createBuffer({.size = kLanes * kSlots * sizeof(Float4), .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostReadback}).transform([&](auto value) { output = std::move(value); }), "Output");
            std::array<ComputeKernel, 2> kernels;
            std::filesystem::create_directories(context.outputDirectory);
            std::ofstream report(context.outputDirectory / "FiberComparison.txt");
            for (uint32_t side = 0; side < 2; ++side) {
                const SlangMacroDefine define{"FIBER_REFERENCE", side == 0 ? "1" : "0"};
                std::string log;
                // Native path intentionally has no RTXCR SDK search directory.
                auto shader = compileSlangShaderToSpirv({.moduleName = "FiberProbe", .entryPointName = "fiberProbeMain",
                    .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .macroDefines = {&define, 1}}, {.enableDiskCache = false}, log);
                check(shader, log);
                std::ofstream file(context.outputDirectory / (side == 0 ? "Reference.spv" : "Native.spv"), std::ios::binary);
                file.write(reinterpret_cast<const char*>(shader->spirv.data()), shader->spirv.size() * sizeof(uint32_t));
                check(kernels[side].initialize(*device, {.spirv = shader->spirv,
                    .parameters = parameterAbi<FiberParams>(kABI, ParameterTransport::InlinePush)}, log), log);
            }
            double maximum = 0;
            for (uint32_t lights : {1u, 8u}) {
                std::vector<Float4> reference;
                for (uint32_t side = 0; side < 2; ++side) {
                    ParameterWriter writer(*device, registry);
                    auto encoded = writer.encode(FiberParams{writer.bufferSpan<Float4>(output.get()), lights}, kABI, ParameterTransport::InlinePush);
                    check(encoded, "Parameters");
                    auto dispatch = kernels[side].prepareDispatch(*encoded, kLanes / 64);
                    check(dispatch, "Dispatch");
                    bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
                    check(gpu.initialize(*device), "Commands");
                    check(dispatch->record(*gpu.commands), "Record");
                    check(gpu.submitAndWait(), "Complete");
                    output->invalidate();
                    const auto* mapped = static_cast<const Float4*>(output->map());
                    check(mapped != nullptr, "Readback");
                    std::vector<Float4> values(mapped, mapped + kLanes * kSlots);
                    output->unmap();
                    std::ofstream file(context.outputDirectory / (std::string(side ? "Native-" : "Reference-") + std::to_string(lights) + ".bin"), std::ios::binary);
                    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(Float4));
                    uint32_t samples = 0;
                    for (uint32_t lane = 0; lane < kLanes; ++lane) {
                        if (values[lane * kSlots + 2][3] != 0) { ++samples; }
                        for (uint32_t slot = 0; slot < kSlots; ++slot) {
                            for (uint32_t c = 0; c < 4; ++c) {
                                const float value = values[lane * kSlots + slot][c];
                                check(std::isfinite(value), "Nonfinite Fiber lane=" + std::to_string(lane) + " slot=" + std::to_string(slot));
                                if (side) {
                                    const float expected = reference[lane * kSlots + slot][c];
                                    const double difference = std::abs(double(value) - expected);
                                    maximum = std::max(maximum, difference);
                                    check(difference <= 2e-5 + 2e-4 * std::abs(expected),
                                        "Fiber mismatch lane=" + std::to_string(lane) + " slot=" + std::to_string(slot) +
                                        " component=" + std::to_string(c) + " reference=" + std::to_string(expected) + " native=" + std::to_string(value));
                                }
                            }
                        }
                        // Independent stage and domain contracts, in addition to the vendor oracle.
                        check(values[lane * kSlots + 14] == Float4{1, float(lights), float(lane * 64), 0}, "Repeated evaluate or lost instance offset");
                        check(values[lane * kSlots + 15] == Float4{}, "Unsupported importance transport is not rejected");
                    }
                    check(samples > kLanes / 2, "Probe did not exercise enough valid samples");
                    report << "lights=" << lights << " side=" << side << " validChiang=" << samples << '\n';
                    if (!side) { reference = std::move(values); }
                }
            }
            check(errors == 0, "Vulkan validation errors");
            report << "maxAbsolute=" << maximum << " validationErrors=" << errors << '\n';
            return RHITestResult::pass("4096 cases: Chiang/SeparateChiang/FarField eval+sampling match upstream; Fiber stages load once for 1/8 lights, projected measure, instance offsets and transport rejection");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(FiberMaterialTest);

class FiberGuideParametersTest final : public RHITest
{
public:
    FiberGuideParametersTest() { name = "material_fiber_guide_parameters"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            constexpr uint32_t kCases = 8, kGuideSlots = 10;
            const std::array<Float4, 4> colors{{
                {0.15f, 0.35f, 0.55f, 0}, {-0.3f, 1.4f, 0.7f, 0},
                {0.8f, 0.1f, 0.25f, 0}, {0.4f, 0.6f, 0.9f, 0}}};
            const std::array roughness{-4.0f, 0.18f, 1.6f, 0.6f};
            const std::array iors{0.2f, 1.7f, 4.5f, 2.3f};
            std::array<SceneMaterialData, kCases> records;
            for (uint32_t index = 0; index < kCases; ++index) {
                auto& record = records[index];
                const uint32_t pair = index / 2;
                std::copy(colors[pair].begin(), colors[pair].end(), record.rtxcrHairBaseColor);
                record.rtxcrHairParams0[0] = roughness[pair];
                record.rtxcrHairParams0[2] = iors[pair];
                record.textureParams[2] = float(MaterialProgramId::RTXCRChiang);
                // Identical Fiber inputs, deliberately conflicting Surface
                // factors and a clear source hair flag. Only the explicit
                // published domain and typed Fiber parameters may guide shading.
                const bool alternate = (index & 1u) != 0;
                record.baseColor[0] = alternate ? 0.97f : 0.02f;
                record.baseColor[1] = alternate ? 0.01f : 0.95f;
                record.baseColor[2] = alternate ? 0.9f : 0.03f;
                record.params[0] = alternate ? 1.0f : 0.0f;
                record.params[1] = alternate ? 0.93f : 0.03f;
                record.glassParams[0] = alternate ? 0.65f : 0.0f;
                record.glassParams[1] = alternate ? 2.9f : 1.2f;
                record.textureParams[0] = alternate ? 50.0f : 0.0f;
                record.textureParams[1] = alternate ? 0.0f : 1.0f;
                record.emissive[0] = alternate ? 10.0f : 0.0f;
                record.emissive[3] = alternate ? 1.0f : 0.0f;
                if (alternate) {
                    record.baseColorTexture.textureIndex = 0u;
                    record.normalTexture.textureIndex = 0u;
                    record.emissiveTexture.textureIndex = 0u;
                }
            }
            std::atomic_uint validationErrors{0};
            bench::TestDevice device;
            check(bench::createTestDevice(context, {.applicationName = "Fiber guide parameters",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true,
                .validationSink = {[](void* target, const ValidationMessage& message) noexcept {
                    if (message.severity == ValidationSeverity::Error && hasFlag(message.type, ValidationCategory::Validation)) {
                        ++*static_cast<std::atomic_uint*>(target);
                    }
                }, &validationErrors}}).transform([&](auto value) { device = std::move(value); }), "Guide device");
            std::unique_ptr<Buffer> input, output;
            check(device->createBuffer({.size = sizeof(records), .structureStride = sizeof(SceneMaterialData),
                .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload})
                .transform([&](auto value) { input = std::move(value); }), "Guide input");
            check(device->createBuffer({.size = kCases * kGuideSlots * sizeof(Float4),
                .structureStride = sizeof(Float4), .usage = BufferUsageBits::Storage,
                .memoryLocation = MemoryLocation::HostReadback})
                .transform([&](auto value) { output = std::move(value); }), "Guide output");
            void* mappedInput = input->map();
            check(mappedInput != nullptr, "Guide input mapping");
            std::memcpy(mappedInput, records.data(), sizeof(records));
            input->flush();
            input->unmap();
            std::string log;
            auto shader = compileSlangShaderToSpirv({.moduleName = "FiberGuideProbe",
                .entryPointName = "fiberGuideProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders"},
                {.enableDiskCache = false}, log);
            check(shader, log);
            const std::array layout{
                ComputeResourceBindingDesc{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials),
                    .kind = ComputeResourceBindingKind::StorageBuffer},
                ComputeResourceBindingDesc{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput),
                    .kind = ComputeResourceBindingKind::StorageBuffer}};
            ComputeProgram program;
            check(program.initialize(*device, {.spirv = shader->spirv, .bindings = layout,
                .requiresRayQuery = false, .resourceParameterSize = sizeof(SceneResourceParameters)}, log), log);
            const std::array bindings{
                ComputeDispatchBinding{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materials),
                    .buffer = input.get()},
                ComputeDispatchBinding{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, probeOutput),
                    .buffer = output.get()}};
            bench::GPUCommands gpu(*device->getQueue(QueueType::Graphics));
            check(gpu.initialize(*device), "Guide commands");
            check(program.dispatch({.commandBuffer = gpu.commands.get(), .bindings = bindings}), "Guide dispatch");
            check(gpu.submitAndWait(), "Guide completion");
            output->invalidate();
            const auto* mapped = static_cast<const Float4*>(output->map());
            check(mapped != nullptr, "Guide readback");
            const std::vector<Float4> actual(mapped, mapped + kCases * kGuideSlots);
            bench::readbackEvidence(context, "FiberGuides.bin", std::span(mapped, kCases * kGuideSlots));
            output->unmap();
            for (uint32_t index = 0; index < kCases; ++index) {
                const uint32_t pair = index / 2;
                const float r = std::clamp(roughness[pair], 0.02f, 1.0f);
                const float ior = std::clamp(iors[pair], 1.01f, 3.0f);
                const float f0 = (ior - 1.0f) / (ior + 1.0f);
                const float red = std::clamp(colors[pair][0], 0.0f, 1.0f);
                const float green = std::clamp(colors[pair][1], 0.0f, 1.0f);
                const float blue = std::clamp(colors[pair][2], 0.0f, 1.0f);
                // Independent scalar evaluation of the established GGX guide
                // approximation at normal incidence; Fiber supplies Fresnel F0.
                const double alpha = double(r) * r;
                const double alphaCubed = alpha * alpha * alpha;
                const double bias = (-0.393023 + 1.72691 * alpha) /
                    (56.04732 + 71.40998 * alpha + 2794.14169 * alphaCubed) *
                    std::clamp(double(f0) * f0 * 50.0, 0.0, 1.0);
                const double scale = (3.7725574 + 5.8797 * alpha) /
                    (3.73232 + 7.2821 * alpha + 17.195586 * alphaCubed);
                const float specular = float(std::clamp(double(f0) * f0 * std::max(scale, 0.0) +
                    std::max(bias, 0.0), 0.0, 1.0));
                const std::array<Float4, kGuideSlots> expected{{
                    {red, green, blue, r}, {red, green, blue, ior},
                    {f0 * f0, f0 * f0, f0 * f0, 1}, {0, -1, 0, 97},
                    {0, 1, 0, 0}, {0, 0, 0, 0}, {red, green, blue, r},
                    {0, -1, 0, 97}, {specular, specular, specular, 2}, {0, 0, 0, 0}}};
                for (uint32_t slot = 0; slot < kGuideSlots; ++slot) {
                    for (uint32_t component = 0; component < 4; ++component) {
                        const float value = actual[index * kGuideSlots + slot][component];
                        check(std::isfinite(value) && std::abs(value - expected[slot][component]) <= 2e-6f,
                            "Fiber guide mismatch case=" + std::to_string(index) + " slot=" + std::to_string(slot) +
                            " component=" + std::to_string(component));
                    }
                    check(actual[index * kGuideSlots + slot] == actual[(index ^ 1u) * kGuideSlots + slot],
                        "Unrelated Surface factors changed Fiber guides");
                }
            }
            check(validationErrors == 0, "Fiber guide Vulkan validation errors");
            return RHITestResult::pass("Standard/OpenPBR production Fiber guides use typed color/roughness/IOR; Surface factors do not affect guides, normal policy or RNG");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(FiberGuideParametersTest);

#if defined(METALLIC_HAS_RTXCR_ASSETS) && METALLIC_HAS_RTXCR_ASSETS && \
    defined(METALLIC_HAS_RTXCR_GEOMETRY) && METALLIC_HAS_RTXCR_GEOMETRY
class FiberAssetRenderingTest final : public RHITest
{
public:
    FiberAssetRenderingTest() { name = "material_fiber_asset_rendering"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        try {
            const auto root = context.outputDirectory / "assets";
            std::filesystem::create_directories(root);
            RenderSampleLoadResult sample;
            std::string log;
            check(loadBuiltInRenderSample("rtxcr-material-sample", sample, log), log);
            const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath;
            std::ifstream input(source);
            auto gltf = nlohmann::json::parse(input);
            for (auto& buffer : gltf["buffers"]) {
                const auto filename = buffer["uri"].get<std::string>();
                std::filesystem::copy_file(source.parent_path() / filename, root / filename,
                    std::filesystem::copy_options::overwrite_existing);
            }
            const auto scenePath = root / "Fiber.gltf";
            std::ofstream(scenePath) << gltf.dump();
            // Reusing an output directory must start from the imported groom,
            // not the edited sidecar produced by this test's previous run.
            std::filesystem::remove(root / "Fiber.metallic_scene.json");
            scene::SceneDocument document;
            check(document.load(scenePath), document.lastLoadResult().error);
            check(document.materials().size() == 1 && document.materials()[0].rtxcrHair, "Expected imported Claire Fiber material");
            const auto definition = material::defaultRTXCRChiangDefinition();
            std::ofstream(root / "Fiber.materialdef") << material::serializeMaterialDefinition(definition);
            material::MaterialInstance instance;
            check(material::createMaterialInstance(document.materials()[0], "asset://Fiber.materialdef", definition, {}, instance, log), log);
            material::MaterialAssetLibrary library(root);
            check(library.save("asset://Fiber.material", instance, log), log);
            auto* pass = sample.graph.findNode("PathTrace");
            check(pass != nullptr, "PathTrace pass");
            pass->properties["path"] = scenePath.string();
            sample.graph.markDirty();
            const auto renderImage = [&](const scene::SceneDocument& scene, const char* name) {
                RenderGraphPreviewRenderer preview;
                preview.bindRuntimeScene(&scene);
                const auto& environment = sample.desc.environment.value();
                preview.setEnvironment({.enabled = environment.enabled, .path = environment.path,
                    .intensity = environment.intensity, .rotationDegrees = environment.rotationDegrees, .visible = environment.visible});
                check(preview.initialize(context.enableValidation, true, false), "Fiber scene device");
                for (uint32_t frame = 0; frame < 64; ++frame) {
                    preview.setRawReadbackEnabled(frame == 63);
                    check(preview.render(sample.graph, 384, 216, "PathTrace.color"), preview.lastLog());
                    if (frame == 0) {
                        std::ofstream(context.outputDirectory / (std::string(name) + ".compile.log")) << preview.lastLog();
                        check(preview.lastLog().find("error[") == std::string::npos &&
                            preview.lastLog().find("error material") == std::string::npos, preview.lastLog());
                    }
                }
                check(preview.readbackFormat() == Format::RGBA32Sfloat, "Fiber HDR format");
                const auto& bytes = preview.readbackBytes();
                check(bytes.size() == 384 * 216 * sizeof(Float4), "Fiber HDR extent");
                std::vector<Float4> pixels(bytes.size() / sizeof(Float4));
                std::memcpy(pixels.data(), bytes.data(), bytes.size());
                for (const auto& pixel : pixels) { for (float value : pixel) { check(std::isfinite(value), "Nonfinite Fiber scene"); } }
                std::ofstream file(context.outputDirectory / (std::string(name) + ".rgba32f"), std::ios::binary);
                file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                return pixels;
            };
            const auto legacy = renderImage(document, "Imported");
            check(document.setMaterialAsset(0, "asset://Fiber.material", root, log), log);
            const auto asset = renderImage(document, "Asset");
            check(asset == legacy, "Binding equivalent Fiber asset changed rendered pixels");
            bench::TestDevice device;
            check(bench::createTestDevice(context, {.applicationName = "Fiber asset bindings",
                .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true})
                .transform([&](auto value) { device = std::move(value); }), "Fiber binding device");
            auto& queue = *device->getQueue(QueueType::Graphics);
            ScenePathTraceResources resources;
            check(resources.beginPrepareAsync(*device, queue, {{"path", scenePath.string()}}, document, log), log);
            bool complete = false;
            scene::SceneLoadProgress progress;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (!complete && std::chrono::steady_clock::now() < deadline) {
                check(resources.pumpPrepareAsync(8, progress, log).transform([&](auto value) { complete = value; }), log);
                check(queue.waitIdle(), "Fiber upload completion");
            }
            check(complete && resources.valid(), "Fiber material upload timed out");
            const auto original = resources.materialGeneration();
            check(bool(original), "Fiber generation missing");
            instance.parameters["melanin"] = 0.15;
            check(library.save("asset://Fiber.material", instance, log), log);
            check(document.reloadMaterialAsset(0, log), log);
            check(document.save(log), log);
            scene::SceneDocument reloaded;
            check(reloaded.load(document.documentPath()), reloaded.documentWarning());
            check(reloaded.documentWarning().empty(), reloaded.documentWarning());
            const auto changed = renderImage(reloaded, "Edited");
            double energy = 0;
            for (size_t i = 0; i < asset.size(); ++i) {
                for (uint32_t c = 0; c < 3; ++c) { energy += std::abs(changed[i][c] - asset[i][c]); }
            }
            check(energy > 10, "Fiber asset parameter edit did not affect scene execution");
            check(resources.syncRuntimeScene(&document, log), log);
            const auto generation = resources.materialGeneration();
            check(generation && generation->programCount() == 1 &&
                generation->instances()[0].program == original->instances()[0].program &&
                generation->parameters()[0].rtxcrHairParams1[0] == 0.15f &&
                generation->instances()[0].program->definition->domain == MaterialDomain::Fiber,
                "Fiber instances did not share one program");
            check(!generation->supports(MaterialEvaluationTarget::VisibilityBuffer, log), "Fiber was accepted by opaque VBuffer");
            pass->properties["bsdf"] = "openpbr";
            sample.graph.markDirty();
            const auto openpbr = renderImage(reloaded, "OpenPBRPathFiber");
            double openpbrEnergy = 0, squaredDifference = 0, squaredReference = 0;
            for (size_t i = 0; i < openpbr.size(); ++i) {
                for (uint32_t c = 0; c < 3; ++c) {
                    openpbrEnergy += openpbr[i][c];
                    const double difference = double(openpbr[i][c]) - changed[i][c];
                    squaredDifference += difference * difference;
                    squaredReference += double(changed[i][c]) * changed[i][c];
                }
            }
            check(openpbrEnergy > 10, "OpenPBR path failed to execute Fiber domain");
            check(squaredDifference <= 1e-8 * squaredReference, "OpenPBR path changed Fiber transport");

            // Restart both production paths at frame zero with the same Fiber
            // asset, geometry and settings, but live, unrelated Surface inputs.
            // A valid map exercises both normal mapping and stochastic texture
            // RNG if Surface evaluation accidentally runs before Fiber dispatch.
            const uint8_t surfaceTexel[] = {255, 32, 128, 255};
            check(saveRgba8Png(root / "SurfaceOnly.png", surfaceTexel, 1, 1, log), log);
            auto contaminated = gltf;
            if (!contaminated.contains("images")) { contaminated["images"] = nlohmann::json::array(); }
            if (!contaminated.contains("textures")) { contaminated["textures"] = nlohmann::json::array(); }
            const auto imageIndex = contaminated["images"].size();
            contaminated["images"].push_back({{"uri", "SurfaceOnly.png"}});
            const auto textureIndex = contaminated["textures"].size();
            contaminated["textures"].push_back({{"source", imageIndex}});
            auto& surfaceInputs = contaminated["materials"][0];
            auto& pbr = surfaceInputs["pbrMetallicRoughness"];
            pbr["baseColorFactor"] = {0.07, 0.8, 0.12, 1.0};
            pbr["metallicFactor"] = 0.9;
            pbr["roughnessFactor"] = 0.04;
            pbr["baseColorTexture"] = {{"index", textureIndex}};
            pbr["metallicRoughnessTexture"] = {{"index", textureIndex}};
            surfaceInputs["emissiveFactor"] = {0.5, 1.0, 2.0};
            surfaceInputs["emissiveTexture"] = {{"index", textureIndex}};
            surfaceInputs["normalTexture"] = {{"index", textureIndex}, {"scale", 8.0}};
            surfaceInputs["occlusionTexture"] = {{"index", textureIndex}, {"strength", 0.0}};
            surfaceInputs["extensions"]["KHR_materials_transmission"] = {{"transmissionFactor", 0.8}};
            surfaceInputs["extensions"]["KHR_materials_ior"] = {{"ior", 2.7}};
            const auto contaminatedPath = root / "FiberSurfaceInputs.gltf";
            std::ofstream(contaminatedPath) << contaminated.dump();
            std::filesystem::remove(root / "FiberSurfaceInputs.metallic_scene.json");
            scene::SceneDocument contaminatedScene;
            check(contaminatedScene.load(contaminatedPath), contaminatedScene.lastLoadResult().error);
            check(contaminatedScene.setMaterialAsset(0, "asset://Fiber.material", root, log), log);
            const auto& contaminatedMaterial = contaminatedScene.materials()[0];
            check(contaminatedMaterial.normalTexture.textureIndex >= 0 &&
                contaminatedMaterial.baseColorTexture.textureIndex >= 0 &&
                contaminatedMaterial.emissiveFactor.x > 0 && contaminatedMaterial.normalTextureScale == 8.0f &&
                contaminatedMaterial.transmissionFactor > 0.0f && contaminatedMaterial.rtxcrHair,
                "Fiber isolation fixture did not retain unrelated Surface inputs");
            pass->properties["path"] = contaminatedPath.string();
            pass->properties["bsdf"] = "standard";
            sample.graph.markDirty();
            const auto contaminatedStandard = renderImage(contaminatedScene, "SurfaceInputsStandard");
            check(contaminatedStandard == changed, "Surface emission, normal maps or texture RNG changed standard Fiber HDR");
            pass->properties["bsdf"] = "openpbr";
            sample.graph.markDirty();
            const auto contaminatedOpenPBR = renderImage(contaminatedScene, "SurfaceInputsOpenPBR");
            check(contaminatedOpenPBR == openpbr, "Surface inputs changed OpenPBR Fiber HDR");
            std::ofstream(context.outputDirectory / "FiberAsset.txt") << "imported==asset; changedAbsoluteRGB=" << energy
                << "; programs=1; opaqueVBufferRejected=true; OpenPBRPathEnergy=" << openpbrEnergy
                << "; SurfaceInputsStandard==Edited; SurfaceInputsOpenPBR==OpenPBRPathFiber\n";
            return RHITestResult::pass("Claire DOTS asset edits affect Fiber HDR; standard/OpenPBR paths ignore live Surface emission, normal maps and texture RNG; opaque VBuffer rejects Fiber");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(FiberAssetRenderingTest);
#endif
} // namespace
} // namespace metallic::tests

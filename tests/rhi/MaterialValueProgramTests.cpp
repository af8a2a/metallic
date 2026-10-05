#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/NamedResourceLayouts.h"
#include "RHITest.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/ColorSpace.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Scene/SceneDocument.h"
#include "Runtime/Material/MaterialAsset.h"
#include "Runtime/Render/Subsystem/EnvironmentLightingSubsystem.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>

namespace metallic::tests {
namespace {
using namespace render;

std::string readProgram(const char* name)
{
    std::ifstream file(std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/LookDev/MaterialPrograms" / name);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

class MaterialValueCompilerTest final : public RHITest
{
public:
    MaterialValueCompilerTest() { name = "material_value_compiler"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext& context) override
    {
        std::string log;
        std::vector<scene::RenderMaterial> materials(1000);
        for (size_t i = 0; i < materials.size(); ++i) {
            materials[i].valueProgram = R"({"version":1,"baseColor":{"op":"parameter","index":0}})";
            materials[i].valueParameters[0] = float(i) / 1000;
        }
        auto first = MaterialValueProgramSet::create(materials, log);
        if (!first || first->programCount() != 1 || first->instances().size() != 1000 ||
            first->manifests()[0].parameterMask != 1 || first->manifests()[0].outputMask != 1 ||
            first->instances()[999].parameters[0] != 0.999f) { return RHITestResult::fail("Instance sharing: " + log); }
        materials.resize(1);
        materials[0].valueProgram = R"({ "baseColor": {"index":0,"op":"parameter"}, "version": 1 })";
        materials[0].valueParameters[0] = 0.25f;
        auto second = MaterialValueProgramSet::create(materials, log);
        if (!second || first->key() != second->key() || first->source() != second->source()) {
            return RHITestResult::fail("Parameters/count/JSON key order changed program identity");
        }
        for (const char* invalid : {
                R"({"version":1,"coverage":1})", R"({"version":2,"metallic":1})",
                R"({"version":1,"baseColor":{"op":"texture"}})",
                R"({"version":1,"baseColor":{"op":"parameter","index":4}})",
                R"({"version":1,"baseColor":{"op":"sin","args":[]}})",
                R"({"version":1,"baseColor":[1,2,3]})", R"({"version":1,"metallic":1e20})", "broken"}) {
            materials[0].valueProgram = invalid;
            if (MaterialValueProgramSet::create(materials, log) || log.empty()) {
                return RHITestResult::fail(std::string("Invalid program accepted: ") + invalid);
            }
        }
        for (uint32_t count : {1u, 16u, 64u, 256u}) {
            materials.assign(count, {});
            for (uint32_t i = 0; i < count; ++i) {
                materials[i].valueProgram = "{\"version\":1,\"emissive\":" + std::to_string(i) + "}";
            }
            auto set = MaterialValueProgramSet::create(materials, log);
            if (count <= kMaxMaterialValuePrograms ? (!set || set->programCount() != count) : bool(set)) {
                return RHITestResult::fail("Program budget did not enforce 64-program limit");
            }
            if (set) {
                std::reverse(materials.begin(), materials.end());
                auto reordered = MaterialValueProgramSet::create(materials, log);
                if (!reordered || set->key() != reordered->key()) { return RHITestResult::fail("Order changed set key"); }
            }
        }
        materials.assign(1, {});
        materials[0].valueProgram = readProgram("ProceduralRust.value.json");
        auto rust = MaterialValueProgramSet::create(materials, log);
        if (!rust) { return RHITestResult::fail(log); }
        std::filesystem::path includeDirectory;
        if (!rust->writeInclude(context.outputDirectory / "programs", includeDirectory, log)) { return RHITestResult::fail(log); }
        materials[0].alphaMode = "MASK";
        if (!MaterialValueProgramSet::create(materials, log)) { return RHITestResult::fail("MASK Surface values rejected: " + log); }
        materials[0].alphaMode = "OPAQUE";
        materials[0].transmissionFactor = 0.5f;
        if (MaterialValueProgramSet::create(materials, log)) { return RHITestResult::fail("Custom transmission accepted"); }
        materials[0].transmissionFactor = 0;
        materials[0].valueParameters[0] = std::numeric_limits<float>::infinity();
        if (MaterialValueProgramSet::create(materials, log)) { return RHITestResult::fail("Infinite parameter accepted"); }
        return RHITestResult::pass("Canonical static sets, 1000 instances, 1/16/64 accepted, 256 rejected; unsupported operations rejected");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialValueCompilerTest);

class PainterLookDevScenesTest final : public RHITest
{
public:
    PainterLookDevScenesTest() { name = "painter_lookdev_scenes"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        auto samples = listBuiltInRenderSamples();
        std::erase_if(samples, [](const auto& sample) { return sample.category != "Painter Validation"; });
        if (samples.empty()) { return RHITestResult::skip("Run Tools/MaterialValidation/BuildLookDevScenes.py first"); }
        if (samples.size() != 24) { return RHITestResult::fail("Expected all 24 Painter scenes"); }
        std::string log;
        RenderGraphPreviewRenderer preview;
        if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Device failed"); }
        preview.setRawReadbackEnabled(true);
        scene::SceneDocument document;
        nlohmann::json evidence = nlohmann::json::array();
        for (const auto& desc : samples) {
            RenderSampleLoadResult sample;
            if (!loadBuiltInRenderSample(desc.id, sample, log) || !document.load(desc.scenePath) ||
                !document.documentWarning().empty()) {
                return RHITestResult::fail(desc.id + ": " + log + document.documentWarning());
            }
            for (const auto& material : document.materials()) {
                if (material.valueProgram.empty()) { return RHITestResult::fail(desc.id + ": material inputs missing"); }
            }
            preview.bindRuntimeScene(&document);
            preview.setEnvironment(document.environment());
            preview.setLighting(document.lighting());
            for (const char* pass : {"Reference", "Deferred"}) {
                auto* node = sample.graph.findNode(pass);
                node->properties["samples"] = 4;
                node->properties["maxDepth"] = 8;
                node->properties["accumulate"] = false;
            }
            sample.graph.markDirty();
            for (const char* output : {"Reference.color", "Deferred.color"}) {
                for (int frame = 0; frame < 3; ++frame) {
                    if (!preview.render(sample.graph, 256, 256, output)) { return RHITestResult::fail(desc.id + ": " + preview.lastLog()); }
                    if (preview.lastLog().find("error material") != std::string::npos ||
                        preview.lastLog().find("error[E") != std::string::npos) {
                        return RHITestResult::fail(desc.id + ": " + preview.lastLog());
                    }
                }
                if (preview.readbackFormat() != Format::RGBA32Sfloat || preview.readbackBytes().empty()) {
                    return RHITestResult::fail("Expected linear float32 output");
                }
                float maximum = 0;
                size_t errorPixels = 0;
                for (size_t offset = 0; offset + 16 <= preview.readbackBytes().size(); offset += 16) {
                    std::array<float, 4> pixel;
                    std::memcpy(pixel.data(), preview.readbackBytes().data() + offset, 16);
                    if ((pixel[0] == 1.0f || pixel[0] == 0.08f) && pixel[1] == 0 && pixel[2] == pixel[0]) { ++errorPixels; }
                }
                if (errorPixels > 16) { return RHITestResult::fail(desc.id + ": material compiler error checker detected"); }
                for (size_t offset = 0; offset < preview.readbackBytes().size(); offset += 4) {
                    float value; std::memcpy(&value, preview.readbackBytes().data() + offset, 4);
                    if (!std::isfinite(value)) { return RHITestResult::fail(desc.id + ": nonfinite HDR"); }
                    if ((offset / 4) % 4 != 3) { maximum = std::max(maximum, value); }
                }
                if (maximum <= 0) { return RHITestResult::fail(desc.id + ": black image"); }
                if (!saveRgba8Png(context.outputDirectory / (desc.id + "-" + output + ".png"),
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), 256, 256, log)) { return RHITestResult::fail(log); }
                evidence.push_back({{"scene", desc.id}, {"output", output}, {"maximum", maximum},
                    {"materials", document.materials().size()}, {"textures", document.textures().size()}});
                std::ofstream(context.outputDirectory / "PainterScenes.json") << evidence.dump(2);
            }
        }
        return RHITestResult::pass("24 complete Painter scenes, sequential PT/Deferred rendering, finite HDR, screenshots saved; no pixel equivalence claim");
    }
};
METALLIC_REGISTER_RHI_TEST(PainterLookDevScenesTest);

class PainterFuzzEnvironmentTest final : public RHITest
{
public:
    PainterFuzzEnvironmentTest() { name = "painter_fuzz_environment"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        const auto samples = listBuiltInRenderSamples();
        auto desc = std::find_if(samples.begin(), samples.end(), [](const auto& value) {
            return value.id == "painter-M05_Fuzz-textured";
        });
        if (desc == samples.end()) { return RHITestResult::skip("Run Tools/MaterialValidation/BuildLookDevScenes.py first"); }
        std::string log;
        RenderSampleLoadResult sample;
        scene::SceneDocument document;
        if (!loadBuiltInRenderSample(desc->id, sample, log) || !document.load(desc->scenePath)) {
            return RHITestResult::fail(log + document.lastLoadResult().error);
        }
        RenderGraphPreviewRenderer preview;
        if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Device failed"); }
        preview.setRawReadbackEnabled(true);
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        preview.setLighting(document.lighting());
        constexpr uint32_t width = 512, height = 256;
        std::array<std::vector<float>, 3> images;
        const std::array budgets{1, 64, 1024};
        for (size_t index = 0; index < budgets.size(); ++index) {
            sample.graph.findNode("Deferred")->properties["samples"] = budgets[index];
            sample.graph.markDirty();
            for (int frame = 0; frame < 3; ++frame) {
                if (!preview.render(sample.graph, width, height, "Deferred.color") ||
                    preview.lastLog().find("error material") != std::string::npos ||
                    preview.lastLog().find("error[E") != std::string::npos) {
                    return RHITestResult::fail(preview.lastLog());
                }
            }
            if (preview.readbackFormat() != Format::RGBA32Sfloat || preview.readbackBytes().size() != width * height * 16) {
                return RHITestResult::fail("Expected linear RGBA32F output");
            }
            auto& pixels = images[index];
            pixels.resize(width * height * 4);
            std::memcpy(pixels.data(), preview.readbackBytes().data(), preview.readbackBytes().size());
            if (std::any_of(pixels.begin(), pixels.end(), [](float value) { return !std::isfinite(value); })) {
                return RHITestResult::fail("Nonfinite Fuzz environment output");
            }
            const auto stem = context.outputDirectory / ("Fuzz-" + std::to_string(budgets[index]));
            std::vector<uint8_t> display(pixels.size(), 255);
            for (size_t offset = 0; offset < pixels.size(); offset += 4) {
                const auto rgb = color::toLinearRec709({pixels[offset], pixels[offset + 1], pixels[offset + 2]});
                for (size_t channel = 0; channel < 3; ++channel) {
                    const float linear = std::max(rgb[channel], 0.0f);
                    const float srgb = linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
                    display[offset + channel] = uint8_t(std::clamp(srgb, 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
            if (!saveRgba8Png(stem.string() + ".png", display.data(), width, height, log)) {
                return RHITestResult::fail(log);
            }
            std::ofstream raw(stem.string() + ".rgba32f", std::ios::binary);
            raw.write(reinterpret_cast<const char*>(pixels.data()), pixels.size() * sizeof(float));
        }
        if (images[0] == images[1] || images[1] == images[2]) {
            return RHITestResult::fail("Deferred ignored or clamped its environment sample budget");
        }
        // Sphere interiors in the fixed M05 camera: avoid background and silhouette
        // coverage masking an IBL regression. Compare single frames, never accumulation.
        double squaredError = 0, referenceEnergy = 0, mean = 0, referenceMean = 0;
        for (uint32_t y = 108; y < 148; ++y) {
            for (uint32_t x = 163; x < 349; ++x) {
                bool interior = false;
                for (int center : {183, 256, 329}) {
                    const int dx = int(x) - center, dy = int(y) - 128;
                    interior |= dx * dx + dy * dy < 20 * 20;
                }
                if (!interior) { continue; }
                for (size_t c = 0; c < 3; ++c) {
                    const size_t offset = (y * width + x) * 4 + c;
                    const double value = images[1][offset], reference = images[2][offset];
                    squaredError += (value - reference) * (value - reference);
                    referenceEnergy += reference * reference;
                    mean += value;
                    referenceMean += reference;
                }
            }
        }
        const double relativeRms = std::sqrt(squaredError / std::max(referenceEnergy, 1e-20));
        const double relativeMean = std::abs(mean / std::max(referenceMean, 1e-20) - 1.0);
        std::ofstream(context.outputDirectory / "FuzzQuality.json") << nlohmann::json{
            {"relativeRms64Vs1024", relativeRms}, {"relativeMean64Vs1024", relativeMean}}.dump(2);
        if (referenceEnergy < 0.01 || relativeRms > 0.08 || relativeMean > 0.08) {
            return RHITestResult::fail("Single-frame Fuzz integration did not converge: RMS=" + std::to_string(relativeRms));
        }
        auto* deferred = sample.graph.findNode("Deferred");
        deferred->properties["samples"] = 64;
        deferred->properties["materialBinning"] = true;
        sample.graph.markDirty();
        for (int frame = 0; frame < 3; ++frame) {
            if (!preview.render(sample.graph, width, height, "Deferred.color") ||
                preview.lastLog().find("error material") != std::string::npos ||
                preview.lastLog().find("error[E") != std::string::npos ||
                preview.readbackBytes().size() != images[1].size() * sizeof(float)) {
                return RHITestResult::fail("Binned Fuzz render failed: " + preview.lastLog());
            }
            for (size_t i = 0; i < images[1].size(); ++i) {
                float value;
                std::memcpy(&value, preview.readbackBytes().data() + i * sizeof(float), sizeof(float));
                if (!std::isfinite(value) || std::abs(value - images[1][i]) > 1e-5f * std::max(1.0f, std::abs(images[1][i]))) {
                    return RHITestResult::fail("Fuzz IBL changed with frame index or material binning");
                }
            }
        }
        return RHITestResult::pass("M05 Fuzz single-frame convergence, sample budgets, frame stability and binned/unbinned equivalence; HDR captures saved");
    }
};
METALLIC_REGISTER_RHI_TEST(PainterFuzzEnvironmentTest);

class MaterialValuePublicationTest : public RHITest
{
public:
    explicit MaterialValuePublicationTest(bool closure = false) : closure_(closure)
    { name = closure ? "material_value_closure_publication" : "material_value_publication"; type = RHITestType::Resource; }
    RHITestResult run(RHITestContext& context) override
    {
        // Write a composition in the test output directory, preserving the M0 asset.
        const auto scenePath = std::filesystem::absolute(context.outputDirectory / "value-roundtrip.metallic_scene.json");
        const auto sourcePath = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/LookDev/OpenPBRDefault/OpenPbrDefault.gltf";
        nlohmann::json manifest{{"version", 4}, {"sources", {{{"id", "lookdev"}, {"path", sourcePath.generic_string()}}}}};
        { std::ofstream file(scenePath); file << manifest.dump(2); }
        scene::SceneDocument document;
        if (!document.load(scenePath)) { return RHITestResult::fail(document.lastLoadResult().error); }
        const auto original = document.materials()[0];
        auto material = original;
        material.valueProgram = R"({"version":1,"baseColor":{"op":"parameter","index":0}})";
        if (closure_) { material.valueProgram = R"({"version":3,"nodes":{},"outputs":{},"closure":{"op":"slab","reflectance":{"op":"parameter","index":0}}})"; }
        material.valueParameters[0] = 0.42f;
        material.alphaMode = "OPAQUE";
        material.unlit = false;
        material.transmissionFactor = material.diffuseTransmissionFactor = 0;
        std::string log;
        if (!document.setMaterialProperties(0, material) || !document.save(log)) { return RHITestResult::fail(log); }
        scene::SceneDocument reloaded;
        if (!reloaded.load(scenePath) || reloaded.materials()[0].valueProgram != material.valueProgram ||
            reloaded.materials()[0].valueParameters != material.valueParameters) { return RHITestResult::fail("Value source/parameters did not roundtrip"); }
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Material Value publication",
            .enableValidation = context.enableValidation, .enableRayTracingAccelerationStructure = true})
            .transform([&](auto value) { device = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Requires ray tracing"); }
        if (!result) { return RHITestResult::fail("Device failed"); }
        auto& queue = *device->getQueue(QueueType::Graphics);
        ScenePathTraceResources resources;
        if (!resources.beginPrepareAsync(*device, queue, {{"path", scenePath.string()}}, document, log, true)) {
            return RHITestResult::fail(log);
        }
        bool complete = false;
        scene::SceneLoadProgress progress;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!complete && std::chrono::steady_clock::now() < deadline) {
            if (!resources.pumpPrepareAsync(8, progress, log).transform([&](bool value) { complete = value; }) ||
                !queue.waitIdle()) { return RHITestResult::fail(log); }
        }
        const auto initial = resources.materialBinding();
        if (!complete || !initial || !initial->valueBuffer() || initial->values()->instances()[0].parameters[0] != 0.42f) {
            return RHITestResult::fail("Async Value generation was not published");
        }
        if (closure_ && (initial->generation()->instances()[0].program->definition->id != MaterialProgramId::SingleSlab ||
            !MaterialGeneration::create(initial->generation()->parameters(), 123, log, document.materials()))) {
            return RHITestResult::fail("Slab family identity lost in published/reused CPU generation: " + log);
        }
        auto invalid = material;
        invalid.valueProgram = R"({"version":1,"coverage":0})";
        document.setMaterialProperties(0, invalid);
        if (resources.syncRuntimeScene(&document, log) || log.empty() || resources.materialBinding() != initial) {
            return RHITestResult::fail("Invalid program replaced previous generation");
        }
        material.valueParameters[0] = 0.73f;
        document.setMaterialProperties(0, material);
        auto refuse = +[](Device&, const BufferDesc&) -> Result<std::unique_ptr<Buffer>> { return makeError(Error::Failure); };
        if (resources.syncRuntimeScene(&document, log, refuse) || resources.materialBinding() != initial) {
            return RHITestResult::fail("Allocation failure published Value generation");
        }
        if (!resources.syncRuntimeScene(&document, log)) { return RHITestResult::fail(log); }
        auto updated = resources.materialBinding();
        if (updated == initial || updated->values()->key() != initial->values()->key() ||
            updated->values()->instances()[0].parameters[0] != 0.73f || initial->values()->instances()[0].parameters[0] != 0.42f) {
            return RHITestResult::fail("Parameter update violated snapshot/key contract");
        }
        resources.clear();
        if (!initial->valueBuffer() || !updated->valueBuffer()) { return RHITestResult::fail("Held Value generation was destroyed"); }
        return RHITestResult::pass("Source/parameter persistence, async upload, rejection rollback, parameter recovery and held buffers");
    }
private:
    bool closure_;
};
METALLIC_REGISTER_RHI_TEST(MaterialValuePublicationTest);
class MaterialValueClosurePublicationTest final : public MaterialValuePublicationTest
{
public:
    MaterialValueClosurePublicationTest() : MaterialValuePublicationTest(true) {}
};
METALLIC_REGISTER_RHI_TEST(MaterialValueClosurePublicationTest);

std::array<scene::RenderMaterial, 4> probeMaterials()
{
    std::array<scene::RenderMaterial, 4> materials;
    materials[1].valueProgram = R"({"version":1,"baseColor":{"op":"parameter","index":0},"roughness":{"op":"parameter","index":3}})";
    materials[2].valueProgram = materials[1].valueProgram;
    materials[3].valueProgram = R"({"version":1,"baseColor":{"op":"position"},"metallic":{"op":"dot","args":[{"op":"geometryNormal"},[0,0.8,0,0]]}})";
    for (size_t i = 0; i < materials.size(); ++i) {
        for (size_t p = 0; p < 16; ++p) { materials[i].valueParameters[p] = float(i * 16 + p) / 100; }
    }
    return materials;
}

class MaterialValueProbePass final : public ComputePass
{
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        auto& output = reflection.addBufferOutput("result").buffer(4 * 25 * 4, 4).storageWrite();
        output.memoryLocation = MemoryLocation::HostReadback;
        return reflection;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const auto values = MaterialValueProgramSet::create(probeMaterials(), log);
        if (!values) { return makeError(Error::Failure); }
        auto result = context.device->createBuffer({.size = values->instances().size_bytes(),
            .structureStride = sizeof(MaterialValueInstance), .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload}).transform([&](auto buffer) { input_ = std::move(buffer); });
        if (!result) { return result; }
        void* mapped = input_->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, values->instances().data(), values->instances().size_bytes());
        input_->flush(); input_->unmap();
        std::filesystem::path directory;
        if (!values->writeInclude(PROJECT_SOURCE_DIR "/.cache/materials", directory, log)) { return makeError(Error::Failure); }
        const auto path = directory.string();
        const std::array paths{path.c_str()};
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "MaterialValueProbe", .entryPointName = "materialValueProbeMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .additionalSearchPaths = paths}, log)
            .transform([&](auto value) { shader = std::move(value); });
        if (!result) { return result; }
        const std::array bindings{ComputeProgramBindingDesc{.binding = 0, .kind = ComputeResourceBindingKind::StorageBuffer},
            ComputeProgramBindingDesc{.binding = kMaterialValueBinding, .kind = ComputeResourceBindingKind::StorageBuffer}};
        const ComputeResourceField fields[] = {
            {0, ComputeResourceBindingKind::StorageBuffer, offsetof(SceneResourceParameters, probeOutput)},
            {kMaterialValueBinding, ComputeResourceBindingKind::StorageBuffer, offsetof(SceneResourceParameters, materialValues)}};
        return program_.initialize(*context.device, {.spirv = shader.spirv, .bindings = bindings, .requiresRayQuery = false,
            .resourceParameters = {sizeof(SceneResourceParameters), fields}}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const std::array bindings{ComputeDispatchBinding{.binding = 0, .buffer = context.outputBuffer("result").buffer()},
            ComputeDispatchBinding{.binding = kMaterialValueBinding, .buffer = input_.get()}};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings});
    }
private:
    std::unique_ptr<Buffer> input_;
    ComputeProgram program_;
};

class MaterialValueGPUTest final : public RHITest
{
public:
    MaterialValueGPUTest() { name = "material_value_gpu_abi"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        registerRenderGraphPassType("MaterialValueProbe", "Value ABI", [] { return std::make_unique<MaterialValueProbePass>(); });
        std::unique_ptr<Device> device;
        auto result = createDevice({.applicationName = "Value GPU ABI", .enableValidation = context.enableValidation,
            .enableBindlessDescriptorHeap = true}).transform([&](auto value) { device = std::move(value); });
        if (hasError(result, Error::Unsupported)) { return RHITestResult::skip("Requires bindless"); }
        if (!result) { return RHITestResult::fail("Device failed"); }
        std::string log;
        RenderGraph graph;
        graph.addNode("MaterialValueProbe", "Probe"); graph.markOutput("Probe.result");
        RenderGraphExecutor executor;
        if (!executor.compile(*device, graph, 1, 1, log)) { return RHITestResult::fail(log); }
        if (!executor.execute({.graphicsQueue = device->getQueue(QueueType::Graphics)}) || !executor.waitForSubmittedWork()) {
            return RHITestResult::fail("Probe execution failed");
        }
        auto* output = executor.outputResource("Probe.result")->buffer;
        output->invalidate();
        const void* mapped = output->map();
        if (!mapped) { return RHITestResult::fail("Probe readback failed"); }
        std::array<float, 100> actual{};
        std::memcpy(actual.data(), mapped, sizeof(actual)); output->unmap();
        auto materials = probeMaterials();
        auto set = MaterialValueProgramSet::create(materials, log);
        for (size_t i = 0; i < 4; ++i) {
            std::array<float, 25> expected{};
            expected[0] = float(set->instances()[i].programId);
            std::copy(materials[i].valueParameters.begin(), materials[i].valueParameters.end(), expected.begin() + 1);
            const std::array<float, 8> model{0.2f, 0.3f, 0.4f, 0.75f, 0.1f, 0.2f, 0.3f, 0.4f};
            std::copy(model.begin(), model.end(), expected.begin() + 17);
            if (i == 1 || i == 2) {
                std::copy_n(materials[i].valueParameters.begin(), 3, expected.begin() + 17);
                expected[22] = materials[i].valueParameters[12];
            } else if (i == 3) { expected[17] = 0.25f; expected[18] = 0.5f; expected[19] = 0.75f; expected[21] = 0.8f; }
            if (i != 0) {
                const auto working = color::fromLinearRec709({expected[17], expected[18], expected[19]});
                std::copy(working.begin(), working.end(), expected.begin() + 17);
            }
            for (size_t p = 0; p < 25; ++p) {
                if (std::abs(actual[i * 25 + p] - expected[p]) > 1e-6f || !std::isfinite(actual[i * 25 + p])) {
                    return RHITestResult::fail("Value ABI/evaluation mismatch instance=" + std::to_string(i) + " field=" + std::to_string(p));
                }
            }
        }
        return RHITestResult::pass("GPU 80-byte stride, all 16 parameter offsets, default and two custom programs, geometry input and alpha preservation");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialValueGPUTest);

class MaterialValueLookDevTest final : public RHITest
{
public:
    MaterialValueLookDevTest() { name = "material_value_lookdev"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        std::string log;
        RenderSampleLoadResult sample;
        if (!loadBuiltInRenderSample("lookdev-vbuffer", sample, log)) { return RHITestResult::fail(log); }
        for (const char* pass : {"Reference", "Deferred"}) {
            auto* node = sample.graph.findNode(pass);
            node->properties["debugView"] = "baseColor";
            node->properties["accumulate"] = false;
            node->properties["samples"] = 1;
        }
        scene::SceneDocument document;
        if (!document.load(std::filesystem::path(PROJECT_SOURCE_DIR) / sample.desc.scenePath)) { return RHITestResult::fail("Load failed"); }
        std::vector<scene::RenderMaterial> originals(document.materials().begin(), document.materials().end());
        std::vector<int32_t> edited;
        for (int32_t i = 0; i < static_cast<int32_t>(originals.size()); ++i) {
            const auto& original = originals[i];
            if (original.alphaMode == "OPAQUE" && !original.rtxcrHair && !original.unlit &&
                original.transmissionFactor == 0 && original.diffuseTransmissionFactor == 0) { edited.push_back(i); }
        }
        if (edited.empty()) { return RHITestResult::fail("No eligible LookDev materials"); }
        RenderGraphPreviewRenderer preview;
        preview.bindRuntimeScene(&document);
        preview.setEnvironment(document.environment());
        preview.setLighting(document.lighting());
        if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Device failed"); }
        preview.setRawReadbackEnabled(true);
        std::array<std::vector<std::byte>, 2> baseline, custom, parameters;
        auto capture = [&](const std::string& label, auto& images) -> bool {
            uint32_t index = 0;
            for (const char* output : {"Reference.color", "Deferred.color"}) {
                if (!preview.render(sample.graph, 256, 256, output)) { log = preview.lastLog(); return false; }
                if (preview.readbackFormat() != Format::RGBA32Sfloat) { log = "Expected float32 LookDev output"; return false; }
                images[index++] = preview.readbackBytes();
                for (size_t offset = 0; offset < preview.readbackBytes().size(); offset += 4) {
                    float v; std::memcpy(&v, preview.readbackBytes().data() + offset, 4);
                    if (!std::isfinite(v)) { log = "Nonfinite LookDev value"; return false; }
                }
                if (!saveRgba8Png(context.outputDirectory / (label + "-" + output + ".png"),
                    reinterpret_cast<const uint8_t*>(preview.pixels().data()), 256, 256, log)) { return false; }
            }
            return true;
        };
        if (!capture("baseline", baseline)) { return RHITestResult::fail(log); }
        for (auto i : edited) {
            auto material = originals[i];
            material.valueProgram = readProgram("ProceduralRust.value.json");
            material.valueParameters = {0.12f, 0.25f, 0.38f, 1, 0.85f, 0.12f, 0.015f, 1, 7, 19, 13, 0, 0.7f, 0, 0, 0};
            if (!document.setMaterialProperties(i, material)) { return RHITestResult::fail("Program edit failed"); }
        }
        const auto set = MaterialValueProgramSet::create(document.materials(), log);
        if (!capture("rust", custom)) { return RHITestResult::fail(log); }
        for (auto i : edited) {
            auto material = document.materials()[i];
            material.valueParameters[0] = 0.9f;
            material.valueParameters[1] = 0.05f;
            document.setMaterialProperties(i, material);
        }
        const auto updated = MaterialValueProgramSet::create(document.materials(), log);
        if (!set || !updated || set->key() != updated->key()) { return RHITestResult::fail("Parameter edit changed program key"); }
        if (!capture("parameters", parameters)) { return RHITestResult::fail(log); }
        for (size_t path = 0; path < 2; ++path) {
            if (baseline[path] == custom[path] || custom[path] == parameters[path]) { return RHITestResult::fail("Value edit did not affect both rendering paths"); }
        }
        for (auto i : edited) {
            auto material = document.materials()[i];
            material.valueProgram = readProgram("Stripes.value.json");
            document.setMaterialProperties(i, material);
        }
        if (!capture("stripes", parameters)) { return RHITestResult::fail(log); }
        for (auto i : edited) { document.setMaterialProperties(i, originals[i]); }
        if (!capture("restored", parameters)) { return RHITestResult::fail(log); }
        if (baseline != parameters) { return RHITestResult::fail("Removing custom programs failed to restore exact baseline"); }
        for (auto i : edited) {
            auto material = originals[i];
            material.valueProgram = readProgram("ProceduralRust.value.json");
            material.valueParameters = {0.12f, 0.25f, 0.38f, 1, 0.85f, 0.12f, 0.015f, 1, 7, 19, 13, 0, 0.7f, 0, 0, 0};
            document.setMaterialProperties(i, material);
        }
        for (const char* pass : {"Reference", "Deferred"}) {
            auto* node = sample.graph.findNode(pass);
            node->properties["debugView"] = "final";
            node->properties["accumulate"] = true;
        }
        sample.graph.markDirty();
        for (uint32_t frame = 0; frame < 32; ++frame) {
            if (!preview.render(sample.graph, 256, 256, "Reference.color", false)) { return RHITestResult::fail(preview.lastLog()); }
        }
        if (!capture("rust-lit", parameters)) { return RHITestResult::fail(log); }
        return RHITestResult::pass("Existing LookDev PT/VBuffer: program/parameter edits, exact baseline restoration, 32 lit frames and finite HDR");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialValueLookDevTest);

class MaterialValueClosureSceneTest final : public RHITest
{
public:
    MaterialValueClosureSceneTest() { name = "material_value_closure_scene"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        using Json = nlohmann::json;
        try {
            std::string log;
            const auto check = [&](const auto& condition, const std::string& message) {
                if (!condition) { throw std::runtime_error(message + ": " + log); }
            };
            const auto root = std::filesystem::absolute(context.outputDirectory);
            const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/LookDev/OpenPBRDefault";
            std::ifstream file(source / "OpenPbrDefault.gltf");
            auto gltf = Json::parse(file);
            for (auto& buffer : gltf["buffers"]) {
                const auto name = buffer["uri"].get<std::string>();
                std::filesystem::copy_file(source / name, root / name, std::filesystem::copy_options::overwrite_existing);
            }
            gltf["materials"].push_back(gltf["materials"][0]);
            gltf["materials"].push_back(gltf["materials"][0]);
            gltf["meshes"][1]["primitives"][0]["material"] = 1;
            // Emissive Slab behind the camera: no primary ray can see it. With
            // no environment/lights, nonzero receiver radiance proves secondary
            // hits execute the scene's dynamic Slab program and emission.
            const std::array<float, 18> vertices{-20,-20,5, -20,20,5, 20,20,5, -20,-20,5, 20,20,5, 20,-20,5};
            { std::ofstream binary(root / "emitter.bin", std::ios::binary); binary.write(reinterpret_cast<const char*>(vertices.data()), sizeof(vertices)); }
            const auto buffer = gltf["buffers"].size(), view = gltf["bufferViews"].size(), accessor = gltf["accessors"].size();
            gltf["buffers"].push_back({{"uri", "emitter.bin"}, {"byteLength", sizeof(vertices)}});
            gltf["bufferViews"].push_back({{"buffer", buffer}, {"byteLength", sizeof(vertices)}});
            gltf["accessors"].push_back({{"bufferView", view}, {"componentType", 5126}, {"count", 6}, {"type", "VEC3"},
                {"min", {-20,-20,5}}, {"max", {20,20,5}}});
            gltf["meshes"].push_back({{"primitives", {{{"attributes", {{"POSITION", accessor}}}, {"material", 2}}}}});
            gltf["nodes"].push_back({{"mesh", 2}, {"name", "Secondary-only Slab emitter"}});
            gltf["scenes"][0]["nodes"].push_back(gltf["nodes"].size() - 1);
            const auto scenePath = root / "ClosureScene.gltf";
            { std::ofstream sceneFile(scenePath); sceneFile << gltf.dump(2); }
            // Avoid inheriting overrides from a previous run in this fixture directory.
            std::filesystem::remove(root / "ClosureScene.metallic_scene.json");
            const Json parameter{{"op", "parameter"}, {"index", 0}};
            const Json slab{{"op", "slab"}, {"reflectance", parameter}};
            const Json texturedSlab{{"op", "slab"}, {"reflectance", {{"op", "mul"}, {"args", {parameter,
                {{"op", "textureSample"}, {"texture", "baseColor"}, {"footprint", "RayCone"}, {"args", {{{"op", "uv"}}, 0}}}}}}}};
            const Json black{{"op", "slab"}, {"reflectance", 0}, {"opticalDepth", {{"op", "parameter"}, {"index", 1}}}};
            const Json mix{{"op", "mix"}, {"a", slab}, {"b", {{"op", "slab"}, {"reflectance", {0.8,0.1,0.05,0}}}},
                {"weight", {{"op", "parameter"}, {"index", 1}}}};
            const Json layer{{"op", "layer"}, {"a", black}, {"b", slab}};
            material::MaterialAssetLibrary library(root);
            const std::array<uint8_t, 16> white{255,255,255,255, 255,255,255,255, 255,255,255,255, 255,255,255,255};
            const std::array<uint8_t, 16> pattern{255,40,10,255, 10,255,40,255, 10,40,255,255, 255,255,255,255};
            check(saveRgba8Png(root / "White.png", white.data(), 2, 2, log), "Write white texture");
            check(saveRgba8Png(root / "Pattern.png", pattern.data(), 2, 2, log), "Write pattern texture");
            for (const auto& [name, closure] : std::array<std::pair<std::string, Json>, 3>{{{"Single", texturedSlab}, {"Mix", mix}, {"Layer", layer}}}) {
                auto definition = material::defaultOpenPBRDefinition();
                definition.implementation = "Slab.Surface";
                definition.surfaceProgram = Json{{"version", 3}, {"nodes", Json::object()},
                    {"outputs", {{"emissive", {{"op", "parameter"}, {"index", 2}}}}}, {"closure", closure}}.dump();
                definition.valueParameters = {{"0", {0.35,0.45,0.55,0}}};
                { std::ofstream definitionFile(root / (name + ".materialdef")); definitionFile << material::serializeMaterialDefinition(definition); }
                material::MaterialInstance instance; instance.definition = "asset://" + name + ".materialdef";
                if (name == "Single") { instance.resources["baseColorTexture"] = "asset://White.png"; }
                check(library.save("asset://" + name + ".material", instance, log), "Save Slab asset");
                if (name == "Single") {
                    instance.resources["baseColorTexture"] = "asset://Pattern.png";
                    check(library.save("asset://Textured.material", instance, log), "Save texture variant");
                }
            }
            scene::SceneDocument document;
            const bool loaded = document.load(scenePath);
            check(loaded, "Load scene " + document.lastLoadResult().error);
            for (int32_t i = 0; i < 3; ++i) {
                check(document.setMaterialAsset(i, std::array{"asset://Single.material", "asset://Mix.material", "asset://Layer.material"}[i], root, log), "Bind asset");
            }
            check(document.save(log), "Save asset scene");
            scene::SceneDocument reloaded;
            check(reloaded.load(document.documentPath()) && reloaded.documentWarning().empty(), "Reload asset scene");
            check(reloaded.materials()[0].valueProgram == document.materials()[0].valueProgram, "Lost persistent Closure program");
            RenderSampleLoadResult sample;
            check(loadBuiltInRenderSample("lookdev-vbuffer", sample, log), "Load graph");
            for (const char* pass : {"Reference", "Deferred", "VBuffer"}) { sample.graph.findNode(pass)->properties["path"] = scenePath.string(); }
            for (const char* pass : {"Reference", "Deferred"}) {
                auto& p = sample.graph.findNode(pass)->properties;
                p["accumulate"] = false; p["samples"] = 1; p["maxDepth"] = 1; p["outputLinear"] = true;
                // The secondary-only emitter is an opaque wall behind the
                // camera; exclude its sun shadow from direct BSDF comparisons.
                p["debugDisableShadows"] = true;
            }
            RenderGraphPreviewRenderer preview;
            preview.bindRuntimeScene(&reloaded);
            scene::EnvironmentSettings environment; environment.enabled = false;
            scene::LightingSettings lighting; lighting.autoExposure.enabled = false;
            scene::PunctualLight sun; sun.properties.type = "directional"; sun.properties.intensity = 2;
            sun.direction = float3(-0.4f, -0.6f, -1); lighting.lights.push_back(sun);
            preview.setEnvironment(environment); preview.setLighting(lighting); preview.setRawReadbackEnabled(true);
            check(preview.initialize(context.enableValidation, true, false), "Initialize device");
            const auto capture = [&](const char* output, const std::string& label) {
                const auto rendered = preview.render(sample.graph, 128, 128, output);
                check(rendered, "Render " + label + " " + preview.lastLog());
                const auto& bytes = preview.readbackBytes();
                check(preview.readbackFormat() == Format::RGBA32Sfloat && bytes.size() == 128*128*16, "HDR readback");
                std::vector<float> pixels(bytes.size()/4); std::memcpy(pixels.data(), bytes.data(), bytes.size());
                for (float value : pixels) { check(std::isfinite(value), "Nonfinite Slab image"); }
                std::ofstream hdr(root / (label + ".rgba32f"), std::ios::binary); hdr.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                check(saveRgba8Png(root / (label + ".png"), reinterpret_cast<const uint8_t*>(preview.pixels().data()), 128, 128, log), "Save evidence");
                return pixels;
            };
            const auto energy = [](const auto& pixels) { double sum = 0; for (size_t i = 0; i < pixels.size(); ++i) { if (i % 4 != 3) { sum += std::abs(pixels[i]); } } return sum; };
            const auto error = [](const auto& a, const auto& b, double scale = 1.0) {
                double e = 0, v = 0; for (size_t i = 0; i < a.size(); ++i) { if (i % 4 != 3) { e += std::abs(a[i] - scale*b[i]); v += std::abs(a[i]); } }
                return e / std::max(v, 1e-20);
            };
            const auto initial = MaterialValueProgramSet::create(reloaded.materials(), log);
            check(initial && initial->programCount() == 3, "Expected three actual Closure programs");
            const auto directPT = capture("Reference.color", "direct-pt");
            const auto direct = capture("Deferred.color", "direct-binned");
            check(energy(directPT) > 1 && energy(direct) > 1, "Slab direct lighting is empty");
            sample.graph.findNode("Deferred")->properties["materialBinning"] = false; sample.graph.markDirty();
            const auto unbinned = capture("Deferred.color", "direct-unbinned");
            check(error(direct, unbinned) < 1e-5, "Actual Program/Family bins changed lighting");
            sample.graph.findNode("Deferred")->properties["materialBinning"] = true; sample.graph.markDirty();
            sample.graph.findNode("Deferred")->properties["programBinning"] = false;
            const auto legacyFallback = capture("Deferred.color", "legacy-bin-fallback");
            check(error(direct, legacyFallback) < 1e-5, "Legacy class schedule failed to fall back for custom programs");
            sample.graph.findNode("Deferred")->properties["programBinning"] = true; sample.graph.markDirty();
            // Mix at weight zero and transparent black top Layer at tau zero
            // must both equal the same single diffuse receiver, in scene lighting.
            check(reloaded.setMaterialAsset(1, "asset://Layer.material", root, log), "Switch Mix to Layer");
            const auto transparent = capture("Deferred.color", "transparent-layer");
            check(error(direct, transparent) < 1e-5, "Layer zero-depth boundary differs from Mix endpoint");
            const auto transparentPT = capture("Reference.color", "transparent-layer-pt");
            check(error(directPT, transparentPT) < 1e-5, "PT Layer boundary differs from Mix endpoint");
            auto edited = reloaded.materials()[1]; edited.valueParameters[4] = edited.valueParameters[5] = edited.valueParameters[6] = 2;
            check(reloaded.setMaterialProperties(1, edited), "Edit optical depth");
            const auto absorbed = capture("Deferred.color", "absorbing-layer");
            check(energy(absorbed) < energy(transparent) * 0.995, "Dynamic optical depth did not attenuate actual Layer lighting");
            const auto absorbedPT = capture("Reference.color", "absorbing-layer-pt");
            check(energy(absorbedPT) < energy(transparentPT) * 0.995, "PT ignored dynamic Layer absorption");
            check(reloaded.setMaterialAsset(1, "asset://Mix.material", root, log), "Restore Mix");
            edited = reloaded.materials()[1]; edited.valueParameters[4] = 1;
            check(reloaded.setMaterialProperties(1, edited), "Edit Mix weight");
            const auto mixed = capture("Deferred.color", "mix-endpoint");
            check(error(direct, mixed) > 0.001, "Dynamic Mix weight did not affect scattering");
            const auto updated = MaterialValueProgramSet::create(reloaded.materials(), log);
            check(updated && updated->key() == initial->key(), "Dynamic values changed code identity");
            check(reloaded.setMaterialAsset(0, "asset://Textured.material", root, log), "Bind texture variant");
            const auto textured = capture("Deferred.color", "textured-slab");
            check(error(mixed, textured) > 0.01, "Value texture did not reach Slab scattering");
            check(MaterialValueProgramSet::create(reloaded.materials(), log)->key() == initial->key(), "Texture resource changed program identity");
            // Only secondary Slab emission now illuminates the receivers.
            lighting.lights.clear(); preview.setLighting(lighting);
            edited = reloaded.materials()[2]; edited.valueParameters[8] = edited.valueParameters[9] = edited.valueParameters[10] = 1;
            check(reloaded.setMaterialProperties(2, edited), "Edit hidden emitter");
            const auto primaryOnly = capture("Reference.color", "primary-only");
            check(energy(primaryOnly) < 1e-5, "Emitter leaked into primary visibility");
            sample.graph.findNode("Reference")->properties["maxDepth"] = 3; sample.graph.markDirty();
            sample.graph.findNode("Reference")->properties["samples"] = 8;
            // PT seeds use global frameIndex even with accumulation disabled.
            // Recreate the renderer for each oracle image to replay frame zero.
            const auto restart = [&] {
                preview = RenderGraphPreviewRenderer{};
                preview.bindRuntimeScene(&reloaded); preview.setEnvironment(environment); preview.setLighting(lighting);
                check(preview.subsystemHost()->configure<EnvironmentLightingSubsystem>({.initialDecodeTimeoutMilliseconds = 10000}, log), "Configure environment");
                preview.setRawReadbackEnabled(true);
                check(preview.initialize(context.enableValidation, true, false), "Restart deterministic renderer");
            };
            restart();
            const auto secondary = capture("Reference.color", "secondary-emission");
            check(energy(secondary) > 1, "Secondary hit did not execute Slab program");
            edited.valueParameters[8] = edited.valueParameters[9] = edited.valueParameters[10] = 2;
            check(reloaded.setMaterialProperties(2, edited), "Edit secondary emission");
            restart();
            const auto doubled = capture("Reference.color", "secondary-emission-double");
            check(error(doubled, secondary, 2) < 1e-5, "Secondary dynamic inputs violate linear emission oracle");
            // Actual realtime Slab IBL uses its BSDF, including Layer absorption.
            environment.enabled = true; environment.path = root / "White.png";
            check(reloaded.setMaterialAsset(1, "asset://Layer.material", root, log), "Bind IBL Layer");
            restart();
            const auto ibl = capture("Deferred.color", "environment-layer");
            const auto& environmentSnapshot = preview.subsystemHost()->get<EnvironmentLightingSubsystem>()->snapshot();
            check(environmentSnapshot.status == EnvironmentLightingStatus::Ready && environmentSnapshot.mapAvailable, "IBL source not ready");
            edited = reloaded.materials()[1]; edited.valueParameters[4] = edited.valueParameters[5] = edited.valueParameters[6] = 2;
            check(reloaded.setMaterialProperties(1, edited), "Edit IBL optical depth");
            const auto iblAbsorbed = capture("Deferred.color", "environment-layer-absorbed");
            check(energy(iblAbsorbed) < energy(ibl) * 0.999, "IBL ignored actual Layer closure");
            // M8: a ray-only graph can mix Fiber receivers and Surface Slab
            // secondary hits. No opaque VBuffer or screen closure is available.
            auto rayProperties = sample.graph.findNode("Reference")->properties;
            const auto rayType = sample.graph.findNode("Reference")->type;
            sample.graph = RenderGraph{};
            sample.graph.addNode(rayType, "Reference", rayProperties);
            sample.graph.markOutput("Reference.color");
            check(reloaded.setMaterialAsset(0, "asset://Materials/Examples/ChestnutFiber.material",
                PROJECT_SOURCE_DIR "/Asset", log), "Bind mixed Fiber receiver");
            environment.enabled = false;
            edited = reloaded.materials()[2];
            edited.valueParameters[8] = edited.valueParameters[9] = edited.valueParameters[10] = 1;
            check(reloaded.setMaterialProperties(2, edited), "Set mixed secondary emitter");
            sample.graph.findNode("Reference")->properties["maxDepth"] = 1; sample.graph.markDirty();
            restart();
            check(energy(capture("Reference.color", "mixed-primary-only")) < 1e-5, "Mixed emitter visible to primary rays");
            sample.graph.findNode("Reference")->properties["maxDepth"] = 3; sample.graph.markDirty();
            restart();
            const auto mixedSecondary = capture("Reference.color", "mixed-fiber-surface-secondary");
            check(energy(mixedSecondary) > 1, "Mixed ray domains produced no secondary radiance");
            edited.valueParameters[8] = edited.valueParameters[9] = edited.valueParameters[10] = 2;
            check(reloaded.setMaterialProperties(2, edited), "Double mixed emitter");
            restart();
            const auto mixedDoubled = capture("Reference.color", "mixed-fiber-surface-secondary-double");
            check(error(mixedDoubled, mixedSecondary, 2) < 1e-5, "Mixed ray execution reused wrong instance/prepared state");
            auto fiberReceiver = reloaded.materials()[0]; fiberReceiver.rtxcrHairMelanin = 0.05f;
            check(reloaded.setMaterialProperties(0, fiberReceiver), "Edit secondary-lit Fiber receiver");
            restart();
            const auto mixedFiberEdited = capture("Reference.color", "mixed-fiber-surface-melanin");
            check(error(mixedFiberEdited, mixedDoubled) > 0.001, "Mixed indirect radiance did not execute Fiber scattering");
            std::ofstream(root / "ClosureSceneAcceptance.json") << Json{{"programs", 3}, {"binRelativeError", error(direct, unbinned)},
                {"ptLayerBoundaryError", error(directPT, transparentPT)},
                {"layerBoundaryError", error(direct, transparent)}, {"secondaryEnergy", energy(secondary)},
                {"secondaryLinearityError", error(doubled, secondary, 2)}, {"mixedSecondaryEnergy", energy(mixedSecondary)},
                {"mixedSecondaryLinearityError", error(mixedDoubled, mixedSecondary, 2)},
                {"mixedFiberEditRelativeDifference", error(mixedFiberEdited, mixedDoubled)}}.dump(2);
            return RHITestResult::pass("Persistent Single/Mix/Layer assets; actual sparse/fallback VBuffer equivalence; dynamic scattering; PT secondary-only emission and linearity oracle");
        } catch (const std::exception& error) { return RHITestResult::fail(error.what()); }
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialValueClosureSceneTest);
} // namespace
} // namespace metallic::tests

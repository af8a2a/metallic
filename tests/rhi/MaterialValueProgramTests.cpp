#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Core/NamedResourceLayouts.h"
#include "RHITest.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Render/Streamer/ScenePathTraceResources.h"
#include "Runtime/Scene/SceneDocument.h"

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
        if (MaterialValueProgramSet::create(materials, log)) { return RHITestResult::fail("Custom mask accepted"); }
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

class MaterialValuePublicationTest final : public RHITest
{
public:
    MaterialValuePublicationTest() { name = "material_value_publication"; type = RHITestType::Resource; }
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
};
METALLIC_REGISTER_RHI_TEST(MaterialValuePublicationTest);

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
} // namespace
} // namespace metallic::tests

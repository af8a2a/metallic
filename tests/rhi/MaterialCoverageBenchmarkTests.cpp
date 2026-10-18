#include "RHITest.h"
#include "TestComputeProgram.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/Material/MaterialCoverageProgram.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Render/Material/MaterialRuntime.h"
#include "Runtime/Render/RenderGraph/RenderGraph.h"
#include "Runtime/Render/RenderSample.h"
#include "Runtime/Scene/SceneDocument.h"
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <initializer_list>

namespace metallic::tests {
namespace {
using namespace render;
using Json = nlohmann::json;

Json expression(const char* op, std::initializer_list<Json> args)
{
    // A single Json in braces can select Json's copy constructor. Build the
    // argument array explicitly so unary operators always receive [operand].
    Json array = Json::array();
    for (const auto& argument : args) { array.push_back(argument); }
    return {{"op", op}, {"args", std::move(array)}};
}

std::vector<scene::RenderMaterial> coverageBenchmarkMaterials(const std::string& workload)
{
    const Json uv{{"op", "uv"}}, alpha{{"op", "alpha"}}, parameter{{"op", "parameter"}, {"index", 0}};
    std::vector<Json> roots;
    if (workload == "control") { roots.emplace_back(); }
    else if (workload == "simple") { roots.push_back(expression("mul", {alpha, expression("add", {uv, parameter})})); }
    else if (workload == "complex" || workload == "mixed") {
        for (int variant = 0; variant < (workload == "mixed" ? 8 : 1); ++variant) {
            Json value = expression("add", {uv, parameter});
            // Four nested operators per layer plus input/output expressions:
            // five layers fit the authoring depth limit of 24 without relaxing it.
            for (int layer = 0; layer < 5; ++layer) {
                value = expression("fract", {expression("add", {
                    expression("sin", {expression("mul", {value, 1.25 + layer * 0.125})}),
                    0.375 + variant * 0.015625})});
            }
            roots.push_back(expression("mul", {alpha, value}));
        }
    } else {
        // Exercise every opcode, arbitrary swizzle/dot dependencies and large UV
        // leaves, including near-zero normalization and negative Select predicates.
        const Json transformed = expression("uvTransform", {uv, Json::array({0.25, -0.5, 0.125, 0}), Json::array({0.5, 0.25, -0.125, 0})});
        Json swizzle = expression("swizzle", {transformed}); swizzle["components"] = "yxzy";
        roots = {0.75, parameter, uv, alpha,
            expression("add", {uv, parameter}), expression("mul", {uv, parameter}),
            expression("dot", {swizzle, parameter}), expression("mix", {uv, parameter, alpha}),
            expression("sin", {uv}), expression("fract", {uv}), expression("abs", {uv}),
            expression("saturate", {uv}), expression("clamp", {uv, -0.25, 0.75}),
            expression("normalize", {parameter}), expression("normalMap", {parameter}),
            transformed, swizzle, expression("select", {parameter, alpha, uv})};
        Json normalizedZ = expression("swizzle", {expression("normalize", {parameter})});
        normalizedZ["components"] = "zzzz";
        roots.push_back(normalizedZ);
    }
    std::vector<scene::RenderMaterial> materials;
    for (size_t i = 0; i < roots.size(); ++i) {
        scene::RenderMaterial material;
        material.alphaMode = "MASK";
        if (!roots[i].is_null()) { material.valueProgram = Json{{"version", 1}, {"coverage", roots[i]}}.dump(); }
        material.valueParameters = {0.125f, 0.25f, 0.75f, -0.5f};
        if (workload == "edge") {
            if (i == 13 || i == 18) { material.valueParameters = {1e-12f, 0, 0, 0}; }
            if (i == 14) { material.valueParameters = {0.5f, 0.5f, 0.5f, 0.5f}; }
            if (i == 17) { material.valueParameters[0] = -1; }
        }
        materials.push_back(std::move(material));
    }
    return materials;
}

struct CoverageBenchmarkState
{
    std::string workload, arm;
    uint32_t size = 512, repeats = 4;
    std::vector<scene::RenderMaterial> materials;
    std::filesystem::path output;
};

class CoverageBenchmarkPass final : public ComputePass
{
public:
    explicit CoverageBenchmarkPass(std::shared_ptr<CoverageBenchmarkState> state) : state_(std::move(state)) {}
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureOutput("color").storageWrite().format = Format::RGBA32Sfloat;
        return reflection;
    }
    Result<> compile(const RenderGraphCompileContext& context, std::string& log) override
    {
        const auto set = MaterialValueProgramSet::create(state_->materials, log);
        if (!set) { return makeError(Error::Failure); }
        std::vector<std::byte> bytes(set->inputBytes().begin(), set->inputBytes().end());
        if (state_->arm == "legacy") {
            // Reconstruct the exact old transport. Production no longer uploads
            // bytecode. Both arms use the same IR and instance parameter values.
            std::vector<MaterialValueInstance> instances(set->instances().begin(), set->instances().end());
            for (size_t i = 0; i < state_->materials.size(); ++i) {
                if (state_->materials[i].valueProgram.empty()) { continue; }
                const auto slice = compileMaterialCoverageSlice(MaterialValueIR::parse(state_->materials[i].valueProgram).slice(true));
                instances[i].coverageOffset = static_cast<uint32_t>(bytes.size());
                const auto* begin = reinterpret_cast<const std::byte*>(slice.instructions.data());
                bytes.insert(bytes.end(), begin, begin + slice.instructions.size() * sizeof(MaterialCoverageInstruction));
            }
            std::memcpy(bytes.data(), instances.data(), instances.size() * sizeof(MaterialValueInstance));
        }
        auto result = context.device->createBuffer({.size = bytes.size(), .structureStride = 4,
            .usage = BufferUsageBits::Storage, .memoryLocation = MemoryLocation::HostUpload})
            .transform([&](auto value) { input_ = std::move(value); });
        if (!result) { return result; }
        void* mapped = input_->map();
        if (!mapped) { return makeError(Error::Failure); }
        std::memcpy(mapped, bytes.data(), bytes.size()); input_->flush(); input_->unmap();
        std::filesystem::path directory;
        if (!set->writeCoverageInclude(PROJECT_SOURCE_DIR "/.cache/materials", directory, log)) { return makeError(Error::Failure); }
        const auto path = directory.string();
        const std::array paths{path.c_str()};
        std::vector<SlangMacroDefine> macros{{"METALLIC_TEST_LEGACY_COVERAGE", state_->arm == "legacy" ? "1" : "0"}};
        if (state_->arm == "native" && set->coverageProgramCount()) { macros.push_back({"METALLIC_CUSTOM_COVERAGE", "1"}); }
        ShaderCompileResult shader;
        result = compileSlangShaderToSpirv({.moduleName = "MaterialCoverageBenchmark", .entryPointName = "materialCoverageBenchmarkMain",
            .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders", .additionalSearchPaths = paths, .macroDefines = macros}, log)
            .transform([&](auto value) { shader = std::move(value); });
        if (!result) { return result; }
        std::ofstream(state_->output / (state_->workload + ".spv"), std::ios::binary)
            .write(reinterpret_cast<const char*>(shader.spirv.data()), shader.spirv.size() * sizeof(uint32_t));
        std::ofstream(state_->output / (state_->workload + ".coverage.slang")) << set->coverageSource();
        const std::array bindings{
            ComputeResourceBindingDesc{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, output), .kind = ComputeResourceBindingKind::StorageImage},
            ComputeResourceBindingDesc{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues), .kind = ComputeResourceBindingKind::StorageBuffer}};
        return program_.initialize(*context.device, {.spirv = shader.spirv, .pushConstantSize = sizeof(uint32_t) * 4,
            .bindings = bindings, .resourceParameterSize = sizeof(SceneResourceParameters)}, log);
    }
    Result<> execute(RenderGraphExecutionContext& context) override
    {
        const std::array bindings{
            ComputeDispatchBinding{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, output), .textureView = context.outputTexture("color").view()},
            ComputeDispatchBinding{.binding = METALLIC_RESOURCE_MEMBER(SceneResourceParameters, materialValues), .buffer = input_.get()}};
        const std::array<uint32_t, 4> push{state_->size, static_cast<uint32_t>(state_->materials.size()), state_->repeats, state_->workload == "edge" ? 1u : 0u};
        return program_.dispatch({.commandBuffer = &context.commandBuffer(), .bindings = bindings,
            .pushData = push.data(), .pushDataSize = sizeof(push), .groupCountX = (state_->size + 7) / 8, .groupCountY = (state_->size + 7) / 8});
    }
private:
    std::shared_ptr<CoverageBenchmarkState> state_;
    std::unique_ptr<Buffer> input_;
    ComputeProgram program_;
};

class MaterialCoverageBenchmarkTest final : public RHITest
{
public:
    MaterialCoverageBenchmarkTest() { name = "material_coverage_codegen_benchmark"; type = RHITestType::Rendering; }
    RHITestResult run(RHITestContext& context) override
    {
        const char* requested = std::getenv("METALLIC_COVERAGE_BENCHMARK_ARM");
        if (!requested) { return RHITestResult::skip("Opt in with METALLIC_COVERAGE_BENCHMARK_ARM=native or legacy"); }
        const std::string arm(requested);
        if (arm != "native" && arm != "legacy") { return RHITestResult::fail("Invalid Coverage benchmark arm"); }
        Json report = {{"version", 1}, {"arm", arm}, {"validation", context.enableValidation},
            {"scope", "Coverage evaluator compute dispatch; excludes rasterization, traversal, lighting and presentation"},
            {"warmupFrames", 16}, {"sampleFrames", 32}, {"cases", Json::array()}};
        for (const char* workload : {"control", "simple", "complex", "mixed", "edge"}) {
            const auto state = std::make_shared<CoverageBenchmarkState>();
            state->workload = workload; state->arm = arm; state->output = context.outputDirectory;
            state->materials = coverageBenchmarkMaterials(workload);
            Json materials = Json::array();
            for (const auto& material : state->materials) { materials.push_back({{"expression", material.valueProgram}, {"parameters", material.valueParameters}}); }
            registerRenderGraphPassType("CoverageBenchmark", "Coverage evaluator benchmark", [state] { return std::make_unique<CoverageBenchmarkPass>(state); });
            RenderGraph graph;
            graph.addNode("CoverageBenchmark", "Coverage"); graph.markOutput("Coverage.color");
            RenderGraphPreviewRenderer preview;
            preview.setEnvironment({.enabled = false});
            if (!preview.initialize(context.enableValidation, true, false)) { return RHITestResult::fail("Coverage benchmark device failed"); }
            Json item = {{"id", workload}, {"size", state->size}, {"repeats", state->repeats}, {"materials", materials},
                {"frames", Json::array()}, {"warmup", Json::array()}};
            for (uint32_t frame = 0; frame < 49; ++frame) {
                const bool capture = frame == 48;
                preview.setRawReadbackEnabled(capture);
                if (!preview.render(graph, state->size, state->size, "Coverage.color", capture)) {
                    std::ofstream(context.outputDirectory / "Failure.log") << preview.lastLog();
                    return RHITestResult::fail(preview.lastLog());
                }
                std::vector<RenderGraphExecutionStats> completed;
                if (!preview.collectCompletedGpuExecutionStats().transform([&](auto value) { completed = std::move(value); }) ||
                    completed.size() != 1 || !completed[0].gpuTimingAvailable || completed[0].profilingOverflow) {
                    return RHITestResult::fail("Coverage benchmark GPU timestamp missing/ambiguous/overflowed");
                }
                if (!capture) {
                    const auto& stats = completed[0];
                    const auto node = std::find_if(stats.nodes.begin(), stats.nodes.end(), [](const auto& entry) { return entry.name == "Coverage"; });
                    if (node == stats.nodes.end() || !node->gpuTimingAvailable) { return RHITestResult::fail("Missing Coverage node timing"); }
                    item[frame < 16 ? "warmup" : "frames"].push_back({{"frame", frame}, {"gpuMs", node->gpuMilliseconds}, {"graphMs", stats.gpuMilliseconds}});
                } else {
                    if (preview.readbackFormat() != Format::RGBA32Sfloat || preview.readbackBytes().size() != state->size * state->size * 16) {
                        return RHITestResult::fail("Unexpected Coverage output layout");
                    }
                    const auto filename = std::string(workload) + ".rgba32f.bin";
                    std::ofstream(context.outputDirectory / filename, std::ios::binary)
                        .write(reinterpret_cast<const char*>(preview.readbackBytes().data()), preview.readbackBytes().size());
                    item["image"] = filename;
                }
            }
            report["cases"].push_back(std::move(item));
            std::ofstream(context.outputDirectory / "CoverageBenchmark.json") << report.dump(2);
        }
        return RHITestResult::pass("Coverage evaluator timestamps and full float readbacks recorded; compare independent legacy/native processes");
    }
};
METALLIC_REGISTER_RHI_TEST(MaterialCoverageBenchmarkTest);
} // namespace
} // namespace metallic::tests

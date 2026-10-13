#include "RHITest.h"
#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"

#include <json.hpp>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace metallic::tests {
namespace {

class RecipeCaptureScope {
public:
    explicit RecipeCaptureScope(const std::filesystem::path& directory)
    {
        const auto* previous = std::getenv("METALLIC_PIPELINE_RECIPE_DIRECTORY");
        if (previous) { previous_ = previous; }
        set(directory.string().c_str());
    }
    ~RecipeCaptureScope() { set(previous_ ? previous_->c_str() : nullptr); }

private:
    static void set(const char* value)
    {
#ifdef _WIN32
        _putenv_s("METALLIC_PIPELINE_RECIPE_DIRECTORY", value ? value : "");
#else
        if (value) { setenv("METALLIC_PIPELINE_RECIPE_DIRECTORY", value, 1); }
        else { unsetenv("METALLIC_PIPELINE_RECIPE_DIRECTORY"); }
#endif
    }
    std::optional<std::string> previous_;
};

std::filesystem::path recipePath(const std::filesystem::path& directory, const char* type, uint64_t hash)
{
    std::ostringstream name;
    name << type << '-' << std::hex << std::setw(16) << std::setfill('0') << hash << ".recipe.json";
    return directory / name.str();
}

class PipelineRecipeTest final : public RHITest {
public:
    PipelineRecipeTest() { type = RHITestType::Resource; name = "pipeline_recipe_compute_graphics_roundtrip"; }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto directory = std::filesystem::absolute(context.outputDirectory / "pipeline-recipes" /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(directory);
        const RecipeCaptureScope capture(directory);
        auto& registry = ShaderRegistry::instance();
        {
            std::ofstream fixture(directory / "RecipeCompute.slang");
            fixture << "[shader(\"compute\")] [numthreads(1, 1, 1)] void recipeMain(uint3 id : SV_DispatchThreadID) {}\n";
            if (!fixture) { return RHITestResult::fail("could not write recipe compute fixture"); }
        }
        const auto sourceDirectory = directory.string();
        const auto codeCache = (directory / "spirv-cache").string();
        const SlangShaderCacheOptions shaderOptions{.cacheDirectory = codeCache.c_str()};
        std::string log;
        auto computeCode = registry.getShader({.moduleName = "RecipeCompute", .entryPointName = "recipeMain",
            .searchPath = sourceDirectory.c_str()}, shaderOptions, log);
        auto vertexCode = registry.getShader({.moduleName = "Features/Samples/Triangle", .entryPointName = "triangleVertexMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shaderOptions, log);
        auto fragmentCode = registry.getShader({.moduleName = "Features/Samples/Triangle", .entryPointName = "triangleFragmentMain",
            .searchPath = PROJECT_SOURCE_DIR "/Shaders"}, shaderOptions, log);
        if (!computeCode || !vertexCode || !fragmentCode) { return RHITestResult::fail("recipe shader compilation failed: " + log); }
        auto compute = registry.getShaderModule(context.device, {.spirv = computeCode->spirv});
        auto vertex = registry.getShaderModule(context.device, {.spirv = vertexCode->spirv});
        auto fragment = registry.getShaderModule(context.device, {.spirv = fragmentCode->spirv});
        auto cache = context.device.createPipelineCache({.saveOnDestroy = false});
        if (!compute || !vertex || !fragment || !cache) { return RHITestResult::fail("recipe module/cache creation failed"); }
        auto computePipeline = registry.getComputePipeline(context.device,
            {.computeShader = {compute->get()}, .pipelineCache = cache->get()});
        const GraphicsPipelineDesc graphicsDesc{.vertexShader = {vertex->get()}, .fragmentShader = {fragment->get()},
            .colorFormats = {Format::RGBA16Sfloat}, .colorAttachmentCount = 1, .depthStencilFormat = Format::D32Sfloat,
            .rasterization = {.cullMode = CullMode::Back, .frontFace = FrontFace::Clockwise},
            .depthStencil = {.depthTestEnable = true, .depthWriteEnable = true, .depthCompareOp = CompareOp::Less},
            .pipelineCache = cache->get()};
        auto graphicsPipeline = registry.getGraphicsPipeline(context.device, graphicsDesc);
        if (!computePipeline || !graphicsPipeline) { return RHITestResult::fail("source recipe pipeline creation failed"); }
        const auto computePath = recipePath(directory, "compute", (*computePipeline)->psoHash());
        const auto graphicsPath = recipePath(directory, "graphics", (*graphicsPipeline)->psoHash());
        auto computeRecipe = readPipelineRecipe(computePath, log);
        auto graphicsRecipe = readPipelineRecipe(graphicsPath, log);
        if (!computeRecipe || !graphicsRecipe) { return RHITestResult::fail("registry recipe export/import failed: " + log); }
        if (computeRecipe->computeShader.spirv != computeCode->spirv ||
            computeRecipe->computeShader.moduleName != "RecipeCompute" ||
            computeRecipe->computeShader.sourceEntryPoint != "recipeMain" ||
            graphicsRecipe->vertexShader.spirv != vertexCode->spirv ||
            graphicsRecipe->fragmentShader.spirv != fragmentCode->spirv ||
            graphicsRecipe->colorFormats[0] != Format::RGBA16Sfloat || graphicsRecipe->colorAttachmentCount != 1 ||
            graphicsRecipe->depthStencilFormat != Format::D32Sfloat ||
            graphicsRecipe->rasterization.cullMode != CullMode::Back ||
            graphicsRecipe->rasterization.frontFace != FrontFace::Clockwise ||
            !graphicsRecipe->depthStencil.depthTestEnable || !graphicsRecipe->depthStencil.depthWriteEnable ||
            graphicsRecipe->depthStencil.depthCompareOp != CompareOp::Less ||
            graphicsRecipe->sourceDevice.validationEnabled != context.enableValidation) {
            return RHITestResult::fail("export did not preserve source bytecode, identities, graphics states or validation settings");
        }
        // Import creates fresh modules directly from the captured input code;
        // neither Slang nor production cache lookup participates in the replay.
        auto replayCompute = context.device.createShaderModule({.spirv = computeRecipe->computeShader.spirv});
        auto replayVertex = context.device.createShaderModule({.spirv = graphicsRecipe->vertexShader.spirv});
        auto replayFragment = context.device.createShaderModule({.spirv = graphicsRecipe->fragmentShader.spirv});
        if (!replayCompute || !replayVertex || !replayFragment) { return RHITestResult::fail("imported recipe module creation failed"); }
        auto replayComputePipeline = context.device.createComputePipeline(computeRecipe->computeDesc(replayCompute->get(), cache->get()));
        auto replayGraphicsPipeline = context.device.createGraphicsPipeline(graphicsRecipe->graphicsDesc(
            replayVertex->get(), nullptr, nullptr, replayFragment->get(), cache->get()));
        if (!replayComputePipeline || !replayGraphicsPipeline ||
            (*replayComputePipeline)->psoHash() != computeRecipe->expectedPsoHash ||
            (*replayGraphicsPipeline)->psoHash() != graphicsRecipe->expectedPsoHash) {
            return RHITestResult::fail("recipe reconstruction changed the created PSO identity");
        }
        auto changedDesc = graphicsRecipe->graphicsDesc(replayVertex->get(), nullptr, nullptr, replayFragment->get(), cache->get());
        changedDesc.rasterization.cullMode = CullMode::Front;
        auto changedPipeline = context.device.createGraphicsPipeline(changedDesc);
        if (!changedPipeline || (*changedPipeline)->psoHash() == graphicsRecipe->expectedPsoHash) {
            return RHITestResult::fail("recipe replay aliased a changed graphics state");
        }
        auto experimental = *computeRecipe;
        experimental.disableOptimization = true;
        experimental.computeBackend = ComputePipelineBackend::PipelineBinary;
        experimental.expectedPsoHash = detail::computePipelineStateHash(experimental.computeDesc(replayCompute->get()));
        const auto experimentalPath = directory / "experimental.recipe.json";
        if (!writePipelineRecipe(experimental, experimentalPath, log)) {
            return RHITestResult::fail("could not serialize experimental compute recipe: " + log);
        }
        auto importedExperiment = readPipelineRecipe(experimentalPath, log);
        if (!importedExperiment || !importedExperiment->disableOptimization ||
            importedExperiment->computeBackend != ComputePipelineBackend::PipelineBinary ||
            detail::computePipelineStateHash(importedExperiment->computeDesc(replayCompute->get())) != experimental.expectedPsoHash) {
            return RHITestResult::fail("recipe import lost driver optimization or compute backend identity");
        }
        changedDesc.disableOptimization = true;
        auto changedOptimization = context.device.createGraphicsPipeline(changedDesc);
        if (!changedOptimization || (*changedOptimization)->psoHash() == (*changedPipeline)->psoHash()) {
            return RHITestResult::fail("graphics creation optimization policy reused the optimized PSO identity");
        }
        nlohmann::json manifest;
        { std::ifstream input(computePath); manifest = nlohmann::json::parse(input); }
        auto legacyManifest = manifest;
        legacyManifest["state"].erase("disable_optimization");
        legacyManifest["state"].erase("compute_backend");
        legacyManifest["shaders"]["compute"].erase("slang_optimization");
        const auto legacyPath = directory / "legacy.recipe.json";
        { std::ofstream output(legacyPath); output << legacyManifest.dump(2); }
        auto legacy = readPipelineRecipe(legacyPath, log);
        if (!legacy || legacy->disableOptimization || legacy->computeBackend != ComputePipelineBackend::Pipeline ||
            legacy->expectedPsoHash != computeRecipe->expectedPsoHash) {
            return RHITestResult::fail("legacy recipe defaults changed its pipeline creation identity");
        }
        auto zeroHashManifest = manifest;
        zeroHashManifest["shaders"]["compute"]["input_spirv_hash"] = "0000000000000000";
        const auto zeroHashPath = directory / "zero-hash.recipe.json";
        { std::ofstream output(zeroHashPath); output << zeroHashManifest.dump(2); }
        if (readPipelineRecipe(zeroHashPath, log)) { return RHITestResult::fail("recipe import accepted an omitted content hash"); }
        const auto blob = directory / manifest["shaders"]["compute"]["spirv_file"].get<std::string>();
        {
            std::fstream output(blob, std::ios::binary | std::ios::in | std::ios::out);
            output.seekp(sizeof(uint32_t));
            const uint32_t corruptVersion = 0;
            output.write(reinterpret_cast<const char*>(&corruptVersion), sizeof(corruptVersion));
        }
        if (readPipelineRecipe(computePath, log)) { return RHITestResult::fail("recipe import accepted corrupted SPIR-V"); }
        {
            std::ofstream output(blob, std::ios::binary | std::ios::trunc);
            output.write(reinterpret_cast<const char*>(computeCode->spirv.data()),
                static_cast<std::streamsize>(computeCode->spirv.size() * sizeof(uint32_t)));
            if (!output) { return RHITestResult::fail("could not restore recipe SPIR-V evidence"); }
        }
        manifest["version"] = 999;
        const auto unsupportedPath = directory / "unsupported.recipe.json";
        { std::ofstream output(unsupportedPath); output << manifest.dump(2); }
        if (readPipelineRecipe(unsupportedPath, log)) { return RHITestResult::fail("recipe import accepted an unsupported schema version"); }
        return RHITestResult::pass("registry capture, exact compute/graphics PSO replay, state variants and corrupt/unsupported import rejection");
    }
};
METALLIC_REGISTER_RHI_TEST(PipelineRecipeTest);

} // namespace
} // namespace metallic::tests

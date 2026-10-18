#include "RHITest.h"
#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"

#include <gtest/gtest.h>
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
            fixture << "module RecipeCompute;\n[shader(\"compute\")] [numthreads(1, 1, 1)] void recipeMain(uint3 id : SV_DispatchThreadID) {}\n";
            if (!fixture) { return RHITestResult::fail("could not write recipe compute fixture"); }
        }
        const auto sourceDirectory = directory.string();
        const auto codeCache = (directory / "spirv-cache").string();
        const SlangShaderCacheOptions shaderOptions{.cacheDirectory = codeCache.c_str()};
        std::string log;
        const SlangShaderDesc computeSource{.moduleName = "RecipeCompute", .entryPointName = "recipeMain",
            .searchPath = sourceDirectory.c_str()};
        auto computeCode = registry.getShader(computeSource, shaderOptions, log);
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
            computeRecipe->computeShader.spirvPostprocess != slangShaderPostprocessModeName(slangShaderPostprocessMode(computeSource)) ||
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
        const auto rejectInvalidExport = [&](const char* policy, const char* version, const char* filename) {
            auto invalid = *computeRecipe;
            invalid.computeShader.spirvPostprocess = policy;
            invalid.computeShader.spirvPostprocessToolVersion = version;
            const auto invalidPath = directory / filename;
            return !writePipelineRecipe(invalid, invalidPath, log) && !std::filesystem::exists(invalidPath);
        };
        if (!rejectInvalidExport("unknown", "", "unknown-postprocess.recipe.json") ||
            !rejectInvalidExport("dead-branches", "", "missing-cleanup-tool.recipe.json") ||
            !rejectInvalidExport("slang-dead-branches", "", "missing-slang-cleanup-tool.recipe.json") ||
            !rejectInvalidExport("dead-branches", "unavailable", "unavailable-cleanup-tool.recipe.json")) {
            return RHITestResult::fail("recipe export accepted an invalid postprocess policy or optimizer version");
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
        experimental.computeShader.spirvPostprocess = "dead-branches";
        experimental.computeShader.spirvPostprocessToolVersion = "roundtrip-test-version";
        experimental.disableOptimization = true;
        experimental.computeBackend = ComputePipelineBackend::PipelineBinary;
        experimental.expectedPsoHash = detail::computePipelineStateHash(experimental.computeDesc(replayCompute->get()));
        const auto experimentalPath = directory / "experimental.recipe.json";
        if (!writePipelineRecipe(experimental, experimentalPath, log)) {
            return RHITestResult::fail("could not serialize experimental compute recipe: " + log);
        }
        auto importedExperiment = readPipelineRecipe(experimentalPath, log);
        if (!importedExperiment || !importedExperiment->disableOptimization ||
            importedExperiment->computeShader.spirvPostprocess != "dead-branches" ||
            importedExperiment->computeShader.spirvPostprocessToolVersion != "roundtrip-test-version" ||
            importedExperiment->computeBackend != ComputePipelineBackend::PipelineBinary ||
            detail::computePipelineStateHash(importedExperiment->computeDesc(replayCompute->get())) != experimental.expectedPsoHash) {
            return RHITestResult::fail("recipe import lost driver optimization or compute backend identity");
        }
        auto slangExperiment = experimental;
        slangExperiment.computeShader.spirvPostprocess = "slang-dead-branches";
        const auto slangPath = directory / "slang-experimental.recipe.json";
        if (!writePipelineRecipe(slangExperiment, slangPath, log)) {
            return RHITestResult::fail("could not serialize Slang downstream cleanup identity: " + log);
        }
        auto importedSlang = readPipelineRecipe(slangPath, log);
        if (!importedSlang || importedSlang->computeShader.spirvPostprocess != "slang-dead-branches" ||
            importedSlang->computeShader.spirvPostprocessToolVersion != "roundtrip-test-version" ||
            importedSlang->computeShader.spirv != experimental.computeShader.spirv ||
            importedSlang->expectedPsoHash != experimental.expectedPsoHash) {
            return RHITestResult::fail("recipe roundtrip lost Slang downstream cleanup provenance or changed bytecode/state");
        }
        auto missingCleanupVersion = nlohmann::json::parse(std::ifstream(experimentalPath));
        missingCleanupVersion["shaders"]["compute"].erase("spirv_postprocess_tool_version");
        const auto missingVersionPath = directory / "missing-cleanup-version.recipe.json";
        { std::ofstream output(missingVersionPath); output << missingCleanupVersion.dump(2); }
        if (readPipelineRecipe(missingVersionPath, log)) {
            return RHITestResult::fail("recipe accepted cleanup without a recorded optimizer version");
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
        legacyManifest["shaders"]["compute"].erase("source_identity_ambiguous");
        const auto legacyPath = directory / "legacy.recipe.json";
        { std::ofstream output(legacyPath); output << legacyManifest.dump(2); }
        auto legacy = readPipelineRecipe(legacyPath, log);
        if (!legacy || legacy->disableOptimization || legacy->computeBackend != ComputePipelineBackend::Pipeline ||
            legacy->expectedPsoHash != computeRecipe->expectedPsoHash || legacy->computeShader.sourceIdentityAmbiguous) {
            return RHITestResult::fail("legacy recipe defaults changed its pipeline creation identity");
        }
        {
            struct DebugModeScope {
                SlangShaderDebugMode previous = slangShaderDebugMode();
                DebugModeScope() { setSlangShaderDebugMode(SlangShaderDebugMode::Disabled); }
                ~DebugModeScope() { setSlangShaderDebugMode(previous); }
            } debugScope;
            // Two single-entry modules emit the same main function. An unusual,
            // valid workgroup shape keeps this fixture distinct from other empty shaders.
            for (const char* moduleName : {"RecipeIdentityFirst", "RecipeIdentitySecond"}) {
                std::ofstream fixture(directory / (std::string(moduleName) + ".slang"));
                fixture << "module " << moduleName << ";\n"
                    << "[shader(\"compute\")] [numthreads(7, 3, 1)] void recipeMain(uint3 id : SV_DispatchThreadID) {}\n";
                fixture.close();
                if (!fixture) { return RHITestResult::fail("could not write source identity conflict fixture"); }
            }
            SlangShaderDesc firstSource{.moduleName = "RecipeIdentityFirst", .entryPointName = "recipeMain",
                .searchPath = sourceDirectory.c_str(), .descriptorHeapMode = SlangDescriptorHeapMode::Mapped,
                .optimizationLevel = SlangShaderOptimizationLevel::O1, .postprocessMode = SlangShaderPostprocessMode::None};
            auto firstCode = registry.getShader(firstSource, shaderOptions, log);
            if (!firstCode) { return RHITestResult::fail("first source identity compilation failed: " + log); }
            auto firstModule = registry.getShaderModule(context.device, {.spirv = firstCode->spirv});
            if (!firstModule) { return RHITestResult::fail("first source identity module creation failed"); }
            const auto firstDirectory = directory / "first-source";
            {
                const RecipeCaptureScope firstCapture(firstDirectory);
                auto firstPipeline = registry.getComputePipeline(context.device,
                    {.computeShader = {firstModule->get()}, .pipelineCache = cache->get()});
                if (!firstPipeline) { return RHITestResult::fail("first source identity pipeline creation failed"); }
                auto firstRecipe = readPipelineRecipe(recipePath(firstDirectory, "compute", (*firstPipeline)->psoHash()), log);
                if (!firstRecipe || firstRecipe->computeShader.sourceIdentityAmbiguous ||
                    firstRecipe->computeShader.moduleName != "RecipeIdentityFirst" ||
                    firstRecipe->computeShader.sourceEntryPoint != "recipeMain") {
                    return RHITestResult::fail("unique source identity was not captured accurately"); }
            }
            auto secondSource = firstSource;
            secondSource.moduleName = "RecipeIdentitySecond";
            // Establish exact output identity before registering the second producer.
            auto secondProbe = compileSlangShaderToSpirv(secondSource, shaderOptions, log);
            if (!secondProbe || firstCode->spirv != secondProbe->spirv) {
                return RHITestResult::fail("source identity conflict fixture must emit identical bytecode"); }
            auto secondCode = registry.getShader(secondSource, shaderOptions, log);
            if (!secondCode || firstCode->spirv != secondCode->spirv) {
                return RHITestResult::fail("source identity conflict fixture must emit identical bytecode"); }
            // Reobserving the original candidate must not clear the conflict.
            auto originalAgain = registry.getShader(firstSource, shaderOptions, log);
            if (!originalAgain || originalAgain->spirv != firstCode->spirv) {
                return RHITestResult::fail("original source identity did not retain its bytecode"); }
            auto secondModule = registry.getShaderModule(context.device, {.spirv = secondCode->spirv});
            if (!secondModule) { return RHITestResult::fail("ambiguous source module creation failed"); }
            const auto secondDirectory = directory / "ambiguous-source";
            const RecipeCaptureScope secondCapture(secondDirectory);
            auto secondPipeline = registry.getComputePipeline(context.device,
                {.computeShader = {secondModule->get()}, .pipelineCache = cache->get()});
            if (!secondPipeline) { return RHITestResult::fail("ambiguous source pipeline creation failed"); }
            auto ambiguous = readPipelineRecipe(recipePath(secondDirectory, "compute", (*secondPipeline)->psoHash()), log);
            if (!ambiguous || !ambiguous->computeShader.sourceIdentityAmbiguous ||
                ambiguous->computeShader.moduleName != "RecipeIdentityFirst" ||
                ambiguous->computeShader.sourceEntryPoint != "recipeMain" ||
                ambiguous->computeShader.spirv != firstCode->spirv) {
                return RHITestResult::fail("same-bytecode source conflict lost ambiguity or overwrote its candidate identity"); }
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

// CPU recipe fixtures: these exercise serialization/content validation only.
// Native RT module validity, device rewriting and PSO creation are separate tests.
std::vector<uint32_t> rayTracingRecipeModule(uint32_t executionModel)
{
    std::vector<uint32_t> words{0x07230203u, 0x00010600u, 0, 7, 0,
        0x00020011u, 1, 0x00020011u, 4479}; // Shader, RayTracingKHR
    constexpr char extension[] = "SPV_KHR_ray_tracing";
    constexpr size_t extensionWords = (sizeof(extension) + 3) / 4;
    words.push_back(static_cast<uint32_t>((extensionWords + 1) << 16) | 10u);
    for (size_t index = 0; index < extensionWords; ++index) {
        uint32_t word = 0;
        for (size_t byte = 0; byte < 4 && index * 4 + byte < sizeof(extension); ++byte) {
            word |= uint32_t(static_cast<unsigned char>(extension[index * 4 + byte])) << (byte * 8);
        }
        words.push_back(word);
    }
    const uint32_t body[]{0x0003000eu, 0, 1,
        0x0005000fu, executionModel, 1, 0x6e69616du, 0,
        0x00040047u, 3, 1, 1000,
        0x00020013u, 4, 0x00030021u, 5, 4,
        0x00040015u, 2, 32, 1, 0x00040032u, 2, 3, 0,
        0x00050036u, 4, 1, 0, 5, 0x000200f8u, 6, 0x000100fdu, 0x00010038u};
    words.insert(words.end(), std::begin(body), std::end(body));
    return words;
}

TEST(PipelineRecipe, RayTracingV3RoundtripAndStageValidation)
{
    using namespace render;
    const auto directory = std::filesystem::temp_directory_path() /
        ("metallic-rt-recipe-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    struct RemoveFixture {
        std::filesystem::path path;
        ~RemoveFixture() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{directory};
    PipelineRecipe recipe;
    recipe.type = PipelineRecipeType::RayTracing;
    recipe.cacheGroup = "ray-tracing-cpu-schema";
    recipe.usesBindlessHeap = true;
    recipe.bindlessUserPushDataSize = 12;
    recipe.maxRecursionDepth = 1;
    recipe.clusterAccelerationStructure = true;
    recipe.disableOptimization = true;
    recipe.sourceDevice.capabilities.rayTracingPipeline = true;
    recipe.sourceDevice.capabilities.rayTracingAccelerationStructure = true;
    recipe.sourceDevice.capabilities.shaderMaximalReconvergence = true;
    const std::array stages{&PipelineRecipe::rayGenerationShader, &PipelineRecipe::missShader,
        &PipelineRecipe::closestHitShader, &PipelineRecipe::anyHitShader};
    const std::array<const char*, 4> names{"ray_generation", "miss", "closest_hit", "any_hit"};
    const std::array<uint32_t, 4> models{5313, 5317, 5316, 5315};
    for (size_t index = 0; index < stages.size(); ++index) {
        auto& shader = recipe.*stages[index];
        shader.spirv = rayTracingRecipeModule(models[index]);
        shader.moduleName = std::string("RecipeRT.") + names[index];
        shader.sourceEntryPoint = std::string(names[index]) + "Main";
        shader.debugMode = "disabled";
        shader.descriptorHeapMode = "mapped";
        shader.optimizationLevel = "O2";
    }
    recipe.rayGenerationShader.specializationConstants = {{1000, ShaderSpecializationType::Int32, 4}};
    // No device is involved: this sentinel verifies exact manifest roundtrip.
    // Actual native PSO hashes are checked by --recipe-file before creation.
    recipe.expectedPsoHash = 0x123456789abcdef0ull;
    std::string log;
    const auto path = directory / "ray-tracing.recipe.json";
    ASSERT_TRUE(writePipelineRecipe(recipe, path, log)) << log;
    const auto manifest = nlohmann::json::parse(std::ifstream(path));
    ASSERT_EQ(manifest.at("version"), 3);
    ASSERT_EQ(manifest.at("pipeline_type"), "ray_tracing");
    auto loaded = readPipelineRecipe(path, log);
    ASSERT_TRUE(loaded) << log;
    EXPECT_EQ(loaded->type, PipelineRecipeType::RayTracing);
    EXPECT_EQ(loaded->expectedPsoHash, recipe.expectedPsoHash);
    EXPECT_EQ(loaded->cacheGroup, recipe.cacheGroup);
    EXPECT_TRUE(loaded->sourceDevice.capabilities.rayTracingPipeline);
    EXPECT_TRUE(loaded->sourceDevice.capabilities.rayTracingAccelerationStructure);
    EXPECT_TRUE(loaded->sourceDevice.capabilities.shaderMaximalReconvergence);
    auto legacyManifest = manifest;
    legacyManifest["source_device"]["enabled_features"].erase("shader_maximal_reconvergence");
    const auto legacyPath = directory / "legacy-capabilities.recipe.json";
    { std::ofstream output(legacyPath); output << legacyManifest.dump(2); }
    auto legacy = readPipelineRecipe(legacyPath, log);
    ASSERT_TRUE(legacy) << log;
    EXPECT_FALSE(legacy->sourceDevice.capabilities.shaderMaximalReconvergence);
    EXPECT_EQ(legacy->expectedPsoHash, recipe.expectedPsoHash);
    for (size_t index = 0; index < stages.size(); ++index) {
        const auto& original = recipe.*stages[index];
        const auto& imported = (*loaded).*stages[index];
        EXPECT_EQ(imported.spirv, original.spirv);
        EXPECT_EQ(imported.specializationConstants, original.specializationConstants);
        EXPECT_EQ(imported.moduleName, original.moduleName);
        EXPECT_EQ(imported.sourceEntryPoint, original.sourceEntryPoint);
        EXPECT_EQ(imported.entryPoint, original.entryPoint);
        EXPECT_EQ(imported.optimizationLevel, original.optimizationLevel);
        EXPECT_NE(imported.inputSpirvHash, 0u);
    }
    const auto desc = loaded->rayTracingDesc(nullptr, nullptr, nullptr, nullptr);
    EXPECT_TRUE(desc.usesBindlessHeap);
    EXPECT_EQ(desc.bindlessUserPushDataSize, recipe.bindlessUserPushDataSize);
    EXPECT_EQ(desc.maxRecursionDepth, recipe.maxRecursionDepth);
    EXPECT_TRUE(desc.clusterAccelerationStructure);
    EXPECT_TRUE(desc.disableOptimization);
    EXPECT_EQ(desc.rayGenerationShader.specializationConstants.size(), 1u);
    EXPECT_TRUE(desc.missShader.specializationConstants.empty());
    EXPECT_TRUE(desc.closestHitShader.specializationConstants.empty());
    EXPECT_TRUE(desc.anyHitShader.specializationConstants.empty());
    EXPECT_EQ(detail::rayTracingPipelineStateHash(desc),
        detail::rayTracingPipelineStateHash(recipe.rayTracingDesc(nullptr, nullptr, nullptr, nullptr)));

    auto serRecipe = recipe;
    serRecipe.enableInvocationReorder = true;
    serRecipe.sourceDevice.capabilities.rayTracingInvocationReorder = true;
    serRecipe.sourceDevice.capabilities.rayTracingInvocationReorderReorderingHint = true;
    const auto serPath = directory / "ray-tracing-ser.recipe.json";
    ASSERT_TRUE(writePipelineRecipe(serRecipe, serPath, log)) << log;
    const auto serManifest = nlohmann::json::parse(std::ifstream(serPath));
    EXPECT_EQ(serManifest.at("version"), 4);
    EXPECT_TRUE(serManifest.at("state").at("enable_invocation_reorder").get<bool>());
    auto loadedSER = readPipelineRecipe(serPath, log);
    ASSERT_TRUE(loadedSER) << log;
    EXPECT_TRUE(loadedSER->enableInvocationReorder);
    EXPECT_TRUE(loadedSER->rayTracingDesc(nullptr, nullptr, nullptr, nullptr).enableInvocationReorder);
    EXPECT_TRUE(loadedSER->sourceDevice.capabilities.rayTracingInvocationReorder);
    EXPECT_TRUE(loadedSER->sourceDevice.capabilities.rayTracingInvocationReorderReorderingHint);
    serRecipe.sourceDevice.capabilities.rayTracingInvocationReorder = false;
    EXPECT_FALSE(writePipelineRecipe(serRecipe, directory / "unsupported-ser.recipe.json", log));

    const auto rejects = [&](nlohmann::json changed) {
        const auto invalidPath = directory / "invalid.recipe.json";
        { std::ofstream output(invalidPath); output << changed.dump(2); }
        return !readPipelineRecipe(invalidPath, log);
    };
    for (size_t index = 0; index < names.size(); ++index) {
        SCOPED_TRACE(names[index]);
        auto changed = manifest; changed["shaders"].erase(names[index]);
        EXPECT_TRUE(rejects(changed));
        changed = manifest;
        changed["shaders"][names[index]] = {{"spirv_file", nullptr}, {"entry_point", "main"}};
        EXPECT_TRUE(rejects(changed));
        changed = manifest; changed["shaders"][names[index]]["input_spirv_hash"] = "0000000000000000";
        EXPECT_TRUE(rejects(changed));
        auto invalid = recipe; invalid.*stages[index] = {};
        EXPECT_FALSE(writePipelineRecipe(invalid, directory / "missing-stage.recipe.json", log));
    }
    for (const char* stage : {"compute", "vertex", "task", "mesh", "fragment"}) {
        auto changed = manifest; changed["shaders"][stage] = changed["shaders"]["ray_generation"];
        EXPECT_TRUE(rejects(changed)) << stage;
    }
    for (const char* stage : {"miss", "closest_hit", "any_hit"}) {
        auto changed = manifest;
        changed["shaders"][stage]["specialization_constants"] = changed["shaders"]["ray_generation"]["specialization_constants"];
        EXPECT_TRUE(rejects(changed)) << stage;
    }
    auto changed = manifest; changed["version"] = 2;
    EXPECT_TRUE(rejects(changed));
    changed = serManifest; changed["version"] = 3;
    EXPECT_TRUE(rejects(changed));
    changed = serManifest; changed["state"].erase("enable_invocation_reorder");
    EXPECT_TRUE(rejects(changed));
    changed = serManifest; changed["source_device"]["enabled_features"]["ray_tracing_invocation_reorder"] = false;
    EXPECT_TRUE(rejects(changed));
    changed = manifest; changed["pipeline_type"] = "compute";
    EXPECT_TRUE(rejects(changed));
    changed = manifest; changed["state"]["max_recursion_depth"] = 0;
    EXPECT_TRUE(rejects(changed));
    changed = manifest; changed["state"].erase("cluster_acceleration_structure");
    EXPECT_TRUE(rejects(changed));
    changed = manifest; changed["state"]["indirect_bindable"] = true;
    EXPECT_TRUE(rejects(changed));
    for (const auto stage : {&PipelineRecipe::computeShader, &PipelineRecipe::vertexShader,
        &PipelineRecipe::taskShader, &PipelineRecipe::meshShader, &PipelineRecipe::fragmentShader}) {
        auto invalid = recipe; invalid.*stage = recipe.rayGenerationShader;
        EXPECT_FALSE(writePipelineRecipe(invalid, directory / "mixed-stage.recipe.json", log));
    }
    auto invalid = recipe;
    invalid.type = PipelineRecipeType::Compute;
    invalid.computeShader = recipe.rayGenerationShader;
    EXPECT_FALSE(writePipelineRecipe(invalid, directory / "mixed-compute.recipe.json", log));
}

TEST(PipelineRecipe, RayTracingHashCoversStagesSpecializationAndNativeState)
{
    using namespace render;
    RayTracingPipelineDesc desc;
    desc.rayGenerationShader.entryPoint = "rayGeneration";
    desc.missShader.entryPoint = "miss";
    desc.closestHitShader.entryPoint = "closestHit";
    desc.anyHitShader.entryPoint = "anyHit";
    const std::array values{ShaderSpecializationConstant{1001, ShaderSpecializationType::Bool32, 1},
        ShaderSpecializationConstant{1000, ShaderSpecializationType::Int32, 4}};
    desc.rayGenerationShader.specializationConstants = values;
    const auto baseline = detail::rayTracingPipelineStateHash(desc);
    auto canonical = detail::canonicalShaderSpecializationConstants(values);
    auto changed = desc; changed.rayGenerationShader.specializationConstants = canonical;
    EXPECT_EQ(detail::rayTracingPipelineStateHash(changed), baseline);
    canonical[0].bits = 8;
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    const std::array stages{&RayTracingPipelineDesc::rayGenerationShader, &RayTracingPipelineDesc::missShader,
        &RayTracingPipelineDesc::closestHitShader, &RayTracingPipelineDesc::anyHitShader};
    for (const auto stage : stages) {
        changed = desc; (changed.*stage).entryPoint = "changed";
        EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
        changed = desc; (changed.*stage).specializationConstants = canonical;
        EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    }
    changed = desc; std::swap(changed.missShader, changed.anyHitShader);
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    changed = desc; changed.usesBindlessHeap = true;
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    changed = desc; changed.bindlessUserPushDataSize = 12;
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    changed = desc; changed.maxRecursionDepth = 2;
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    changed = desc; changed.clusterAccelerationStructure = !desc.clusterAccelerationStructure;
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    changed = desc; changed.disableOptimization = true;
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
    changed = desc; changed.enableInvocationReorder = true;
    EXPECT_NE(detail::rayTracingPipelineStateHash(changed), baseline);
}

} // namespace
} // namespace metallic::tests

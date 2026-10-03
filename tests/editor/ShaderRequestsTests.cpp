#include "Runtime/Render/Core/BuiltinShaderRequests.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>

using namespace metallic::render;

namespace {

SceneShaderOptions standardSceneOptions()
{
    const std::string rtxcrInclude = METALLIC_RTXCR_SHADER_INCLUDE_DIR;
    return {.hasRTXCR = !rtxcrInclude.empty(), .positionFetch = true, .rtxcrInclude = rtxcrInclude};
}

bool contains(const std::vector<ShaderRequest>& catalog, const ShaderRequest& request)
{
    return std::find(catalog.begin(), catalog.end(), request) != catalog.end();
}

class ShaderRequestCompileTest : public testing::Test {
protected:
    void SetUp() override
    {
        oldMode_ = slangShaderDebugMode();
        setSlangShaderDebugMode(SlangShaderDebugMode::Disabled);
    }
    void TearDown() override { setSlangShaderDebugMode(oldMode_); }
private:
    SlangShaderDebugMode oldMode_;
};

TEST(ShaderRequests, OwnsStringsAcrossCopiesAndMoves)
{
    ShaderRequest original{.module = "A/LongModuleName", .entry = "main",
        .capabilities = {"spvRayQueryKHR"}, .defines = {{"CUSTOM_VALUE", "123"}}, .searchPaths = {"C:/Original"}};
    ShaderRequest copied = original;
    original.module.clear(); original.defines.clear(); original.searchPaths.clear();
    ShaderRequest moved = std::move(copied);
    const ShaderRequestView source(moved);
    const auto desc = source.desc();
    EXPECT_STREQ(desc.moduleName, "A/LongModuleName");
    ASSERT_EQ(desc.macroDefines.size(), 1u);
    EXPECT_STREQ(desc.macroDefines[0].name, "CUSTOM_VALUE");
    EXPECT_STREQ(desc.macroDefines[0].value, "123");
    EXPECT_STREQ(desc.additionalSearchPaths[0], "C:/Original");
    EXPECT_STREQ(desc.capabilities[0], "spvRayQueryKHR");
    EXPECT_STREQ(desc.profileName, kDefaultSlangProfileName);
}

TEST(ShaderRequests, IncludesTheRuntimeCustomMaterialDefaultInItsOriginalPosition)
{
    const auto request = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, {});
    const std::vector<std::pair<std::string, std::string>> expected{
        {"METALLIC_CUSTOM_MATERIALS", "0"}, {"METALLIC_STREAM_MATERIALS", "0"},
        {"METALLIC_STREAM_RAY_QUERIES", "0"}, {"METALLIC_GLOBAL_VIEW", "0"},
        {"METALLIC_HAS_RTXCR", "0"}, {"METALLIC_HAS_NTC", "0"}, {"METALLIC_NTC_COOPERATIVE_VECTOR", "0"},
        {"SCENE_RAYQUERY_ENABLE_POSITION_FETCH", "0"}, {"METALLIC_DEFERRED_LIGHT_GRID", "0"},
        {"METALLIC_REALTIME_DEFERRED", "0"}, {"METALLIC_DEFERRED_UPSCALER_GUIDES", "0"},
    };
    EXPECT_EQ(request.defines, expected);
    EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "spvGroupNonUniformBallot"}));
}

TEST(ShaderRequests, PreservesCustomMaterialAndSDKSearchPrecedence)
{
    const SceneShaderOptions options{.customMaterials = true, .hasRTXCR = true, .hasNTC = true,
        .cooperativeVector = true, .positionFetch = true,
        .materialInclude = "C:/Values", .rtxcrInclude = "C:/RTXCR", .ntcInclude = "C:/NTC"};
    const SlangMacroDefine cacheDefine{"NRC_QUERY", "1"};
    const auto request = makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&cacheDefine, 1});
    EXPECT_EQ(request.searchPaths, (std::vector<std::string>{"C:/Values", "C:/RTXCR", "C:/NTC"}));
    EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "spvGroupNonUniformBallot",
        "spvRayQueryPositionFetchKHR", "spvCooperativeVectorNV"}));
    EXPECT_EQ(request.defines.front(), (std::pair<std::string, std::string>{"METALLIC_CUSTOM_MATERIALS", "1"}));
    EXPECT_EQ(request.defines.back(), (std::pair<std::string, std::string>{"NRC_QUERY", "1"}));
    const auto maintenance = makeSceneShaderRequest(SceneShaderProgram::SharcClear, options);
    EXPECT_TRUE(maintenance.defines.empty());
    EXPECT_TRUE(maintenance.searchPaths.empty());
    EXPECT_EQ(maintenance.capabilities, request.capabilities);
}

TEST(ShaderRequests, CanonicalizesUnreachableContinuationAndRetainsTransmissionVariants)
{
    for (bool streamed : {false, true}) {
        for (bool rays : {false, true}) {
            SceneShaderOptions off{.streamMaterials = streamed, .streamRayQueries = rays};
            auto on = off; on.supplementaryPathTracing = true;
            for (int materialClass = 0; materialClass < 5; ++materialClass) {
                const std::string value = std::to_string(materialClass);
                const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
                const auto a = makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, off, {&define, 1});
                const auto b = makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, on, {&define, 1});
                EXPECT_EQ(a == b, materialClass < 4 || (streamed && !rays));
            }
            const auto a = makeSceneShaderRequest(SceneShaderProgram::Deferred, off);
            const auto b = makeSceneShaderRequest(SceneShaderProgram::Deferred, on);
            EXPECT_EQ(a == b, streamed && !rays);
        }
    }
}

TEST(ShaderRequests, CatalogCoversProductionSceneVariantsWithoutDuplicateRequests)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    size_t deferredCount = 0;
    for (size_t i = 0; i < catalog.size(); ++i) {
        EXPECT_EQ(std::count(catalog.begin(), catalog.end(), catalog[i]), 1);
        if (catalog[i].module == "Features/VisibilityBuffer/VisibilityBufferDeferred") { ++deferredCount; }
    }
    EXPECT_EQ(deferredCount, 44u);
    for (bool fetch : {false, true}) {
        auto options = standardSceneOptions(); options.positionFetch = fetch;
        for (auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
                SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides,
                SceneShaderProgram::RealtimeLighting, SceneShaderProgram::SharcClear,
                SceneShaderProgram::SharcResolve, SceneShaderProgram::Tonemap}) {
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(program, options)));
        }
        for (const char* name : {"SHARC_UPDATE", "SHARC_QUERY", "NRC_UPDATE", "NRC_QUERY"}) {
            const SlangMacroDefine define{name, "1"};
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&define, 1})));
        }
    }
    for (int mode = 0; mode < 4; ++mode) {
        for (bool realtime : {false, true}) {
            if (mode >= 2 && !realtime) { continue; }
            for (bool continuation : {false, true}) {
                if (mode == 3 && !continuation) { continue; }
                auto options = standardSceneOptions();
                options.positionFetch = mode == 1; options.streamMaterials = mode >= 2;
                options.streamRayQueries = mode == 3; options.globalView = true;
                options.realtimeDeferred = realtime; options.upscalerGuides = realtime;
                options.supplementaryPathTracing = continuation;
                EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::Deferred, options)));
                for (int materialClass = 0; materialClass < 5; ++materialClass) {
                    const std::string value = std::to_string(materialClass);
                    const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
                    EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, options, {&define, 1})));
                }
            }
        }
    }
}

TEST(ShaderRequests, RayQueriesAndShadowsRetainOptionalNTCRequests)
{
    const auto ray = makeSceneRayQueryRequest(SceneRayQueryProgram::RTXDI,
        {.positionFetch = true, .hasNTC = true, .cooperativeVector = true, .ntcInclude = "C:/NTC"});
    EXPECT_EQ(ray.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "spvRayQueryPositionFetchKHR", "spvCooperativeVectorNV"}));
    EXPECT_EQ(ray.searchPaths, (std::vector<std::string>{"C:/NTC"}));
    const auto shadow = makeShadowShaderRequest({.hasNTC = true, .cooperativeVector = true, .ntcInclude = "C:/NTC"});
    EXPECT_EQ(shadow.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "spvCooperativeVectorNV"}));
    EXPECT_EQ(shadow.searchPaths, ray.searchPaths);
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    EXPECT_TRUE(contains(catalog, makeSceneRayQueryRequest(SceneRayQueryProgram::RTXDI, {.positionFetch = true})));
    EXPECT_TRUE(contains(catalog, makeSceneRayQueryRequest(SceneRayQueryProgram::MaterialVisualization, {})));
    EXPECT_TRUE(contains(catalog, makeShadowShaderRequest({.streamed = true, .streamTlas = true})));
    EXPECT_FALSE(contains(catalog, ray)); // Optional SDK permutations are still compiled on demand.
}

TEST_F(ShaderRequestCompileTest, PrewarmedOpenPBRHitsTheRuntimeRequestCache)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    const auto runtime = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, standardSceneOptions());
    const auto warmup = std::find(catalog.begin(), catalog.end(), runtime);
    ASSERT_NE(warmup, catalog.end());
    const auto directory = std::filesystem::path(TEST_BINARY_DIR) / "shader-request-cache" /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::string cacheDirectory = directory.string();
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = cacheDirectory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    const ShaderRequestView warmupSource(*warmup);
    const auto precompiled = compileSlangShaderToSpirv(warmupSource.desc(), cache, diagnostics);
    ASSERT_TRUE(precompiled) << diagnostics;
    EXPECT_FALSE(hit);
    const ShaderRequestView runtimeSource(runtime);
    const auto reused = compileSlangShaderToSpirv(runtimeSource.desc(), cache, diagnostics);
    ASSERT_TRUE(reused) << diagnostics;
    EXPECT_TRUE(hit);
    EXPECT_EQ(precompiled->spirv, reused->spirv);
}

TEST_F(ShaderRequestCompileTest, FoldedDeferredClassesProduceIdenticalSPIRV)
{
    for (int materialClass = 0; materialClass < 4; ++materialClass) {
        SCOPED_TRACE(materialClass);
        auto options = standardSceneOptions(); options.globalView = true;
        options.realtimeDeferred = true; options.upscalerGuides = true; options.supplementaryPathTracing = true;
        const std::string value = std::to_string(materialClass);
        const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
        const auto normalized = makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, options, {&define, 1});
        auto original = normalized;
        for (auto& [name, v] : original.defines) {
            if (name == "METALLIC_DEFERRED_PATH_TRACING") { v = "1"; }
        }
        const ShaderRequestView a(normalized), b(original);
        const SlangShaderCacheOptions cache{.enableDiskCache = false};
        std::string diagnostics;
        const auto compiledA = compileSlangShaderToSpirv(a.desc(), cache, diagnostics);
        ASSERT_TRUE(compiledA) << diagnostics;
        const auto compiledB = compileSlangShaderToSpirv(b.desc(), cache, diagnostics);
        ASSERT_TRUE(compiledB) << diagnostics;
        EXPECT_TRUE(compiledA->spirv == compiledB->spirv);
    }
}

TEST_F(ShaderRequestCompileTest, FoldedStreamedDeferredProducesIdenticalSPIRV)
{
    auto options = standardSceneOptions();
    options.positionFetch = false; options.streamMaterials = true; options.globalView = true;
    options.realtimeDeferred = true; options.upscalerGuides = true; options.supplementaryPathTracing = true;
    const SlangMacroDefine transmission{"MATERIAL_CLASS", "4"};
    for (auto program : {SceneShaderProgram::Deferred, SceneShaderProgram::DeferredBinned}) {
        SCOPED_TRACE(static_cast<int>(program));
        const auto normalized = makeSceneShaderRequest(program, options,
            program == SceneShaderProgram::DeferredBinned ? std::span<const SlangMacroDefine>(&transmission, 1)
                                                         : std::span<const SlangMacroDefine>{});
        auto original = normalized;
        for (auto& [name, value] : original.defines) {
            if (name == "METALLIC_DEFERRED_PATH_TRACING") { value = "1"; }
        }
        const ShaderRequestView a(normalized), b(original);
        const SlangShaderCacheOptions cache{.enableDiskCache = false};
        std::string diagnostics;
        const auto compiledA = compileSlangShaderToSpirv(a.desc(), cache, diagnostics);
        ASSERT_TRUE(compiledA) << diagnostics;
        const auto compiledB = compileSlangShaderToSpirv(b.desc(), cache, diagnostics);
        ASSERT_TRUE(compiledB) << diagnostics;
        EXPECT_TRUE(compiledA->spirv == compiledB->spirv);
    }
}

} // namespace

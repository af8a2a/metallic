#include "Runtime/Render/Core/BuiltinShaderRequests.h"
#include "Tools/MaterialShaderWarmupRequests.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Scene/scene.h"

#include <gtest/gtest.h>
#include <json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unordered_map>

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

ShaderRequest nativeHeapStrideProbe(DescriptorHeapShaderStrides strides = {})
{
    return {.module = "NativeHeapStrideProbe", .entry = "main",
        .searchPath = PROJECT_SOURCE_DIR "/tests/editor/shaders",
        .descriptorHeapMode = SlangDescriptorHeapMode::Native, .descriptorHeapStrides = strides};
}

std::string uniqueShaderCacheDirectory(const char* name)
{
    return (std::filesystem::path(TEST_BINARY_DIR) / "shader-request-cache" / name /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())).string();
}

void expectNativeHeapStrides(const std::vector<uint32_t>& words, DescriptorHeapShaderStrides expected,
    bool requireImages = true, bool requireSamplers = true, bool requireBuffers = true)
{
    ASSERT_GE(words.size(), 5u);
    std::unordered_map<uint32_t, uint32_t> opaqueTypes;
    std::unordered_map<uint32_t, uint32_t> arrays;
    std::unordered_map<uint32_t, uint32_t> literalStrides;
    for (size_t offset = 5; offset < words.size();) {
        const uint32_t count = words[offset] >> 16, opcode = words[offset] & 0xffffu;
        ASSERT_GT(count, 0u);
        ASSERT_LE(offset + count, words.size());
        EXPECT_NE(opcode, 5129u); // OpConstantSizeOfEXT must never reach the cache or the driver.
        if (opcode == 25u || opcode == 26u || opcode == 5115u) { // Image, Sampler, BufferEXT
            ASSERT_GE(count, 2u);
            opaqueTypes.emplace(words[offset + 1], opcode);
        }
        if (opcode == 29u) { // OpTypeRuntimeArray
            ASSERT_EQ(count, 3u);
            arrays.emplace(words[offset + 1], words[offset + 2]);
        }
        if (opcode == 71u && count >= 4u && words[offset + 2] == 6u) { // OpDecorate ArrayStride
            ASSERT_EQ(count, 4u);
            literalStrides.emplace(words[offset + 1], words[offset + 3]);
        }
        if (opcode == 332u && count >= 3u) { // OpDecorateId
            EXPECT_NE(words[offset + 2], 6u); // Descriptor ArrayStride must be a literal, not an ID.
        }
        offset += count;
    }
    size_t bufferCount = 0, imageCount = 0, samplerCount = 0;
    for (const auto& [array, element] : arrays) {
        const auto type = opaqueTypes.find(element);
        if (type == opaqueTypes.end()) { continue; } // Ignore the buffer's ordinary data-element stride.
        SCOPED_TRACE(array);
        const auto stride = literalStrides.find(array);
        ASSERT_NE(stride, literalStrides.end());
        EXPECT_EQ(stride->second, type->second == 26u ? expected.sampler : expected.resource);
        if (type->second == 26u) { ++samplerCount; }
        else if (type->second == 25u) { ++imageCount; }
        else { ++bufferCount; }
    }
    if (requireBuffers) { EXPECT_GT(bufferCount, 0u); }
    if (requireImages) { EXPECT_GT(imageCount, 0u); }
    if (requireSamplers) { EXPECT_GT(samplerCount, 0u); }
}

class ShaderRequestCompileTest : public testing::Test {
protected:
    void SetUp() override
    {
        oldMode_ = slangShaderDebugMode();
        oldStrides_ = slangDescriptorHeapShaderStrides();
        setSlangShaderDebugMode(SlangShaderDebugMode::Disabled);
    }
    void TearDown() override
    {
        setSlangShaderDebugMode(oldMode_);
        setSlangDescriptorHeapShaderStrides(oldStrides_);
    }
private:
    SlangShaderDebugMode oldMode_;
    DescriptorHeapShaderStrides oldStrides_;
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

TEST(ShaderRequests, OwnsNativeDescriptorHeapStridesAcrossCopiesAndMoves)
{
    auto original = nativeHeapStrideProbe({48, 16});
    auto copied = original;
    original.descriptorHeapStrides = {64, 32};
    const auto moved = std::move(copied);
    EXPECT_NE(original, moved);
    const ShaderRequestView source(moved);
    EXPECT_EQ(source.desc().descriptorHeapStrides, (DescriptorHeapShaderStrides{48, 16}));
}

TEST_F(ShaderRequestCompileTest, NativeDescriptorHeapStridesSeparateTheCompilerCache)
{
    const std::string directory = uniqueShaderCacheDirectory("native-heap-strides");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    std::vector<std::vector<uint32_t>> binaries;
    for (const auto strides : {DescriptorHeapShaderStrides{48, 16},
            DescriptorHeapShaderStrides{64, 16}, DescriptorHeapShaderStrides{48, 32}}) {
        SCOPED_TRACE(strides.resource);
        SCOPED_TRACE(strides.sampler);
        const auto request = nativeHeapStrideProbe(strides);
        const ShaderRequestView source(request);
        const auto compiled = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
        ASSERT_TRUE(compiled) << diagnostics;
        EXPECT_FALSE(hit); // Both sizes participate in cache identity, independently.
        expectNativeHeapStrides(compiled->spirv, strides);
        binaries.push_back(compiled->spirv);
        const auto reused = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
        ASSERT_TRUE(reused) << diagnostics;
        EXPECT_TRUE(hit);
        EXPECT_EQ(compiled->spirv, reused->spirv);
    }
    EXPECT_NE(binaries[0], binaries[1]);
    EXPECT_NE(binaries[0], binaries[2]);
}

TEST_F(ShaderRequestCompileTest, NativeDescriptorHeapStridesStayLiteralWithShaderDebugInformation)
{
    const auto request = nativeHeapStrideProbe({48, 16});
    const ShaderRequestView source(request);
    std::string diagnostics;
    for (const auto mode : {SlangShaderDebugMode::CaptureSymbols, SlangShaderDebugMode::ShaderDebug}) {
        SCOPED_TRACE(static_cast<int>(mode));
        setSlangShaderDebugMode(mode);
        const auto compiled = compileSlangShaderToSpirv(source.desc(), {.enableDiskCache = false}, diagnostics);
        ASSERT_TRUE(compiled) << diagnostics;
        expectNativeHeapStrides(compiled->spirv, {48, 16});
    }
}

TEST_F(ShaderRequestCompileTest, MappedDescriptorHeapIgnoresNativeStridesInTheCompilerCache)
{
    const std::string directory = uniqueShaderCacheDirectory("mapped-heap-strides");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    std::vector<uint32_t> reference;
    for (const auto strides : {DescriptorHeapShaderStrides{}, DescriptorHeapShaderStrides{48, 16},
            DescriptorHeapShaderStrides{64, 32}}) {
        auto request = nativeHeapStrideProbe(strides);
        request.descriptorHeapMode = SlangDescriptorHeapMode::Mapped;
        const ShaderRequestView source(request);
        const auto compiled = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
        ASSERT_TRUE(compiled) << diagnostics;
        EXPECT_EQ(hit, !reference.empty());
        if (reference.empty()) { reference = compiled->spirv; }
        else { EXPECT_EQ(compiled->spirv, reference); }
    }
}

TEST_F(ShaderRequestCompileTest, NativeDescriptorHeapRequiresValidDeviceStrides)
{
    setSlangDescriptorHeapShaderStrides({});
    std::string diagnostics;
    const uint32_t tooLarge = uint32_t(std::numeric_limits<int32_t>::max()) + 1u;
    for (const auto strides : {DescriptorHeapShaderStrides{}, DescriptorHeapShaderStrides{48, 0},
            DescriptorHeapShaderStrides{0, 16}, DescriptorHeapShaderStrides{4, 16},
            DescriptorHeapShaderStrides{tooLarge, 16}, DescriptorHeapShaderStrides{48, tooLarge}}) {
        SCOPED_TRACE(strides.resource);
        SCOPED_TRACE(strides.sampler);
        const auto request = nativeHeapStrideProbe(strides);
        const ShaderRequestView source(request);
        const auto compiled = compileSlangShaderToSpirv(source.desc(), {.enableDiskCache = false}, diagnostics);
        EXPECT_FALSE(compiled);
        EXPECT_FALSE(diagnostics.empty());
    }
}

TEST_F(ShaderRequestCompileTest, NativeDescriptorHeapFallbackAndExplicitStridesShareTheCompilerCache)
{
    const std::string directory = uniqueShaderCacheDirectory("native-heap-device-policy");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    setSlangDescriptorHeapShaderStrides({48, 16});
    const auto implicit = nativeHeapStrideProbe();
    const ShaderRequestView implicitSource(implicit);
    const auto compiled = compileSlangShaderToSpirv(implicitSource.desc(), cache, diagnostics);
    ASSERT_TRUE(compiled) << diagnostics;
    EXPECT_FALSE(hit);
    expectNativeHeapStrides(compiled->spirv, {48, 16});
    setSlangDescriptorHeapShaderStrides({64, 32});
    const auto explicitRequest = nativeHeapStrideProbe({48, 16});
    const ShaderRequestView explicitSource(explicitRequest);
    const auto reused = compileSlangShaderToSpirv(explicitSource.desc(), cache, diagnostics);
    ASSERT_TRUE(reused) << diagnostics;
    EXPECT_TRUE(hit); // An explicit request takes priority over the current device's pair.
    EXPECT_EQ(compiled->spirv, reused->spirv);
    const auto otherDevice = compileSlangShaderToSpirv(implicitSource.desc(), cache, diagnostics);
    ASSERT_TRUE(otherDevice) << diagnostics;
    EXPECT_FALSE(hit);
    expectNativeHeapStrides(otherDevice->spirv, {64, 32});
    EXPECT_NE(compiled->spirv, otherDevice->spirv);
}

TEST_F(ShaderRequestCompileTest, NativeDescriptorHeapCapabilityRequiresTheSameDeviceStrides)
{
    setSlangDescriptorHeapShaderStrides({});
    auto request = nativeHeapStrideProbe();
    request.descriptorHeapMode = SlangDescriptorHeapMode::Mapped;
    request.capabilities = {"spvDescriptorHeapEXT"};
    const ShaderRequestView source(request);
    std::string diagnostics;
    EXPECT_FALSE(compileSlangShaderToSpirv(source.desc(), {.enableDiskCache = false}, diagnostics));
    request.descriptorHeapStrides = {48, 16};
    const auto compiled = compileSlangShaderToSpirv(source.desc(), {.enableDiskCache = false}, diagnostics);
    ASSERT_TRUE(compiled) << diagnostics;
    expectNativeHeapStrides(compiled->spirv, {48, 16});
}

TEST_F(ShaderRequestCompileTest, NativeWarmupExplicitStridesHitTheRuntimeDeviceCache)
{
    auto runtime = makeColorResizeShaderRequest();
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    const auto entry = std::find(catalog.begin(), catalog.end(), runtime);
    ASSERT_NE(entry, catalog.end());
    auto warmup = *entry;
    // Warmup enumerates device ABIs; runtime obtains the selected device ABI.
    warmup.descriptorHeapMode = SlangDescriptorHeapMode::Native;
    runtime.descriptorHeapMode = SlangDescriptorHeapMode::Native;
    const std::string directory = uniqueShaderCacheDirectory("native-warmup-runtime");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    for (const auto strides : {DescriptorHeapShaderStrides{48, 16}, DescriptorHeapShaderStrides{64, 32}}) {
        warmup.descriptorHeapStrides = strides;
        setSlangDescriptorHeapShaderStrides({});
        const ShaderRequestView warmupSource(warmup);
        const auto warmed = compileSlangShaderToSpirv(warmupSource.desc(), cache, diagnostics);
        ASSERT_TRUE(warmed) << diagnostics;
        EXPECT_FALSE(hit);
        expectNativeHeapStrides(warmed->spirv, strides, true, false, false);
        setSlangDescriptorHeapShaderStrides(strides);
        const ShaderRequestView runtimeSource(runtime);
        const auto reused = compileSlangShaderToSpirv(runtimeSource.desc(), cache, diagnostics);
        ASSERT_TRUE(reused) << diagnostics;
        EXPECT_TRUE(hit);
        EXPECT_EQ(warmed->spirv, reused->spirv);
    }
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
    const SlangMacroDefine cacheDefine{"SHARC_QUERY", "1"};
    const auto request = makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&cacheDefine, 1});
    EXPECT_EQ(request.searchPaths, (std::vector<std::string>{"C:/Values", "C:/RTXCR", "C:/NTC"}));
    EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "spvGroupNonUniformBallot",
        "spvRayQueryPositionFetchKHR", "spvCooperativeVectorNV"}));
    EXPECT_EQ(request.defines.front(), (std::pair<std::string, std::string>{"METALLIC_CUSTOM_MATERIALS", "1"}));
    EXPECT_EQ(request.defines.back(), (std::pair<std::string, std::string>{"SHARC_QUERY", "1"}));
    const auto maintenance = makeSceneShaderRequest(SceneShaderProgram::SharcClear, options);
    EXPECT_TRUE(maintenance.defines.empty());
    EXPECT_TRUE(maintenance.searchPaths.empty());
    EXPECT_EQ(maintenance.capabilities, request.capabilities);
}

TEST(ShaderRequests, DeferredDoesNotRequestRayTracingCapabilities)
{
    auto options = standardSceneOptions();
    options.streamRayQueries = true;
    for (auto program : {SceneShaderProgram::Deferred, SceneShaderProgram::DeferredBinned}) {
        const auto request = makeSceneShaderRequest(program, options);
        EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvGroupNonUniformBallot"}));
        EXPECT_TRUE(request.searchPaths.empty());
        EXPECT_NE(std::find(request.defines.begin(), request.defines.end(),
            std::pair<std::string, std::string>{"METALLIC_STREAM_RAY_QUERIES", "0"}), request.defines.end());
    }
}

TEST(ShaderRequests, CatalogCoversProductionSceneVariantsWithoutDuplicateRequests)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    EXPECT_TRUE(contains(catalog, makeColorResizeShaderRequest()));
    size_t deferredCount = 0;
    for (size_t i = 0; i < catalog.size(); ++i) {
        EXPECT_EQ(std::count(catalog.begin(), catalog.end(), catalog[i]), 1);
        if (catalog[i].module == "Features/VisibilityBuffer/VisibilityBufferDeferred") { ++deferredCount; }
    }
    EXPECT_EQ(deferredCount, 96u);
    for (bool fetch : {false, true}) {
        auto options = standardSceneOptions(); options.positionFetch = fetch;
        for (auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
                SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides,
                SceneShaderProgram::RealtimeLighting, SceneShaderProgram::SharcClear,
                SceneShaderProgram::SharcResolve}) {
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(program, options)));
        }
        for (const char* name : {"SHARC_UPDATE", "SHARC_QUERY"}) {
            const SlangMacroDefine define{name, "1"};
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&define, 1})));
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, options, {&define, 1})));
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&define, 1})));
        }
    }
    for (bool streamed : {false, true}) {
        for (bool guides : {false, true}) {
            auto options = standardSceneOptions();
            options.streamMaterials = streamed; options.globalView = true; options.upscalerGuides = guides;
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::Deferred, options)));
            for (int materialClass = 0; materialClass < 5; ++materialClass) {
                const std::string value = std::to_string(materialClass);
                const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
                EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, options, {&define, 1})));
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
    const ShaderRequestView runtimeSource(runtime);
    for (const auto mode : {SlangShaderDebugMode::Disabled,
            SlangShaderDebugMode::CaptureSymbols, SlangShaderDebugMode::ShaderDebug}) {
        SCOPED_TRACE(static_cast<int>(mode));
        setSlangShaderDebugMode(mode);
        const auto precompiled = compileSlangShaderToSpirv(warmupSource.desc(), cache, diagnostics);
        ASSERT_TRUE(precompiled) << diagnostics;
        EXPECT_FALSE(hit); // Each mode must have a separate cache identity.
        bool hasCalls = false;
        bool hasFunctionDebug = false;
        const auto& words = precompiled->spirv;
        for (size_t offset = 5; offset < words.size();) {
            const uint32_t count = words[offset] >> 16, opcode = words[offset] & 0xffffu;
            ASSERT_GT(count, 0u);
            ASSERT_LE(offset + count, words.size());
            if (opcode == 57) { hasCalls = true; } // OpFunctionCall
            if (opcode == 11 && count > 2) { // OpExtInstImport
                const std::string_view name(reinterpret_cast<const char*>(&words[offset + 2]),
                    (count - 2u) * sizeof(uint32_t));
                hasFunctionDebug |= name.starts_with("NonSemantic.Shader.DebugInfo");
            }
            offset += count;
        }
        EXPECT_TRUE(hasCalls); // Do not exhaustively inline the production scene shader.
        EXPECT_EQ(hasFunctionDebug, mode != SlangShaderDebugMode::Disabled);
        const auto reused = compileSlangShaderToSpirv(runtimeSource.desc(), cache, diagnostics);
        ASSERT_TRUE(reused) << diagnostics;
        EXPECT_TRUE(hit);
        EXPECT_EQ(precompiled->spirv, reused->spirv);
    }
}

TEST_F(ShaderRequestCompileTest, DeferredSPIRVIsRasterOnlyWithNativeFloat16)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    for (const auto& request : catalog) {
        if (request.module != "Features/VisibilityBuffer/VisibilityBufferDeferred") { continue; }
        SCOPED_TRACE(request.entry);
        const ShaderRequestView source(request);
        std::string diagnostics;
        const auto compiled = compileSlangShaderToSpirv(source.desc(), {}, diagnostics);
        ASSERT_TRUE(compiled) << diagnostics;
        const auto& words = compiled->spirv;
        bool float16 = false;
        for (size_t offset = 5; offset < words.size();) {
            const uint32_t count = words[offset] >> 16, opcode = words[offset] & 0xffffu;
            ASSERT_GT(count, 0u);
            ASSERT_LE(offset + count, words.size());
            if (opcode == 17) { // OpCapability
                EXPECT_NE(words[offset + 1], 4472u); // RayQueryKHR
                EXPECT_NE(words[offset + 1], 4479u); // RayTracingKHR
            }
            EXPECT_NE(opcode, 4472u); // OpTypeRayQueryKHR
            EXPECT_NE(opcode, 5341u); // OpTypeAccelerationStructureKHR
            if (opcode == 22 && words[offset + 2] == 16u) { float16 = true; }
            offset += count;
        }
        const bool background = std::find(request.defines.begin(), request.defines.end(),
            std::pair<std::string, std::string>{"MATERIAL_CLASS", "0"}) != request.defines.end();
        const bool half = std::find(request.defines.begin(), request.defines.end(),
            std::pair<std::string, std::string>{"METALLIC_DEFERRED_FP16", "1"}) != request.defines.end();
        if (!background && half) { EXPECT_TRUE(float16); }
    }
}

TEST_F(ShaderRequestCompileTest, LocalMaterialProgramsWarmBeforeRuntimeCreatesTheInclude)
{
    const auto root = std::filesystem::path(TEST_BINARY_DIR) / "local-material-warmup" /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const auto write = [&](const std::filesystem::path& path, const nlohmann::json& value) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << value.dump();
        ASSERT_TRUE(file);
    };
    const std::string valueProgram = R"({"version":1,"roughness":{"op":"parameter","index":0}})";
    write(root / "Sample/Scene.gltf", nlohmann::json::object());
    write(root / "Sample/Scene.metallic_scene.json",
        {{"materials", {{{"properties", {{"valueProgram", valueProgram}}}}}}});
    write(root / "Sample/Graph.json", {{"nodes", {
        {{"type", "ScenePathTracePass"}, {"properties", {{"bsdf", "openpbr"}}}},
        {{"type", "VisibilityBufferDeferredPass"}, {"properties", {{"materialBinning", false}}}}
    }}});
    const nlohmann::json sample{{"id", "painter-test"}, {"name", "Test"}, {"description", ""}, {"environment", ""},
        {"scenePath", "Sample/Scene.gltf"}, {"graphPath", "Sample/Graph.json"}};
    // Duplicate catalogs must not cause duplicate compilation.
    write(root / "build/MaterialValidation/PainterLookDev/Catalog.json", {{"version", 1}, {"samples", {sample}}});
    auto studio = sample;
    studio["id"] = "studio-white-test";
    write(root / "build/MaterialValidation/WhiteStudio02/Catalog.json", {{"version", 1}, {"samples", {studio}}});
    const auto requests = metallic::tools::materialShaderWarmupRequests(root, METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    ASSERT_EQ(requests.size(), 4u);
    metallic::scene::RenderMaterial material;
    material.valueProgram = valueProgram;
    std::string diagnostics;
    const auto runtimePrograms = MaterialValueProgramSet::create({&material, 1}, diagnostics);
    ASSERT_TRUE(runtimePrograms) << diagnostics;
    std::filesystem::path directory;
    ASSERT_TRUE(runtimePrograms->writeInclude(root / ".cache/materials", directory, diagnostics)) << diagnostics;
    const std::string cacheDirectory = (root / "spirv").string();
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = cacheDirectory.c_str(), .outCacheHit = &hit};
    for (const auto program : {SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::Deferred}) {
        auto options = standardSceneOptions();
        options.customMaterials = true;
        options.materialInclude = directory.string();
        options.globalView = program == SceneShaderProgram::Deferred;
        const auto runtime = makeSceneShaderRequest(program, options);
        const auto warmed = std::find(requests.begin(), requests.end(), runtime);
        ASSERT_NE(warmed, requests.end());
        const ShaderRequestView warmupSource(*warmed);
        const auto compiled = compileSlangShaderToSpirv(warmupSource.desc(), cache, diagnostics);
        ASSERT_TRUE(compiled) << diagnostics;
        EXPECT_FALSE(hit);
        const ShaderRequestView runtimeSource(runtime);
        const auto reused = compileSlangShaderToSpirv(runtimeSource.desc(), cache, diagnostics);
        ASSERT_TRUE(reused) << diagnostics;
        EXPECT_TRUE(hit);
        EXPECT_EQ(compiled->spirv, reused->spirv);
    }
    std::filesystem::remove(root / "Sample/Scene.gltf");
    EXPECT_TRUE(metallic::tools::materialShaderWarmupRequests(root, METALLIC_RTXCR_SHADER_INCLUDE_DIR).empty());
}

TEST(ShaderRequests, CoversStreamPrecisionViewAndDisplaySettings)
{
    const auto requests = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    auto scene = standardSceneOptions();
    scene.streamMaterials = true;
    scene.positionFetch = false;
    EXPECT_TRUE(contains(requests, makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, scene)));
    for (bool view : {false, true}) {
        for (bool half : {false, true}) {
            scene.globalView = view;
            scene.deferredFloat16 = half;
            EXPECT_TRUE(contains(requests, makeSceneShaderRequest(SceneShaderProgram::Deferred, scene)));
        }
    }
    EXPECT_TRUE(contains(requests, makeEditorDisplayShaderRequest("editorDisplayFragment", false,
        true, true, std::to_string(203.0f))));
    EXPECT_TRUE(contains(requests, makeEditorDisplayShaderRequest("editorOutputPQFragment", true,
        false, false, std::to_string(80.0f))));
}

} // namespace

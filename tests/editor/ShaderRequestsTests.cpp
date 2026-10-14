#include "Runtime/Render/Core/BuiltinShaderRequests.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Tools/MaterialShaderWarmupRequests.h"
#include "Runtime/Render/Material/MaterialValueProgram.h"
#include "Runtime/Scene/scene.h"

#include <gtest/gtest.h>
#include <json.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <unordered_map>

using namespace metallic::render;

namespace {

SceneShaderOptions standardSceneOptions()
{
    const std::string rtxcrInclude = METALLIC_RTXCR_SHADER_INCLUDE_DIR;
    return {.hasRTXCR = !rtxcrInclude.empty(), .rtxcrInclude = rtxcrInclude};
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

class ScopedOptimizationEnvironment {
public:
    ScopedOptimizationEnvironment()
    {
        if (const char* value = std::getenv("METALLIC_SLANG_OPTIMIZATION")) { original_ = value; }
        set(nullptr);
    }
    ~ScopedOptimizationEnvironment() { set(original_ ? original_->c_str() : nullptr); }

    void set(const char* value)
    {
#if defined(_WIN32)
        _putenv_s("METALLIC_SLANG_OPTIMIZATION", value ? value : "");
#else
        if (value) { setenv("METALLIC_SLANG_OPTIMIZATION", value, 1); }
        else { unsetenv("METALLIC_SLANG_OPTIMIZATION"); }
#endif
    }

private:
    std::optional<std::string> original_;
};

ShaderRequest optimizationProbe()
{
    return {.module = "OptimizationProbe", .entry = "main",
        .searchPath = PROJECT_SOURCE_DIR "/tests/editor/shaders"};
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
    ScopedOptimizationEnvironment optimizationEnvironment_;

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

TEST(ShaderRequests, OwnsOptimizationLevelAcrossCopiesAndMoves)
{
    auto original = optimizationProbe();
    original.optimizationLevel = SlangShaderOptimizationLevel::O2;
    auto copied = original;
    original.optimizationLevel = SlangShaderOptimizationLevel::O0;
    const auto moved = std::move(copied);
    EXPECT_NE(original, moved);
    const ShaderRequestView source(moved);
    EXPECT_EQ(source.desc().optimizationLevel, SlangShaderOptimizationLevel::O2);
}

TEST_F(ShaderRequestCompileTest, OptimizationLevelsSeparateTheCompilerCache)
{
    const std::string directory = uniqueShaderCacheDirectory("optimization-levels");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    auto request = optimizationProbe();
    for (const auto level : {SlangShaderOptimizationLevel::O0, SlangShaderOptimizationLevel::O1,
            SlangShaderOptimizationLevel::O2, SlangShaderOptimizationLevel::O3}) {
        SCOPED_TRACE(slangShaderOptimizationLevelName(level));
        request.optimizationLevel = level;
        const ShaderRequestView source(request);
        const auto compiled = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
        ASSERT_TRUE(compiled) << diagnostics;
        EXPECT_FALSE(hit);
        EXPECT_EQ(slangShaderOptimizationLevel(source.desc()), level);
        const auto reused = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
        ASSERT_TRUE(reused) << diagnostics;
        EXPECT_TRUE(hit);
        EXPECT_EQ(compiled->spirv, reused->spirv);
    }
    request.optimizationLevel = SlangShaderOptimizationLevel::Default;
    const ShaderRequestView source(request);
    const auto defaultResult = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(defaultResult) << diagnostics;
    EXPECT_TRUE(hit); // Preserve the existing release default's effective O1 identity.
}

TEST_F(ShaderRequestCompileTest, CompilationMetricsBelongToTheCurrentInvocation)
{
    const std::string directory = uniqueShaderCacheDirectory("compilation-metrics");
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str()};
    const auto request = optimizationProbe();
    const ShaderRequestView source(request);
    std::string diagnostics;
    const auto compiled = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(compiled) << diagnostics;
    const auto& metrics = compiled->metrics;
    EXPECT_TRUE(metrics.compilerExecuted);
    EXPECT_FALSE(metrics.cacheHit);
    EXPECT_FALSE(metrics.detailedCompilerPerfRequested);
    const uint64_t phaseSum = metrics.cacheLookupTimeNanoseconds + metrics.compilerSetupTimeNanoseconds +
        metrics.moduleLoadTimeNanoseconds + metrics.dependencyTrackingTimeNanoseconds +
        metrics.entryPointCompositionTimeNanoseconds + metrics.linkTimeNanoseconds +
        metrics.codeGenerationTimeNanoseconds + metrics.targetFinalizeTimeNanoseconds +
        metrics.dependencyValidationTimeNanoseconds + metrics.cacheWriteTimeNanoseconds;
    EXPECT_LE(phaseSum, metrics.totalTimeNanoseconds);

    const auto cached = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(cached) << diagnostics;
    EXPECT_EQ(cached->spirv, compiled->spirv);
    const auto& reused = cached->metrics;
    EXPECT_TRUE(reused.cacheHit);
    EXPECT_FALSE(reused.compilerExecuted);
    EXPECT_EQ(reused.compilerSetupTimeNanoseconds, 0u);
    EXPECT_EQ(reused.moduleLoadTimeNanoseconds, 0u);
    EXPECT_EQ(reused.entryPointCompositionTimeNanoseconds, 0u);
    EXPECT_EQ(reused.linkTimeNanoseconds, 0u);
    EXPECT_EQ(reused.codeGenerationTimeNanoseconds, 0u);
    EXPECT_EQ(reused.targetFinalizeTimeNanoseconds, 0u);
    EXPECT_EQ(reused.dependencyValidationTimeNanoseconds, 0u);
    EXPECT_EQ(reused.cacheWriteTimeNanoseconds, 0u);
    EXPECT_EQ(reused.compilerReportedTotalTimeNanoseconds, 0u);
    EXPECT_EQ(reused.compilerReportedDownstreamTimeNanoseconds, 0u);
    EXPECT_FALSE(reused.compilerReportedTimingAvailable);
    EXPECT_LE(reused.cacheLookupTimeNanoseconds + reused.dependencyTrackingTimeNanoseconds,
        reused.totalTimeNanoseconds);
}

TEST_F(ShaderRequestCompileTest, DetailedCompilerProfilingPreservesBytecodeAndCacheIdentity)
{
    const std::string directory = uniqueShaderCacheDirectory("detailed-compiler-profiling");
    const auto request = optimizationProbe();
    const ShaderRequestView source(request);
    std::string diagnostics;
    const auto baseline = compileSlangShaderToSpirv(source.desc(), {.enableDiskCache = false}, diagnostics);
    ASSERT_TRUE(baseline) << diagnostics;
    const auto profiled = compileSlangShaderToSpirv(source.desc(),
        {.cacheDirectory = directory.c_str(), .reportDetailedCompilerPerf = true}, diagnostics);
    ASSERT_TRUE(profiled) << diagnostics;
    EXPECT_EQ(profiled->spirv, baseline->spirv);
    EXPECT_TRUE(profiled->metrics.compilerExecuted);
    EXPECT_TRUE(profiled->metrics.detailedCompilerPerfRequested);

    const auto normalHit = compileSlangShaderToSpirv(source.desc(),
        {.cacheDirectory = directory.c_str()}, diagnostics);
    ASSERT_TRUE(normalHit) << diagnostics;
    EXPECT_TRUE(normalHit->metrics.cacheHit);
    EXPECT_FALSE(normalHit->metrics.detailedCompilerPerfRequested);
    const auto requestedHit = compileSlangShaderToSpirv(source.desc(),
        {.cacheDirectory = directory.c_str(), .reportDetailedCompilerPerf = true}, diagnostics);
    ASSERT_TRUE(requestedHit) << diagnostics;
    EXPECT_TRUE(requestedHit->metrics.cacheHit);
    EXPECT_TRUE(requestedHit->metrics.detailedCompilerPerfRequested);
    EXPECT_FALSE(requestedHit->metrics.compilerExecuted);
}

TEST_F(ShaderRequestCompileTest, ReusedCompilerKeepsSourceEditsAndMacroPermutationsFresh)
{
    const std::filesystem::path root = uniqueShaderCacheDirectory("compiler-reuse");
    const auto implementation = root / "Modules/Reuse/Value.slang";
    std::filesystem::create_directories(implementation.parent_path());
    const auto write = [&](const std::filesystem::path& path, const char* source) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream << source;
        return static_cast<bool>(stream);
    };
    ASSERT_TRUE(write(root / "CompilerReuseProbe.slang",
        "module CompilerReuseProbe;\nimport CompilerReuseValues;\n"
        "RWStructuredBuffer<uint> outputBuffer;\n"
        "[shader(\"compute\")] [numthreads(1, 1, 1)]\n"
        "void main() { outputBuffer[0] = compilerReuseValue(); }\n"));
    ASSERT_TRUE(write(root / "Modules/CompilerReuseValues.slang",
        "module CompilerReuseValues;\n__include \"Reuse/Value.slang\";\n"));
    ASSERT_TRUE(write(implementation,
        "implementing CompilerReuseValues;\n"
        "#ifndef REUSE_SCALE\n#define REUSE_SCALE 2u\n#endif\n"
        "public uint compilerReuseValue() { return REUSE_SCALE * 3u; }\n"));
    const std::string sourceRoot = root.string();
    SlangShaderDesc desc{.moduleName = "CompilerReuseProbe", .entryPointName = "main",
        .searchPath = sourceRoot.c_str()};
    const SlangShaderCacheOptions options{.enableDiskCache = false};
    std::string diagnostics;
    const auto baseline = compileSlangShaderToSpirv(desc, options, diagnostics);
    ASSERT_TRUE(baseline) << diagnostics;

    // A content-only edit must reach a fresh request session even though the
    // thread's global compiler and the source file metadata are unchanged.
    const auto originalTime = std::filesystem::last_write_time(implementation);
    const auto originalSize = std::filesystem::file_size(implementation);
    ASSERT_TRUE(write(implementation,
        "implementing CompilerReuseValues;\n"
        "#ifndef REUSE_SCALE\n#define REUSE_SCALE 2u\n#endif\n"
        "public uint compilerReuseValue() { return REUSE_SCALE * 7u; }\n"));
    ASSERT_EQ(std::filesystem::file_size(implementation), originalSize);
    std::filesystem::last_write_time(implementation, originalTime);
    const auto edited = compileSlangShaderToSpirv(desc, options, diagnostics);
    ASSERT_TRUE(edited) << diagnostics;
    EXPECT_NE(edited->spirv, baseline->spirv);
    EXPECT_EQ(edited->dependencies, baseline->dependencies);

    const SlangMacroDefine macro{"REUSE_SCALE", "3u"};
    desc.macroDefines = {&macro, 1};
    const auto variant = compileSlangShaderToSpirv(desc, options, diagnostics);
    ASSERT_TRUE(variant) << diagnostics;
    EXPECT_NE(variant->spirv, edited->spirv);
    desc.macroDefines = {};
    const auto restored = compileSlangShaderToSpirv(desc, options, diagnostics);
    ASSERT_TRUE(restored) << diagnostics;
    EXPECT_EQ(restored->spirv, edited->spirv);

    ASSERT_TRUE(write(implementation,
        "implementing CompilerReuseValues;\npublic uint compilerReuseValue() { return ; }\n"));
    EXPECT_FALSE(compileSlangShaderToSpirv(desc, options, diagnostics));
    EXPECT_FALSE(diagnostics.empty());
    ASSERT_TRUE(write(implementation,
        "implementing CompilerReuseValues;\npublic uint compilerReuseValue() { return 19u; }\n"));
    const auto recovered = compileSlangShaderToSpirv(desc, options, diagnostics);
    ASSERT_TRUE(recovered) << diagnostics;
    EXPECT_NE(recovered->spirv, edited->spirv);
}

TEST_F(ShaderRequestCompileTest, ReusedCompilerReportsOnlyCurrentInvocationTiming)
{
    const auto request = optimizationProbe();
    const ShaderRequestView source(request);
    std::string diagnostics;
    std::vector<uint32_t> expected;
    bool timingAvailable = false;
    for (size_t iteration = 0; iteration < 8; ++iteration) {
        SCOPED_TRACE(iteration);
        const auto compiled = compileSlangShaderToSpirv(source.desc(),
            {.enableDiskCache = false}, diagnostics);
        ASSERT_TRUE(compiled) << diagnostics;
        EXPECT_TRUE(compiled->metrics.compilerExecuted);
        EXPECT_FALSE(compiled->metrics.cacheHit);
        if (iteration == 0) { expected = compiled->spirv; }
        else { EXPECT_EQ(compiled->spirv, expected); }
        const auto& metrics = compiled->metrics;
        if (metrics.compilerReportedTimingAvailable) {
            timingAvailable = true;
            // SDK compiler time is contained in this call's measured wall time.
            // Allow 1 ms for differences in the SDK's clock resolution.
            EXPECT_LE(metrics.compilerReportedTotalTimeNanoseconds,
                metrics.totalTimeNanoseconds + 1'000'000u);
            EXPECT_LE(metrics.compilerReportedDownstreamTimeNanoseconds,
                metrics.totalTimeNanoseconds + 1'000'000u);
        }
    }
    if (!timingAvailable) { GTEST_SKIP() << "Slang SDK elapsed-time counters are unavailable"; }
}

TEST_F(ShaderRequestCompileTest, OptimizationOverrideDefaultsAndDebugInformationStayIndependent)
{
    const auto request = optimizationProbe();
    const ShaderRequestView source(request);
    EXPECT_EQ(slangShaderOptimizationLevel(source.desc()), SlangShaderOptimizationLevel::O1);
    setSlangShaderDebugMode(SlangShaderDebugMode::ShaderDebug);
    EXPECT_EQ(slangShaderOptimizationLevel(source.desc()), SlangShaderOptimizationLevel::O0);
    optimizationEnvironment_.set("O2");
    EXPECT_EQ(slangShaderOptimizationLevel(source.desc()), SlangShaderOptimizationLevel::O2);
    std::string diagnostics;
    const auto debugOptimized = compileSlangShaderToSpirv(source.desc(), {.enableDiskCache = false}, diagnostics);
    ASSERT_TRUE(debugOptimized) << diagnostics;
    optimizationEnvironment_.set("default");
    EXPECT_EQ(slangShaderOptimizationLevel(source.desc()), SlangShaderOptimizationLevel::O0);
    setSlangShaderDebugMode(SlangShaderDebugMode::CaptureSymbols);
    EXPECT_EQ(slangShaderOptimizationLevel(source.desc()), SlangShaderOptimizationLevel::O1);
}

TEST_F(ShaderRequestCompileTest, EnvironmentAndExplicitOptimizationShareOnlyTheSameEffectiveCache)
{
    const std::string directory = uniqueShaderCacheDirectory("optimization-environment");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    auto request = optimizationProbe();
    const ShaderRequestView source(request);
    optimizationEnvironment_.set("o2");
    const auto fromEnvironment = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(fromEnvironment) << diagnostics;
    EXPECT_FALSE(hit);
    request.optimizationLevel = SlangShaderOptimizationLevel::O2;
    optimizationEnvironment_.set("O0");
    const auto explicitO2 = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(explicitO2) << diagnostics;
    EXPECT_TRUE(hit);
    EXPECT_EQ(fromEnvironment->spirv, explicitO2->spirv);
    request.optimizationLevel = SlangShaderOptimizationLevel::Default;
    const auto fromO0 = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(fromO0) << diagnostics;
    EXPECT_FALSE(hit);
    optimizationEnvironment_.set("O2");
    const auto reusedO2 = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(reusedO2) << diagnostics;
    EXPECT_TRUE(hit);
    EXPECT_EQ(fromEnvironment->spirv, reusedO2->spirv);
}

TEST_F(ShaderRequestCompileTest, InvalidOptimizationFailsWithoutCreatingCacheEntries)
{
    const std::string directory = uniqueShaderCacheDirectory("invalid-optimization");
    bool hit = true;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    auto request = optimizationProbe();
    const ShaderRequestView source(request);
    optimizationEnvironment_.set("O4");
    EXPECT_EQ(slangShaderOptimizationLevel(source.desc()), SlangShaderOptimizationLevel::Default);
    const auto invalidEnvironment = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_FALSE(invalidEnvironment);
    EXPECT_EQ(invalidEnvironment.error(), Error::InvalidArgument);
    EXPECT_FALSE(hit);
    EXPECT_FALSE(diagnostics.empty());
    EXPECT_FALSE(std::filesystem::exists(directory));
    request.optimizationLevel = static_cast<SlangShaderOptimizationLevel>(255);
    const auto invalidRequest = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_FALSE(invalidRequest);
    EXPECT_EQ(invalidRequest.error(), Error::InvalidArgument);
    EXPECT_FALSE(std::filesystem::exists(directory));
    request.optimizationLevel = SlangShaderOptimizationLevel::O1;
    const auto validExplicit = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
    ASSERT_TRUE(validExplicit) << diagnostics; // An explicit valid level ignores an invalid process default.
    EXPECT_FALSE(hit);
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
        {"METALLIC_CUSTOM_MATERIALS", "0"}, {"METALLIC_GLOBAL_VIEW", "0"},
        {"METALLIC_HAS_RTXCR", "0"},
        {"METALLIC_DEFERRED_LIGHT_GRID", "0"},
        {"METALLIC_REALTIME_DEFERRED", "0"}, {"METALLIC_DEFERRED_UPSCALER_GUIDES", "0"},
    };
    EXPECT_EQ(request.defines, expected);
    EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV", "spvGroupNonUniformBallot"}));
}

TEST(ShaderRequests, PreservesCustomMaterialAndSDKSearchPrecedence)
{
    const SceneShaderOptions options{.customMaterials = true, .hasRTXCR = true,
        .materialInclude = "C:/Values", .rtxcrInclude = "C:/RTXCR"};
    const SlangMacroDefine cacheDefine{"SHARC_QUERY", "1"};
    const auto request = makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&cacheDefine, 1});
    EXPECT_EQ(request.searchPaths, (std::vector<std::string>{"C:/Values", "C:/RTXCR"}));
    EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV", "spvGroupNonUniformBallot"}));
    EXPECT_EQ(request.defines.front(), (std::pair<std::string, std::string>{"METALLIC_CUSTOM_MATERIALS", "1"}));
    EXPECT_EQ(request.defines.back(), (std::pair<std::string, std::string>{"SHARC_QUERY", "1"}));
    const auto maintenance = makeSceneShaderRequest(SceneShaderProgram::SharcClear, options);
    EXPECT_TRUE(maintenance.defines.empty());
    EXPECT_TRUE(maintenance.searchPaths.empty());
    EXPECT_EQ(maintenance.capabilities, request.capabilities);
}

TEST(ShaderRequests, DirectionalCacheNeverUsesSpatialKeyReprojection)
{
    const auto spatial = makeSceneShaderRequest(SceneShaderProgram::SharcResolve, {});
    const auto directional = makeSceneShaderRequest(SceneShaderProgram::OpenPBRSharcResolve, {});
    EXPECT_NE(spatial, directional);
    EXPECT_TRUE(spatial.defines.empty());
    EXPECT_EQ(directional.defines, (std::vector<std::pair<std::string, std::string>>{{"SHARC_BLEND_ADJACENT_LEVELS", "0"}}));
}

TEST(ShaderRequests, DeferredDoesNotRequestRayTracingCapabilities)
{
    auto options = standardSceneOptions();
    for (auto program : {SceneShaderProgram::Deferred, SceneShaderProgram::DeferredBinned}) {
        const auto request = makeSceneShaderRequest(program, options);
        EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvGroupNonUniformBallot"}));
        EXPECT_TRUE(request.searchPaths.empty());
        EXPECT_EQ(std::find_if(request.defines.begin(), request.defines.end(), [](const auto& define) {
            return define.first == "METALLIC_STREAM_RAY_QUERIES" || define.first == "SCENE_RAYQUERY_ENABLE_POSITION_FETCH";
        }), request.defines.end());
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
    EXPECT_EQ(deferredCount, 44u);
    {
        auto options = standardSceneOptions();
        for (auto program : {SceneShaderProgram::PathTrace, SceneShaderProgram::PathTraceGuides,
                SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides,
                SceneShaderProgram::RealtimeLighting, SceneShaderProgram::SharcClear,
                SceneShaderProgram::SharcResolve, SceneShaderProgram::OpenPBRSharcResolve}) {
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(program, options)));
        }
        for (const char* name : {"SHARC_UPDATE", "SHARC_QUERY"}) {
            const SlangMacroDefine define{name, "1"};
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::PathTrace, options, {&define, 1})));
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, options, {&define, 1})));
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&define, 1})));
        }
    }
    {
        for (bool guides : {false, true}) {
            auto options = standardSceneOptions();
            options.globalView = true; options.upscalerGuides = guides;
            EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::Deferred, options)));
            for (int materialClass = 0; materialClass < 5; ++materialClass) {
                const std::string value = std::to_string(materialClass);
                const SlangMacroDefine define{"MATERIAL_CLASS", value.c_str()};
                EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, options, {&define, 1})));
            }
        }
    }
}

TEST(ShaderRequests, OrdinaryShadersExcludeNeuralTextureVariants)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    EXPECT_TRUE(contains(catalog, makeSceneRayQueryRequest(SceneRayQueryProgram::RTXDI)));
    EXPECT_TRUE(contains(catalog, makeSceneRayQueryRequest(SceneRayQueryProgram::MaterialVisualization)));
    EXPECT_TRUE(contains(catalog, makeShadowShaderRequest()));
    for (const auto& request : catalog) {
        EXPECT_EQ(std::count(request.capabilities.begin(), request.capabilities.end(), "spvCooperativeVectorNV"), 0);
        for (const auto& define : request.defines) {
            EXPECT_EQ(define.first.find("NTC"), std::string::npos);
        }
    }
    for (const auto& request : {makeSceneRayQueryRequest(SceneRayQueryProgram::RTXDI),
            makeSceneRayQueryRequest(SceneRayQueryProgram::MaterialVisualization), makeShadowShaderRequest()}) {
        EXPECT_TRUE(request.defines.empty());
        EXPECT_TRUE(request.searchPaths.empty());
        EXPECT_EQ(request.capabilities, (std::vector<std::string>{"spvRayQueryKHR", "SPV_NV_cluster_acceleration_structure", "spvRayTracingClusterAccelerationStructureNV"}));
    }
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
        bool hasFunctionDebug = false;
        const auto& words = precompiled->spirv;
        for (size_t offset = 5; offset < words.size();) {
            const uint32_t count = words[offset] >> 16, opcode = words[offset] & 0xffffu;
            ASSERT_GT(count, 0u);
            ASSERT_LE(offset + count, words.size());
            if (opcode == 11 && count > 2) { // OpExtInstImport
                const std::string_view name(reinterpret_cast<const char*>(&words[offset + 2]),
                    (count - 2u) * sizeof(uint32_t));
                hasFunctionDebug |= name.starts_with("NonSemantic.Shader.DebugInfo");
            }
            offset += count;
        }
        // Default optimization may inline all calls; cache identity and debug metadata
        // must remain correct regardless of the optimized function graph.
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
    // One CLAS path trace request plus deferred render and deferred material-kernel requests.
    ASSERT_EQ(requests.size(), 3u);
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
        if (program == SceneShaderProgram::OpenPBRPathTrace) {
            // The custom-material wrapper must expose component evaluation and
            // sampling in both SHaRC paths, including the guide entry point.
            for (const char* mode : {"SHARC_QUERY", "SHARC_UPDATE"}) {
                for (const auto entry : {SceneShaderProgram::OpenPBRPathTrace, SceneShaderProgram::OpenPBRPathTraceGuides}) {
                    const SlangMacroDefine define{mode, "1"};
                    const auto request = makeSceneShaderRequest(entry, options, {&define, 1});
                    const ShaderRequestView source(request);
                    const auto cachedPath = compileSlangShaderToSpirv(source.desc(), cache, diagnostics);
                    ASSERT_TRUE(cachedPath) << mode << ": " << request.entry << '\n' << diagnostics;
                }
            }
        }
    }
    std::filesystem::remove(root / "Sample/Scene.gltf");
    EXPECT_TRUE(metallic::tools::materialShaderWarmupRequests(root, METALLIC_RTXCR_SHADER_INCLUDE_DIR).empty());
}

TEST(ShaderRequests, CoversStreamPrecisionViewAndDisplaySettings)
{
    const auto requests = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    auto scene = standardSceneOptions();
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

TEST(ShaderRequests, HDR10UsesDynamicResourcesOutsideImGuiABI)
{
    const auto imgui = makeEditorDisplayShaderRequest("editorDisplayFragment", true, false, false, "203.0");
    EXPECT_EQ(imgui.module, "Features/PostProcess/EditorDisplay");
    EXPECT_EQ(imgui.descriptorHeapMode, SlangDescriptorHeapMode::Mapped);
    for (const char* entry : {"editorOutputVertex", "editorOutputPQFragment"}) {
        const auto output = makeEditorDisplayShaderRequest(entry, true, false, false, "203.0");
        EXPECT_EQ(output.module, "Features/PostProcess/EditorOutput");
        EXPECT_EQ(output.descriptorHeapMode, SlangDescriptorHeapMode::Native);
        // PQ encodes absolute scRGB nits; UI paper white is already composed.
        EXPECT_EQ(output, makeEditorDisplayShaderRequest(entry, false, true, true, "80.0"));
    }
}

TEST(ShaderRequests, EditorDisplayVertexSharesAllFragmentDisplaySettings)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    const auto vertex = makeEditorDisplayShaderRequest("editorDisplayVertex", false, false, false,
        std::to_string(80.0f));
    EXPECT_EQ(vertex.module, "Features/PostProcess/EditorDisplay");
    EXPECT_EQ(vertex.descriptorHeapMode, SlangDescriptorHeapMode::Mapped);
    EXPECT_TRUE(vertex.defines.empty());
    EXPECT_EQ(std::count(catalog.begin(), catalog.end(), vertex), 1);
    std::vector<ShaderRequest> fragments;
    for (bool hdr : {false, true}) {
        for (bool scRgb : {false, true}) {
            for (bool sRGB : {false, true}) {
                for (float white : {80.0f, 203.0f}) {
                    const auto whiteNits = std::to_string(white);
                    EXPECT_EQ(makeEditorDisplayShaderRequest("editorDisplayVertex", hdr, scRgb, sRGB, whiteNits), vertex);
                    const auto fragment = makeEditorDisplayShaderRequest("editorDisplayFragment", hdr, scRgb, sRGB, whiteNits);
                    const bool identityHdr = hdr && !scRgb && whiteNits == "80.000000";
                    const bool canonicalHdr = hdr && !identityHdr;
                    const bool canonicalAttachment = identityHdr || (!hdr && sRGB);
                    EXPECT_EQ(fragment.defines, (std::vector<std::pair<std::string, std::string>>{
                        {"DISPLAY_HDR", canonicalHdr ? "1" : "0"},
                        {"DISPLAY_SCRGB_IMAGE", scRgb ? "1" : "0"},
                        {"DISPLAY_SRGB_ATTACHMENT", canonicalAttachment ? "1" : "0"},
                        {"DISPLAY_WHITE_NITS", canonicalHdr == scRgb ? "203.000000" : whiteNits}}));
                    EXPECT_EQ(std::count(catalog.begin(), catalog.end(), fragment), 1);
                    if (!contains(fragments, fragment)) { fragments.push_back(fragment); }
                }
            }
        }
    }
    EXPECT_EQ(std::count_if(catalog.begin(), catalog.end(), [](const auto& request) {
        return request.module == "Features/PostProcess/EditorDisplay" && request.entry == "editorDisplayVertex";
    }), 1);
    EXPECT_EQ(fragments.size(), 8u);
    EXPECT_EQ(std::count_if(catalog.begin(), catalog.end(), [](const auto& request) {
        return request.module == "Features/PostProcess/EditorDisplay" && request.entry == "editorDisplayFragment";
    }), fragments.size());
}

TEST(ShaderRequests, EditorDisplayFragmentOnlyCanonicalizesKnownInactiveSettings)
{
    const auto sRGBOutput = makeEditorDisplayShaderRequest("editorDisplayFragment", false, false, true, "203.000000");
    EXPECT_EQ(makeEditorDisplayShaderRequest("editorDisplayFragment", true, false, false, "80.000000"), sRGBOutput);
    for (const char* white : {"80.0", "80", "8e1", "(40.0 * 2.0)"}) {
        const auto hdr = makeEditorDisplayShaderRequest("editorDisplayFragment", true, false, true, white);
        EXPECT_NE(hdr, sRGBOutput);
        EXPECT_EQ(hdr.defines, (std::vector<std::pair<std::string, std::string>>{
            {"DISPLAY_HDR", "1"}, {"DISPLAY_SCRGB_IMAGE", "0"},
            {"DISPLAY_SRGB_ATTACHMENT", "0"}, {"DISPLAY_WHITE_NITS", white}}));
    }
    const auto unknown = makeEditorDisplayShaderRequest("futureDisplayEntry", true, true, true, "80.000000");
    EXPECT_EQ(unknown.defines, (std::vector<std::pair<std::string, std::string>>{
        {"DISPLAY_HDR", "1"}, {"DISPLAY_SCRGB_IMAGE", "1"},
        {"DISPLAY_SRGB_ATTACHMENT", "1"}, {"DISPLAY_WHITE_NITS", "80.000000"}}));
    EXPECT_NE(makeEditorDisplayShaderRequest("editorDisplayFragment", false, true, false, "80.000000"),
        makeEditorDisplayShaderRequest("editorDisplayFragment", false, true, false, "203.000000"));
}

TEST(ShaderRequests, DeferredBackgroundOnlySharesUnambiguousUnusedPrecision)
{
    const SceneShaderOptions half{.customMaterials = true, .globalView = true, .hasRTXCR = true,
        .deferredFloat16 = true, .upscalerGuides = true, .materialInclude = "C:/MaterialPrograms",
        .rtxcrInclude = "C:/RTXCR"};
    auto full = half;
    full.deferredFloat16 = false;
    const SlangMacroDefine background[]{{"BEFORE_CLASS", "3"}, {"MATERIAL_CLASS", "0"}, {"AFTER_CLASS", "7"}};
    const auto canonical = makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, half, background);
    EXPECT_EQ(canonical, makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, full, background));
    EXPECT_EQ(canonical.searchPaths, (std::vector<std::string>{half.materialInclude}));
    EXPECT_EQ((std::vector<std::pair<std::string, std::string>>(canonical.defines.end() - 4, canonical.defines.end())),
        (std::vector<std::pair<std::string, std::string>>{{"METALLIC_DEFERRED_FP16", "0"},
            {"BEFORE_CLASS", "3"}, {"MATERIAL_CLASS", "0"}, {"AFTER_CLASS", "7"}}));
    EXPECT_NE(makeSceneShaderRequest(SceneShaderProgram::Deferred, half, background),
        makeSceneShaderRequest(SceneShaderProgram::Deferred, full, background));
    EXPECT_NE(makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, half),
        makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, full));
    for (const char* value : {"00", "0u", "1", "2", "3", "4", "-1"}) {
        const SlangMacroDefine materialClass{"MATERIAL_CLASS", value};
        EXPECT_NE(makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, half, {&materialClass, 1}),
            makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, full, {&materialClass, 1}));
    }
    for (const auto& values : {std::pair{"0", "0"}, std::pair{"0", "1"}, std::pair{"1", "0"}}) {
        const SlangMacroDefine classes[]{{"MATERIAL_CLASS", values.first}, {"MATERIAL_CLASS", values.second}};
        EXPECT_NE(makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, half, classes),
            makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, full, classes));
    }
    for (const char* precision : {"0", "1"}) {
        const SlangMacroDefine overrides[]{{"MATERIAL_CLASS", "0"}, {"METALLIC_DEFERRED_FP16", precision}};
        const auto request = makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, half, overrides);
        EXPECT_NE(request, makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, full, overrides));
        EXPECT_EQ((std::vector<std::pair<std::string, std::string>>(request.defines.end() - 3, request.defines.end())),
            (std::vector<std::pair<std::string, std::string>>{{"METALLIC_DEFERRED_FP16", "1"},
                {"MATERIAL_CLASS", "0"}, {"METALLIC_DEFERRED_FP16", precision}}));
    }
}

TEST_F(ShaderRequestCompileTest, EditorDisplayCanonicalizationPreservesAllFinalBytecodeAndRuntimeCacheHits)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    const std::string directory = uniqueShaderCacheDirectory("display-canonicalization");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    std::vector<ShaderRequest> compiledRequests;
    std::vector<std::vector<uint32_t>> uniqueBytecode;
    for (bool hdr : {false, true}) {
        for (bool scRgb : {false, true}) {
            for (bool sRGB : {false, true}) {
                for (float white : {80.0f, 203.0f}) {
                    SCOPED_TRACE(hdr);
                    SCOPED_TRACE(scRgb);
                    SCOPED_TRACE(sRGB);
                    SCOPED_TRACE(white);
                    const auto whiteNits = std::to_string(white);
                    const ShaderRequest original{.module = "Features/PostProcess/EditorDisplay", .entry = "editorDisplayFragment",
                        .defines = {{"DISPLAY_HDR", hdr ? "1" : "0"}, {"DISPLAY_SCRGB_IMAGE", scRgb ? "1" : "0"},
                            {"DISPLAY_SRGB_ATTACHMENT", sRGB ? "1" : "0"}, {"DISPLAY_WHITE_NITS", whiteNits}},
                        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped, .optimizationLevel = SlangShaderOptimizationLevel::O1};
                    const ShaderRequestView originalSource(original);
                    const auto reference = compileSlangShaderToSpirv(originalSource.desc(), {.enableDiskCache = false}, diagnostics);
                    ASSERT_TRUE(reference) << diagnostics;
                    const auto runtime = makeEditorDisplayShaderRequest("editorDisplayFragment", hdr, scRgb, sRGB, whiteNits);
                    const auto warmup = std::find(catalog.begin(), catalog.end(), runtime);
                    ASSERT_NE(warmup, catalog.end());
                    const ShaderRequestView warmupSource(*warmup);
                    const auto compiled = ShaderRegistry::instance().getShader(warmupSource.desc(), cache, diagnostics);
                    ASSERT_TRUE(compiled) << diagnostics;
                    EXPECT_EQ(hit, contains(compiledRequests, runtime));
                    EXPECT_EQ(compiled->spirv, reference->spirv);
                    if (!contains(compiledRequests, runtime)) { compiledRequests.push_back(runtime); }
                    if (std::find(uniqueBytecode.begin(), uniqueBytecode.end(), reference->spirv) == uniqueBytecode.end()) {
                        uniqueBytecode.push_back(reference->spirv);
                    }
                    const ShaderRequestView runtimeSource(runtime);
                    const auto reused = ShaderRegistry::instance().getShader(runtimeSource.desc(), cache, diagnostics);
                    ASSERT_TRUE(reused) << diagnostics;
                    EXPECT_TRUE(hit);
                    EXPECT_EQ(reused->spirv, reference->spirv);
                    EXPECT_FALSE(reused->metrics.compilerExecuted);
                }
            }
        }
    }
    EXPECT_EQ(compiledRequests.size(), 8u);
    EXPECT_EQ(uniqueBytecode.size(), 8u);
}

TEST_F(ShaderRequestCompileTest, DeferredBackgroundCanonicalizationPreservesAllFinalBytecodeAndRuntimeCacheHits)
{
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    const std::string directory = uniqueShaderCacheDirectory("deferred-background-canonicalization");
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    std::vector<ShaderRequest> compiledRequests;
    const SlangMacroDefine materialClass{"MATERIAL_CLASS", "0"};
    for (bool guides : {false, true}) {
        for (bool view : {false, true}) {
            for (bool half : {false, true}) {
                SCOPED_TRACE(guides);
                SCOPED_TRACE(view);
                SCOPED_TRACE(half);
                const SceneShaderOptions options{.globalView = view, .deferredFloat16 = half, .upscalerGuides = guides};
                const auto runtime = makeSceneShaderRequest(SceneShaderProgram::DeferredBinned, options, {&materialClass, 1});
                auto original = runtime;
                original.descriptorHeapMode = SlangDescriptorHeapMode::Mapped;
                original.optimizationLevel = SlangShaderOptimizationLevel::O1;
                const auto precision = std::find_if(original.defines.begin(), original.defines.end(), [](const auto& define) {
                    return define.first == "METALLIC_DEFERRED_FP16";
                });
                ASSERT_NE(precision, original.defines.end());
                precision->second = half ? "1" : "0";
                const ShaderRequestView originalSource(original);
                const auto reference = compileSlangShaderToSpirv(originalSource.desc(), {.enableDiskCache = false}, diagnostics);
                ASSERT_TRUE(reference) << diagnostics;
                const auto warmup = std::find(catalog.begin(), catalog.end(), runtime);
                ASSERT_NE(warmup, catalog.end());
                auto canonical = *warmup;
                canonical.descriptorHeapMode = SlangDescriptorHeapMode::Mapped;
                const ShaderRequestView warmupSource(canonical);
                const auto compiled = ShaderRegistry::instance().getShader(warmupSource.desc(), cache, diagnostics);
                ASSERT_TRUE(compiled) << diagnostics;
                EXPECT_EQ(hit, contains(compiledRequests, runtime));
                EXPECT_EQ(compiled->spirv, reference->spirv);
                if (!contains(compiledRequests, runtime)) { compiledRequests.push_back(runtime); }
                auto mappedRuntime = runtime;
                mappedRuntime.descriptorHeapMode = SlangDescriptorHeapMode::Mapped;
                const ShaderRequestView runtimeSource(mappedRuntime);
                const auto reused = ShaderRegistry::instance().getShader(runtimeSource.desc(), cache, diagnostics);
                ASSERT_TRUE(reused) << diagnostics;
                EXPECT_TRUE(hit);
                EXPECT_EQ(reused->spirv, reference->spirv);
                EXPECT_FALSE(reused->metrics.compilerExecuted);
            }
        }
    }
    EXPECT_EQ(compiledRequests.size(), 4u);
}

TEST(ShaderRequests, OpenPBRGuideTrainingSharesThePathTraceRequest)
{
    const SlangMacroDefine update{"SHARC_UPDATE", "1"};
    const auto options = standardSceneOptions();
    const auto path = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, options, {&update, 1});
    const auto guides = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&update, 1});
    EXPECT_EQ(guides, path);
    const auto catalog = builtinShaderWarmupRequests(METALLIC_RTXCR_SHADER_INCLUDE_DIR);
    EXPECT_EQ(std::count(catalog.begin(), catalog.end(), path), 1);
    EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options)));
    const SlangMacroDefine query{"SHARC_QUERY", "1"};
    EXPECT_TRUE(contains(catalog, makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&query, 1})));
}

TEST(ShaderRequests, OpenPBRGuideTrainingPreservesCustomOptionsAndDefineOrder)
{
    const SceneShaderOptions options{.customMaterials = true, .globalView = true, .hasRTXCR = true,
        .materialInclude = "C:/MaterialPrograms", .rtxcrInclude = "C:/RTXCR"};
    const SlangMacroDefine defines[]{{"BEFORE_UPDATE", "3"}, {"SHARC_UPDATE", "1"}, {"AFTER_UPDATE", "7"}};
    const auto path = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTrace, options, defines);
    const auto guides = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, defines);
    EXPECT_EQ(guides, path);
    ASSERT_GE(guides.defines.size(), 3u);
    EXPECT_EQ((std::vector<std::pair<std::string, std::string>>(guides.defines.end() - 3, guides.defines.end())),
        (std::vector<std::pair<std::string, std::string>>{{"BEFORE_UPDATE", "3"}, {"SHARC_UPDATE", "1"}, {"AFTER_UPDATE", "7"}}));
}

TEST(ShaderRequests, OpenPBRGuideTrainingOnlySharesAnUnambiguousUpdateDefine)
{
    const auto options = standardSceneOptions();
    const auto ordinary = makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options);
    for (const char* value : {"0", "01", "true"}) {
        const SlangMacroDefine update{"SHARC_UPDATE", value};
        auto expected = ordinary;
        expected.defines.emplace_back(update.name, update.value);
        EXPECT_EQ(makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, {&update, 1}), expected);
    }
    for (const auto& values : {std::pair{"1", "0"}, std::pair{"0", "1"}, std::pair{"1", "1"}}) {
        const SlangMacroDefine updates[]{{"SHARC_UPDATE", values.first}, {"SHARC_UPDATE", values.second}};
        auto expected = ordinary;
        expected.defines.emplace_back(updates[0].name, updates[0].value);
        expected.defines.emplace_back(updates[1].name, updates[1].value);
        EXPECT_EQ(makeSceneShaderRequest(SceneShaderProgram::OpenPBRPathTraceGuides, options, updates), expected);
    }
}

} // namespace

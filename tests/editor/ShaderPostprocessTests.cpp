#include "Runtime/Render/Core/SlangCompiler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace metallic::render;

namespace {

constexpr SlangShaderPostprocessMode kCleanupModes[] = {
    SlangShaderPostprocessMode::DeadBranches, SlangShaderPostprocessMode::SlangDeadBranches};

size_t opcodeCount(std::span<const uint32_t> spirv, uint16_t opcode)
{
    if (spirv.size() < 5 || spirv.front() != 0x07230203u) {
        throw std::runtime_error("Invalid probe SPIR-V header");
    }
    size_t count = 0;
    for (size_t offset = 5; offset < spirv.size();) {
        const uint32_t length = spirv[offset] >> 16;
        if (!length || offset + length > spirv.size()) {
            throw std::runtime_error("Invalid probe SPIR-V instruction length");
        }
        count += (spirv[offset] & 0xffffu) == opcode;
        offset += length;
    }
    return count;
}

size_t literalConditionalCount(std::span<const uint32_t> spirv)
{
    (void)opcodeCount(spirv, 250); // Validate bounds before reading operands.
    std::vector<uint32_t> literalIds;
    for (size_t offset = 5; offset < spirv.size(); offset += spirv[offset] >> 16) {
        const auto opcode = spirv[offset] & 0xffffu;
        if ((opcode == 41 || opcode == 42) && (spirv[offset] >> 16) == 3) {
            literalIds.push_back(spirv[offset + 2]); // OpConstantTrue / OpConstantFalse result ID.
        }
    }
    size_t count = 0;
    for (size_t offset = 5; offset < spirv.size(); offset += spirv[offset] >> 16) {
        if ((spirv[offset] & 0xffffu) == 250 && (spirv[offset] >> 16) >= 4 &&
            std::find(literalIds.begin(), literalIds.end(), spirv[offset + 1]) != literalIds.end()) {
            ++count;
        }
    }
    return count;
}

class ScopedShaderEnvironment {
public:
    explicit ScopedShaderEnvironment(const char* name) : name_(name)
    {
        if (const char* value = std::getenv(name)) { original_ = value; }
        set(nullptr);
    }
    ~ScopedShaderEnvironment() { set(original_ ? original_->c_str() : nullptr); }

    void set(const char* value)
    {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value) { setenv(name_.c_str(), value, 1); }
        else { unsetenv(name_.c_str()); }
#endif
    }

private:
    std::string name_;
    std::optional<std::string> original_;
};

class ShaderPostprocessTest : public testing::Test {
protected:
    ScopedShaderEnvironment postprocessEnvironment_{"METALLIC_SPIRV_POSTPROCESS"};
    ScopedShaderEnvironment optimizationEnvironment_{"METALLIC_SLANG_OPTIMIZATION"};
    ScopedShaderEnvironment descriptorEnvironment_{"METALLIC_SLANG_DESCRIPTOR_MODE"};

    void SetUp() override
    {
        oldMode_ = slangShaderDebugMode();
        setSlangShaderDebugMode(SlangShaderDebugMode::Disabled);
        directory_ = (std::filesystem::path(TEST_BINARY_DIR) / "shader-postprocess-cache" /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())).string();
    }
    void TearDown() override { setSlangShaderDebugMode(oldMode_); }

    SlangShaderDesc probe() const
    {
        return {.moduleName = "OptimizationProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/editor/shaders",
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped,
            .optimizationLevel = SlangShaderOptimizationLevel::O1};
    }

    SlangShaderCacheOptions cache() const { return {.cacheDirectory = directory_.c_str()}; }
    bool available(SlangShaderPostprocessMode mode = SlangShaderPostprocessMode::DeadBranches) const
    {
        return std::string_view(slangShaderPostprocessToolVersion(mode)) != "unavailable";
    }

    std::string directory_;
    std::string diagnostics_;

private:
    SlangShaderDebugMode oldMode_;
};

TEST_F(ShaderPostprocessTest, DefaultAndExplicitNonePreserveBytecodeAndShareCache)
{
    auto desc = probe();
    EXPECT_EQ(slangShaderPostprocessMode(desc), SlangShaderPostprocessMode::None);
    const auto baseline = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_TRUE(baseline) << diagnostics_;
    EXPECT_FALSE(baseline->metrics.cacheHit);
    EXPECT_EQ(baseline->metrics.postprocessTimeNanoseconds, 0u);
    EXPECT_EQ(baseline->metrics.rawSpirvByteSize, baseline->spirv.size() * sizeof(uint32_t));
    EXPECT_EQ(baseline->metrics.outputSpirvByteSize, baseline->spirv.size() * sizeof(uint32_t));

    desc.postprocessMode = SlangShaderPostprocessMode::None;
    for (const auto mode : kCleanupModes) {
        postprocessEnvironment_.set(slangShaderPostprocessModeName(mode));
        EXPECT_EQ(slangShaderPostprocessMode(desc), SlangShaderPostprocessMode::None);
        const auto explicitNone = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_TRUE(explicitNone) << diagnostics_;
        EXPECT_TRUE(explicitNone->metrics.cacheHit);
        EXPECT_EQ(explicitNone->spirv, baseline->spirv);
        EXPECT_EQ(explicitNone->metrics.postprocessTimeNanoseconds, 0u);
        EXPECT_EQ(explicitNone->metrics.rawSpirvByteSize, 0u);
        EXPECT_EQ(explicitNone->metrics.outputSpirvByteSize, baseline->spirv.size() * sizeof(uint32_t));
    }

    desc.postprocessMode = SlangShaderPostprocessMode::Default;
    for (const char* value : {"default", "none"}) {
        postprocessEnvironment_.set(value);
        const auto reused = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_TRUE(reused) << diagnostics_;
        EXPECT_TRUE(reused->metrics.cacheHit);
        EXPECT_EQ(reused->spirv, baseline->spirv);
    }
}

TEST_F(ShaderPostprocessTest, CleanupUsesASeparateCacheAndRunsOnlyOnCacheMiss)
{
    auto desc = probe();
    const auto unchanged = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_TRUE(unchanged) << diagnostics_;
    std::optional<std::vector<uint32_t>> externalBytes;
    for (const auto mode : kCleanupModes) {
        SCOPED_TRACE(slangShaderPostprocessModeName(mode));
        desc.postprocessMode = mode;
        const auto cleaned = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        if (!available(mode)) {
            ASSERT_FALSE(cleaned);
            EXPECT_FALSE(diagnostics_.empty());
            continue;
        }
        ASSERT_TRUE(cleaned) << diagnostics_;
        EXPECT_FALSE(cleaned->metrics.cacheHit); // Separate keys even when both modes emit identical bytes.
        EXPECT_TRUE(cleaned->metrics.compilerExecuted);
        EXPECT_GT(cleaned->metrics.postprocessTimeNanoseconds, 0u);
        const auto expectedRaw = mode == SlangShaderPostprocessMode::SlangDeadBranches
            ? cleaned->spirv.size() : unchanged->spirv.size();
        EXPECT_EQ(cleaned->metrics.rawSpirvByteSize, expectedRaw * sizeof(uint32_t));
        EXPECT_EQ(cleaned->metrics.outputSpirvByteSize, cleaned->spirv.size() * sizeof(uint32_t));
        if (mode == SlangShaderPostprocessMode::DeadBranches) { externalBytes = cleaned->spirv; }
        else if (externalBytes) { EXPECT_EQ(cleaned->spirv, *externalBytes); }

        desc.postprocessMode = SlangShaderPostprocessMode::Default;
        postprocessEnvironment_.set(slangShaderPostprocessModeName(mode));
        EXPECT_EQ(slangShaderPostprocessMode(desc), mode);
        const auto reused = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_TRUE(reused) << diagnostics_;
        EXPECT_TRUE(reused->metrics.cacheHit);
        EXPECT_FALSE(reused->metrics.compilerExecuted);
        EXPECT_EQ(reused->spirv, cleaned->spirv);
        EXPECT_EQ(reused->metrics.codeGenerationTimeNanoseconds, 0u);
        EXPECT_EQ(reused->metrics.targetFinalizeTimeNanoseconds, 0u);
        EXPECT_EQ(reused->metrics.postprocessTimeNanoseconds, 0u);
        EXPECT_EQ(reused->metrics.rawSpirvByteSize, 0u);
        EXPECT_EQ(reused->metrics.outputSpirvByteSize, cleaned->spirv.size() * sizeof(uint32_t));
    }

    desc.postprocessMode = SlangShaderPostprocessMode::None;
    const auto originalHit = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_TRUE(originalHit) << diagnostics_;
    EXPECT_TRUE(originalHit->metrics.cacheHit);
    EXPECT_EQ(originalHit->spirv, unchanged->spirv);
}

TEST_F(ShaderPostprocessTest, SlangAppendedPassesPreserveSelectedO0AndEliminateResolvedO1Branches)
{
    if (!available(SlangShaderPostprocessMode::SlangDeadBranches) || !available()) {
        GTEST_SKIP() << "Both appended Slang passes and the external cleanup control require SPIRV-Tools";
    }
    for (const auto level : {SlangShaderOptimizationLevel::O0, SlangShaderOptimizationLevel::O1}) {
        SCOPED_TRACE(slangShaderOptimizationLevelName(level));
        auto desc = probe();
        desc.moduleName = "ShaderPostprocessProbe";
        desc.optimizationLevel = level;
        desc.postprocessMode = SlangShaderPostprocessMode::None;
        const auto baseline = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_TRUE(baseline) << diagnostics_;
        ASSERT_GT(opcodeCount(baseline->spirv, 250), 0u);
        if (level == SlangShaderOptimizationLevel::O0) {
            // O0 preserves enableExtra() as a call. Its result is not a direct
            // literal branch condition; these two passes do not inline it.
            EXPECT_GT(opcodeCount(baseline->spirv, 57), 0u);
            EXPECT_EQ(literalConditionalCount(baseline->spirv), 0u);
        } else {
            ASSERT_EQ(opcodeCount(baseline->spirv, 57), 0u);
            ASSERT_EQ(literalConditionalCount(baseline->spirv), 1u);
        }
        desc.postprocessMode = SlangShaderPostprocessMode::DeadBranches;
        const auto external = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_TRUE(external) << diagnostics_;
        desc.postprocessMode = SlangShaderPostprocessMode::SlangDeadBranches;
        const auto appended = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_TRUE(appended) << diagnostics_;
        EXPECT_FALSE(appended->metrics.cacheHit);
        EXPECT_TRUE(appended->metrics.compilerExecuted);
        EXPECT_EQ(appended->spirv, external->spirv);
        if (level == SlangShaderOptimizationLevel::O0) {
            EXPECT_EQ(appended->spirv, baseline->spirv); // Preserve O0; do not silently promote optimization.
        } else {
            EXPECT_NE(appended->spirv, baseline->spirv);
            EXPECT_LT(appended->spirv.size(), baseline->spirv.size());
            EXPECT_LT(opcodeCount(appended->spirv, 248), opcodeCount(baseline->spirv, 248));
            EXPECT_EQ(opcodeCount(appended->spirv, 250), 0u);
        }
        EXPECT_EQ(appended->metrics.rawSpirvByteSize, appended->spirv.size() * sizeof(uint32_t));
        EXPECT_GT(appended->metrics.postprocessTimeNanoseconds, 0u); // Final validation, not a second optimizer run.
        const auto reused = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_TRUE(reused) << diagnostics_;
        EXPECT_TRUE(reused->metrics.cacheHit);
        EXPECT_FALSE(reused->metrics.compilerExecuted);
        EXPECT_EQ(reused->spirv, appended->spirv);
        EXPECT_EQ(reused->metrics.codeGenerationTimeNanoseconds, 0u);
        EXPECT_EQ(reused->metrics.postprocessTimeNanoseconds, 0u);
    }
}

TEST_F(ShaderPostprocessTest, InvalidPolicyFailsBeforeWritingCacheAndExplicitNoneOverridesIt)
{
    auto desc = probe();
    postprocessEnvironment_.set("dead-branch");
    EXPECT_EQ(slangShaderPostprocessMode(desc), SlangShaderPostprocessMode::Default);
    const auto invalidEnvironment = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_FALSE(invalidEnvironment);
    EXPECT_EQ(invalidEnvironment.error(), Error::InvalidArgument);
    EXPECT_FALSE(diagnostics_.empty());
    EXPECT_FALSE(std::filesystem::exists(directory_));

    desc.postprocessMode = static_cast<SlangShaderPostprocessMode>(255);
    const auto invalidRequest = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_FALSE(invalidRequest);
    EXPECT_EQ(invalidRequest.error(), Error::InvalidArgument);
    EXPECT_FALSE(std::filesystem::exists(directory_));

    desc.postprocessMode = SlangShaderPostprocessMode::None;
    const auto explicitNone = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_TRUE(explicitNone) << diagnostics_;
    EXPECT_EQ(explicitNone->metrics.postprocessTimeNanoseconds, 0u);
}

TEST_F(ShaderPostprocessTest, NativeCleanupIsRejectedBeforeCacheLookup)
{
    auto desc = probe();
    desc.descriptorHeapMode = SlangDescriptorHeapMode::Native;
    desc.descriptorHeapStrides = {64, 32};
    for (const auto mode : kCleanupModes) {
        SCOPED_TRACE(slangShaderPostprocessModeName(mode));
        desc.postprocessMode = mode;
        const auto result = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), Error::InvalidArgument);
        EXPECT_FALSE(diagnostics_.empty());
        EXPECT_FALSE(std::filesystem::exists(directory_));
    }
}

TEST_F(ShaderPostprocessTest, UnavailableOptimizerDoesNotFallBackToSourceOrWriteCache)
{
    auto desc = probe();
    bool unavailable = false;
    for (const auto mode : kCleanupModes) {
        if (available(mode)) { continue; }
        unavailable = true;
        SCOPED_TRACE(slangShaderPostprocessModeName(mode));
        desc.postprocessMode = mode;
        const auto result = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
        ASSERT_FALSE(result);
        EXPECT_FALSE(diagnostics_.empty());
        EXPECT_FALSE(std::filesystem::exists(directory_));
    }
    if (!unavailable) { GTEST_SKIP() << "Run with METALLIC_ENABLE_SPIRV_TOOLS=OFF to exercise unavailable backend"; }
}

} // namespace

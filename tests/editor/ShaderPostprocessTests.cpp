#include "Runtime/Render/Core/SlangCompiler.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

using namespace metallic::render;

namespace {

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
    bool available() const { return std::string_view(slangShaderPostprocessToolVersion()) != "unavailable"; }

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
    postprocessEnvironment_.set("dead-branches");
    EXPECT_EQ(slangShaderPostprocessMode(desc), SlangShaderPostprocessMode::None);
    const auto explicitNone = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_TRUE(explicitNone) << diagnostics_;
    EXPECT_TRUE(explicitNone->metrics.cacheHit);
    EXPECT_EQ(explicitNone->spirv, baseline->spirv);
    EXPECT_EQ(explicitNone->metrics.postprocessTimeNanoseconds, 0u);
    EXPECT_EQ(explicitNone->metrics.rawSpirvByteSize, 0u);
    EXPECT_EQ(explicitNone->metrics.outputSpirvByteSize, baseline->spirv.size() * sizeof(uint32_t));

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
    desc.postprocessMode = SlangShaderPostprocessMode::DeadBranches;
    const auto cleaned = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    if (!available()) {
        ASSERT_FALSE(cleaned);
        EXPECT_FALSE(diagnostics_.empty());
        return;
    }
    ASSERT_TRUE(cleaned) << diagnostics_;
    EXPECT_FALSE(cleaned->metrics.cacheHit); // Policy separates the cache even when the bytes happen to match.
    EXPECT_TRUE(cleaned->metrics.compilerExecuted);
    EXPECT_GT(cleaned->metrics.postprocessTimeNanoseconds, 0u);
    EXPECT_EQ(cleaned->metrics.rawSpirvByteSize, unchanged->spirv.size() * sizeof(uint32_t));
    EXPECT_EQ(cleaned->metrics.outputSpirvByteSize, cleaned->spirv.size() * sizeof(uint32_t));

    desc.postprocessMode = SlangShaderPostprocessMode::Default;
    postprocessEnvironment_.set("dead-branches");
    const auto reused = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_TRUE(reused) << diagnostics_;
    EXPECT_TRUE(reused->metrics.cacheHit);
    EXPECT_FALSE(reused->metrics.compilerExecuted);
    EXPECT_EQ(reused->spirv, cleaned->spirv);
    EXPECT_EQ(reused->metrics.postprocessTimeNanoseconds, 0u);
    EXPECT_EQ(reused->metrics.rawSpirvByteSize, 0u);
    EXPECT_EQ(reused->metrics.outputSpirvByteSize, cleaned->spirv.size() * sizeof(uint32_t));

    desc.postprocessMode = SlangShaderPostprocessMode::None;
    const auto originalHit = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_TRUE(originalHit) << diagnostics_;
    EXPECT_TRUE(originalHit->metrics.cacheHit);
    EXPECT_EQ(originalHit->spirv, unchanged->spirv);
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
    desc.postprocessMode = SlangShaderPostprocessMode::DeadBranches;
    const auto result = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_FALSE(result);
    EXPECT_FALSE(diagnostics_.empty());
    EXPECT_FALSE(std::filesystem::exists(directory_));
}

TEST_F(ShaderPostprocessTest, UnavailableOptimizerDoesNotFallBackToSourceOrWriteCache)
{
    if (available()) { GTEST_SKIP() << "Run with METALLIC_ENABLE_SPIRV_TOOLS=OFF to exercise unavailable backend"; }
    auto desc = probe();
    desc.postprocessMode = SlangShaderPostprocessMode::DeadBranches;
    const auto result = compileSlangShaderToSpirv(desc, cache(), diagnostics_);
    ASSERT_FALSE(result);
    EXPECT_FALSE(diagnostics_.empty());
    EXPECT_FALSE(std::filesystem::exists(directory_));
}

} // namespace

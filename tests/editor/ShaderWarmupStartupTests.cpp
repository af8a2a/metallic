#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Tools/ShaderWarmupRequests.h"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {

std::atomic<size_t> compileCalls{0};
std::atomic<bool> failCompilation{false};
std::atomic<bool> throwCompilation{false};
std::atomic<bool> wrongDebugMode{false};
metallic::render::SlangShaderDebugMode debugMode = metallic::render::SlangShaderDebugMode::CaptureSymbols;

class ShaderWarmupStartupTest : public testing::Test {
protected:
    void SetUp() override
    {
        compileCalls = 0;
        failCompilation = false;
        throwCompilation = false;
        wrongDebugMode = false;
        debugMode = metallic::render::SlangShaderDebugMode::CaptureSymbols;
        oldLogLevel_ = spdlog::get_level();
        oldFlags_ = std::cout.flags();
        oldPrecision_ = std::cout.precision();
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
    }

    void TearDown() override
    {
        testing::internal::GetCapturedStdout();
        testing::internal::GetCapturedStderr();
        spdlog::set_level(oldLogLevel_);
        std::cout.flags(oldFlags_);
        std::cout.precision(oldPrecision_);
    }

private:
    spdlog::level::level_enum oldLogLevel_;
    std::ios::fmtflags oldFlags_;
    std::streamsize oldPrecision_;
};

TEST_F(ShaderWarmupStartupTest, DefaultsToCompleteCatalogAndPreservesApplicationState)
{
    metallic::render::ShaderWarmupLaunchOptions options;
    EXPECT_FALSE(options.skip);
    spdlog::set_level(spdlog::level::debug);
    std::cout << std::scientific;
    std::cout.precision(7);
    const auto flags = std::cout.flags();
    ASSERT_EQ(metallic::render::warmupShadersForStartup(options.skip), 0);
    EXPECT_EQ(compileCalls.load(), metallic::tools::shaderWarmupRequests().size());
    EXPECT_FALSE(wrongDebugMode.load());
    EXPECT_EQ(spdlog::get_level(), spdlog::level::debug);
    EXPECT_EQ(std::cout.flags(), flags);
    EXPECT_EQ(std::cout.precision(), 7);
}

TEST_F(ShaderWarmupStartupTest, OnlyExplicitSkipOptionBypassesCompilation)
{
    metallic::render::ShaderWarmupLaunchOptions options;
    EXPECT_FALSE(options.consume("--smoke-test"));
    EXPECT_FALSE(options.consume("--skip-shader-warmup=false"));
    EXPECT_FALSE(options.skip);
    EXPECT_TRUE(options.consume("--skip-shader-warmup"));
    failCompilation = true;
    EXPECT_EQ(metallic::render::warmupShadersForStartup(options.skip), 0);
    EXPECT_EQ(compileCalls.load(), 0u);
}

TEST_F(ShaderWarmupStartupTest, CompilationFailurePreventsStartup)
{
    failCompilation = true;
    EXPECT_NE(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_EQ(compileCalls.load(), metallic::tools::shaderWarmupRequests().size());
}

TEST_F(ShaderWarmupStartupTest, WorkerExceptionPreventsStartupAndRestoresLogging)
{
    throwCompilation = true;
    spdlog::set_level(spdlog::level::info);
    EXPECT_NE(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_EQ(compileCalls.load(), metallic::tools::shaderWarmupRequests().size());
    EXPECT_EQ(spdlog::get_level(), spdlog::level::info);
}

} // namespace

// Link a deterministic compiler substitute, not Slang/Vulkan. This exercises
// the production startup gate and workers without GPU/SDK/cache dependencies.
namespace metallic::render {

void setSlangShaderDebugMode(SlangShaderDebugMode mode) noexcept
{
    debugMode = mode;
}

SlangShaderDebugMode slangShaderDebugMode() noexcept
{
    return debugMode;
}

Result<ShaderCompileResult> compileSlangShaderToSpirv(
    const SlangShaderDesc&,
    const SlangShaderCacheOptions& options,
    std::string& diagnostics)
{
    ++compileCalls;
    if (debugMode != SlangShaderDebugMode::CaptureSymbols) {
        wrongDebugMode = true;
    }
    if (throwCompilation) {
        throw std::runtime_error("injected compiler exception");
    }
    if (failCompilation) {
        diagnostics = "injected compiler failure";
        return makeError(Error::Failure);
    }
    if (options.outCacheHit != nullptr) {
        *options.outCacheHit = true;
    }
    return ShaderCompileResult{};
}

} // namespace metallic::render

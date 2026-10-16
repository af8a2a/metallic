#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Runtime/Render/Core/SlangCompiler.h"
#include "Runtime/Render/GAPI/ShaderTarget.h"
#include "Tools/ShaderWarmupRequests.h"

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

std::atomic<size_t> compileCalls{0};
std::atomic<size_t> batchCalls{0};
std::atomic<size_t> batchJobs{0};
std::atomic<bool> failCompilation{false};
std::atomic<bool> throwCompilation{false};
std::atomic<bool> wrongDebugMode{false};
std::atomic<size_t> adapterQueries{0};
std::atomic<size_t> firstNativeStrideCalls{0};
std::atomic<bool> wrongNativeStride{false};
bool failAdapterQuery = false;
bool noAdapters = false;
bool nativeCompilation = false;
std::string nativeEntry;
metallic::render::SlangShaderDebugMode debugMode = metallic::render::SlangShaderDebugMode::CaptureSymbols;

void setWorkerEnvironment(const char* value)
{
#ifdef _WIN32
    _putenv_s("METALLIC_SHADER_WARMUP_JOBS", value ? value : "");
#else
    if (value) {
        setenv("METALLIC_SHADER_WARMUP_JOBS", value, 1);
    } else {
        unsetenv("METALLIC_SHADER_WARMUP_JOBS");
    }
#endif
}

class ShaderWarmupStartupTest : public testing::Test {
protected:
    void SetUp() override
    {
        if (const char* value = std::getenv("METALLIC_SHADER_WARMUP_JOBS")) {
            oldWorkerEnvironment_ = value;
        }
        setWorkerEnvironment(nullptr);
        compileCalls = 0;
        batchCalls = 0;
        batchJobs = 0;
        failCompilation = false;
        throwCompilation = false;
        wrongDebugMode = false;
        adapterQueries = 0;
        firstNativeStrideCalls = 0;
        wrongNativeStride = false;
        failAdapterQuery = false;
        noAdapters = false;
        nativeCompilation = false;
        nativeEntry.clear();
        debugMode = metallic::render::SlangShaderDebugMode::CaptureSymbols;
        oldLogLevel_ = spdlog::get_level();
        oldFlags_ = std::cout.flags();
        oldPrecision_ = std::cout.precision();
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
    }

    void TearDown() override
    {
        setWorkerEnvironment(oldWorkerEnvironment_ ? oldWorkerEnvironment_->c_str() : nullptr);
        const auto output = testing::internal::GetCapturedStdout();
        const auto errors = testing::internal::GetCapturedStderr();
        if (HasFailure()) {
            std::cout << output;
            std::cerr << errors;
        }
        spdlog::set_level(oldLogLevel_);
        std::cout.flags(oldFlags_);
        std::cout.precision(oldPrecision_);
    }

private:
    std::optional<std::string> oldWorkerEnvironment_;
    spdlog::level::level_enum oldLogLevel_;
    std::ios::fmtflags oldFlags_;
    std::streamsize oldPrecision_;
};

TEST_F(ShaderWarmupStartupTest, StartupHonorsWorkerEnvironment)
{
    setWorkerEnvironment("12");
    EXPECT_EQ(metallic::render::warmupShadersForStartup(false), 0);
    const auto output = testing::internal::GetCapturedStdout();
    testing::internal::CaptureStdout();
    EXPECT_NE(output.find("Shader warmup workers: 12\n"), std::string::npos);
    EXPECT_EQ(compileCalls.load(), metallic::tools::shaderWarmupRequests().size());
    EXPECT_EQ(batchCalls.load(), 1u);
    EXPECT_EQ(batchJobs.load(), 12u);
    EXPECT_TRUE(metallic::render::startupShaderWarmupEnabled());
}

TEST_F(ShaderWarmupStartupTest, InvalidWorkerEnvironmentRejectsStartupBeforeCompilation)
{
    for (const char* value : {"0", "-1", "8x", "99999999999999999999999999999"}) {
        setWorkerEnvironment(value);
        EXPECT_EQ(metallic::render::warmupShadersForStartup(false), 2) << value;
    }
    EXPECT_EQ(compileCalls.load(), 0u);
    EXPECT_EQ(adapterQueries.load(), 0u);
    EXPECT_EQ(batchCalls.load(), 0u);
    EXPECT_FALSE(metallic::render::startupShaderWarmupEnabled());
}

TEST_F(ShaderWarmupStartupTest, ExplicitJobsOverridesInvalidWorkerEnvironment)
{
    setWorkerEnvironment("invalid");
    char program[] = "MetallicShaderCompiler";
    char jobs[] = "--jobs";
    char value[] = "1";
    char* arguments[] = {program, jobs, value};
    EXPECT_EQ(metallic::render::runShaderWarmup(3, arguments), 0);
    const auto output = testing::internal::GetCapturedStdout();
    testing::internal::CaptureStdout();
    EXPECT_NE(output.find("Shader warmup workers: 1\n"), std::string::npos);
}

TEST_F(ShaderWarmupStartupTest, InspectionAndSkipIgnoreInvalidWorkerEnvironment)
{
    setWorkerEnvironment("invalid");
    char program[] = "MetallicShaderCompiler";
    char list[] = "--list";
    char help[] = "--help";
    char* listArguments[] = {program, list};
    char* helpArguments[] = {program, help};
    EXPECT_EQ(metallic::render::runShaderWarmup(2, listArguments), 0);
    EXPECT_EQ(metallic::render::runShaderWarmup(2, helpArguments), 0);
    EXPECT_EQ(metallic::render::warmupShadersForStartup(true), 0);
    EXPECT_EQ(compileCalls.load(), 0u);
}

TEST_F(ShaderWarmupStartupTest, DefaultsToCompleteCatalogAndPreservesApplicationState)
{
    const auto requests = metallic::tools::shaderWarmupRequests();
    const bool hasNativeRequests = std::ranges::any_of(requests, [](const auto& request) {
        return request.descriptorHeapMode == metallic::render::SlangDescriptorHeapMode::Native;
    });
    metallic::render::ShaderWarmupLaunchOptions options;
    EXPECT_FALSE(options.skip);
    spdlog::set_level(spdlog::level::debug);
    std::cout << std::scientific;
    std::cout.precision(7);
    const auto flags = std::cout.flags();
    ASSERT_EQ(metallic::render::warmupShadersForStartup(options.skip), 0);
    EXPECT_EQ(compileCalls.load(), requests.size());
    EXPECT_FALSE(wrongDebugMode.load());
    EXPECT_EQ(adapterQueries.load(), hasNativeRequests ? 1u : 0u);
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
    EXPECT_FALSE(metallic::render::startupShaderWarmupEnabled());
    EXPECT_EQ(compileCalls.load(), 0u);
    EXPECT_EQ(adapterQueries.load(), 0u);
}

TEST_F(ShaderWarmupStartupTest, CompilationFailurePreventsStartup)
{
    failCompilation = true;
    EXPECT_NE(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_EQ(compileCalls.load(), metallic::tools::shaderWarmupRequests().size());
    EXPECT_FALSE(metallic::render::startupShaderWarmupEnabled());
}

TEST_F(ShaderWarmupStartupTest, SkipAndFailureClearPreviouslyEnabledDevicePhase)
{
    ASSERT_EQ(metallic::render::warmupShadersForStartup(false), 0);
    ASSERT_TRUE(metallic::render::startupShaderWarmupEnabled());
    EXPECT_EQ(metallic::render::warmupShadersForStartup(true), 0);
    EXPECT_FALSE(metallic::render::startupShaderWarmupEnabled());
    ASSERT_EQ(metallic::render::warmupShadersForStartup(false), 0);
    failCompilation = true;
    EXPECT_NE(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_FALSE(metallic::render::startupShaderWarmupEnabled());
}

TEST_F(ShaderWarmupStartupTest, StandaloneCompilerDoesNotEnableDevicePhase)
{
    ASSERT_EQ(metallic::render::warmupShadersForStartup(true), 0);
    char program[] = "MetallicShaderCompiler";
    char* arguments[] = {program};
    EXPECT_EQ(metallic::render::runShaderWarmup(1, arguments), 0);
    EXPECT_FALSE(metallic::render::startupShaderWarmupEnabled());
}

TEST_F(ShaderWarmupStartupTest, WorkerExceptionPreventsStartupAndRestoresLogging)
{
    throwCompilation = true;
    spdlog::set_level(spdlog::level::info);
    EXPECT_NE(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_EQ(compileCalls.load(), metallic::tools::shaderWarmupRequests().size());
    EXPECT_EQ(spdlog::get_level(), spdlog::level::info);
}

TEST_F(ShaderWarmupStartupTest, NativeStridesApplyOnlyToNativeRequestsWithoutExpandingCatalog)
{
    const auto requests = metallic::tools::shaderWarmupRequests();
    ASSERT_FALSE(requests.empty());
    nativeEntry = requests.front().entry;
    const size_t nativeCount = std::ranges::count_if(requests, [](const auto& request) {
        return request.descriptorHeapMode == metallic::render::SlangDescriptorHeapMode::Native ||
            request.entry == nativeEntry;
    });
    ASSERT_GT(nativeCount, 0u);
    ASSERT_LT(nativeCount, requests.size());
    ASSERT_EQ(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_EQ(adapterQueries.load(), 1u);
    EXPECT_EQ(compileCalls.load(), requests.size());
    EXPECT_EQ(firstNativeStrideCalls.load(), nativeCount);
    EXPECT_FALSE(wrongNativeStride.load());
}

TEST_F(ShaderWarmupStartupTest, AdapterQueryFailurePreventsNativeCompilation)
{
    nativeCompilation = true;
    failAdapterQuery = true;
    EXPECT_NE(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_EQ(adapterQueries.load(), 1u);
    EXPECT_EQ(compileCalls.load(), 0u);
}

TEST_F(ShaderWarmupStartupTest, MissingNativeStridesPreventsCompilation)
{
    nativeCompilation = true;
    noAdapters = true;
    EXPECT_NE(metallic::render::warmupShadersForStartup(false), 0);
    EXPECT_EQ(adapterQueries.load(), 1u);
    EXPECT_EQ(compileCalls.load(), 0u);
}

TEST_F(ShaderWarmupStartupTest, NativeListAndExplicitSkipDoNotQueryAdapters)
{
    nativeCompilation = true;
    failAdapterQuery = true;
    char program[] = "MetallicShaderCompiler";
    char list[] = "--list";
    char* arguments[] = {program, list};
    EXPECT_EQ(metallic::render::runShaderWarmup(2, arguments), 0);
    EXPECT_EQ(metallic::render::warmupShadersForStartup(true), 0);
    EXPECT_EQ(adapterQueries.load(), 0u);
    EXPECT_EQ(compileCalls.load(), 0u);
}

TEST_F(ShaderWarmupStartupTest, ExplicitNativeStridesAllowOfflineWarmup)
{
    nativeCompilation = true;
    failAdapterQuery = true;
    char program[] = "MetallicShaderCompiler";
    char resource[] = "--resource-heap-stride";
    char resourceValue[] = "64";
    char sampler[] = "--sampler-heap-stride";
    char samplerValue[] = "16";
    char* arguments[] = {program, resource, resourceValue, sampler, samplerValue};
    EXPECT_EQ(metallic::render::runShaderWarmup(5, arguments), 0);
    EXPECT_EQ(adapterQueries.load(), 0u);
    EXPECT_EQ(compileCalls.load(), metallic::tools::shaderWarmupRequests().size());
    EXPECT_EQ(firstNativeStrideCalls.load(), compileCalls.load());
    EXPECT_FALSE(wrongNativeStride.load());
}

TEST_F(ShaderWarmupStartupTest, IncompleteExplicitStridesAreRejectedBeforeCompilation)
{
    char program[] = "MetallicShaderCompiler";
    char resource[] = "--resource-heap-stride";
    char resourceValue[] = "64";
    char* arguments[] = {program, resource, resourceValue};
    EXPECT_EQ(metallic::render::runShaderWarmup(3, arguments), 2);
    EXPECT_EQ(adapterQueries.load(), 0u);
    EXPECT_EQ(compileCalls.load(), 0u);
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

SlangShaderOptimizationLevel slangShaderOptimizationLevel(const SlangShaderDesc& desc) noexcept
{
    if (desc.optimizationLevel != SlangShaderOptimizationLevel::Default) {
        return desc.optimizationLevel;
    }
    return debugMode == SlangShaderDebugMode::ShaderDebug
        ? SlangShaderOptimizationLevel::O0 : SlangShaderOptimizationLevel::O1;
}

const char* slangShaderOptimizationLevelName(SlangShaderOptimizationLevel level) noexcept
{
    switch (level) {
    case SlangShaderOptimizationLevel::O0: return "O0";
    case SlangShaderOptimizationLevel::O1: return "O1";
    case SlangShaderOptimizationLevel::O2: return "O2";
    case SlangShaderOptimizationLevel::O3: return "O3";
    default: return "default";
    }
}

SlangShaderPostprocessMode slangShaderPostprocessMode(const SlangShaderDesc& desc) noexcept
{
    return desc.postprocessMode == SlangShaderPostprocessMode::Default
        ? SlangShaderPostprocessMode::None : desc.postprocessMode;
}

const char* slangShaderPostprocessModeName(SlangShaderPostprocessMode mode) noexcept
{
    switch (mode) {
    case SlangShaderPostprocessMode::None: return "none";
    case SlangShaderPostprocessMode::DeadBranches: return "dead-branches";
    case SlangShaderPostprocessMode::SlangDeadBranches: return "slang-dead-branches";
    default: return "default";
    }
}

const char* slangShaderPostprocessToolVersion(SlangShaderPostprocessMode) noexcept
{
    return "startup-test-validator";
}

bool slangUsesNativeDescriptorHeap(const SlangShaderDesc& desc) noexcept
{
    return nativeCompilation || desc.descriptorHeapMode == SlangDescriptorHeapMode::Native ||
        (!nativeEntry.empty() && desc.entryPointName != nullptr && nativeEntry == desc.entryPointName);
}

bool queryVulkanDescriptorHeapShaderStrides(DescriptorHeapShaderStrides& strides, std::string& diagnostics)
{
    ++adapterQueries;
    if (failAdapterQuery) {
        diagnostics = "injected adapter query failure";
        return false;
    }
    strides = noAdapters ? DescriptorHeapShaderStrides{} : DescriptorHeapShaderStrides{64, 16};
    return true;
}

Result<ShaderCompileResult> compileSlangShaderToSpirv(
    const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& options,
    std::string& diagnostics)
{
    ++compileCalls;
    if (debugMode != SlangShaderDebugMode::CaptureSymbols) {
        wrongDebugMode = true;
    }
    if (slangUsesNativeDescriptorHeap(desc)) {
        if (desc.descriptorHeapStrides == DescriptorHeapShaderStrides{64, 16}) {
            ++firstNativeStrideCalls;
        } else {
            wrongNativeStride = true;
        }
    } else if (desc.descriptorHeapStrides != DescriptorHeapShaderStrides{}) {
        wrongNativeStride = true;
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
    ShaderCompileResult result;
    // This fixture tests startup dispatch/registration, not shader behavior.
    result.spirv = {0x07230203, 0x00010600, 0, 1, 0};
#ifdef _WIN32
    result.metrics.cacheHit = true;
#else
    result.metrics.compilerExecuted = true;
#endif
    return result;
}

Result<SlangWarmupBatchResult> compileSlangShaderWarmupBatch(
    std::span<const SlangShaderDesc> requests, const SlangWarmupBatchOptions& options, std::string& diagnostics)
{
    ++batchCalls;
    batchJobs = options.jobs;
    SlangWarmupBatchResult batch;
    bool failed = false;
    for (const auto& request : requests) {
        try {
            auto shader = compileSlangShaderToSpirv(request, options.cache, diagnostics);
            if (!shader) {
                failed = true;
                continue;
            }
            batch.existingCacheHits += shader->metrics.cacheHit ? 1 : 0;
            batch.shaders.push_back(std::move(*shader));
        } catch (const std::exception& error) {
            diagnostics = error.what();
            failed = true;
        }
    }
    if (failed || batch.shaders.size() != requests.size()) {
        return makeError(Error::Failure);
    }
    batch.sessionGroups = 1;
    batch.completedAllRequests = true;
#ifdef _WIN32
    batch.completeSourceEpochValidated = true;
#endif
    return batch;
}

} // namespace metallic::render

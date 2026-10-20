#include "Runtime/Render/Core/SlangCompiler.h"
#include <slang.h>

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

using namespace metallic::render;

namespace {

class CompilerProducerSealTest : public testing::Test {
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

#if defined(_WIN32)
void expectMutationOpenDenied(const std::filesystem::path& path)
{
    SCOPED_TRACE(path.string());
    // Request every sharing mode ourselves: rejection must come from the
    // producer's retained read-only sharing contract. Never write or delete.
    for (const DWORD access : {GENERIC_WRITE, DELETE}) {
        SCOPED_TRACE(access);
        const HANDLE handle = CreateFileW(path.c_str(), access,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        const DWORD error = GetLastError();
        if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); }
        EXPECT_EQ(handle, INVALID_HANDLE_VALUE);
        EXPECT_EQ(error, ERROR_SHARING_VIOLATION);
    }
}
#endif

TEST_F(CompilerProducerSealTest, CacheHitRetainsHostAndSDKWriteDeleteExclusion)
{
#if defined(_WIN32)
    const std::string directory = (std::filesystem::path(TEST_BINARY_DIR) / "producer-seal-cache" /
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())).string();
    const SlangShaderDesc desc{.moduleName = "OptimizationProbe", .entryPointName = "main",
        .searchPath = PROJECT_SOURCE_DIR "/tests/editor/shaders",
        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped,
        .optimizationLevel = SlangShaderOptimizationLevel::O1,
        .postprocessMode = SlangShaderPostprocessMode::None};
    bool hit = false;
    const SlangShaderCacheOptions cache{.cacheDirectory = directory.c_str(), .outCacheHit = &hit};
    std::string diagnostics;
    const auto compiled = compileSlangShaderToSpirv(desc, cache, diagnostics);
    ASSERT_TRUE(compiled) << diagnostics;
    ASSERT_FALSE(hit);
    ASSERT_TRUE(compiled->metrics.compilerExecuted);

    // Identify the actual compiler host rather than assuming that it is the EXE.
    HMODULE host{};
    ASSERT_TRUE(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&slangShaderDebugMode), &host));
    HMODULE slang{};
    ASSERT_TRUE(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&slang_createGlobalSession2), &slang));
    std::array<wchar_t, 32768> hostPath{}, slangPath{};
    const DWORD hostLength = GetModuleFileNameW(host, hostPath.data(), static_cast<DWORD>(hostPath.size()));
    const DWORD slangLength = GetModuleFileNameW(slang, slangPath.data(), static_cast<DWORD>(slangPath.size()));
    ASSERT_GT(hostLength, 0u);
    ASSERT_LT(hostLength, hostPath.size());
    ASSERT_GT(slangLength, 0u);
    ASSERT_LT(slangLength, slangPath.size());
    const std::filesystem::path sdkDirectory = std::filesystem::path(slangPath.data()).parent_path();
    const std::array<std::filesystem::path, 4> producers{
        std::filesystem::path(hostPath.data()), sdkDirectory / "slang.dll",
        sdkDirectory / "slang-compiler.dll", sdkDirectory / "slang-glslang.dll"};
    for (const auto& producer : producers) { expectMutationOpenDenied(producer); }

    const auto cached = compileSlangShaderToSpirv(desc, cache, diagnostics);
    ASSERT_TRUE(cached) << diagnostics;
    EXPECT_TRUE(hit);
    EXPECT_TRUE(cached->metrics.cacheHit);
    EXPECT_FALSE(cached->metrics.compilerExecuted);
    EXPECT_EQ(cached->spirv, compiled->spirv);
    for (const auto& producer : producers) { expectMutationOpenDenied(producer); }
#else
    GTEST_SKIP() << "Compiler producer pinning requires Windows sharing semantics";
#endif
}

} // namespace

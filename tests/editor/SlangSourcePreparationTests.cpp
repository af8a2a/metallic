#include "Runtime/Render/Core/SlangCompiler.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <tuple>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace {
using namespace metallic::render;
namespace fs = std::filesystem;

void writeText(const fs::path& path, std::string_view text)
{
    fs::create_directories(path.parent_path());
#ifdef _WIN32
    struct File {
        HANDLE handle;
        ~File() { if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); } }
    } file{CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
        nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr)};
    if (file.handle == INVALID_HANDLE_VALUE) { throw std::runtime_error("Cannot open source preparation fixture"); }
    DWORD written = 0;
    if (!WriteFile(file.handle, text.data(), static_cast<DWORD>(text.size()), &written, nullptr) ||
        written != text.size() || !FlushFileBuffers(file.handle)) {
        throw std::runtime_error("Source preparation fixture write/flush failed");
    }
    if (!CloseHandle(file.handle)) { throw std::runtime_error("Source preparation fixture close failed"); }
    file.handle = INVALID_HANDLE_VALUE;
#else
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    stream.close();
    if (!stream) { throw std::runtime_error("Source preparation fixture write failed"); }
#endif
}

using CacheEntry = std::tuple<fs::path, uintmax_t, fs::file_time_type>;
std::vector<CacheEntry> cacheEntries(const fs::path& directory)
{
    std::vector<CacheEntry> result;
    if (!fs::exists(directory)) { return result; }
    for (const auto& entry : fs::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            result.emplace_back(entry.path(), entry.file_size(), entry.last_write_time());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

class SlangSourcePreparationTest : public testing::Test
{
protected:
    void SetUp() override
    {
#ifndef _WIN32
        GTEST_SKIP() << "Live source epoch qualification requires Windows directory notifications";
#else
        static std::atomic<uint64_t> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = fs::absolute(TEST_BINARY_DIR) / "SlangSourcePreparationFixtures" /
            (std::to_string(stamp) + "-" + std::to_string(sequence++));
        source_ = directory_ / "source";
        cache_ = directory_ / "cache";
        fs::create_directories(source_);
        fs::create_directories(cache_);
        sourcePath_ = source_.string();
        cachePath_ = cache_.string();
        oldMode_ = slangShaderDebugMode();
        setSlangShaderDebugMode(SlangShaderDebugMode::Disabled);
#endif
    }

    void TearDown() override
    {
#ifdef _WIN32
        resetSlangShaderHotReloadTracking();
        setSlangShaderDebugMode(oldMode_);
        std::error_code error;
        fs::remove_all(directory_, error);
#endif
    }

    SlangShaderDesc request(const char* moduleName = "PreparationProbe") const
    {
        return {.moduleName = moduleName, .entryPointName = "probe", .searchPath = sourcePath_.c_str(),
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped,
            .optimizationLevel = SlangShaderOptimizationLevel::O1,
            .postprocessMode = SlangShaderPostprocessMode::None};
    }

    SlangShaderCacheOptions cacheOptions(bool* hit = nullptr) const
    {
        return {.cacheDirectory = cachePath_.c_str(), .outCacheHit = hit};
    }

    Result<SlangWarmupBatchResult> primeCache(const SlangShaderDesc& desc, std::string& log) const
    {
        // The explicit batch writes an artifact without populating the ordinary
        // source epoch pool, so the next prepare/acquire must establish an epoch.
        std::string setupRetries;
        for (size_t attempt = 0; attempt < 8; ++attempt) {
            auto result = compileSlangShaderWarmupBatch(std::span(&desc, 1),
                {.cache = cacheOptions(), .jobs = 1}, log);
            if (result && result->completeSourceEpochValidated) {
                log = setupRetries + log;
                return result;
            }
            constexpr std::string_view prefix = "Source snapshot reuse unavailable: "
                "Unstable snapshot capture: source epoch notification advanced: ";
            bool setupNotification = false;
            if (log.starts_with(prefix)) {
                const auto end = log.find('\n', prefix.size());
                const auto root = fs::u8path(log.substr(prefix.size(), end - prefix.size())).lexically_normal();
                setupNotification = (root == source_.lexically_normal() || root == (directory_ / "extra").lexically_normal()) &&
                    (end == std::string::npos || log.find_first_not_of("\r\n", end) == std::string::npos);
            }
            // Even after flushing the writer handle, its final close can
            // deliver setup notifications after capture starts. Retry only
            // that exact fixture-root failure, before any behavior under test.
            if (attempt < 7 && result && setupNotification) {
                setupRetries += "[FixtureSetup] Recapturing after delayed setup notification:\n" + log;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            log = setupRetries + log;
            if (!result) { return result; }
            return makeError(Error::Failure);
        }
        return makeError(Error::Failure);
    }

    fs::path directory_, source_, cache_;
    std::string sourcePath_, cachePath_;
    SlangShaderDebugMode oldMode_ = SlangShaderDebugMode::Disabled;
};

TEST_F(SlangSourcePreparationTest, SearchOrderKeepsRequestIdentityAndOutputWhileSharingSourceImage)
{
    const auto first = source_ / "first", second = source_ / "second";
    writeText(first / "Selected.h", "#define PREPARATION_VALUE 17u\n");
    writeText(second / "Selected.h", "#define PREPARATION_VALUE 29u\n");
    writeText(source_ / "PreparationProbe.slang", "module PreparationProbe;\n#include \"Selected.h\"\n"
        "RWStructuredBuffer<uint> output;\n[shader(\"compute\")] [numthreads(1,1,1)] "
        "void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = PREPARATION_VALUE; }\n");
    const auto firstPath = first.string(), secondPath = second.string();
    const std::array firstOrder{firstPath.c_str(), secondPath.c_str()};
    const std::array secondOrder{secondPath.c_str(), firstPath.c_str()};
    auto a = request(), b = request();
    a.additionalSearchPaths = firstOrder;
    b.additionalSearchPaths = secondOrder;
    std::string log;
    const auto identityA = slangShaderRequestIdentity(a, log), identityB = slangShaderRequestIdentity(b, log);
    ASSERT_TRUE(identityA) << log;
    ASSERT_TRUE(identityB) << log;
    EXPECT_NE(*identityA, *identityB);

    bool hit = false;
    auto compiledA = compileSlangShaderToSpirv(a, cacheOptions(&hit), log);
    ASSERT_TRUE(compiledA) << log;
    EXPECT_FALSE(hit);
    EXPECT_FALSE(compiledA->metrics.sourceEpochReused);
    EXPECT_GT(compiledA->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    auto compiledB = compileSlangShaderToSpirv(b, cacheOptions(&hit), log);
    ASSERT_TRUE(compiledB) << log;
    EXPECT_FALSE(hit);
    EXPECT_TRUE(compiledB->metrics.sourceEpochReused);
    EXPECT_EQ(compiledB->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_NE(compiledA->spirv, compiledB->spirv);

    auto cachedA = compileSlangShaderToSpirv(a, cacheOptions(&hit), log);
    ASSERT_TRUE(cachedA) << log;
    EXPECT_TRUE(hit);
    EXPECT_EQ(cachedA->spirv, compiledA->spirv);
    auto cachedB = compileSlangShaderToSpirv(b, cacheOptions(&hit), log);
    ASSERT_TRUE(cachedB) << log;
    EXPECT_TRUE(hit);
    EXPECT_EQ(cachedB->spirv, compiledB->spirv);
}

TEST_F(SlangSourcePreparationTest, PreparationDoesNotWriteCacheAndSharesOnlyEquivalentCaptureRoots)
{
    writeText(source_ / "PreparationProbe.slang", "module PreparationProbe; RWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = 17u; }\n");
    const auto covered = source_ / "covered", extra = directory_ / "extra";
    writeText(covered / "Unused.h", "#define UNUSED 1\n");
    writeText(extra / "Unused.h", "#define UNUSED 2\n");
    const auto coveredPath = covered.string(), extraPath = extra.string();
    const char* coveredPointer = coveredPath.c_str();
    const char* extraPointer = extraPath.c_str();
    auto coveredRequest = request(), expandedRequest = request();
    coveredRequest.additionalSearchPaths = std::span(&coveredPointer, 1);
    expandedRequest.additionalSearchPaths = std::span(&extraPointer, 1);
    std::string log;
    auto primed = primeCache(expandedRequest, log);
    ASSERT_TRUE(primed) << log;
    ASSERT_TRUE(primed->completeSourceEpochValidated) << log;
    const auto fixtureCacheBefore = cacheEntries(cache_);
    ASSERT_FALSE(fixtureCacheBefore.empty()) << log;
    const auto defaultCache = fs::path(PROJECT_SOURCE_DIR) / ".cache/shaders/spirv";
    const auto defaultCacheBefore = cacheEntries(defaultCache);
    const std::array roots{source_};
    auto prepared = prepareSlangSourceEpoch(roots, log);
    ASSERT_TRUE(prepared) << log;
    EXPECT_FALSE(prepared->sourceEpochReused);
    EXPECT_GT(prepared->sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_EQ(cacheEntries(cache_), fixtureCacheBefore);
    EXPECT_EQ(cacheEntries(defaultCache), defaultCacheBefore);

    bool hit = false;
    auto coveredShader = compileSlangShaderToSpirv(coveredRequest, cacheOptions(&hit), log);
    ASSERT_TRUE(coveredShader) << log;
    EXPECT_TRUE(coveredShader->metrics.sourceEpochReused);
    EXPECT_EQ(coveredShader->metrics.sourceEpochCaptureTimeNanoseconds, 0u);

    auto expandedShader = compileSlangShaderToSpirv(expandedRequest, cacheOptions(&hit), log);
    ASSERT_TRUE(expandedShader) << log;
    EXPECT_TRUE(hit) << log;
    EXPECT_FALSE(expandedShader->metrics.sourceEpochReused);
    EXPECT_GT(expandedShader->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_EQ(expandedShader->spirv, coveredShader->spirv);
}

TEST_F(SlangSourcePreparationTest, ChangedDependencyInvalidatesPreparedImageEvenWithRestoredTimestamp)
{
    const auto header = source_ / "Selected.h";
    writeText(header, "#define PREPARATION_VALUE 17u\n");
    writeText(source_ / "PreparationProbe.slang", "module PreparationProbe;\n#include \"Selected.h\"\n"
        "RWStructuredBuffer<uint> output;\n[shader(\"compute\")] [numthreads(1,1,1)] "
        "void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = PREPARATION_VALUE; }\n");
    const auto desc = request();
    std::string log;
    auto primed = primeCache(desc, log);
    ASSERT_TRUE(primed) << log;
    const std::array roots{source_};
    auto prepared = prepareSlangSourceEpoch(roots, log);
    ASSERT_TRUE(prepared) << log;
    const auto timestamp = fs::last_write_time(header);
    writeText(header, "#define PREPARATION_VALUE 31u\n");
    fs::last_write_time(header, timestamp);

    bool hit = false;
    auto changed = compileSlangShaderToSpirv(desc, cacheOptions(&hit), log);
    ASSERT_TRUE(changed) << log;
    EXPECT_FALSE(hit);
    EXPECT_FALSE(changed->metrics.sourceEpochReused);
    EXPECT_GT(changed->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_NE(changed->spirv, primed->shaders[0].spirv);
}

TEST_F(SlangSourcePreparationTest, MissingSearchPathOutsideCapturedNamespaceRejectsReuse)
{
    const auto uncaptured = directory_ / "uncaptured";
    fs::create_directories(uncaptured);
    const std::array roots{source_};
    std::string log;
    auto prepared = prepareSlangSourceEpoch(roots, log);
    ASSERT_TRUE(prepared) << log;

    // Planning omits this missing directory, leaving an equivalent image key.
    // The retained image must still prove each new request's search namespace.
    const std::array expandedRoots{source_, uncaptured / "missing"};
    auto unqualified = prepareSlangSourceEpoch(expandedRoots, log);
    EXPECT_FALSE(unqualified);
    EXPECT_NE(log.find("Effective search path has no complete captured namespace/ancestor"), std::string::npos) << log;
}

TEST_F(SlangSourcePreparationTest, NewShadowingHeaderInvalidatesPreparedNamespace)
{
    const auto first = source_ / "first", second = source_ / "second";
    fs::create_directories(first);
    writeText(second / "Selected.h", "#define PREPARATION_VALUE 17u\n");
    writeText(source_ / "PreparationProbe.slang", "module PreparationProbe;\n#include \"Selected.h\"\n"
        "RWStructuredBuffer<uint> output;\n[shader(\"compute\")] [numthreads(1,1,1)] "
        "void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = PREPARATION_VALUE; }\n");
    const auto firstPath = first.string(), secondPath = second.string();
    const std::array paths{firstPath.c_str(), secondPath.c_str()};
    auto desc = request();
    desc.additionalSearchPaths = paths;
    std::string log;
    auto primed = primeCache(desc, log);
    ASSERT_TRUE(primed) << log;
    const std::array roots{source_};
    auto prepared = prepareSlangSourceEpoch(roots, log);
    ASSERT_TRUE(prepared) << log;
    writeText(first / "Selected.h", "#define PREPARATION_VALUE 47u\n");

    bool hit = false;
    auto changed = compileSlangShaderToSpirv(desc, cacheOptions(&hit), log);
    ASSERT_TRUE(changed) << log;
    EXPECT_FALSE(hit);
    EXPECT_FALSE(changed->metrics.sourceEpochReused);
    EXPECT_GT(changed->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_NE(changed->spirv, primed->shaders[0].spirv);
}

TEST_F(SlangSourcePreparationTest, ConcurrentPreparationAndOrdinaryAcquisitionCaptureOnce)
{
    writeText(source_ / "PreparationProbe.slang", "module PreparationProbe; RWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = 17u; }\n");
    const auto desc = request();
    std::string log;
    auto primed = primeCache(desc, log);
    ASSERT_TRUE(primed) << log;
    const std::array roots{source_};
    struct Observation {
        bool succeeded = false, reused = false, cacheHit = false;
        uint64_t captureTimeNanoseconds = 0;
        std::string log;
        std::vector<uint32_t> spirv;
    };
    constexpr size_t kWorkers = 8;
    std::barrier start(static_cast<std::ptrdiff_t>(kWorkers));
    std::vector<std::future<Observation>> futures;
    for (size_t index = 0; index < kWorkers; ++index) {
        futures.push_back(std::async(std::launch::async, [&, index] {
            Observation observation;
            start.arrive_and_wait();
            if (index % 2 == 0) {
                auto prepared = prepareSlangSourceEpoch(roots, observation.log);
                observation.succeeded = bool(prepared);
                if (prepared) {
                    observation.reused = prepared->sourceEpochReused;
                    observation.captureTimeNanoseconds = prepared->sourceEpochCaptureTimeNanoseconds;
                }
            } else {
                auto shader = compileSlangShaderToSpirv(desc, cacheOptions(&observation.cacheHit), observation.log);
                observation.succeeded = bool(shader);
                if (shader) {
                    observation.reused = shader->metrics.sourceEpochReused;
                    observation.captureTimeNanoseconds = shader->metrics.sourceEpochCaptureTimeNanoseconds;
                    observation.spirv = std::move(shader->spirv);
                }
            }
            return observation;
        }));
    }
    size_t captures = 0, reuses = 0;
    for (size_t index = 0; index < futures.size(); ++index) {
        const auto observation = futures[index].get();
        ASSERT_TRUE(observation.succeeded) << observation.log;
        if (observation.captureTimeNanoseconds > 0) { ++captures; }
        if (observation.reused) { ++reuses; }
        if (index % 2 != 0) {
            EXPECT_TRUE(observation.cacheHit);
            EXPECT_EQ(observation.spirv, primed->shaders[0].spirv);
        }
    }
    EXPECT_EQ(captures, 1u);
    EXPECT_EQ(reuses, kWorkers - 1);
}

TEST_F(SlangSourcePreparationTest, PreparedImageSurvivesWorkerExitAndStillDetectsDependencyChanges)
{
    const auto header = source_ / "Selected.h";
    writeText(header, "#define PREPARATION_VALUE 17u\n");
    writeText(source_ / "PreparationProbe.slang", "module PreparationProbe;\n#include \"Selected.h\"\n"
        "RWStructuredBuffer<uint> output;\n[shader(\"compute\")] [numthreads(1,1,1)] "
        "void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = PREPARATION_VALUE; }\n");
    const auto desc = request();
    std::string log;
    auto primed = primeCache(desc, log);
    ASSERT_TRUE(primed) << log;
    const std::array roots{source_};
    Result<SlangSourcePreparationMetrics> prepared = makeError(Error::Failure);
    std::string preparationLog;
    // Use an actual thread lifetime: std::async may retain its OS worker in a
    // pool and conceal cancellation of directory I/O when the issuer exits.
    std::thread worker([&] { prepared = prepareSlangSourceEpoch(roots, preparationLog); });
    worker.join();
    ASSERT_TRUE(prepared) << preparationLog;

    bool hit = false;
    auto cached = compileSlangShaderToSpirv(desc, cacheOptions(&hit), log);
    ASSERT_TRUE(cached) << log;
    EXPECT_TRUE(hit);
    EXPECT_TRUE(cached->metrics.sourceEpochReused);
    EXPECT_EQ(cached->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_EQ(cached->spirv, primed->shaders[0].spirv);

    writeText(header, "#define PREPARATION_VALUE 31u\n");
    auto changed = compileSlangShaderToSpirv(desc, cacheOptions(&hit), log);
    ASSERT_TRUE(changed) << log;
    EXPECT_FALSE(hit);
    EXPECT_FALSE(changed->metrics.sourceEpochReused);
    EXPECT_GT(changed->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_NE(changed->spirv, cached->spirv);
}

} // namespace

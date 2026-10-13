#include "Runtime/Render/Core/ShaderRegistry.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace metallic::render;

namespace {

using ArtifactResolution = std::optional<Result<ShaderArtifactOverrideResult>>;

class CallbackArtifactOverride final : public ShaderArtifactOverride {
public:
    using Callback = std::function<ArtifactResolution(const SlangShaderDesc&, std::string&)>;

    explicit CallbackArtifactOverride(Callback callback, std::function<void()> onDestroy = {})
        : callback_(std::move(callback)), onDestroy_(std::move(onDestroy)) {}

    ~CallbackArtifactOverride() override
    {
        if (onDestroy_) { onDestroy_(); }
    }

    ArtifactResolution resolve(const SlangShaderDesc& desc, std::string& log) override
    {
        return callback_(desc, log);
    }

private:
    Callback callback_;
    std::function<void()> onDestroy_;
};

// A complete, resource-free compute module. A distinct generator word gives
// each test independent byte-content identity in the process singleton.
std::vector<uint32_t> artifactSpirv(uint32_t generator = 1)
{
    return {0x07230203, 0x00010600, generator, 5, 0,
        0x00020011, 1,                         // OpCapability Shader
        0x0003000e, 0, 1,                      // OpMemoryModel Logical GLSL450
        0x0005000f, 5, 3, 0x6e69616d, 0,      // OpEntryPoint GLCompute %3 "main"
        0x00060010, 3, 17, 1, 1, 1,           // OpExecutionMode %3 LocalSize 1 1 1
        0x00020013, 1,                         // %1 = OpTypeVoid
        0x00030021, 2, 1,                      // %2 = OpTypeFunction %1
        0x00050036, 1, 3, 0, 2,               // %3 = OpFunction %1 None %2
        0x000200f8, 4,                         // %4 = OpLabel
        0x000100fd,                            // OpReturn
        0x00010038};                           // OpFunctionEnd
}

ShaderArtifactOverrideResult artifact(uint32_t generator = 1)
{
    ShaderArtifactOverrideResult result;
    result.shader.spirv = artifactSpirv(generator);
    result.shader.diagnostics = "sealed artifact supplied";
    result.identity.moduleName = "Artifacts/PathBackend/CPU-" + std::to_string(generator);
    result.identity.sourceEntryPoint = "main";
    result.identity.entryPoint = "main";
    result.identity.descriptorHeapMode = "mapped";
    result.identity.optimizationLevel = "O1";
    result.identity.debugMode = "release";
    result.identity.spirvPostprocess = "none";
    return result;
}

uint64_t spirvHash(const std::vector<uint32_t>& spirv)
{
    uint64_t hash = 14695981039346656037ull;
    const auto* bytes = reinterpret_cast<const unsigned char*>(spirv.data());
    for (size_t index = 0; index < spirv.size() * sizeof(uint32_t); ++index) {
        hash = (hash ^ bytes[index]) * 1099511628211ull;
    }
    return hash;
}

void expectArtifactMetrics(const ShaderCompileMetrics& metrics, size_t byteSize)
{
    EXPECT_FALSE(metrics.cacheHit);
    EXPECT_FALSE(metrics.compilerExecuted);
    EXPECT_FALSE(metrics.compilerReportedTimingAvailable);
    EXPECT_FALSE(metrics.detailedCompilerPerfRequested);
    EXPECT_EQ(metrics.totalTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.cacheLookupTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.compilerSetupTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.moduleLoadTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.dependencyTrackingTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.entryPointCompositionTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.linkTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.codeGenerationTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.targetFinalizeTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.postprocessTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.rawSpirvByteSize, 0u);
    EXPECT_EQ(metrics.outputSpirvByteSize, byteSize);
    EXPECT_EQ(metrics.dependencyValidationTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.cacheWriteTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.compilerReportedTotalTimeNanoseconds, 0u);
    EXPECT_EQ(metrics.compilerReportedDownstreamTimeNanoseconds, 0u);
}

class ShaderArtifactOverrideTest : public testing::Test {
protected:
    void SetUp() override
    {
        oldDebugMode_ = slangShaderDebugMode();
        setSlangShaderDebugMode(SlangShaderDebugMode::Disabled);
        cacheDirectory_ = (std::filesystem::path(TEST_BINARY_DIR) / "shader-artifact-override-cache" /
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())).string();
    }

    void TearDown() override
    {
        setSlangShaderDebugMode(oldDebugMode_);
    }

    SlangShaderDesc probe() const
    {
        return {.moduleName = "OptimizationProbe", .entryPointName = "main",
            .searchPath = PROJECT_SOURCE_DIR "/tests/editor/shaders",
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped,
            .optimizationLevel = SlangShaderOptimizationLevel::O1,
            .postprocessMode = SlangShaderPostprocessMode::None};
    }

    SlangShaderCacheOptions cache(bool* hit = nullptr) const
    {
        return {.cacheDirectory = cacheDirectory_.c_str(), .outCacheHit = hit,
            .reportDetailedCompilerPerf = true};
    }

    void expectCacheAbsent() const
    {
        if (std::filesystem::exists(cacheDirectory_)) {
            EXPECT_TRUE(std::filesystem::is_empty(cacheDirectory_));
        }
    }

    ShaderRegistry& registry_ = ShaderRegistry::instance();
    std::string cacheDirectory_;
    std::string log_;

private:
    SlangShaderDebugMode oldDebugMode_;
};

TEST_F(ShaderArtifactOverrideTest, ArtifactUsesReturnedBytesWithoutCacheWritesOrInventedTimings)
{
    auto value = artifact(11);
    // A producer cannot smuggle compiler measurements from preparation into a
    // later artifact acquisition, nor turn it into an application cache hit.
    value.shader.metrics = {.totalTimeNanoseconds = 99, .codeGenerationTimeNanoseconds = 98,
        .postprocessTimeNanoseconds = 97, .rawSpirvByteSize = 96, .outputSpirvByteSize = 95,
        .cacheWriteTimeNanoseconds = 94, .compilerReportedTotalTimeNanoseconds = 93,
        .cacheHit = true, .compilerExecuted = true, .compilerReportedTimingAvailable = true,
        .detailedCompilerPerfRequested = true};
    value.identity.spirv = value.shader.spirv;
    value.identity.inputSpirvHash = spirvHash(value.shader.spirv);
    auto override = std::make_shared<CallbackArtifactOverride>([value](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{value};
    });
    auto installed = registry_.installArtifactOverrideForTesting(override);
    ASSERT_TRUE(installed);
    bool hit = true;
    const auto result = registry_.getShader(probe(), cache(&hit), log_);
    ASSERT_TRUE(result) << log_;
    EXPECT_EQ(result->spirv, value.shader.spirv);
    EXPECT_EQ(result->diagnostics, value.shader.diagnostics);
    EXPECT_FALSE(hit);
    expectArtifactMetrics(result->metrics, value.shader.spirv.size() * sizeof(uint32_t));
    expectCacheAbsent();
}

TEST_F(ShaderArtifactOverrideTest, ExactNonTargetFallsThroughAndScopeRemovalRestoresCompilerCache)
{
    std::atomic<uint32_t> visits = 0;
    auto override = std::make_shared<CallbackArtifactOverride>([&](const SlangShaderDesc& desc, std::string&) {
        ++visits;
        if (std::string_view(desc.moduleName) != "ArtifactOnlyTarget" ||
            std::string_view(desc.entryPointName) != "main") {
            return ArtifactResolution{};
        }
        return ArtifactResolution{artifact(12)};
    });
    std::vector<uint32_t> ordinaryBytes;
    {
        auto installed = registry_.installArtifactOverrideForTesting(override);
        ASSERT_TRUE(installed);
        auto target = probe();
        target.moduleName = "ArtifactOnlyTarget"; // No source exists: success requires the override.
        auto loaded = registry_.getShader(target, cache(), log_);
        ASSERT_TRUE(loaded) << log_;
        EXPECT_EQ(loaded->spirv, artifactSpirv(12));
        expectCacheAbsent();

        bool hit = true;
        const auto ordinary = registry_.getShader(probe(), cache(&hit), log_);
        ASSERT_TRUE(ordinary) << log_;
        EXPECT_FALSE(hit);
        EXPECT_TRUE(ordinary->metrics.compilerExecuted);
        EXPECT_FALSE(ordinary->metrics.cacheHit);
        ordinaryBytes = ordinary->spirv;
        EXPECT_NE(ordinaryBytes, loaded->spirv);
    }
    bool hit = false;
    const auto reused = registry_.getShader(probe(), cache(&hit), log_);
    ASSERT_TRUE(reused) << log_;
    EXPECT_TRUE(hit);
    EXPECT_TRUE(reused->metrics.cacheHit);
    EXPECT_FALSE(reused->metrics.compilerExecuted);
    EXPECT_EQ(reused->spirv, ordinaryBytes);
    EXPECT_EQ(visits.load(), 2u); // Destroyed scope was not called for the normal cache hit.
}

TEST_F(ShaderArtifactOverrideTest, EngagedErrorsAndExceptionsNeverFallBackToValidSource)
{
    for (const bool throws : {false, true}) {
        SCOPED_TRACE(throws);
        auto override = std::make_shared<CallbackArtifactOverride>([throws](const SlangShaderDesc&, std::string& log)
            -> ArtifactResolution {
            log = "artifact identity mismatch";
            if (throws) { throw std::runtime_error("artifact read failed"); }
            return Result<ShaderArtifactOverrideResult>{makeError(Error::InvalidArgument)};
        });
        auto installed = registry_.installArtifactOverrideForTesting(override);
        ASSERT_TRUE(installed);
        bool hit = true;
        const auto result = registry_.getShader(probe(), cache(&hit), log_);
        ASSERT_FALSE(result); // OptimizationProbe would otherwise compile successfully.
        EXPECT_EQ(result.error(), throws ? Error::Failure : Error::InvalidArgument);
        EXPECT_FALSE(hit);
        EXPECT_FALSE(log_.empty());
        expectCacheAbsent();
    }
}

TEST_F(ShaderArtifactOverrideTest, MalformedSpirvAndContradictoryIdentityRejectBeforeFallback)
{
    using Mutation = std::pair<const char*, std::function<void(ShaderArtifactOverrideResult&)>>;
    const std::vector<Mutation> mutations = {
        {"missing header", [](auto& value) { value.shader.spirv.resize(4); }},
        {"magic", [](auto& value) { value.shader.spirv[0] = 0; }},
        {"future version", [](auto& value) { value.shader.spirv[1] = 0x00010700; }},
        {"zero bound", [](auto& value) { value.shader.spirv[3] = 0; }},
        {"reserved schema", [](auto& value) { value.shader.spirv[4] = 1; }},
        {"zero instruction length", [](auto& value) { value.shader.spirv[5] = 17; }},
        {"truncated instruction", [](auto& value) { value.shader.spirv[5] = 0xffff0011; }},
        {"native entry missing", [](auto& value) { value.identity.entryPoint = "notInModule"; }},
        {"source tag missing", [](auto& value) { value.identity.moduleName.clear(); }},
        {"source tag contains nul", [](auto& value) { value.identity.moduleName = std::string("Artifacts/a\0b", 12); }},
        {"source entry mismatch", [](auto& value) { value.identity.sourceEntryPoint = "otherMain"; }},
        {"descriptor ABI", [](auto& value) { value.identity.descriptorHeapMode = "native"; }},
        {"optimization", [](auto& value) { value.identity.optimizationLevel = "default"; }},
        {"debug", [](auto& value) { value.identity.debugMode = "unknown"; }},
        {"postprocess", [](auto& value) { value.identity.spirvPostprocess = "unknown"; }},
        {"cleanup version absent", [](auto& value) { value.identity.spirvPostprocess = "dead-branches"; }},
        {"cleanup version unavailable", [](auto& value) {
            value.identity.spirvPostprocess = "slang-dead-branches";
            value.identity.spirvPostprocessToolVersion = "unavailable";
        }},
        {"byte hash mismatch", [](auto& value) { value.identity.inputSpirvHash = 1; }},
        {"identity bytecode mismatch", [](auto& value) { value.identity.spirv = artifactSpirv(900); }}
    };
    uint32_t generator = 30;
    for (const auto& [name, mutate] : mutations) {
        SCOPED_TRACE(name);
        auto value = artifact(generator++);
        mutate(value);
        auto override = std::make_shared<CallbackArtifactOverride>([value](const SlangShaderDesc&, std::string&) {
            return ArtifactResolution{value};
        });
        auto installed = registry_.installArtifactOverrideForTesting(override);
        ASSERT_TRUE(installed);
        bool hit = true;
        const auto result = registry_.getShader(probe(), cache(&hit), log_);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), Error::InvalidArgument);
        EXPECT_FALSE(hit);
        EXPECT_FALSE(log_.empty());
        expectCacheAbsent();
    }
}

TEST_F(ShaderArtifactOverrideTest, NullAndNestedInstallRejectWithoutReplacingOwner)
{
    const auto empty = registry_.installArtifactOverrideForTesting(nullptr);
    ASSERT_FALSE(empty);
    EXPECT_EQ(empty.error(), Error::InvalidArgument);
    auto first = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(60)};
    });
    auto second = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(61)};
    });
    auto installed = registry_.installArtifactOverrideForTesting(first);
    ASSERT_TRUE(installed);
    const auto rejected = registry_.installArtifactOverrideForTesting(second);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error(), Error::InvalidArgument);
    const auto result = registry_.getShader(probe(), cache(), log_);
    ASSERT_TRUE(result) << log_;
    EXPECT_EQ(result->spirv, artifactSpirv(60));
}

TEST_F(ShaderArtifactOverrideTest, MoveTransfersRemovalAndStaleScopeCannotRemoveLaterInstall)
{
    static_assert(!std::is_copy_constructible_v<ShaderArtifactOverrideScope>);
    static_assert(!std::is_copy_assignable_v<ShaderArtifactOverrideScope>);
    auto override = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(62)};
    });
    auto installed = registry_.installArtifactOverrideForTesting(override);
    ASSERT_TRUE(installed);
    ShaderArtifactOverrideScope moved(std::move(*installed));
    EXPECT_FALSE(bool(*installed));
    EXPECT_TRUE(bool(moved));
    installed->reset();
    auto loaded = registry_.getShader(probe(), cache(), log_);
    ASSERT_TRUE(loaded) << log_;
    EXPECT_EQ(loaded->spirv, artifactSpirv(62));

    ShaderArtifactOverrideScope assigned;
    assigned = std::move(moved);
    EXPECT_FALSE(bool(moved));
    EXPECT_TRUE(bool(assigned));
    moved.reset();
    EXPECT_FALSE(registry_.installArtifactOverrideForTesting(override));
    assigned.reset();
    EXPECT_FALSE(bool(assigned));
    auto replacement = registry_.installArtifactOverrideForTesting(override);
    ASSERT_TRUE(replacement);
    assigned.reset();
    installed->reset();
    EXPECT_FALSE(registry_.installArtifactOverrideForTesting(override));
    loaded = registry_.getShader(probe(), cache(), log_);
    ASSERT_TRUE(loaded) << log_;
    EXPECT_EQ(loaded->spirv, artifactSpirv(62));
}

TEST_F(ShaderArtifactOverrideTest, CallbackCanReenterRegistrationWithoutHoldingRegistryLock)
{
    std::atomic<bool> reentered = false;
    auto nestedOverride = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(66)};
    });
    auto override = std::make_shared<CallbackArtifactOverride>([&](const SlangShaderDesc&, std::string&)
        -> ArtifactResolution {
        auto nested = registry_.installArtifactOverrideForTesting(nestedOverride);
        reentered = !nested && nested.error() == Error::InvalidArgument;
        return artifact(63);
    });
    auto installed = registry_.installArtifactOverrideForTesting(override);
    ASSERT_TRUE(installed);
    const auto result = registry_.getShader(probe(), cache(), log_);
    ASSERT_TRUE(result) << log_;
    EXPECT_TRUE(reentered.load());
}

TEST_F(ShaderArtifactOverrideTest, LastOwnerDestructionCanInstallReplacementOutsideRegistryLock)
{
    auto replacement = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(69)};
    });
    Result<ShaderArtifactOverrideScope> replacementScope = makeError(Error::Failure);
    auto retiring = std::make_shared<CallbackArtifactOverride>(
        [](const SlangShaderDesc&, std::string&) { return ArtifactResolution{artifact(70)}; },
        [&] { replacementScope = registry_.installArtifactOverrideForTesting(replacement); });
    auto installed = registry_.installArtifactOverrideForTesting(retiring);
    ASSERT_TRUE(installed);
    std::weak_ptr<ShaderArtifactOverride> retired = retiring;
    retiring.reset();
    installed->reset();
    EXPECT_TRUE(retired.expired());
    ASSERT_TRUE(replacementScope);
    const auto loaded = registry_.getShader(probe(), cache(), log_);
    ASSERT_TRUE(loaded) << log_;
    EXPECT_EQ(loaded->spirv, artifactSpirv(69));
}

TEST_F(ShaderArtifactOverrideTest, ConcurrentInstallPublishesExactlyOneOwner)
{
    auto firstOverride = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(67)};
    });
    auto secondOverride = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(68)};
    });
    Result<ShaderArtifactOverrideScope> first = makeError(Error::Failure);
    Result<ShaderArtifactOverrideScope> second = makeError(Error::Failure);
    std::mutex mutex;
    std::condition_variable changed;
    uint32_t ready = 0;
    bool start = false;
    const auto install = [&](const auto& override, auto& result) {
        {
            std::unique_lock lock(mutex);
            ++ready;
            changed.notify_all();
            changed.wait(lock, [&] { return start; });
        }
        result = registry_.installArtifactOverrideForTesting(override);
    };
    std::jthread firstWorker([&] { install(firstOverride, first); });
    std::jthread secondWorker([&] { install(secondOverride, second); });
    {
        std::unique_lock lock(mutex);
        EXPECT_TRUE(changed.wait_for(lock, std::chrono::seconds(5), [&] { return ready == 2; }));
        start = true;
    }
    changed.notify_all();
    firstWorker.join();
    secondWorker.join();
    ASSERT_NE(bool(first), bool(second));
    EXPECT_EQ(first ? second.error() : first.error(), Error::InvalidArgument);
    const auto loaded = registry_.getShader(probe(), cache(), log_);
    ASSERT_TRUE(loaded) << log_;
    EXPECT_EQ(loaded->spirv, artifactSpirv(first ? 67 : 68));
}

TEST_F(ShaderArtifactOverrideTest, ConcurrentCallbacksAreIndependentAndScopeResetKeepsInflightOwnerAlive)
{
    struct Gate {
        std::mutex mutex;
        std::condition_variable changed;
        uint32_t entered = 0;
        bool released = false;
    };
    auto gate = std::make_shared<Gate>();
    auto override = std::make_shared<CallbackArtifactOverride>([gate](const SlangShaderDesc&, std::string& log)
        -> ArtifactResolution {
        std::unique_lock lock(gate->mutex);
        ++gate->entered;
        gate->changed.notify_all();
        if (!gate->changed.wait_for(lock, std::chrono::seconds(10), [&] { return gate->released; })) {
            log = "Test callback gate timed out";
            return Result<ShaderArtifactOverrideResult>{makeError(Error::Failure)};
        }
        return artifact(64);
    });
    std::weak_ptr<ShaderArtifactOverride> inflightOwner = override;
    auto installed = registry_.installArtifactOverrideForTesting(override);
    ASSERT_TRUE(installed);
    override.reset();

    Result<ShaderCompileResult> first = makeError(Error::Failure);
    Result<ShaderCompileResult> second = makeError(Error::Failure);
    std::string firstLog, secondLog;
    std::jthread firstWorker([&] { first = registry_.getShader(probe(), cache(), firstLog); });
    std::jthread secondWorker([&] { second = registry_.getShader(probe(), cache(), secondLog); });
    bool bothEntered = false;
    {
        std::unique_lock lock(gate->mutex);
        bothEntered = gate->changed.wait_for(lock, std::chrono::seconds(5), [&] { return gate->entered == 2; });
    }
    EXPECT_TRUE(bothEntered); // The registry must not serialize callbacks under its source mutex.
    installed->reset();
    EXPECT_FALSE(inflightOwner.expired());
    auto replacement = std::make_shared<CallbackArtifactOverride>([](const SlangShaderDesc&, std::string&) {
        return ArtifactResolution{artifact(65)};
    });
    auto replacementScope = registry_.installArtifactOverrideForTesting(replacement);
    EXPECT_TRUE(replacementScope);
    if (replacementScope) {
        const auto latest = registry_.getShader(probe(), cache(), log_);
        EXPECT_TRUE(latest) << log_;
        if (latest) { EXPECT_EQ(latest->spirv, artifactSpirv(65)); }
    }
    {
        std::lock_guard lock(gate->mutex);
        gate->released = true;
    }
    gate->changed.notify_all();
    firstWorker.join();
    secondWorker.join();
    ASSERT_TRUE(first) << firstLog;
    ASSERT_TRUE(second) << secondLog;
    EXPECT_EQ(first->spirv, artifactSpirv(64));
    EXPECT_EQ(second->spirv, artifactSpirv(64));
    expectArtifactMetrics(first->metrics, first->spirv.size() * sizeof(uint32_t));
    expectArtifactMetrics(second->metrics, second->spirv.size() * sizeof(uint32_t));
    EXPECT_TRUE(inflightOwner.expired());
    expectCacheAbsent();
}

} // namespace

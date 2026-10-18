#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/Core/SlangWarmupSchedule.h"
#include "Runtime/Render/Core/SlangSnapshotFileSystem.h"
#include "Runtime/Render/Core/SlangWarmupCacheTransaction.h"

#include <gtest/gtest.h>
#include <chrono>
#include <mutex>

namespace metallic::render {
struct ShaderRegistryIdentityTestAccess
{
    static void registerShader(const SlangShaderDesc& desc, const ShaderCompileResult& shader)
    {
        ShaderRegistry::instance().registerCompiledShader(desc, shader);
    }
    static PipelineRecipeShader recipe(std::span<const uint32_t> spirv)
    {
        return ShaderRegistry::instance().recipeShader(spirv);
    }
    static std::string deviceGroup(std::span<const uint32_t> spirv, uint64_t deviceHash)
    {
        auto& registry = ShaderRegistry::instance();
        registry.registerShaderModule(spirv, deviceHash);
        return registry.cacheGroup(deviceHash);
    }
};
} // namespace metallic::render

namespace {
using namespace metallic::render;
using namespace metallic::render::slang_snapshot;

class WarmupDirectory
{
public:
    WarmupDirectory()
    {
        static std::atomic<uint64_t> sequence{0};
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::absolute(TEST_BINARY_DIR) / "ShaderWarmupBatchFixtures" /
            (std::to_string(stamp) + "-" + std::to_string(sequence++));
        if (fs::exists(path)) { throw std::runtime_error("Shader fixture directory already exists"); }
        fs::create_directories(path / "source");
        fs::create_directories(path / "cache");
    }
    ~WarmupDirectory()
    {
        resetSlangShaderHotReloadTracking();
        std::error_code error;
        fs::remove_all(path, error);
    }
    fs::path path;
};

void writeFile(const fs::path& path, std::span<const uint8_t> bytes)
{
    fs::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    stream.close();
    if (!stream) { throw std::runtime_error("Shader fixture write failed"); }
}
void writeText(const fs::path& path, std::string_view text)
{
    writeFile(path, {reinterpret_cast<const uint8_t*>(text.data()), text.size()});
}
void deliver(SlangSnapshotFileSystem& snapshot, const fs::path& path)
{
    Slang::ComPtr<ISlangBlob> blob;
    ASSERT_TRUE(SLANG_SUCCEEDED(snapshot.loadFile(pathText(path).c_str(), blob.writeRef())));
}

const std::vector<uint32_t> kMinimalCode{
    0x07230203, 0x00010600, 0, 5, 0,
    0x00020011, 1, 0x0003000e, 0, 1, 0x0005000f, 5, 3, 0x6e69616d, 0,
    0x00060010, 3, 17, 1, 1, 1, 0x00020013, 1, 0x00030021, 2, 1,
    0x00050036, 1, 3, 0, 2, 0x000200f8, 4, 0x000100fd, 0x00010038};

TEST(ShaderWarmupBatch, StreamingDeliversOriginalIndicesBeforeAcceptanceAndRejectsConsumerFailure)
{
    WarmupDirectory directory;
    const auto source = directory.path / "source", cache = directory.path / "cache";
    writeText(source / "StreamingA.slang", "module StreamingA; RWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = 17u; }\n");
    writeText(source / "StreamingB.slang", "module StreamingB; RWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = 29u; }\n");
    const ShaderRequest a{.module = "StreamingA", .entry = "probe", .searchPath = source.string(),
        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped, .optimizationLevel = SlangShaderOptimizationLevel::O1};
    auto b = a;
    b.module = "StreamingB";
    const std::array requests{a, b, a};
    const auto cachePath = cache.string();
    std::mutex mutex;
    bool acceptanceStarted = false;
    std::array<size_t, 3> delivered{};
    std::array<std::vector<uint32_t>, 3> copies;
    SlangWarmupBatchOptions options{.cache = {.cacheDirectory = cachePath.c_str()}, .jobs = 2,
        .onProgress = [&](const SlangWarmupBatchProgress& progress) {
            if (progress.phase && (std::string_view(progress.phase) == "validating" ||
                std::string_view(progress.phase) == "finished")) {
                std::lock_guard lock(mutex);
                acceptanceStarted = true;
                EXPECT_EQ(delivered, (std::array<size_t, 3>{1, 1, 1}));
            }
        },
        .onShaderReady = [&](size_t index, const ShaderCompileResult& shader) {
            std::lock_guard lock(mutex);
            EXPECT_FALSE(acceptanceStarted);
            ASSERT_LT(index, delivered.size());
            EXPECT_FALSE(shader.spirv.empty());
            ++delivered[index];
            copies[index] = shader.spirv;
        }};
    std::string log;
    auto result = ShaderRegistry::instance().getWarmupShadersWithOverrides(requests, options, log);
    ASSERT_TRUE(result) << log;
    EXPECT_EQ(result->sources.existingCacheHits + result->sources.compilerExecutions, 2u);
    ASSERT_EQ(result->sources.shaders.size(), copies.size());
    for (size_t index = 0; index < copies.size(); ++index) { EXPECT_EQ(copies[index], result->sources.shaders[index].spirv); }
    EXPECT_EQ(copies[0], copies[2]);

    options.onProgress = {};
    options.onShaderReady = [](size_t, const ShaderCompileResult&) { throw std::runtime_error("consumer rejected provisional shader"); };
    EXPECT_FALSE(ShaderRegistry::instance().getWarmupShadersWithOverrides(requests, options, log));
    EXPECT_NE(log.find("consumer rejected provisional shader"), std::string::npos);

#ifdef _WIN32
    bool mutated = false;
    options.jobs = 1;
    options.onShaderReady = [&](size_t, const ShaderCompileResult&) {
        if (!mutated) {
            mutated = true;
            writeText(source / "StreamingA.slang", "module StreamingA; // changed after provisional completion\n");
        }
    };
    EXPECT_FALSE(ShaderRegistry::instance().getWarmupShadersWithOverrides(requests, options, log));
    EXPECT_TRUE(mutated);
#endif
}

TEST(ShaderWarmupBatch, SchedulingPrioritizesHeavyEntriesWithinSharedGroups)
{
    const SlangMacroDefine sharc{"SHARC_UPDATE", "1"};
    const SlangShaderDesc light{.moduleName = "Features/PathTracing/OpenPBRRayQueryPathTrace"};
    auto heavy = light;
    heavy.macroDefines = std::span(&sharc, 1);
    const SlangShaderDesc small{.moduleName = "Features/Other/Small"};
    const std::array requests{light, heavy, small};
    std::vector<WarmupGroupPlan> groups{{.requestIndices = {2}}, {.requestIndices = {0, 1}}};
    estimateAndSortWarmupGroups(requests, groups);
    ASSERT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups[0].requestIndices, (std::vector<size_t>{1, 0}));
    EXPECT_EQ(groups[1].requestIndices, (std::vector<size_t>{2}));
}

TEST(ShaderWarmupBatch, RejectedOriginalAliasDoesNotRegisterRepresentativeIdentity)
{
#ifndef _WIN32
    GTEST_SKIP() << "Windows process environment policy injection";
#else
    struct OptimizationEnvironment {
        std::optional<std::string> previous;
        OptimizationEnvironment()
        {
            if (const char* value = std::getenv("METALLIC_SLANG_OPTIMIZATION")) { previous = value; }
            _putenv_s("METALLIC_SLANG_OPTIMIZATION", "O2");
        }
        ~OptimizationEnvironment()
        {
            _putenv_s("METALLIC_SLANG_OPTIMIZATION", previous ? previous->c_str() : "");
        }
    } environment;
    WarmupDirectory directory;
    const auto source = directory.path / "source";
    writeText(source / "UnacceptedAlias.slang", "module UnacceptedAlias; RWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = 371938u; }\n");
    const ShaderRequest automatic{.module = "UnacceptedAlias", .entry = "probe", .searchPath = source.string(),
        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped};
    auto explicitPolicy = automatic;
    explicitPolicy.optimizationLevel = SlangShaderOptimizationLevel::O2;
    const std::array requests{automatic, explicitPolicy};
    const auto cachePath = (directory.path / "cache").string();
    bool changedPolicy = false;
    std::vector<uint32_t> provisional;
    const SlangWarmupBatchOptions options{.cache = {.cacheDirectory = cachePath.c_str()}, .jobs = 1,
        .onProgress = [&](const SlangWarmupBatchProgress& progress) {
            if (!changedPolicy && progress.phase && std::string_view(progress.phase) == "preparing") {
                changedPolicy = true;
                _putenv_s("METALLIC_SLANG_OPTIMIZATION", "O3");
            }
        },
        .onShaderReady = [&](size_t, const ShaderCompileResult& shader) {
            provisional = shader.spirv;
            EXPECT_TRUE(ShaderRegistryIdentityTestAccess::recipe(provisional).moduleName.empty());
        }};
    std::string log;
    EXPECT_FALSE(ShaderRegistry::instance().getWarmupShadersWithOverrides(requests, options, log));
    EXPECT_TRUE(changedPolicy);
    ASSERT_FALSE(provisional.empty());
    EXPECT_NE(log.find("Effective shader request identity changed"), std::string::npos);
    EXPECT_TRUE(ShaderRegistryIdentityTestAccess::recipe(provisional).moduleName.empty());
#endif
}

TEST(ShaderWarmupBatch, RealSlangColdAllHitMixedPrivateHeaderShadowAndOrdinaryCacheFailures)
{
#ifndef _WIN32
    GTEST_SKIP() << "Live epoch qualification uses Windows notifications; fresh fallback is covered separately";
#else
    WarmupDirectory directory;
    const auto source = directory.path / "source", cache = directory.path / "cache";
    writeText(source / "AOnly.h", "#define A_VALUE 17u\n");
    writeText(source / "BatchA.slang", "module BatchA;\n#include \"AOnly.h\"\nRWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = A_VALUE; }\n");
    writeText(source / "BatchB.slang", "module BatchB;\n#ifndef B_VALUE\n#define B_VALUE 29u\n#endif\nRWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = B_VALUE; }\n");
    const auto sourcePath = source.string(), cachePath = cache.string();
    const SlangShaderDesc a{.moduleName = "BatchA", .entryPointName = "probe", .searchPath = sourcePath.c_str(),
        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped, .optimizationLevel = SlangShaderOptimizationLevel::O1,
        .postprocessMode = SlangShaderPostprocessMode::None};
    auto b = a; b.moduleName = "BatchB";
    std::array requests{a, b};
    std::mutex progressMutex;
    std::vector<SlangWarmupBatchProgress> progressEvents;
    bool throwFromProgress = false;
    const SlangWarmupBatchOptions options{.cache = {.cacheDirectory = cachePath.c_str()}, .jobs = 2,
        .onProgress = [&](const SlangWarmupBatchProgress& progress) {
            {
                std::lock_guard lock(progressMutex);
                progressEvents.push_back(progress);
            }
            if (throwFromProgress) { throw std::runtime_error("Diagnostic callback failure"); }
        }};
    const auto checkProgress = [&](std::array<bool, 2> expectedHits) {
        std::lock_guard lock(progressMutex);
        std::vector<std::string> phases;
        std::array<size_t, 2> completionCounts{};
        for (const auto& event : progressEvents) {
            ASSERT_NE(event.phase, nullptr);
            if (std::string_view(event.phase) == "request-complete") {
                // Results may arrive in either order, but only during compilation.
                ASSERT_FALSE(phases.empty());
                EXPECT_EQ(phases.back(), "compiling");
                ASSERT_LT(event.requestIndex, expectedHits.size());
                ++completionCounts[event.requestIndex];
                EXPECT_EQ(event.cacheHit, expectedHits[event.requestIndex]);
            } else {
                EXPECT_EQ(event.requestIndex, std::numeric_limits<size_t>::max());
                phases.emplace_back(event.phase);
            }
        }
        const std::vector<std::string> expectedPhases{
            "preparing", "capturing", "compiling", "validating", "publishing", "finished"};
        EXPECT_EQ(phases, expectedPhases);
        EXPECT_EQ(completionCounts[0], 1u);
        EXPECT_EQ(completionCounts[1], 1u);
        ASSERT_FALSE(progressEvents.empty());
        EXPECT_STREQ(progressEvents.back().phase, "finished");
        progressEvents.clear();
    };
    auto& registry = ShaderRegistry::instance();
    std::string log;
    auto cold = registry.getWarmupShaders(requests, options, log);
    ASSERT_TRUE(cold) << log;
    ASSERT_TRUE(cold->completedAllRequests && cold->completeSourceEpochValidated);
    EXPECT_EQ(cold->compilerExecutions, 2u); EXPECT_EQ(cold->existingCacheHits, 0u); EXPECT_EQ(cold->sessionGroups, 1u);
    checkProgress({false, false});
    auto allHit = registry.getWarmupShaders(requests, options, log);
    ASSERT_TRUE(allHit) << log;
    EXPECT_EQ(allHit->compilerExecutions, 0u); EXPECT_EQ(allHit->existingCacheHits, 2u);
    EXPECT_EQ(allHit->shaders[0].spirv, cold->shaders[0].spirv); EXPECT_EQ(allHit->shaders[1].spirv, cold->shaders[1].spirv);
    checkProgress({true, true});
    const SlangMacroDefine define{"B_VALUE", "41u"};
    b.macroDefines = std::span(&define, 1); requests[1] = b;
    // Exercise concurrent session groups and exceptions on every callback;
    // diagnostic failures must not change compilation or publication acceptance.
    throwFromProgress = true;
    auto mixed = registry.getWarmupShaders(requests, options, log);
    ASSERT_TRUE(mixed) << log;
    EXPECT_EQ(mixed->compilerExecutions, 1u); EXPECT_EQ(mixed->existingCacheHits, 1u);
    EXPECT_EQ(mixed->shaders[0].spirv, cold->shaders[0].spirv); EXPECT_NE(mixed->shaders[1].spirv, cold->shaders[1].spirv);
    checkProgress({true, false});
    bool hit = false;
    const SlangShaderCacheOptions ordinaryOptions{.cacheDirectory = cachePath.c_str(), .outCacheHit = &hit};
    auto ordinary = registry.getShader(a, ordinaryOptions, log);
    ASSERT_TRUE(ordinary) << log; ASSERT_TRUE(hit);
    EXPECT_FALSE(ordinary->metrics.compilerExecuted); EXPECT_EQ(ordinary->spirv, cold->shaders[0].spirv);
    EXPECT_FALSE(ordinary->metrics.sourceEpochReused);
    EXPECT_GT(ordinary->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    auto reused = registry.getShader(a, ordinaryOptions, log);
    ASSERT_TRUE(reused) << log; ASSERT_TRUE(hit);
    EXPECT_TRUE(reused->metrics.sourceEpochReused);
    EXPECT_EQ(reused->metrics.sourceEpochCaptureTimeNanoseconds, 0u);
    EXPECT_GT(reused->metrics.producerValidationTimeNanoseconds, 0u);
    EXPECT_GT(reused->metrics.sourceValidationTimeNanoseconds, 0u);
    EXPECT_FALSE(reused->metrics.compilerExecuted);
    EXPECT_EQ(reused->spirv, ordinary->spirv);
    const auto headerTime = fs::last_write_time(source / "AOnly.h");
    writeText(source / "AOnly.h", "#define A_VALUE 31u\n"); fs::last_write_time(source / "AOnly.h", headerTime);
    auto changed = registry.getShader(a, ordinaryOptions, log);
    ASSERT_TRUE(changed) << log; EXPECT_FALSE(hit); EXPECT_NE(changed->spirv, cold->shaders[0].spirv);
    EXPECT_FALSE(changed->metrics.sourceEpochReused);

    // Device warmup consumes this batch entry directly. A same-size edit with
    // a restored timestamp must invalidate it as well as the ordinary path.
    const SlangWarmupBatchOptions invalidationOptions{.cache = {.cacheDirectory = cachePath.c_str()}, .jobs = 1};
    writeText(source / "AOnly.h", "#define A_VALUE 47u\n"); fs::last_write_time(source / "AOnly.h", headerTime);
    auto changedBatch = registry.getWarmupShaders(std::span(&a, 1), invalidationOptions, log);
    ASSERT_TRUE(changedBatch) << log;
    ASSERT_TRUE(changedBatch->completedAllRequests && changedBatch->completeSourceEpochValidated);
    EXPECT_EQ(changedBatch->compilerExecutions, 1u); EXPECT_EQ(changedBatch->existingCacheHits, 0u);
    ASSERT_EQ(changedBatch->shaders.size(), 1u);
    EXPECT_NE(changedBatch->shaders[0].spirv, changed->spirv);
    auto changedBatchHit = registry.getWarmupShaders(std::span(&a, 1), invalidationOptions, log);
    ASSERT_TRUE(changedBatchHit) << log;
    EXPECT_EQ(changedBatchHit->compilerExecutions, 0u); EXPECT_EQ(changedBatchHit->existingCacheHits, 1u);
    EXPECT_EQ(changedBatchHit->shaders[0].spirv, changedBatch->shaders[0].spirv);
    auto changedOrdinaryHit = registry.getShader(a, ordinaryOptions, log);
    ASSERT_TRUE(changedOrdinaryHit) << log; EXPECT_TRUE(hit);
    EXPECT_EQ(changedOrdinaryHit->spirv, changedBatch->shaders[0].spirv);

    const auto first = directory.path / "first", second = directory.path / "second";
    fs::create_directories(first); fs::create_directories(second);
    writeText(second / "BatchValues.slang", "module BatchValues; public uint value() { return 17u; }\n");
    writeText(first / "BatchC.slang", "module BatchC; import BatchValues; RWStructuredBuffer<uint> output;\n"
        "[shader(\"compute\")] [numthreads(1,1,1)] void probe(uint3 id : SV_DispatchThreadID) { output[id.x] = value(); }\n");
    const auto firstPath = first.string(), secondPath = second.string(); const char* extra = secondPath.c_str();
    auto c = a; c.moduleName = "BatchC"; c.searchPath = firstPath.c_str(); c.additionalSearchPaths = std::span(&extra, 1);
    auto oldShadow = registry.getShader(c, ordinaryOptions, log); ASSERT_TRUE(oldShadow) << log;
    auto shadowHit = registry.getShader(c, ordinaryOptions, log); ASSERT_TRUE(shadowHit) << log; ASSERT_TRUE(hit);
    writeText(first / "BatchValues.slang", "module BatchValues; public uint value() { return 31u; }\n");
    auto shadow = registry.getShader(c, ordinaryOptions, log); ASSERT_TRUE(shadow) << log;
    EXPECT_FALSE(hit); EXPECT_NE(shadow->spirv, oldShadow->spirv);
    // Removing and then restoring the higher-priority module changes resolver
    // meaning without changing the request or either module's surviving bytes.
    ASSERT_TRUE(fs::remove(first / "BatchValues.slang"));
    auto removedShadowBatch = registry.getWarmupShaders(std::span(&c, 1), invalidationOptions, log);
    ASSERT_TRUE(removedShadowBatch) << log;
    ASSERT_TRUE(removedShadowBatch->completedAllRequests && removedShadowBatch->completeSourceEpochValidated);
    EXPECT_EQ(removedShadowBatch->compilerExecutions, 1u); EXPECT_EQ(removedShadowBatch->existingCacheHits, 0u);
    ASSERT_EQ(removedShadowBatch->shaders.size(), 1u);
    EXPECT_EQ(removedShadowBatch->shaders[0].spirv, oldShadow->spirv);
    writeText(first / "BatchValues.slang", "module BatchValues; public uint value() { return 31u; }\n");
    auto restoredShadowBatch = registry.getWarmupShaders(std::span(&c, 1), invalidationOptions, log);
    ASSERT_TRUE(restoredShadowBatch) << log;
    ASSERT_TRUE(restoredShadowBatch->completedAllRequests && restoredShadowBatch->completeSourceEpochValidated);
    EXPECT_EQ(restoredShadowBatch->compilerExecutions, 1u); EXPECT_EQ(restoredShadowBatch->existingCacheHits, 0u);
    ASSERT_EQ(restoredShadowBatch->shaders.size(), 1u);
    EXPECT_EQ(restoredShadowBatch->shaders[0].spirv, shadow->spirv);
    auto restoredOrdinaryHit = registry.getShader(c, ordinaryOptions, log);
    ASSERT_TRUE(restoredOrdinaryHit) << log; EXPECT_TRUE(hit);
    EXPECT_EQ(restoredOrdinaryHit->spirv, shadow->spirv);
    const auto blocked = directory.path / "cache-is-file"; writeText(blocked, "not a directory");
    const auto blockedPath = blocked.string();
    auto noCache = registry.getShader(c, {.cacheDirectory = blockedPath.c_str()}, log);
    ASSERT_TRUE(noCache) << log; EXPECT_FALSE(noCache->metrics.cacheHit); EXPECT_EQ(noCache->spirv, shadow->spirv);
    auto overlap = registry.getShader(c, {.cacheDirectory = firstPath.c_str()}, log);
    ASSERT_TRUE(overlap) << log; EXPECT_EQ(overlap->spirv, shadow->spirv);
    for (const auto& item : fs::recursive_directory_iterator(first)) { EXPECT_NE(item.path().extension(), ".spv"); }
#endif
}

TEST(ShaderWarmupBatch, IssuedTokensAcceptedHitUnionAndActualFilePublicationGates)
{
#ifndef _WIN32
    GTEST_SKIP() << "Windows notification epoch contract";
#else
    WarmupDirectory directory;
    const auto source = directory.path / "source", cache = directory.path / "cache";
    const auto a = source / "A.slang", b = source / "B.slang", header = source / "AOnly.h";
    writeText(a, "module A;\n#include \"AOnly.h\"\n"); writeText(b, "module B;\n"); writeText(header, "#define A_VALUE 17u\n");
    const Json graph{{"mappings", Json::array({{{"logical", pathText(source)}, {"physical", pathText(source)}}})}};
    constexpr uint64_t producer = 0x7103;
    slang_shader_cache::Bytes cacheA;
    {
        SlangSnapshotFileSystem snapshot(graph, 1);
        Slang::ComPtr<ISlangBlob> identity;
        ASSERT_TRUE(SLANG_SUCCEEDED(snapshot.getFileUniqueIdentity(pathText(a).c_str(), identity.writeRef())));
        const std::string token(static_cast<const char*>(identity->getBufferPointer()));
        ShaderSourceReadProof readProof; fs::file_time_type time;
        EXPECT_FALSE(snapshot.actualIssuedDependency(token.c_str(), readProof, time));
        deliver(snapshot, a); deliver(snapshot, header);
        ASSERT_TRUE(snapshot.actualIssuedDependency(token.c_str(), readProof, time));
        EXPECT_EQ(readProof.canonicalPath, pathText(a));
        EXPECT_FALSE(snapshot.actualIssuedDependency("999:foreign", readProof, time));
        cacheA = slang_shader_cache::encode(0xa, snapshot.sourceProof(producer), kMinimalCode);
    }
    for (const bool mixed : {false, true}) {
        SlangSnapshotFileSystem snapshot(graph, 2); std::string log;
        slang_shader_cache::CachedPayload decoded;
        ASSERT_TRUE(snapshot.decodeAndAcceptCache(cacheA, 0xa, producer, decoded, log)) << log;
        if (mixed) { deliver(snapshot, b); }
        const auto proof = snapshot.sourceProof(producer);
        EXPECT_TRUE(std::ranges::any_of(proof.actualReads, [&](const auto& item) { return item.canonicalPath == pathText(header); }));
        SlangWarmupCacheTransaction<SlangSnapshotFileSystem> transaction(snapshot, producer, cache / (mixed ? "mixed" : "all-hit"));
        ASSERT_TRUE(transaction.stage({1, 0xa, kMinimalCode}, log));
        if (mixed) { ASSERT_TRUE(transaction.stage({2, 0xb, kMinimalCode}, log)); }
        size_t registered = 0;
        ASSERT_TRUE(transaction.publish(mixed ? 2 : 1, [](const auto& path, std::span<const uint8_t> bytes) {
            writeFile(path, bytes); return true;
        }, [&](size_t) { ++registered; }, log)) << log;
        EXPECT_EQ(registered, mixed ? 2u : 1u);
    }
    {
        SlangSnapshotFileSystem snapshot(graph, 3);
        auto damaged = cacheA;
        slang_shader_cache::Cursor headerCursor{damaged, 24}; uint64_t proofSize = 0;
        ASSERT_TRUE(headerCursor.integer(proofSize, 8));
        damaged[48 + static_cast<size_t>(proofSize)] ^= 1;
        const auto checksum = slang_shader_cache::checksum(std::span<const uint8_t>(damaged).subspan(48));
        for (size_t byte = 0; byte < 8; ++byte) { damaged[40 + byte] = static_cast<uint8_t>(checksum >> (byte * 8)); }
        bool qualifierPassed = false;
        std::string log; slang_shader_cache::CachedPayload decoded;
        EXPECT_FALSE(slang_shader_cache::decode(damaged, 0xa, [&](const auto& old, auto& current) {
            qualifierPassed = snapshot.qualifyCachedProof(old, producer, current, log); return qualifierPassed;
        }, decoded));
        EXPECT_TRUE(qualifierPassed); // Valid proof and CRC, then bad SPIR-V.
        EXPECT_FALSE(snapshot.decodeAndAcceptCache(damaged, 0xa, producer, decoded, log));
        EXPECT_THROW(snapshot.sourceProof(producer), std::runtime_error);
        auto legacy = cacheA; legacy[8] = 2;
        size_t callbacks = 0;
        EXPECT_FALSE(slang_shader_cache::decode(legacy, 0xa, [&](const auto&, auto&) { ++callbacks; return true; }, decoded));
        EXPECT_EQ(callbacks, 0u);
    }
    {
        const auto oldTime = fs::last_write_time(header);
        writeText(header, "#define A_VALUE 31u\n"); fs::last_write_time(header, oldTime);
        SlangSnapshotFileSystem snapshot(graph, 4);
        const auto cached = read(cache / "mixed/000000000000000a.spv");
        std::string log; slang_shader_cache::CachedPayload decoded;
        EXPECT_FALSE(snapshot.decodeAndAcceptCache(cached, 0xa, producer, decoded, log));
        EXPECT_THROW(snapshot.sourceProof(producer), std::runtime_error);
    }
    {
        SlangSnapshotFileSystem snapshot(graph, 5); deliver(snapshot, b);
        const auto callback = +[](SlangPathType, const char*, void*) {};
        EXPECT_TRUE(SLANG_FAILED(snapshot.enumeratePathContents(pathText(directory.path / "outside/missing").c_str(), callback, nullptr)));
        EXPECT_THROW(snapshot.sourceProof(producer), std::runtime_error);
    }
    {
        SlangSnapshotFileSystem snapshot(graph, 6); deliver(snapshot, b);
        writeText(b, "module B; public uint value = 7;\n"); writeText(b, "module B;\n");
        std::string reason;
        bool advanced = false;
        for (size_t attempt = 0; attempt < 100; ++attempt) {
            if (!snapshot.validateNotifications(reason)) { advanced = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        EXPECT_TRUE(advanced);
        EXPECT_FALSE(snapshot.validate(reason)); // Restored bytes still invalidate the old epoch.
    }
    for (const std::string mode : {"normal", "writer-failure", "short", "trailing", "damaged", "corrupt-proof", "source-edit"}) {
        fs::create_directories(cache / mode);
        SlangSnapshotFileSystem snapshot(graph, 3); deliver(snapshot, b);
        SlangWarmupCacheTransaction<SlangSnapshotFileSystem> transaction(snapshot, producer, cache / mode);
        std::string log; ASSERT_TRUE(transaction.stage({1, 0xa, kMinimalCode}, log));
        ASSERT_TRUE(transaction.stage({2, 0xb, kMinimalCode}, log));
        const auto proof = snapshot.sourceProof(producer);
        size_t writes = 0, registered = 0;
        const bool published = transaction.publish(2, [&](const auto& path, std::span<const uint8_t> bytes) {
            ++writes;
            EXPECT_EQ(registered, 0u);
            if (writes == 2 && mode == "writer-failure") { return false; }
            slang_shader_cache::Bytes altered(bytes.begin(), bytes.end());
            if (writes == 2) {
                if (mode == "short") { altered.pop_back(); }
                else if (mode == "trailing") { altered.push_back(0); }
                else if (mode == "damaged" || mode == "corrupt-proof") {
                    if (mode == "damaged") { altered.back() ^= 1; }
                    else { altered[slang_shader_cache::kHeaderSize + 4] ^= 1; }
                    const auto checksum = slang_shader_cache::checksum(
                        std::span<const uint8_t>(altered).subspan(slang_shader_cache::kHeaderSize));
                    for (size_t byte = 0; byte < 8; ++byte) { altered[40 + byte] = static_cast<uint8_t>(checksum >> (8 * byte)); }
                }
            }
            writeFile(path, altered);
            if (writes == 2 && mode == "source-edit") {
                const auto timestamp = fs::last_write_time(b);
                writeText(b, "module B; public uint changed = 1;\n");
                fs::last_write_time(b, timestamp);
            }
            return true;
        }, [&](size_t) { EXPECT_EQ(writes, 2u); ++registered; }, log);
        EXPECT_EQ(published, mode == "normal") << mode << ": " << log;
        EXPECT_EQ(writes, 2u);
        EXPECT_EQ(registered, published ? 2u : 0u);
        EXPECT_EQ(transaction.published(), published);
        if (published) {
            EXPECT_EQ(read(cache / mode / "000000000000000a.spv"), slang_shader_cache::encode(0xa, proof, kMinimalCode));
            EXPECT_EQ(read(cache / mode / "000000000000000b.spv"), slang_shader_cache::encode(0xb, proof, kMinimalCode));
        }
    }
#endif
}

TEST(ShaderWarmupBatch, GenericSnapshotsKeepPerFileProofQualification)
{
#ifndef _WIN32
    GTEST_SKIP() << "Windows snapshot proof fixture";
#else
    WarmupDirectory directory;
    const auto source = directory.path / "source";
    const auto input = source / "Batch.slang";
    writeText(input, "module Batch;\n");
    const Json graph{{"mappings", Json::array({{{"logical", pathText(source)}, {"physical", pathText(source)}}})}};
    constexpr uint64_t producer = 0x7103;
    ShaderNamespaceProof proof;
    {
        SlangSnapshotFileSystem actual(graph, 1);
        deliver(actual, input);
        proof = actual.sourceProof(producer);
    }
    // This compatible snapshot deliberately has no notification API. Its proof
    // qualification must still run for each actual persisted envelope.
    struct GenericSnapshot {
        ShaderNamespaceProof proof;
        size_t validations = 0, qualifications = 0;
        size_t failAtQualification = 0;
        bool validate(std::string&) { ++validations; return true; }
        ShaderNamespaceProof sourceProof(uint64_t) { return proof; }
        bool qualifyCachedProof(const ShaderNamespaceProof& cached, uint64_t,
            ShaderNamespaceProof& current, std::string&)
        {
            ++qualifications; current = proof;
            if (qualifications == failAtQualification) { return false; }
            return namespaceProofMatches(cached, current);
        }
    } snapshot{proof};
    SlangWarmupCacheTransaction<GenericSnapshot> transaction(snapshot, producer, directory.path / "cache");
    std::string log;
    ASSERT_TRUE(transaction.stage({1, 0xa, kMinimalCode}, log));
    ASSERT_TRUE(transaction.stage({2, 0xb, kMinimalCode}, log));
    size_t writes = 0, registered = 0;
    ASSERT_TRUE(transaction.publish(2, [&](const auto& path, std::span<const uint8_t> bytes) {
        ++writes; EXPECT_EQ(registered, 0u); writeFile(path, bytes); return true;
    }, [&](size_t) {
        EXPECT_EQ(writes, 2u); EXPECT_EQ(snapshot.validations, 3u); ++registered;
    }, log)) << log;
    EXPECT_EQ(registered, 2u);
    EXPECT_EQ(snapshot.qualifications, 3u);
    EXPECT_EQ(snapshot.validations, 3u);

    // Initial proof and the first persisted envelope qualify; rejection of the
    // second persisted envelope must prevent every normal registration.
    GenericSnapshot rejected{proof};
    rejected.failAtQualification = 3;
    SlangWarmupCacheTransaction<GenericSnapshot> failed(rejected, producer, directory.path / "generic-last-rejected");
    ASSERT_TRUE(failed.stage({1, 0xa, kMinimalCode}, log));
    ASSERT_TRUE(failed.stage({2, 0xb, kMinimalCode}, log));
    size_t failedWrites = 0, failedRegistrations = 0;
    EXPECT_FALSE(failed.publish(2, [&](const auto& path, std::span<const uint8_t> bytes) {
        ++failedWrites; EXPECT_EQ(failedRegistrations, 0u); writeFile(path, bytes); return true;
    }, [&](size_t) { ++failedRegistrations; }, log));
    EXPECT_EQ(failedWrites, 2u);
    EXPECT_EQ(rejected.qualifications, 3u);
    EXPECT_EQ(rejected.validations, 2u);
    EXPECT_EQ(failedRegistrations, 0u);
    EXPECT_FALSE(failed.published());
#endif
}

TEST(ShaderWarmupBatch, NormalIdentityRegistrationPreservesMetricsAndStickyAmbiguity)
{
    auto code = kMinimalCode; code[2] = 0x724911;
    ShaderCompileResult shader{.spirv = code}; shader.metrics.totalTimeNanoseconds = 111; shader.metrics.cacheHit = true;
    const SlangShaderDesc desc{.moduleName = "Features/PathTracing/IdentityFixture", .entryPointName = "sourceEntry",
        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped, .optimizationLevel = SlangShaderOptimizationLevel::O1,
        .postprocessMode = SlangShaderPostprocessMode::None};
    ShaderRegistryIdentityTestAccess::registerShader(desc, shader);
    auto identity = ShaderRegistryIdentityTestAccess::recipe(code);
    EXPECT_EQ(identity.moduleName, desc.moduleName); EXPECT_EQ(identity.sourceEntryPoint, desc.entryPointName);
    EXPECT_EQ(identity.descriptorHeapMode, "mapped"); EXPECT_EQ(identity.optimizationLevel, "O1");
    EXPECT_EQ(identity.spirvPostprocess, "none"); EXPECT_FALSE(identity.sourceIdentityAmbiguous);
    ShaderRegistryIdentityTestAccess::registerShader(desc, shader);
    EXPECT_FALSE(ShaderRegistryIdentityTestAccess::recipe(code).sourceIdentityAmbiguous);
    EXPECT_EQ(ShaderRegistryIdentityTestAccess::deviceGroup(code, 0x7147), "ScenePathTracePass");
    auto different = desc; different.moduleName = "Features/GPUDriven/GPUDrivenStreamAsset";
    different.optimizationLevel = SlangShaderOptimizationLevel::O3;
    ShaderRegistryIdentityTestAccess::registerShader(different, shader);
    EXPECT_TRUE(ShaderRegistryIdentityTestAccess::recipe(code).sourceIdentityAmbiguous);
    EXPECT_EQ(ShaderRegistryIdentityTestAccess::recipe(code).moduleName, desc.moduleName);
    EXPECT_EQ(ShaderRegistryIdentityTestAccess::deviceGroup(code, 0x7147), "GPUDrivenStreamAssetPass");
    EXPECT_EQ(shader.metrics.totalTimeNanoseconds, 111u); EXPECT_TRUE(shader.metrics.cacheHit);
}
} // namespace

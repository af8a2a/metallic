#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/SlangSnapshotFileSystem.h"
#include "Runtime/Render/Core/SlangWarmupCacheTransaction.h"

#include <gtest/gtest.h>
#include <chrono>

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
    const SlangWarmupBatchOptions options{.cache = {.cacheDirectory = cachePath.c_str()}, .jobs = 2};
    auto& registry = ShaderRegistry::instance();
    std::string log;
    auto cold = registry.getWarmupShaders(requests, options, log);
    ASSERT_TRUE(cold) << log;
    ASSERT_TRUE(cold->completedAllRequests && cold->completeSourceEpochValidated);
    EXPECT_EQ(cold->compilerExecutions, 2u); EXPECT_EQ(cold->existingCacheHits, 0u); EXPECT_EQ(cold->sessionGroups, 1u);
    auto allHit = registry.getWarmupShaders(requests, options, log);
    ASSERT_TRUE(allHit) << log;
    EXPECT_EQ(allHit->compilerExecutions, 0u); EXPECT_EQ(allHit->existingCacheHits, 2u);
    EXPECT_EQ(allHit->shaders[0].spirv, cold->shaders[0].spirv); EXPECT_EQ(allHit->shaders[1].spirv, cold->shaders[1].spirv);
    const SlangMacroDefine define{"B_VALUE", "41u"};
    b.macroDefines = std::span(&define, 1); requests[1] = b;
    auto mixed = registry.getWarmupShaders(requests, options, log);
    ASSERT_TRUE(mixed) << log;
    EXPECT_EQ(mixed->compilerExecutions, 1u); EXPECT_EQ(mixed->existingCacheHits, 1u);
    EXPECT_EQ(mixed->shaders[0].spirv, cold->shaders[0].spirv); EXPECT_NE(mixed->shaders[1].spirv, cold->shaders[1].spirv);
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

#include "RHITest.h"
#include "Runtime/Render/Core/DeferredShaderCacheWriter.h"
#include "Runtime/Render/Core/ShaderRegistry.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <future>
#include <thread>

namespace metallic::tests {
namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsedNanoseconds(Clock::time_point begin)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
}

struct SaveSample {
    uint64_t callTimeNanoseconds = 0;
    uint64_t flushTimeNanoseconds = 0;
    render::PipelineCacheStats cache;
};

void writeSamples(std::ofstream& file, const std::vector<SaveSample>& samples)
{
    file << '[';
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i != 0) { file << ','; }
        const auto& sample = samples[i];
        file << "{\"call_ns\":" << sample.callTimeNanoseconds
             << ",\"flush_ns\":" << sample.flushTimeNanoseconds
             << ",\"extract_ns\":" << sample.cache.lastExtractTimeNanoseconds
             << ",\"write_ns\":" << sample.cache.lastWriteTimeNanoseconds
             << ",\"save_ns\":" << sample.cache.lastSaveTimeNanoseconds
             << ",\"backend_bytes\":" << sample.cache.backendDataSize << '}';
    }
    file << ']';
}

class PipelineCacheDeferredSaveTest final : public RHITest {
public:
    PipelineCacheDeferredSaveTest()
    {
        type = RHITestType::Resource;
        name = "pipeline_cache_deferred_save_and_snapshot";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const auto root = std::filesystem::absolute(context.outputDirectory / "pipeline-cache-save" /
            std::to_string(Clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        {
            std::ofstream source(root / "DeferredCacheProbe.slang");
            source << "#ifndef SAVE_THREADS\n#define SAVE_THREADS 31\n#endif\n"
                      "[shader(\"compute\")] [numthreads(SAVE_THREADS, 1, 1)]\n"
                      "void cacheProbeMain(uint3 id : SV_DispatchThreadID) {}\n";
            if (!source) { return RHITestResult::fail("could not write cache-save source fixture"); }
        }
        const std::string sourceRoot = root.string();
        const std::string sourceCacheRoot = (root / "spirv").string();
        auto& registry = ShaderRegistry::instance();
        std::vector<std::unique_ptr<ShaderModule>> modules;
        std::string log;
        constexpr uint32_t kVariantCount = 10;
        for (uint32_t i = 0; i < kVariantCount; ++i) {
            const std::string threadCount = std::to_string(31 + i);
            const SlangMacroDefine macro{"SAVE_THREADS", threadCount.c_str()};
            auto code = registry.getShader({.moduleName = "DeferredCacheProbe", .entryPointName = "cacheProbeMain",
                .searchPath = sourceRoot.c_str(), .macroDefines = {&macro, 1}},
                {.cacheDirectory = sourceCacheRoot.c_str()}, log);
            if (!code) { return RHITestResult::fail("cache-save source compilation failed: " + log); }
            auto module = registry.getShaderModule(context.device, {.spirv = code->spirv});
            if (!module) { return RHITestResult::fail("cache-save shader module creation failed"); }
            modules.push_back(std::move(*module));
        }

        const std::filesystem::path productionFixture = PROJECT_SOURCE_DIR "/.cache/pso/ScenePathTracePass.pso";
        const auto syncPath = root / "synchronous.pso";
        const auto deferredPath = root / "deferred.pso";
        std::error_code fileError;
        const bool fixtureAvailable = std::filesystem::is_regular_file(productionFixture, fileError);
        if (fileError == std::errc::no_such_file_or_directory) { fileError.clear(); }
        if (fileError) { return RHITestResult::fail("could not inspect optional production cache: " + fileError.message()); }
        if (fixtureAvailable) {
            // Keep the user's cache immutable; all native exports target these copies.
            std::filesystem::copy_file(productionFixture, syncPath, fileError);
            if (fileError) { return RHITestResult::fail("could not copy cache fixture: " + fileError.message()); }
            std::filesystem::copy_file(syncPath, deferredPath, fileError);
            if (fileError) { return RHITestResult::fail("could not duplicate cache fixture: " + fileError.message()); }
        }
        const std::string syncPathString = syncPath.string();
        const std::string deferredPathString = deferredPath.string();
        auto synchronous = context.device.createPipelineCache({.filePath = syncPathString.c_str(), .saveOnDestroy = false});
        auto deferred = context.device.createPipelineCache({.filePath = deferredPathString.c_str(), .saveOnDestroy = false});
        if (!synchronous || !deferred) { return RHITestResult::fail("cache-save fixture creation failed"); }
        const auto initialStats = (*synchronous)->stats();
        if (initialStats.storedPsoCount != (*deferred)->stats().storedPsoCount ||
            initialStats.backendDataSize != (*deferred)->stats().backendDataSize) {
            return RHITestResult::fail("matched cache fixtures did not load identical data");
        }
        const auto createVariant = [&](PipelineCache& cache, uint32_t variant) {
            return context.device.createComputePipeline({.computeShader = {modules[variant].get()}, .pipelineCache = &cache});
        };
        std::vector<uint64_t> hashes(kVariantCount);
        std::vector<SaveSample> syncSamples;
        std::vector<SaveSample> deferredSamples;
        for (uint32_t i = 0; i < 3; ++i) {
            auto pipeline = createVariant(**synchronous, i);
            if (!pipeline || (*pipeline)->pipelineCacheHit()) {
                return RHITestResult::fail("synchronous fixture did not create a new PSO variant");
            }
            hashes[i] = (*pipeline)->psoHash();
            const auto begin = Clock::now();
            const auto saved = (*synchronous)->save();
            const auto elapsed = elapsedNanoseconds(begin);
            const auto stats = (*synchronous)->stats();
            if (!saved || stats.persistedRevision != stats.dirtyRevision || stats.saveCount != i + 1) {
                return RHITestResult::fail("synchronous save did not persist its complete revision");
            }
            syncSamples.push_back({elapsed, 0, stats});
        }

        // A long quiet period isolates queue cost from timer scheduling; flush is
        // the explicit durability boundary. Production uses its default delay.
        // Failed callbacks can be retained until writer destruction, so every
        // captured resource is declared before the writer and outlives it.
        std::atomic_bool saved = false;
        std::atomic_uint burstSaves = 0;
        DeferredShaderCacheWriter writer(std::chrono::milliseconds(30'000), std::chrono::milliseconds(30'000));
        for (uint32_t i = 0; i < 3; ++i) {
            auto pipeline = createVariant(**deferred, i);
            if (!pipeline || (*pipeline)->pipelineCacheHit() || (*pipeline)->psoHash() != hashes[i]) {
                return RHITestResult::fail("deferred fixture changed the matched PSO workload");
            }
            const auto revision = (*deferred)->stats().dirtyRevision;
            saved = false;
            const auto begin = Clock::now();
            writer.request("deferred", revision, [&] {
                const bool succeeded = static_cast<bool>((*deferred)->save());
                saved = succeeded;
                return succeeded;
            });
            const auto enqueueElapsed = elapsedNanoseconds(begin);
            const auto flushBegin = Clock::now();
            writer.flush();
            const auto flushElapsed = elapsedNanoseconds(flushBegin);
            const auto stats = (*deferred)->stats();
            if (!saved.load() || stats.persistedRevision != stats.dirtyRevision || stats.saveCount != i + 1) {
                return RHITestResult::fail("deferred flush did not persist its complete revision");
            }
            deferredSamples.push_back({enqueueElapsed, flushElapsed, stats});
        }
        const auto beforeBurst = (*deferred)->stats().saveCount;
        for (uint32_t i = 3; i < 6; ++i) {
            auto pipeline = createVariant(**deferred, i);
            if (!pipeline || (*pipeline)->pipelineCacheHit()) { return RHITestResult::fail("burst PSO creation failed"); }
            hashes[i] = (*pipeline)->psoHash();
            writer.request("deferred", (*deferred)->stats().dirtyRevision, [&] {
                ++burstSaves;
                return static_cast<bool>((*deferred)->save());
            });
        }
        writer.flush();
        if (burstSaves != 1 || (*deferred)->stats().saveCount != beforeBurst + 1 ||
            (*deferred)->stats().persistedRevision != (*deferred)->stats().dirtyRevision) {
            return RHITestResult::fail("a burst of PSO requests was not coalesced into one complete save");
        }

        auto beforeConcurrent = createVariant(**deferred, 6);
        if (!beforeConcurrent || (*beforeConcurrent)->pipelineCacheHit()) {
            return RHITestResult::fail("concurrent-save seed PSO creation failed");
        }
        hashes[6] = (*beforeConcurrent)->psoHash();
        const auto seedRevision = (*deferred)->stats().dirtyRevision;
        const auto failuresBeforeConcurrent = (*deferred)->stats().saveFailureCount;
        std::promise<void> saveEntered;
        auto entered = saveEntered.get_future();
        auto saveJob = std::async(std::launch::async, [&] {
            saveEntered.set_value();
            return (*deferred)->save();
        });
        entered.wait();
        bool observedSaveInProgress = false;
        const auto observeDeadline = Clock::now() + std::chrono::milliseconds(100);
        while (Clock::now() < observeDeadline && saveJob.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            if ((*deferred)->stats().saveInProgress) { observedSaveInProgress = true; break; }
            std::this_thread::yield();
        }
        for (uint32_t i = 7; i < kVariantCount; ++i) {
            auto pipeline = createVariant(**deferred, i);
            if (!pipeline || (*pipeline)->pipelineCacheHit()) {
                // Join before returning so no native cache can outlive its owner.
                saveJob.wait();
                return RHITestResult::fail("PSO creation concurrent with cache export failed");
            }
            hashes[i] = (*pipeline)->psoHash();
        }
        const bool concurrentSaveSucceeded = static_cast<bool>(saveJob.get());
        const auto afterConcurrent = (*deferred)->stats();
        if (observedSaveInProgress && concurrentSaveSucceeded &&
            (afterConcurrent.persistedRevision != seedRevision ||
             afterConcurrent.dirtyRevision <= afterConcurrent.persistedRevision)) {
            return RHITestResult::fail("concurrent save committed newer PSO identities outside its captured snapshot");
        }
        // Concurrent native growth can exhaust extraction retries. That attempt
        // must retain its dirty revision so a complete final save can recover.
        if (!(*deferred)->save()) { return RHITestResult::fail("final cache save after concurrent creation failed"); }
        const auto finalStats = (*deferred)->stats();
        if (!concurrentSaveSucceeded && finalStats.saveFailureCount != failuresBeforeConcurrent + 1) {
            return RHITestResult::fail("failed concurrent extraction did not record its retryable failure");
        }
        if (finalStats.saveInProgress || finalStats.dirtyRevision != finalStats.persistedRevision ||
            finalStats.storedPsoCount != initialStats.storedPsoCount + kVariantCount) {
            return RHITestResult::fail("concurrent export lost a newly created PSO or its dirty revision");
        }
        auto reloaded = context.device.createPipelineCache({.filePath = deferredPathString.c_str(), .saveOnDestroy = false});
        if (!reloaded || (*reloaded)->stats().loadStatus != PipelineCacheLoadStatus::Loaded ||
            (*reloaded)->stats().storedPsoCount != finalStats.storedPsoCount) {
            return RHITestResult::fail("deferred cache did not reload all persisted identities");
        }
        for (uint32_t i = 0; i < kVariantCount; ++i) {
            auto pipeline = createVariant(**reloaded, i);
            if (!pipeline || !(*pipeline)->pipelineCacheHit() || (*pipeline)->psoHash() != hashes[i]) {
                return RHITestResult::fail("reloaded cache missed a variant created during deferred/concurrent saving");
            }
        }

        // A failed atomic write must remain dirty and be retryable after repair.
        const auto blockedParent = root / "blocked-parent";
        { std::ofstream blocker(blockedParent); blocker << "owned test fixture"; }
        const std::string retryPath = (blockedParent / "retry.pso").string();
        auto retryCache = context.device.createPipelineCache({.filePath = retryPath.c_str(), .saveOnDestroy = false});
        if (!retryCache) { return RHITestResult::fail("retry cache creation failed"); }
        auto retryPipeline = createVariant(**retryCache, 0);
        if (!retryPipeline || (*retryCache)->save()) { return RHITestResult::fail("blocked cache write did not fail"); }
        const auto failedStats = (*retryCache)->stats();
        if (failedStats.saveFailureCount != 1 || failedStats.dirtyRevision == failedStats.persistedRevision ||
            failedStats.saveInProgress || failedStats.saveCount != 0) {
            return RHITestResult::fail("failed cache write discarded the dirty revision or retained its save lock");
        }
        std::filesystem::remove(blockedParent, fileError);
        if (fileError) { return RHITestResult::fail("could not repair owned cache fixture: " + fileError.message()); }
        std::filesystem::create_directory(blockedParent);
        if (!(*retryCache)->save() || (*retryCache)->stats().dirtyRevision != (*retryCache)->stats().persistedRevision) {
            return RHITestResult::fail("cache write did not recover after repairing its output directory");
        }

        const auto metricsPath = root / "save-metrics.json";
        std::ofstream metrics(metricsPath);
        metrics << "{\"fixture_available\":" << (fixtureAvailable ? "true" : "false")
                << ",\"fixture_loaded\":" << (initialStats.loadStatus == PipelineCacheLoadStatus::Loaded ? "true" : "false")
                << ",\"fixture_backend_bytes\":" << initialStats.backendDataSize << ",\"synchronous\":";
        writeSamples(metrics, syncSamples);
        metrics << ",\"deferred\":";
        writeSamples(metrics, deferredSamples);
        metrics << ",\"burst_save_count\":" << burstSaves.load()
                << ",\"observed_save_in_progress\":" << (observedSaveInProgress ? "true" : "false")
                << ",\"concurrent_save_succeeded\":" << (concurrentSaveSucceeded ? "true" : "false")
                << ",\"reloaded_variant_count\":" << kVariantCount << "}\n";
        if (!metrics) { return RHITestResult::fail("could not write cache-save timing evidence"); }
        return RHITestResult::pass("matched save/queue phase timings, burst coalescing, concurrent export, retry and all variants reloaded; " +
            metricsPath.string());
    }
};
METALLIC_REGISTER_RHI_TEST(PipelineCacheDeferredSaveTest);

class ShaderRegistryDeferredTeardownTest final : public RHITest {
public:
    ShaderRegistryDeferredTeardownTest()
    {
        type = RHITestType::Resource;
        name = "shader_registry_deferred_save_device_teardown";
    }

    RHITestResult run(RHITestContext& context) override
    {
        using namespace render;
        const uint64_t nonce = static_cast<uint64_t>(Clock::now().time_since_epoch().count());
        const auto root = std::filesystem::absolute(context.outputDirectory / "registry-save-teardown" / std::to_string(nonce));
        std::filesystem::create_directories(root);
        const std::string moduleName = "DeferredLifetime" + std::to_string(nonce);
        {
            // Nonce constants survive optimization in the vertex output, giving
            // this test a new source identity and a newly owned Registry group.
            std::ofstream source(root / (moduleName + ".slang"));
            source << "[shader(\"vertex\")] float4 lifetimeVertexMain(uint id : SV_VertexID) : SV_Position {\n"
                   << "return float4(float(id & 1u) + asfloat(" << (0x3f000000u | uint32_t(nonce & 0x007fffffu))
                   << "u), float((id >> 1u) & 1u) + asfloat(" << (0x3f000000u | uint32_t((nonce >> 23u) & 0x007fffffu))
                   << "u), 0.0, 1.0); }\n"
                      "[shader(\"fragment\")] float4 lifetimeFragmentMain() : SV_Target { return float4(0.0, 1.0, 0.0, 1.0); }\n";
            if (!source) { return RHITestResult::fail("could not write teardown source fixture"); }
        }
        const std::string sourceRoot = root.string();
        const std::string sourceCacheRoot = (root / "spirv").string();
        auto& registry = ShaderRegistry::instance();
        std::string log;
        auto vertexCode = registry.getShader({.moduleName = moduleName.c_str(), .entryPointName = "lifetimeVertexMain",
            .searchPath = sourceRoot.c_str()}, {.cacheDirectory = sourceCacheRoot.c_str()}, log);
        auto fragmentCode = registry.getShader({.moduleName = moduleName.c_str(), .entryPointName = "lifetimeFragmentMain",
            .searchPath = sourceRoot.c_str()}, {.cacheDirectory = sourceCacheRoot.c_str()}, log);
        if (!vertexCode || !fragmentCode) { return RHITestResult::fail("teardown source compilation failed: " + log); }
        const DeviceDesc desc{.applicationName = "Registry deferred save teardown", .enableValidation = context.enableValidation};
        uint64_t savedHash = 0;
        std::string group;
        {
            auto device = createDevice(desc);
            if (!device) { return RHITestResult::fail("first teardown device creation failed"); }
            auto vertex = registry.getShaderModule(**device, {.spirv = vertexCode->spirv});
            auto fragment = registry.getShaderModule(**device, {.spirv = fragmentCode->spirv});
            if (!vertex || !fragment) { return RHITestResult::fail("first teardown shader creation failed"); }
            auto pipeline = registry.getGraphicsPipeline(**device, {.vertexShader = {vertex->get()},
                .fragmentShader = {fragment->get()}, .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1});
            if (!pipeline || (*pipeline)->pipelineCacheHit()) { return RHITestResult::fail("first teardown PSO was not cold"); }
            savedHash = (*pipeline)->psoHash();
            auto stats = registry.pipelineCacheStats(**device);
            if (!stats || stats->size() != 1 || stats->front().cache.dirtyRevision == 0) {
                return RHITestResult::fail("teardown device did not enqueue a new dirty PSO");
            }
            group = stats->front().group;
            // Deliberately omit flush: destruction must drain the device worker.
        }
        auto device = createDevice(desc);
        if (!device) { return RHITestResult::fail("second teardown device creation failed"); }
        auto vertex = registry.getShaderModule(**device, {.spirv = vertexCode->spirv});
        auto fragment = registry.getShaderModule(**device, {.spirv = fragmentCode->spirv});
        if (!vertex || !fragment) { return RHITestResult::fail("second teardown shader creation failed"); }
        auto pipeline = registry.getGraphicsPipeline(**device, {.vertexShader = {vertex->get()},
            .fragmentShader = {fragment->get()}, .colorFormats = {Format::RGBA8Unorm}, .colorAttachmentCount = 1});
        auto stats = registry.pipelineCacheStats(**device);
        if (!pipeline || !(*pipeline)->pipelineCacheHit() || (*pipeline)->psoHash() != savedHash ||
            !stats || stats->size() != 1 || stats->front().group != group ||
            stats->front().cache.loadStatus != PipelineCacheLoadStatus::Loaded || stats->front().cache.storedPsoCount != 1) {
            return RHITestResult::fail("device teardown did not persist a pending Registry cache before native destruction");
        }
        return RHITestResult::pass("pending worker save drained at device teardown and an independent device reloaded its PSO");
    }
};
METALLIC_REGISTER_RHI_TEST(ShaderRegistryDeferredTeardownTest);

} // namespace
} // namespace metallic::tests

#include "RHITest.h"
#include "harness/Fixtures.h"
#include "Runtime/Render/Core/NativeShaderWarmup.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ShaderRequests.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/RenderPass/BuiltinPass/BuiltinPassCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

namespace metallic::tests {
namespace {

using namespace render;

// Observe the actual Vulkan boundary, including calls satisfied by a driver
// binary cache. Registry statistics alone cannot prove that it was bypassed.
class WarmupNativeCalls {
public:
    explicit WarmupNativeCalls(Device& device)
        : functions_(*const_cast<VolkDeviceTable*>(vulkan::nativeDevice(device).functions)),
          compute_(functions_.vkCreateComputePipelines), graphics_(functions_.vkCreateGraphicsPipelines)
    {
        active_ = this;
        functions_.vkCreateComputePipelines = createCompute;
        functions_.vkCreateGraphicsPipelines = createGraphics;
    }

    ~WarmupNativeCalls()
    {
        functions_.vkCreateComputePipelines = compute_;
        functions_.vkCreateGraphicsPipelines = graphics_;
        active_ = nullptr;
    }

    std::atomic<uint32_t> computeCalls{0};
    std::atomic<uint32_t> graphicsCalls{0};
    std::atomic<uint32_t> activeComputeCalls{0};
    std::atomic<uint32_t> completedComputeCalls{0};
    uint32_t peakComputeCalls = 0;
    bool rendezvousCompute = false;
    bool failCompute = false;

private:
    static VKAPI_ATTR VkResult VKAPI_CALL createCompute(VkDevice device, VkPipelineCache cache,
        uint32_t count, const VkComputePipelineCreateInfo* infos,
        const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
    {
        auto& self = *active_;
        self.computeCalls.fetch_add(count);
        {
            std::unique_lock lock(self.rendezvousMutex_);
            const auto active = self.activeComputeCalls.fetch_add(1) + 1;
            self.peakComputeCalls = std::max(self.peakComputeCalls, active);
            self.rendezvousCondition_.notify_all();
            if (self.rendezvousCompute) {
                // A bounded rendezvous proves overlap without hanging the test
                // when a regression serializes the registry or driver boundary.
                self.rendezvousCondition_.wait_for(lock, std::chrono::seconds(2), [&] {
                    return self.peakComputeCalls >= 2;
                });
            }
        }
        const VkResult result = self.failCompute ? VK_ERROR_OUT_OF_HOST_MEMORY
            : self.compute_(device, cache, count, infos, allocator, pipelines);
        self.activeComputeCalls.fetch_sub(1);
        self.completedComputeCalls.fetch_add(1);
        return result;
    }

    static VKAPI_ATTR VkResult VKAPI_CALL createGraphics(VkDevice device, VkPipelineCache cache,
        uint32_t count, const VkGraphicsPipelineCreateInfo* infos,
        const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
    {
        auto& self = *active_;
        self.graphicsCalls.fetch_add(count);
        return self.graphics_(device, cache, count, infos, allocator, pipelines);
    }

    inline static WarmupNativeCalls* active_ = nullptr;
    VolkDeviceTable& functions_;
    PFN_vkCreateComputePipelines compute_;
    PFN_vkCreateGraphicsPipelines graphics_;
    std::mutex rendezvousMutex_;
    std::condition_variable rendezvousCondition_;
};

class WarmupJobsEnvironment {
public:
    explicit WarmupJobsEnvironment(const char* value)
    {
        if (const char* previous = std::getenv("METALLIC_SHADER_WARMUP_JOBS")) { previous_ = previous; }
        set(value);
    }

    ~WarmupJobsEnvironment() { set(previous_ ? previous_->c_str() : nullptr); }

private:
    static void set(const char* value)
    {
#ifdef _WIN32
        _putenv_s("METALLIC_SHADER_WARMUP_JOBS", value ? value : "");
#else
        if (value) { setenv("METALLIC_SHADER_WARMUP_JOBS", value, 1); }
        else { unsetenv("METALLIC_SHADER_WARMUP_JOBS"); }
#endif
    }

    std::optional<std::string> previous_;
};

ShaderRequest bufferRequest(const char* entry)
{
    return {.module = builtin_pass::kRenderGraphBufferShaderModuleName, .entry = entry,
        .descriptorHeapMode = SlangDescriptorHeapMode::Mapped};
}

Result<bench::TestDevice> warmupTestDevice(RHITestContext& context)
{
    auto device = bench::createTestDevice(context, {.applicationName = "Native shader warmup test",
        .enableValidation = context.enableValidation, .enableBindlessDescriptorHeap = true}, true);
    if (device && !(*device)->capabilities().bindlessDescriptorHeap) { return makeError(Error::Unsupported); }
    return device;
}

bool reusedWithoutNativeWork(const ComputePipeline& pipeline)
{
    const auto stats = pipeline.creationStats();
    return stats.nativeObjectReused && stats.nativeCreateTimeNanoseconds == 0 &&
        stats.driverDurationNanoseconds == 0 && !stats.feedbackValid;
}

class NativeShaderWarmupReuseTest final : public RHITest {
public:
    NativeShaderWarmupReuseTest()
    {
        type = RHITestType::Resource;
        name = "native_shader_warmup_runtime_reuse_and_push_abi";
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto device = warmupTestDevice(context);
        if (!device) { return device.error() == Error::Unsupported
            ? RHITestResult::skip("bindless native warmup unavailable")
            : RHITestResult::fail("could not create native warmup test device"); }
        auto& registry = ShaderRegistry::instance();
        // Set this before warming: no incidental LRU owner may keep the result
        // alive after the warmup's temporary module/pipeline wrappers disappear.
        if (!registry.setNativePipelineCacheCapacity(**device, 0)) {
            return RHITestResult::fail("could not disable native LRU retention");
        }
        WarmupNativeCalls calls(**device);
        const std::array requests{
            bufferRequest(builtin_pass::kRenderGraphBufferWriteEntryPoint),
            bufferRequest(builtin_pass::kRenderGraphBufferCopyEntryPoint)};
        std::string log;
        auto warmed = warmupNativeShaders(**device, requests, log);
        if (!warmed || warmed->nativeCreateCount != 2 || warmed->retainedPipelineCount != 2 ||
            warmed->nativeObjectReuseCount != 0 || calls.computeCalls != 2 || !warmed->sourceBatchUsed ||
            warmed->sourceAcquireTimeNanoseconds == 0) {
            return RHITestResult::fail("warmup did not create and retain both real buffer PSOs: " + log);
        }
        auto repeated = warmupNativeShaders(**device, requests, log);
        if (!repeated || repeated->nativeCreateCount != 0 || repeated->nativeObjectReuseCount != 2 ||
            repeated->retainedPipelineCount != 2 || calls.computeCalls != 2 || !repeated->sourceBatchUsed) {
            return RHITestResult::fail("repeated warmup recompiled a retained native PSO: " + log);
        }
        if (repeated->sourceEpochValidated && (repeated->sourceCacheHits != requests.size() ||
            repeated->sourceCompilerExecutions != 0)) {
            return RHITestResult::fail("validated repeated native source batch missed the warmed disk cache: " + log);
        }
        const std::array duplicateRequests{requests[0], requests[1], requests[0]};
        auto duplicates = warmupNativeShaders(**device, duplicateRequests, log);
        if (!duplicates || duplicates->sourceCount != duplicateRequests.size() || !duplicates->sourceBatchUsed ||
            duplicates->nativeCreateCount != 0 || duplicates->nativeObjectReuseCount != duplicateRequests.size() ||
            duplicates->retainedPipelineCount != 2 || calls.computeCalls != 2) {
            return RHITestResult::fail("duplicate source requests lost their native recipe order or reuse: " + log);
        }
        auto definedRequest = requests[0];
        definedRequest.defines.emplace_back("NATIVE_WARMUP_IDENTITY_TEST", "1");
        auto mappedStridesAlias = requests[0];
        mappedStridesAlias.descriptorHeapStrides = {.resource = 32, .sampler = 16};
        auto pathAlias = requests[0];
        pathAlias.searchPath.insert(pathAlias.searchPath.find_last_of("/\\") + 1, "./");
        auto optimizationAlias = requests[0];
        const ShaderRequestView defaultPolicy(requests[0]);
        optimizationAlias.optimizationLevel = slangShaderOptimizationLevel(defaultPolicy.desc());
        auto descriptorAlias = requests[0];
        descriptorAlias.descriptorHeapMode = SlangDescriptorHeapMode::Default;
        const ShaderRequestView defaultDescriptor(descriptorAlias);
        // Under a native-descriptor test environment, Default is not an alias
        // of the explicitly mapped source, so retain the explicit mapped policy.
        if (slangUsesNativeDescriptorHeap(defaultDescriptor.desc())) { descriptorAlias = requests[0]; }
        const std::array identityRequests{requests[0], requests[1], definedRequest, requests[0],
            mappedStridesAlias, pathAlias, optimizationAlias, descriptorAlias};
        auto identityBatch = registry.getWarmupShadersWithOverrides(identityRequests, {}, log);
        if (!identityBatch || identityBatch->artifactOverrideActive ||
            identityBatch->sources.shaders.size() != identityRequests.size() ||
            identityBatch->sources.existingCacheHits + identityBatch->sources.compilerExecutions != 3 ||
            identityBatch->sources.shaders[0].spirv != identityBatch->sources.shaders[3].spirv ||
            identityBatch->sources.shaders[0].spirv != identityBatch->sources.shaders[4].spirv ||
            identityBatch->sources.shaders[0].spirv != identityBatch->sources.shaders[5].spirv ||
            identityBatch->sources.shaders[0].spirv != identityBatch->sources.shaders[6].spirv ||
            identityBatch->sources.shaders[0].spirv != identityBatch->sources.shaders[7].spirv) {
            return RHITestResult::fail("source batch lost effective policy aliases, distinct defines, or duplicate mapping: " + log);
        }
#ifdef _WIN32
        {
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
            auto explicitPolicy = requests[0];
            explicitPolicy.optimizationLevel = SlangShaderOptimizationLevel::O2;
            const std::array policyAliases{requests[0], explicitPolicy};
            bool changedPolicy = false;
            const SlangWarmupBatchOptions driftOptions{.onProgress = [&](const SlangWarmupBatchProgress& progress) {
                if (!changedPolicy && progress.phase && std::string_view(progress.phase) == "preparing") {
                    _putenv_s("METALLIC_SLANG_OPTIMIZATION", "O3");
                    changedPolicy = true;
                }
            }};
            auto drifted = registry.getWarmupShadersWithOverrides(policyAliases, driftOptions, log);
            if (drifted || !changedPolicy || log.empty() || calls.computeCalls != 2) {
                return RHITestResult::fail("policy drift returned an incorrectly deduplicated source alias");
            }
            log.clear();
        }
#endif
        for (const auto& request : requests) {
            const ShaderRequestView view(request);
            auto code = registry.getShader(view.desc(), log);
            if (!code) { return RHITestResult::fail("runtime buffer source acquisition failed: " + log); }
            auto module = registry.getShaderModule(**device, {.spirv = code->spirv});
            if (!module) { return RHITestResult::fail("runtime buffer module acquisition failed"); }
            const ComputePipelineDesc desc{.computeShader = {module->get(), "main"},
                .usesBindlessHeap = true,
                .bindlessUserPushDataSize = sizeof(builtin_pass::RenderGraphBufferUserPush)};
            const uint32_t before = calls.computeCalls;
            auto runtime = registry.getComputePipeline(**device, desc);
            if (!runtime || !reusedWithoutNativeWork(**runtime) || calls.computeCalls != before) {
                return RHITestResult::fail("runtime buffer ABI did not reuse the warmed PSO");
            }
            auto changedPush = desc;
            changedPush.bindlessUserPushDataSize += sizeof(uint32_t);
            auto distinct = registry.getComputePipeline(**device, changedPush);
            if (!distinct || (*distinct)->creationStats().nativeObjectReused || calls.computeCalls != before + 1 ||
                (*distinct)->psoHash() == (*runtime)->psoHash() ||
                vulkan::nativePipeline(**distinct).pipeline == vulkan::nativePipeline(**runtime).pipeline) {
                return RHITestResult::fail("changed push ABI aliased the warmed native PSO");
            }
            const ShaderSpecializationConstant missing{std::numeric_limits<uint32_t>::max(),
                ShaderSpecializationType::Int32, 1};
            auto malformed = desc;
            malformed.computeShader.specializationConstants = {&missing, 1};
            if (registry.getComputePipeline(**device, malformed) || calls.computeCalls != before + 1) {
                return RHITestResult::fail("warm native reuse bypassed specialization declaration validation");
            }
        }
        if (calls.graphicsCalls != 0) { return RHITestResult::fail("compute warmup created a graphics pipeline"); }
        return RHITestResult::pass("real buffer warmup survives capacity zero; runtime reuses native objects without Vulkan calls; changed push and invalid specialization cannot alias");
    }
};

class NativeShaderWarmupConcurrencyTest final : public RHITest {
public:
    NativeShaderWarmupConcurrencyTest()
    {
        type = RHITestType::Resource;
        name = "native_shader_warmup_parallel_creation_and_failure_join";
    }

    RHITestResult run(RHITestContext& context) override
    {
        struct Scenario {
            const char* jobs;
            uint32_t expectedWorkers;
            bool fail;
            bool duplicates;
        };
        const std::array scenarios{Scenario{"1", 1, false, false}, Scenario{"2", 2, false, false},
            Scenario{"2", 2, true, false}, Scenario{"2", 2, false, true}};
        for (const auto& scenario : scenarios) {
            WarmupJobsEnvironment environment(scenario.jobs);
            auto device = warmupTestDevice(context);
            if (!device) { return device.error() == Error::Unsupported
                ? RHITestResult::skip("bindless native warmup unavailable")
                : RHITestResult::fail("could not create native warmup concurrency test device"); }
            auto& registry = ShaderRegistry::instance();
            if (!registry.setNativePipelineCacheCapacity(**device, 0)) {
                return RHITestResult::fail("could not disable native LRU retention");
            }
            WarmupNativeCalls calls(**device);
            calls.rendezvousCompute = scenario.expectedWorkers > 1 && !scenario.duplicates;
            calls.failCompute = scenario.fail;
            const auto write = bufferRequest(builtin_pass::kRenderGraphBufferWriteEntryPoint);
            const auto copy = bufferRequest(builtin_pass::kRenderGraphBufferCopyEntryPoint);
            std::vector requests{write, copy};
            if (scenario.duplicates) {
                requests.push_back(write);
                requests.push_back(copy);
            }
            std::string log;
            auto warmed = warmupNativeShaders(**device, requests, log);
            if (calls.activeComputeCalls != 0 || calls.completedComputeCalls != calls.computeCalls ||
                calls.peakComputeCalls > scenario.expectedWorkers || calls.peakComputeCalls == 0 ||
                (!scenario.duplicates && calls.peakComputeCalls != scenario.expectedWorkers)) {
                return RHITestResult::fail("native warmup did not honor worker concurrency or join all native calls: " + log);
            }
            if (scenario.fail) {
                if (warmed || warmed.error() != Error::OutOfMemory || log.empty() || calls.computeCalls != 2) {
                    return RHITestResult::fail("parallel native creation failures did not reject warmup: " + log);
                }
                calls.failCompute = false;
                log.clear();
                warmed = warmupNativeShaders(**device, requests, log);
                if (calls.computeCalls != 4 || calls.activeComputeCalls != 0 || calls.completedComputeCalls != 4) {
                    return RHITestResult::fail("parallel native failure poisoned retry or left active calls: " + log);
                }
            } else if (calls.computeCalls != 2) {
                return RHITestResult::fail("duplicate warmup requests created duplicate native PSOs: " + log);
            }
            if (!warmed || warmed->nativeWorkerCount != scenario.expectedWorkers ||
                warmed->nativeCreateCount != 2 || warmed->nativeObjectReuseCount != (scenario.duplicates ? 2u : 0u) ||
                warmed->retainedPipelineCount != 2) {
                return RHITestResult::fail("parallel native warmup lost results, duplicate reuse, or retained PSOs: " + log);
            }
        }
        return RHITestResult::pass("worker overrides control real Vulkan overlap; duplicates create once; parallel failures join and retry successfully");
    }
};

class RejectedWarmupSource final : public ShaderArtifactOverride {
public:
    std::optional<Result<ShaderArtifactOverrideResult>> resolve(const SlangShaderDesc&, std::string& log) override
    {
        log = "intentional native warmup source acquisition failure";
        return Result<ShaderArtifactOverrideResult>(makeError(Error::Failure));
    }
};

class NativeShaderWarmupFailureTest final : public RHITest {
public:
    NativeShaderWarmupFailureTest()
    {
        type = RHITestType::Resource;
        name = "native_shader_warmup_skips_and_failures";
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto device = warmupTestDevice(context);
        if (!device) { return device.error() == Error::Unsupported
            ? RHITestResult::skip("bindless native warmup unavailable")
            : RHITestResult::fail("could not create native warmup failure test device"); }
        auto& registry = ShaderRegistry::instance();
        WarmupNativeCalls calls(**device);
        const std::array skipped{
            ShaderRequest{.module = "Features/Samples/Triangle", .entry = "triangleVertexMain",
                .descriptorHeapMode = SlangDescriptorHeapMode::Mapped},
            ShaderRequest{.module = "ShaderSpecializationProbe", .entry = "specializationProbeMain",
                // The immutable batch must also capture the parent directory
                // containing the fixture's ../TestResourceParameters.h include.
                .searchPaths = {PROJECT_SOURCE_DIR "/tests/rhi"},
                .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
                .descriptorHeapMode = SlangDescriptorHeapMode::Mapped}};
        std::string log;
        auto skips = warmupNativeShaders(**device, skipped, log);
        if (!skips || skips->skippedGraphicsCount != 1 || skips->skippedUnownedABICount != 1 ||
            skips->nativeCreateCount != 0 || skips->retainedPipelineCount != 0 ||
            calls.computeCalls != 0 || calls.graphicsCalls != 0) {
            return RHITestResult::fail("graphics/unknown ABI requests were not explicitly skipped: " + log);
        }
        const std::array requests{bufferRequest(builtin_pass::kRenderGraphBufferWriteEntryPoint)};
        // Even a valid earlier source must not create a PSO before every source
        // has been accepted. This catches an accidental per-request fallback.
        auto invalid = requests.front();
        invalid.entry = "missingNativeWarmupEntry";
        const std::array rejectedBatch{requests.front(), invalid};
        auto rejected = warmupNativeShaders(**device, rejectedBatch, log);
        if (rejected || log.empty() || calls.computeCalls != 0 || calls.graphicsCalls != 0) {
            return RHITestResult::fail("partial source batch created native work before source validation completed");
        }
        log.clear();
        {
            auto overrideScope = registry.installArtifactOverrideForTesting(std::make_shared<RejectedWarmupSource>());
            if (!overrideScope) { return RHITestResult::fail("could not install source rejection fixture"); }
            auto failed = warmupNativeShaders(**device, requests, log);
            if (failed || log.find("intentional native warmup source acquisition failure") == std::string::npos ||
                calls.computeCalls != 0) {
                return RHITestResult::fail("source acquisition failure was reported as successful warmup");
            }
        }
        calls.failCompute = true;
        log.clear();
        auto failed = warmupNativeShaders(**device, requests, log);
        if (failed || failed.error() != Error::OutOfMemory || log.empty() || calls.computeCalls != 1) {
            return RHITestResult::fail("native creation failure was reported as successful warmup");
        }
        calls.failCompute = false;
        auto retry = warmupNativeShaders(**device, requests, log);
        if (!retry || retry->nativeCreateCount != 1 || retry->retainedPipelineCount != 1 || calls.computeCalls != 2) {
            return RHITestResult::fail("failed warmup poisoned a subsequent native retry: " + log);
        }
        return RHITestResult::pass("graphics/unknown ABI skips are explicit; source and Vulkan failures reject success, and native failure can be retried");
    }
};

class SpecializationWarmupArtifact final : public ShaderArtifactOverride {
public:
    explicit SpecializationWarmupArtifact(ShaderCompileResult code) : code_(std::move(code)) {}

    std::optional<Result<ShaderArtifactOverrideResult>> resolve(const SlangShaderDesc& desc, std::string&) override
    {
        if (!desc.moduleName || std::string_view(desc.moduleName) != "Features/Debug/GPUProbe") { return std::nullopt; }
        ++resolveCalls;
        if (scopeToRemove) {
            scopeToRemove->reset();
            scopeToRemove = nullptr;
        }
        // Reuse the existing scalar specialization fixture through the testing
        // override. The separate buffer test exercises genuine production code.
        ShaderArtifactOverrideResult result;
        result.shader = code_;
        result.identity.moduleName = "Tests/NativeWarmup/ShaderSpecializationProbe";
        result.identity.sourceEntryPoint = desc.entryPointName;
        result.identity.entryPoint = "main";
        result.identity.descriptorHeapMode = "mapped";
        result.identity.optimizationLevel = "O1";
        result.identity.debugMode = "release";
        result.identity.spirvPostprocess = "none";
        return Result<ShaderArtifactOverrideResult>(std::move(result));
    }

    ShaderArtifactOverrideScope* scopeToRemove = nullptr;
    size_t resolveCalls = 0;

private:
    ShaderCompileResult code_;
};

class NativeShaderWarmupSpecializationTest final : public RHITest {
public:
    NativeShaderWarmupSpecializationTest()
    {
        type = RHITestType::Resource;
        name = "native_shader_warmup_specialization_identity";
    }

    RHITestResult run(RHITestContext& context) override
    {
        auto& registry = ShaderRegistry::instance();
        std::string log;
        auto code = registry.getShader({.moduleName = "ShaderSpecializationProbe",
            .entryPointName = "specializationProbeMain", .searchPath = PROJECT_SOURCE_DIR "/tests/rhi/shaders",
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped,
            .optimizationLevel = SlangShaderOptimizationLevel::O1,
            .postprocessMode = SlangShaderPostprocessMode::None}, log);
        if (!code) { return RHITestResult::fail("existing specialization fixture compilation failed: " + log); }
        // Observe and validate the actual scalar declaration, as in the existing
        // specialization RHI tests; Slang auto IDs are not a stable source API.
        std::optional<uint32_t> intId;
        for (size_t offset = 5; offset < code->spirv.size();) {
            const uint32_t count = code->spirv[offset] >> 16, opcode = code->spirv[offset] & 0xffffu;
            if (!count || count > code->spirv.size() - offset) {
                return RHITestResult::fail("malformed specialization fixture SPIR-V");
            }
            if (opcode == 71 && count == 4 && code->spirv[offset + 2] == 1) {
                const ShaderSpecializationConstant declaration{code->spirv[offset + 3], ShaderSpecializationType::Int32, 0};
                if (detail::validateShaderSpecializationModule(code->spirv, {&declaration, 1}, log)) {
                    if (intId) { return RHITestResult::fail("ambiguous specialization fixture Int32 declaration"); }
                    intId = declaration.id;
                }
            }
            offset += count;
        }
        if (!intId) { return RHITestResult::fail("specialization fixture lost its Int32 declaration"); }
        auto device = warmupTestDevice(context);
        if (!device) { return device.error() == Error::Unsupported
            ? RHITestResult::skip("bindless native warmup unavailable")
            : RHITestResult::fail("could not create specialization warmup test device"); }
        WarmupNativeCalls calls(**device);
        auto artifact = std::make_shared<SpecializationWarmupArtifact>(*code);
        auto overrideScope = registry.installArtifactOverrideForTesting(artifact);
        if (!overrideScope) { return RHITestResult::fail("could not install specialization fixture override"); }
        // Removal during the first callback must not split the acquisition
        // between this artifact and the ordinary production shader.
        artifact->scopeToRemove = &*overrideScope;
        const ShaderRequest request{.module = "Features/Debug/GPUProbe", .entry = "probe",
            .descriptorHeapMode = SlangDescriptorHeapMode::Mapped};
        const std::array requests{request, request};
        auto warmed = warmupNativeShaders(**device, requests, log);
        if (!warmed || warmed->nativeCreateCount != 1 || warmed->nativeObjectReuseCount != 1 || calls.computeCalls != 1 ||
            warmed->sourceBatchUsed || warmed->sourceEpochValidated || warmed->sourceCacheHits != 0 ||
            warmed->sourceCompilerExecutions != 0 || artifact->resolveCalls != 2 || bool(*overrideScope)) {
            return RHITestResult::fail("specialization fixture warmup failed: " + log);
        }
        auto module = registry.getShaderModule(**device, {.spirv = code->spirv});
        if (!module) { return RHITestResult::fail("could not acquire specialization fixture module"); }
        ComputePipelineDesc desc{.computeShader = {module->get(), "main"}, .usesBindlessHeap = true,
            .bindlessUserPushDataSize = sizeof(GPUBufferSpan)};
        auto original = registry.getComputePipeline(**device, desc);
        if (!original || !reusedWithoutNativeWork(**original) || calls.computeCalls != 1) {
            return RHITestResult::fail("runtime default specialization did not reuse warmup");
        }
        ShaderSpecializationConstant value{*intId, ShaderSpecializationType::Int32, 1};
        desc.computeShader.specializationConstants = {&value, 1};
        auto first = registry.getComputePipeline(**device, desc);
        value.bits = 2;
        auto second = registry.getComputePipeline(**device, desc);
        if (!first || !second || calls.computeCalls != 3 ||
            (*first)->creationStats().nativeObjectReused || (*second)->creationStats().nativeObjectReused ||
            (*first)->psoHash() == (*original)->psoHash() || (*second)->psoHash() == (*first)->psoHash() ||
            vulkan::nativePipeline(**first).pipeline == vulkan::nativePipeline(**second).pipeline) {
            return RHITestResult::fail("valid specialization values aliased a warmed native PSO");
        }
        value.bits = 1;
        auto repeated = registry.getComputePipeline(**device, desc);
        if (!repeated || !reusedWithoutNativeWork(**repeated) || calls.computeCalls != 3 ||
            vulkan::nativePipeline(**repeated).pipeline != vulkan::nativePipeline(**first).pipeline) {
            return RHITestResult::fail("matching specialization tuple did not reuse its native object");
        }
        return RHITestResult::pass("default warmup reuse, distinct actual scalar specialization values, and same-value runtime reuse verified at the Vulkan call boundary");
    }
};

METALLIC_REGISTER_RHI_TEST(NativeShaderWarmupReuseTest);
METALLIC_REGISTER_RHI_TEST(NativeShaderWarmupConcurrencyTest);
METALLIC_REGISTER_RHI_TEST(NativeShaderWarmupFailureTest);
METALLIC_REGISTER_RHI_TEST(NativeShaderWarmupSpecializationTest);

} // namespace
} // namespace metallic::tests

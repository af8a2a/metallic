#include "Runtime/Render/Core/NativeShaderWarmup.h"
#include "Runtime/Render/Core/NativeShaderWarmupRequests.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "../../../../Tools/ShaderWarmupRequests.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

struct RetainedWarmupPipeline
{
    uint64_t psoHash = 0;
    std::vector<ShaderSpecializationConstant> specialization;
    PreparedExecution execution;
};

struct NativeWarmupState
{
    std::mutex mutex;
    std::vector<RetainedWarmupPipeline> pipelines;
};

struct NativeWarmupTask
{
    ShaderModule* module = nullptr;
    std::string name;
    NativeComputeWarmupVariant variant;
    size_t variantIndex = 0;
    size_t variantCount = 0;
    size_t estimatedCost = 0;
};

struct NativeWarmupTaskResult
{
    std::unique_ptr<ComputePipeline> pipeline;
    std::optional<Error> error;
    std::exception_ptr exception;
};

struct ShaderRequirements
{
    bool compute = false;
    bool nativeDescriptorHeap = false;
    bool float16 = false;
    bool float64 = false;
    bool int16 = false;
    bool int64 = false;
    bool rayQuery = false;
    bool clusterAccelerationStructure = false;
    bool subgroupBallotArithmetic = false;
    bool subgroupShuffle = false;
    bool maximalReconvergence = false;
    bool int64Atomics = false;
    bool integerDotProduct = false;
    bool cooperativeVector = false;
};

Result<ShaderRequirements> inspectShader(std::span<const uint32_t> words)
{
    if (words.size() < 5 || words[0] != 0x07230203u) { return makeError(Error::InvalidArgument); }
    ShaderRequirements requirements;
    std::unordered_set<uint32_t> int64Types;
    size_t entryCount = 0;
    // Inspect emitted operations, not requested Slang capabilities: a shared
    // source request can advertise ray/subgroup support without using it.
    for (size_t offset = 5; offset < words.size();) {
        const uint32_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 0xffffu;
        if (!count || count > words.size() - offset) { return makeError(Error::InvalidArgument); }
        if (opcode == 15u) { // OpEntryPoint
            if (count < 4) { return makeError(Error::InvalidArgument); }
            ++entryCount;
            requirements.compute = words[offset + 1] == 5u; // GLCompute
        }
        if (opcode == 21u && count == 4 && words[offset + 2] == 64u) { // OpTypeInt
            int64Types.insert(words[offset + 1]);
        }
        if (opcode == 16u) { // OpExecutionMode
            if (count < 3u) { return makeError(Error::InvalidArgument); }
            requirements.maximalReconvergence |= words[offset + 2] == 6023u; // MaximallyReconvergesKHR
        }
        if (opcode == 17u && count == 2u) { // Actual SPIR-V OpCapability
            const uint32_t capability = words[offset + 1];
            // These are requirements in the emitted module, not the broader
            // capabilities Slang was permitted to use for its source request.
            requirements.nativeDescriptorHeap |= capability == 5128u; // DescriptorHeapEXT
            requirements.float16 |= capability == 9u;
            requirements.float64 |= capability == 10u;
            requirements.int64 |= capability == 11u;
            requirements.int16 |= capability == 22u;
        }
        requirements.rayQuery |= (opcode >= 4473u && opcode <= 4479u) ||
            (opcode >= 6016u && opcode <= 6032u);
        requirements.clusterAccelerationStructure |= opcode == 5345u; // OpRayQueryGetClusterIdNV
        requirements.subgroupBallotArithmetic |= (opcode >= 339u && opcode <= 344u) ||
            (opcode >= 349u && opcode <= 364u);
        requirements.subgroupShuffle |= opcode >= 345u && opcode <= 348u;
        requirements.integerDotProduct |= opcode >= 4450u && opcode <= 4455u;
        requirements.cooperativeVector |= (opcode >= 5289u && opcode <= 5292u) ||
            opcode == 5302u || opcode == 5303u;
        // AtomicLoad and returning integer atomic operations carry result type.
        // The authored kernels use returning atomics for their uint64 buffers.
        if ((opcode == 227u || (opcode >= 229u && opcode <= 242u)) && count >= 3) {
            requirements.int64Atomics |= int64Types.contains(words[offset + 1]);
        }
        offset += count;
    }
    if (entryCount != 1) { return makeError(Error::InvalidArgument); }
    return requirements;
}

const char* unsupportedReason(const ShaderRequirements& shader, const DeviceCapabilities& caps,
    const vulkan::NativeDevice& nativeDevice)
{
    if (!caps.bindlessDescriptorHeap) { return "bindless descriptor heap is unavailable"; }
    if (shader.nativeDescriptorHeap && (!nativeDevice.descriptorHeapEnabled || !nativeDevice.shaderUntypedPointersEnabled)) {
        return "emitted native descriptor heap requires shader untyped pointers";
    }
    if (shader.float16 && !nativeDevice.shaderFloat16Enabled) { return "emitted Float16 capability is unavailable"; }
    if (shader.float64 && !nativeDevice.shaderFloat64Enabled) { return "emitted Float64 capability is unavailable"; }
    if (shader.int16 && !nativeDevice.shaderInt16Enabled) { return "emitted Int16 capability is unavailable"; }
    if (shader.int64 && !nativeDevice.shaderInt64Enabled) { return "emitted Int64 capability is unavailable"; }
    if (shader.rayQuery && (!caps.rayQuery || !caps.rayTracingAccelerationStructure)) {
        return "emitted ray-query operations require ray query and acceleration structures";
    }
    if (shader.clusterAccelerationStructure && !caps.clusterAccelerationStructure) {
        return "emitted cluster-ray-query operations require cluster acceleration structures";
    }
    if (shader.subgroupBallotArithmetic && !caps.computeSubgroupBallotArithmetic) {
        return "emitted subgroup ballot/arithmetic operations are unavailable";
    }
    if (shader.subgroupShuffle && !caps.computeSubgroupShuffle) {
        return "emitted subgroup shuffle operations are unavailable";
    }
    if (shader.maximalReconvergence && !caps.shaderMaximalReconvergence) {
        return "emitted maximal reconvergence execution mode is unavailable";
    }
    if (shader.int64Atomics && !caps.shaderBufferInt64Atomics) { return "emitted 64-bit buffer atomics are unavailable"; }
    if (shader.integerDotProduct && !caps.shaderIntegerDotProduct) { return "emitted integer dot products are unavailable"; }
    if (shader.cooperativeVector && !caps.cooperativeVector) { return "emitted cooperative vectors are unavailable"; }
    return nullptr;
}

uint64_t elapsedNanoseconds(std::chrono::steady_clock::time_point begin)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count());
}

} // namespace

Result<NativeShaderWarmupReport> warmupNativeShaders(Device& device,
    std::span<const ShaderRequest> requests, std::string& log)
{
    static const char stateKey = 0;
    auto shared = device.sharedState(&stateKey, []() -> Result<std::shared_ptr<void>> {
        return std::make_shared<NativeWarmupState>();
    });
    if (!shared) { return makeError(shared.error()); }
    auto state = std::static_pointer_cast<NativeWarmupState>(*shared);
    // The native registry still arbitrates concurrent runtime requests. This
    // lock only serializes warmup bookkeeping and avoids duplicate strong pins.
    std::lock_guard lock(state->mutex);
    const auto begin = std::chrono::steady_clock::now();
    NativeShaderWarmupReport report;
    report.sourceCount = requests.size();
    auto& registry = ShaderRegistry::instance();
    // Keep persistence out of source/native compilation. This scope outlives
    // all worker joins, including error exits, and coalesces each cache group.
    auto cacheSaveDeferral = registry.deferPipelineCacheSaves(device);
    if (!cacheSaveDeferral) { return makeError(cacheSaveDeferral.error()); }
    const auto nativeDevice = vulkan::nativeDevice(device);
    const bool nativeDescriptorHeapAvailable = nativeDevice.descriptorHeapEnabled && nativeDevice.shaderUntypedPointersEnabled;
    // Own every string/span until acquisition completes. Native heap requests
    // must use this device's ABI, even when an offline batch already ran.
    std::vector<ShaderRequest> deviceRequests;
    deviceRequests.reserve(requests.size());
    for (const auto& request : requests) {
        const ShaderRequestView view(request);
        auto owned = request;
        if (slangUsesNativeDescriptorHeap(view.desc())) {
            owned.descriptorHeapStrides = device.descriptorHeapShaderStrides();
            if (!nativeDescriptorHeapAvailable || !owned.descriptorHeapStrides.resource || !owned.descriptorHeapStrides.sampler) {
                // This device cannot compile the native heap ABI. Do not fall
                // back to process-global strides belonging to another device.
                const auto variants = nativeComputeWarmupVariants(request);
                report.skippedUnsupportedCount += std::max(size_t{1}, variants.size());
                spdlog::info("[Native shader warmup] skip {}.{}: native descriptor heap, untyped pointers, or descriptor strides are unavailable", request.module, request.entry);
                continue;
            }
        }
        deviceRequests.push_back(std::move(owned));
    }
    std::vector<ShaderCompileResult> shaders;
    size_t configuredJobs = std::clamp(std::thread::hardware_concurrency(), 1u, 12u);
    if (!deviceRequests.empty()) {
        if (const char* value = std::getenv("METALLIC_SHADER_WARMUP_JOBS"); value && value[0]) {
            const std::string_view text(value);
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), configuredJobs);
            if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || configuredJobs == 0) {
                log += "METALLIC_SHADER_WARMUP_JOBS requires a positive integer\n";
                return makeError(Error::InvalidArgument);
            }
        }
        const size_t jobs = std::min(configuredJobs, deviceRequests.size());
        const auto acquireBegin = std::chrono::steady_clock::now();
        spdlog::info("[Native shader warmup] acquiring {} device-qualified sources (workers={})", deviceRequests.size(), jobs);
        std::string diagnostics;
        const SlangWarmupBatchOptions options{.jobs = jobs,
            .onProgress = [](const SlangWarmupBatchProgress& progress) {
                if (progress.phase && std::string_view(progress.phase) != "request-complete") {
                    spdlog::info("[Native shader warmup] source batch {}", progress.phase);
                }
            }};
        auto acquired = registry.getWarmupShadersWithOverrides(deviceRequests, options, diagnostics);
        report.sourceAcquireTimeNanoseconds = elapsedNanoseconds(acquireBegin);
        if (!acquired) {
            log += "Native warmup source batch failed:\n" + diagnostics;
            return makeError(acquired.error());
        }
        const auto& batch = acquired->sources;
        if (!batch.completedAllRequests || batch.shaders.size() != deviceRequests.size()) {
            log += "Incomplete native warmup source batch\n";
            return makeError(Error::Failure);
        }
        report.sourceBatchUsed = !acquired->artifactOverrideActive;
        report.sourceEpochValidated = batch.completeSourceEpochValidated;
        report.sourceSnapshotTimeNanoseconds = batch.snapshotCaptureTimeNanoseconds;
        report.sourceBoundaryValidationTimeNanoseconds = batch.boundaryValidationTimeNanoseconds;
        report.sourcePublicationTimeNanoseconds = batch.cachePublicationTimeNanoseconds;
        report.sourceCacheHits = batch.existingCacheHits;
        report.sourceCompilerExecutions = batch.compilerExecutions;
        shaders = std::move(acquired->sources.shaders);
        spdlog::info("[Native shader warmup] sources ready mode={} hits={} compiles={} epochValidated={} acquireMs={:.3f} snapshotMs={:.3f} validationMs={:.3f} publicationMs={:.3f}",
            report.sourceBatchUsed ? "batch" : "artifact-override", report.sourceCacheHits, report.sourceCompilerExecutions,
            report.sourceEpochValidated, double(report.sourceAcquireTimeNanoseconds) / 1e6,
            double(report.sourceSnapshotTimeNanoseconds) / 1e6, double(report.sourceBoundaryValidationTimeNanoseconds) / 1e6,
            double(report.sourcePublicationTimeNanoseconds) / 1e6);
    }
    // Accept no source result until the complete batch has passed its source,
    // namespace, compiler and publication checks. Ordinary runtime acquisition
    // and dependency tracking remain unchanged for subsequent hot reloads.
    const auto nativeBegin = std::chrono::steady_clock::now();
    // Modules and owned specialization arrays stay alive through every join.
    std::vector<std::unique_ptr<ShaderModule>> modules;
    std::vector<NativeWarmupTask> tasks;
    for (size_t index = 0; index < deviceRequests.size(); ++index) {
        const auto& request = deviceRequests[index];
        const auto& shader = shaders[index];
        const std::string name = request.module + "." + request.entry;
        spdlog::debug("[Native shader warmup] source {}/{} {}", index + 1, deviceRequests.size(), name);
        auto requirements = inspectShader(shader.spirv);
        if (!requirements) {
            log += "Native warmup requires a valid single-entry SPIR-V module: " + name + "\n";
            return makeError(requirements.error());
        }
        if (!requirements->compute) {
            ++report.skippedGraphicsCount;
            spdlog::info("[Native shader warmup] skip {}: graphics requires a complete graphics PSO recipe", name);
            continue;
        }
        ++report.computeSourceCount;
        const auto variants = nativeComputeWarmupVariants(request);
        if (variants.empty()) {
            ++report.skippedUnownedABICount;
            spdlog::info("[Native shader warmup] skip {}: no authored native ABI recipe", name);
            continue;
        }
        if (const char* reason = unsupportedReason(*requirements, device.capabilities(), nativeDevice)) {
            report.skippedUnsupportedCount += variants.size();
            spdlog::info("[Native shader warmup] skip {} ({} variants): {}", name, variants.size(), reason);
            continue;
        }
        auto module = registry.getShaderModule(device, {.spirv = shader.spirv, .debugName = name.c_str()});
        if (!module) {
            log += "Native warmup shader module failed: " + name + ": " + resultToString(module) + "\n";
            return makeError(module.error());
        }
        // Keep admission tied to the actual device module if RHI lowering is
        // introduced later. Unexpected module/native failures remain failures.
        requirements = inspectShader(vulkan::nativeShaderSpirv(**module));
        if (!requirements) { return makeError(requirements.error()); }
        if (const char* reason = unsupportedReason(*requirements, device.capabilities(), nativeDevice)) {
            report.skippedUnsupportedCount += variants.size();
            spdlog::info("[Native shader warmup] skip finalized {}: {}", name, reason);
            continue;
        }
        for (size_t variantIndex = 0; variantIndex < variants.size(); ++variantIndex) {
            // SPIR-V size is only a scheduling estimate, not measured driver
            // history. Variants of one module share the same estimate.
            tasks.push_back({module->get(), name, variants[variantIndex], variantIndex,
                variants.size(), vulkan::nativeShaderSpirv(**module).size()});
        }
        modules.push_back(std::move(*module));
    }
    std::stable_sort(tasks.begin(), tasks.end(), [](const auto& left, const auto& right) {
        return left.estimatedCost > right.estimatedCost;
    });
    report.nativeTaskCount = tasks.size();
    report.nativeWorkerCount = std::min(configuredJobs, tasks.size());
    std::vector<NativeWarmupTaskResult> results(tasks.size());
    std::atomic<size_t> nextTask{0};
    std::atomic<bool> cancelled{false};
    spdlog::info("[Native shader warmup] creating {} native tasks (workers={}, largest SPIR-V first)",
        report.nativeTaskCount, report.nativeWorkerCount);
    const auto worker = [&] {
        while (!cancelled.load(std::memory_order_relaxed)) {
            const size_t taskIndex = nextTask.fetch_add(1, std::memory_order_relaxed);
            if (taskIndex >= tasks.size()) { return; }
            const auto& task = tasks[taskIndex];
            auto& result = results[taskIndex];
            try {
                const auto& variant = task.variant;
                const ComputePipelineDesc desc{
                    .computeShader = {task.module, "main", variant.specializationConstants},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = variant.userPushDataSize,
                    .indirectBindable = variant.indirectBindable,
                };
                // Log before entering the driver; a large material PSO can take
                // tens of seconds even when its source shader was already cached.
                spdlog::info("[Native shader warmup] creating task {}/{} {} variant {}/{} estimatedSpirvWords={}",
                    taskIndex + 1, tasks.size(), task.name, task.variantIndex + 1, task.variantCount, task.estimatedCost);
                const auto taskBegin = std::chrono::steady_clock::now();
                auto pipeline = registry.getComputePipeline(device, desc);
                if (!pipeline) {
                    result.error = pipeline.error();
                    cancelled.store(true, std::memory_order_relaxed);
                    return;
                }
                result.pipeline = std::move(*pipeline);
                const auto stats = result.pipeline->creationStats();
                spdlog::info("[Native shader warmup] {} {} variant {}/{} taskMs={:.3f} createMs={:.3f}",
                    stats.nativeObjectReused ? "reused" : "created", task.name, task.variantIndex + 1, task.variantCount,
                    double(elapsedNanoseconds(taskBegin)) / 1e6, double(stats.nativeCreateTimeNanoseconds) / 1e6);
            } catch (...) {
                result.exception = std::current_exception();
                cancelled.store(true, std::memory_order_relaxed);
                return;
            }
        }
    };
    std::exception_ptr launchException;
    {
        // jthread joins already-launched workers even if thread creation fails.
        std::vector<std::jthread> workers;
        try {
            workers.reserve(report.nativeWorkerCount);
            for (size_t index = 0; index < report.nativeWorkerCount; ++index) {
                workers.emplace_back(worker);
            }
        } catch (...) {
            launchException = std::current_exception();
            cancelled.store(true, std::memory_order_relaxed);
        }
    }
    report.nativePipelineTimeNanoseconds = elapsedNanoseconds(nativeBegin);
    std::optional<Error> failure;
    const auto recordException = [&](std::exception_ptr exception) {
        try {
            std::rethrow_exception(exception);
        } catch (const std::exception& error) {
            log += std::string("Native warmup worker failed: ") + error.what() + "\n";
        } catch (...) {
            log += "Native warmup worker failed with an unknown exception\n";
        }
        failure = Error::Failure;
    };
    if (launchException) { recordException(launchException); }
    // Merge bookkeeping only after joining. Successful objects remain pinned
    // even when a sibling task failed; registry single-flight handles duplicates.
    for (size_t index = 0; index < results.size(); ++index) {
        auto& result = results[index];
        if (result.error) {
            log += "Native warmup pipeline creation failed: " + tasks[index].name + ": " + errorToString(*result.error) + "\n";
            if (!failure) { failure = *result.error; }
        }
        if (result.exception) { recordException(result.exception); }
        if (result.pipeline) {
            const auto stats = result.pipeline->creationStats();
            report.nativeCreateCount += stats.nativeObjectReused ? 0 : 1;
            report.nativeObjectReuseCount += stats.nativeObjectReused ? 1 : 0;
            report.nativeCreateTimeNanoseconds += stats.nativeCreateTimeNanoseconds;
            const auto specialization = detail::canonicalShaderSpecializationConstants(tasks[index].variant.specializationConstants);
            const uint64_t hash = result.pipeline->psoHash();
            if (std::none_of(state->pipelines.begin(), state->pipelines.end(), [&](const auto& retained) {
                    return retained.psoHash == hash && retained.specialization == specialization;
                })) {
                state->pipelines.push_back({hash, specialization, result.pipeline->execution()});
            }
        }
    }
    report.retainedPipelineCount = state->pipelines.size();
    report.totalTimeNanoseconds = elapsedNanoseconds(begin);
    if (failure) { return makeError(*failure); }
    spdlog::info("[Native shader warmup] complete sources={} sourceHits={} sourceCompiles={} compute={} created={} reused={} graphicsSkipped={} unsupportedSkipped={} unknownABISkipped={} retained={} elapsedMs={:.3f} nativeCreateMs={:.3f} sourceAcquireMs={:.3f} nativeTasks={} nativeWorkers={} nativePipelineMs={:.3f}",
        report.sourceCount, report.sourceCacheHits, report.sourceCompilerExecutions, report.computeSourceCount,
        report.nativeCreateCount, report.nativeObjectReuseCount, report.skippedGraphicsCount, report.skippedUnsupportedCount,
        report.skippedUnownedABICount, report.retainedPipelineCount, double(report.totalTimeNanoseconds) / 1e6,
        double(report.nativeCreateTimeNanoseconds) / 1e6, double(report.sourceAcquireTimeNanoseconds) / 1e6,
        report.nativeTaskCount, report.nativeWorkerCount, double(report.nativePipelineTimeNanoseconds) / 1e6);
    return report;
}

Result<NativeShaderWarmupReport> warmupNativeShadersPipelined(Device& device,
    std::span<const ShaderRequest> requests, std::string& log)
{
    auto& registry = ShaderRegistry::instance();
    if (!registry.supportsStagedComputeWarmup()) { return warmupNativeShaders(device, requests, log); }
    auto staged = registry.beginStagedComputeWarmup(device);
    if (!staged) {
        if (staged.error() == Error::Unsupported) { return warmupNativeShaders(device, requests, log); }
        return makeError(staged.error());
    }
    static const char stateKey = 0;
    auto shared = device.sharedState(&stateKey, []() -> Result<std::shared_ptr<void>> {
        return std::make_shared<NativeWarmupState>();
    });
    if (!shared) { return makeError(shared.error()); }
    auto state = std::static_pointer_cast<NativeWarmupState>(*shared);
    std::lock_guard stateLock(state->mutex);
    auto deferral = registry.deferPipelineCacheSaves(device);
    if (!deferral) { return makeError(deferral.error()); }
    const auto begin = std::chrono::steady_clock::now();
    NativeShaderWarmupReport report;
    report.pipelined = true;
    report.sourceCount = requests.size();
    const auto nativeDevice = vulkan::nativeDevice(device);
    std::vector<ShaderRequest> deviceRequests;
    for (const auto& request : requests) {
        auto owned = request;
        const ShaderRequestView view(request);
        if (slangUsesNativeDescriptorHeap(view.desc())) {
            owned.descriptorHeapStrides = device.descriptorHeapShaderStrides();
            if (!nativeDevice.descriptorHeapEnabled || !nativeDevice.shaderUntypedPointersEnabled ||
                !owned.descriptorHeapStrides.resource || !owned.descriptorHeapStrides.sampler) {
                report.skippedUnsupportedCount += std::max(size_t{1}, nativeComputeWarmupVariants(request).size());
                continue;
            }
        }
        deviceRequests.push_back(std::move(owned));
    }
    if (deviceRequests.empty()) { return report; }
    size_t jobs = std::clamp(std::thread::hardware_concurrency(), 1u, 12u);
    if (const char* value = std::getenv("METALLIC_SHADER_WARMUP_JOBS"); value && value[0]) {
        const std::string_view text(value);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), jobs);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || jobs == 0 || jobs > 256) {
            log += "METALLIC_SHADER_WARMUP_JOBS requires an integer in [1, 256]\n";
            return makeError(Error::InvalidArgument);
        }
    }
    jobs = std::min(jobs, deviceRequests.size());
    struct Job {
        NativeWarmupTask task;
        NativeWarmupTaskResult result;
        size_t sourceIndex = 0;
        size_t sequence = 0;
    };
    struct Priority {
        bool operator()(const std::shared_ptr<Job>& a, const std::shared_ptr<Job>& b) const
        {
            if (a->task.estimatedCost != b->task.estimatedCost) { return a->task.estimatedCost < b->task.estimatedCost; }
            return a->sequence > b->sequence;
        }
    };
    std::vector<std::unique_ptr<ShaderModule>> modules(deviceRequests.size());
    // Destroy staged descriptors/owners before their borrowed module wrappers.
    auto pipelineStaging = std::move(*staged);
    std::vector<std::shared_ptr<Job>> allJobs;
    std::priority_queue<std::shared_ptr<Job>, std::vector<std::shared_ptr<Job>>, Priority> ready;
    std::mutex mutex;
    std::condition_variable changed;
    bool producerDone = false;
    bool cancelled = false;
    bool sourceAccepted = false;
    size_t sourceReadyCount = 0;
    size_t nativeFinishedCount = 0;
    size_t nativeActiveCount = 0;
    std::optional<std::chrono::steady_clock::time_point> firstNative;
    report.nativeWorkerCount = jobs;
    const auto worker = [&] {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return cancelled || producerDone || !ready.empty(); });
                if (cancelled || ready.empty()) { return; }
                job = ready.top(); ready.pop();
                ++nativeActiveCount;
                if (!firstNative) { firstNative = std::chrono::steady_clock::now(); }
                if (!sourceAccepted) { ++report.nativeTasksBeforeSourceAcceptance; }
            }
            try {
                const auto& task = job->task;
                const auto& variant = task.variant;
                const ComputePipelineDesc desc{
                    .computeShader = {task.module, "main", variant.specializationConstants},
                    .usesBindlessHeap = true, .bindlessUserPushDataSize = variant.userPushDataSize,
                    .indirectBindable = variant.indirectBindable};
                spdlog::info("[Native shader warmup] provisional create {} variant {}/{} estimatedSpirvWords={}",
                    task.name, task.variantIndex + 1, task.variantCount, task.estimatedCost);
                auto pipeline = pipelineStaging->getComputePipeline(deviceRequests[job->sourceIndex].module.c_str(), desc);
                if (pipeline) { job->result.pipeline = std::move(*pipeline); }
                else { job->result.error = pipeline.error(); }
            } catch (...) { job->result.exception = std::current_exception(); }
            {
                std::lock_guard lock(mutex);
                --nativeActiveCount;
                ++nativeFinishedCount;
                changed.notify_all();
                if (job->result.error || job->result.exception) {
                    cancelled = true;
                    changed.notify_all();
                    return;
                }
            }
        }
    };
    // Callback consumes immutable bytes immediately into a device module. The
    // module and specialization storage outlive every worker and commit.
    const auto onReady = [&](size_t index, const ShaderCompileResult& shader) {
        std::lock_guard lock(mutex);
        ++sourceReadyCount;
        if (cancelled) { return; }
        if (index >= modules.size()) { throw std::runtime_error("Invalid source stream index"); }
        const auto& request = deviceRequests[index];
        const auto name = request.module + "." + request.entry;
        auto requirements = inspectShader(shader.spirv);
        if (!requirements) { throw std::runtime_error("Invalid provisional SPIR-V: " + name); }
        if (!requirements->compute) { ++report.skippedGraphicsCount; return; }
        ++report.computeSourceCount;
        const auto variants = nativeComputeWarmupVariants(request);
        if (variants.empty()) { ++report.skippedUnownedABICount; return; }
        if (unsupportedReason(*requirements, device.capabilities(), nativeDevice)) {
            report.skippedUnsupportedCount += variants.size(); return;
        }
        auto module = device.createShaderModule({.spirv = shader.spirv, .debugName = name.c_str()});
        if (!module) { throw std::runtime_error("Provisional shader module failed: " + name); }
        requirements = inspectShader(vulkan::nativeShaderSpirv(**module));
        if (!requirements) { throw std::runtime_error("Invalid finalized provisional SPIR-V: " + name); }
        if (unsupportedReason(*requirements, device.capabilities(), nativeDevice)) {
            report.skippedUnsupportedCount += variants.size(); return;
        }
        modules[index] = std::move(*module);
        for (size_t variant = 0; variant < variants.size(); ++variant) {
            auto job = std::make_shared<Job>();
            job->task = {modules[index].get(), name, variants[variant], variant, variants.size(),
                vulkan::nativeShaderSpirv(*modules[index]).size()};
            job->sourceIndex = index;
            job->sequence = allJobs.size();
            allJobs.push_back(job);
            ready.push(std::move(job));
        }
        changed.notify_all();
    };
    Result<ShaderRegistryWarmupResult> acquired = makeError(Error::Failure);
    std::string diagnostics;
    std::vector<std::jthread> workers;
    std::condition_variable_any heartbeatWake;
    std::jthread heartbeat;
    try {
        workers.reserve(jobs);
        for (size_t index = 0; index < jobs; ++index) { workers.emplace_back(worker); }
        heartbeat = std::jthread([&](std::stop_token stop) {
            std::unique_lock lock(mutex);
            while (!stop.stop_requested()) {
                heartbeatWake.wait_for(lock, stop, std::chrono::seconds(2), [] { return false; });
                if (stop.stop_requested()) { return; }
                const auto sources = sourceReadyCount;
                const auto finished = nativeFinishedCount;
                const auto active = nativeActiveCount;
                const auto pending = ready.size();
                const auto accepted = sourceAccepted;
                lock.unlock();
                spdlog::info("[Native shader warmup] pipeline progress sources={}/{} nativeFinished={} active={} queued={} sourceAccepted={} elapsedMs={:.3f}",
                    sources, deviceRequests.size(), finished, active, pending, accepted, double(elapsedNanoseconds(begin)) / 1e6);
                lock.lock();
            }
        });
        spdlog::info("[Native shader warmup] pipelined acquisition sources={} sourceWorkers={} nativeWorkers={} (provisional; largest ready SPIR-V first)",
            deviceRequests.size(), jobs, jobs);
        const SlangWarmupBatchOptions options{.jobs = jobs,
            .onProgress = [](const SlangWarmupBatchProgress& progress) {
                if (progress.phase && std::string_view(progress.phase) != "request-complete") {
                    spdlog::info("[Native shader warmup] source batch {}", progress.phase);
                }
            }, .onShaderReady = onReady};
        acquired = registry.getWarmupShadersWithOverrides(deviceRequests, options, diagnostics);
    } catch (const std::exception& error) { diagnostics += error.what(); }
    catch (...) { diagnostics += "Unknown pipelined acquisition exception"; }
    report.sourceAcquireTimeNanoseconds = elapsedNanoseconds(begin);
    {
        std::lock_guard lock(mutex);
        sourceAccepted = acquired && acquired->sources.completedAllRequests && acquired->sources.shaders.size() == deviceRequests.size();
        if (!sourceAccepted) { cancelled = true; }
        producerDone = true;
    }
    changed.notify_all();
    workers.clear(); // Join before checking acceptance or destroying staged data.
    heartbeat.request_stop();
    if (heartbeat.joinable()) { heartbeat.join(); }
    if (!sourceAccepted) {
        log += "Pipelined source batch rejected; discarded provisional native work:\n" + diagnostics;
        return makeError(acquired ? Error::Failure : acquired.error());
    }
    if (firstNative) { report.nativePipelineTimeNanoseconds = elapsedNanoseconds(*firstNative); }
    report.nativeTaskCount = allJobs.size();
    const auto& batch = acquired->sources;
    report.sourceBatchUsed = !acquired->artifactOverrideActive;
    report.sourceEpochValidated = batch.completeSourceEpochValidated;
    report.sourceCacheHits = batch.existingCacheHits;
    report.sourceCompilerExecutions = batch.compilerExecutions;
    report.sourceSnapshotTimeNanoseconds = batch.snapshotCaptureTimeNanoseconds;
    report.sourceBoundaryValidationTimeNanoseconds = batch.boundaryValidationTimeNanoseconds;
    report.sourcePublicationTimeNanoseconds = batch.cachePublicationTimeNanoseconds;
    for (size_t index = 0; index < modules.size(); ++index) {
        if (modules[index]) { registry.registerWarmupShaderModule(device, *modules[index], batch.shaders[index].spirv); }
    }
    auto committed = pipelineStaging->commit();
    if (!committed) { return makeError(committed.error()); }
    std::optional<Error> failure;
    for (const auto& job : allJobs) {
        if (job->result.error || job->result.exception) {
            failure = job->result.error.value_or(Error::Failure);
            log += "Pipelined native creation failed: " + job->task.name + "\n";
        }
        if (!job->result.pipeline) { continue; }
        const auto& pipeline = job->result.pipeline;
        const auto stats = pipeline->creationStats();
        report.nativeCreateCount += !stats.nativeObjectReused;
        report.nativeObjectReuseCount += stats.nativeObjectReused;
        report.nativeCreateTimeNanoseconds += stats.nativeCreateTimeNanoseconds;
        const auto specialization = detail::canonicalShaderSpecializationConstants(job->task.variant.specializationConstants);
        if (std::none_of(state->pipelines.begin(), state->pipelines.end(), [&](const auto& retained) {
                return retained.psoHash == pipeline->psoHash() && retained.specialization == specialization;
            })) {
            state->pipelines.push_back({pipeline->psoHash(), specialization, pipeline->execution()});
        }
    }
    report.retainedPipelineCount = state->pipelines.size();
    report.totalTimeNanoseconds = elapsedNanoseconds(begin);
    if (failure) { return makeError(*failure); }
    spdlog::info("[Native shader warmup] complete sources={} sourceHits={} sourceCompiles={} compute={} created={} reused={} graphicsSkipped={} unsupportedSkipped={} unknownABISkipped={} retained={} elapsedMs={:.3f} nativeCreateMs={:.3f} sourceAcquireMs={:.3f} nativeTasks={} nativeWorkers={} nativePipelineMs={:.3f} pipelined=1 provisionalTasks={}",
        report.sourceCount, report.sourceCacheHits, report.sourceCompilerExecutions, report.computeSourceCount,
        report.nativeCreateCount, report.nativeObjectReuseCount, report.skippedGraphicsCount, report.skippedUnsupportedCount,
        report.skippedUnownedABICount, report.retainedPipelineCount, double(report.totalTimeNanoseconds) / 1e6,
        double(report.nativeCreateTimeNanoseconds) / 1e6, double(report.sourceAcquireTimeNanoseconds) / 1e6,
        report.nativeTaskCount, report.nativeWorkerCount, double(report.nativePipelineTimeNanoseconds) / 1e6,
        report.nativeTasksBeforeSourceAcceptance);
    return report;
}

Result<> warmupNativeShadersForStartup(Device& device)
{
    if (!startupShaderWarmupEnabled()) { return {}; }
    try {
        const auto requests = tools::shaderWarmupRequests();
        std::string log;
        auto result = warmupNativeShadersPipelined(device, requests, log);
        if (!result) {
            spdlog::error("[Native shader warmup] Startup cancelled: {}", log);
            return makeError(result.error());
        }
        return {};
    } catch (const std::exception& error) {
        spdlog::error("[Native shader warmup] Startup cancelled: {}", error.what());
        return makeError(Error::Failure);
    }
}

} // namespace metallic::render

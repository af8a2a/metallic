#include "Runtime/Render/Core/NativeShaderWarmup.h"
#include "Runtime/Render/Core/NativeShaderWarmupRequests.h"
#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ShaderWarmup.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "../../../../Tools/ShaderWarmupRequests.h"

#include <algorithm>
#include <chrono>
#include <mutex>
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
    const auto nativeDevice = vulkan::nativeDevice(device);
    const bool nativeDescriptorHeapAvailable = nativeDevice.descriptorHeapEnabled && nativeDevice.shaderUntypedPointersEnabled;
    for (size_t index = 0; index < requests.size(); ++index) {
        const auto& request = requests[index];
        const std::string name = request.module + "." + request.entry;
        spdlog::info("[Native shader warmup] source {}/{} {}", index + 1, requests.size(), name);
        const ShaderRequestView view(request);
        auto source = view.desc();
        if (slangUsesNativeDescriptorHeap(source)) {
            source.descriptorHeapStrides = device.descriptorHeapShaderStrides();
            if (!nativeDescriptorHeapAvailable || !source.descriptorHeapStrides.resource || !source.descriptorHeapStrides.sampler) {
                // This device cannot compile the native heap ABI. Do not fall
                // back to process-global strides belonging to another device.
                const auto variants = nativeComputeWarmupVariants(request);
                report.skippedUnsupportedCount += std::max(size_t{1}, variants.size());
                spdlog::info("[Native shader warmup] skip {}: native descriptor heap, untyped pointers, or descriptor strides are unavailable", name);
                continue;
            }
        }
        std::string diagnostics;
        auto shader = registry.getShader(source, diagnostics);
        if (!shader) {
            log += "Native warmup source failed: " + name + "\n" + diagnostics;
            return makeError(shader.error());
        }
        report.sourceCacheHits += shader->metrics.cacheHit ? 1 : 0;
        report.sourceCompilerExecutions += shader->metrics.compilerExecuted ? 1 : 0;
        auto requirements = inspectShader(shader->spirv);
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
        auto module = registry.getShaderModule(device, {.spirv = shader->spirv, .debugName = name.c_str()});
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
            const auto& variant = variants[variantIndex];
            const ComputePipelineDesc desc{
                .computeShader = {module->get(), "main", variant.specializationConstants},
                .usesBindlessHeap = true, .bindlessUserPushDataSize = variant.userPushDataSize,
                .indirectBindable = variant.indirectBindable,
            };
            // Log before entering the driver; a large material PSO can take
            // tens of seconds even when its source shader was already cached.
            spdlog::info("[Native shader warmup] creating {} variant {}/{}", name, variantIndex + 1, variants.size());
            auto pipeline = registry.getComputePipeline(device, desc);
            if (!pipeline) {
                log += "Native warmup pipeline creation failed: " + name + ": " + resultToString(pipeline) + "\n";
                return makeError(pipeline.error());
            }
            const auto stats = (*pipeline)->creationStats();
            report.nativeCreateCount += stats.nativeObjectReused ? 0 : 1;
            report.nativeObjectReuseCount += stats.nativeObjectReused ? 1 : 0;
            report.nativeCreateTimeNanoseconds += stats.nativeCreateTimeNanoseconds;
            const auto specialization = detail::canonicalShaderSpecializationConstants(variant.specializationConstants);
            const uint64_t hash = (*pipeline)->psoHash();
            if (std::none_of(state->pipelines.begin(), state->pipelines.end(), [&](const auto& retained) {
                    return retained.psoHash == hash && retained.specialization == specialization;
                })) {
                state->pipelines.push_back({hash, specialization, (*pipeline)->execution()});
            }
            spdlog::info("[Native shader warmup] {} {} variant {} createMs={:.3f}",
                stats.nativeObjectReused ? "reused" : "created", name, variantIndex + 1,
                double(stats.nativeCreateTimeNanoseconds) / 1e6);
        }
    }
    report.retainedPipelineCount = state->pipelines.size();
    report.totalTimeNanoseconds = elapsedNanoseconds(begin);
    spdlog::info("[Native shader warmup] complete sources={} sourceHits={} sourceCompiles={} compute={} created={} reused={} graphicsSkipped={} unsupportedSkipped={} unknownABISkipped={} retained={} elapsedMs={:.3f} nativeCreateMs={:.3f}",
        report.sourceCount, report.sourceCacheHits, report.sourceCompilerExecutions, report.computeSourceCount,
        report.nativeCreateCount, report.nativeObjectReuseCount, report.skippedGraphicsCount, report.skippedUnsupportedCount,
        report.skippedUnownedABICount, report.retainedPipelineCount, double(report.totalTimeNanoseconds) / 1e6,
        double(report.nativeCreateTimeNanoseconds) / 1e6);
    return report;
}

Result<> warmupNativeShadersForStartup(Device& device)
{
    if (!startupShaderWarmupEnabled()) { return {}; }
    try {
        const auto requests = tools::shaderWarmupRequests();
        std::string log;
        auto result = warmupNativeShaders(device, requests, log);
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

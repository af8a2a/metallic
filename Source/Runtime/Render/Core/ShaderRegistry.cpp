#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ComputeProgram.h"

#include <map>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

struct DeviceShaders {
    std::mutex mutex;
    std::map<std::string, std::unique_ptr<PipelineCache>> caches;
};

Result<std::shared_ptr<DeviceShaders>> deviceShaders(Device& device)
{
    static const char key = 0;
    auto state = device.sharedState(&key, []() -> Result<std::shared_ptr<void>> {
        return std::make_shared<DeviceShaders>();
    });
    if (!state) { return makeError(state.error()); }
    return std::static_pointer_cast<DeviceShaders>(*state);
}

Result<PipelineCache*> acquireCache(Device& device, const std::string& group)
{
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    std::lock_guard lock((*state)->mutex);
    auto& cache = (*state)->caches[group];
    if (!cache) {
        const std::string path = std::string(PROJECT_SOURCE_DIR "/.cache/pso/") + group + ".pso";
        auto created = device.createPipelineCache({.filePath = path.c_str()});
        if (!created) {
            (*state)->caches.erase(group);
            return makeError(created.error());
        }
        cache = std::move(*created);
    }
    return cache.get();
}

void persist(PipelineCache& cache, const std::string& group)
{
    // save() already avoids I/O on unchanged hits. Persist newly compiled PSOs
    // here, including early-return/error-program paths in a caller's compile.
    const auto result = cache.save();
    const auto stats = cache.stats();
    spdlog::info("[ShaderRegistry] PSO cache group={} hits={} misses={}", group, stats.hitCount, stats.missCount);
    if (!result) {
        spdlog::warn("[ShaderRegistry] Could not persist PSO cache group={}: {}", group, resultToString(result));
    }
}

} // namespace

Result<std::unique_ptr<ShaderModule>> ShaderRegistry::getShaderModule(Device& device, const ShaderModuleDesc& desc)
{
    auto module = device.createShaderModule(desc);
    if (module) { registerShaderModule(desc.spirv, (*module)->contentHash()); }
    return module;
}

Result<> ShaderRegistry::getComputeKernel(Device& device, const SlangShaderDesc& source,
    const ComputeKernelDesc& layout, ComputeKernel& kernel, std::string& log)
{
    auto shader = getShader(source, log);
    if (!shader) { return makeError(shader.error()); }
    auto desc = layout;
    desc.spirv = shader->spirv;
    ComputeKernel candidate;
    std::string creationLog;
    auto result = candidate.initialize(device, desc, creationLog);
    log += creationLog;
    if (result) { kernel = std::move(candidate); }
    return result;
}

Result<> ShaderRegistry::getComputeProgram(Device& device, const SlangShaderDesc& source,
    const ComputeProgramDesc& layout, ComputeProgram& program, std::string& log)
{
    auto shader = getShader(source, log);
    if (!shader) { return makeError(shader.error()); }
    auto desc = layout;
    desc.spirv = shader->spirv;
    ComputeProgram candidate;
    std::string creationLog;
    auto result = candidate.initialize(device, desc, creationLog);
    log += creationLog;
    if (result) { program = std::move(candidate); }
    return result;
}

Result<std::unique_ptr<ComputePipeline>> ShaderRegistry::getComputePipeline(Device& device, const ComputePipelineDesc& desc)
{
    if (!desc.computeShader.module) { return makeError(Error::InvalidArgument); }
    if (desc.pipelineCache) { return device.createComputePipeline(desc); }
    const std::string group = cacheGroup(desc.computeShader.module->contentHash());
    auto cache = acquireCache(device, group);
    if (!cache) { return makeError(cache.error()); }
    auto managed = desc;
    managed.pipelineCache = *cache;
    auto pipeline = device.createComputePipeline(managed);
    if (pipeline) { persist(**cache, group); }
    return pipeline;
}

Result<std::unique_ptr<GraphicsPipeline>> ShaderRegistry::getGraphicsPipeline(Device& device, const GraphicsPipelineDesc& desc)
{
    const ShaderModule* primary = desc.meshShader.module ? desc.meshShader.module : desc.vertexShader.module;
    if (!primary) { return makeError(Error::InvalidArgument); }
    if (desc.pipelineCache) { return device.createGraphicsPipeline(desc); }
    const std::string group = cacheGroup(primary->contentHash());
    auto cache = acquireCache(device, group);
    if (!cache) { return makeError(cache.error()); }
    auto managed = desc;
    managed.pipelineCache = *cache;
    auto pipeline = device.createGraphicsPipeline(managed);
    if (pipeline) { persist(**cache, group); }
    return pipeline;
}

Result<std::unique_ptr<GraphicsShaderObjectProgram>> ShaderRegistry::getGraphicsShaderObjectProgram(
    Device& device, const GraphicsShaderObjectProgramDesc& desc)
{
    if (!desc.vertexShader.module) { return makeError(Error::InvalidArgument); }
    const std::string group = cacheGroup(desc.vertexShader.module->contentHash());
    const std::string directory = std::string(PROJECT_SOURCE_DIR "/.cache/shader-objects/") + group;
    auto managed = desc;
    if (!managed.binaryCacheDirectory) { managed.binaryCacheDirectory = directory.c_str(); }
    auto program = device.createGraphicsShaderObjectProgram(managed);
    if (program) {
        const auto stats = (*program)->cacheStats();
        spdlog::info("[ShaderRegistry] ShaderObject cache group={} key={:016x} binaryHit={} persisted={} bytes={} createMs={:.3f}",
            group, stats.programHash, stats.binaryCacheHit, stats.persisted, stats.binaryDataSize,
            static_cast<double>(stats.creationTimeNanoseconds) / 1'000'000.0);
    }
    return program;
}

Result<> ShaderRegistry::getExternalGraphicsPipeline(Device& device, uint64_t shaderHash,
    const std::function<Result<>(PipelineCache&)>& factory)
{
    if (!factory) { return makeError(Error::InvalidArgument); }
    const std::string group = cacheGroup(shaderHash);
    auto cache = acquireCache(device, group);
    if (!cache) { return makeError(cache.error()); }
    auto result = factory(**cache);
    if (result) { persist(**cache, group); }
    return result;
}

Result<std::vector<ShaderRegistryCacheStats>> ShaderRegistry::pipelineCacheStats(Device& device)
{
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    std::lock_guard lock((*state)->mutex);
    std::vector<ShaderRegistryCacheStats> result;
    for (const auto& [group, cache] : (*state)->caches) { result.push_back({group, cache->stats()}); }
    return result;
}

} // namespace metallic::render

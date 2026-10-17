#include "Runtime/Render/Core/ShaderRegistry.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Core/DeferredShaderCacheWriter.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <iomanip>
#include <set>
#include <sstream>
#include <type_traits>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

struct DeviceShaders {
    std::mutex mutex;
    std::map<std::string, std::unique_ptr<PipelineCache>> caches;
    // Input code is retained only for an explicitly requested recipe capture.
    std::map<const ShaderModule*, PipelineRecipeShader> recipeShaders;
    std::set<std::string> exportedRecipes;
    detail::NativePipelineCache nativePipelines;
    // Reverse member destruction drains/joins the writer before cache handles.
    DeferredShaderCacheWriter writer;
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

std::string recipeDirectory()
{
    const auto* value = std::getenv("METALLIC_PIPELINE_RECIPE_DIRECTORY");
    return value ? value : "";
}

PipelineRecipeDevice recipeDevice(Device& device)
{
    PipelineRecipeDevice result;
    result.capabilities = device.capabilities();
    const auto native = vulkan::nativeDevice(device);
    result.validationEnabled = native.validationEnabled;
    result.synchronizationValidationEnabled = native.synchronizationValidationEnabled;
    if (native.properties) {
        const auto& core = native.properties->core;
        result.deviceName = core.deviceName;
        result.driverName = native.properties->driver.driverName;
        result.vendorId = core.vendorID;
        result.deviceId = core.deviceID;
        result.driverVersion = core.driverVersion;
        result.apiVersion = core.apiVersion;
        std::ostringstream uuid;
        for (const auto byte : core.pipelineCacheUUID) { uuid << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte); }
        result.pipelineCacheUUID = uuid.str();
    }
    return result;
}

bool recipeStage(const DeviceShaders& state, const ShaderStageDesc& stage, PipelineRecipeShader& shader)
{
    if (!stage.module) {
        shader.entryPoint = stage.entryPoint ? stage.entryPoint : "main";
        return true;
    }
    const auto found = state.recipeShaders.find(stage.module);
    if (found == state.recipeShaders.end()) { return false; }
    shader = found->second;
    shader.entryPoint = stage.entryPoint ? stage.entryPoint : "main";
    shader.specializationConstants = detail::canonicalShaderSpecializationConstants(stage.specializationConstants);
    return true;
}

template<typename Desc>
void exportRecipe(Device& device, const Desc& desc, uint64_t psoHash, const std::string& group)
{
    const auto directory = recipeDirectory();
    if (directory.empty()) { return; }
    auto state = deviceShaders(device);
    if (!state) { return; }
    // Capture is diagnostic and opt-in. Serialize exports per device so shared
    // stages and repeated requests publish each manifest once.
    std::lock_guard lock((*state)->mutex);
    std::ostringstream filename;
    filename << (std::is_same_v<Desc, ComputePipelineDesc> ? "compute-" :
        std::is_same_v<Desc, RayTracingPipelineDesc> ? "ray-tracing-" : "graphics-")
        << std::hex << std::setw(16) << std::setfill('0') << psoHash << ".recipe.json";
    const auto path = std::filesystem::path(directory) / filename.str();
    if ((*state)->exportedRecipes.contains(path.string())) { return; }
    PipelineRecipe recipe;
    recipe.expectedPsoHash = psoHash;
    recipe.cacheGroup = group;
    recipe.sourceDevice = recipeDevice(device);
    recipe.usesBindlessHeap = desc.usesBindlessHeap;
    if constexpr (!std::is_same_v<Desc, RayTracingPipelineDesc>) { recipe.indirectBindable = desc.indirectBindable; }
    recipe.disableOptimization = desc.disableOptimization;
    bool complete;
    if constexpr (std::is_same_v<Desc, ComputePipelineDesc>) {
        recipe.type = PipelineRecipeType::Compute;
        recipe.computeBackend = desc.backend;
        recipe.bindlessUserPushDataSize = desc.bindlessUserPushDataSize;
        complete = recipeStage(**state, desc.computeShader, recipe.computeShader);
    } else if constexpr (std::is_same_v<Desc, RayTracingPipelineDesc>) {
        recipe.type = PipelineRecipeType::RayTracing;
        recipe.bindlessUserPushDataSize = desc.bindlessUserPushDataSize;
        recipe.maxRecursionDepth = desc.maxRecursionDepth;
        recipe.clusterAccelerationStructure = desc.clusterAccelerationStructure;
        recipe.enableInvocationReorder = desc.enableInvocationReorder;
        complete = recipeStage(**state, desc.rayGenerationShader, recipe.rayGenerationShader) &&
            recipeStage(**state, desc.missShader, recipe.missShader) &&
            recipeStage(**state, desc.closestHitShader, recipe.closestHitShader) &&
            recipeStage(**state, desc.anyHitShader, recipe.anyHitShader);
    } else {
        recipe.type = PipelineRecipeType::Graphics;
        recipe.taskRequiredSubgroupSize = desc.taskRequiredSubgroupSize;
        recipe.taskRequireFullSubgroups = desc.taskRequireFullSubgroups;
        recipe.colorFormats = desc.colorFormats;
        recipe.colorAttachmentCount = desc.colorAttachmentCount;
        recipe.depthStencilFormat = desc.depthStencilFormat;
        recipe.topology = desc.topology;
        recipe.rasterization = desc.rasterization;
        recipe.depthStencil = desc.depthStencil;
        complete = recipeStage(**state, desc.vertexShader, recipe.vertexShader) &&
            recipeStage(**state, desc.taskShader, recipe.taskShader) &&
            recipeStage(**state, desc.meshShader, recipe.meshShader) &&
            recipeStage(**state, desc.fragmentShader, recipe.fragmentShader);
    }
    if (!complete) {
        spdlog::warn("[ShaderRegistry] Recipe {} requires modules created through the registry while capture is enabled", path.string());
        return;
    }
    std::string log;
    if (!writePipelineRecipe(recipe, path, log)) {
        spdlog::warn("[ShaderRegistry] {}", log);
    } else {
        (*state)->exportedRecipes.insert(path.string());
        spdlog::info("[ShaderRegistry] Exported pipeline recipe {}", path.string());
    }
}

Result<PipelineCache*> acquireCache(Device& device, const std::string& group)
{
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    std::lock_guard lock((*state)->mutex);
    auto& cache = (*state)->caches[group];
    if (!cache) {
        const char* cacheDirectory = std::getenv("METALLIC_PSO_CACHE_DIRECTORY");
        const std::filesystem::path directory = cacheDirectory && cacheDirectory[0]
            ? std::filesystem::path(cacheDirectory)
            : std::filesystem::path(PROJECT_SOURCE_DIR "/.cache/pso");
        const std::string path = (directory / (group + ".pso")).string();
        auto created = device.createPipelineCache({.filePath = path.c_str()});
        if (!created) {
            (*state)->caches.erase(group);
            return makeError(created.error());
        }
        cache = std::move(*created);
    }
    return cache.get();
}

void persist(Device& device, PipelineCache& cache, const std::string& group)
{
    const auto stats = cache.stats();
    spdlog::info("[ShaderRegistry] PSO cache group={} hits={} misses={} nativeHits={} nativeMisses={} nativeUnknown={} createMs={:.3f}",
        group, stats.hitCount, stats.missCount, stats.nativeHitCount, stats.nativeMissCount,
        stats.unknownFeedbackCount, static_cast<double>(stats.lastNativeCreateTimeNanoseconds) / 1'000'000.0);
    if (stats.dirtyRevision == stats.persistedRevision) { return; }
    auto state = deviceShaders(device);
    if (!state) {
        spdlog::warn("[ShaderRegistry] Could not queue PSO cache group={}", group);
        return;
    }
    // Native hits do not extend the deadline. Bursts of backend updates share a save, and
    // ongoing work has a bounded deadline instead of deferring indefinitely.
    (*state)->writer.request(group, stats.dirtyRevision, [&cache, group] {
        const auto result = cache.save();
        if (!result) {
            spdlog::warn("[ShaderRegistry] Could not persist PSO cache group={}: {}", group, resultToString(result));
        }
        return static_cast<bool>(result);
    });
}

} // namespace

Result<std::unique_ptr<ShaderModule>> ShaderRegistry::getShaderModule(Device& device, const ShaderModuleDesc& desc)
{
    auto module = device.createShaderModule(desc);
    if (module) {
        registerShaderModule(desc.spirv, (*module)->contentHash());
        if (!recipeDirectory().empty()) {
            auto state = deviceShaders(device);
            if (state) {
                auto shader = recipeShader(desc.spirv);
                std::lock_guard lock((*state)->mutex);
                // Different inputs can finalize to identical native code. Keep
                // capture associated with the exact live module wrapper; an
                // allocator-reused address replaces its previous input snapshot.
                (*state)->recipeShaders.insert_or_assign(module->get(), std::move(shader));
            }
        }
    }
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

Result<std::unique_ptr<ComputePipeline>> ShaderRegistry::getComputePipeline(Device& device, const ComputePipelineDesc& requestedDesc)
{
    auto desc = requestedDesc;
    std::vector<ShaderSpecializationConstant> specialization;
    try {
        specialization = detail::canonicalShaderSpecializationConstants(desc.computeShader.specializationConstants);
    } catch (const std::invalid_argument&) {
        return makeError(Error::InvalidArgument);
    }
    desc.computeShader.specializationConstants = specialization;
    const char* optimization = std::getenv("METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION");
    if (optimization && optimization[0] && std::strcmp(optimization, "1") != 0 && std::strcmp(optimization, "0") != 0) {
        return makeError(Error::InvalidArgument);
    }
    if (optimization && std::strcmp(optimization, "1") == 0) { desc.disableOptimization = true; }
    const char* backend = std::getenv("METALLIC_VK_COMPUTE_BACKEND");
    if (backend && backend[0] && std::strcmp(backend, "shader-object") != 0 &&
        std::strcmp(backend, "pipeline-binary") != 0 && std::strcmp(backend, "pipeline") != 0) {
        return makeError(Error::InvalidArgument);
    }
    if (desc.backend == ComputePipelineBackend::Pipeline && !desc.pipelineCache && !desc.nativeBinaryCacheDirectory) {
        if (backend && std::strcmp(backend, "shader-object") == 0) { desc.backend = ComputePipelineBackend::ShaderObject; }
        if (backend && std::strcmp(backend, "pipeline-binary") == 0) { desc.backend = ComputePipelineBackend::PipelineBinary; }
    }
    if (desc.backend != ComputePipelineBackend::Pipeline && !desc.nativeBinaryCacheDirectory) {
        desc.nativeBinaryCacheDirectory = std::getenv("METALLIC_VK_NATIVE_BINARY_CACHE_DIR");
    }
    auto valid = device.validateComputePipelineDesc(desc);
    if (!valid) { return makeError(valid.error()); }
    const std::string group = cacheGroup(desc.computeShader.module->contentHash());
    if (desc.pipelineCache || (desc.nativeBinaryCacheDirectory && desc.nativeBinaryCacheDirectory[0])) {
        auto pipeline = device.createComputePipeline(desc);
        if (pipeline) { exportRecipe(device, desc, (*pipeline)->psoHash(), group); }
        return pipeline;
    }
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    const auto key = detail::NativePipelineKey{detail::NativePipelineType::Compute, detail::computePipelineStateHash(desc), specialization};
    auto acquired = (*state)->nativePipelines.acquire(key, [&]() -> Result<std::shared_ptr<void>> {
        if (desc.backend != ComputePipelineBackend::Pipeline) {
            auto pipeline = device.createComputePipeline(desc);
            if (!pipeline) { return makeError(pipeline.error()); }
            return std::shared_ptr<void>((*pipeline)->impl_);
        }
        auto cache = acquireCache(device, group);
        if (!cache) { return makeError(cache.error()); }
        auto managed = desc;
        managed.pipelineCache = *cache;
        auto pipeline = device.createComputePipeline(managed);
        if (!pipeline) { return makeError(pipeline.error()); }
        persist(device, **cache, group);
        return std::shared_ptr<void>((*pipeline)->impl_);
    });
    if (!acquired) { return makeError(acquired.error()); }
    auto pipeline = std::unique_ptr<ComputePipeline>(new ComputePipeline(
        std::static_pointer_cast<detail::ComputePipelineImpl>(acquired->owner)));
    if (acquired->nativeObjectReused) {
        pipeline->acquisitionStats_ = PipelineCreationStats{
            .singleFlightWaitTimeNanoseconds = acquired->singleFlightWaitTimeNanoseconds,
            .nativeObjectReused = true, .computeBackend = desc.backend};
    }
    exportRecipe(device, desc, pipeline->psoHash(), group);
    return pipeline;
}

Result<std::unique_ptr<RayTracingPipeline>> ShaderRegistry::getRayTracingPipeline(Device& device, const RayTracingPipelineDesc& requestedDesc)
{
    auto desc = requestedDesc;
    std::vector<ShaderSpecializationConstant> specialization;
    try {
        specialization = detail::canonicalShaderSpecializationConstants(desc.rayGenerationShader.specializationConstants);
    } catch (const std::invalid_argument&) {
        return makeError(Error::InvalidArgument);
    }
    desc.rayGenerationShader.specializationConstants = specialization;
    const char* optimization = std::getenv("METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION");
    if (optimization && optimization[0] && std::strcmp(optimization, "1") != 0 && std::strcmp(optimization, "0") != 0) {
        return makeError(Error::InvalidArgument);
    }
    if (optimization && std::strcmp(optimization, "1") == 0) { desc.disableOptimization = true; }
    if (auto valid = device.validateRayTracingPipelineDesc(desc); !valid) { return makeError(valid.error()); }
    const std::string group = "ray-tracing-" + cacheGroup(desc.rayGenerationShader.module->contentHash());
    if (desc.pipelineCache) {
        auto pipeline = device.createRayTracingPipeline(desc);
        if (pipeline) { exportRecipe(device, desc, (*pipeline)->psoHash(), group); }
        return pipeline;
    }
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    const auto key = detail::NativePipelineKey{detail::NativePipelineType::RayTracing, detail::rayTracingPipelineStateHash(desc), specialization};
    auto acquired = (*state)->nativePipelines.acquire(key, [&]() -> Result<std::shared_ptr<void>> {
        auto cache = acquireCache(device, group);
        if (!cache) { return makeError(cache.error()); }
        auto managed = desc;
        managed.pipelineCache = *cache;
        auto pipeline = device.createRayTracingPipeline(managed);
        if (!pipeline) { return makeError(pipeline.error()); }
        persist(device, **cache, group);
        return std::shared_ptr<void>((*pipeline)->impl_);
    });
    if (!acquired) { return makeError(acquired.error()); }
    auto pipeline = std::unique_ptr<RayTracingPipeline>(new RayTracingPipeline(
        std::static_pointer_cast<detail::RayTracingPipelineImpl>(acquired->owner)));
    if (acquired->nativeObjectReused) {
        pipeline->acquisitionStats_ = PipelineCreationStats{
            .singleFlightWaitTimeNanoseconds = acquired->singleFlightWaitTimeNanoseconds, .nativeObjectReused = true};
    }
    exportRecipe(device, desc, pipeline->psoHash(), group);
    return pipeline;
}

Result<std::unique_ptr<GraphicsPipeline>> ShaderRegistry::getGraphicsPipeline(Device& device, const GraphicsPipelineDesc& requestedDesc)
{
    auto desc = requestedDesc;
    const char* optimization = std::getenv("METALLIC_VK_DISABLE_PIPELINE_OPTIMIZATION");
    if (optimization && optimization[0] && std::strcmp(optimization, "1") != 0 && std::strcmp(optimization, "0") != 0) {
        return makeError(Error::InvalidArgument);
    }
    if (optimization && std::strcmp(optimization, "1") == 0) { desc.disableOptimization = true; }
    auto valid = device.validateGraphicsPipelineDesc(desc);
    if (!valid) { return makeError(valid.error()); }
    const ShaderModule* primary = desc.meshShader.module ? desc.meshShader.module : desc.vertexShader.module;
    const std::string group = cacheGroup(primary->contentHash());
    if (desc.pipelineCache) {
        auto pipeline = device.createGraphicsPipeline(desc);
        if (pipeline) { exportRecipe(device, desc, (*pipeline)->psoHash(), group); }
        return pipeline;
    }
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    const auto key = detail::NativePipelineKey{detail::NativePipelineType::Graphics, detail::graphicsPipelineStateHash(desc)};
    auto acquired = (*state)->nativePipelines.acquire(key, [&]() -> Result<std::shared_ptr<void>> {
        auto cache = acquireCache(device, group);
        if (!cache) { return makeError(cache.error()); }
        auto managed = desc;
        managed.pipelineCache = *cache;
        auto pipeline = device.createGraphicsPipeline(managed);
        if (!pipeline) { return makeError(pipeline.error()); }
        persist(device, **cache, group);
        return std::shared_ptr<void>((*pipeline)->impl_);
    });
    if (!acquired) { return makeError(acquired.error()); }
    auto pipeline = std::unique_ptr<GraphicsPipeline>(new GraphicsPipeline(
        std::static_pointer_cast<detail::GraphicsPipelineImpl>(acquired->owner)));
    if (acquired->nativeObjectReused) {
        pipeline->acquisitionStats_ = PipelineCreationStats{
            .singleFlightWaitTimeNanoseconds = acquired->singleFlightWaitTimeNanoseconds, .nativeObjectReused = true};
    }
    exportRecipe(device, desc, pipeline->psoHash(), group);
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
    if (result) { persist(device, **cache, group); }
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

Result<NativePipelineCacheStats> ShaderRegistry::nativePipelineCacheStats(Device& device)
{
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    return (*state)->nativePipelines.stats();
}

Result<> ShaderRegistry::setNativePipelineCacheCapacity(Device& device, size_t strongCapacity)
{
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    (*state)->nativePipelines.setCapacity(strongCapacity);
    return {};
}

Result<> ShaderRegistry::flushPipelineCaches(Device& device)
{
    auto state = deviceShaders(device);
    if (!state) { return makeError(state.error()); }
    // Never hold the map mutex while waiting for callbacks. The device-owned
    // state keeps all borrowed cache pointers alive through this barrier.
    (*state)->writer.flush();
    std::lock_guard lock((*state)->mutex);
    for (const auto& [group, cache] : (*state)->caches) {
        const auto stats = cache->stats();
        if (stats.dirtyRevision != stats.persistedRevision) { return makeError(Error::Failure); }
    }
    return {};
}

} // namespace metallic::render

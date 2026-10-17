#pragma once

#include "SlangCompiler.h"
#include "Runtime/Render/Core/PipelineRecipe.h"
#include "Runtime/Render/Core/NativePipelineCache.h"

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace metallic::render {

struct ShaderRegistryCacheStats {
    std::string group;
    PipelineCacheStats cache;
};

class ComputeKernel;
struct ComputeKernelDesc;
class ShaderRegistry;

struct ShaderArtifactOverrideResult {
    ShaderCompileResult shader;
    // moduleName identifies the actual artifact producer, not the requested
    // production module. Full request/SHA/validation evidence belongs to the
    // test loader. Empty spirv is allowed; supplied words must match shader.
    PipelineRecipeShader identity;
};

class ShaderArtifactOverride {
public:
    virtual ~ShaderArtifactOverride() = default;
    // nullopt leaves a non-target request on the ordinary compiler path. An
    // engaged error or exception rejects a target without compiling fallback.
    // Implementations own immutable bytes and synchronize concurrent resolves.
    [[nodiscard]] virtual std::optional<Result<ShaderArtifactOverrideResult>> resolve(
        const SlangShaderDesc& desc, std::string& log) = 0;
};

class ShaderArtifactOverrideScope {
public:
    ShaderArtifactOverrideScope() = default;
    ~ShaderArtifactOverrideScope();
    ShaderArtifactOverrideScope(const ShaderArtifactOverrideScope&) = delete;
    ShaderArtifactOverrideScope& operator=(const ShaderArtifactOverrideScope&) = delete;
    ShaderArtifactOverrideScope(ShaderArtifactOverrideScope&& other) noexcept;
    ShaderArtifactOverrideScope& operator=(ShaderArtifactOverrideScope&& other) noexcept;
    void reset() noexcept;
    [[nodiscard]] explicit operator bool() const noexcept { return registry_ != nullptr; }

private:
    friend class ShaderRegistry;
    ShaderArtifactOverrideScope(ShaderRegistry* registry, uint64_t installation) noexcept;
    ShaderRegistry* registry_ = nullptr;
    uint64_t installation_ = 0;
};

// The process singleton owns source identities only. Native cache state belongs
// to Device::sharedState and is released before that device's native teardown.
class ShaderRegistry {
public:
    static ShaderRegistry& instance();
    ShaderRegistry(const ShaderRegistry&) = delete;
    ShaderRegistry& operator=(const ShaderRegistry&) = delete;

    // Ordinary requests validate dependencies through Slang's disk-cache path;
    // unchanged/precompiled code is reused, missing or stale code is compiled.
    [[nodiscard]] Result<ShaderCompileResult> getShader(const SlangShaderDesc& desc, std::string& log);
    [[nodiscard]] Result<ShaderCompileResult> getShader(const SlangShaderDesc& desc,
        const SlangShaderCacheOptions& options, std::string& log);
    [[nodiscard]] Result<SlangWarmupBatchResult> getWarmupShaders(std::span<const SlangShaderDesc> requests,
        const SlangWarmupBatchOptions& options, std::string& log);
    // Opt-in tests only. At most one scope can be installed. Resolve callbacks
    // run outside the registry lock and remain alive for in-flight requests
    // after scope removal. Handled requests never read/write Slang's cache or
    // report compiler execution/cache hits, even when the loader supplied them.
    [[nodiscard]] Result<ShaderArtifactOverrideScope> installArtifactOverrideForTesting(
        std::shared_ptr<ShaderArtifactOverride> artifactOverride);
    [[nodiscard]] Result<std::unique_ptr<ShaderModule>> getShaderModule(Device& device, const ShaderModuleDesc& desc);
    // Resolve source, acquire its PSO and publish only a successful generation.
    // layout.spirv is replaced by the requested source's code.
    [[nodiscard]] Result<> getComputeKernel(Device& device, const SlangShaderDesc& source,
        const ComputeKernelDesc& layout, ComputeKernel& kernel, std::string& log);
    // Null desc.pipelineCache uses the device's single-flight native object
    // cache above its persistent binary cache. Explicit caches bypass object
    // reuse and always enter the RHI; those callers own binary-cache saving.
    [[nodiscard]] Result<std::unique_ptr<ComputePipeline>> getComputePipeline(Device& device, const ComputePipelineDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<RayTracingPipeline>> getRayTracingPipeline(Device& device, const RayTracingPipelineDesc& desc);
    [[nodiscard]] Result<std::unique_ptr<GraphicsPipeline>> getGraphicsPipeline(Device& device, const GraphicsPipelineDesc& desc);
    // Native interop adapters with external descriptor/vertex ABIs still use
    // automatic cache selection and persistence. The factory records its full
    // stable state identity through the backend's cached native creation API.
    [[nodiscard]] Result<> getExternalGraphicsPipeline(Device& device, uint64_t shaderHash,
        const std::function<Result<>(PipelineCache&)>& factory);
    // Linked stages persist as one driver-binary pair. Null directory selects
    // the registry cache; an explicit directory supports isolated RHI tests.
    [[nodiscard]] Result<std::unique_ptr<GraphicsShaderObjectProgram>> getGraphicsShaderObjectProgram(
        Device& device, const GraphicsShaderObjectProgramDesc& desc);
    [[nodiscard]] Result<std::vector<ShaderRegistryCacheStats>> pipelineCacheStats(Device& device);
    [[nodiscard]] Result<NativePipelineCacheStats> nativePipelineCacheStats(Device& device);
    // Capacity counts native objects across compute and graphics. Zero keeps
    // weak reuse/single-flight while disabling strong retention; default 128.
    [[nodiscard]] Result<> setNativePipelineCacheCapacity(Device& device, size_t strongCapacity);
    // Automatic saves run on the device-owned worker. This explicit durability
    // barrier blocks until pending attempts finish; ordinary acquisition never
    // calls it. Device teardown drains the worker before native cache teardown.
    [[nodiscard]] Result<> flushPipelineCaches(Device& device);

private:
    friend struct ShaderRegistryIdentityTestAccess;
    friend class ShaderArtifactOverrideScope;
    ShaderRegistry() = default;
    void removeArtifactOverride(uint64_t installation) noexcept;
    // Normal compiler/warmup result registration; never the testing override path.
    void registerCompiledShader(const SlangShaderDesc& desc, const ShaderCompileResult& compiled);
    std::string cacheGroup(uint64_t shaderHash);
    void registerShaderModule(std::span<const uint32_t> spirv, uint64_t deviceShaderHash);
    PipelineRecipeShader recipeShader(std::span<const uint32_t> spirv);
    std::mutex sourceMutex_;
    std::unordered_map<uint64_t, std::string> sourceGroups_;
    std::unordered_map<uint64_t, PipelineRecipeShader> sourceIdentities_;
    std::shared_ptr<ShaderArtifactOverride> artifactOverride_;
    uint64_t artifactOverrideInstallation_ = 0;
    uint64_t nextArtifactOverrideInstallation_ = 0;
};

} // namespace metallic::render

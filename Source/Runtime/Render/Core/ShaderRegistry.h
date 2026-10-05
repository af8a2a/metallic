#pragma once

#include "Runtime/Render/Core/SlangCompiler.h"

#include <mutex>
#include <unordered_map>

namespace metallic::render {

struct ShaderRegistryCacheStats {
    std::string group;
    PipelineCacheStats cache;
};

class ComputeKernel;
struct ComputeKernelDesc;
class ComputeProgram;
struct ComputeProgramDesc;

// The process singleton owns source identities only. Native cache state belongs
// to Device::sharedState and is released before that device's native teardown.
class ShaderRegistry {
public:
    static ShaderRegistry& instance();
    ShaderRegistry(const ShaderRegistry&) = delete;
    ShaderRegistry& operator=(const ShaderRegistry&) = delete;

    // Always validates dependencies through Slang's existing disk-cache path;
    // unchanged/precompiled code is reused, missing or stale code is compiled.
    [[nodiscard]] Result<ShaderCompileResult> getShader(const SlangShaderDesc& desc, std::string& log);
    [[nodiscard]] Result<ShaderCompileResult> getShader(const SlangShaderDesc& desc,
        const SlangShaderCacheOptions& options, std::string& log);
    [[nodiscard]] Result<std::unique_ptr<ShaderModule>> getShaderModule(Device& device, const ShaderModuleDesc& desc);
    // Resolve source, acquire its PSO and publish only a successful generation.
    // layout.spirv is replaced by the requested source's code.
    [[nodiscard]] Result<> getComputeKernel(Device& device, const SlangShaderDesc& source,
        const ComputeKernelDesc& layout, ComputeKernel& kernel, std::string& log);
    [[nodiscard]] Result<> getComputeProgram(Device& device, const SlangShaderDesc& source,
        const ComputeProgramDesc& layout, ComputeProgram& program, std::string& log);
    // A null desc.pipelineCache automatically uses a device-owned persistent
    // cache selected from shader identity. Explicit RHI caches remain supported
    // for callers testing/controlling their own cache; those callers own saving.
    [[nodiscard]] Result<std::unique_ptr<ComputePipeline>> getComputePipeline(Device& device, const ComputePipelineDesc& desc);
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

private:
    ShaderRegistry() = default;
    std::string cacheGroup(uint64_t shaderHash);
    void registerShaderModule(std::span<const uint32_t> spirv, uint64_t deviceShaderHash);
    std::mutex sourceMutex_;
    std::unordered_map<uint64_t, std::string> sourceGroups_;
};

} // namespace metallic::render

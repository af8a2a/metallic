#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"

#include <memory>
#include <span>
#include <string>

namespace metallic::render {

struct RayTracingKernelDesc {
    std::span<const uint32_t> rayGenerationSpirv;
    std::span<const uint32_t> missSpirv;
    std::span<const uint32_t> closestHitSpirv;
    std::span<const uint32_t> anyHitSpirv;
    ParameterABI parameters;
    const char* debugName = nullptr;
    // Null automatically selects ShaderRegistry's persistent cache.
    PipelineCache* pipelineCache = nullptr;
    // Path-tracing policy belongs to ray generation; hit/miss stages are shared.
    std::span<const ShaderSpecializationConstant> specializationConstants;
    bool enableInvocationReorder = false;
};

class RayTracingKernel;

// Retains the parameters, all four shader modules, native pipeline and its SBT.
// Frame-scoped parameters can only be recorded in their original generation.
class PreparedRayTracingDispatch {
public:
    bool valid() const { return impl_ != nullptr; }
    Result<> record(CommandBuffer& commands) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    friend class RayTracingKernel;
};

// A fixed ray-generation/miss/triangle-hit pipeline with a shared parameter ABI.
// Resource identity and allocation lifetime remain owned by ResourceRegistry.
class RayTracingKernel {
public:
    Result<> initialize(Device& device, const RayTracingKernelDesc& desc, std::string& log);
    bool valid() const { return impl_ != nullptr; }
    void clear() { impl_.reset(); }
    ParameterABI parameterABI() const;
    // Dimensions are ray launch pixels, never compute workgroup counts.
    [[nodiscard]] Result<PreparedRayTracingDispatch> prepareDispatch(const EncodedParameters& params,
        uint32_t width, uint32_t height = 1, uint32_t depth = 1) const;
    Result<> dispatch(CommandBuffer& commands, const EncodedParameters& params,
        uint32_t width, uint32_t height = 1, uint32_t depth = 1) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render

#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"

#include <memory>
#include <span>
#include <string>

namespace metallic::render {

struct ComputeKernelDesc {
    std::span<const uint32_t> spirv;
    ParameterABI parameters;
    const char* debugName = nullptr;
    PipelineCache* pipelineCache = nullptr;
};

class ComputeKernel;

// Immutable dispatches retain parameters, executable state and indirect allocations.
// Frame-scoped parameters can only be recorded in their original frame generation.
class PreparedComputeDispatch {
public:
    bool valid() const { return impl_ != nullptr; }
    Result<> record(CommandBuffer& commands, const BarrierDesc& betweenDispatches = {}) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    friend class ComputeKernel;
};

struct ComputeIndirectParameters {
    EncodedParameters parameters;
    BufferSlice arguments;
    // Optional permutation; its parameter ABI must match this kernel.
    const ComputeKernel* kernel = nullptr;
};

// Owns executable code and its parameter ABI only. Resource identity belongs to ResourceRegistry.
class ComputeKernel {
public:
    Result<> initialize(Device& device, const ComputeKernelDesc& desc, std::string& log);
    bool valid() const { return impl_ != nullptr; }
    void clear() { impl_.reset(); }
    [[nodiscard]] Result<PreparedComputeDispatch> prepareDispatch(const EncodedParameters& params,
        uint32_t x, uint32_t y = 1, uint32_t z = 1) const;
    [[nodiscard]] Result<PreparedComputeDispatch> prepareIndirectBatch(
        std::span<const ComputeIndirectParameters> dispatches) const;
    Result<> dispatch(CommandBuffer& commands, const EncodedParameters& params,
        uint32_t x, uint32_t y = 1, uint32_t z = 1) const;
    Result<> dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params, const BufferSlice& arguments) const;
    Result<> dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params,
        Buffer& arguments, uint64_t offset = 0) const;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render

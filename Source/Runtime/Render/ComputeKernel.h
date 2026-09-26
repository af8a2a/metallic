#pragma once

#include "Runtime/Render/ResourceRegistry.h"

#include <memory>
#include <span>
#include <string>

namespace metallic::render {

struct ComputeKernelDesc {
    std::span<const uint32_t> spirv;
    ParameterAbi parameters;
    const char* debugName = nullptr;
    PipelineCache* pipelineCache = nullptr;
};

// Owns executable code and its parameter ABI only. Resource identity belongs to ResourceRegistry.
class ComputeKernel {
public:
    Result initialize(Device& device, const ComputeKernelDesc& desc, std::string& log);
    bool valid() const { return impl_ != nullptr; }
    void clear() { impl_.reset(); }
    Result dispatch(CommandBuffer& commands, const EncodedParameters& params,
        uint32_t x, uint32_t y = 1, uint32_t z = 1) const;
    Result dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params, const BufferSlice& arguments) const;
    Result dispatchIndirect(CommandBuffer& commands, const EncodedParameters& params,
        Buffer& arguments, uint64_t offset = 0) const;
private:
    Result bind(CommandBuffer& commands, const EncodedParameters& params) const;
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render

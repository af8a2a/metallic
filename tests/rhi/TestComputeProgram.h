#pragma once
#include "Runtime/Render/Core/ComputeResourceEncoder.h"
#include "Runtime/Render/Material/MaterialExecutable.h"

namespace metallic::render {
class ComputeProgram;
struct TestComputeIndirectDispatch {
    const void* pushData = nullptr;
    uint64_t argumentOffset = 0;
    const ComputeProgram* program = nullptr;
};

// Test fixture only: production owns and passes executable and input layout separately.
class ComputeProgram {
public:
    ComputeKernel kernel;
    ComputeResourceEncoder encoder;
    Result<> initialize(Device& device, const ResourceComputeKernelDesc& desc, std::string& log)
    {
        clear();
        return initializeResourceKernel(device, desc, kernel, encoder, log);
    }
    bool valid() const { return kernel.valid() && encoder.valid(); }
    void clear() { kernel.clear(); encoder.clear(); }
    ComputeProgram share() const { return *this; }
    Result<> dispatch(const ComputeDispatchDesc& desc)
    {
        return dispatchResources(kernel, encoder, desc);
    }
    Result<PreparedComputeDispatch> prepareDispatch(RenderFrameContext& frame, const ComputeDispatchDesc& desc) const
    {
        return prepareResourceDispatch(kernel, encoder, &frame, desc);
    }
    static std::vector<ComputeIndirectDispatch> convert(std::span<const TestComputeIndirectDispatch> items)
    {
        std::vector<ComputeIndirectDispatch> result;
        for (const auto& item : items) {
            result.push_back({item.pushData, item.argumentOffset,
                item.program ? &item.program->kernel : nullptr, item.program ? &item.program->encoder : nullptr});
        }
        return result;
    }
    Result<PreparedComputeDispatch> prepareIndirectBatch(RenderFrameContext& frame, const ComputeDispatchDesc& desc,
        std::span<const TestComputeIndirectDispatch> items) const
    {
        if (items.empty() || !desc.indirectArguments) { return makeError(Error::InvalidArgument); }
        return prepareResourceDispatch(kernel, encoder, &frame, desc, convert(items));
    }
    Result<> dispatchIndirectBatch(const ComputeDispatchDesc& desc, std::span<const TestComputeIndirectDispatch> items,
        const BarrierDesc& barriers = {}) const
    {
        if (items.empty() || !desc.indirectArguments) { return makeError(Error::InvalidArgument); }
        return dispatchResources(kernel, encoder, desc, convert(items), barriers);
    }
};

inline Result<> compileMaterialExecutable(Device& device, const SlangShaderDesc& source,
    const ResourceComputeKernelDesc& layout, ComputeProgram& program,
    std::shared_ptr<const MaterialExecutableArtifact>& artifact, std::string& log,
    MaterialProgramCompileOptions options = {})
{
    auto result = compileMaterialExecutable(device, source, layout, program.kernel, artifact, log, options);
    if (result) { program.encoder = artifact->encoder; }
    return result;
}

inline Result<> initializeMaterialErrorProgram(Device& device, ComputeProgram& program, std::string& log)
{
    return initializeMaterialErrorProgram(device, program.kernel, program.encoder, log);
}
} // namespace metallic::render

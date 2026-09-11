#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace metallic::render {

enum class ComputeResourceBindingKind : uint8_t {
    AccelerationStructure,
    PartitionedAccelerationStructure,
    StorageImage,
    StorageBuffer,
    SampledImage,
    Sampler,
};

struct ComputeProgramBindingDesc {
    // Application resource-table slot, not a Vulkan descriptor binding.
    uint32_t binding = 0;
    ComputeResourceBindingKind kind = ComputeResourceBindingKind::StorageBuffer;
    uint32_t descriptorCount = 1;
};

struct ComputeProgramDesc {
    const uint32_t* spirv = nullptr;
    uint64_t byteSize = 0;
    uint32_t pushConstantSize = 0;
    const ComputeProgramBindingDesc* bindings = nullptr;
    uint32_t bindingCount = 0;
    const char* debugName = nullptr;
    uint32_t resourceTableCount = 1;
    bool requiresRayQuery = true;
    // Native DescriptorHandle shaders include ComputeResources.slang. The
    // mapped path is retained only for explicit legacy shader diagnostics.
    bool usesResourceTable = true;
};

struct ComputeDispatchBinding {
    uint32_t binding = 0;
    union {
        RayTracingAccelerationStructure* accelerationStructure = nullptr;
        const SamplerDesc* sampler;
    };
    PartitionedAccelerationStructure* partitionedAccelerationStructure = nullptr;
    TextureView* textureView = nullptr;
    TextureView* const* textureViews = nullptr;
    uint32_t textureViewCount = 0;
    Buffer* buffer = nullptr;
    uint64_t offset = 0;
    uint64_t size = UINT64_MAX;
};

struct ComputeDispatchDesc {
    CommandBuffer* commandBuffer = nullptr;
    const ComputeDispatchBinding* bindings = nullptr;
    uint32_t bindingCount = 0;
    const void* pushData = nullptr;
    uint32_t pushDataSize = 0;
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
    uint32_t resourceTableIndex = 0;
    // When present, GPU-generated counts replace groupCountX/Y/Z. The caller
    // transitions this buffer to IndirectArgument and retains it until completion.
    Buffer* indirectArguments = nullptr;
    uint64_t indirectOffset = 0;
};

class ComputeProgram;

struct ComputeIndirectDispatch {
    const void* pushData = nullptr;
    uint64_t argumentOffset = 0;
    // Optional permutation with exactly the same descriptor and push-data layout.
    const ComputeProgram* program = nullptr;
};

class ComputeProgram {
public:
    ComputeProgram();
    ~ComputeProgram();

    ComputeProgram(ComputeProgram&&) noexcept;
    ComputeProgram& operator=(ComputeProgram&&) noexcept;

    ComputeProgram(const ComputeProgram&) = delete;
    ComputeProgram& operator=(const ComputeProgram&) = delete;

    Result initialize(Device& device, const ComputeProgramDesc& desc, std::string& log);
    void clear();
    bool valid() const;
    Result dispatch(const ComputeDispatchDesc& desc);
    // Bind one immutable descriptor table for the batch. Every item supplies
    // pushDataSize bytes and an offset into desc.indirectArguments. Optional
    // barriers separate dispatches sharing writable resources. Compatible per-item
    // programs share this table; the entire batch is validated before recording.
    Result dispatchIndirectBatch(const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches = {});

private:
    Result dispatchImpl(const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches);
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render

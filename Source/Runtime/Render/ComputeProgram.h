#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

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
    // Native DescriptorHandle shaders import the Core module. The
    // mapped path is retained only for explicit legacy shader diagnostics.
    bool usesResourceTable = true;
    // Optional cache borrowed only during pipeline creation.
    PipelineCache* pipelineCache = nullptr;
};

struct CpuProfileRecorder;

// Publish through shared_ptr<const ...> and never mutate afterwards. The owner
// retains the underlying images, while views supply stable ownership identities.
struct ComputeSampledImageSnapshot {
    std::shared_ptr<const void> owner;
    std::vector<std::shared_ptr<TextureView>> views;
};

struct ComputeDispatchStats {
    uint32_t sampledImageWrites = 0;
    uint32_t sampledImageCacheHits = 0;
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
    // Optional immutable sampled-image array; takes precedence over textureViews.
    // The shared registry also deduplicates individual resource registrations.
    std::shared_ptr<const ComputeSampledImageSnapshot> sampledImages;
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
    CpuProfileRecorder* profiler = nullptr;
    ComputeDispatchStats* stats = nullptr;
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
    Result dispatchShared(const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches);
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render

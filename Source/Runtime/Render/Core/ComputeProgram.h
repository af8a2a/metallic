#pragma once

#include "Runtime/Render/Core/ComputeKernel.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace metallic::render {

enum class ComputeResourceBindingKind : uint8_t {
    AccelerationStructure,
    StorageImage,
    StorageBuffer,
    DataBuffer,
    SampledImage,
    Sampler,
};

struct ComputeProgramBindingDesc {
    // CPU input ID. Named layouts map it to a field; legacy tests use it as a slot.
    uint32_t binding = 0;
    ComputeResourceBindingKind kind = ComputeResourceBindingKind::StorageBuffer;
    uint32_t descriptorCount = 1;
    // DataBuffer only: explicit element ABI for a descriptor-backed buffer span.
    uint32_t dataStride = 0;
    uint32_t dataAlignment = 0;
    bool operator==(const ComputeProgramBindingDesc&) const = default;
};

enum class ComputeResourceFieldFormat : uint8_t { Handle, IndexSpan, DataSpan };

struct ComputeResourceField {
    uint32_t binding = 0;
    ComputeResourceBindingKind kind = ComputeResourceBindingKind::StorageBuffer;
    uint32_t offset = 0;
    ComputeResourceFieldFormat format = ComputeResourceFieldFormat::Handle;
    bool operator==(const ComputeResourceField&) const = default;
};

struct ComputeResourceLayout {
    uint32_t size = 0;
    std::span<const ComputeResourceField> fields;
};

struct ComputeProgramDesc {
    std::span<const uint32_t> spirv;
    uint32_t pushConstantSize = 0;
    std::span<const ComputeProgramBindingDesc> bindings;
    const char* debugName = nullptr;
    bool requiresRayQuery = true;
    // Optional cache borrowed only during pipeline creation.
    PipelineCache* pipelineCache = nullptr;
    // Direct CPU/Slang resource struct. Empty preserves the low-level test adapter.
    ComputeResourceLayout resourceParameters;
};

struct CPUProfileRecorder;

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
    TextureView* textureView = nullptr;
    std::span<TextureView* const> textureViews;
    Buffer* buffer = nullptr;
    BufferRange range;
    // DataBuffer only. Supply either a slice or buffer/range, never both.
    BufferSlice data;
    // Optional immutable sampled-image array; takes precedence over textureViews.
    // The shared registry also deduplicates individual resource registrations.
    std::shared_ptr<const ComputeSampledImageSnapshot> sampledImages;
};

struct ComputeDispatchDesc {
    CommandBuffer* commandBuffer = nullptr;
    std::span<const ComputeDispatchBinding> bindings;
    const void* pushData = nullptr;
    uint32_t pushDataSize = 0;
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
    // When present, GPU-generated counts replace groupCountX/Y/Z. The caller
    // transitions this buffer to IndirectArgument and retains it until completion.
    Buffer* indirectArguments = nullptr;
    uint64_t indirectOffset = 0;
    CPUProfileRecorder* profiler = nullptr;
    ComputeDispatchStats* stats = nullptr;
};

class ComputeProgram;
class RenderFrameContext;

struct ComputeIndirectDispatch {
    const void* pushData = nullptr;
    uint64_t argumentOffset = 0;
    // Optional permutation with exactly the same descriptor and push-data layout.
    const ComputeProgram* program = nullptr;
};

// CPU binding adapter for Core resource parameters. ComputeKernel owns execution.
class ComputeProgram {
public:
    ComputeProgram();
    ~ComputeProgram();

    ComputeProgram(ComputeProgram&&) noexcept;
    ComputeProgram& operator=(ComputeProgram&&) noexcept;

    ComputeProgram(const ComputeProgram&) = delete;
    ComputeProgram& operator=(const ComputeProgram&) = delete;

    Result<> initialize(Device& device, const ComputeProgramDesc& desc, std::string& log);
    void clear();
    bool valid() const;
    Result<> dispatch(const ComputeDispatchDesc& desc);
    // No command buffer access. Concurrent preparations require stable program,
    // input wrappers and frame generation until all jobs join. A packet is returned
    // only on success. Encoding delegates execution to ComputeKernel.
    [[nodiscard]] Result<PreparedComputeDispatch> prepareDispatch(
        RenderFrameContext& frame,
        const ComputeDispatchDesc& desc) const;
    [[nodiscard]] Result<PreparedComputeDispatch> prepareIndirectBatch(
        RenderFrameContext& frame,
        const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches) const;
    // Bind one immutable descriptor table for the batch. Every item supplies
    // pushDataSize bytes and an offset into desc.indirectArguments. Optional
    // barriers separate dispatches sharing writable resources. Compatible per-item
    // programs share this table; the entire batch is validated before recording.
    Result<> dispatchIndirectBatch(const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches = {});

private:
    Result<> validateDispatch(const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches) const;
    [[nodiscard]] Result<PreparedComputeDispatch> prepare(
        RenderFrameContext* frame,
        const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches) const;
    Result<> dispatchImpl(const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches);
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace metallic::render

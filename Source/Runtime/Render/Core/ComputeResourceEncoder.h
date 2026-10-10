#pragma once

#include "Runtime/Render/Core/ComputeKernel.h"

#include <cstdint>
#include <compare>
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

enum class ResourceMemberFormat : uint8_t { Handle, IndexSpan, DataSpan };

// A reference to a shared parameter member, never an arbitrary numeric slot.
class ComputeResourceMember {
public:
    constexpr ComputeResourceMember() = default;
    constexpr bool valid() const { return key_ != UINT32_MAX; }
    constexpr uint32_t offset() const { return key_ & 0xffffu; }
    constexpr ComputeResourceBindingKind kind() const { return ComputeResourceBindingKind((key_ >> 16) & 0xffu); }
    constexpr ResourceMemberFormat format() const { return ResourceMemberFormat(key_ >> 24); }
    constexpr uint32_t key() const { return key_; }
    auto operator<=>(const ComputeResourceMember&) const = default;
private:
    template<typename Owner, typename Member, bool Data>
    friend consteval ComputeResourceMember resourceMember(uint32_t offset);
    constexpr explicit ComputeResourceMember(uint32_t key) : key_(key) {}
    uint32_t key_ = UINT32_MAX;
};

struct ComputeResourceBindingDesc {
    // Construct with METALLIC_RESOURCE_MEMBER / METALLIC_DATA_MEMBER.
    ComputeResourceMember binding;
    ComputeResourceBindingKind kind = ComputeResourceBindingKind::StorageBuffer;
    uint32_t descriptorCount = 1;
    // DataBuffer only: explicit element ABI for a descriptor-backed buffer span.
    uint32_t dataStride = 0;
    uint32_t dataAlignment = 0;
    // A missing dispatch input retains the invalid named-field sentinel.
    // Present inputs still require a valid allocation and matching resource kind.
    bool optional = false;
    bool operator==(const ComputeResourceBindingDesc&) const = default;
};

// Creation description for dynamic resource manifests. New static passes use typed parameters.
struct ResourceComputeKernelDesc {
    std::span<const uint32_t> spirv;
    uint32_t pushConstantSize = 0;
    std::span<const ComputeResourceBindingDesc> bindings;
    const char* debugName = nullptr;
    bool requiresRayQuery = true;
    // Optional explicit cache borrowed during creation. Null uses ShaderRegistry.
    PipelineCache* pipelineCache = nullptr;
    // Required direct CPU/Slang resource struct. There is no implicit slot layout.
    uint32_t resourceParameterSize = 0;
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
    ComputeResourceMember binding;
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

class ComputeResourceEncoder;

struct ComputeIndirectDispatch {
    const void* pushData = nullptr;
    uint64_t argumentOffset = 0;
    // A permutation must supply both its executable and its resource contract.
    const ComputeKernel* kernel = nullptr;
    const ComputeResourceEncoder* encoder = nullptr;
};

// Immutable CPU input layout only. Owns no shader, pipeline or executable.
// Retained for dynamic resource manifests; new passes encode typed parameters directly.
class ComputeResourceEncoder {
public:
    Result<> initialize(Device& device, const ResourceComputeKernelDesc& desc, std::string& log);
    bool valid() const { return impl_ != nullptr; }
    void clear() { impl_.reset(); }
    bool compatible(const ComputeResourceEncoder& other) const;
    ParameterABI parameterABI() const;
    [[nodiscard]] Result<std::vector<EncodedParameters>> encode(
        RenderFrameContext* frame, const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches = {}) const;
private:
    Result<> validate(const ComputeDispatchDesc& desc,
        std::span<const ComputeIndirectDispatch> dispatches) const;
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

// Transactionally publish the executable and its input encoder together.
Result<> initializeResourceKernel(Device& device, const ResourceComputeKernelDesc& desc,
    ComputeKernel& kernel, ComputeResourceEncoder& encoder, std::string& log);
[[nodiscard]] Result<PreparedComputeDispatch> prepareResourceDispatch(
    const ComputeKernel& kernel, const ComputeResourceEncoder& encoder,
    RenderFrameContext* frame, const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches = {});
Result<> dispatchResources(const ComputeKernel& kernel, const ComputeResourceEncoder& encoder,
    const ComputeDispatchDesc& desc, std::span<const ComputeIndirectDispatch> dispatches = {},
    const BarrierDesc& betweenDispatches = {});

} // namespace metallic::render

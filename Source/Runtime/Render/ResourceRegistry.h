#pragma once

#include "Runtime/Render/GAPI/Rhi.h"
#include "Runtime/Render/RenderFrameContext.h"

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace metallic::render {

enum class ShaderResourceKind : uint8_t { Buffer, SampledImage, StorageImage, Sampler, AccelerationStructure, PartitionedAccelerationStructure };

// GPU wire values are independent of the CPU allocator's BindlessHandle.
template<ShaderResourceKind Kind>
struct ShaderResourceHandle {
    uint64_t value = UINT64_MAX;
};
using ShaderBuffer = ShaderResourceHandle<ShaderResourceKind::Buffer>;
using ShaderSampledImage = ShaderResourceHandle<ShaderResourceKind::SampledImage>;
using ShaderStorageImage = ShaderResourceHandle<ShaderResourceKind::StorageImage>;
using ShaderSampler = ShaderResourceHandle<ShaderResourceKind::Sampler>;
using ShaderAccelerationStructure = ShaderResourceHandle<ShaderResourceKind::AccelerationStructure>;

// Matches Slang DataSpan<T>. No descriptor is allocated for ordinary data.
struct ShaderDataSpan {
    uint64_t address = 0;
    uint32_t count = 0;
    uint32_t stride = 0;
};
static_assert(sizeof(ShaderDataSpan) == 16);

struct ParameterAbi {
    uint64_t id = 0;
    uint32_t size = 0;
    uint32_t alignment = 0;
    bool operator==(const ParameterAbi&) const = default;
};

template<typename T>
constexpr ParameterAbi parameterAbi(uint64_t id)
{
    static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>);
    return {id, sizeof(T), alignof(T)};
}

namespace detail { struct RegistryState; struct ResourceLeaseState; struct ParameterPacket; }

class ResourceLease {
public:
    bool valid() const { return state_ != nullptr; }
    uint64_t shaderValue() const;
    uint32_t shaderIndex() const { return static_cast<uint32_t>(shaderValue()); }
    ShaderResourceKind kind() const;
private:
    std::shared_ptr<detail::ResourceLeaseState> state_;
    friend class ResourceRegistry;
    friend class ParameterWriter;
};

class EncodedParameters {
public:
    bool valid() const { return packet_ != nullptr; }
    ParameterAbi abi() const;
    uint64_t address() const;
    bool compatible(const CommandBuffer& commands, ParameterAbi abi) const;
private:
    Result bindResources(CommandBuffer& commands) const;
    std::shared_ptr<detail::ParameterPacket> packet_;
    friend class ParameterWriter;
    friend class ComputeKernel;
    friend class ComputeProgram;
};

struct ResourceRegistryStats {
    uint64_t descriptorWrites = 0;
    uint64_t cacheHits = 0;
    uint64_t liveDescriptors = 0;
    uint64_t parameterBytes = 0;
    uint64_t parameterCapacity = 0;
};

// One immutable-index heap epoch. Capacity exhaustion is explicit; never relocates live indices.
// Device must outlive the registry, all leases and all submitted parameter packets.
class ResourceRegistry {
public:
    ResourceRegistry() = default;
    Result initialize(Device& device, const BindlessHeapDesc& capacity = {
        .maxSamplers = 64, .maxSampledImages = 8192, .maxStorageImages = 1024, .maxBuffers = 8192});
    Result storageBuffer(Buffer& buffer, ResourceLease& out);
    Result sampledImage(TextureView& view, ResourceLease& out, ResourceState layout = ResourceState::ShaderRead);
    Result storageImage(TextureView& view, ResourceLease& out);
    Result sampler(const SamplerDesc& sampler, ResourceLease& out);
    Result accelerationStructure(RayTracingAccelerationStructure& structure, ResourceLease& out);
    Result partitionedAccelerationStructure(PartitionedAccelerationStructure& structure, ResourceLease& out);
    void collect();
    ResourceRegistryStats stats() const;
    // Borrowed heap for prepared raster/SDK pipelines. Only registry registration writes descriptors.
    BindlessHeap* heap() const;
    Result bind(CommandBuffer& commands) const;
    Result retain(CommandBuffer& commands, const ResourceLease& lease) const;
private:
    Result image(TextureView& view, ResourceLease& out, ShaderResourceKind kind, ResourceState layout);
    std::shared_ptr<detail::RegistryState> state_;
    friend class ParameterWriter;
};

// Resources must be registered/leased through this writer before encoding their wire handles.
// Each registration contributes a strong allocation lease. The writer belongs to one submission.
// The first error is sticky, so a failed registration cannot publish a partial packet.
class ParameterWriter {
public:
    ParameterWriter(Device& device, RenderFrameContext& frame, ResourceRegistry& registry);
    ShaderBuffer buffer(Buffer* buffer);
    ShaderDataSpan dataBuffer(const BufferSlice& slice, uint32_t stride, uint32_t alignment);
    ShaderDataSpan dataBuffer(Buffer* buffer, uint32_t stride, uint32_t alignment);
    template<typename T>
    ShaderDataSpan dataBuffer(const BufferSlice& slice)
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        return dataBuffer(slice, sizeof(T), alignof(T));
    }
    ShaderSampledImage sampledImage(TextureView* view, ResourceState layout = ResourceState::ShaderRead);
    ShaderStorageImage storageImage(TextureView* view);
    ShaderSampler sampler(const SamplerDesc& sampler);
    ShaderAccelerationStructure accelerationStructure(RayTracingAccelerationStructure* structure);
    // Immutable GPU array of uint2 handles, with no contiguous descriptor allocation requirement.
    uint64_t sampledImages(std::span<TextureView* const> views);
    // Immutable ABI payload owned by the same submission as the root packet.
    uint64_t data(const void* bytes, uint64_t size, uint64_t alignment = 16);
    Result use(const ResourceLease& lease);
    // Retain transitive owners (for example a scene's BLAS set behind a TLAS).
    void retain(std::shared_ptr<void> owner) { if (owner) { arrays_.push_back(std::move(owner)); } }
    Result status() const { return result_; }

    template<typename T>
    Result encode(const T& params, uint64_t abiId, EncodedParameters& out)
    {
        return encodeBytes(&params, parameterAbi<T>(abiId), out);
    }
private:
    Result encodeBytes(const void* params, ParameterAbi abi, EncodedParameters& out);
    Result upload(const void* data, uint64_t size, uint64_t alignment,
        uint64_t& address, std::shared_ptr<void>& allocation);
    uint64_t append(Result result, ResourceLease lease);
    Device& device_;
    RenderFrameContext& frame_;
    GpuCompletionPoint completion_;
    std::shared_ptr<detail::RegistryState> registry_;
    Result result_;
    std::vector<std::shared_ptr<detail::ResourceLeaseState>> resources_;
    std::vector<std::shared_ptr<void>> arrays_;
};

} // namespace metallic::render

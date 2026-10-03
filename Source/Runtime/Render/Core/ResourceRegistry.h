#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ShaderResourceABI.h"

#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

namespace metallic::render {

enum class ShaderResourceKind : uint8_t { Buffer, SampledImage, StorageImage, Sampler, AccelerationStructure };

using ShaderBuffer = GPUResourceHandle<ResourceViewKind::RawBuffer>;
using ShaderSampledImage = GPUResourceHandle<ResourceViewKind::SampledImage>;
using ShaderStorageImage = GPUResourceHandle<ResourceViewKind::StorageImage>;
using ShaderSampler = GPUSamplerHandle;
// AS retains the explicitly documented Vulkan device-address transport.
struct ShaderAccelerationStructure { uint64_t value = UINT64_MAX; };

enum class ParameterTransport : uint8_t { DescriptorBuffer, InlinePush };

struct ParameterABI {
    uint64_t id = 0;
    uint32_t size = 0;
    uint32_t alignment = 0;
    ParameterTransport transport = ParameterTransport::DescriptorBuffer;
    bool operator==(const ParameterABI&) const = default;
};

template<typename T>
constexpr ParameterABI parameterAbi(uint64_t id, ParameterTransport transport = ParameterTransport::DescriptorBuffer)
{
    static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>);
    return {id, sizeof(T), alignof(T), transport};
}

namespace detail { struct RegistryState; struct ResourceLeaseState; struct ParameterPacket; struct ParameterChunk; }

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
    ParameterABI abi() const;
    GPUBufferSpan root() const;
    std::span<const uint8_t> inlineData() const;
    const void* deviceIdentity() const;
    bool compatible(const CommandBuffer& commands, ParameterABI abi) const;
    // Also usable by raw bindless raster/compute paths: retain this immutable
    // packet locally and bind its registry without changing execution state.
    Result<> bindResources(CommandBuffer& commands) const;
private:
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
    [[nodiscard]] static Result<std::shared_ptr<ResourceRegistry>> forDevice(Device& device);
    ResourceRegistry() = default;
    Result<> initialize(Device& device, const BindlessHeapDesc& capacity = {
        .maxSamplers = 64, .maxSampledImages = 8192, .maxStorageImages = 1024, .maxBuffers = 8192});
    [[nodiscard]] Result<ResourceLease> storageBuffer(Buffer& buffer);
    [[nodiscard]] Result<ResourceLease> storageBuffer(const BufferSlice& buffer);
    [[nodiscard]] Result<ResourceLease> sampledImage(
        TextureView& view,
        ResourceState layout = ResourceState::ShaderRead,
        bool* descriptorWritten = nullptr);
    [[nodiscard]] Result<ResourceLease> storageImage(TextureView& view);
    [[nodiscard]] Result<ResourceLease> sampler(const SamplerDesc& sampler);
    [[nodiscard]] Result<ResourceLease> accelerationStructure(RayTracingAccelerationStructure& structure);
    void collect();
    ResourceRegistryStats stats() const;
    // Borrowed heap for prepared raster/SDK pipelines. Only registry registration writes descriptors.
    BindlessHeap* heap() const;
    // Immutable provenance check for assembling packets without a command buffer.
    bool owns(const ResourceLease& lease) const;
    Result<> bind(CommandBuffer& commands) const;
    Result<> retain(CommandBuffer& commands, const ResourceLease& lease) const;
private:
    [[nodiscard]] Result<ResourceLease> image(
        TextureView& view,
        ShaderResourceKind kind,
        ResourceState layout,
        bool* descriptorWritten = nullptr);
    std::shared_ptr<detail::RegistryState> state_;
    friend class ParameterWriter;
};

// Resources must be registered/leased through this writer before encoding their wire handles.
// Each registration contributes a strong allocation lease. Frame writers use a submission arena;
// standalone writers own their storage and can be used by commands without a frame.
// The first error is sticky, so a failed registration cannot publish a partial packet.
class ParameterWriter {
public:
    ParameterWriter(Device& device, RenderFrameContext& frame, ResourceRegistry& registry);
    ParameterWriter(Device& device, ResourceRegistry& registry, RenderFrameContext* frame = nullptr);
    ShaderBuffer buffer(Buffer* buffer);
    GPUResourceHandle<ResourceViewKind::SampledImage> sampledImageHandle(
        TextureView* view, ResourceState layout = ResourceState::ShaderRead);
    GPUResourceHandle<ResourceViewKind::StorageImage> storageImageHandle(TextureView* view);
    GPUSamplerHandle samplerHandle(const SamplerDesc& sampler);
    // One full-allocation descriptor is shared by all subranges. Offset/count
    // are checked before publishing the packet; its lease retains the allocation.
    GPUBufferSpan bufferSpan(Buffer* buffer, BufferRange range, uint32_t stride, uint32_t alignment);
    template<typename T>
    GPUBufferSpan bufferSpan(Buffer* buffer, BufferRange range = {})
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        return bufferSpan(buffer, range, sizeof(T), alignof(T));
    }
    GPUBufferSpan bufferSpan(const BufferSlice& slice, uint32_t stride, uint32_t alignment);
    GPUBufferSpan bufferSpan(Buffer* buffer, uint32_t stride, uint32_t alignment);
    template<typename T>
    GPUBufferSpan bufferSpan(const BufferSlice& slice)
    {
        static_assert(std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>);
        return bufferSpan(slice, sizeof(T), alignof(T));
    }
    ShaderSampledImage sampledImage(TextureView* view, ResourceState layout = ResourceState::ShaderRead);
    ShaderStorageImage storageImage(TextureView* view);
    ShaderSampler sampler(const SamplerDesc& sampler);
    ShaderAccelerationStructure accelerationStructure(RayTracingAccelerationStructure* structure);
    // Immutable descriptor-backed array of 32-bit handles, with no contiguous descriptor allocation requirement.
    GPUBufferSpan sampledImages(std::span<TextureView* const> views);
    // Descriptor-backed immutable payload, retained with the encoded packet.
    GPUBufferSpan dataSpan(const void* bytes, uint64_t size, uint32_t stride = 4, uint32_t alignment = 4);
    Result<> use(const ResourceLease& lease);
    // Retain transitive owners (for example a scene's BLAS set behind a TLAS).
    void retain(std::shared_ptr<void> owner) { if (owner) { arrays_.push_back(std::move(owner)); } }
    Result<> status() const { return result_; }

    template<typename T>
    [[nodiscard]] Result<EncodedParameters> encode(const T& params, uint64_t abiId,
        ParameterTransport transport = ParameterTransport::DescriptorBuffer)
    {
        return encodeBytes(&params, parameterAbi<T>(abiId, transport));
    }
private:
    [[nodiscard]] Result<EncodedParameters> encodeBytes(const void* params, ParameterABI abi);
    Result<> upload(const void* data, uint64_t size, uint64_t alignment,
        std::shared_ptr<void>& allocation, BufferSlice& slice);
    uint64_t append(Result<ResourceLease> lease);
    Device& device_;
    RenderFrameContext* frame_ = nullptr;
    GPUCompletionPoint completion_;
    std::shared_ptr<detail::RegistryState> registry_;
    Result<> result_;
    std::vector<std::shared_ptr<detail::ResourceLeaseState>> resources_;
    std::vector<std::shared_ptr<void>> arrays_;
    std::vector<std::shared_ptr<detail::ParameterChunk>> standaloneChunks_;
};

} // namespace metallic::render

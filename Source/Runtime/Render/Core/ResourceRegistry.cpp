#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>

namespace metallic::render {

Result<std::shared_ptr<ResourceRegistry>> ResourceRegistry::forDevice(Device& device)
{
    static const char key = 0;
    auto state = device.sharedState(&key, [&]() -> Result<std::shared_ptr<void>> {
        auto registry = std::make_shared<ResourceRegistry>();
        auto result = registry->initialize(device);
        if (!result) { return makeError(result.error()); }
        return registry;
    });
    if (!state) { return makeError(state.error()); }
    return std::static_pointer_cast<ResourceRegistry>(*state);
}
namespace detail {

struct RegistryEntry {
    ShaderResourceKind kind;
    BindlessHandle handle;
    uint64_t value = 0;
    std::weak_ptr<void> allocation;
    bool permanent = false;
};

struct ParameterChunk {
    std::unique_ptr<Buffer> buffer;
    GPUCompletionPoint completion;
    uint64_t used = 0;
};

struct RegistryState {
    const void* device = nullptr;
    std::unique_ptr<BindlessHeap> heap;
    mutable std::mutex mutex;
    using Key = std::array<uint64_t, 12>;
    std::map<Key, std::shared_ptr<RegistryEntry>> entries;
    std::vector<std::shared_ptr<ParameterChunk>> chunks;
    ResourceRegistryStats stats;

    void collectLocked()
    {
        for (auto it = entries.begin(); it != entries.end();) {
            auto& entry = it->second;
            if (!entry->permanent && entry->allocation.expired() && entry.use_count() == 1) {
                if (entry->handle.valid()) { heap->release(entry->handle); --stats.liveDescriptors; }
                it = entries.erase(it);
            } else { ++it; }
        }
    }
};

struct ResourceLeaseState {
    std::shared_ptr<RegistryState> registry;
    std::shared_ptr<RegistryEntry> entry;
    std::shared_ptr<void> allocation;
};

struct ParameterPacket {
    std::shared_ptr<RegistryState> registry;
    std::shared_ptr<void> allocation;
    std::vector<std::shared_ptr<ResourceLeaseState>> resources;
    std::vector<std::shared_ptr<void>> arrays;
    GPUCompletionPoint completion;
    ParameterABI abi;
    GPUBufferSpan root;
    std::vector<uint8_t> inlineData;
};

} // namespace detail
namespace {

using Key = detail::RegistryState::Key;
Key keyFor(ShaderResourceKind kind, const std::shared_ptr<void>& allocation)
{
    return {uint64_t(kind), reinterpret_cast<uintptr_t>(allocation.get())};
}

Result<std::shared_ptr<detail::ResourceLeaseState>> acquire(const std::shared_ptr<detail::RegistryState>& state, const Key& key,
    ShaderResourceKind kind, std::shared_ptr<void> allocation, bool permanent,
    const std::function<Result<>(detail::RegistryEntry&)>& write)
{
    if (!state || (!allocation && !permanent)) { return makeError(Error::InvalidArgument); }
    std::lock_guard lock(state->mutex);
    auto it = state->entries.find(key);
    // A native allocation address can be reused after its last owner disappears.
    // Check that key directly; sweeping every hit makes scene registration quadratic.
    if (it != state->entries.end() && !it->second->permanent && it->second->allocation.expired()) {
        if (it->second->handle.valid()) { state->heap->release(it->second->handle); --state->stats.liveDescriptors; }
        state->entries.erase(it);
        it = state->entries.end();
    }
    if (it == state->entries.end()) {
        auto entry = std::make_shared<detail::RegistryEntry>();
        entry->kind = kind;
        entry->allocation = allocation;
        entry->permanent = permanent;
        auto result = write(*entry);
        if (hasError(result, Error::OutOfMemory)) {
            if (entry->handle.valid()) { state->heap->release(entry->handle); entry->handle = {}; }
            state->collectLocked();
            result = write(*entry);
        }
        if (!result) {
            if (entry->handle.valid()) { state->heap->release(entry->handle); }
            return makeError(result.error());
        }
        if (entry->handle.valid()) { ++state->stats.descriptorWrites; ++state->stats.liveDescriptors; }
        it = state->entries.emplace(key, std::move(entry)).first;
    } else { ++state->stats.cacheHits; }
    return std::make_shared<detail::ResourceLeaseState>(state, it->second, std::move(allocation));
}

} // namespace

uint64_t ResourceLease::shaderValue() const
{
    return state_ ? state_->entry->value : UINT64_MAX;
}

ShaderResourceKind ResourceLease::kind() const
{
    return state_ ? state_->entry->kind : ShaderResourceKind::Buffer;
}

ParameterABI EncodedParameters::abi() const
{
    return packet_ ? packet_->abi : ParameterABI{};
}

GPUBufferSpan EncodedParameters::root() const
{
    return packet_ ? packet_->root : GPUBufferSpan{};
}

std::span<const uint8_t> EncodedParameters::inlineData() const
{
    return packet_ ? std::span<const uint8_t>(packet_->inlineData) : std::span<const uint8_t>{};
}

const void* EncodedParameters::deviceIdentity() const
{
    return packet_ ? packet_->registry->device : nullptr;
}

bool EncodedParameters::compatible(const CommandBuffer& commands, ParameterABI abi) const
{
    auto* frame = metallic::render::RenderFrameContext::from(commands);
    return packet_ && packet_->abi == abi && commands.recording() &&
        packet_->registry->device == commands.deviceIdentity() &&
        (!packet_->completion.valid() || (frame && frame->recording() &&
            packet_->completion.sameSubmission(frame->completion())));
}

Result<> EncodedParameters::bindResources(CommandBuffer& commands) const
{
    if (!compatible(commands, abi())) { return makeError(Error::InvalidArgument); }
    auto retained = commands.retainResource(packet_);
    if (!retained) { return retained; }
    return commands.bindBindlessHeap(*packet_->registry->heap);
}

Result<> ResourceRegistry::initialize(Device& device, const BindlessHeapDesc& capacity)
{
    if (state_) { return makeError(Error::InvalidArgument); }
    auto state = std::make_shared<detail::RegistryState>();
    auto result = device.createBindlessHeap(capacity).transform([&](auto rhiValue) { state->heap = std::move(rhiValue); });
    if (!result) { return result; }
    state->device = device.identity();
    state_ = std::move(state);
    return {};
}

Result<ResourceLease> ResourceRegistry::storageBuffer(Buffer& buffer)
{
    auto slice = buffer.slice();
    return slice ? storageBuffer(*slice) : makeError(slice.error());
}

Result<ResourceLease> ResourceRegistry::storageBuffer(const BufferSlice& buffer)
{
    if (!state_ || !buffer.valid() || buffer.deviceIdentity() != state_->device ||
        (uint32_t(buffer.allocationDesc().usage) & uint32_t(BufferUsageBits::Storage)) == 0) {
        return makeError(Error::InvalidArgument);
    }
    auto allocation = buffer.retainAllocation();
    return acquire(state_, keyFor(ShaderResourceKind::Buffer, allocation), ShaderResourceKind::Buffer,
        allocation, false, [&](auto& entry) {
            auto result = state_->heap->allocate(BindlessHandleKind::Buffer).transform([&](auto rhiValue) { entry.handle = std::move(rhiValue); });
            if (!result) { return result; }
            entry.value = entry.handle.shaderIndex;
            return state_->heap->writeStorageBuffer(entry.handle, buffer);
        }).transform([](auto state) {
        ResourceLease lease;
        lease.state_ = std::move(state);
        return lease;
    });
}

Result<ResourceLease> ResourceRegistry::image(
    TextureView& view,
    ShaderResourceKind kind,
    TextureLayout layout,
    bool* descriptorWritten)
{
    if (descriptorWritten) { *descriptorWritten = false; }
    if (!state_ || view.deviceIdentity() != state_->device) { return makeError(Error::InvalidArgument); }
    auto allocation = view.retainTexture();
    auto key = keyFor(kind, allocation);
    const auto& desc = view.desc();
    key[2] = uint64_t(desc.format); key[3] = desc.range.baseMip; key[4] = desc.range.mipCount;
    key[5] = desc.range.baseLayer; key[6] = desc.range.layerCount; key[7] = uint64_t(layout);
    for (size_t i = 0; i < 4; ++i) { key[8 + i] = uint64_t(desc.swizzle[i]); }
    return acquire(state_, key, kind, allocation, false, [&](auto& entry) {
        auto result = state_->heap->allocate(kind == ShaderResourceKind::SampledImage
            ? BindlessHandleKind::SampledImage : BindlessHandleKind::StorageImage)
            .transform([&](auto rhiValue) { entry.handle = std::move(rhiValue); });
        if (!result) { return result; }
        entry.value = entry.handle.shaderIndex;
        result = kind == ShaderResourceKind::SampledImage ? state_->heap->writeSampledImage(entry.handle, view, layout)
            : state_->heap->writeStorageImage(entry.handle, view);
        if (descriptorWritten) { *descriptorWritten = bool(result); }
        return result;
    }).transform([](auto state) {
        ResourceLease lease;
        lease.state_ = std::move(state);
        return lease;
    });
}

Result<ResourceLease> ResourceRegistry::sampledImage(TextureView& view, TextureLayout layout, bool* descriptorWritten)
{
    if (descriptorWritten) { *descriptorWritten = false; }
    if (layout != TextureLayout::ShaderRead && layout != TextureLayout::General) {
        return makeError(Error::InvalidArgument);
    }
    return image(view, ShaderResourceKind::SampledImage, layout, descriptorWritten);
}

Result<ResourceLease> ResourceRegistry::storageImage(TextureView& view)
{
    return image(view, ShaderResourceKind::StorageImage, TextureLayout::General);
}

Result<ResourceLease> ResourceRegistry::sampler(const SamplerDesc& sampler)
{
    Key key{uint64_t(ShaderResourceKind::Sampler), uint64_t(sampler.minFilter), uint64_t(sampler.magFilter),
        uint64_t(sampler.mipFilter), uint64_t(sampler.addressU), uint64_t(sampler.addressV), uint64_t(sampler.addressW),
        std::bit_cast<uint32_t>(sampler.minLod), std::bit_cast<uint32_t>(sampler.maxLod)};
    return acquire(state_, key, ShaderResourceKind::Sampler, {}, true, [&](auto& entry) {
        auto result = state_->heap->allocate(BindlessHandleKind::Sampler).transform([&](auto rhiValue) { entry.handle = std::move(rhiValue); });
        if (!result) { return result; }
        entry.value = entry.handle.shaderIndex;
        return state_->heap->writeSampler(entry.handle, sampler);
    }).transform([](auto state) {
        ResourceLease lease;
        lease.state_ = std::move(state);
        return lease;
    });
}

Result<ResourceLease> ResourceRegistry::accelerationStructure(RayTracingAccelerationStructure& structure)
{
    if (!state_ || structure.deviceIdentity() != state_->device || !structure.valid() ||
        structure.desc().type != RayTracingAccelerationStructureType::TopLevel) {
        return makeError(Error::InvalidArgument);
    }
    auto allocation = structure.retainAllocation();
    return acquire(state_, keyFor(ShaderResourceKind::AccelerationStructure, allocation),
        ShaderResourceKind::AccelerationStructure, allocation, false, [&](auto& entry) {
            // Matches Core.resolveDescriptor's explicit AS address contract in both modes.
            entry.value = structure.deviceAddress();
            return Result<>{};
        }).transform([](auto state) {
        ResourceLease lease;
        lease.state_ = std::move(state);
        return lease;
    });
}

void ResourceRegistry::collect()
{
    if (state_) { std::lock_guard lock(state_->mutex); state_->collectLocked(); }
}

ResourceRegistryStats ResourceRegistry::stats() const
{
    if (!state_) { return {}; }
    std::lock_guard lock(state_->mutex);
    return state_->stats;
}

BindlessHeap* ResourceRegistry::heap() const
{
    return state_ ? state_->heap.get() : nullptr;
}

Result<> ResourceRegistry::bind(CommandBuffer& commands) const
{
    if (!state_ || commands.deviceIdentity() != state_->device) { return makeError(Error::InvalidArgument); }
    auto result = commands.retainResource(state_);
    if (result) { result = commands.bindBindlessHeap(*state_->heap); }
    return result;
}

bool ResourceRegistry::owns(const ResourceLease& lease) const
{
    return state_ && lease.state_ && lease.state_->registry == state_;
}

Result<> ResourceRegistry::retain(CommandBuffer& commands, const ResourceLease& lease) const
{
    if (!owns(lease) || commands.deviceIdentity() != state_->device) {
        return makeError(Error::InvalidArgument);
    }
    return commands.retainResource(lease.state_);
}

ParameterWriter::ParameterWriter(Device& device, RenderFrameContext& frame, ResourceRegistry& registry)
    : ParameterWriter(device, registry, &frame)
{
}

ParameterWriter::ParameterWriter(Device& device, ResourceRegistry& registry, RenderFrameContext* frame)
    : device_(device), frame_(frame), completion_(frame ? frame->completion() : GPUCompletionPoint{}),
      registry_(registry.state_)
{
    if (!registry_ || registry_->device != device.identity() || (frame && !frame->recording())) {
        result_ = makeError(Error::InvalidArgument);
    }
}

Result<> ParameterWriter::use(const ResourceLease& lease)
{
    if (result_ && (!lease.state_ || lease.state_->registry != registry_)) { result_ = makeError(Error::InvalidArgument); }
    if (result_) { resources_.push_back(lease.state_); }
    return result_;
}

uint64_t ParameterWriter::append(Result<ResourceLease> lease)
{
    if (result_ && !lease) { result_ = makeError(lease.error()); }
    return result_ && use(*lease) ? lease->shaderValue() : UINT64_MAX;
}

ShaderBuffer ParameterWriter::buffer(Buffer* buffer)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ && buffer ? registry.storageBuffer(*buffer) : makeError(Error::InvalidArgument);
    return {static_cast<uint32_t>(append(std::move(result)))};
}

GPUResourceHandle<ResourceViewKind::SampledImage> ParameterWriter::sampledImageHandle(
    TextureView* view, TextureLayout layout)
{
    return {static_cast<uint32_t>(sampledImage(view, layout).index)};
}

GPUResourceHandle<ResourceViewKind::StorageImage> ParameterWriter::storageImageHandle(TextureView* view)
{
    return {static_cast<uint32_t>(storageImage(view).index)};
}

GPUSamplerHandle ParameterWriter::samplerHandle(const SamplerDesc& sampler)
{
    return {static_cast<uint32_t>(this->sampler(sampler).index)};
}

GPUBufferSpan ParameterWriter::bufferSpan(Buffer* buffer, BufferRange range, uint32_t stride, uint32_t alignment)
{
    if (!result_) { return {}; }
    auto slice = buffer ? buffer->slice(range) : makeError(Error::InvalidArgument);
    if (!slice) { result_ = makeError(slice.error()); return {}; }
    return bufferSpan(*slice, stride, alignment);
}

GPUBufferSpan ParameterWriter::bufferSpan(const BufferSlice& slice, uint32_t stride, uint32_t alignment)
{
    if (!result_) { return {}; }
    // Raw descriptor addressing is 32-bit. Check the exclusive end before narrowing.
    if (!slice.valid() || slice.deviceIdentity() != device_.identity() || !slice.size() ||
        !stride || !std::has_single_bit(alignment) || stride % alignment || stride % 4 ||
        slice.offset() % std::max(4u, alignment) || slice.size() % stride ||
        slice.offset() > UINT32_MAX || slice.size() > uint64_t(UINT32_MAX) + 1 - slice.offset()) {
        result_ = makeError(Error::InvalidArgument);
        return {};
    }
    ResourceRegistry registry; registry.state_ = registry_;
    const auto index = append(registry.storageBuffer(slice));
    if (!result_) { return {}; }
    return {{static_cast<uint32_t>(index)}, static_cast<uint32_t>(slice.offset()),
        static_cast<uint32_t>(slice.size() / stride)};
}

GPUBufferSpan ParameterWriter::bufferSpan(Buffer* buffer, uint32_t stride, uint32_t alignment)
{
    return bufferSpan(buffer, {}, stride, alignment);
}

ShaderSampledImage ParameterWriter::sampledImage(TextureView* view, TextureLayout layout)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ && view ? registry.sampledImage(*view, layout) : makeError(Error::InvalidArgument);
    return {static_cast<uint32_t>(append(std::move(result)))};
}

ShaderStorageImage ParameterWriter::storageImage(TextureView* view)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ && view ? registry.storageImage(*view) : makeError(Error::InvalidArgument);
    return {static_cast<uint32_t>(append(std::move(result)))};
}

ShaderSampler ParameterWriter::sampler(const SamplerDesc& sampler)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ ? registry.sampler(sampler) : makeError(result_.error());
    return {static_cast<uint32_t>(append(std::move(result)))};
}

ShaderAccelerationStructure ParameterWriter::accelerationStructure(RayTracingAccelerationStructure* structure)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ && structure ? registry.accelerationStructure(*structure) : makeError(Error::InvalidArgument);
    return {append(std::move(result))};
}

Result<> ParameterWriter::upload(const void* data, uint64_t size, uint64_t alignment,
    std::shared_ptr<void>& allocation, BufferSlice& slice)
{
    if (!result_) { return result_; }
    if ((frame_ && (!frame_->recording() || !completion_.sameSubmission(frame_->completion()))) ||
        !data || size == 0 || alignment == 0 || !std::has_single_bit(alignment) ||
        alignment > 4096 || size > UINT32_MAX) { return makeError(Error::InvalidArgument); }
    std::lock_guard lock(registry_->mutex);
    std::shared_ptr<detail::ParameterChunk> chunk;
    uint64_t offset = 0;
    auto& chunks = frame_ ? registry_->chunks : standaloneChunks_;
    for (const auto& candidate : chunks) {
        if (frame_ && candidate->completion.isComplete()) {
            candidate->completion = completion_; candidate->used = 0;
        }
        // A later batch can upload while an accepted prefix reads this chunk.
        // Start each write in a distinct flush atom (including the prior tail).
        const uint64_t writeAlignment = std::max(alignment, candidate->buffer->hostWriteAlignment());
        offset = (candidate->used + writeAlignment - 1) & ~(writeAlignment - 1);
        if ((!frame_ || candidate->completion.sameSubmission(completion_)) &&
            offset <= candidate->buffer->desc().size && size <= candidate->buffer->desc().size - offset) {
            chunk = candidate; break;
        }
    }
    if (!chunk) {
        chunk = std::make_shared<detail::ParameterChunk>();
        auto result = device_.createBuffer({.size = std::max<uint64_t>(64 * 1024, size),
            .usage = BufferUsageBits::Storage,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { chunk->buffer = std::move(rhiValue); });
        if (!result) { return result; }
        chunk->completion = completion_; offset = 0;
        if (frame_) { registry_->stats.parameterCapacity += chunk->buffer->desc().size; }
        chunks.push_back(chunk);
    }
    auto* mapped = static_cast<uint8_t*>(chunk->buffer->map());
    if (!mapped || (offset & (alignment - 1)) != 0) {
        if (mapped) { chunk->buffer->unmap(); }
        return makeError(Error::Failure);
    }
    std::memcpy(mapped + offset, data, size);
    chunk->buffer->flush({offset, size});
    chunk->buffer->unmap();
    chunk->used = offset + size;
    registry_->stats.parameterBytes += size;
    allocation = chunk;
    slice = *chunk->buffer->slice({offset, size});
    return {};
}

GPUBufferSpan ParameterWriter::sampledImages(std::span<TextureView* const> views)
{
    if (!result_) { return {}; }
    if (views.empty()) { result_ = makeError(Error::InvalidArgument); return {}; }
    std::vector<ShaderSampledImage> handles;
    handles.reserve(views.size());
    for (auto* view : views) { handles.push_back(sampledImage(view)); }
    return dataSpan(handles.data(), handles.size() * sizeof(ShaderSampledImage), sizeof(ShaderSampledImage), alignof(ShaderSampledImage));
}

GPUBufferSpan ParameterWriter::dataSpan(const void* bytes, uint64_t size, uint32_t stride, uint32_t alignment)
{
    if (!result_) { return {}; }
    std::shared_ptr<void> allocation;
    BufferSlice slice;
    result_ = upload(bytes, size, alignment, allocation, slice);
    if (!result_) { return {}; }
    arrays_.push_back(std::move(allocation));
    return bufferSpan(slice, stride, alignment);
}

Result<EncodedParameters> ParameterWriter::encodeBytes(const void* params, ParameterABI abi)
{
    EncodedParameters out;
    if (!result_) { return makeError(result_.error()); }
    if (!abi.id || !params || !abi.size || !std::has_single_bit(abi.alignment) || abi.alignment > 4096 ||
        (abi.transport != ParameterTransport::DescriptorBuffer && abi.transport != ParameterTransport::InlinePush) ||
        (abi.size & 3u) ||
        (frame_ && (!frame_->recording() || !completion_.sameSubmission(frame_->completion())))) {
        result_ = makeError(Error::InvalidArgument); return makeError(result_.error());
    }
    auto packet = std::make_shared<detail::ParameterPacket>();
    if (abi.transport == ParameterTransport::InlinePush) {
        const auto* bytes = static_cast<const uint8_t*>(params);
        packet->inlineData.assign(bytes, bytes + abi.size);
    } else {
        BufferSlice slice;
        result_ = upload(params, abi.size, abi.alignment, packet->allocation, slice);
        if (result_) { packet->root = bufferSpan(slice, 4, 4); }
    }
    if (!result_) { return makeError(result_.error()); }
    packet->registry = registry_; packet->completion = completion_; packet->abi = abi;
    packet->resources = resources_; packet->arrays = arrays_;
    out.packet_ = std::move(packet);
    return out;
}

} // namespace metallic::render

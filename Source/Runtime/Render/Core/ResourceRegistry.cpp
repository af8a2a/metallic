#include "Runtime/Render/Core/ResourceRegistry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>

namespace metallic::render {
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
    uint64_t address = 0;
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

uint64_t EncodedParameters::address() const
{
    return packet_ ? packet_->address : 0;
}

const void* EncodedParameters::deviceIdentity() const
{
    return packet_ ? packet_->registry->device : nullptr;
}

bool EncodedParameters::compatible(const CommandBuffer& commands, ParameterABI abi) const
{
    auto* frame = commands.frameContext();
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
    commands.bindBindlessHeap(*packet_->registry->heap);
    return {};
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
    if (!state_ || buffer.deviceIdentity() != state_->device ||
        (uint32_t(buffer.desc().usage) & uint32_t(BufferUsageBits::Storage)) == 0) {
        return makeError(Error::InvalidArgument);
    }
    auto allocation = buffer.retainAllocation();
    return acquire(state_, keyFor(ShaderResourceKind::Buffer, allocation), ShaderResourceKind::Buffer,
        allocation, false, [&](auto& entry) {
            auto result = state_->heap->allocateBuffer().transform([&](auto rhiValue) { entry.handle = std::move(rhiValue); });
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
    ResourceState layout,
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
        auto result = kind == ShaderResourceKind::SampledImage
            ? state_->heap->allocateSampledImage().transform([&](auto rhiValue) { entry.handle = std::move(rhiValue); }) : state_->heap->allocateStorageImage().transform([&](auto rhiValue) { entry.handle = std::move(rhiValue); });
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

Result<ResourceLease> ResourceRegistry::sampledImage(TextureView& view, ResourceState layout, bool* descriptorWritten)
{
    if (descriptorWritten) { *descriptorWritten = false; }
    if (layout != ResourceState::ShaderRead && layout != ResourceState::General) {
        return makeError(Error::InvalidArgument);
    }
    return image(view, ShaderResourceKind::SampledImage, layout, descriptorWritten);
}

Result<ResourceLease> ResourceRegistry::storageImage(TextureView& view)
{
    return image(view, ShaderResourceKind::StorageImage, ResourceState::General);
}

Result<ResourceLease> ResourceRegistry::sampler(const SamplerDesc& sampler)
{
    Key key{uint64_t(ShaderResourceKind::Sampler), uint64_t(sampler.minFilter), uint64_t(sampler.magFilter),
        uint64_t(sampler.mipFilter), uint64_t(sampler.addressU), uint64_t(sampler.addressV), uint64_t(sampler.addressW),
        std::bit_cast<uint32_t>(sampler.minLod), std::bit_cast<uint32_t>(sampler.maxLod)};
    return acquire(state_, key, ShaderResourceKind::Sampler, {}, true, [&](auto& entry) {
        auto result = state_->heap->allocateSampler().transform([&](auto rhiValue) { entry.handle = std::move(rhiValue); });
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
    if (result) { commands.bindBindlessHeap(*state_->heap); }
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
    return {append(std::move(result))};
}

ShaderDataSpan ParameterWriter::dataBuffer(const BufferSlice& slice, uint32_t stride, uint32_t alignment)
{
    if (!result_) { return {}; }
    result_ = slice.validateData(device_.identity(), stride, alignment);
    if (!result_) { return {}; }
    arrays_.push_back(slice.retainAllocation());
    return {slice.deviceAddress(), static_cast<uint32_t>(slice.size() / stride), stride};
}

ShaderDataSpan ParameterWriter::dataBuffer(Buffer* buffer, uint32_t stride, uint32_t alignment)
{
    if (!result_) { return {}; }
    auto slice = buffer ? buffer->slice() : makeError(Error::InvalidArgument);
    if (!slice) {
        result_ = makeError(slice.error());
        return {};
    }
    return dataBuffer(*slice, stride, alignment);
}

ShaderSampledImage ParameterWriter::sampledImage(TextureView* view, ResourceState layout)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ && view ? registry.sampledImage(*view, layout) : makeError(Error::InvalidArgument);
    return {append(std::move(result))};
}

ShaderStorageImage ParameterWriter::storageImage(TextureView* view)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ && view ? registry.storageImage(*view) : makeError(Error::InvalidArgument);
    return {append(std::move(result))};
}

ShaderSampler ParameterWriter::sampler(const SamplerDesc& sampler)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ ? registry.sampler(sampler) : makeError(result_.error());
    return {append(std::move(result))};
}

ShaderAccelerationStructure ParameterWriter::accelerationStructure(RayTracingAccelerationStructure* structure)
{
    ResourceRegistry registry; registry.state_ = registry_;
    auto result = result_ && structure ? registry.accelerationStructure(*structure) : makeError(Error::InvalidArgument);
    return {append(std::move(result))};
}

Result<> ParameterWriter::upload(const void* data, uint64_t size, uint64_t alignment,
    uint64_t& address, std::shared_ptr<void>& allocation)
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
            .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
            .memoryLocation = MemoryLocation::HostUpload,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute}).transform([&](auto rhiValue) { chunk->buffer = std::move(rhiValue); });
        if (!result) { return result; }
        chunk->completion = completion_; offset = 0;
        if (frame_) { registry_->stats.parameterCapacity += chunk->buffer->desc().size; }
        chunks.push_back(chunk);
    }
    auto* mapped = static_cast<uint8_t*>(chunk->buffer->map());
    const auto base = chunk->buffer->deviceAddress();
    if (!mapped || !base || ((base + offset) & (alignment - 1)) != 0) {
        if (mapped) { chunk->buffer->unmap(); }
        return makeError(Error::Failure);
    }
    std::memcpy(mapped + offset, data, size);
    chunk->buffer->flush({offset, size});
    chunk->buffer->unmap();
    chunk->used = offset + size;
    registry_->stats.parameterBytes += size;
    address = base + offset;
    allocation = chunk;
    return {};
}

uint64_t ParameterWriter::sampledImages(std::span<TextureView* const> views)
{
    if (!result_) { return 0; }
    if (views.empty()) { result_ = makeError(Error::InvalidArgument); return 0; }
    std::vector<ShaderSampledImage> handles;
    handles.reserve(views.size());
    for (auto* view : views) { handles.push_back(sampledImage(view)); }
    uint64_t address = 0;
    std::shared_ptr<void> allocation;
    result_ = upload(handles.data(), handles.size() * sizeof(ShaderSampledImage), alignof(ShaderSampledImage), address, allocation);
    if (result_) { arrays_.push_back(std::move(allocation)); }
    return address;
}

uint64_t ParameterWriter::data(const void* bytes, uint64_t size, uint64_t alignment)
{
    uint64_t address = 0;
    std::shared_ptr<void> allocation;
    result_ = upload(bytes, size, alignment, address, allocation);
    if (result_) { arrays_.push_back(std::move(allocation)); }
    return address;
}

Result<EncodedParameters> ParameterWriter::encodeBytes(const void* params, ParameterABI abi)
{
    EncodedParameters out;
    if (!result_) { return makeError(result_.error()); }
    if (!abi.id) { result_ = makeError(Error::InvalidArgument); return makeError(result_.error()); }
    auto packet = std::make_shared<detail::ParameterPacket>();
    result_ = upload(params, abi.size, abi.alignment, packet->address, packet->allocation);
    if (!result_) { return makeError(result_.error()); }
    packet->registry = registry_; packet->completion = completion_; packet->abi = abi;
    packet->resources = resources_; packet->arrays = arrays_;
    out.packet_ = std::move(packet);
    return out;
}

} // namespace metallic::render

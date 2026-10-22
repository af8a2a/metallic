#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeResourceEncoder.h"
#include "Runtime/Render/Core/ComputeResourceParameters.h"
#include "Runtime/Render/Core/ResourceMember.h"
#include "Runtime/Render/Profiling/CPUProfile.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <mutex>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

constexpr uint32_t kMaxComputeResourceBindings = 256;

const ComputeDispatchBinding* findDispatchBinding(
    const ComputeDispatchDesc& desc,
    ComputeResourceMember binding)
{
    if (desc.bindings.empty()) {
        return nullptr;
    }
    for (uint32_t index = 0; index < desc.bindings.size(); ++index) {
        if (desc.bindings[index].binding == binding) {
            return &desc.bindings[index];
        }
    }
    return nullptr;
}

bool hasDuplicateBindings(std::span<const ComputeResourceBindingDesc> bindings)
{
    for (uint32_t lhs = 0; lhs < bindings.size(); ++lhs) {
        for (uint32_t rhs = lhs + 1; rhs < bindings.size(); ++rhs) {
            if (bindings[lhs].binding == bindings[rhs].binding) {
                return true;
            }
        }
    }
    return false;
}

bool usesImageHeap(ComputeResourceBindingKind kind)
{
    return kind == ComputeResourceBindingKind::SampledImage ||
        kind == ComputeResourceBindingKind::StorageImage;
}

} // namespace

namespace detail {

struct ComputeSampledImageTable {
    // Keep the generation's transitive owners without retaining the snapshot
    // which owns this cache. Packets retain this table until GPU completion.
    std::shared_ptr<const void> owner;
    std::vector<ResourceLease> images;
    std::unique_ptr<Buffer> indices;
    ResourceLease indexLease;
    GPUBufferSpan span;
};

} // namespace detail

Result<std::shared_ptr<detail::ComputeSampledImageTable>> ComputeSampledImageSnapshot::table(
    Device& device, ResourceRegistry& registry, uint32_t count, ComputeDispatchStats* stats) const
{
    if (!count || views.size() < count || uint64_t(count) * sizeof(uint32_t) > UINT32_MAX) {
        return makeError(Error::InvalidArgument);
    }
    std::lock_guard lock(tableMutex_);
    for (const auto& cached : tables_) {
        // A lease owns the exact registry epoch, so neither a recycled Device
        // nor a reused registry address can match a table from another heap.
        if (cached->span.count == count && registry.owns(cached->indexLease)) {
            if (stats) { ++stats->sampledImageTableCacheHits; }
            return cached;
        }
    }
    auto cached = std::make_shared<detail::ComputeSampledImageTable>();
    cached->owner = owner;
    cached->images.reserve(count);
    std::vector<uint32_t> indices;
    indices.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!views[i]) { return makeError(Error::InvalidArgument); }
        bool written = false;
        auto lease = registry.sampledImage(*views[i], TextureLayout::ShaderRead, &written);
        if (!lease) { return makeError(lease.error()); }
        if (stats) {
            if (written) { ++stats->sampledImageWrites; }
            else { ++stats->sampledImageCacheHits; }
        }
        indices.push_back(lease->shaderIndex());
        cached->images.push_back(std::move(*lease));
    }
    const uint64_t bytes = uint64_t(count) * sizeof(uint32_t);
    auto buffer = device.createBuffer({.size = bytes, .usage = BufferUsageBits::Storage,
        .memoryLocation = MemoryLocation::HostUpload,
        .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute});
    if (!buffer) { return makeError(buffer.error()); }
    cached->indices = std::move(*buffer);
    auto* mapped = cached->indices->map();
    if (!mapped) { return makeError(Error::Failure); }
    std::memcpy(mapped, indices.data(), bytes);
    cached->indices->flush({0, bytes});
    cached->indices->unmap();
    auto lease = registry.storageBuffer(*cached->indices);
    if (!lease) { return makeError(lease.error()); }
    cached->indexLease = std::move(*lease);
    cached->span = {{cached->indexLease.shaderIndex()}, 0, count};
    // This immutable allocation is independent of frame upload chunks. A
    // cancelled first recording cannot recycle storage used by a later frame.
    tables_.push_back(cached);
    if (stats) {
        ++stats->sampledImageTableBuilds;
        stats->sampledImageTableUploadBytes += bytes;
    }
    return cached;
}

struct ComputeResourceEncoder::Impl {
    uint64_t abiId = 0;
    Device* device = nullptr;
    std::shared_ptr<ResourceRegistry> registry;
    std::vector<ComputeResourceBindingDesc> bindings;
    uint32_t pushConstantSize = 0;
    uint32_t resourceParameterSize = 0;
    uint32_t resourceParameterAlignment = 4;
    ParameterTransport rootTransport = ParameterTransport::DescriptorBuffer;
    std::string debugName;

    bool hasCompatibleBindings(const Impl& other) const
    {
        return device == other.device && pushConstantSize == other.pushConstantSize && bindings == other.bindings &&
            resourceParameterSize == other.resourceParameterSize && rootTransport == other.rootTransport;
    }
};

Result<> ComputeResourceEncoder::initialize(Device& device, const ResourceComputeKernelDesc& desc, std::string& log)
{
    clear();
    log.clear();
    if (desc.rootTransport != ParameterTransport::DescriptorBuffer && desc.rootTransport != ParameterTransport::InlinePush) {
        log = "ComputeResourceEncoder has an invalid root transport";
        return makeError(Error::InvalidArgument);
    }
    if (desc.bindings.size() > kMaxComputeResourceBindings || hasDuplicateBindings(desc.bindings)) {
        log = "Resource encoder requires unique resource members";
        return makeError(Error::InvalidArgument);
    }
    if ((desc.requiresRayQuery && (!device.capabilities().rayQuery || !device.capabilities().rayTracingAccelerationStructure)) ||
        !device.capabilities().bindlessDescriptorHeap) {
        log = "ComputeResourceEncoder requires unavailable device capabilities";
        return makeError(Error::Unsupported);
    }
    auto impl = std::make_shared<Impl>();
    impl->device = &device;
    impl->pushConstantSize = desc.pushConstantSize;
    impl->rootTransport = desc.rootTransport;
    impl->debugName = desc.debugName ? desc.debugName : "ComputeResourceEncoder";
    for (const auto& binding : desc.bindings) {
        if (!binding.binding.valid() || binding.binding.kind() != binding.kind || binding.descriptorCount == 0 ||
            binding.kind > ComputeResourceBindingKind::Sampler ||
            (!usesImageHeap(binding.kind) && binding.descriptorCount != 1) ||
            (binding.kind == ComputeResourceBindingKind::DataBuffer &&
                (!binding.dataStride || !std::has_single_bit(binding.dataAlignment) ||
                 binding.dataStride % binding.dataAlignment != 0 || binding.dataStride % 4 != 0))) {
            log = "ComputeResourceEncoder has an invalid resource binding";
            return makeError(Error::InvalidArgument);
        }
        impl->bindings.push_back(binding);
    }
    // CPU member identities never depend on descriptor allocation order.
    std::ranges::sort(impl->bindings, {}, &ComputeResourceBindingDesc::binding);
    {
        const auto size = desc.resourceParameterSize;
        if (!size || size > 65536 || (size & 3u)) {
            log = "ComputeResourceEncoder requires a valid resource parameter size";
            return makeError(Error::InvalidArgument);
        }
        impl->resourceParameterSize = size;
        const auto memberSize = [](ComputeResourceMember member) -> uint32_t {
            return member.format() != ResourceMemberFormat::Handle ? sizeof(GPUBufferSpan) :
                member.kind() == ComputeResourceBindingKind::AccelerationStructure ? 8u : 4u;
        };
        for (size_t index = 0; index < impl->bindings.size(); ++index) {
            const auto& binding = impl->bindings[index];
            const auto member = binding.binding;
            const uint32_t fieldSize = memberSize(member);
            const uint32_t alignment = member.kind() == ComputeResourceBindingKind::AccelerationStructure ? 8u : 4u;
            if (member.offset() % alignment || size % alignment ||
                member.offset() > size || fieldSize > size - member.offset() ||
                (member.format() == ResourceMemberFormat::Handle && binding.descriptorCount != 1)) {
                log = "Invalid resource member at offset " + std::to_string(member.offset());
                return makeError(Error::InvalidArgument);
            }
            for (size_t previousIndex = 0; previousIndex < index; ++previousIndex) {
                const auto previous = impl->bindings[previousIndex].binding;
                if (member.offset() < previous.offset() + memberSize(previous) && previous.offset() < member.offset() + fieldSize) {
                    log = "Overlapping resource members";
                    return makeError(Error::InvalidArgument);
                }
            }
            impl->resourceParameterAlignment = std::max(impl->resourceParameterAlignment, alignment);
        }
    }
    auto registry = metallic::render::ResourceRegistry::forDevice(device);
    if (!registry) { return makeError(registry.error()); }
    impl->registry = std::move(*registry);
    // Intern exact live contracts, not just their hash. The ID is CPU-only and
    // couples a kernel to its encoder even though their shader root has the same
    // two-span shape. Weak entries never extend the Device's lifetime.
    static std::mutex contractMutex;
    static std::vector<std::weak_ptr<const Impl>> contracts;
    static uint64_t nextABI = 0x4352455300000000ull;
    std::lock_guard lock(contractMutex);
    std::erase_if(contracts, [](const auto& entry) { return entry.expired(); });
    for (const auto& entry : contracts) {
        auto existing = entry.lock();
        if (existing && impl->hasCompatibleBindings(*existing)) {
            impl_ = std::move(existing);
            return {};
        }
    }
    impl->abiId = ++nextABI;
    impl_ = std::move(impl);
    contracts.push_back(impl_);
    return {};
}

ParameterABI ComputeResourceEncoder::parameterABI() const
{
    return impl_ ? parameterAbi<ComputeResourceParameters>(impl_->abiId, impl_->rootTransport) : ParameterABI{};
}

bool ComputeResourceEncoder::compatible(const ComputeResourceEncoder& other) const
{
    return impl_ && other.impl_ && impl_->hasCompatibleBindings(*other.impl_);
}

Result<> ComputeResourceEncoder::validate(const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches) const
{
    if (desc.stats) { *desc.stats = {}; }
    if (!valid() || desc.bindings.size() > UINT32_MAX ||
        (desc.indirectArguments == nullptr &&
         (desc.groupCountX == 0 || desc.groupCountY == 0 || desc.groupCountZ == 0)) ||
        (impl_->pushConstantSize > 0 &&
         (desc.pushData == nullptr || desc.pushDataSize != impl_->pushConstantSize)) ||
        (impl_->pushConstantSize == 0 && desc.pushDataSize != 0)) {
        spdlog::error("[ComputeResourceEncoder:{}] invalid dispatch description", impl_ ? impl_->debugName : "uninitialized");
        return makeError(Error::InvalidArgument);
    }

    if (desc.indirectArguments != nullptr &&
        ((static_cast<uint32_t>(desc.indirectArguments->desc().usage) &
          static_cast<uint32_t>(BufferUsageBits::Indirect)) == 0 ||
         (desc.indirectOffset & 3u) != 0 ||
         desc.indirectOffset > desc.indirectArguments->desc().size ||
         3 * sizeof(uint32_t) > desc.indirectArguments->desc().size - desc.indirectOffset)) {
        return makeError(Error::InvalidArgument);
    }
    if (!dispatches.empty() && !desc.indirectArguments) { return makeError(Error::InvalidArgument); }
    for (const auto& item : dispatches) {
        if ((item.kernel == nullptr) != (item.encoder == nullptr) ||
            (item.kernel && (!item.kernel->valid() || !compatible(*item.encoder) ||
                item.kernel->parameterABI() != parameterABI()))) {
            return makeError(Error::InvalidArgument);
        }
        if ((impl_->pushConstantSize > 0 && item.pushData == nullptr) ||
            (item.argumentOffset & 3u) != 0 || item.argumentOffset > desc.indirectArguments->desc().size ||
            3 * sizeof(uint32_t) > desc.indirectArguments->desc().size - item.argumentOffset) {
            return makeError(Error::InvalidArgument);
        }
    }
    return {};
}

Result<std::vector<EncodedParameters>> ComputeResourceEncoder::encode(
    RenderFrameContext* frame,
    const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches) const
{
    auto result = validate(desc, dispatches);
    if (!result) { return makeError(result.error()); }
    auto& registry = *impl_->registry;
    ParameterWriter writer(*impl_->device, registry, frame);
    auto upload = [&](const void* bytes, uint64_t size, uint32_t stride = 4, uint32_t alignment = 4) -> GPUBufferSpan {
        if (!result || size == 0) { return {}; }
        const auto span = writer.dataSpan(bytes, size, stride, alignment);
        result = writer.status();
        return span;
    };
    // Absent optional fields are invalid sentinel bytes, never descriptor index zero.
    std::vector<uint8_t> parameters(impl_->resourceParameterSize, 0xff);
    for (size_t input = 0; input < impl_->bindings.size(); ++input) {
        const auto& expected = impl_->bindings[input];
        const auto member = expected.binding;
        auto* destination = parameters.data() + member.offset();
        const auto* binding = findDispatchBinding(desc, expected.binding);
        if (!binding) {
            if (expected.optional) { continue; }
            return makeError(Error::InvalidArgument);
        }
        if (expected.kind == ComputeResourceBindingKind::DataBuffer) {
            BufferSlice slice = binding->data;
            if (slice.valid()) {
                if (binding->buffer || binding->range.offset != 0 || binding->range.size != UINT64_MAX) {
                    return makeError(Error::InvalidArgument);
                }
            } else {
                if (!binding->buffer) { return makeError(Error::InvalidArgument); }
                result = binding->buffer->slice({binding->range.offset, binding->range.size}).transform([&](auto rhiValue) { slice = std::move(rhiValue); });
                if (!result) { return makeError(result.error()); }
            }
            auto span = writer.bufferSpan(slice, expected.dataStride, expected.dataAlignment);
            result = writer.status();
            if (!result) { return makeError(result.error()); }
            // Named spans count words; typedBufferSpan restores the authored element type.
            const uint64_t words = uint64_t(span.count) * expected.dataStride / 4;
            if (words > UINT32_MAX) { return makeError(Error::InvalidArgument); }
            span.count = static_cast<uint32_t>(words);
            std::memcpy(destination, &span, sizeof(span));
            continue;
        }
        if (binding->data.valid()) { return makeError(Error::InvalidArgument); }
        const uint32_t count = std::max(expected.descriptorCount, 1u);
        if (binding->sampledImages && expected.kind == ComputeResourceBindingKind::SampledImage &&
            member.format() == ResourceMemberFormat::IndexSpan) {
            auto table = binding->sampledImages->table(*impl_->device, registry, count, desc.stats);
            if (!table) { return makeError(table.error()); }
            writer.retain(*table);
            std::memcpy(destination, &(*table)->span, sizeof(GPUBufferSpan));
            continue;
        }
        if (binding->sampledImages) {
            writer.retain(std::const_pointer_cast<ComputeSampledImageSnapshot>(binding->sampledImages));
        }
        std::vector<uint64_t> handles;
        handles.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            Result<ResourceLease> lease;
            switch (expected.kind) {
            case ComputeResourceBindingKind::DataBuffer:
                return makeError(Error::InvalidArgument);
            case ComputeResourceBindingKind::StorageBuffer:
                if (!binding->buffer || binding->range.offset != 0 ||
                    (binding->range.size != UINT64_MAX && binding->range.size != binding->buffer->desc().size)) {
                    return makeError(Error::InvalidArgument);
                }
                lease = registry.storageBuffer(*binding->buffer);
                break;
            case ComputeResourceBindingKind::AccelerationStructure:
                if (!binding->accelerationStructure) { return makeError(Error::InvalidArgument); }
                lease = registry.accelerationStructure(*binding->accelerationStructure);
                break;
            case ComputeResourceBindingKind::Sampler:
                if (!binding->sampler) { return makeError(Error::InvalidArgument); }
                lease = registry.sampler(*binding->sampler);
                break;
            case ComputeResourceBindingKind::SampledImage:
            case ComputeResourceBindingKind::StorageImage: {
                TextureView* view = nullptr;
                if (binding->sampledImages && expected.kind == ComputeResourceBindingKind::SampledImage) {
                    if (binding->sampledImages->views.size() < count) { return makeError(Error::InvalidArgument); }
                    view = binding->sampledImages->views[i].get();
                } else if (!binding->textureViews.empty() && binding->textureViews.size() >= count) {
                    view = binding->textureViews[i];
                } else if (count == 1) { view = binding->textureView; }
                if (!view) { return makeError(Error::InvalidArgument); }
                bool written = false;
                lease = expected.kind == ComputeResourceBindingKind::SampledImage
                    ? registry.sampledImage(*view, TextureLayout::ShaderRead, &written) : registry.storageImage(*view);
                if (lease && desc.stats && expected.kind == ComputeResourceBindingKind::SampledImage) {
                    if (written) { ++desc.stats->sampledImageWrites; }
                    else { ++desc.stats->sampledImageCacheHits; }
                }
                break;
            }
            }
            if (!lease) { return makeError(lease.error()); }
            result = writer.use(*lease);
            if (!result) { return makeError(result.error()); }
            handles.push_back(lease->shaderValue());
        }
        if (member.format() == ResourceMemberFormat::IndexSpan) {
            std::vector<uint32_t> indices;
            indices.reserve(handles.size());
            for (auto handle : handles) { indices.push_back(static_cast<uint32_t>(handle)); }
            const auto span = upload(indices.data(), indices.size() * sizeof(uint32_t));
            std::memcpy(destination, &span, sizeof(span));
        } else {
            const uint32_t size = expected.kind == ComputeResourceBindingKind::AccelerationStructure ? 8u : 4u;
            std::memcpy(destination, handles.data(), size);
        }
        if (!result) { return makeError(result.error()); }
    }
    ComputeResourceParameters push;
    push.resources = upload(parameters.data(), parameters.size(), impl_->resourceParameterSize, impl_->resourceParameterAlignment);
    push.resources.count = impl_->resourceParameterSize / 4;
    const uint64_t stride = (uint64_t(impl_->pushConstantSize) + 15) & ~uint64_t(15);
    const size_t count = std::max<size_t>(1, dispatches.size());
    if (stride && (count > SIZE_MAX / stride || count > UINT32_MAX / stride)) { return makeError(Error::InvalidArgument); }
    std::vector<uint8_t> constants(stride * count);
    if (impl_->pushConstantSize) {
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(constants.data() + stride * i,
                dispatches.empty() ? desc.pushData : dispatches[i].pushData, impl_->pushConstantSize);
        }
        push.constants = upload(constants.data(), constants.size());
    }
    if (!result) { return makeError(result.error()); }
    std::vector<EncodedParameters> items;
    items.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto parameters = push;
        if (parameters.constants.count) {
            parameters.constants.byteOffset += static_cast<uint32_t>(stride * i);
            parameters.constants.count = impl_->pushConstantSize / 4;
        }
        auto encoded = writer.encode(parameters, impl_->abiId, impl_->rootTransport);
        if (!encoded) { return makeError(encoded.error()); }
        items.push_back(std::move(*encoded));
    }
    return items;
}

Result<> initializeResourceKernel(Device& device, const ResourceComputeKernelDesc& desc,
    ComputeKernel& kernel, ComputeResourceEncoder& encoder, std::string& log)
{
    ComputeResourceEncoder nextEncoder;
    auto result = nextEncoder.initialize(device, desc, log);
    if (!result) { return result; }
    ComputeKernel nextKernel;
    result = nextKernel.initialize(device, {.spirv = desc.spirv,
        .parameters = nextEncoder.parameterABI(),
        .debugName = desc.debugName, .pipelineCache = desc.pipelineCache,
        .specializationConstants = desc.specializationConstants}, log);
    if (!result) { return result; }
    kernel = std::move(nextKernel);
    encoder = std::move(nextEncoder);
    return {};
}

Result<PreparedComputeDispatch> prepareResourceDispatch(
    const ComputeKernel& kernel, const ComputeResourceEncoder& encoder,
    RenderFrameContext* frame, const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches)
{
    if (desc.commandBuffer || !kernel.valid() || kernel.parameterABI() != encoder.parameterABI() ||
        (frame && !frame->recording())) {
        return makeError(Error::InvalidArgument);
    }
    auto first = desc;
    if (!dispatches.empty()) {
        if (!desc.indirectArguments) { return makeError(Error::InvalidArgument); }
        first.pushData = dispatches.front().pushData;
        first.indirectOffset = dispatches.front().argumentOffset;
    }
    auto encoded = encoder.encode(frame, first, dispatches);
    if (!encoded) { return makeError(encoded.error()); }
    if (!desc.indirectArguments) {
        return kernel.prepareDispatch(encoded->front(), desc.groupCountX, desc.groupCountY, desc.groupCountZ);
    }
    std::vector<ComputeIndirectParameters> items;
    items.reserve(encoded->size());
    for (size_t i = 0; i < encoded->size(); ++i) {
        auto arguments = desc.indirectArguments->slice({
            dispatches.empty() ? desc.indirectOffset : dispatches[i].argumentOffset, 12});
        if (!arguments) { return makeError(arguments.error()); }
        items.push_back({std::move((*encoded)[i]), std::move(*arguments),
            dispatches.empty() ? nullptr : dispatches[i].kernel});
    }
    return kernel.prepareIndirectBatch(items);
}

Result<> dispatchResources(const ComputeKernel& kernel, const ComputeResourceEncoder& encoder,
    const ComputeDispatchDesc& desc, std::span<const ComputeIndirectDispatch> dispatches,
    const BarrierDesc& betweenDispatches)
{
    if (!desc.commandBuffer || !desc.commandBuffer->recording()) { return makeError(Error::InvalidArgument); }
    CPUProfileScope profile(desc.profiler, "Encode compute parameters");
    auto inputs = desc;
    inputs.commandBuffer = nullptr;
    auto prepared = prepareResourceDispatch(kernel, encoder,
        RenderFrameContext::from(*desc.commandBuffer), inputs, dispatches);
    if (!prepared) { return makeError(prepared.error()); }
    profile.next("Record dispatch commands");
    return prepared->record(*desc.commandBuffer, betweenDispatches);
}

Result<> initializeResourceKernel(Device& device, const ResourceRayTracingKernelDesc& desc,
    RayTracingKernel& kernel, ComputeResourceEncoder& encoder, std::string& log)
{
    ComputeResourceEncoder nextEncoder;
    auto result = nextEncoder.initialize(device, {
        .pushConstantSize = desc.pushConstantSize,
        .bindings = desc.bindings,
        .debugName = desc.debugName,
        .requiresRayQuery = false,
        .resourceParameterSize = desc.resourceParameterSize,
    }, log);
    if (!result) { return result; }
    RayTracingKernel nextKernel;
    result = nextKernel.initialize(device, {
        .rayGenerationSpirv = desc.rayGenerationSpirv,
        .missSpirv = desc.missSpirv,
        .closestHitSpirv = desc.closestHitSpirv,
        .anyHitSpirv = desc.anyHitSpirv,
        .parameters = nextEncoder.parameterABI(),
        .debugName = desc.debugName,
        .pipelineCache = desc.pipelineCache,
        .specializationConstants = desc.specializationConstants,
        .enableInvocationReorder = desc.enableInvocationReorder,
    }, log);
    if (!result) { return result; }
    kernel = std::move(nextKernel);
    encoder = std::move(nextEncoder);
    return {};
}

Result<PreparedRayTracingDispatch> prepareResourceDispatch(
    const RayTracingKernel& kernel, const ComputeResourceEncoder& encoder,
    RenderFrameContext* frame, const RayTracingDispatchDesc& desc)
{
    if (desc.stats) { *desc.stats = {}; }
    if (desc.commandBuffer || !kernel.valid() || kernel.parameterABI() != encoder.parameterABI() ||
        !desc.width || !desc.height || !desc.depth || (frame && !frame->recording())) {
        return makeError(Error::InvalidArgument);
    }
    // ComputeResourceEncoder encodes resource data only. Its group counts stay
    // at the neutral default; the launch dimensions belong to the RT packet.
    const ComputeDispatchDesc inputs{
        .bindings = desc.bindings,
        .pushData = desc.pushData,
        .pushDataSize = desc.pushDataSize,
        .profiler = desc.profiler,
        .stats = desc.stats,
    };
    auto encoded = encoder.encode(frame, inputs);
    if (!encoded) { return makeError(encoded.error()); }
    return kernel.prepareDispatch(encoded->front(), desc.width, desc.height, desc.depth);
}

Result<> dispatchResources(const RayTracingKernel& kernel, const ComputeResourceEncoder& encoder,
    const RayTracingDispatchDesc& desc)
{
    if (!desc.commandBuffer || !desc.commandBuffer->recording()) { return makeError(Error::InvalidArgument); }
    CPUProfileScope profile(desc.profiler, "Encode ray tracing parameters");
    auto inputs = desc;
    inputs.commandBuffer = nullptr;
    auto prepared = prepareResourceDispatch(kernel, encoder,
        RenderFrameContext::from(*desc.commandBuffer), inputs);
    if (!prepared) { return makeError(prepared.error()); }
    profile.next("Record ray tracing commands");
    return prepared->record(*desc.commandBuffer);
}

} // namespace metallic::render

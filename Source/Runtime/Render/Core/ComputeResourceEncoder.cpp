#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeResourceEncoder.h"
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

// ParameterRoot payload; matches Core.ComputeResourceParameters in Slang.
struct ComputeResourceParameters {
    GPUBufferSpan resources;
    GPUBufferSpan constants;
};
static_assert(sizeof(ComputeResourceParameters) == 24);

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

struct ComputeResourceEncoder::Impl {
    uint64_t abiId = 0;
    Device* device = nullptr;
    std::shared_ptr<ResourceRegistry> registry;
    std::vector<ComputeResourceBindingDesc> bindings;
    uint32_t pushConstantSize = 0;
    uint32_t resourceParameterSize = 0;
    uint32_t resourceParameterAlignment = 4;
    std::string debugName;

    bool hasCompatibleBindings(const Impl& other) const
    {
        return device == other.device && pushConstantSize == other.pushConstantSize && bindings == other.bindings &&
            resourceParameterSize == other.resourceParameterSize;
    }
};

Result<> ComputeResourceEncoder::initialize(Device& device, const ResourceComputeKernelDesc& desc, std::string& log)
{
    clear();
    log.clear();
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
    return impl_ ? parameterAbi<ComputeResourceParameters>(impl_->abiId) : ParameterABI{};
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
        if (binding->sampledImages) {
            writer.retain(std::const_pointer_cast<ComputeSampledImageSnapshot>(binding->sampledImages));
        }
        const uint32_t count = std::max(expected.descriptorCount, 1u);
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
        auto encoded = writer.encode(parameters, impl_->abiId);
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
        .debugName = desc.debugName, .pipelineCache = desc.pipelineCache}, log);
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

} // namespace metallic::render

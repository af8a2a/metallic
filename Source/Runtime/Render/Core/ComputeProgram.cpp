#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Profiling/CPUProfile.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

constexpr uint32_t kMaxComputeResourceSlots = 256;
constexpr uint64_t kComputeResourceABI = 0x434f4d5055544503ull;

// ParameterRoot payload; matches Core.ComputeResourceParameters in Slang.
struct ComputeResourceParameters {
    uint64_t resources = 0;
    uint64_t constants = 0;
};
static_assert(sizeof(ComputeResourceParameters) == 16);

const ComputeDispatchBinding* findDispatchBinding(
    const ComputeDispatchDesc& desc,
    uint32_t binding)
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

bool hasDuplicateBindings(std::span<const ComputeProgramBindingDesc> bindings)
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

struct ComputeProgram::Impl {
    Device* device = nullptr;
    std::shared_ptr<ResourceRegistry> registry;
    ComputeKernel kernel;
    std::vector<ComputeProgramBindingDesc> bindings;
    uint32_t pushConstantSize = 0;
    uint32_t resourceSlotCount = 0;
    std::string debugName;

    bool hasCompatibleBindings(const Impl& other) const
    {
        return device == other.device && pushConstantSize == other.pushConstantSize && bindings == other.bindings;
    }
};

ComputeProgram::ComputeProgram() = default;
ComputeProgram::~ComputeProgram() = default;
ComputeProgram::ComputeProgram(ComputeProgram&&) noexcept = default;
ComputeProgram& ComputeProgram::operator=(ComputeProgram&&) noexcept = default;

Result<> ComputeProgram::initialize(Device& device, const ComputeProgramDesc& desc, std::string& log)
{
    clear();
    log.clear();
    if (desc.spirv.size() < 5 || desc.spirv[0] != 0x07230203u ||
        desc.bindings.size() > kMaxComputeResourceSlots || hasDuplicateBindings(desc.bindings)) {
        log = "ComputeProgram requires valid SPIR-V and unique resource slots";
        return makeError(Error::InvalidArgument);
    }
    if ((desc.requiresRayQuery && (!device.capabilities().rayQuery || !device.capabilities().rayTracingAccelerationStructure)) ||
        !device.capabilities().bindlessDescriptorHeap) {
        log = "ComputeProgram requires unavailable device capabilities";
        return makeError(Error::Unsupported);
    }
    auto impl = std::make_shared<Impl>();
    impl->device = &device;
    impl->pushConstantSize = desc.pushConstantSize;
    impl->debugName = desc.debugName ? desc.debugName : "ComputeProgram";
    for (const auto& binding : desc.bindings) {
        if (binding.binding >= kMaxComputeResourceSlots || binding.descriptorCount == 0 ||
            binding.kind > ComputeResourceBindingKind::Sampler ||
            (!usesImageHeap(binding.kind) && binding.descriptorCount != 1) ||
            (binding.kind == ComputeResourceBindingKind::DataBuffer &&
                (!binding.dataStride || !std::has_single_bit(binding.dataAlignment) ||
                 binding.dataStride % binding.dataAlignment != 0))) {
            log = "ComputeProgram has an invalid resource binding";
            return makeError(Error::InvalidArgument);
        }
        impl->resourceSlotCount = std::max(impl->resourceSlotCount, binding.binding + 1);
        impl->bindings.push_back(binding);
    }
    // Resource slots are independent of declaration and descriptor allocation order.
    std::ranges::sort(impl->bindings, {}, &ComputeProgramBindingDesc::binding);
    auto registry = device.resourceRegistry();
    if (!registry) { return makeError(registry.error()); }
    impl->registry = std::move(*registry);
    auto result = impl->kernel.initialize(device, {
        .spirv = desc.spirv,
        .parameters = parameterAbi<ComputeResourceParameters>(kComputeResourceABI),
        .debugName = desc.debugName,
        .pipelineCache = desc.pipelineCache,
    }, log);
    if (!result) { return result; }
    impl_ = std::move(impl);
    return {};
}

void ComputeProgram::clear()
{
    impl_.reset();
}

bool ComputeProgram::valid() const
{
    return impl_ && impl_->kernel.valid();
}

Result<> ComputeProgram::dispatch(const ComputeDispatchDesc& desc)
{
    return dispatchImpl(desc, {}, {});
}

Result<> ComputeProgram::dispatchIndirectBatch(const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches)
{
    if (!desc.indirectArguments || dispatches.empty()) { return makeError(Error::InvalidArgument); }
    auto first = desc;
    first.pushData = dispatches.front().pushData;
    first.indirectOffset = dispatches.front().argumentOffset;
    return dispatchImpl(first, dispatches, betweenDispatches);
}

Result<> ComputeProgram::validateDispatch(const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches) const
{
    if (desc.stats) { *desc.stats = {}; }
    if (!valid() || desc.bindings.size() > UINT32_MAX ||
        (desc.indirectArguments == nullptr &&
         (desc.groupCountX == 0 || desc.groupCountY == 0 || desc.groupCountZ == 0)) ||
        (impl_->pushConstantSize > 0 &&
         (desc.pushData == nullptr || desc.pushDataSize != impl_->pushConstantSize)) ||
        (impl_->pushConstantSize == 0 && desc.pushDataSize != 0)) {
        spdlog::error("[ComputeProgram:{}] invalid dispatch description", impl_ ? impl_->debugName : "uninitialized");
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
        if (item.program != nullptr &&
            (!item.program->valid() || !impl_->hasCompatibleBindings(*item.program->impl_))) {
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

Result<> ComputeProgram::dispatchImpl(const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches)
{
    if (!valid() || !desc.commandBuffer || !desc.commandBuffer->recording() ||
        desc.commandBuffer->deviceIdentity() != impl_->device->identity()) {
        return makeError(Error::InvalidArgument);
    }
    CPUProfileScope profile(desc.profiler, "Encode compute parameters");
    auto prepared = prepare(desc.commandBuffer->frameContext(), desc, dispatches);
    if (!prepared) { return makeError(prepared.error()); }
    profile.next("Record dispatch commands");
    return prepared->record(*desc.commandBuffer, betweenDispatches);
}

Result<PreparedComputeDispatch> ComputeProgram::prepareDispatch(
    RenderFrameContext& frame,
    const ComputeDispatchDesc& desc) const
{
    if (desc.commandBuffer || !frame.recording()) { return makeError(Error::InvalidArgument); }
    return prepare(&frame, desc, {});
}

Result<PreparedComputeDispatch> ComputeProgram::prepareIndirectBatch(
    RenderFrameContext& frame,
    const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches) const
{
    if (desc.commandBuffer || !frame.recording() || !desc.indirectArguments || dispatches.empty()) {
        return makeError(Error::InvalidArgument);
    }
    auto first = desc;
    first.pushData = dispatches.front().pushData;
    first.indirectOffset = dispatches.front().argumentOffset;
    return prepare(&frame, first, dispatches);
}

Result<PreparedComputeDispatch> ComputeProgram::prepare(
    RenderFrameContext* frame,
    const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches) const
{
    auto result = validateDispatch(desc, dispatches);
    if (!result) { return makeError(result.error()); }
    auto& registry = *impl_->registry;
    ParameterWriter writer(*impl_->device, registry, frame);
    auto upload = [&](const void* bytes, uint64_t size) -> uint64_t {
        if (!result || size == 0) { return 0; }
        const auto address = writer.data(bytes, size);
        result = writer.status();
        return address;
    };
    // Matches Core.ComputeResourceSlot: direct access stays scalar, arrays carry
    // explicit handles so unrelated consumers never need contiguous descriptors.
    // Ordinary data uses the same payload words for element count and stride.
    struct ResourceSlot { uint64_t handle = UINT64_MAX; uint64_t payload = 0; };
    static_assert(sizeof(ResourceSlot) == 16);
    std::vector<ResourceSlot> slots(impl_->resourceSlotCount);
    for (const auto& expected : impl_->bindings) {
        const auto* binding = findDispatchBinding(desc, expected.binding);
        if (!binding) { return makeError(Error::InvalidArgument); }
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
            result = slice.validateData(impl_->device->identity(), expected.dataStride, expected.dataAlignment);
            if (!result) { return makeError(result.error()); }
            writer.retain(slice.retainAllocation());
            auto& slot = slots[expected.binding];
            slot.handle = slice.deviceAddress();
            slot.payload = (uint64_t(expected.dataStride) << 32) | uint32_t(slice.size() / expected.dataStride);
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
                    ? registry.sampledImage(*view, ResourceState::ShaderRead, &written) : registry.storageImage(*view);
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
        auto& slot = slots[expected.binding];
        slot.handle = handles.front();
        if (usesImageHeap(expected.kind)) { slot.payload = upload(handles.data(), handles.size() * sizeof(uint64_t)); }
        if (!result) { return makeError(result.error()); }
    }
    ComputeResourceParameters push;
    push.resources = upload(slots.data(), slots.size() * sizeof(ResourceSlot));
    const uint64_t stride = (uint64_t(impl_->pushConstantSize) + 15) & ~uint64_t(15);
    const size_t count = std::max<size_t>(1, dispatches.size());
    if (stride && count > SIZE_MAX / stride) { return makeError(Error::InvalidArgument); }
    std::vector<uint8_t> constants(stride * count);
    if (impl_->pushConstantSize) {
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(constants.data() + stride * i,
                dispatches.empty() ? desc.pushData : dispatches[i].pushData, impl_->pushConstantSize);
        }
        push.constants = upload(constants.data(), constants.size());
    }
    if (!result) { return makeError(result.error()); }
    std::vector<ComputeIndirectParameters> items;
    items.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const ComputeResourceParameters parameters{push.resources, push.constants ? push.constants + stride * i : 0};
        auto encoded = writer.encode(parameters, kComputeResourceABI);
        if (!encoded) { return makeError(encoded.error()); }
        if (!desc.indirectArguments) {
            return impl_->kernel.prepareDispatch(*encoded, desc.groupCountX, desc.groupCountY, desc.groupCountZ);
        }
        auto arguments = desc.indirectArguments->slice({
            dispatches.empty() ? desc.indirectOffset : dispatches[i].argumentOffset, 3 * sizeof(uint32_t)});
        if (!arguments) { return makeError(arguments.error()); }
        const auto& program = !dispatches.empty() && dispatches[i].program ? dispatches[i].program->impl_ : impl_;
        items.push_back({std::move(*encoded), std::move(*arguments), &program->kernel});
    }
    return impl_->kernel.prepareIndirectBatch(items);
}

} // namespace metallic::render

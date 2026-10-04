#include "Runtime/Render/Core/RenderFrameContext.h"
#include "Runtime/Render/Core/ResourceRegistry.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include "Runtime/Render/Profiling/CPUProfile.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <spdlog/spdlog.h>

namespace metallic::render {
namespace {

constexpr uint32_t kMaxComputeResourceBindings = 256;
constexpr uint64_t kComputeResourceABI = 0x434f4d5055544505ull;

// ParameterRoot payload; matches Core.ComputeResourceParameters in Slang.
struct ComputeResourceParameters {
    GPUBufferSpan resources;
    GPUBufferSpan constants;
};
static_assert(sizeof(ComputeResourceParameters) == 24);

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
    uint32_t resourceParameterSize = 0;
    uint32_t resourceParameterAlignment = 4;
    std::vector<ComputeResourceField> resourceFields;
    std::string debugName;

    bool hasCompatibleBindings(const Impl& other) const
    {
        return device == other.device && pushConstantSize == other.pushConstantSize && bindings == other.bindings &&
            resourceParameterSize == other.resourceParameterSize && resourceFields == other.resourceFields;
    }
};

ComputeProgram::ComputeProgram() = default;
ComputeProgram::~ComputeProgram() = default;
ComputeProgram::ComputeProgram(ComputeProgram&&) noexcept = default;
ComputeProgram& ComputeProgram::operator=(ComputeProgram&&) noexcept = default;

ComputeProgram ComputeProgram::share() const
{
    ComputeProgram shared;
    shared.impl_ = impl_;
    return shared;
}

Result<> ComputeProgram::initialize(Device& device, const ComputeProgramDesc& desc, std::string& log)
{
    clear();
    log.clear();
    if (desc.spirv.size() < 5 || desc.spirv[0] != 0x07230203u ||
        desc.bindings.size() > kMaxComputeResourceBindings || hasDuplicateBindings(desc.bindings)) {
        log = "ComputeProgram requires valid SPIR-V and unique resource input IDs";
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
        if (binding.descriptorCount == 0 ||
            binding.kind > ComputeResourceBindingKind::Sampler ||
            (!usesImageHeap(binding.kind) && binding.descriptorCount != 1) ||
            (binding.kind == ComputeResourceBindingKind::DataBuffer &&
                (!binding.dataStride || !std::has_single_bit(binding.dataAlignment) ||
                 binding.dataStride % binding.dataAlignment != 0 || binding.dataStride % 4 != 0))) {
            log = "ComputeProgram has an invalid resource binding";
            return makeError(Error::InvalidArgument);
        }
        impl->bindings.push_back(binding);
    }
    // CPU input IDs are independent of field offsets and descriptor allocation order.
    std::ranges::sort(impl->bindings, {}, &ComputeProgramBindingDesc::binding);
    {
        const auto& layout = desc.resourceParameters;
        if (!layout.size || layout.size > 65536 || (layout.size & 3u) || layout.fields.empty()) {
            log = "ComputeProgram requires an explicit named resource layout";
            return makeError(Error::InvalidArgument);
        }
        impl->resourceParameterSize = layout.size;
        for (const auto& binding : impl->bindings) {
            const ComputeResourceField* selected = nullptr;
            for (const auto& field : layout.fields) {
                if (field.binding != binding.binding || field.kind != binding.kind) { continue; }
                if (selected) { log = "Duplicate named resource input " + std::to_string(binding.binding); return makeError(Error::InvalidArgument); }
                selected = &field;
            }
            if (!selected) { log = "Missing named resource field for input " + std::to_string(binding.binding); return makeError(Error::InvalidArgument); }
            const auto& field = *selected;
            const bool span = field.format != ComputeResourceFieldFormat::Handle;
            const uint32_t size = span ? sizeof(GPUBufferSpan) :
                field.kind == ComputeResourceBindingKind::AccelerationStructure ? 8u : 4u;
            const uint32_t alignment = !span && size == 8 ? 8u : 4u;
            if (field.format > ComputeResourceFieldFormat::DataSpan || field.offset % alignment || layout.size % alignment ||
                field.offset > layout.size || size > layout.size - field.offset ||
                (field.format == ComputeResourceFieldFormat::IndexSpan && !usesImageHeap(field.kind)) ||
                (field.format == ComputeResourceFieldFormat::DataSpan) != (field.kind == ComputeResourceBindingKind::DataBuffer) ||
                (field.format == ComputeResourceFieldFormat::Handle && binding.descriptorCount != 1)) {
                log = "Invalid named resource field for input " + std::to_string(binding.binding);
                return makeError(Error::InvalidArgument);
            }
            for (const auto& previous : impl->resourceFields) {
                const uint32_t previousSize = previous.format != ComputeResourceFieldFormat::Handle ? sizeof(GPUBufferSpan) :
                    previous.kind == ComputeResourceBindingKind::AccelerationStructure ? 8u : 4u;
                if (field.offset < previous.offset + previousSize && previous.offset < field.offset + size) {
                    log = "Overlapping named resource inputs " + std::to_string(previous.binding) + " and " + std::to_string(binding.binding);
                    return makeError(Error::InvalidArgument);
                }
            }
            impl->resourceFields.push_back(field);
            impl->resourceParameterAlignment = std::max(impl->resourceParameterAlignment, alignment);
        }
    }
    auto registry = metallic::render::ResourceRegistry::forDevice(device);
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
    auto prepared = prepare(metallic::render::RenderFrameContext::from(*desc.commandBuffer), desc, dispatches);
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
        const auto& field = impl_->resourceFields[input];
        auto* destination = parameters.data() + field.offset;
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
        if (field.format == ComputeResourceFieldFormat::IndexSpan) {
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
    std::vector<ComputeIndirectParameters> items;
    items.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto parameters = push;
        if (parameters.constants.count) {
            parameters.constants.byteOffset += static_cast<uint32_t>(stride * i);
            parameters.constants.count = impl_->pushConstantSize / 4;
        }
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

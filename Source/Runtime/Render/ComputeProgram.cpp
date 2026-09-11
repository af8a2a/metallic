#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderFrameContext.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace metallic::render {
namespace {

constexpr uint32_t kMaxComputeResourceSlots = 256;

// The RHI prepends its two-word heap header. These addresses match
// ComputeResourcePush in Shaders/Libraries/Resources/ComputeResources.slang.
struct ComputeResourcePush {
    uint64_t resources = 0;
    uint64_t constants = 0;
};
static_assert(sizeof(ComputeResourcePush) == 16);

std::string resultMessage(const char* action, Result result)
{
    return std::string(action) + " returned " + resultToString(result);
}

const ComputeDispatchBinding* findDispatchBinding(
    const ComputeDispatchDesc& desc,
    uint32_t binding)
{
    if (desc.bindings == nullptr) {
        return nullptr;
    }
    for (uint32_t index = 0; index < desc.bindingCount; ++index) {
        if (desc.bindings[index].binding == binding) {
            return &desc.bindings[index];
        }
    }
    return nullptr;
}

bool hasDuplicateBindings(const ComputeProgramBindingDesc* bindings, uint32_t bindingCount)
{
    if (bindings == nullptr) {
        return bindingCount != 0;
    }
    for (uint32_t lhs = 0; lhs < bindingCount; ++lhs) {
        for (uint32_t rhs = lhs + 1; rhs < bindingCount; ++rhs) {
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

bool usesSamplerHeap(ComputeResourceBindingKind kind)
{
    // Samplers occupy their own bindless descriptor heap range and binding union member.
    return kind == ComputeResourceBindingKind::Sampler;
}

ShaderBindingType shaderBindingType(ComputeResourceBindingKind kind)
{
    switch (kind) {
    case ComputeResourceBindingKind::Sampler:
        return ShaderBindingType::Sampler;
    case ComputeResourceBindingKind::AccelerationStructure:
        return ShaderBindingType::AccelerationStructure;
    case ComputeResourceBindingKind::PartitionedAccelerationStructure:
        return ShaderBindingType::PartitionedAccelerationStructure;
    case ComputeResourceBindingKind::StorageImage:
        return ShaderBindingType::StorageImage;
    case ComputeResourceBindingKind::StorageBuffer:
        return ShaderBindingType::StorageBuffer;
    case ComputeResourceBindingKind::SampledImage:
        return ShaderBindingType::SampledImage;
    }
    return ShaderBindingType::StorageBuffer;
}

} // namespace

struct ComputeDescriptorTables {
    struct BindingState {
        ComputeProgramBindingDesc desc;
        uint32_t heapIndexOffset = 0;
        std::vector<BindlessHandle> handles;
    };

    std::unique_ptr<BindlessHeap> heap;
    std::vector<BindingState> bindings;
    std::vector<uint32_t> samplerBaseShaderIndices;
    std::vector<uint32_t> imageBaseShaderIndices;
    std::vector<uint32_t> bufferBaseShaderIndices;
    GpuCompletionPoint completion;
    std::vector<bool> usedTables;
    // One immutable packet per table and submission. Indirect batches store
    // all of their constants in the same packet, with a distinct address each.
    std::vector<std::unique_ptr<Buffer>> resourcePackets;
};

struct ComputeProgram::Impl : ComputeDescriptorTables {
    Device* device = nullptr;
    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    uint32_t pushConstantSize = 0;
    uint32_t resourceTableCount = 0;
    uint32_t bindlessPushDataSize = 0;
    uint32_t samplerBasePushDataOffset = UINT32_MAX;
    uint32_t imageBasePushDataOffset = UINT32_MAX;
    uint32_t bufferBasePushDataOffset = UINT32_MAX;
    bool usesResourceTable = true;
    uint32_t resourceSlotCount = 0;
    std::string debugName = "ComputeProgram";
    std::vector<std::shared_ptr<ComputeDescriptorTables>> frameTables;

    bool hasCompatibleBindings(const Impl& other) const
    {
        if (device != other.device || usesResourceTable != other.usesResourceTable ||
            resourceSlotCount != other.resourceSlotCount || pushConstantSize != other.pushConstantSize ||
            resourceTableCount != other.resourceTableCount || bindlessPushDataSize != other.bindlessPushDataSize ||
            samplerBasePushDataOffset != other.samplerBasePushDataOffset ||
            imageBasePushDataOffset != other.imageBasePushDataOffset || bufferBasePushDataOffset != other.bufferBasePushDataOffset ||
            samplerBaseShaderIndices != other.samplerBaseShaderIndices ||
            imageBaseShaderIndices != other.imageBaseShaderIndices || bufferBaseShaderIndices != other.bufferBaseShaderIndices ||
            bindings.size() != other.bindings.size()) { return false; }
        // Constant heap indices are baked into SPIR-V; matching descriptor types
        // alone is insufficient. Require identical allocation order and indices.
        for (size_t i = 0; i < bindings.size(); ++i) {
            const auto& a = bindings[i];
            const auto& b = other.bindings[i];
            if (a.desc.binding != b.desc.binding || a.desc.kind != b.desc.kind ||
                a.desc.descriptorCount != b.desc.descriptorCount || a.heapIndexOffset != b.heapIndexOffset ||
                a.handles.size() != b.handles.size()) { return false; }
            for (size_t j = 0; j < a.handles.size(); ++j) {
                if (a.handles[j].shaderIndex != b.handles[j].shaderIndex) { return false; }
            }
        }
        return true;
    }
    Result acquireTables(RenderFrameContext& frame, uint32_t tableIndex,
        std::shared_ptr<ComputeDescriptorTables>& outTables)
    {
        for (const auto& tables : frameTables) {
            if (tables->completion.isComplete()) {
                tables->completion = frame.completion();
                std::fill(tables->usedTables.begin(), tables->usedTables.end(), false);
            }
            if (tables->completion.sameSubmission(frame.completion()) && !tables->usedTables[tableIndex]) {
                tables->usedTables[tableIndex] = true;
                outTables = tables;
                return {};
            }
        }

        auto tables = std::make_shared<ComputeDescriptorTables>();
        Result result = device->createBindlessHeap(heap->desc(), tables->heap);
        if (!result) {
            return result;
        }
        tables->bindings = bindings;
        tables->samplerBaseShaderIndices = samplerBaseShaderIndices;
        tables->imageBaseShaderIndices = imageBaseShaderIndices;
        tables->bufferBaseShaderIndices = bufferBaseShaderIndices;
        // Allocate in exactly the original order so shader mappings remain valid
        // for both constant-base and pushed-base pipelines.
        for (auto& binding : tables->bindings) {
            binding.handles.clear();
        }
        for (uint32_t set = 0; set < resourceTableCount; ++set) {
            for (size_t bindingIndex = 0; bindingIndex < bindings.size(); ++bindingIndex) {
                auto& binding = tables->bindings[bindingIndex];
                const uint32_t count = std::max(binding.desc.descriptorCount, 1u);
                for (uint32_t index = 0; index < count; ++index) {
                    BindlessHandle handle;
                    switch (binding.desc.kind) {
                    case ComputeResourceBindingKind::Sampler:
                        result = tables->heap->allocateSampler(handle); break;
                    case ComputeResourceBindingKind::AccelerationStructure:
                        result = tables->heap->allocateAccelerationStructure(handle); break;
                    case ComputeResourceBindingKind::PartitionedAccelerationStructure:
                        result = tables->heap->allocatePartitionedAccelerationStructure(handle); break;
                    case ComputeResourceBindingKind::StorageImage:
                        result = tables->heap->allocateStorageImage(handle); break;
                    case ComputeResourceBindingKind::StorageBuffer:
                        result = tables->heap->allocateBuffer(handle); break;
                    case ComputeResourceBindingKind::SampledImage:
                        result = tables->heap->allocateSampledImage(handle); break;
                    }
                    if (!result) {
                        return result;
                    }
                    if (handle.shaderIndex != bindings[bindingIndex].handles[set * count + index].shaderIndex) {
                        return makeError(Error::Failure);
                    }
                    binding.handles.push_back(handle);
                }
            }
        }
        tables->usedTables.assign(resourceTableCount, false);
        tables->resourcePackets.resize(resourceTableCount);
        tables->usedTables[tableIndex] = true;
        tables->completion = frame.completion();
        frameTables.push_back(tables);
        outTables = std::move(tables);
        return {};
    }

    ~Impl()
    {
        destroy();
    }

    void destroy()
    {
        pipeline.reset();
        shader.reset();
        heap.reset();
        frameTables.clear();
        pushConstantSize = 0;
        resourceTableCount = 0;
        bindlessPushDataSize = 0;
        samplerBasePushDataOffset = UINT32_MAX;
        imageBasePushDataOffset = UINT32_MAX;
        bufferBasePushDataOffset = UINT32_MAX;
        resourcePackets.clear();
        bindings.clear();
        samplerBaseShaderIndices.clear();
        imageBaseShaderIndices.clear();
        bufferBaseShaderIndices.clear();
        debugName = "ComputeProgram";
    }
};

ComputeProgram::ComputeProgram()
    : impl_(std::make_shared<Impl>())
{
}

ComputeProgram::~ComputeProgram() = default;
ComputeProgram::ComputeProgram(ComputeProgram&&) noexcept = default;
ComputeProgram& ComputeProgram::operator=(ComputeProgram&&) noexcept = default;

Result ComputeProgram::initialize(
    Device& device,
    const ComputeProgramDesc& desc,
    std::string& log)
{
    if (impl_ == nullptr) {
        impl_ = std::make_shared<Impl>();
    }
    log.clear();

    if (desc.spirv == nullptr ||
        desc.byteSize == 0 ||
        (desc.byteSize % sizeof(uint32_t)) != 0 ||
        desc.bindings == nullptr ||
        desc.bindingCount == 0 ||
        desc.resourceTableCount == 0 ||
        hasDuplicateBindings(desc.bindings, desc.bindingCount)) {
        log = "ComputeProgramDesc is invalid";
        return makeError(Error::InvalidArgument);
    }
    if (desc.requiresRayQuery &&
        (!device.capabilities().rayTracingAccelerationStructure || !device.capabilities().rayQuery)) {
        log = "ComputeProgram requires rayTracingAccelerationStructure and rayQuery capabilities";
        return makeError(Error::Unsupported);
    }
    if (!device.capabilities().bindlessDescriptorHeap) {
        log = "ComputeProgram requires bindlessDescriptorHeap capability";
        return makeError(Error::Unsupported);
    }

    impl_ = std::make_shared<Impl>();
    impl_->device = &device;
    impl_->pushConstantSize = desc.pushConstantSize;
    impl_->resourceTableCount = desc.resourceTableCount;
    impl_->usesResourceTable = desc.usesResourceTable;
    impl_->resourcePackets.resize(desc.resourceTableCount);
    impl_->debugName = desc.debugName != nullptr ? desc.debugName : "ComputeProgram";

    uint64_t samplerCount = 0;
    uint64_t sampledImageCount = 0;
    uint64_t storageImageCount = 0;
    uint64_t bufferCount = 0;
    impl_->bindings.reserve(desc.bindingCount);
    for (uint32_t bindingIndex = 0; bindingIndex < desc.bindingCount; ++bindingIndex) {
        const ComputeProgramBindingDesc& binding = desc.bindings[bindingIndex];
        if (desc.usesResourceTable) {
            if (binding.binding >= kMaxComputeResourceSlots) {
                log = "ComputeProgram resource slot exceeds the native table limit";
                clear();
                return makeError(Error::InvalidArgument);
            }
            impl_->resourceSlotCount = std::max(impl_->resourceSlotCount, binding.binding + 1);
        }
        const uint32_t descriptorCount = std::max(binding.descriptorCount, 1u);
        if ((binding.kind == ComputeResourceBindingKind::Sampler ||
             binding.kind == ComputeResourceBindingKind::AccelerationStructure ||
             binding.kind == ComputeResourceBindingKind::PartitionedAccelerationStructure ||
             binding.kind == ComputeResourceBindingKind::StorageBuffer) &&
            descriptorCount != 1) {
            log = "ComputeProgram buffer and RTAS bindings must have descriptorCount 1";
            clear();
            return makeError(Error::InvalidArgument);
        }
        const uint64_t slotCount =
            static_cast<uint64_t>(descriptorCount) * desc.resourceTableCount;
        switch (binding.kind) {
        case ComputeResourceBindingKind::Sampler:
            samplerCount += slotCount;
            break;
        case ComputeResourceBindingKind::SampledImage:
            sampledImageCount += slotCount;
            break;
        case ComputeResourceBindingKind::StorageImage:
            storageImageCount += slotCount;
            break;
        case ComputeResourceBindingKind::StorageBuffer:
        case ComputeResourceBindingKind::AccelerationStructure:
        case ComputeResourceBindingKind::PartitionedAccelerationStructure:
            bufferCount += slotCount;
            break;
        }
        if (samplerCount > UINT32_MAX ||
            sampledImageCount + storageImageCount > UINT32_MAX ||
            bufferCount > UINT32_MAX) {
            log = "ComputeProgram bindless heap sizing overflowed";
            clear();
            return makeError(Error::InvalidArgument);
        }
        impl_->bindings.push_back(Impl::BindingState{
            .desc = binding,
        });
    }

    const bool hasSamplerBindings = samplerCount != 0;
    const bool hasImageBindings = sampledImageCount + storageImageCount != 0;
    const bool hasBufferBindings = bufferCount != 0;
    const bool dynamicDescriptorTables = !desc.usesResourceTable && desc.resourceTableCount > 1;
    const uint32_t pushedHeapBaseCount = dynamicDescriptorTables
        ? static_cast<uint32_t>(hasSamplerBindings) +
            static_cast<uint32_t>(hasImageBindings) +
            static_cast<uint32_t>(hasBufferBindings)
        : 0u;
    if (desc.pushConstantSize >
        UINT32_MAX - pushedHeapBaseCount * static_cast<uint32_t>(sizeof(uint32_t))) {
        log = "ComputeProgram bindless push-data sizing overflowed";
        clear();
        return makeError(Error::InvalidArgument);
    }
    uint32_t nextPushDataOffset = desc.pushConstantSize;
    if (hasSamplerBindings) {
        impl_->samplerBaseShaderIndices.assign(desc.resourceTableCount, UINT32_MAX);
        if (dynamicDescriptorTables) {
            impl_->samplerBasePushDataOffset = nextPushDataOffset;
            nextPushDataOffset += sizeof(uint32_t);
        }
    }
    if (hasImageBindings) {
        impl_->imageBaseShaderIndices.assign(desc.resourceTableCount, UINT32_MAX);
        if (dynamicDescriptorTables) {
            impl_->imageBasePushDataOffset = nextPushDataOffset;
            nextPushDataOffset += sizeof(uint32_t);
        }
    }
    if (hasBufferBindings) {
        impl_->bufferBaseShaderIndices.assign(desc.resourceTableCount, UINT32_MAX);
        if (dynamicDescriptorTables) {
            impl_->bufferBasePushDataOffset = nextPushDataOffset;
            nextPushDataOffset += sizeof(uint32_t);
        }
    }
    impl_->bindlessPushDataSize = desc.usesResourceTable ? sizeof(ComputeResourcePush) : nextPushDataOffset;

    Result result = device.createBindlessHeap(
        BindlessHeapDesc{
            .maxSamplers = static_cast<uint32_t>(samplerCount),
            .maxSampledImages = static_cast<uint32_t>(sampledImageCount),
            .maxStorageImages = static_cast<uint32_t>(storageImageCount),
            .maxBuffers = static_cast<uint32_t>(bufferCount),
        },
        impl_->heap);
    if (!result) {
        log = resultMessage("createBindlessHeap(ComputeProgram)", result);
        clear();
        return result;
    }

    for (Impl::BindingState& binding : impl_->bindings) {
        const uint32_t descriptorCount = std::max(binding.desc.descriptorCount, 1u);
        binding.handles.reserve(descriptorCount * desc.resourceTableCount);
    }
    for (uint32_t resourceTableIndex = 0;
         resourceTableIndex < desc.resourceTableCount;
         ++resourceTableIndex) {
        for (Impl::BindingState& binding : impl_->bindings) {
            const uint32_t descriptorCount = std::max(binding.desc.descriptorCount, 1u);
            std::vector<uint32_t>* groupBases = nullptr;
            if (usesSamplerHeap(binding.desc.kind)) {
                groupBases = &impl_->samplerBaseShaderIndices;
            } else if (usesImageHeap(binding.desc.kind)) {
                groupBases = &impl_->imageBaseShaderIndices;
            } else {
                groupBases = &impl_->bufferBaseShaderIndices;
            }
            for (uint32_t descriptorIndex = 0;
                 descriptorIndex < descriptorCount;
                 ++descriptorIndex) {
                BindlessHandle handle;
                switch (binding.desc.kind) {
                case ComputeResourceBindingKind::Sampler:
                    result = impl_->heap->allocateSampler(handle);
                    break;
                case ComputeResourceBindingKind::AccelerationStructure:
                    result = impl_->heap->allocateAccelerationStructure(handle);
                    break;
                case ComputeResourceBindingKind::PartitionedAccelerationStructure:
                    result = impl_->heap->allocatePartitionedAccelerationStructure(handle);
                    break;
                case ComputeResourceBindingKind::StorageImage:
                    result = impl_->heap->allocateStorageImage(handle);
                    break;
                case ComputeResourceBindingKind::StorageBuffer:
                    result = impl_->heap->allocateBuffer(handle);
                    break;
                case ComputeResourceBindingKind::SampledImage:
                    result = impl_->heap->allocateSampledImage(handle);
                    break;
                }
                if (!result) {
                    log = resultMessage("allocate bindless ComputeProgram slot", result);
                    clear();
                    return result;
                }
                if ((*groupBases)[resourceTableIndex] == UINT32_MAX) {
                    (*groupBases)[resourceTableIndex] = handle.shaderIndex;
                }
                if (resourceTableIndex == 0 && descriptorIndex == 0) {
                    binding.heapIndexOffset =
                        handle.shaderIndex - (*groupBases)[resourceTableIndex];
                }
                const uint32_t expectedShaderIndex = (*groupBases)[resourceTableIndex] +
                    binding.heapIndexOffset + descriptorIndex;
                if (handle.shaderIndex != expectedShaderIndex) {
                    log = "ComputeProgram descriptor-table allocation is not contiguous";
                    clear();
                    return makeError(Error::Failure);
                }
                binding.handles.push_back(handle);
            }
        }
    }

    std::vector<ShaderBindingMappingDesc> mappings;
    mappings.reserve(impl_->bindings.size());
    for (const Impl::BindingState& binding : impl_->bindings) {
        if (desc.usesResourceTable) {
            break;
        }
        const bool samplerBinding = usesSamplerHeap(binding.desc.kind);
        const bool imageBinding = usesImageHeap(binding.desc.kind);
        mappings.push_back(ShaderBindingMappingDesc{
            .descriptorSet = 0,
            .firstBinding = binding.desc.binding,
            .bindingCount = 1,
            .type = shaderBindingType(binding.desc.kind),
            .source = dynamicDescriptorTables
                ? ShaderBindingSource::HeapIndexFromPushData
                : ShaderBindingSource::HeapConstantOffset,
            .pushDataOffset = dynamicDescriptorTables
                ? samplerBinding
                    ? impl_->samplerBasePushDataOffset
                    : imageBinding
                    ? impl_->imageBasePushDataOffset
                    : impl_->bufferBasePushDataOffset
                : 0u,
            .heapIndexOffset = dynamicDescriptorTables
                ? binding.heapIndexOffset
                : binding.handles.front().shaderIndex,
        });
    }

    result = device.createShaderModule(
        ShaderModuleDesc{
            .code = desc.spirv,
            .byteSize = desc.byteSize,
            .debugName = impl_->debugName.c_str(),
        },
        impl_->shader);
    if (!result) {
        log = resultMessage("createShaderModule(ComputeProgram)", result);
        clear();
        return result;
    }
    result = device.createComputePipeline(
        ComputePipelineDesc{
            .computeShader = impl_->shader.get(),
            .computeEntryPoint = "main",
            .usesBindlessHeap = true,
            .bindlessUserPushDataSize = impl_->bindlessPushDataSize,
            .bindingMappings = mappings.data(),
            .bindingMappingCount = static_cast<uint32_t>(mappings.size()),
            .pipelineCache = desc.pipelineCache,
        },
        impl_->pipeline);
    if (!result) {
        log = resultMessage("createComputePipeline(ComputeProgram bindless)", result);
        clear();
        return result;
    }

    return {};
}

void ComputeProgram::clear()
{
    impl_ = std::make_shared<Impl>();
}

bool ComputeProgram::valid() const
{
    return impl_ != nullptr &&
        impl_->shader != nullptr &&
        impl_->pipeline != nullptr &&
        impl_->heap != nullptr &&
        impl_->resourceTableCount != 0;
}

Result ComputeProgram::dispatch(const ComputeDispatchDesc& desc)
{
    return dispatchImpl(desc, {}, {});
}

Result ComputeProgram::dispatchIndirectBatch(const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches)
{
    if (desc.indirectArguments == nullptr || dispatches.empty()) {
        return makeError(Error::InvalidArgument);
    }
    ComputeDispatchDesc first = desc;
    first.pushData = dispatches.front().pushData;
    first.indirectOffset = dispatches.front().argumentOffset;
    return dispatchImpl(first, dispatches, betweenDispatches);
}

Result ComputeProgram::dispatchImpl(const ComputeDispatchDesc& desc,
    std::span<const ComputeIndirectDispatch> dispatches, const BarrierDesc& betweenDispatches)
{
    if (!valid() ||
        desc.commandBuffer == nullptr ||
        (desc.indirectArguments == nullptr &&
         (desc.groupCountX == 0 || desc.groupCountY == 0 || desc.groupCountZ == 0)) ||
        desc.resourceTableIndex >= impl_->resourceTableCount ||
        (impl_->pushConstantSize > 0 &&
         (desc.pushData == nullptr || desc.pushDataSize != impl_->pushConstantSize)) ||
        (impl_->pushConstantSize == 0 && desc.pushDataSize != 0)) {
        spdlog::error("[ComputeProgram:{}] invalid dispatch description", impl_->debugName);
        return makeError(Error::InvalidArgument);
    }

    std::shared_ptr<ComputeDescriptorTables> retainedTables;
    if (desc.indirectArguments != nullptr &&
        ((static_cast<uint32_t>(desc.indirectArguments->desc().usage) &
          static_cast<uint32_t>(BufferUsageBits::Indirect)) == 0 ||
         (desc.indirectOffset & 3u) != 0 ||
         desc.indirectOffset > desc.indirectArguments->desc().size ||
         3 * sizeof(uint32_t) > desc.indirectArguments->desc().size - desc.indirectOffset)) {
        return makeError(Error::InvalidArgument);
    }
    ComputeDescriptorTables* tables = impl_.get();
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
    if (RenderFrameContext* frame = desc.commandBuffer->frameContext()) {
        if (!frame->recording()) {
            return makeError(Error::InvalidArgument);
        }
        Result result = impl_->acquireTables(*frame, desc.resourceTableIndex, retainedTables);
        if (!result) {
            return result;
        }
        tables = retainedTables.get();
        frame->retain(retainedTables);
        frame->retain(impl_);
        for (const auto& item : dispatches) {
            if (item.program != nullptr && item.program != this) { frame->retain(item.program->impl_); }
        }
    }

    std::vector<uint8_t> pushData(impl_->bindlessPushDataSize, 0);
    if (!impl_->usesResourceTable && impl_->pushConstantSize != 0) {
        std::memcpy(pushData.data(), desc.pushData, impl_->pushConstantSize);
    }
    if (impl_->imageBasePushDataOffset != UINT32_MAX) {
        const uint32_t imageBase =
            tables->imageBaseShaderIndices[desc.resourceTableIndex];
        std::memcpy(
            pushData.data() + impl_->imageBasePushDataOffset,
            &imageBase,
            sizeof(imageBase));
    }
    if (impl_->samplerBasePushDataOffset != UINT32_MAX) {
        const uint32_t samplerBase =
            tables->samplerBaseShaderIndices[desc.resourceTableIndex];
        std::memcpy(
            pushData.data() + impl_->samplerBasePushDataOffset,
            &samplerBase,
            sizeof(samplerBase));
    }
    if (impl_->bufferBasePushDataOffset != UINT32_MAX) {
        const uint32_t bufferBase =
            tables->bufferBaseShaderIndices[desc.resourceTableIndex];
        std::memcpy(
            pushData.data() + impl_->bufferBasePushDataOffset,
            &bufferBase,
            sizeof(bufferBase));
    }

    for (const Impl::BindingState& expectedBinding : tables->bindings) {
        const ComputeDispatchBinding* binding =
            findDispatchBinding(desc, expectedBinding.desc.binding);
        if (binding == nullptr) {
            std::string providedBindings;
            for (uint32_t index = 0; index < desc.bindingCount; ++index) {
                if (!providedBindings.empty()) {
                    providedBindings += ',';
                }
                providedBindings += std::to_string(desc.bindings[index].binding);
            }
            spdlog::error(
                "[ComputeProgram:{}] missing binding {}; provided count={} bindings=[{}]",
                impl_->debugName,
                expectedBinding.desc.binding,
                desc.bindingCount,
                providedBindings);
            return makeError(Error::InvalidArgument);
        }
        const uint32_t descriptorCount =
            std::max(expectedBinding.desc.descriptorCount, 1u);
        const uint32_t firstHandle = desc.resourceTableIndex * descriptorCount;
        if (firstHandle >= expectedBinding.handles.size() ||
            descriptorCount > expectedBinding.handles.size() - firstHandle) {
            return makeError(Error::Failure);
        }
        Result result;
        switch (expectedBinding.desc.kind) {
        case ComputeResourceBindingKind::Sampler: {
            if (binding->sampler == nullptr) {
                spdlog::error(
                    "[ComputeProgram:{}] invalid sampler binding {}",
                    impl_->debugName,
                    expectedBinding.desc.binding);
                return makeError(Error::InvalidArgument);
            }
            result = tables->heap->writeSampler(
                expectedBinding.handles[firstHandle],
                *binding->sampler);
            break;
        }
        case ComputeResourceBindingKind::AccelerationStructure: {
            if (binding->accelerationStructure == nullptr ||
                !binding->accelerationStructure->valid()) {
                spdlog::error(
                    "[ComputeProgram:{}] invalid RTAS binding {}",
                    impl_->debugName,
                    expectedBinding.desc.binding);
                return makeError(Error::InvalidArgument);
            }
            result = tables->heap->writeAccelerationStructure(
                expectedBinding.handles[firstHandle],
                *binding->accelerationStructure);
            break;
        }
        case ComputeResourceBindingKind::PartitionedAccelerationStructure: {
            if (binding->partitionedAccelerationStructure == nullptr ||
                !binding->partitionedAccelerationStructure->valid()) {
                spdlog::error(
                    "[ComputeProgram:{}] invalid partitioned RTAS binding {}",
                    impl_->debugName,
                    expectedBinding.desc.binding);
                return makeError(Error::InvalidArgument);
            }
            result = tables->heap->writePartitionedAccelerationStructure(
                expectedBinding.handles[firstHandle],
                *binding->partitionedAccelerationStructure);
            break;
        }
        case ComputeResourceBindingKind::StorageImage: {
            const bool useTextureArray =
                binding->textureViews != nullptr && binding->textureViewCount >= descriptorCount;
            if (!useTextureArray && (descriptorCount != 1u || binding->textureView == nullptr)) {
                spdlog::error(
                    "[ComputeProgram:{}] invalid storage image binding {}",
                    impl_->debugName,
                    expectedBinding.desc.binding);
                return makeError(Error::InvalidArgument);
            }
            for (uint32_t index = 0; index < descriptorCount; ++index) {
                TextureView* textureView = useTextureArray
                    ? binding->textureViews[index]
                    : binding->textureView;
                if (textureView == nullptr) {
                    spdlog::error(
                        "[ComputeProgram:{}] null storage image binding {}[{}]",
                        impl_->debugName,
                        expectedBinding.desc.binding,
                        index);
                    return makeError(Error::InvalidArgument);
                }
                result = tables->heap->writeStorageImage(
                    expectedBinding.handles[firstHandle + index],
                    *textureView);
                if (!result) {
                    return result;
                }
            }
            break;
        }
        case ComputeResourceBindingKind::SampledImage: {
            if (binding->textureViews == nullptr || binding->textureViewCount < descriptorCount) {
                spdlog::error(
                    "[ComputeProgram:{}] sampled image binding {} has {} views, expected {}",
                    impl_->debugName,
                    expectedBinding.desc.binding,
                    binding->textureViewCount,
                    descriptorCount);
                return makeError(Error::InvalidArgument);
            }
            for (uint32_t index = 0; index < descriptorCount; ++index) {
                TextureView* textureView = binding->textureViews[index];
                if (textureView == nullptr) {
                    spdlog::error(
                        "[ComputeProgram:{}] null sampled image binding {}[{}]",
                        impl_->debugName,
                        expectedBinding.desc.binding,
                        index);
                    return makeError(Error::InvalidArgument);
                }
                result = tables->heap->writeSampledImage(
                    expectedBinding.handles[firstHandle + index],
                    *textureView,
                    ResourceState::ShaderRead);
                if (!result) {
                    return result;
                }
            }
            break;
        }
        case ComputeResourceBindingKind::StorageBuffer: {
            if (binding->buffer == nullptr || binding->offset != 0 ||
                (binding->size != UINT64_MAX &&
                 binding->size != binding->buffer->desc().size)) {
                spdlog::error(
                    "[ComputeProgram:{}] invalid storage buffer binding {} offset={} size={}",
                    impl_->debugName,
                    expectedBinding.desc.binding,
                    binding->offset,
                    binding->size);
                return makeError(Error::InvalidArgument);
            }
            result = tables->heap->writeStorageBuffer(
                expectedBinding.handles[firstHandle],
                *binding->buffer);
            break;
        }
        }
        if (!result) {
            spdlog::error(
                "[ComputeProgram:{}] descriptor write failed at binding {}: {}",
                impl_->debugName,
                expectedBinding.desc.binding,
                resultToString(result));
            return result;
        }
    }

    uint64_t constantsAddress = 0;
    const uint64_t constantStride = (uint64_t(impl_->pushConstantSize) + 15u) & ~uint64_t(15u);
    if (impl_->usesResourceTable) {
        const uint64_t constantsOffset = (uint64_t(impl_->resourceSlotCount) * sizeof(uint64_t) + 15u) & ~uint64_t(15u);
        const uint64_t dispatchCount = std::max<size_t>(dispatches.size(), 1);
        if (constantStride != 0 && dispatchCount > (UINT64_MAX - constantsOffset) / constantStride) {
            return makeError(Error::InvalidArgument);
        }
        const uint64_t packetSize = constantsOffset + constantStride * dispatchCount;
        auto& packet = tables->resourcePackets[desc.resourceTableIndex];
        if (packet == nullptr || packet->desc().size < packetSize) {
            Result result = impl_->device->createBuffer(BufferDesc{
                .size = packetSize,
                .usage = BufferUsageBits::Storage | BufferUsageBits::ShaderDeviceAddress,
                .memoryLocation = MemoryLocation::HostUpload,
                .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Compute,
            }, packet);
            if (!result) {
                return result;
            }
        }
        auto* bytes = static_cast<uint8_t*>(packet->map());
        if (bytes == nullptr || packet->deviceAddress() == 0) {
            if (bytes != nullptr) { packet->unmap(); }
            return makeError(Error::Failure);
        }
        std::memset(bytes, 0xff, static_cast<size_t>(constantsOffset));
        for (const auto& binding : tables->bindings) {
            const uint32_t descriptorCount = std::max(binding.desc.descriptorCount, 1u);
            uint64_t handle = binding.handles[desc.resourceTableIndex * descriptorCount].shaderIndex;
            // Slang 2026.1.2 lowers DescriptorHandle<RTAS> to an address cast,
            // unlike its image/buffer/sampler handles, which index heap arrays.
            const auto* resource = findDispatchBinding(desc, binding.desc.binding);
            if (binding.desc.kind == ComputeResourceBindingKind::AccelerationStructure) {
                handle = resource->accelerationStructure->deviceAddress();
            } else if (binding.desc.kind == ComputeResourceBindingKind::PartitionedAccelerationStructure) {
                handle = resource->partitionedAccelerationStructure->deviceAddress();
            }
            std::memcpy(bytes + binding.desc.binding * sizeof(uint64_t), &handle, sizeof(handle));
        }
        if (impl_->pushConstantSize != 0) {
            for (size_t index = 0; index < dispatchCount; ++index) {
                const void* constants = dispatches.empty() ? desc.pushData : dispatches[index].pushData;
                std::memcpy(bytes + constantsOffset + constantStride * index, constants, impl_->pushConstantSize);
            }
        }
        packet->flush();
        packet->unmap();
        constantsAddress = packet->deviceAddress() + constantsOffset;
        const ComputeResourcePush push{packet->deviceAddress(), constantsAddress};
        std::memcpy(pushData.data(), &push, sizeof(push));
    }

    desc.commandBuffer->bindBindlessHeap(*tables->heap);
    desc.commandBuffer->bindComputePipeline(*impl_->pipeline);
    if (!dispatches.empty()) {
        const Impl* boundProgram = impl_.get();
        for (size_t index = 0; index < dispatches.size(); ++index) {
            const Impl* program = dispatches[index].program != nullptr ? dispatches[index].program->impl_.get() : impl_.get();
            if (program != boundProgram) {
                desc.commandBuffer->bindComputePipeline(*program->pipeline);
                boundProgram = program;
            }
            if (impl_->usesResourceTable) {
                const uint64_t address = constantsAddress + constantStride * index;
                std::memcpy(pushData.data() + offsetof(ComputeResourcePush, constants), &address, sizeof(address));
            } else if (impl_->pushConstantSize > 0) {
                std::memcpy(pushData.data(), dispatches[index].pushData, impl_->pushConstantSize);
            }
            desc.commandBuffer->pushBindlessData(pushData.data(), static_cast<uint32_t>(pushData.size()));
            auto result = desc.commandBuffer->dispatchIndirect(*desc.indirectArguments, dispatches[index].argumentOffset);
            if (!result) { return result; }
            if (index + 1 < dispatches.size() && (betweenDispatches.bufferCount > 0 || betweenDispatches.textureCount > 0)) {
                desc.commandBuffer->barrier(betweenDispatches);
            }
        }
        return {};
    }
    desc.commandBuffer->pushBindlessData(
        pushData.data(),
        static_cast<uint32_t>(pushData.size()));
    if (desc.indirectArguments != nullptr) {
        return desc.commandBuffer->dispatchIndirect(*desc.indirectArguments, desc.indirectOffset);
    }
    desc.commandBuffer->dispatch(
        desc.groupCountX,
        desc.groupCountY,
        desc.groupCountZ);
    return {};
}

} // namespace metallic::render

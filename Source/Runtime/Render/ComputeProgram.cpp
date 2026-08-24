#include "Runtime/Render/ComputeProgram.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace metallic::render {
namespace {

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

struct ComputeProgram::Impl {
    struct BindingState {
        ComputeProgramBindingDesc desc;
        uint32_t heapIndexOffset = 0;
        std::vector<BindlessHandle> handles;
    };

    std::unique_ptr<ShaderModule> shader;
    std::unique_ptr<ComputePipeline> pipeline;
    std::unique_ptr<BindlessHeap> heap;
    uint32_t pushConstantSize = 0;
    uint32_t descriptorSetCount = 0;
    uint32_t bindlessPushDataSize = 0;
    uint32_t samplerBasePushDataOffset = UINT32_MAX;
    uint32_t imageBasePushDataOffset = UINT32_MAX;
    uint32_t bufferBasePushDataOffset = UINT32_MAX;
    std::string debugName = "ComputeProgram";
    std::vector<BindingState> bindings;
    std::vector<uint32_t> samplerBaseShaderIndices;
    std::vector<uint32_t> imageBaseShaderIndices;
    std::vector<uint32_t> bufferBaseShaderIndices;

    ~Impl()
    {
        destroy();
    }

    void destroy()
    {
        pipeline.reset();
        shader.reset();
        heap.reset();
        pushConstantSize = 0;
        descriptorSetCount = 0;
        bindlessPushDataSize = 0;
        samplerBasePushDataOffset = UINT32_MAX;
        imageBasePushDataOffset = UINT32_MAX;
        bufferBasePushDataOffset = UINT32_MAX;
        bindings.clear();
        samplerBaseShaderIndices.clear();
        imageBaseShaderIndices.clear();
        bufferBaseShaderIndices.clear();
        debugName = "ComputeProgram";
    }
};

ComputeProgram::ComputeProgram()
    : impl_(std::make_unique<Impl>())
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
        impl_ = std::make_unique<Impl>();
    }
    log.clear();

    if (desc.spirv == nullptr ||
        desc.byteSize == 0 ||
        (desc.byteSize % sizeof(uint32_t)) != 0 ||
        desc.bindings == nullptr ||
        desc.bindingCount == 0 ||
        desc.descriptorSetCount == 0 ||
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

    impl_->destroy();
    impl_->pushConstantSize = desc.pushConstantSize;
    impl_->descriptorSetCount = desc.descriptorSetCount;
    impl_->debugName = desc.debugName != nullptr ? desc.debugName : "ComputeProgram";

    uint64_t samplerCount = 0;
    uint64_t sampledImageCount = 0;
    uint64_t storageImageCount = 0;
    uint64_t bufferCount = 0;
    impl_->bindings.reserve(desc.bindingCount);
    for (uint32_t bindingIndex = 0; bindingIndex < desc.bindingCount; ++bindingIndex) {
        const ComputeProgramBindingDesc& binding = desc.bindings[bindingIndex];
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
            static_cast<uint64_t>(descriptorCount) * desc.descriptorSetCount;
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
    const bool dynamicDescriptorTables = desc.descriptorSetCount > 1;
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
        impl_->samplerBaseShaderIndices.assign(desc.descriptorSetCount, UINT32_MAX);
        if (dynamicDescriptorTables) {
            impl_->samplerBasePushDataOffset = nextPushDataOffset;
            nextPushDataOffset += sizeof(uint32_t);
        }
    }
    if (hasImageBindings) {
        impl_->imageBaseShaderIndices.assign(desc.descriptorSetCount, UINT32_MAX);
        if (dynamicDescriptorTables) {
            impl_->imageBasePushDataOffset = nextPushDataOffset;
            nextPushDataOffset += sizeof(uint32_t);
        }
    }
    if (hasBufferBindings) {
        impl_->bufferBaseShaderIndices.assign(desc.descriptorSetCount, UINT32_MAX);
        if (dynamicDescriptorTables) {
            impl_->bufferBasePushDataOffset = nextPushDataOffset;
            nextPushDataOffset += sizeof(uint32_t);
        }
    }
    impl_->bindlessPushDataSize = nextPushDataOffset;

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
        binding.handles.reserve(descriptorCount * desc.descriptorSetCount);
    }
    for (uint32_t descriptorSetIndex = 0;
         descriptorSetIndex < desc.descriptorSetCount;
         ++descriptorSetIndex) {
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
                if ((*groupBases)[descriptorSetIndex] == UINT32_MAX) {
                    (*groupBases)[descriptorSetIndex] = handle.shaderIndex;
                }
                if (descriptorSetIndex == 0 && descriptorIndex == 0) {
                    binding.heapIndexOffset =
                        handle.shaderIndex - (*groupBases)[descriptorSetIndex];
                }
                const uint32_t expectedShaderIndex = (*groupBases)[descriptorSetIndex] +
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
    if (impl_ != nullptr) {
        impl_->destroy();
    }
}

bool ComputeProgram::valid() const
{
    return impl_ != nullptr &&
        impl_->shader != nullptr &&
        impl_->pipeline != nullptr &&
        impl_->heap != nullptr &&
        impl_->descriptorSetCount != 0;
}

Result ComputeProgram::dispatch(const ComputeDispatchDesc& desc)
{
    if (!valid() ||
        desc.commandBuffer == nullptr ||
        desc.groupCountX == 0 ||
        desc.groupCountY == 0 ||
        desc.groupCountZ == 0 ||
        desc.descriptorSetIndex >= impl_->descriptorSetCount ||
        (impl_->pushConstantSize > 0 &&
         (desc.pushData == nullptr || desc.pushDataSize != impl_->pushConstantSize)) ||
        (impl_->pushConstantSize == 0 && desc.pushDataSize != 0)) {
        spdlog::error("[ComputeProgram:{}] invalid dispatch description", impl_->debugName);
        return makeError(Error::InvalidArgument);
    }

    std::vector<uint8_t> pushData(impl_->bindlessPushDataSize, 0);
    if (impl_->pushConstantSize != 0) {
        std::memcpy(pushData.data(), desc.pushData, impl_->pushConstantSize);
    }
    if (impl_->imageBasePushDataOffset != UINT32_MAX) {
        const uint32_t imageBase =
            impl_->imageBaseShaderIndices[desc.descriptorSetIndex];
        std::memcpy(
            pushData.data() + impl_->imageBasePushDataOffset,
            &imageBase,
            sizeof(imageBase));
    }
    if (impl_->samplerBasePushDataOffset != UINT32_MAX) {
        const uint32_t samplerBase =
            impl_->samplerBaseShaderIndices[desc.descriptorSetIndex];
        std::memcpy(
            pushData.data() + impl_->samplerBasePushDataOffset,
            &samplerBase,
            sizeof(samplerBase));
    }
    if (impl_->bufferBasePushDataOffset != UINT32_MAX) {
        const uint32_t bufferBase =
            impl_->bufferBaseShaderIndices[desc.descriptorSetIndex];
        std::memcpy(
            pushData.data() + impl_->bufferBasePushDataOffset,
            &bufferBase,
            sizeof(bufferBase));
    }

    for (const Impl::BindingState& expectedBinding : impl_->bindings) {
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
        const uint32_t firstHandle = desc.descriptorSetIndex * descriptorCount;
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
            result = impl_->heap->writeSampler(
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
            result = impl_->heap->writeAccelerationStructure(
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
            result = impl_->heap->writePartitionedAccelerationStructure(
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
                result = impl_->heap->writeStorageImage(
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
                result = impl_->heap->writeSampledImage(
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
            result = impl_->heap->writeStorageBuffer(
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

    desc.commandBuffer->bindBindlessHeap(*impl_->heap);
    desc.commandBuffer->bindComputePipeline(*impl_->pipeline);
    desc.commandBuffer->pushBindlessData(
        pushData.data(),
        static_cast<uint32_t>(pushData.size()));
    desc.commandBuffer->dispatch(
        desc.groupCountX,
        desc.groupCountY,
        desc.groupCountZ);
    return {};
}

} // namespace metallic::render

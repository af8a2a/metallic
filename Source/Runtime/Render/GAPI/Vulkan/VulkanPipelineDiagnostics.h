#pragma once
#include "Runtime/Render/GAPI/RHI.h"
#include "Runtime/Render/GAPI/PipelineStateHash.h"
#include "Runtime/Render/GAPI/Hash.h"

namespace metallic::render::detail {

// Raw descriptor decoration experiments only. Production pipelines use the
// default backend mapping and the named ComputeKernel resource ABI.
enum class ShaderBindingType : uint8_t {
    Sampler,
    SampledImage,
    StorageImage,
    ConstantBuffer,
    StorageBuffer,
    AccelerationStructure,
};

enum class ShaderBindingSource : uint8_t {
    HeapConstantOffset,
    HeapIndexFromPushData,
    DeviceAddressFromPushData,
};

// Maps existing DescriptorSet/Binding decorations onto descriptor-heap data.
// pushDataOffset is relative to the user payload passed to pushBindlessData().
// heapIndexOffset is expressed in descriptors. It is an absolute heap index
// for HeapConstantOffset and is added to the pushed heap index for
// HeapIndexFromPushData, allowing several bindings to share one pushed base.
struct ShaderBindingMappingDesc {
    uint32_t descriptorSet = 0;
    uint32_t firstBinding = 0;
    uint32_t bindingCount = 1;
    ShaderBindingType type = ShaderBindingType::SampledImage;
    ShaderBindingSource source = ShaderBindingSource::HeapIndexFromPushData;
    uint32_t pushDataOffset = 0;
    uint32_t heapIndexOffset = 0;
};

Result<std::unique_ptr<ComputePipeline>> createMappedComputePipeline(Device& device,
    const ComputePipelineDesc& desc, std::span<const ShaderBindingMappingDesc> mappings);

inline uint64_t mappedComputePipelineStateHash(const ComputePipelineDesc& desc,
    std::span<const ShaderBindingMappingDesc> mappings)
{
    uint64_t hash = computePipelineStateHash(desc);
    if (mappings.empty()) { return hash; }
    hash = hashValue(hash, uint32_t{0x564b4d50}); // Vulkan mapping diagnostic domain.
    hash = hashValue(hash, static_cast<uint32_t>(mappings.size()));
    for (const auto& mapping : mappings) {
        hash = hashValue(hash, mapping.descriptorSet);
        hash = hashValue(hash, mapping.firstBinding);
        hash = hashValue(hash, mapping.bindingCount);
        hash = hashValue(hash, static_cast<uint32_t>(mapping.type));
        hash = hashValue(hash, static_cast<uint32_t>(mapping.source));
        hash = hashValue(hash, mapping.pushDataOffset);
        hash = hashValue(hash, mapping.heapIndexOffset);
    }
    return hash;
}
} // namespace metallic::render::detail

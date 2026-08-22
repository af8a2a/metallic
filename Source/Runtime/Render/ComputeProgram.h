#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cstdint>
#include <memory>
#include <string>

namespace metallic::render {

enum class ComputeResourceBindingKind : uint8_t {
    AccelerationStructure,
    PartitionedAccelerationStructure,
    StorageImage,
    StorageBuffer,
    SampledImage,
};

struct ComputeProgramBindingDesc {
    uint32_t binding = 0;
    ComputeResourceBindingKind kind = ComputeResourceBindingKind::StorageBuffer;
    uint32_t descriptorCount = 1;
};

struct ComputeProgramDesc {
    const uint32_t* spirv = nullptr;
    uint64_t byteSize = 0;
    uint32_t pushConstantSize = 0;
    const ComputeProgramBindingDesc* bindings = nullptr;
    uint32_t bindingCount = 0;
    const char* debugName = nullptr;
    uint32_t descriptorSetCount = 1;
    bool requiresRayQuery = true;
};

struct ComputeDispatchBinding {
    uint32_t binding = 0;
    RayTracingAccelerationStructure* accelerationStructure = nullptr;
    PartitionedAccelerationStructure* partitionedAccelerationStructure = nullptr;
    TextureView* textureView = nullptr;
    TextureView* const* textureViews = nullptr;
    uint32_t textureViewCount = 0;
    Buffer* buffer = nullptr;
    uint64_t offset = 0;
    uint64_t size = UINT64_MAX;
};

struct ComputeDispatchDesc {
    CommandBuffer* commandBuffer = nullptr;
    const ComputeDispatchBinding* bindings = nullptr;
    uint32_t bindingCount = 0;
    const void* pushData = nullptr;
    uint32_t pushDataSize = 0;
    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;
    uint32_t descriptorSetIndex = 0;
};

class ComputeProgram {
public:
    ComputeProgram();
    ~ComputeProgram();

    ComputeProgram(ComputeProgram&&) noexcept;
    ComputeProgram& operator=(ComputeProgram&&) noexcept;

    ComputeProgram(const ComputeProgram&) = delete;
    ComputeProgram& operator=(const ComputeProgram&) = delete;

    Result initialize(Device& device, const ComputeProgramDesc& desc, std::string& log);
    void clear();
    bool valid() const;
    Result dispatch(const ComputeDispatchDesc& desc);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metallic::render

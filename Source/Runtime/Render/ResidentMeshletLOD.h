#pragma once

#include "Runtime/Render/MeshletLOD.h"
#include "Runtime/Render/Core/ComputeKernel.h"
#include "Runtime/Render/Subsystem/GPUScene.h"

namespace metallic::render {

class ResidentMeshletLOD {
public:
    Result<> initialize(Device& device, uint32_t capacity, std::string& log);
    Result<> record(CommandBuffer& commands, ResourceRegistry& registry,
        const GPUSceneGlobalBufferViews& inputs, const MeshletLODView& view,
        GPUSceneRasterDrawRange candidates, uint32_t instanceCount, uint32_t groupCount, uint32_t manualLevel = UINT32_MAX);
    Buffer& selections() const { return *selections_; }
    Buffer& arguments() const { return *arguments_; }
    Buffer& scratch() const { return *scratch_; }
    uint32_t capacity() const { return capacity_; }

private:
    std::unique_ptr<Buffer> selections_;
    std::unique_ptr<Buffer> arguments_;
    std::unique_ptr<Buffer> scratch_;
    Device* device_ = nullptr;
    std::array<ComputeKernel, 4> kernels_;
    uint32_t capacity_ = 0;
    bool initialized_ = false;
};

} // namespace metallic::render

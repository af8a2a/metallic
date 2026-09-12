#pragma once

#include "Runtime/Render/MeshletLod.h"
#include "Runtime/Render/Subsystem/GPUScene.h"

namespace metallic::render {

class ResidentMeshletLod {
public:
    Result initialize(Device& device, uint32_t capacity, std::string& log);
    Result record(CommandBuffer& commands, BindlessHeap& heap,
        const GPUSceneConsumerBindings& bindings, const MeshletLodView& view,
        GPUSceneRasterDrawRange candidates, uint32_t instanceCount, uint32_t groupCount,
        BindlessHandle output, BindlessHandle arguments, BindlessHandle scratch, uint32_t manualLevel = UINT32_MAX);
    Buffer& selections() const { return *selections_; }
    Buffer& arguments() const { return *arguments_; }
    Buffer& scratch() const { return *scratch_; }
    uint32_t capacity() const { return capacity_; }

private:
    std::unique_ptr<Buffer> selections_;
    std::unique_ptr<Buffer> arguments_;
    std::unique_ptr<Buffer> scratch_;
    std::array<std::unique_ptr<ShaderModule>, 4> shaders_;
    std::array<std::unique_ptr<ComputePipeline>, 4> pipelines_;
    uint32_t capacity_ = 0;
    bool initialized_ = false;
};

} // namespace metallic::render

#pragma once

#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderFrameContext.h"

#include <array>
#include <vector>

namespace metallic::render {

inline constexpr uint32_t kMaterialClassCount = 5;
inline constexpr uint32_t kMaterialTileWidth = 8;
inline constexpr uint32_t kMaterialTileHeight = 4;

struct MaterialBinningDesc {
    TextureView* visibility = nullptr;
    Buffer* records = nullptr;
    Buffer* instances = nullptr;
    Buffer* materials = nullptr;
    Buffer* shadingMaterials = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct MaterialBinningResult {
    Buffer* bins = nullptr; // uint2 {task offset, task count}, indexed by shading class
    Buffer* tiles = nullptr; // uint2 {linear 8x4 tile index, wave32 lane mask}
    Buffer* arguments = nullptr; // three uint32 dispatch counts per bin
    uint32_t binCount = 0;
};

// Substrate-style feature classification with separate masks for mixed tiles.
// Native wave32 is required; scratch storage is retained until frame completion.
class MaterialBinning {
public:
    Result record(Device& device, CommandBuffer& commands, const MaterialBinningDesc& desc,
        MaterialBinningResult& output, std::string& log);
    void clear();

private:
    struct Allocation;
    std::array<ComputeProgram, 3> programs_;
    std::vector<std::shared_ptr<Allocation>> allocations_;
};

} // namespace metallic::render

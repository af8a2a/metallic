#pragma once

#include "Runtime/Render/ComputeProgram.h"
#include "Runtime/Render/RenderFrameContext.h"

#include <array>
#include <vector>

namespace metallic::render {

struct MaterialBinningDesc {
    TextureView* visibility = nullptr;
    Buffer* records = nullptr;
    Buffer* instances = nullptr;
    Buffer* materials = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t materialCount = 0;
};

struct MaterialBinningResult {
    Buffer* bins = nullptr; // uint2 {offset, count}, indexed by source material + 1
    Buffer* pixels = nullptr; // packed linear pixel IDs, background is bin 0
    Buffer* arguments = nullptr; // three uint32 dispatch counts per bin
    uint32_t binCount = 0;
};

// GPU-only count/allocation/scatter. All scratch storage is frame-retained;
// dimensions and material counts may change while older frames are in flight.
class MaterialBinning {
public:
    Result record(Device& device, CommandBuffer& commands, const MaterialBinningDesc& desc,
        MaterialBinningResult& output, std::string& log);
    void clear();

private:
    struct Allocation;
    std::array<ComputeProgram, 4> programs_;
    std::vector<std::shared_ptr<Allocation>> allocations_;
};

} // namespace metallic::render

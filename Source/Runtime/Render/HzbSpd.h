#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <cstdint>

namespace metallic::render {

inline constexpr uint32_t kHzbSpdTileSize = 64;
inline constexpr uint32_t kHzbSpdMaxDimension = 4096;
inline constexpr const char* kHzbSpdModule = "Features/GPUDriven/HzbSpd";
inline constexpr const char* kHzbSpdEntryPoint = "hzbSpdMain";
inline constexpr const char* kHzbSpdWaveOpsDefine = "HZB_SPD_WAVE_OPS";

inline bool supportsHzbSpdWaveOps(const DeviceCapabilities& capabilities)
{
    // Compiled as SPIR-V 1.6: 256 threads in X guarantees full subgroups for
    // every possible wave size. Each shuffle operates within a 16-lane block.
    return capabilities.computeSubgroupShuffle && capabilities.minSubgroupSize >= 16 &&
        capabilities.maxSubgroupSize <= 256 && capabilities.minSubgroupSize <= capabilities.maxSubgroupSize;
}

// Native bindless header is prepended by the RHI. See HzbSpd.slang.
struct HzbSpdUserPush {
    uint32_t depthImage = 0;
    uint32_t hzbBuffer = 0;
    uint32_t counterBuffer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 0; // Includes full-resolution mip 0.
    uint32_t reversedZ = 1;
};
static_assert(sizeof(HzbSpdUserPush) == 28);

} // namespace metallic::render

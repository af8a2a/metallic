#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <cstdint>

namespace metallic::render {

inline constexpr uint32_t kHZBSPDTileSize = 64;
inline constexpr uint32_t kHZBSPDMaxDimension = 4096;
inline constexpr const char* kHZBSPDModule = "Features/GPUDriven/HZBSPD";
inline constexpr const char* kHZBSPDEntryPoint = "hzbSpdMain";
inline constexpr const char* kHZBSPDWaveOpsDefine = "HZB_SPD_WAVE_OPS";

inline bool supportsHzbSpdWaveOps(const DeviceCapabilities& capabilities)
{
    // Compiled as SPIR-V 1.6: 256 threads in X guarantees full subgroups for
    // every possible wave size. Each shuffle operates within a 16-lane block.
    return capabilities.computeSubgroupShuffle && capabilities.minSubgroupSize >= 16 &&
        capabilities.maxSubgroupSize <= 256 && capabilities.minSubgroupSize <= capabilities.maxSubgroupSize;
}

// Native bindless header is prepended by the RHI. See HZBSPD.slang.
struct HZBSPDUserPush {
    uint32_t depthImage = 0;
    uint32_t hzbBuffer = 0;
    uint32_t counterBuffer = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 0; // Includes full-resolution mip 0.
};
static_assert(sizeof(HZBSPDUserPush) == 24);

} // namespace metallic::render

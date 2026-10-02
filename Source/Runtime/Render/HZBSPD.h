#pragma once

#include "Runtime/Render/GAPI/RHI.h"

#include <cstdint>

namespace metallic::render {

inline constexpr const char* kHZBModule = "Features/GPUDriven/HZB";
inline constexpr const char* kHZBEntryPoint = "hzbMain";

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

} // namespace metallic::render

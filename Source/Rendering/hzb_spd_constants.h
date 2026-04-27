#ifndef HZB_SPD_CONSTANTS_H
#define HZB_SPD_CONSTANTS_H

#ifndef HZB_CONSTANTS_H
#include "hzb_constants.h"
#endif

#define SPD_MAX_MIPS 12u
#define SPD_THREADGROUP_SIZE 256u
#define SPD_TILE_SIZE 64u
#define HZB_SPD_SOURCE_TEXTURE_BINDING 0u
#define HZB_SPD_MIP_TEXTURE_BINDING_BASE 1u
#define HZB_SPD_MIP6_LEVEL 6u
#define HZB_SPD_MIP6_TEXTURE_BINDING 14u
#define HZB_SPD_ATOMIC_COUNTER_BINDING 15u

#ifdef __cplusplus

#include <algorithm>
#include <cstdint>

static constexpr uint32_t kSpdMaxMips = SPD_MAX_MIPS;
static constexpr uint32_t kSpdThreadgroupSize = SPD_THREADGROUP_SIZE;
static constexpr uint32_t kSpdTileSize = SPD_TILE_SIZE;
static constexpr uint32_t kHzbSpdSourceTextureBinding = HZB_SPD_SOURCE_TEXTURE_BINDING;
static constexpr uint32_t kHzbSpdMipTextureBindingBase = HZB_SPD_MIP_TEXTURE_BINDING_BASE;
static constexpr uint32_t kHzbSpdMip6Level = HZB_SPD_MIP6_LEVEL;
static constexpr uint32_t kHzbSpdMip6TextureBinding = HZB_SPD_MIP6_TEXTURE_BINDING;
static constexpr uint32_t kHzbSpdAtomicCounterBinding = HZB_SPD_ATOMIC_COUNTER_BINDING;

static_assert(kHzbSpdMip6TextureBinding == kHzbSpdMipTextureBindingBase + kHzbMaxLevels);
static_assert(kHzbSpdAtomicCounterBinding == kHzbSpdMip6TextureBinding + 1u);

struct SpdSetupInfo {
    uint32_t dispatchX = 0;
    uint32_t dispatchY = 0;
    uint32_t mipCount = 0;
    uint32_t numWorkGroups = 0;
};

inline SpdSetupInfo spdSetup(uint32_t width, uint32_t height) {
    SpdSetupInfo info;
    info.dispatchX = (width + kSpdTileSize - 1u) / kSpdTileSize;
    info.dispatchY = (height + kSpdTileSize - 1u) / kSpdTileSize;
    info.numWorkGroups = info.dispatchX * info.dispatchY;

    uint32_t maxDim = std::max(width, height);
    info.mipCount = 0;
    while (maxDim > 1u) {
        info.mipCount++;
        maxDim >>= 1;
    }
    info.mipCount = std::min(info.mipCount, kSpdMaxMips);
    return info;
}

#endif
#endif

#ifndef HZB_CONSTANTS_H
#define HZB_CONSTANTS_H

#define HZB_MAX_LEVELS 10u

#ifdef __cplusplus

#include <algorithm>
#include <cstdint>
#include <string>

#include "rhi_backend.h"

static constexpr uint32_t kHzbMaxLevels = HZB_MAX_LEVELS;

inline uint32_t hzbLevelDimension(uint32_t baseDimension, uint32_t level) {
    uint32_t dimension = std::max(baseDimension, 1u);
    for (uint32_t currentLevel = 0; currentLevel < level; ++currentLevel) {
        dimension = std::max(1u, (dimension + 1u) / 2u);
    }
    return dimension;
}

inline uint32_t computeHzbLevelCount(uint32_t width, uint32_t height) {
    uint32_t levels = 0;
    uint32_t currentWidth = std::max(width, 1u);
    uint32_t currentHeight = std::max(height, 1u);

    while (levels < kHzbMaxLevels) {
        ++levels;
        if (currentWidth == 1u && currentHeight == 1u) {
            break;
        }
        currentWidth = std::max(1u, (currentWidth + 1u) / 2u);
        currentHeight = std::max(1u, (currentHeight + 1u) / 2u);
    }

    return levels;
}

inline RhiTextureDesc makeHzbTextureDesc(uint32_t width, uint32_t height, uint32_t level) {
    return RhiTextureDesc::storageTexture(
        hzbLevelDimension(width, level),
        hzbLevelDimension(height, level),
        RhiFormat::R32Float);
}

inline std::string hzbHistoryResourceName(uint32_t level) {
    return "history.hzb.mip" + std::to_string(level);
}

#endif

#endif

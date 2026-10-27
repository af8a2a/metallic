#pragma once

// Shared CPU/Slang publication ABI. Entries use current lighting source slots;
// stable identities are independent of those slots and survive list reordering.
#ifdef __cplusplus
#include <cstdint>
namespace metallic::render {
#define VSM_PUBLIC
#else
#define VSM_PUBLIC public
#endif

VSM_PUBLIC struct VirtualShadowMapLightsHeader
{
    uint32_t version;
    uint32_t count;
    uint32_t capacity;
    uint32_t totalFaces;
    uint32_t admittedCount;
    uint32_t budgetDroppedCount;
    uint32_t disabledCount;
    uint32_t physicalPageBudget;
    uint32_t sceneIdentity[2];
    uint32_t frameIndex[2];
    uint32_t lightGeneration;
    uint32_t sourceSlotCount;
    uint32_t reserved[2]; // [0] bit 0: compact publication, one layer per light.
};

VSM_PUBLIC struct VirtualShadowMapLightEntry
{
    uint32_t source; // Celestial tag | Sun/Moon slot, or local GPUScene source slot.
    uint32_t kind; // 0 directional, 1 spot, 2 point.
    uint32_t baseLayer; // Entry index * 6, or entry index for compact publication.
    uint32_t faceCount;
    uint32_t stableId[2];
    uint32_t pageBudget; // Total across this light's faces.
    uint32_t status; // 0 shadow disabled, 1 admitted, 2 coarse budget rejected.
    float positionRange[4];
    uint32_t reserved[4];
};

#ifdef __cplusplus
static_assert(sizeof(VirtualShadowMapLightsHeader) == 64);
static_assert(sizeof(VirtualShadowMapLightEntry) == 64);
inline constexpr uint32_t kVirtualShadowMapLightsVersion = 1;
inline constexpr uint32_t kVirtualShadowMapLightDisabled = 0;
inline constexpr uint32_t kVirtualShadowMapLightAdmitted = 1;
inline constexpr uint32_t kVirtualShadowMapLightBudgetDropped = 2;
inline constexpr uint32_t kVirtualShadowMapLightFaceCapacity = 6;
inline constexpr uint32_t kVirtualShadowMapFaceStatisticsSize = 32 * sizeof(uint32_t);
} // namespace metallic::render
#endif
#undef VSM_PUBLIC

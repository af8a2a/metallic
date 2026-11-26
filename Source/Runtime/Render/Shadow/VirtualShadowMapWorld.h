#pragma once

#ifdef __cplusplus
#include <cstdint>
namespace metallic::render {
#define VSM_WORLD_PUBLIC
#else
#define VSM_WORLD_PUBLIC public
#endif

// Six canonical faces per source slot. Slots 0/1 are celestial, followed by
// the GPUScene local light slots. A zero pageSize denotes an unpublished face.
VSM_WORLD_PUBLIC struct VirtualShadowMapWorldFace
{
    uint32_t atlas, pages, projections, statistics;
    uint32_t physicalPagesX, clipmapCount, pageSize, physicalPagesY;
};

#ifdef __cplusplus
static_assert(sizeof(VirtualShadowMapWorldFace) == 32);
} // namespace metallic::render
#endif
#undef VSM_WORLD_PUBLIC

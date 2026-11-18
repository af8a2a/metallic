#pragma once

#ifdef __cplusplus
#include <cstdint>
namespace metallic::render {
#define VSM_POOL_INLINE inline
#else
#define uint32_t uint
#define VSM_POOL_INLINE
#endif

// One dispatch group owns the global admission and physical-page prefix work.
// Jobs contains the current frame's ShadowPush for every admitted face.
struct VirtualShadowMapPhysicalPoolPush
{
    uint32_t jobs;
    uint32_t metadata;
    uint32_t previousHistory; // UINT32_MAX starts an identity LRU order.
    uint32_t history;
    uint32_t faceCount;
    uint32_t capacity;
    uint32_t demandDriven;
    uint32_t frameIndex;
};

struct VirtualShadowMapPhysicalPoolPublishPush
{
    uint32_t jobs;
    uint32_t metadata;
    uint32_t publicationFaces; // uint entryIndex * 6 + canonicalFace per job.
    uint32_t lights;
    uint32_t budgetDiagnostics;
    uint32_t faceCount;
    uint32_t frameIndex;
    uint32_t reserved;
};

// Metadata consists of uint4 records:
//   [0]                   = {faceCount, capacity, jobs, frameIndex}
//   [1 + face]            = {grantFine, missingFine, missingBase, missingCoarse}
//   [1 + faceCount + p]   = {face + 1, virtualPage + 1, pinned, lastRequestedFrame}
//   [1 + faceCount + capacity + rank] = {physicalPage, lastRequestedFrame, 0, 0}
//   [1 + faceCount + capacity * 2] = {total requested fine before admission, 0, 0, 0}
//   [clearHeader]         = {compositeClearCount, staticClearCount, 0, 0}
//   [clearArg(layer, b)]  = {groupCountX, 1, 1, 0}
//   [clearList ...]       = tightly packed compositeSlots[capacity], staticSlots[capacity] (uint32_t)
// History contains capacity uint2(physicalPage, lastRequestedFrame) records in
// oldest-first order. All current owners are rebuilt after Prepare and before
// Physical, including unrequested cached pages; every admitted page is pinned.

VSM_POOL_INLINE uint32_t vsmSharedPoolClearBatches(uint32_t capacity)
{
    return (capacity + 65534u) / 65535u;
}

VSM_POOL_INLINE uint32_t vsmSharedPoolClearHeaderQuad(uint32_t faceCount, uint32_t capacity)
{
    return 2u + faceCount + capacity * 2u;
}

VSM_POOL_INLINE uint32_t vsmSharedPoolClearArgumentQuad(uint32_t faceCount, uint32_t capacity, uint32_t staticLayer, uint32_t batch)
{
    return vsmSharedPoolClearHeaderQuad(faceCount, capacity) + 1u + staticLayer * vsmSharedPoolClearBatches(capacity) + batch;
}

VSM_POOL_INLINE uint32_t vsmSharedPoolClearListQuad(uint32_t faceCount, uint32_t capacity)
{
    return vsmSharedPoolClearHeaderQuad(faceCount, capacity) + 1u + 2u * vsmSharedPoolClearBatches(capacity);
}

VSM_POOL_INLINE uint32_t vsmSharedPoolTotalMetadataQuads(uint32_t faceCount, uint32_t capacity)
{
    return vsmSharedPoolClearListQuad(faceCount, capacity) + (capacity * 2u + 3u) / 4u;
}

#undef VSM_POOL_INLINE
#ifdef __cplusplus
static_assert(sizeof(VirtualShadowMapPhysicalPoolPush) == 32);
static_assert(sizeof(VirtualShadowMapPhysicalPoolPublishPush) == 32);
} // namespace metallic::render
#else
#undef uint32_t
#endif

#ifndef METALLIC_SHADOW_RAY_CAPTURE_INTEROP
#define METALLIC_SHADOW_RAY_CAPTURE_INTEROP

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
namespace metallic::render {
#define SHADOW_CAPTURE_PUBLIC
#else
#define SHADOW_CAPTURE_PUBLIC public
#endif

// One record per image pixel. Candidate counters are shader-visible query
// events, not hardware BVH node or triangle-test counts.
SHADOW_CAPTURE_PUBLIC struct ShadowRayCaptureRecord
{
    SHADOW_CAPTURE_PUBLIC float originX, originY, originZ, tMin;
    SHADOW_CAPTURE_PUBLIC float directionX, directionY, directionZ, tMax;
    SHADOW_CAPTURE_PUBLIC uint32_t active, hit, proceedCalls, nonOpaqueCandidates;
    SHADOW_CAPTURE_PUBLIC float rawCommittedT, shadowDistance, penumbra, visibility;
    SHADOW_CAPTURE_PUBLIC uint32_t instanceId, clusterId, primitiveId, acceptedCandidates;
};

SHADOW_CAPTURE_PUBLIC struct ShadowRayCapturePush
{
    SHADOW_CAPTURE_PUBLIC uint32_t materialTextureCount, padding;
    SHADOW_CAPTURE_PUBLIC uint64_t outputAddress;
};

#ifdef __cplusplus
static_assert(sizeof(ShadowRayCaptureRecord) == 80);
static_assert(offsetof(ShadowRayCaptureRecord, active) == 32);
static_assert(offsetof(ShadowRayCaptureRecord, rawCommittedT) == 48);
static_assert(offsetof(ShadowRayCaptureRecord, instanceId) == 64);
static_assert(sizeof(ShadowRayCapturePush) == 16);
static_assert(offsetof(ShadowRayCapturePush, outputAddress) == 8);
} // namespace metallic::render
#endif
#undef SHADOW_CAPTURE_PUBLIC
#endif

#ifndef VISIBILITY_CONSTANTS_H
#define VISIBILITY_CONSTANTS_H

// Shared visibility-buffer bit packing constants for CPU + Slang shaders.
#define VISIBILITY_INSTANCE_BITS 11u
#define VISIBILITY_TRIANGLE_BITS 7u
#define VISIBILITY_INSTANCE_MASK ((1u << VISIBILITY_INSTANCE_BITS) - 1u)
#define VISIBILITY_TRIANGLE_MASK ((1u << VISIBILITY_TRIANGLE_BITS) - 1u)
#define VISIBILITY_MESHLET_SHIFT (VISIBILITY_TRIANGLE_BITS + VISIBILITY_INSTANCE_BITS)
#define VISIBILITY_MESHLET_MASK ((1u << (32u - VISIBILITY_MESHLET_SHIFT)) - 1u)

#ifdef __cplusplus
#include <cstdint>
static constexpr uint32_t kVisibilityInstanceBits = VISIBILITY_INSTANCE_BITS;
static constexpr uint32_t kVisibilityInstanceMask = VISIBILITY_INSTANCE_MASK;
static constexpr uint32_t kVisibilityTriangleBits = VISIBILITY_TRIANGLE_BITS;
static constexpr uint32_t kVisibilityTriangleMask = VISIBILITY_TRIANGLE_MASK;
static constexpr uint32_t kVisibilityMeshletShift = VISIBILITY_MESHLET_SHIFT;
static constexpr uint32_t kVisibilityMeshletMask = VISIBILITY_MESHLET_MASK;
#endif

#endif

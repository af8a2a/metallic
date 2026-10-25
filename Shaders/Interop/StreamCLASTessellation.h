#ifndef METALLIC_STREAM_CLAS_TESSELLATION_INTEROP
#define METALLIC_STREAM_CLAS_TESSELLATION_INTEROP

#ifdef __cplusplus
#include <cstdint>
namespace metallic::render {
#define TESS_PUBLIC
#else
#define TESS_PUBLIC public
#endif

#define METALLIC_TESS_VERTICES_PER_PATCH 64u
#define METALLIC_TESS_TRIANGLES_PER_PATCH 96u
#include "MeshletStreamBLAS.h"

TESS_PUBLIC struct StreamCLASTessellationLeaf
{
    TESS_PUBLIC uint32_t instanceIndex, pageIndex, clusterIndex, sourceTriangle;
    TESS_PUBLIC uint32_t patch0, patch1, patch2, rates;
    // RightCorner publishes its chosen guide corner in reserved0; geometry
    // decoders ignore it. Other template shapes and triangle builds keep zero.
    TESS_PUBLIC uint32_t vertexCount, triangleCount, reserved0, reserved1;
};

TESS_PUBLIC struct StreamCLASTessellationVertex
{
    TESS_PUBLIC float x, y, z, sourceU;
    TESS_PUBLIC float sourceV, reserved0, reserved1, reserved2;
};

// Published with the TLAS. The generated namespace begins after every asset
// page's fixed CLAS range; the host checks base + capacity without wrapping.
TESS_PUBLIC struct StreamCLASTessellationGeometry
{
    TESS_PUBLIC uint64_t leaves, vertices, triangles;
    TESS_PUBLIC uint32_t baseClusterId, maxPatches;
};

TESS_PUBLIC struct StreamCLASTessellationInstance
{
    TESS_PUBLIC uint32_t patchCount, firstPatch, buildIndex, status;
    TESS_PUBLIC uint32_t hash0, hash1, reserved0, reserved1;
};

// phases: 0 count (one 64-thread group/instance), 1 prefix/admit (one group),
// 2 emit leaves (one group/instance), 3 dice/build records (one group/patch),
// CLAS build, 4 BLAS records (64 threads/instance), BLAS build,
// base TLAS input generation, 5 publish (64 threads/instance), TLAS build.
// counts: patchCount, blasCount, rejectedInstances, invalidInstances,
// unboundedFallbackPatchesThisFrame, anyUnboundedFallbackSinceAllocation.
TESS_PUBLIC struct StreamCLASTessellationParameters
{
    TESS_PUBLIC StreamCLASTessellationGeometry geometry;
    TESS_PUBLIC uint64_t pages, pageTable, instances, primitives, groups;
    TESS_PUBLIC uint64_t representations, representationGroups, representationState, streamParams;
    TESS_PUBLIC uint64_t instanceStates, counts, clasBuildInfos, clasAddresses;
    TESS_PUBLIC uint64_t blasBuildInfos, blasAddresses, tlasInstances, blasHeader, history;
    TESS_PUBLIC uint32_t materialData, instanceCount, enabled, initializeHistory;
    TESS_PUBLIC float edgePixels;
    TESS_PUBLIC uint32_t maxFactor, maxSplitDepth, revision;
    TESS_PUBLIC uint32_t pageDescriptor, pageTableDescriptor;
    TESS_PUBLIC uint64_t templateAddresses;
    TESS_PUBLIC uint32_t buildMode, initializeBoundStatistics;
    TESS_PUBLIC uint64_t templateBoundsAddress;
    // 512 patterns per guide variant, followed by the optional bounded copies.
    // RightCorner has three unbounded variants (1536); other shapes have 512.
    TESS_PUBLIC uint32_t templateBoundsRequested, templateUnboundedCount;
};

TESS_PUBLIC struct StreamCLASTessellationPush
{
    TESS_PUBLIC uint64_t parameters;
    TESS_PUBLIC uint32_t phase, reserved;
};

#ifdef __cplusplus
static_assert(sizeof(StreamCLASTessellationLeaf) == 48);
static_assert(sizeof(StreamCLASTessellationVertex) == 32);
static_assert(sizeof(StreamCLASTessellationGeometry) == 32);
static_assert(sizeof(StreamCLASTessellationInstance) == 32);
static_assert(sizeof(StreamCLASTessellationParameters) == 248);
static_assert(sizeof(StreamCLASTessellationPush) == 16);
} // namespace metallic::render
#endif
#undef TESS_PUBLIC
#endif

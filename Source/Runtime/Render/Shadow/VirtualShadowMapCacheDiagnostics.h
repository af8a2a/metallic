#pragma once

// Independent 32-uint per-recording diagnostics. Read after GPU completion.
// CPU words 0..12, 15 and 28..29; GPU words 13..14, 16..27 and 30..31. Page counts include only the named scope.
// Raster counters 20..23 are available only when word 28 is nonzero.
// Triangle/primitive words 20..23 require collectGPUWork; otherwise zero.
#ifdef __cplusplus
#include <array>
#include <cstdint>
#include <string_view>
namespace metallic::render {
#endif

static const uint32_t kVSMCacheReasons = 0;
static const uint32_t kVSMCacheLayered = 1;
static const uint32_t kVSMCacheReused = 2;
static const uint32_t kVSMCacheStaticCasters = 3;
static const uint32_t kVSMCacheDynamicCasters = 4;
static const uint32_t kVSMCacheAddedCasters = 5;
static const uint32_t kVSMCacheRemovedCasters = 6;
static const uint32_t kVSMCacheChangedCasters = 7;
static const uint32_t kVSMCacheTransformChanges = 8;
static const uint32_t kVSMCacheMaterialChanges = 9;
static const uint32_t kVSMCachePromotedCasters = 10;
static const uint32_t kVSMCacheDirtyBounds = 11;
static const uint32_t kVSMCacheStaticDirtyBounds = 12;
static const uint32_t kVSMCacheDirtyRectangles = 13;
static const uint32_t kVSMCacheStaticDirtyRectangles = 14;
static const uint32_t kVSMCacheCancelledRecordings = 15;
static const uint32_t kVSMCacheStaticDirtyFinePages = 16;
static const uint32_t kVSMCacheStaticDirtyCoarsePages = 17;
static const uint32_t kVSMCacheCompositeDirtyFinePages = 18;
static const uint32_t kVSMCacheCompositeDirtyCoarsePages = 19;
static const uint32_t kVSMCacheStaticDecodedTriangles = 20;
static const uint32_t kVSMCacheDynamicDecodedTriangles = 21;
static const uint32_t kVSMCacheStaticOutputPrimitives = 22;
static const uint32_t kVSMCacheDynamicOutputPrimitives = 23;
static const uint32_t kVSMCacheRestoredPages = 24;
static const uint32_t kVSMCacheStaticReusedFinePages = 25;
static const uint32_t kVSMCacheStaticInvalidatedResidentPages = 26;
static const uint32_t kVSMCacheCompositeInvalidatedResidentPages = 27;
static const uint32_t kVSMCacheRasterWorkAvailable = 28;
static const uint32_t kVSMCacheLODChangedCasters = 29;
static const uint32_t kVSMCacheStaticHZBBuiltPages = 30; // HZB counters require collectGPUWork.
static const uint32_t kVSMCacheHZBSavedInvalidations = 31;
static const uint32_t kVirtualShadowMapCacheDiagnosticCount = 32;
static const uint32_t kVirtualShadowMapCacheDiagnosticSize = 128;

// Private metadata following the public diagnostics. Valid on demotion frames.
static const uint32_t kVSMCacheMergeRemap = 41;
static const uint32_t kVSMCacheMergeGeometry = 42;
static const uint32_t kVSMCacheMergeDrawOffset = 43; // Indirect argument triples, after draw and compact ranges.
static const uint32_t kVSMCacheMergeClustersBase = 44; // uint4 records in the page changes buffer.
static const uint32_t kVSMCacheMergeClusterCount = 45;
static const uint32_t kVSMCacheInstanceTable = 48;
static const uint32_t kVSMCacheInstanceCount = 49;

// Word 50 holds the GPU LOD context. These private words accompany the fine
// cluster/page raster path. Its three GPU counters require collectGPUWork.
static const uint32_t kVSMHybridDepth = 51;
static const uint32_t kVSMHybridPageCount = 52;
static const uint32_t kVSMHybridSubpixelBits = 53;
static const uint32_t kVSMHybridSoftwareTickets = 54;
static const uint32_t kVSMHybridHardwareTickets = 55;
static const uint32_t kVSMHybridCoveredSamples = 56;
static const uint32_t kVSMHybridDrawOffset = 57; // First composite HW indirect triple; static follows one fine batch range later.
static const uint32_t kVSMRasterHZB = 58;
static const uint32_t kVSMHZBDeferredTickets = 59;
static const uint32_t kVSMHZBRejectedTickets = 60;
static const uint32_t kVSMHZBRecoveredTickets = 61;
// Live queue counters for the composite and static layers, independent of diagnostics.
static const uint32_t kVSMHZBDeferredCount = 62;
static const uint32_t kVSMHZBPostCount = 64;
static const uint32_t kVSMHZBPageCount = 66;
static const uint32_t kVSMRasterHZBWordsPerSlot = 85; // 8x8, 4x4, 2x2, 1x1.

static const uint32_t kVSMHierarchyHZBNodes = 68;
static const uint32_t kVSMHierarchyHZBRootCount = 69;
static const uint32_t kVSMHierarchyHZBQueue = 70;
static const uint32_t kVSMHierarchyHZBDeferredCount = 71;
static const uint32_t kVSMHierarchyHZBMainTickets = 72;
static const uint32_t kVSMHierarchyHZBDeferredInstances = 73;
static const uint32_t kVSMHierarchyHZBDeferredNodes = 74;
static const uint32_t kVSMHierarchyHZBRejectedInstances = 75;
static const uint32_t kVSMHierarchyHZBRejectedNodes = 76;
static const uint32_t kVSMHierarchyHZBRecoveredInstances = 77;
static const uint32_t kVSMHierarchyHZBRecoveredNodes = 78;
static const uint32_t kVSMActiveFineViewMask = 79;
static const uint32_t kVSMActiveInitialBuildRows = 80;
static const uint32_t kVSMCachePreviousSlotAssignments = 81;
static const uint32_t kVSMCachePoolSerialLow = 82;
static const uint32_t kVSMCachePoolSerialHigh = 83;
static const uint32_t kVSMCacheRestoreInactive = 64u; // ShadowPush.cacheMode
static const uint32_t kVSMHierarchyHZBPostRaster = 16u;
static const uint32_t kVSMHierarchyHZBTreeStride = 15u;

static const uint32_t kVSMCacheReasonCacheDisabled = 1u << 0;
static const uint32_t kVSMCacheReasonCold = 1u << 1;
static const uint32_t kVSMCacheReasonDevice = 1u << 2;
static const uint32_t kVSMCacheReasonLight = 1u << 3;
static const uint32_t kVSMCacheReasonSettings = 1u << 4;
static const uint32_t kVSMCacheReasonScene = 1u << 5;
static const uint32_t kVSMCacheReasonStructure = 1u << 6;
static const uint32_t kVSMCacheReasonGeometryPublication = 1u << 7;
static const uint32_t kVSMCacheReasonTexturePublication = 1u << 8;
static const uint32_t kVSMCacheReasonTessellation = 1u << 9;
static const uint32_t kVSMCacheReasonProjectionRange = 1u << 10;
static const uint32_t kVSMCacheReasonForceRedraw = 1u << 11;
static const uint32_t kVSMCacheReasonLayerMode = 1u << 12;

#ifdef __cplusplus
inline constexpr std::array<std::string_view, 32> kVirtualShadowMapCacheDiagnosticNames{
    "reasons", "layered", "reused", "staticCasters", "dynamicCasters", "addedCasters", "removedCasters", "changedCasters", "transformChanges", "materialChanges", "promotedCasters", "dirtyBounds", "staticDirtyBounds", "dirtyRectangles", "staticDirtyRectangles", "cancelledRecordings", "staticDirtyFinePages", "staticDirtyCoarsePages", "compositeDirtyFinePages", "compositeDirtyCoarsePages", "staticDecodedTriangles", "dynamicDecodedTriangles", "staticOutputPrimitives", "dynamicOutputPrimitives", "restoredPages", "staticReusedFinePages", "staticInvalidatedResidentPages", "compositeInvalidatedResidentPages", "rasterWorkAvailable", "lodChangedCasters", "staticHZBBuiltPages", "hzbSavedInvalidations"
};
} // namespace metallic::render
#endif

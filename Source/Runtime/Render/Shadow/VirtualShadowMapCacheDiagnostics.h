#pragma once

// Independent 32-uint per-recording diagnostics. Read after GPU completion.
// CPU words 0..15 and 28..29; GPU words 16..27. Page counts include only the named scope.
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
static const uint32_t kVSMCacheReserved30 = 30;
static const uint32_t kVSMCacheReserved31 = 31;
static const uint32_t kVirtualShadowMapCacheDiagnosticCount = 32;
static const uint32_t kVirtualShadowMapCacheDiagnosticSize = 128;

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
    "reasons", "layered", "reused", "staticCasters", "dynamicCasters", "addedCasters", "removedCasters", "changedCasters", "transformChanges", "materialChanges", "promotedCasters", "dirtyBounds", "staticDirtyBounds", "dirtyRectangles", "staticDirtyRectangles", "cancelledRecordings", "staticDirtyFinePages", "staticDirtyCoarsePages", "compositeDirtyFinePages", "compositeDirtyCoarsePages", "staticDecodedTriangles", "dynamicDecodedTriangles", "staticOutputPrimitives", "dynamicOutputPrimitives", "restoredPages", "staticReusedFinePages", "staticInvalidatedResidentPages", "compositeInvalidatedResidentPages", "rasterWorkAvailable", "lodChangedCasters", "reserved30", "reserved31"
};
} // namespace metallic::render
#endif

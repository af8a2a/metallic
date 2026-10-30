#pragma once

// Workload diagnostics extend the private statistics buffer after the unchanged
// 32-word public statistics ABI. Each counter is a low/high uint32 pair so GPU
// collection does not require 64-bit atomics. Read only after GPU completion.
#ifdef __cplusplus
#include <array>
#include <cstdint>
#include <string_view>
namespace metallic::render {
#endif

static const uint32_t kVSMWorkMarkReceiverDispatches = 0;
static const uint32_t kVSMWorkMarkSamplePositions = 1;
static const uint32_t kVSMWorkMarkLevelTests = 2;
static const uint32_t kVSMWorkMarkPageRequests = 3;
static const uint32_t kVSMWorkBuildActiveTriangleSlots = 4;
static const uint32_t kVSMWorkBuildDecodedTriangles = 5;
static const uint32_t kVSMWorkBuildLevelTests = 6;
static const uint32_t kVSMWorkBuildPageTests = 7; // Actual fine triangle uint4 page-table reads.
static const uint32_t kVSMWorkRasterCoarseGroups = 8;
static const uint32_t kVSMWorkRasterFineGroups = 9;
static const uint32_t kVSMWorkRasterDecodedTriangles = 10;
static const uint32_t kVSMWorkRasterOutputPrimitives = 11;
static const uint32_t kVSMWorkRasterClearedPages = 12;
static const uint32_t kVSMWorkProjectionReceivers = 13;
static const uint32_t kVSMWorkProjectionRays = 14;
static const uint32_t kVSMWorkProjectionAtlasLoads = 15;
static const uint32_t kVirtualShadowMapWorkCounterCount = 16;
static const uint32_t kVirtualShadowMapWorkloadSize = 128;

// Optional build diagnostics follow the original 32 statistics words and
// 16 workload counters without changing either ABI. Values are low/high uint32
// pairs at byte offset 256. Disabled cluster culling leaves this segment zero.
static const uint32_t kVSMBuildDiagnosticClusterGroups = 0;
static const uint32_t kVSMBuildDiagnosticClusterBoundsValidGroups = 1;
static const uint32_t kVSMBuildDiagnosticClusterSurvivingGroups = 2;
static const uint32_t kVSMBuildDiagnosticClusterLevelTests = 3;
static const uint32_t kVSMBuildDiagnosticClusterPageTests = 4;
static const uint32_t kVSMBuildDiagnosticClusterPageHits = 5;
static const uint32_t kVSMBuildDiagnosticClusterNearPlaneKeeps = 6;
static const uint32_t kVSMBuildDiagnosticClusterOutsideLevels = 7;
static const uint32_t kVSMBuildDiagnosticClusterEmptyLevels = 8;
static const uint32_t kVSMBuildDiagnosticClusterPageScanHistogram = 9;
static const uint32_t kVSMBuildDiagnosticClusterBitsetLevelTests = 25;
static const uint32_t kVSMBuildDiagnosticClusterBitsetWordTests = 26;
static const uint32_t kVSMBuildDiagnosticClusterBitsetScanHistogram = 27;
static const uint32_t kVirtualShadowMapBuildDiagnosticHistogramCount = 16;
static const uint32_t kVSMBuildDiagnosticTriangleBitsetWordTests = 43;
static const uint32_t kVSMBuildDiagnosticTriangleBitsetPageHits = 44;
static const uint32_t kVirtualShadowMapBuildDiagnosticCounterCount = 45;
static const uint32_t kVirtualShadowMapBuildDiagnosticSize = 360;
static const uint32_t kVirtualShadowMapBuildDiagnosticWordOffset = 64;

// Groups enter only after enabled/range/nonzero-dirty guards. Each valid group
// tests every fine level; each level contributes exactly one hit/near/outside/
// empty outcome. Legacy page tests count actual page-table loads, including the
// first hit. Bitset mode instead counts actual uint-word loads; raw-word reuse
// across rows is not another load. The inactive scan mode's counters stay zero.
// Each level contributes one observation to its mode's histogram, including
// zero for near/outside exits: 0, 1, 2-3, ..., 8192-16383, 16384+.
// Version 3 appends exact-triangle bitset uint loads and set-bit page hits.
// These replace fine triangle page-table reads, not cluster scan counters.
// Page hits count every emitted candidate, including work-capacity overflow.
#ifdef __cplusplus
static_assert(kVirtualShadowMapBuildDiagnosticWordOffset == 32u + kVirtualShadowMapWorkCounterCount * 2u);
static_assert(kVSMBuildDiagnosticClusterBitsetLevelTests == kVSMBuildDiagnosticClusterPageScanHistogram +
    kVirtualShadowMapBuildDiagnosticHistogramCount);
static_assert(kVSMBuildDiagnosticTriangleBitsetWordTests == kVSMBuildDiagnosticClusterBitsetScanHistogram +
    kVirtualShadowMapBuildDiagnosticHistogramCount);
static_assert(kVirtualShadowMapBuildDiagnosticCounterCount == kVSMBuildDiagnosticTriangleBitsetPageHits + 1u);
static_assert(kVirtualShadowMapBuildDiagnosticSize == kVirtualShadowMapBuildDiagnosticCounterCount * sizeof(uint64_t));
inline constexpr std::array<std::string_view, kVirtualShadowMapWorkCounterCount> kVirtualShadowMapWorkCounterNames{
    "markReceiverDispatches", "markSamplePositions", "markLevelTests", "markPageRequests",
    "buildActiveTriangleSlots", "buildDecodedTriangles", "buildLevelTests", "buildPageTests",
    "rasterCoarseGroups", "rasterFineGroups", "rasterDecodedTriangles", "rasterOutputPrimitives",
    "rasterClearedPages", "projectionReceivers", "projectionRays", "projectionAtlasLoads"};
inline constexpr std::array<std::string_view, kVirtualShadowMapBuildDiagnosticCounterCount> kVirtualShadowMapBuildDiagnosticCounterNames{
    "clusterGroups", "clusterBoundsValidGroups", "clusterSurvivingGroups", "clusterLevelTests", "clusterPageTests",
    "clusterPageHits", "clusterNearPlaneKeeps", "clusterOutsideLevels", "clusterEmptyLevels",
    "clusterPageScan0", "clusterPageScan1", "clusterPageScan2To3", "clusterPageScan4To7",
    "clusterPageScan8To15", "clusterPageScan16To31", "clusterPageScan32To63", "clusterPageScan64To127",
    "clusterPageScan128To255", "clusterPageScan256To511", "clusterPageScan512To1023", "clusterPageScan1024To2047",
    "clusterPageScan2048To4095", "clusterPageScan4096To8191", "clusterPageScan8192To16383", "clusterPageScan16384Plus",
    "clusterBitsetLevelTests", "clusterBitsetWordTests",
    "clusterBitsetScan0", "clusterBitsetScan1", "clusterBitsetScan2To3", "clusterBitsetScan4To7",
    "clusterBitsetScan8To15", "clusterBitsetScan16To31", "clusterBitsetScan32To63", "clusterBitsetScan64To127",
    "clusterBitsetScan128To255", "clusterBitsetScan256To511", "clusterBitsetScan512To1023", "clusterBitsetScan1024To2047",
    "clusterBitsetScan2048To4095", "clusterBitsetScan4096To8191", "clusterBitsetScan8192To16383", "clusterBitsetScan16384Plus",
    "triangleBitsetWordTests", "triangleBitsetPageHits"};
} // namespace metallic::render
#endif

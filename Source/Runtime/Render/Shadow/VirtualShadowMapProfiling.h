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
static const uint32_t kVSMWorkBuildPageTests = 7;
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
static const uint32_t kVirtualShadowMapBuildDiagnosticHistogramCount = 16;
static const uint32_t kVirtualShadowMapBuildDiagnosticCounterCount = 25;
static const uint32_t kVirtualShadowMapBuildDiagnosticSize = 200;
static const uint32_t kVirtualShadowMapBuildDiagnosticWordOffset = 64;

// Groups enter only after enabled/range/nonzero-dirty guards. Each valid group
// tests every fine level; each level contributes exactly one hit/near/outside/
// empty outcome. Page tests count actual loads, including the first hit. One
// histogram observation records that level's scan length, including zero for
// near/outside exits: 0, 1, 2-3, ..., 8192-16383, 16384+.
#ifdef __cplusplus
static_assert(kVirtualShadowMapBuildDiagnosticWordOffset == 32u + kVirtualShadowMapWorkCounterCount * 2u);
static_assert(kVirtualShadowMapBuildDiagnosticCounterCount == kVSMBuildDiagnosticClusterPageScanHistogram +
    kVirtualShadowMapBuildDiagnosticHistogramCount);
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
    "clusterPageScan2048To4095", "clusterPageScan4096To8191", "clusterPageScan8192To16383", "clusterPageScan16384Plus"};
} // namespace metallic::render
#endif

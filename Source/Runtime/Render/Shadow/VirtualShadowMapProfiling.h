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

#ifdef __cplusplus
inline constexpr std::array<std::string_view, kVirtualShadowMapWorkCounterCount> kVirtualShadowMapWorkCounterNames{
    "markReceiverDispatches", "markSamplePositions", "markLevelTests", "markPageRequests",
    "buildActiveTriangleSlots", "buildDecodedTriangles", "buildLevelTests", "buildPageTests",
    "rasterCoarseGroups", "rasterFineGroups", "rasterDecodedTriangles", "rasterOutputPrimitives",
    "rasterClearedPages", "projectionReceivers", "projectionRays", "projectionAtlasLoads"};
} // namespace metallic::render
#endif

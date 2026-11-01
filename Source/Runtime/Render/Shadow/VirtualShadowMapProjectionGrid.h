#pragma once

// Shared CPU/Slang one-pass projection lookup. Slots 0/1 are Sun/Moon;
// slots 2 + source index correspond to the unchanged GPUScene light domain.
#ifdef __cplusplus
#include <array>
#include <cstdint>
#include <string_view>
namespace metallic::render {
#define VSM_GRID_PUBLIC
#else
#define VSM_GRID_PUBLIC public
#endif

VSM_GRID_PUBLIC struct VirtualShadowMapProjectionJobRange
{
    uint32_t first;
    uint32_t count;
};

VSM_GRID_PUBLIC struct VirtualShadowMapProjectionGrid
{
    uint32_t parameters, cells, lightIndices, candidates;
    uint32_t admittedLow, admittedHigh, diagnostics, localJobCount;
    VirtualShadowMapProjectionJobRange jobs[64];
};

static const uint32_t kVSMGridReceiverLookups = 0;
static const uint32_t kVSMGridFallbacks = 1;
static const uint32_t kVSMGridLocalCandidates = 2;
static const uint32_t kVSMGridLocalMemberships = 3;
static const uint32_t kVSMGridWaveLocalJobs = 4;
static const uint32_t kVSMGridSkippedWaveLocalJobs = 5;
static const uint32_t kVSMGridProjectedLaneJobs = 6;
static const uint32_t kVSMGridPossibleLaneJobs = 7;
static const uint32_t kVirtualShadowMapProjectionGridCounterCount = 8;
static const uint32_t kVirtualShadowMapProjectionGridDiagnosticSize = 64;

#ifdef __cplusplus
static_assert(sizeof(VirtualShadowMapProjectionJobRange) == 8);
static_assert(sizeof(VirtualShadowMapProjectionGrid) == 544);
inline constexpr std::array<std::string_view, kVirtualShadowMapProjectionGridCounterCount>
    kVirtualShadowMapProjectionGridCounterNames{
        "receiverLookups", "gridFallback", "localCandidates", "localMemberships",
        "waveLocalJobs", "skippedWaveLocalJobs", "projectedLaneJobs", "possibleLaneJobs"};
} // namespace metallic::render
#endif
#undef VSM_GRID_PUBLIC

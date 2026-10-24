#pragma once

#include <cstdint>

namespace metallic::render
{
// Raw device-address ABI of StreamBLASAllocator.slang. All configuration is
// immutable for the lifetime of an arena; only submitted GPU work mutates it.
// Phases 5/1/6 gather releases, retire them and prepare dirty builds; phase 2
// allocates/plans moves. Phase 8 retires staged move sources and plans their
// return; phase 3 publishes after that return.
// nodeCapacity >= 4 * ownerCount + 1 covers live sources and pending targets.
struct MeshletStreamBLASAllocatorParameters
{
    uint64_t state, gaps, allocations, owners, header, buildInfos;
    uint64_t buildOwners, sizes, sourceAddresses, moveSources, moveDestinations, moveCounts;
    uint64_t clusterReferenceAddress, arena;
    uint32_t ownerCount, alignment, storageBytes, relocateBudget;
    uint32_t nodeCapacity, relocateScanCount, reusePadding, mixedTilePadding;
    uint64_t relocationArena;
    uint64_t priorityOrder, importance;
    uint32_t referenceCapacity, buildCapacity;
    // GPU control words: exact sort-input change, sort-reused feedback.
    // Storage-budget changes require
    // an arena reset, because budget comparisons cannot resize the gap topology.
    uint64_t priorityCacheControl;
};
static_assert(sizeof(MeshletStreamBLASAllocatorParameters) == 184);

struct MeshletStreamBLASAllocatorGap
{
    uint32_t offset, size, previous, next;
    uint32_t binPrevious, binNext, free, unusedNext;
};
struct MeshletStreamBLASAllocatorAllocation
{
    uint32_t node, encodedBytes, rejectedEpoch, oldNode;
    uint32_t pendingNode, pendingOffset, pendingBytes, status;
    // Submitted GPU history only: phase 5 compares inputs before overwriting;
    // phase 6 records the rejection/wait decision for parallel idle restoration.
    uint32_t inputDynamic, inputSelectedClusterCount, prepareDecision, reserved;
};
static_assert(sizeof(MeshletStreamBLASAllocatorGap) == 32);
static_assert(sizeof(MeshletStreamBLASAllocatorAllocation) == 48);

struct MeshletStreamBLASAllocatorPush
{
    uint64_t parameters;
    uint32_t phase, reserved;
};
static_assert(sizeof(MeshletStreamBLASAllocatorPush) == 16);

inline constexpr uint32_t kBLASAllocatorStateWords = 144;
inline constexpr uint32_t kBLASAllocatorEpochWord = 99;
inline constexpr uint32_t kBLASAllocatorEncodedBytesWord = 102;
inline constexpr uint32_t kBLASAllocatorAllocatedBytesWord = 103;
inline constexpr uint32_t kBLASAllocatorLargestFreeWord = 104;
inline constexpr uint32_t kBLASAllocatorFreeBytesWord = 105;
inline constexpr uint32_t kBLASAllocatorSuppressedBuildsWord = 106;
inline constexpr uint32_t kBLASAllocatorRelocatedCountWord = 107;
inline constexpr uint32_t kBLASAllocatorRelocatedBytesWord = 108;
inline constexpr uint32_t kBLASAllocatorFailedAllocationsWord = 109;
inline constexpr uint32_t kBLASAllocatorPressureTargetWord = 110;
inline constexpr uint32_t kBLASAllocatorRequestedVisibleWord = 112;
inline constexpr uint32_t kBLASAllocatorRequestedOffscreenWord = 113;
inline constexpr uint32_t kBLASAllocatorRejectedVisibleWord = 114;
inline constexpr uint32_t kBLASAllocatorRejectedOffscreenWord = 115;
inline constexpr uint32_t kBLASAllocatorRejectedMaxErrorWord = 116;
inline constexpr uint32_t kBLASAllocatorRejectedVisibleMaxErrorWord = 117;
inline constexpr uint32_t kBLASAllocatorAdmittedOffscreenWord = 118;
inline constexpr uint32_t kBLASAllocatorPriorityEvictedWord = 119;
inline constexpr uint32_t kBLASAllocatorCompactionTargetWord = 110;
inline constexpr uint32_t kBLASAllocatorWaitingCountWord = 111;
inline constexpr uint32_t kBLASAllocatorPostBuildBudgetRejectedWord = 120;
inline constexpr uint32_t kBLASAllocatorPhysicalAllocationFailedWord = 121;
inline constexpr uint32_t kBLASAllocatorWaitingClaimBytesWord = 122;
inline constexpr uint32_t kBLASAllocatorMaxWaitingBytesWord = 123;
inline constexpr uint32_t kBLASAllocatorUnknownBuildCountWord = 124;
inline constexpr uint32_t kBLASAllocatorProjectedRetryDeferredCountWord = 125;
inline constexpr uint32_t kBLASAllocatorProjectedDirtyClaimBytesWord = 126;
inline constexpr uint32_t kBLASAllocatorSortReusedWord = 128;
inline constexpr uint32_t kBLASAllocatorAdmissionReusedWord = 129;
} // namespace metallic::render

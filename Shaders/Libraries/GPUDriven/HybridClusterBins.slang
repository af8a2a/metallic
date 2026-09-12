#ifndef HYBRID_CLUSTER_BINS_SLANG
#define HYBRID_CLUSTER_BINS_SLANG
#include "HybridRasterCommon.slang"

static const uint kHybridBinCount = 5u;
static const uint kHybridSoftwareBin = 4u;
static const uint kHybridBinHeaderWords = 16u;
static const uint kHybridDispatchWidth = 65535u;

// Header: five counters, capacity, width, height, maxPixels, reversedZ,
// subpixelBits, producer pixel descriptor, candidate count, reserved, overflow, reserved.
// Lists contain original record indices, never a reordered visibility namespace.
uint hybridFlatGroup(uint3 group)
{
    return group.x + group.y * kHybridDispatchWidth;
}

bool hybridBinRecord(StructuredBuffer<uint> bins, uint bin, uint item, out uint record)
{
    record = 0u;
    if (item >= min(bins[bin], bins[5])) { return false; }
    record = bins[kHybridBinHeaderWords + bin * bins[5] + item];
    return true;
}

static const uint kHybridBinBlockSize = 128u;
uint hybridCandidateBase(uint capacity) { return kHybridBinHeaderWords + kHybridBinCount * capacity; }
uint hybridRankBase(uint capacity) { return hybridCandidateBase(capacity) + 2u * capacity; }
uint hybridBlockBase(uint capacity) { return hybridRankBase(capacity) + capacity; }

// Classification writes one unique slot per input. Stable GPU compaction keeps
// equal-depth hardware primitives in the original submission order.
void hybridClassifyCluster(RWStructuredBuffer<uint> bins, uint bin, uint record, uint candidate)
{
    if (candidate >= bins[12]) { return; }
    uint offset = hybridCandidateBase(bins[5]) + candidate * 2u;
    bins[offset] = record;
    bins[offset + 1u] = bin;
}
#endif

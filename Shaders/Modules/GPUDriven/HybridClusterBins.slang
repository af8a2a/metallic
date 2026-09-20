#language slang 2026
implementing GPUDriven;

namespace Metallic.GPUDriven
{
public static const uint kHybridBinCount = 5u;
public static const uint kHybridSoftwareBin = 4u;
public static const uint kHybridBinHeaderWords = 16u;
public static const uint kHybridDispatchWidth = 65535u;

// Header: five counters, capacity, width, height, maxPixels, reversedZ,
// subpixelBits, producer pixel descriptor, candidate count, max group clusters,
// overflow, active group count (the last group fields are stream-only).
// Lists contain original record indices, never a reordered visibility namespace.
public uint hybridFlatGroup(uint3 group)
{
    return group.x + group.y * kHybridDispatchWidth;
}

public bool hybridBinRecord(StructuredBuffer<uint> bins, uint bin, uint item, out uint record)
{
    record = 0u;
    if (item >= min(bins[bin], bins[5])) { return false; }
    record = bins[kHybridBinHeaderWords + bin * bins[5] + item];
    return true;
}

public static const uint kHybridBinBlockSize = 128u;
public uint hybridCandidateBase(uint capacity) { return kHybridBinHeaderWords + kHybridBinCount * capacity; }
public uint hybridRankBase(uint capacity) { return hybridCandidateBase(capacity) + 2u * capacity; }
public uint hybridBlockBase(uint capacity) { return hybridRankBase(capacity) + capacity; }

// Stream culling aliases four of the five output-list words per capacity
// item with compact classification work (candidate, vertex base, counts,
// triangle base). Header word 0 temporarily counts this work. Stable binning
// consumes original candidate tags and overwrites both regions afterwards.
// During stream culling header 1/2 count metadata SW/forced HW shortcuts;
// those diagnostic counters are also overwritten by stable binning.
public uint hybridStreamClassifyBase(uint item) { return kHybridBinHeaderWords + 4u * item; }

// Per-active-group retry masks for stream early/late culling. At most capacity groups.
public uint hybridRetryBase(uint capacity) { return hybridBlockBase(capacity) + ((capacity + 127u) / 128u) * kHybridBinCount; }

// Classification writes one unique slot per input. Stable GPU compaction keeps
// equal-depth hardware primitives in the original submission order.
public void hybridClassifyCluster(RWStructuredBuffer<uint> bins, uint bin, uint record, uint candidate)
{
    if (candidate >= bins[12]) { return; }
    uint offset = hybridCandidateBase(bins[5]) + candidate * 2u;
    bins[offset] = record;
    bins[offset + 1u] = bin;
}
} // namespace Metallic.GPUDriven

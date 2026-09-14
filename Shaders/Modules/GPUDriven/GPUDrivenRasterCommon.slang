#language slang 2026
implementing GPUDriven;

namespace Metallic.GPUDriven
{
public static const uint kVisibilityTriangleBits = 7u;
public static const uint kVisibilityTriangleMask = (1u << kVisibilityTriangleBits) - 1u;
public static const uint kVisibilityRecordBits = 32u - kVisibilityTriangleBits;
// Encoded record zero is reserved for background.
public static const uint kVisibilityMaxRecordCount = 0xffffffffu >> kVisibilityTriangleBits;
public static const uint kVisibilityMaxRecordIndex = kVisibilityMaxRecordCount - 1u;
public static const uint kVisibleClusterDrawBucketMask = 0x0fu;
public static const uint kVisibleClusterSourceShift = 28u;
public static const uint kVisibleClusterSourceMask = 0x3u << kVisibleClusterSourceShift;
public static const uint kVisibleClusterSourceResident = 0u;
public static const uint kVisibleClusterSourceStreamPage = 1u;

public struct VisibleClusterRecord
{
    public uint clusterIndex;
    public uint instanceIndex;
    public uint dataIndex;
    public uint flags;
};

public uint visibleClusterDrawBucket(VisibleClusterRecord record)
{
    return record.flags & kVisibleClusterDrawBucketMask;
}

public uint visibleClusterSource(VisibleClusterRecord record)
{
    return (record.flags & kVisibleClusterSourceMask) >> kVisibleClusterSourceShift;
}

public bool visibilityRecordIndexValid(uint recordIndex, uint recordCapacity)
{
    return recordIndex < recordCapacity && recordIndex < kVisibilityMaxRecordCount;
}

public bool visibilityRecordRangeValid(uint recordBase, uint recordCapacity)
{
    return recordBase <= kVisibilityMaxRecordCount &&
        recordCapacity <= kVisibilityMaxRecordCount - recordBase;
}

public bool resolveVisibilityRecordIndex(
    uint localRecordIndex,
    uint recordBase,
    uint recordCapacity,
    out uint recordIndex)
{
    recordIndex = 0u;
    if (!visibilityRecordRangeValid(recordBase, recordCapacity) ||
        localRecordIndex >= recordCapacity) {
        return false;
    }
    recordIndex = recordBase + localRecordIndex;
    return visibilityRecordIndexValid(
        recordIndex,
        recordBase + recordCapacity);
}

public uint packVisibilityId(uint recordIndex, uint triangleIndex)
{
    return ((recordIndex + 1u) << kVisibilityTriangleBits) |
        (triangleIndex & kVisibilityTriangleMask);
}

public bool unpackVisibilityId(
    uint visibilityId,
    out uint recordIndex,
    out uint triangleIndex)
{
    uint encodedRecord = visibilityId >> kVisibilityTriangleBits;
    triangleIndex = visibilityId & kVisibilityTriangleMask;
    recordIndex = encodedRecord == 0u ? 0u : encodedRecord - 1u;
    return encodedRecord != 0u;
}
} // namespace Metallic.GPUDriven

static const uint kVisibilityTriangleBits = 7u;
static const uint kVisibilityTriangleMask = (1u << kVisibilityTriangleBits) - 1u;
static const uint kVisibilityRecordBits = 32u - kVisibilityTriangleBits;
// Encoded record zero is reserved for background.
static const uint kVisibilityMaxRecordCount = 0xffffffffu >> kVisibilityTriangleBits;
static const uint kVisibilityMaxRecordIndex = kVisibilityMaxRecordCount - 1u;
static const uint kVisibleClusterDrawBucketMask = 0x0fu;
static const uint kVisibleClusterSourceShift = 28u;
static const uint kVisibleClusterSourceMask = 0x3u << kVisibleClusterSourceShift;
static const uint kVisibleClusterSourceResident = 0u;
static const uint kVisibleClusterSourceStreamPage = 1u;

struct VisibleClusterRecord
{
    uint clusterIndex;
    uint instanceIndex;
    uint dataIndex;
    uint flags;
};

uint visibleClusterDrawBucket(VisibleClusterRecord record)
{
    return record.flags & kVisibleClusterDrawBucketMask;
}

uint visibleClusterSource(VisibleClusterRecord record)
{
    return (record.flags & kVisibleClusterSourceMask) >> kVisibleClusterSourceShift;
}

bool visibilityRecordIndexValid(uint recordIndex, uint recordCapacity)
{
    return recordIndex < recordCapacity && recordIndex < kVisibilityMaxRecordCount;
}

bool visibilityRecordRangeValid(uint recordBase, uint recordCapacity)
{
    return recordBase <= kVisibilityMaxRecordCount &&
        recordCapacity <= kVisibilityMaxRecordCount - recordBase;
}

bool resolveVisibilityRecordIndex(
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

uint packVisibilityId(uint recordIndex, uint triangleIndex)
{
    return ((recordIndex + 1u) << kVisibilityTriangleBits) |
        (triangleIndex & kVisibilityTriangleMask);
}

bool unpackVisibilityId(
    uint visibilityId,
    out uint recordIndex,
    out uint triangleIndex)
{
    uint encodedRecord = visibilityId >> kVisibilityTriangleBits;
    triangleIndex = visibilityId & kVisibilityTriangleMask;
    recordIndex = encodedRecord == 0u ? 0u : encodedRecord - 1u;
    return encodedRecord != 0u;
}

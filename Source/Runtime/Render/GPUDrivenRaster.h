#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace metallic::render {

inline constexpr uint32_t kVisibilityTriangleBits = 7u;
inline constexpr uint32_t kVisibilityTriangleMask =
    (1u << kVisibilityTriangleBits) - 1u;
inline constexpr uint32_t kVisibilityRecordBits = 32u - kVisibilityTriangleBits;
// Encoded record zero is reserved for background, so an N-bit encoded-record
// field can address only (2^N - 1) records.
inline constexpr uint32_t kVisibilityMaxRecordCount =
    0xffffffffu >> kVisibilityTriangleBits;
inline constexpr uint32_t kVisibilityMaxRecordIndex =
    kVisibilityMaxRecordCount - 1u;
inline constexpr uint32_t kVisibleClusterSourceShift = 28u;
inline constexpr uint32_t kVisibleClusterSourceMask = 0x3u << kVisibleClusterSourceShift;
inline constexpr uint32_t kVisibleClusterDrawBucketMask = 0x0fu;

constexpr bool visibilityRecordCapacityFitsId(uint64_t recordCapacity)
{
    return recordCapacity <= kVisibilityMaxRecordCount;
}

constexpr bool visibilityRecordRangeFitsId(
    uint64_t recordBase,
    uint64_t recordCapacity)
{
    return recordBase <= kVisibilityMaxRecordCount &&
        recordCapacity <= kVisibilityMaxRecordCount - recordBase;
}

static_assert(kVisibilityTriangleBits == 7u);
static_assert(kVisibilityTriangleMask == 0x7fu);
static_assert(kVisibilityRecordBits == 25u);
static_assert(kVisibilityMaxRecordCount == 0x01ffffffu);
static_assert(kVisibilityMaxRecordIndex == 0x01fffffeu);
static_assert(
    (((kVisibilityMaxRecordIndex + 1u) << kVisibilityTriangleBits) |
        kVisibilityTriangleMask) == 0xffffffffu);
static_assert(
    ((kVisibilityMaxRecordCount + 1u) << kVisibilityTriangleBits) == 0u);
static_assert(visibilityRecordCapacityFitsId(kVisibilityMaxRecordCount));
static_assert(!visibilityRecordCapacityFitsId(
    static_cast<uint64_t>(kVisibilityMaxRecordCount) + 1u));
static_assert(visibilityRecordRangeFitsId(0u, kVisibilityMaxRecordCount));
static_assert(visibilityRecordRangeFitsId(kVisibilityMaxRecordIndex, 1u));
static_assert(!visibilityRecordRangeFitsId(kVisibilityMaxRecordCount, 1u));
static_assert(!visibilityRecordRangeFitsId(
    static_cast<uint64_t>(kVisibilityMaxRecordCount) + 1u,
    0u));

enum class VisibleClusterSource : uint32_t {
    Resident = 0u,
    StreamPage = 1u,
};

constexpr uint32_t visibleClusterFlags(
    VisibleClusterSource source,
    uint32_t drawBucket = 0u,
    uint32_t flags = 0u)
{
    return (static_cast<uint32_t>(source) << kVisibleClusterSourceShift) |
        (drawBucket & kVisibleClusterDrawBucketMask) |
        (flags & ~(kVisibleClusterSourceMask | kVisibleClusterDrawBucketMask));
}

constexpr VisibleClusterSource visibleClusterSource(uint32_t flags)
{
    return static_cast<VisibleClusterSource>(
        (flags & kVisibleClusterSourceMask) >> kVisibleClusterSourceShift);
}

// Stable visibility-buffer indirection shared by resident GPUScene meshes and
// streamed page clusters. dataIndex addresses geometry for Resident records
// and the active-group table for StreamPage records.
struct alignas(16) VisibleClusterRecord {
    uint32_t clusterIndex = 0;
    uint32_t instanceIndex = 0;
    uint32_t dataIndex = 0;
    uint32_t flags = 0;
};

static_assert(sizeof(VisibleClusterRecord) == 16);
static_assert(alignof(VisibleClusterRecord) == 16);
static_assert(offsetof(VisibleClusterRecord, clusterIndex) == 0);
static_assert(offsetof(VisibleClusterRecord, instanceIndex) == 4);
static_assert(offsetof(VisibleClusterRecord, dataIndex) == 8);
static_assert(offsetof(VisibleClusterRecord, flags) == 12);
static_assert(std::is_trivially_copyable_v<VisibleClusterRecord>);

} // namespace metallic::render

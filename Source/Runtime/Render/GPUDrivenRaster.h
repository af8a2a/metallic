#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace metallic::render {

// Fullscreen debug consumer only; descriptor indices use its own bindless heap.
struct VisibilityBufferCompositeUserPush {
    uint32_t paramsBuffer = 0;
    uint32_t visibilityImage = 0;
    uint32_t depthImage = 0;
    uint32_t residentRecords = 0;
    uint32_t meshletBuffer = 0;
    uint32_t residentRecordCapacity = 0;
    uint32_t streamRecords = UINT32_MAX;
    uint32_t streamGroups = UINT32_MAX;
    uint32_t shadedColors = 0;
};
static_assert(sizeof(VisibilityBufferCompositeUserPush) == 36);

// CPU-authored metadata accompanying visibility/depth. Consumers reconstruct
// from the actual raster camera, including when the culling camera is frozen.
struct alignas(16) VisibilityBufferFrameInfo {
    float eye[4] = {};
    float center[4] = {};
    float upProjection[4] = {};
    float viewport[4] = {};
    float clipOrtho[4] = {};
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t residentRecordCount = 0;
    uint32_t hasStreamGeometry = 0;
    uint64_t sceneIdentity = 0;
    uint64_t reserved = 0;
    uint32_t lightGridViewIndex = 0;
    uint32_t lightGridViewGeneration = 0;
    uint32_t lightGridFrameSlot = 0;
    uint32_t temporalJitter = 0;
    float jitter[2] = {};
    uint64_t frameIndex = 0;
    // Producer settings (LOD, displacement, culling, etc.) can change the
    // primary surface without changing the camera or the source scene.
    uint64_t rasterSettingsRevision = 0;
};
static_assert(sizeof(VisibilityBufferFrameInfo) == 160);
// VisibilityBufferDeferred.slang reads these words from the shared frame buffer.
static_assert(offsetof(VisibilityBufferFrameInfo, residentRecordCount) == 88);
static_assert(offsetof(VisibilityBufferFrameInfo, hasStreamGeometry) == 92);
static_assert(offsetof(VisibilityBufferFrameInfo, reserved) == 104);

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

// Stream records keep their sparse visibility ID. Instance identity comes from
// the active group; resident records retain the general 16-byte layout.
struct CompactStreamVisibleRecord {
    uint32_t packed = 0; // cluster[0:4], group+1[5:29], raster flags[30:31]
};
static_assert(sizeof(CompactStreamVisibleRecord) == 4);
static_assert(kVisibilityMaxRecordCount == 0x01ffffffu);
inline constexpr CompactStreamVisibleRecord packStreamVisibleRecord(uint32_t group, uint32_t cluster, uint32_t flags)
{
    return {((group + 1u) << 5u) | (cluster & 31u) | ((flags & 3u) << 30u)};
}
inline constexpr VisibleClusterRecord unpackStreamVisibleRecord(CompactStreamVisibleRecord value, uint32_t instance = UINT32_MAX)
{
    const uint32_t group = (value.packed >> 5u) & 0x01ffffffu;
    return {value.packed & 31u, instance, group - 1u,
        group ? ((1u << 28u) | (value.packed >> 30u)) : 0u};
}
static_assert(packStreamVisibleRecord(kVisibilityMaxRecordCount - 1u, 31u, 3u).packed == UINT32_MAX);
static_assert(unpackStreamVisibleRecord({UINT32_MAX}).dataIndex == kVisibilityMaxRecordCount - 1u);
static_assert(unpackStreamVisibleRecord({UINT32_MAX}).clusterIndex == 31u);
static_assert(unpackStreamVisibleRecord({}).dataIndex == UINT32_MAX);



} // namespace metallic::render

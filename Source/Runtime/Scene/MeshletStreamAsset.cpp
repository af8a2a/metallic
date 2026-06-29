#include "Runtime/Scene/MeshletStreamAsset.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <system_error>
#include <type_traits>
#include <unordered_map>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace metallic::scene {
namespace {

constexpr std::array<char, 8> kMeshletStreamMagic{'M', 'T', 'L', 'M', 'S', 'T', 'R', 'M'};
constexpr uint32_t kMeshletStreamVersion = 2;
constexpr uint32_t kMeshletStreamEndian = 0x01020304;
constexpr uint32_t kPayloadMagic = 0x4d535047u; // "GSPM"
constexpr uint32_t kPayloadVersion = 2;
constexpr uint64_t kFileAlignment = 16;
constexpr uint64_t kPageSlotAlignment = 256;
constexpr uint32_t kMeshletClusterMaxVertices = 128;
constexpr uint32_t kMeshletClusterMinTriangles = 32;
constexpr uint32_t kMeshletClusterMaxTriangles = 128;
constexpr uint32_t kMeshletLodGroupSize = 32;
constexpr float kMeshletClusterFillWeight = 0.5f;
constexpr float kMeshletLodErrorMergePrevious = 1.5f;
constexpr float kMeshletLodErrorMergeAdditive = 0.0f;

struct MeshletStreamFileHeader {
    char magic[8] = {};
    uint32_t version = 0;
    uint32_t endian = 0;
    uint64_t fileSize = 0;
    uint64_t sourceFileSize = 0;
    int64_t sourceWriteTime = 0;
    uint32_t primitiveCount = 0;
    uint32_t instanceCount = 0;
    uint32_t geometryCount = 0;
    uint32_t reserved0 = 0;
    uint32_t lodLevelCount = 0;
    uint32_t pageCount = 0;
    uint32_t maxPagePayloadBytes = 0;
    uint32_t pagePayloadAlignment = 0;
    uint32_t maxVertices = 0;
    uint32_t minTriangles = 0;
    uint32_t maxTriangles = 0;
    uint32_t lodGroupSize = 0;
    uint32_t reserved = 0;
    uint64_t primitiveOffset = 0;
    uint64_t instanceOffset = 0;
    uint64_t geometryOffset = 0;
    uint64_t lodLevelOffset = 0;
    uint64_t pageInfoOffset = 0;
    uint64_t pageOffsetTableOffset = 0;
    float fillWeight = 0.0f;
    float lodErrorMergePrevious = 0.0f;
    float lodErrorMergeAdditive = 0.0f;
    uint32_t reserved1 = 0;
};

static_assert(std::is_trivially_copyable_v<MeshletStreamFileHeader>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPrimitiveInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamInstanceInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamGeometryInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamLodLevelInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPageInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPayloadHeader>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPayloadCluster>);
static_assert(sizeof(MeshletStreamPayloadHeader) == 96);
static_assert(sizeof(MeshletStreamPayloadCluster) == 32);

bool meshletStreamBuildParamsMatch(const MeshletStreamFileHeader& header)
{
    return header.pagePayloadAlignment == kPageSlotAlignment &&
        header.maxVertices == kMeshletClusterMaxVertices &&
        header.minTriangles == kMeshletClusterMinTriangles &&
        header.maxTriangles == kMeshletClusterMaxTriangles &&
        header.lodGroupSize == kMeshletLodGroupSize &&
        header.fillWeight == kMeshletClusterFillWeight &&
        header.lodErrorMergePrevious == kMeshletLodErrorMergePrevious &&
        header.lodErrorMergeAdditive == kMeshletLodErrorMergeAdditive;
}

uint64_t alignUp(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

uint64_t sourceFileSizeFor(const std::filesystem::path& path)
{
    std::error_code error;
    const uint64_t size = std::filesystem::file_size(path, error);
    return error ? 0 : size;
}

int64_t sourceWriteTimeFor(const std::filesystem::path& path)
{
    std::error_code error;
    const auto writeTime = std::filesystem::last_write_time(path, error);
    if (error) {
        return 0;
    }
    return static_cast<int64_t>(writeTime.time_since_epoch().count());
}

MeshletStreamBounds makeStreamBounds(const Bounds& bounds)
{
    MeshletStreamBounds result;
    if (bounds.valid) {
        result.min[0] = bounds.min.x;
        result.min[1] = bounds.min.y;
        result.min[2] = bounds.min.z;
        result.max[0] = bounds.max.x;
        result.max[1] = bounds.max.y;
        result.max[2] = bounds.max.z;
        result.valid = 1;
    }
    return result;
}

Bounds makeBounds(const MeshletStreamBounds& bounds)
{
    Bounds result;
    if (bounds.valid != 0) {
        result.min = float3(bounds.min[0], bounds.min[1], bounds.min[2]);
        result.max = float3(bounds.max[0], bounds.max[1], bounds.max[2]);
        result.valid = true;
    }
    return result;
}

template <typename T>
bool rangeWithin(uint64_t fileSize, uint64_t offset, uint64_t count)
{
    if (count == 0) {
        return true;
    }
    if (offset > fileSize) {
        return false;
    }
    const uint64_t byteSize = count * sizeof(T);
    if (count != 0 && byteSize / sizeof(T) != count) {
        return false;
    }
    return byteSize <= fileSize - offset;
}

template <typename T>
std::span<const T> makeSpan(const uint8_t* base, uint64_t offset, uint64_t count)
{
    if (count == 0) {
        return {};
    }
    return std::span<const T>(reinterpret_cast<const T*>(base + offset), static_cast<size_t>(count));
}

bool byteRangeWithin(uint64_t byteSize, uint64_t offset, uint64_t rangeSize)
{
    return offset <= byteSize && rangeSize <= byteSize - offset;
}

bool meshletStreamCompressionSupported(uint32_t compressionMode)
{
    return compressionMode == static_cast<uint32_t>(MeshletStreamPayloadCompression::None) ||
        compressionMode == static_cast<uint32_t>(MeshletStreamPayloadCompression::ByteRle);
}

bool encodeByteRle(std::span<const uint8_t> source, std::vector<uint8_t>& outBytes)
{
    outBytes.clear();
    outBytes.reserve(source.size());

    size_t offset = 0;
    while (offset < source.size()) {
        size_t runLength = 1;
        while (offset + runLength < source.size() &&
            source[offset + runLength] == source[offset] &&
            runLength < 128) {
            ++runLength;
        }

        if (runLength >= 3) {
            outBytes.push_back(static_cast<uint8_t>(0x80u | (runLength - 1u)));
            outBytes.push_back(source[offset]);
            offset += runLength;
            continue;
        }

        const size_t literalStart = offset;
        size_t literalEnd = offset;
        while (literalEnd < source.size() && literalEnd - literalStart < 128) {
            runLength = 1;
            while (literalEnd + runLength < source.size() &&
                source[literalEnd + runLength] == source[literalEnd] &&
                runLength < 128) {
                ++runLength;
            }
            if (runLength >= 3) {
                break;
            }
            ++literalEnd;
        }

        const size_t literalLength = literalEnd - literalStart;
        if (literalLength == 0 || literalLength > 128) {
            return false;
        }
        outBytes.push_back(static_cast<uint8_t>(literalLength - 1u));
        outBytes.insert(
            outBytes.end(),
            source.data() + literalStart,
            source.data() + literalStart + literalLength);
        offset = literalEnd;
    }
    return true;
}

bool decodeByteRle(std::span<const uint8_t> source, std::span<uint8_t> destination)
{
    size_t sourceOffset = 0;
    size_t destinationOffset = 0;
    while (sourceOffset < source.size()) {
        const uint8_t command = source[sourceOffset++];
        const size_t count = static_cast<size_t>(command & 0x7fu) + 1u;
        if ((command & 0x80u) != 0u) {
            if (sourceOffset >= source.size() || count > destination.size() - destinationOffset) {
                return false;
            }
            std::memset(destination.data() + destinationOffset, source[sourceOffset++], count);
            destinationOffset += count;
            continue;
        }

        if (count > source.size() - sourceOffset || count > destination.size() - destinationOffset) {
            return false;
        }
        std::memcpy(destination.data() + destinationOffset, source.data() + sourceOffset, count);
        sourceOffset += count;
        destinationOffset += count;
    }
    return destinationOffset == destination.size();
}

bool encodePayloadForStorage(
    const std::vector<uint8_t>& devicePayload,
    MeshletStreamPayloadCompression compressionMode,
    std::vector<uint8_t>& outStoredPayload,
    MeshletStreamPageInfo& pageInfo,
    std::string& reason)
{
    if (devicePayload.size() < sizeof(MeshletStreamPayloadHeader) ||
        devicePayload.size() > std::numeric_limits<uint32_t>::max()) {
        reason = "meshlet page payload is too small or too large for storage";
        return false;
    }

    if (compressionMode == MeshletStreamPayloadCompression::None) {
        outStoredPayload = devicePayload;
    } else if (compressionMode == MeshletStreamPayloadCompression::ByteRle) {
        std::vector<uint8_t> encodedBody;
        const std::span<const uint8_t> body(
            devicePayload.data() + sizeof(MeshletStreamPayloadHeader),
            devicePayload.size() - sizeof(MeshletStreamPayloadHeader));
        if (!encodeByteRle(body, encodedBody)) {
            reason = "meshlet page ByteRle compression failed";
            return false;
        }

        outStoredPayload.assign(
            devicePayload.begin(),
            devicePayload.begin() + static_cast<std::ptrdiff_t>(sizeof(MeshletStreamPayloadHeader)));
        outStoredPayload.insert(outStoredPayload.end(), encodedBody.begin(), encodedBody.end());
    } else {
        reason = "meshlet page compression mode is unsupported";
        return false;
    }

    if (outStoredPayload.size() > std::numeric_limits<uint32_t>::max()) {
        reason = "meshlet page stored payload exceeds 32-bit shader offsets";
        return false;
    }

    auto* storedHeader = reinterpret_cast<MeshletStreamPayloadHeader*>(outStoredPayload.data());
    storedHeader->payloadByteSize = static_cast<uint32_t>(outStoredPayload.size());
    storedHeader->uncompressedPayloadByteSize = static_cast<uint32_t>(devicePayload.size());
    storedHeader->compressionMode = static_cast<uint32_t>(compressionMode);

    pageInfo.payloadSize = outStoredPayload.size();
    pageInfo.uncompressedSize = devicePayload.size();
    pageInfo.compressionMode = static_cast<uint32_t>(compressionMode);
    return true;
}

bool validatePayloadHeader(
    const uint8_t* base,
    uint64_t fileSize,
    const MeshletStreamPageInfo& page,
    uint32_t pageIndex,
    std::string& reason)
{
    if (page.payloadOffset % kFileAlignment != 0 ||
        page.payloadSize == 0 ||
        page.payloadSize > std::numeric_limits<uint32_t>::max() ||
        !byteRangeWithin(fileSize, page.payloadOffset, page.payloadSize) ||
        page.payloadSize < sizeof(MeshletStreamPayloadHeader)) {
        reason = "streamasset page payload metadata is invalid";
        return false;
    }

    MeshletStreamPayloadHeader payloadHeader;
    std::memcpy(&payloadHeader, base + page.payloadOffset, sizeof(payloadHeader));
    if (payloadHeader.magic != kPayloadMagic ||
        payloadHeader.version != kPayloadVersion ||
        payloadHeader.clusterCount != page.clusterCount ||
        payloadHeader.vertexCount != page.vertexCount ||
        payloadHeader.triangleIndexCount != page.triangleIndexCount ||
        payloadHeader.primitiveIndex != page.primitiveIndex ||
        payloadHeader.materialIndex != page.materialIndex ||
        payloadHeader.lodLevel != page.lodLevel ||
        payloadHeader.lodGroupIndex != page.lodGroupIndex ||
        payloadHeader.payloadByteSize != page.payloadSize ||
        payloadHeader.uncompressedPayloadByteSize != page.uncompressedSize ||
        payloadHeader.attributeFlags != page.attributeFlags ||
        payloadHeader.compressionMode != page.compressionMode) {
        reason = "streamasset page payload header does not match page directory";
        return false;
    }
    if (!meshletStreamCompressionSupported(page.compressionMode)) {
        reason = "streamasset page compression mode is unsupported by the v2 runtime reader";
        return false;
    }
    if (page.compressionMode == static_cast<uint32_t>(MeshletStreamPayloadCompression::None) &&
        page.payloadSize != page.uncompressedSize) {
        reason = "streamasset uncompressed page has mismatched stored and device sizes";
        return false;
    }
    if (page.compressionMode != static_cast<uint32_t>(MeshletStreamPayloadCompression::None) &&
        page.uncompressedSize < sizeof(MeshletStreamPayloadHeader)) {
        reason = "streamasset compressed page device payload metadata is invalid";
        return false;
    }

    const uint64_t payloadSize = page.uncompressedSize;
    const uint64_t clusterBytes =
        static_cast<uint64_t>(payloadHeader.clusterCount) * sizeof(MeshletStreamPayloadCluster);
    const uint64_t positionBytes = static_cast<uint64_t>(payloadHeader.vertexCount) * sizeof(float) * 4u;
    const uint64_t normalBytes =
        (payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeNormal) != 0u
            ? static_cast<uint64_t>(payloadHeader.vertexCount) * sizeof(float) * 4u
            : 0u;
    const uint64_t texcoord0Bytes =
        (payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeTexcoord0) != 0u
            ? static_cast<uint64_t>(payloadHeader.vertexCount) * sizeof(float) * 2u
            : 0u;
    const uint64_t materialBytes =
        (payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeMaterial) != 0u
            ? static_cast<uint64_t>(payloadHeader.materialCount) * sizeof(uint32_t)
            : 0u;
    const uint64_t triangleBytes = payloadHeader.triangleIndexCount;
    if (payloadHeader.clusterOffsetBytes % 16u != 0 ||
        payloadHeader.positionOffsetBytes % 16u != 0 ||
        ((payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeNormal) != 0u &&
            payloadHeader.normalOffsetBytes % 16u != 0) ||
        ((payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeTexcoord0) != 0u &&
            payloadHeader.texcoord0OffsetBytes % 8u != 0) ||
        ((payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeMaterial) != 0u &&
            payloadHeader.materialOffsetBytes % 4u != 0) ||
        payloadHeader.triangleOffsetBytes % 4u != 0 ||
        !byteRangeWithin(payloadSize, payloadHeader.clusterOffsetBytes, clusterBytes) ||
        !byteRangeWithin(payloadSize, payloadHeader.positionOffsetBytes, positionBytes) ||
        !byteRangeWithin(payloadSize, payloadHeader.normalOffsetBytes, normalBytes) ||
        !byteRangeWithin(payloadSize, payloadHeader.texcoord0OffsetBytes, texcoord0Bytes) ||
        !byteRangeWithin(payloadSize, payloadHeader.materialOffsetBytes, materialBytes) ||
        !byteRangeWithin(payloadSize, payloadHeader.triangleOffsetBytes, triangleBytes)) {
        reason = "streamasset page payload section ranges are invalid";
        return false;
    }

    if (payloadHeader.clusterCount == 0 ||
        payloadHeader.vertexCount == 0 ||
        payloadHeader.triangleIndexCount == 0 ||
        payloadHeader.triangleIndexCount % 3u != 0) {
        reason = "streamasset page payload counts are invalid";
        return false;
    }
    if ((payloadHeader.attributeFlags & kMeshletStreamPayloadAttributePosition) == 0u ||
        payloadHeader.positionFormat != static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x4) ||
        ((payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeNormal) != 0u &&
            payloadHeader.normalFormat != static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x4)) ||
        ((payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeTexcoord0) != 0u &&
            payloadHeader.texcoord0Format != static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x2)) ||
        ((payloadHeader.attributeFlags & kMeshletStreamPayloadAttributeMaterial) != 0u &&
            (payloadHeader.materialFormat != static_cast<uint32_t>(MeshletStreamPayloadFormat::Uint32) ||
                payloadHeader.materialCount != payloadHeader.clusterCount))) {
        reason = "streamasset page payload attribute metadata is invalid";
        return false;
    }

    (void)pageIndex;
    return true;
}

bool writeZeros(std::ostream& stream, uint64_t byteCount)
{
    static constexpr std::array<char, 16> kZeros{};
    while (byteCount > 0) {
        const uint64_t chunk = std::min<uint64_t>(byteCount, kZeros.size());
        stream.write(kZeros.data(), static_cast<std::streamsize>(chunk));
        if (!stream) {
            return false;
        }
        byteCount -= chunk;
    }
    return true;
}

bool alignStream(std::ostream& stream, uint64_t alignment)
{
    const std::streampos pos = stream.tellp();
    if (pos == std::streampos(-1)) {
        return false;
    }
    const uint64_t offset = static_cast<uint64_t>(pos);
    const uint64_t aligned = alignUp(offset, alignment);
    return writeZeros(stream, aligned - offset);
}

template <typename T>
bool writePod(std::ostream& stream, const T& value)
{
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return stream.good();
}

template <typename T>
bool writeArray(std::ostream& stream, const std::vector<T>& values)
{
    if (values.empty()) {
        return true;
    }
    stream.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T)));
    return stream.good();
}

void appendBytes(std::vector<uint8_t>& bytes, const void* data, size_t byteSize)
{
    const uint8_t* first = static_cast<const uint8_t*>(data);
    bytes.insert(bytes.end(), first, first + byteSize);
}

void appendPadding(std::vector<uint8_t>& bytes, uint64_t alignment)
{
    const uint64_t aligned = alignUp(bytes.size(), alignment);
    bytes.resize(static_cast<size_t>(aligned));
}

struct PayloadPosition {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct PayloadNormal {
    float x = 0.0f;
    float y = 0.0f;
    float z = 1.0f;
    float w = 0.0f;
};

struct PayloadTexcoord {
    float x = 0.0f;
    float y = 0.0f;
};

struct PagePayloadBuildInput {
    const RenderPrimitive* primitive = nullptr;
    const std::vector<MeshletCluster>* clusters = nullptr;
    const std::vector<uint32_t>* vertices = nullptr;
    const std::vector<uint8_t>* triangles = nullptr;
    uint32_t firstCluster = 0;
    uint32_t clusterCount = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t lodGroupIndex = 0;
    uint32_t materialIndex = 0;
    Bounds bounds;
    float maxQuadricError = 0.0f;
};

bool buildPagePayload(
    const PagePayloadBuildInput& input,
    std::vector<uint8_t>& outPayload,
    MeshletStreamPageInfo& outPageInfo,
    std::string& reason)
{
    outPayload.clear();
    outPageInfo = {};
    if (input.primitive == nullptr ||
        input.clusters == nullptr ||
        input.vertices == nullptr ||
        input.triangles == nullptr ||
        input.clusterCount == 0 ||
        input.firstCluster > input.clusters->size() ||
        input.clusterCount > input.clusters->size() - input.firstCluster) {
        reason = "invalid meshlet page build input";
        return false;
    }

    std::vector<MeshletStreamPayloadCluster> payloadClusters;
    std::vector<PayloadPosition> payloadPositions;
    std::vector<PayloadNormal> payloadNormals;
    std::vector<PayloadTexcoord> payloadTexcoords0;
    std::vector<uint32_t> payloadMaterials;
    std::vector<uint8_t> payloadTriangles;
    payloadClusters.reserve(input.clusterCount);
    const bool hasNormals = input.primitive->normals.size() == input.primitive->positions.size();
    const bool hasTexcoords0 = input.primitive->texcoords0.size() == input.primitive->positions.size();
    if (hasNormals) {
        payloadNormals.reserve(input.clusterCount * kMeshletClusterMaxVertices);
    }
    if (hasTexcoords0) {
        payloadTexcoords0.reserve(input.clusterCount * kMeshletClusterMaxVertices);
    }
    payloadMaterials.reserve(input.clusterCount);

    Bounds pageBounds;
    for (uint32_t localCluster = 0; localCluster < input.clusterCount; ++localCluster) {
        const MeshletCluster& cluster =
            (*input.clusters)[static_cast<size_t>(input.firstCluster) + localCluster];
        if (cluster.vertexCount == 0 ||
            cluster.vertexCount > kMeshletClusterMaxVertices ||
            cluster.triangleCount == 0 ||
            cluster.triangleCount > kMeshletClusterMaxTriangles ||
            cluster.vertexOffset > input.vertices->size() ||
            cluster.vertexCount > input.vertices->size() - cluster.vertexOffset ||
            cluster.triangleOffset > input.triangles->size() ||
            cluster.triangleCount * 3u > input.triangles->size() - cluster.triangleOffset) {
            reason = "meshlet page contains invalid cluster ranges";
            return false;
        }

        const uint32_t vertexOffset = static_cast<uint32_t>(payloadPositions.size());
        const uint32_t triangleOffset = static_cast<uint32_t>(payloadTriangles.size());

        for (uint32_t vertex = 0; vertex < cluster.vertexCount; ++vertex) {
            const uint32_t sourceVertex =
                (*input.vertices)[static_cast<size_t>(cluster.vertexOffset) + vertex];
            if (sourceVertex >= input.primitive->positions.size()) {
                reason = "meshlet page contains an out-of-range vertex reference";
                return false;
            }
            const float3& position = input.primitive->positions[sourceVertex];
            payloadPositions.push_back(PayloadPosition{position.x, position.y, position.z, 1.0f});
            if (hasNormals) {
                const float3& normal = input.primitive->normals[sourceVertex];
                payloadNormals.push_back(PayloadNormal{normal.x, normal.y, normal.z, 0.0f});
            }
            if (hasTexcoords0) {
                const float2& texcoord = input.primitive->texcoords0[sourceVertex];
                payloadTexcoords0.push_back(PayloadTexcoord{texcoord.x, texcoord.y});
            }
            pageBounds.include(position);
        }

        for (uint32_t index = 0; index < cluster.triangleCount * 3u; ++index) {
            const uint8_t localVertex =
                (*input.triangles)[static_cast<size_t>(cluster.triangleOffset) + index];
            if (localVertex >= cluster.vertexCount) {
                reason = "meshlet page contains an out-of-range local triangle index";
                return false;
            }
            payloadTriangles.push_back(localVertex);
        }

        payloadClusters.push_back(MeshletStreamPayloadCluster{
            .vertexOffset = vertexOffset,
            .vertexCount = cluster.vertexCount,
            .triangleOffset = triangleOffset,
            .triangleCount = cluster.triangleCount,
            .primitiveIndex = input.primitiveIndex,
            .materialIndex = input.materialIndex,
            .lodLevel = input.lodLevel,
            .lodGroupIndex = input.lodGroupIndex,
        });
        payloadMaterials.push_back(input.materialIndex);
    }

    MeshletStreamPayloadHeader header;
    header.magic = kPayloadMagic;
    header.version = kPayloadVersion;
    header.clusterCount = static_cast<uint32_t>(payloadClusters.size());
    header.vertexCount = static_cast<uint32_t>(payloadPositions.size());
    header.triangleIndexCount = static_cast<uint32_t>(payloadTriangles.size());
    header.primitiveIndex = input.primitiveIndex;
    header.materialIndex = input.materialIndex;
    header.lodLevel = input.lodLevel;
    header.lodGroupIndex = input.lodGroupIndex;
    header.attributeFlags = kMeshletStreamPayloadAttributePosition | kMeshletStreamPayloadAttributeMaterial;
    header.compressionMode = static_cast<uint32_t>(MeshletStreamPayloadCompression::None);
    header.positionFormat = static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x4);
    header.materialFormat = static_cast<uint32_t>(MeshletStreamPayloadFormat::Uint32);
    header.materialCount = static_cast<uint32_t>(payloadMaterials.size());
    if (hasNormals) {
        header.attributeFlags |= kMeshletStreamPayloadAttributeNormal;
        header.normalFormat = static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x4);
    }
    if (hasTexcoords0) {
        header.attributeFlags |= kMeshletStreamPayloadAttributeTexcoord0;
        header.texcoord0Format = static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x2);
    }

    outPayload.resize(sizeof(MeshletStreamPayloadHeader));
    appendPadding(outPayload, 16);
    header.clusterOffsetBytes = static_cast<uint32_t>(outPayload.size());
    appendBytes(outPayload, payloadClusters.data(), payloadClusters.size() * sizeof(MeshletStreamPayloadCluster));
    appendPadding(outPayload, 16);
    header.positionOffsetBytes = static_cast<uint32_t>(outPayload.size());
    appendBytes(outPayload, payloadPositions.data(), payloadPositions.size() * sizeof(PayloadPosition));
    if (hasNormals) {
        appendPadding(outPayload, 16);
        header.normalOffsetBytes = static_cast<uint32_t>(outPayload.size());
        appendBytes(outPayload, payloadNormals.data(), payloadNormals.size() * sizeof(PayloadNormal));
    }
    if (hasTexcoords0) {
        appendPadding(outPayload, 8);
        header.texcoord0OffsetBytes = static_cast<uint32_t>(outPayload.size());
        appendBytes(outPayload, payloadTexcoords0.data(), payloadTexcoords0.size() * sizeof(PayloadTexcoord));
    }
    appendPadding(outPayload, 4);
    header.materialOffsetBytes = static_cast<uint32_t>(outPayload.size());
    appendBytes(outPayload, payloadMaterials.data(), payloadMaterials.size() * sizeof(uint32_t));
    appendPadding(outPayload, 4);
    header.triangleOffsetBytes = static_cast<uint32_t>(outPayload.size());
    appendBytes(outPayload, payloadTriangles.data(), payloadTriangles.size());
    appendPadding(outPayload, 16);
    if (outPayload.size() > std::numeric_limits<uint32_t>::max()) {
        reason = "meshlet page payload exceeds 32-bit shader offsets";
        return false;
    }
    header.payloadByteSize = static_cast<uint32_t>(outPayload.size());
    header.uncompressedPayloadByteSize = header.payloadByteSize;
    std::memcpy(outPayload.data(), &header, sizeof(header));

    outPageInfo.uncompressedSize = outPayload.size();
    outPageInfo.payloadSize = outPayload.size();
    outPageInfo.primitiveIndex = input.primitiveIndex;
    outPageInfo.lodLevel = input.lodLevel;
    outPageInfo.lodGroupIndex = input.lodGroupIndex;
    outPageInfo.clusterCount = header.clusterCount;
    outPageInfo.vertexCount = header.vertexCount;
    outPageInfo.triangleIndexCount = header.triangleIndexCount;
    outPageInfo.materialIndex = input.materialIndex;
    outPageInfo.attributeFlags = header.attributeFlags;
    outPageInfo.compressionMode = header.compressionMode;
    outPageInfo.bounds = pageBounds.valid ? makeStreamBounds(pageBounds) : makeStreamBounds(input.bounds);
    outPageInfo.maxQuadricError = input.maxQuadricError;
    return true;
}

bool writePayload(
    std::ostream& stream,
    const std::vector<uint8_t>& payload,
    uint64_t& outOffset)
{
    if (!alignStream(stream, kFileAlignment)) {
        return false;
    }
    const std::streampos pos = stream.tellp();
    if (pos == std::streampos(-1)) {
        return false;
    }
    outOffset = static_cast<uint64_t>(pos);
    if (!payload.empty()) {
        stream.write(
            reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
    }
    return stream.good();
}

void copyMatrix(const float4x4& matrix, float outValues[16])
{
    for (size_t index = 0; index < 16; ++index) {
        outValues[index] = matrix.a[index];
    }
}

} // namespace

struct MeshletStreamAsset::Impl {
    ~Impl()
    {
#ifdef _WIN32
        if (data != nullptr) {
            UnmapViewOfFile(data);
        }
        if (mapping != nullptr) {
            CloseHandle(mapping);
        }
        if (file != INVALID_HANDLE_VALUE) {
            CloseHandle(file);
        }
#endif
    }

#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#endif
    std::vector<uint8_t> ownedData;
    const uint8_t* data = nullptr;
    uint64_t dataSize = 0;
    MeshletStreamFileHeader header;
    std::span<const MeshletStreamPrimitiveInfo> primitives;
    std::span<const MeshletStreamInstanceInfo> instances;
    std::span<const MeshletStreamGeometryInfo> geometries;
    std::span<const MeshletStreamLodLevelInfo> lodLevels;
    std::span<const MeshletStreamPageInfo> pages;
    std::span<const uint64_t> pageOffsets;
};

MeshletStreamAsset::MeshletStreamAsset() = default;

MeshletStreamAsset::~MeshletStreamAsset()
{
    close();
}

MeshletStreamAsset::MeshletStreamAsset(MeshletStreamAsset&& other) noexcept
    : path_(std::move(other.path_))
    , impl_(other.impl_)
{
    other.impl_ = nullptr;
}

MeshletStreamAsset& MeshletStreamAsset::operator=(MeshletStreamAsset&& other) noexcept
{
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

bool MeshletStreamAsset::open(const std::filesystem::path& path, std::string& reason)
{
    close();
    reason.clear();

    auto impl = std::make_unique<Impl>();
    impl->dataSize = sourceFileSizeFor(path);
    if (impl->dataSize == 0) {
        reason = "streamasset file is empty or cannot be stat'ed";
        return false;
    }

#ifdef _WIN32
    impl->file = CreateFileW(
        path.wstring().c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr);
    if (impl->file == INVALID_HANDLE_VALUE) {
        reason = "streamasset file cannot be opened";
        return false;
    }
    LARGE_INTEGER fileSize{};
    if (!GetFileSizeEx(impl->file, &fileSize) || fileSize.QuadPart <= 0) {
        reason = "streamasset file size is invalid";
        return false;
    }
    impl->dataSize = static_cast<uint64_t>(fileSize.QuadPart);
    impl->mapping = CreateFileMappingW(impl->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (impl->mapping == nullptr) {
        reason = "streamasset file cannot be memory mapped";
        return false;
    }
    impl->data = static_cast<const uint8_t*>(MapViewOfFile(impl->mapping, FILE_MAP_READ, 0, 0, 0));
    if (impl->data == nullptr) {
        reason = "streamasset file mapping view cannot be opened";
        return false;
    }
#else
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        reason = "streamasset file cannot be opened";
        return false;
    }
    impl->ownedData.resize(static_cast<size_t>(impl->dataSize));
    file.read(reinterpret_cast<char*>(impl->ownedData.data()), static_cast<std::streamsize>(impl->ownedData.size()));
    if (!file) {
        reason = "streamasset file cannot be read";
        return false;
    }
    impl->data = impl->ownedData.data();
#endif

    if (impl->dataSize < sizeof(MeshletStreamFileHeader)) {
        reason = "streamasset header is truncated";
        return false;
    }

    std::memcpy(&impl->header, impl->data, sizeof(impl->header));
    if (std::memcmp(impl->header.magic, kMeshletStreamMagic.data(), kMeshletStreamMagic.size()) != 0 ||
        impl->header.version != kMeshletStreamVersion ||
        impl->header.endian != kMeshletStreamEndian) {
        reason = "streamasset header magic or version is unsupported";
        return false;
    }
    if (impl->header.fileSize != impl->dataSize ||
        !meshletStreamBuildParamsMatch(impl->header) ||
        impl->header.maxPagePayloadBytes == 0) {
        reason = "streamasset header metadata is invalid";
        return false;
    }
    if (!rangeWithin<MeshletStreamPrimitiveInfo>(impl->dataSize, impl->header.primitiveOffset, impl->header.primitiveCount) ||
        !rangeWithin<MeshletStreamInstanceInfo>(impl->dataSize, impl->header.instanceOffset, impl->header.instanceCount) ||
        !rangeWithin<MeshletStreamGeometryInfo>(impl->dataSize, impl->header.geometryOffset, impl->header.geometryCount) ||
        !rangeWithin<MeshletStreamLodLevelInfo>(impl->dataSize, impl->header.lodLevelOffset, impl->header.lodLevelCount) ||
        !rangeWithin<MeshletStreamPageInfo>(impl->dataSize, impl->header.pageInfoOffset, impl->header.pageCount) ||
        !rangeWithin<uint64_t>(impl->dataSize, impl->header.pageOffsetTableOffset, impl->header.pageCount)) {
        reason = "streamasset directory exceeds file bounds";
        return false;
    }

    impl->primitives = makeSpan<MeshletStreamPrimitiveInfo>(impl->data, impl->header.primitiveOffset, impl->header.primitiveCount);
    impl->instances = makeSpan<MeshletStreamInstanceInfo>(impl->data, impl->header.instanceOffset, impl->header.instanceCount);
    impl->geometries = makeSpan<MeshletStreamGeometryInfo>(impl->data, impl->header.geometryOffset, impl->header.geometryCount);
    impl->lodLevels = makeSpan<MeshletStreamLodLevelInfo>(impl->data, impl->header.lodLevelOffset, impl->header.lodLevelCount);
    impl->pages = makeSpan<MeshletStreamPageInfo>(impl->data, impl->header.pageInfoOffset, impl->header.pageCount);
    impl->pageOffsets = makeSpan<uint64_t>(impl->data, impl->header.pageOffsetTableOffset, impl->header.pageCount);

    for (uint32_t primitiveIndex = 0; primitiveIndex < impl->primitives.size(); ++primitiveIndex) {
        const MeshletStreamPrimitiveInfo& primitive = impl->primitives[primitiveIndex];
        if (primitive.lodLevelCount == 0 ||
            primitive.pageCount == 0 ||
            primitive.fallbackPageCount == 0 ||
            primitive.lodLevelOffset > impl->lodLevels.size() ||
            primitive.lodLevelCount > impl->lodLevels.size() - primitive.lodLevelOffset ||
            primitive.pageOffset > impl->pages.size() ||
            primitive.pageCount > impl->pages.size() - primitive.pageOffset ||
            primitive.fallbackPageOffset > impl->pages.size() ||
            primitive.fallbackPageCount > impl->pages.size() - primitive.fallbackPageOffset) {
            reason = "streamasset primitive directory contains invalid ranges";
            return false;
        }
    }

    if (impl->geometries.size() != impl->primitives.size()) {
        reason = "streamasset geometry directory does not match primitive directory";
        return false;
    }
    for (uint32_t geometryIndex = 0; geometryIndex < impl->geometries.size(); ++geometryIndex) {
        const MeshletStreamGeometryInfo& geometry = impl->geometries[geometryIndex];
        if (geometry.primitiveIndex >= impl->primitives.size() ||
            geometry.pageCount == 0 ||
            geometry.pageOffset > impl->pages.size() ||
            geometry.pageCount > impl->pages.size() - geometry.pageOffset ||
            geometry.pagePayloadOffsetTableOffset > impl->pageOffsets.size() ||
            geometry.pagePayloadOffsetTableCount != geometry.pageCount ||
            geometry.pagePayloadOffsetTableCount >
                impl->pageOffsets.size() - geometry.pagePayloadOffsetTableOffset) {
            reason = "streamasset geometry directory contains invalid ranges";
            return false;
        }

        const MeshletStreamPrimitiveInfo& primitive = impl->primitives[geometry.primitiveIndex];
        if (geometry.pageOffset != primitive.pageOffset ||
            geometry.pageCount != primitive.pageCount ||
            geometry.renderPrimitiveIndex != primitive.renderPrimitiveIndex ||
            geometry.pagePayloadOffsetTableOffset != primitive.pageOffset) {
            reason = "streamasset geometry directory does not match primitive ranges";
            return false;
        }
    }

    for (const MeshletStreamInstanceInfo& instance : impl->instances) {
        if (instance.primitiveIndex >= impl->primitives.size()) {
            reason = "streamasset instance directory references an invalid primitive";
            return false;
        }
    }

    for (uint32_t lodIndex = 0; lodIndex < impl->lodLevels.size(); ++lodIndex) {
        const MeshletStreamLodLevelInfo& lod = impl->lodLevels[lodIndex];
        if (lod.primitiveIndex >= impl->primitives.size() ||
            lod.pageCount == 0 ||
            lod.pageOffset > impl->pages.size() ||
            lod.pageCount > impl->pages.size() - lod.pageOffset) {
            reason = "streamasset LOD directory contains invalid ranges";
            return false;
        }
        for (uint32_t localPage = 0; localPage < lod.pageCount; ++localPage) {
            const MeshletStreamPageInfo& page = impl->pages[lod.pageOffset + localPage];
            if (page.primitiveIndex != lod.primitiveIndex || page.lodLevel != lod.lodLevel) {
                reason = "streamasset LOD directory does not match page directory";
                return false;
            }
        }
    }

    for (uint32_t pageIndex = 0; pageIndex < impl->header.pageCount; ++pageIndex) {
        const MeshletStreamPageInfo& page = impl->pages[pageIndex];
        if (page.primitiveIndex >= impl->primitives.size() ||
            page.payloadOffset != impl->pageOffsets[pageIndex] ||
            page.payloadOffset > impl->dataSize ||
            page.payloadSize > impl->dataSize - page.payloadOffset ||
            page.uncompressedSize == 0 ||
            page.payloadSize == 0 ||
            page.uncompressedSize > impl->header.maxPagePayloadBytes) {
            reason = "streamasset page payload exceeds file bounds";
            return false;
        }
        if (!validatePayloadHeader(impl->data, impl->dataSize, page, pageIndex, reason)) {
            return false;
        }
    }

    for (const MeshletStreamGeometryInfo& geometry : impl->geometries) {
        uint64_t payloadFileBegin = std::numeric_limits<uint64_t>::max();
        uint64_t payloadFileEnd = 0;
        for (uint32_t page = 0; page < geometry.pageCount; ++page) {
            const MeshletStreamPageInfo& pageInfo = impl->pages[geometry.pageOffset + page];
            payloadFileBegin = std::min(payloadFileBegin, pageInfo.payloadOffset);
            payloadFileEnd = std::max(payloadFileEnd, pageInfo.payloadOffset + pageInfo.payloadSize);
        }
        const uint64_t expectedOffset =
            payloadFileBegin == std::numeric_limits<uint64_t>::max() ? 0u : payloadFileBegin;
        const uint64_t expectedSize = payloadFileEnd > payloadFileBegin ? payloadFileEnd - payloadFileBegin : 0u;
        if (geometry.payloadFileOffset != expectedOffset || geometry.payloadFileSize != expectedSize) {
            reason = "streamasset geometry payload span does not match page directory";
            return false;
        }
    }

    path_ = path;
    impl_ = impl.release();
    return true;
}

void MeshletStreamAsset::close()
{
    if (impl_ == nullptr) {
        return;
    }
    delete impl_;
    impl_ = nullptr;
    path_.clear();
}

bool MeshletStreamAsset::valid() const
{
    return impl_ != nullptr && impl_->data != nullptr;
}

bool MeshletStreamAsset::isCurrentForSource(const std::filesystem::path& sourcePath) const
{
    return valid() &&
        impl_->header.sourceFileSize == sourceFileSizeFor(sourcePath) &&
        impl_->header.sourceWriteTime == sourceWriteTimeFor(sourcePath) &&
        meshletStreamBuildParamsMatch(impl_->header);
}

uint32_t MeshletStreamAsset::primitiveCount() const
{
    return valid() ? impl_->header.primitiveCount : 0;
}

uint32_t MeshletStreamAsset::instanceCount() const
{
    return valid() ? impl_->header.instanceCount : 0;
}

uint32_t MeshletStreamAsset::geometryCount() const
{
    return valid() ? impl_->header.geometryCount : 0;
}

uint32_t MeshletStreamAsset::lodLevelCount() const
{
    return valid() ? impl_->header.lodLevelCount : 0;
}

uint32_t MeshletStreamAsset::pageCount() const
{
    return valid() ? impl_->header.pageCount : 0;
}

uint32_t MeshletStreamAsset::maxPagePayloadBytes() const
{
    return valid() ? impl_->header.maxPagePayloadBytes : 0;
}

uint64_t MeshletStreamAsset::sourceFileSize() const
{
    return valid() ? impl_->header.sourceFileSize : 0;
}

int64_t MeshletStreamAsset::sourceWriteTime() const
{
    return valid() ? impl_->header.sourceWriteTime : 0;
}

std::span<const MeshletStreamPrimitiveInfo> MeshletStreamAsset::primitives() const
{
    return valid() ? impl_->primitives : std::span<const MeshletStreamPrimitiveInfo>{};
}

std::span<const MeshletStreamInstanceInfo> MeshletStreamAsset::instances() const
{
    return valid() ? impl_->instances : std::span<const MeshletStreamInstanceInfo>{};
}

std::span<const MeshletStreamGeometryInfo> MeshletStreamAsset::geometries() const
{
    return valid() ? impl_->geometries : std::span<const MeshletStreamGeometryInfo>{};
}

std::span<const MeshletStreamLodLevelInfo> MeshletStreamAsset::lodLevels() const
{
    return valid() ? impl_->lodLevels : std::span<const MeshletStreamLodLevelInfo>{};
}

std::span<const MeshletStreamPageInfo> MeshletStreamAsset::pages() const
{
    return valid() ? impl_->pages : std::span<const MeshletStreamPageInfo>{};
}

std::span<const uint64_t> MeshletStreamAsset::pagePayloadOffsets() const
{
    return valid() ? impl_->pageOffsets : std::span<const uint64_t>{};
}

std::span<const uint64_t> MeshletStreamAsset::geometryPagePayloadOffsets(uint32_t geometryIndex) const
{
    if (!valid() || geometryIndex >= impl_->geometries.size()) {
        return {};
    }
    const MeshletStreamGeometryInfo& geometry = impl_->geometries[geometryIndex];
    return std::span<const uint64_t>(
        impl_->pageOffsets.data() + geometry.pagePayloadOffsetTableOffset,
        static_cast<size_t>(geometry.pagePayloadOffsetTableCount));
}

std::span<const uint8_t> MeshletStreamAsset::pagePayload(uint32_t pageIndex) const
{
    if (!valid() || pageIndex >= impl_->pages.size()) {
        return {};
    }
    const MeshletStreamPageInfo& page = impl_->pages[pageIndex];
    return std::span<const uint8_t>(
        impl_->data + page.payloadOffset,
        static_cast<size_t>(page.payloadSize));
}

bool decodeMeshletStreamPayloadForDevice(
    const MeshletStreamPageInfo& page,
    std::span<const uint8_t> storedPayload,
    std::vector<uint8_t>& scratchPayload,
    std::span<const uint8_t>& outDevicePayload,
    std::string& reason)
{
    reason.clear();
    outDevicePayload = {};
    if (storedPayload.empty() ||
        storedPayload.size() != page.payloadSize ||
        page.uncompressedSize == 0 ||
        page.uncompressedSize > std::numeric_limits<uint32_t>::max()) {
        reason = "streamasset stored payload does not match page metadata";
        return false;
    }

    if (page.compressionMode == static_cast<uint32_t>(MeshletStreamPayloadCompression::None)) {
        if (page.payloadSize != page.uncompressedSize) {
            reason = "streamasset uncompressed payload has mismatched sizes";
            return false;
        }
        outDevicePayload = storedPayload;
        return true;
    }

    if (page.compressionMode != static_cast<uint32_t>(MeshletStreamPayloadCompression::ByteRle)) {
        reason = "streamasset compressed payload mode is unsupported";
        return false;
    }
    if (storedPayload.size() < sizeof(MeshletStreamPayloadHeader) ||
        page.uncompressedSize < sizeof(MeshletStreamPayloadHeader)) {
        reason = "streamasset compressed payload is too small";
        return false;
    }

    MeshletStreamPayloadHeader storedHeader;
    std::memcpy(&storedHeader, storedPayload.data(), sizeof(storedHeader));
    if (storedHeader.payloadByteSize != page.payloadSize ||
        storedHeader.uncompressedPayloadByteSize != page.uncompressedSize ||
        storedHeader.compressionMode != page.compressionMode) {
        reason = "streamasset compressed payload header does not match page metadata";
        return false;
    }

    scratchPayload.resize(static_cast<size_t>(page.uncompressedSize));
    std::memcpy(scratchPayload.data(), storedPayload.data(), sizeof(MeshletStreamPayloadHeader));

    const std::span<const uint8_t> encodedBody(
        storedPayload.data() + sizeof(MeshletStreamPayloadHeader),
        storedPayload.size() - sizeof(MeshletStreamPayloadHeader));
    std::span<uint8_t> decodedBody(
        scratchPayload.data() + sizeof(MeshletStreamPayloadHeader),
        scratchPayload.size() - sizeof(MeshletStreamPayloadHeader));
    if (!decodeByteRle(encodedBody, decodedBody)) {
        reason = "streamasset ByteRle payload decompression failed";
        return false;
    }

    auto* deviceHeader = reinterpret_cast<MeshletStreamPayloadHeader*>(scratchPayload.data());
    deviceHeader->payloadByteSize = static_cast<uint32_t>(page.uncompressedSize);
    deviceHeader->uncompressedPayloadByteSize = static_cast<uint32_t>(page.uncompressedSize);
    deviceHeader->compressionMode = static_cast<uint32_t>(MeshletStreamPayloadCompression::None);
    outDevicePayload = std::span<const uint8_t>(scratchPayload.data(), scratchPayload.size());
    return true;
}

std::filesystem::path meshletStreamAssetPathFor(const std::filesystem::path& sourcePath)
{
    std::filesystem::path path = sourcePath;
    path += kMeshletStreamAssetSuffix;
    return path;
}

bool buildMeshletStreamAsset(const MeshletStreamAssetBuildDesc& desc, std::string& reason)
{
    reason.clear();
    if (desc.scene == nullptr || !desc.scene->valid()) {
        reason = "streamasset build requires a loaded scene";
        return false;
    }

    const std::filesystem::path outputPath = desc.outputPath.empty()
        ? meshletStreamAssetPathFor(desc.sourcePath)
        : desc.outputPath;
    std::error_code createError;
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path(), createError);
        if (createError) {
            reason = createError.message();
            return false;
        }
    }

    std::ofstream stream(outputPath, std::ios::binary | std::ios::trunc);
    if (!stream) {
        reason = "streamasset output file cannot be opened";
        return false;
    }

    MeshletStreamFileHeader header;
    std::memcpy(header.magic, kMeshletStreamMagic.data(), kMeshletStreamMagic.size());
    header.version = kMeshletStreamVersion;
    header.endian = kMeshletStreamEndian;
    header.sourceFileSize = sourceFileSizeFor(desc.sourcePath);
    header.sourceWriteTime = sourceWriteTimeFor(desc.sourcePath);
    header.pagePayloadAlignment = static_cast<uint32_t>(kPageSlotAlignment);
    header.maxVertices = kMeshletClusterMaxVertices;
    header.minTriangles = kMeshletClusterMinTriangles;
    header.maxTriangles = kMeshletClusterMaxTriangles;
    header.lodGroupSize = kMeshletLodGroupSize;
    header.fillWeight = kMeshletClusterFillWeight;
    header.lodErrorMergePrevious = kMeshletLodErrorMergePrevious;
    header.lodErrorMergeAdditive = kMeshletLodErrorMergeAdditive;

    if (!writePod(stream, header)) {
        reason = "streamasset header placeholder write failed";
        return false;
    }

    const std::vector<RenderPrimitive>& renderPrimitives = desc.scene->renderPrimitives();
    std::vector<int32_t> primitiveMap(renderPrimitives.size(), kInvalidSceneIndex);
    std::vector<MeshletStreamPrimitiveInfo> primitives;
    std::vector<MeshletStreamInstanceInfo> instances;
    std::vector<MeshletStreamGeometryInfo> geometries;
    std::vector<MeshletStreamLodLevelInfo> lodLevels;
    std::vector<MeshletStreamPageInfo> pages;
    std::vector<uint64_t> pageOffsets;

    std::vector<uint8_t> payload;
    std::vector<uint8_t> storedPayload;
    for (size_t renderPrimitiveIndex = 0; renderPrimitiveIndex < renderPrimitives.size(); ++renderPrimitiveIndex) {
        const RenderPrimitive& primitive = renderPrimitives[renderPrimitiveIndex];
        if (primitive.mode != 4 || primitive.positions.empty()) {
            continue;
        }
        const bool hasLodGroups =
            !primitive.meshletLodLevels.empty() &&
            !primitive.meshletLodGroups.empty() &&
            !primitive.meshletLodClusters.empty() &&
            !primitive.meshletLodVertices.empty() &&
            !primitive.meshletLodTriangles.empty();
        const bool hasBaseMeshlets =
            !primitive.meshletClusters.empty() &&
            !primitive.meshletVertices.empty() &&
            !primitive.meshletTriangles.empty();
        if (!hasLodGroups && !hasBaseMeshlets) {
            continue;
        }

        const uint32_t primitiveIndex = static_cast<uint32_t>(primitives.size());
        primitiveMap[renderPrimitiveIndex] = static_cast<int32_t>(primitiveIndex);

        MeshletStreamPrimitiveInfo primitiveInfo;
        primitiveInfo.renderPrimitiveIndex = static_cast<uint32_t>(renderPrimitiveIndex);
        primitiveInfo.materialIndex = static_cast<uint32_t>(std::max(primitive.materialIndex, 0));
        primitiveInfo.lodLevelOffset = static_cast<uint32_t>(lodLevels.size());
        primitiveInfo.pageOffset = static_cast<uint32_t>(pages.size());
        primitiveInfo.bounds = makeStreamBounds(primitive.localBounds);

        if (hasLodGroups) {
            uint32_t bestFallbackPageCount = std::numeric_limits<uint32_t>::max();
            for (uint32_t lodLevelIndex = 0; lodLevelIndex < primitive.meshletLodLevels.size(); ++lodLevelIndex) {
                const MeshletLodLevel& sourceLevel = primitive.meshletLodLevels[lodLevelIndex];
                if (sourceLevel.groupOffset > primitive.meshletLodGroups.size() ||
                    sourceLevel.groupCount > primitive.meshletLodGroups.size() - sourceLevel.groupOffset) {
                    reason = "primitive meshlet LOD level has invalid group range";
                    return false;
                }

                MeshletStreamLodLevelInfo lodInfo;
                lodInfo.primitiveIndex = primitiveIndex;
                lodInfo.lodLevel = lodLevelIndex;
                lodInfo.pageOffset = static_cast<uint32_t>(pages.size());
                lodInfo.minBoundingSphereRadius = sourceLevel.minBoundingSphereRadius;
                lodInfo.minMaxQuadricError = sourceLevel.minMaxQuadricError;

                for (uint32_t groupChild = 0; groupChild < sourceLevel.groupCount; ++groupChild) {
                    const uint32_t groupIndex = sourceLevel.groupOffset + groupChild;
                    const MeshletLodGroup& group = primitive.meshletLodGroups[groupIndex];
                    PagePayloadBuildInput pageInput{
                        .primitive = &primitive,
                        .clusters = &primitive.meshletLodClusters,
                        .vertices = &primitive.meshletLodVertices,
                        .triangles = &primitive.meshletLodTriangles,
                        .firstCluster = group.clusterOffset,
                        .clusterCount = group.clusterCount,
                        .primitiveIndex = primitiveIndex,
                        .lodLevel = lodLevelIndex,
                        .lodGroupIndex = groupIndex,
                        .materialIndex = primitiveInfo.materialIndex,
                        .bounds = makeBounds(makeStreamBounds(group.bounds)),
                        .maxQuadricError = group.maxQuadricError,
                    };
                    MeshletStreamPageInfo pageInfo;
                    if (!buildPagePayload(pageInput, payload, pageInfo, reason)) {
                        return false;
                    }
                    if (!encodePayloadForStorage(
                            payload,
                            desc.compressionMode,
                            storedPayload,
                            pageInfo,
                            reason)) {
                        return false;
                    }
                    uint64_t payloadOffset = 0;
                    if (!writePayload(stream, storedPayload, payloadOffset)) {
                        reason = "streamasset page payload write failed";
                        return false;
                    }
                    pageInfo.payloadOffset = payloadOffset;
                    header.maxPagePayloadBytes = std::max<uint32_t>(
                        header.maxPagePayloadBytes,
                        static_cast<uint32_t>(pageInfo.uncompressedSize));
                    lodInfo.clusterCount += pageInfo.clusterCount;
                    pages.push_back(pageInfo);
                    pageOffsets.push_back(payloadOffset);
                }

                lodInfo.pageCount = static_cast<uint32_t>(pages.size()) - lodInfo.pageOffset;
                if (lodInfo.pageCount > 0 && lodInfo.pageCount < bestFallbackPageCount) {
                    primitiveInfo.fallbackPageOffset = lodInfo.pageOffset;
                    primitiveInfo.fallbackPageCount = lodInfo.pageCount;
                    bestFallbackPageCount = lodInfo.pageCount;
                }
                lodLevels.push_back(lodInfo);
            }
        } else {
            MeshletStreamLodLevelInfo lodInfo;
            lodInfo.primitiveIndex = primitiveIndex;
            lodInfo.lodLevel = 0;
            lodInfo.pageOffset = static_cast<uint32_t>(pages.size());

            PagePayloadBuildInput pageInput{
                .primitive = &primitive,
                .clusters = &primitive.meshletClusters,
                .vertices = &primitive.meshletVertices,
                .triangles = &primitive.meshletTriangles,
                .firstCluster = 0,
                .clusterCount = static_cast<uint32_t>(primitive.meshletClusters.size()),
                .primitiveIndex = primitiveIndex,
                .lodLevel = 0,
                .lodGroupIndex = 0,
                .materialIndex = primitiveInfo.materialIndex,
                .bounds = primitive.localBounds,
            };
            MeshletStreamPageInfo pageInfo;
            if (!buildPagePayload(pageInput, payload, pageInfo, reason)) {
                return false;
            }
            if (!encodePayloadForStorage(
                    payload,
                    desc.compressionMode,
                    storedPayload,
                    pageInfo,
                    reason)) {
                return false;
            }
            uint64_t payloadOffset = 0;
            if (!writePayload(stream, storedPayload, payloadOffset)) {
                reason = "streamasset fallback page payload write failed";
                return false;
            }
            pageInfo.payloadOffset = payloadOffset;
            header.maxPagePayloadBytes =
                std::max<uint32_t>(header.maxPagePayloadBytes, static_cast<uint32_t>(pageInfo.uncompressedSize));
            lodInfo.pageCount = 1;
            lodInfo.clusterCount = pageInfo.clusterCount;
            primitiveInfo.fallbackPageOffset = lodInfo.pageOffset;
            primitiveInfo.fallbackPageCount = 1;
            pages.push_back(pageInfo);
            pageOffsets.push_back(payloadOffset);
            lodLevels.push_back(lodInfo);
        }

        primitiveInfo.lodLevelCount = static_cast<uint32_t>(lodLevels.size()) - primitiveInfo.lodLevelOffset;
        primitiveInfo.pageCount = static_cast<uint32_t>(pages.size()) - primitiveInfo.pageOffset;
        if (primitiveInfo.pageCount == 0 || primitiveInfo.fallbackPageCount == 0) {
            primitiveMap[renderPrimitiveIndex] = kInvalidSceneIndex;
            continue;
        }
        uint64_t payloadFileBegin = std::numeric_limits<uint64_t>::max();
        uint64_t payloadFileEnd = 0;
        for (uint32_t page = 0; page < primitiveInfo.pageCount; ++page) {
            const MeshletStreamPageInfo& pageInfo = pages[primitiveInfo.pageOffset + page];
            payloadFileBegin = std::min(payloadFileBegin, pageInfo.payloadOffset);
            payloadFileEnd = std::max(payloadFileEnd, pageInfo.payloadOffset + pageInfo.payloadSize);
        }
        geometries.push_back(MeshletStreamGeometryInfo{
            .primitiveIndex = primitiveIndex,
            .renderPrimitiveIndex = primitiveInfo.renderPrimitiveIndex,
            .pageOffset = primitiveInfo.pageOffset,
            .pageCount = primitiveInfo.pageCount,
            .pagePayloadOffsetTableOffset = primitiveInfo.pageOffset,
            .pagePayloadOffsetTableCount = primitiveInfo.pageCount,
            .payloadFileOffset = payloadFileBegin == std::numeric_limits<uint64_t>::max() ? 0u : payloadFileBegin,
            .payloadFileSize = payloadFileEnd > payloadFileBegin ? payloadFileEnd - payloadFileBegin : 0u,
        });
        primitives.push_back(primitiveInfo);
    }

    if (primitives.empty() || pages.empty()) {
        reason = "scene contains no meshlet data suitable for streamasset";
        return false;
    }

    for (size_t renderNodeIndex = 0; renderNodeIndex < desc.scene->renderNodes().size(); ++renderNodeIndex) {
        const RenderNode& renderNode = desc.scene->renderNodes()[renderNodeIndex];
        if (renderNode.renderPrimitiveIndex < 0 ||
            static_cast<size_t>(renderNode.renderPrimitiveIndex) >= primitiveMap.size()) {
            continue;
        }
        const int32_t primitiveIndex = primitiveMap[static_cast<size_t>(renderNode.renderPrimitiveIndex)];
        if (primitiveIndex < 0) {
            continue;
        }

        MeshletStreamInstanceInfo instance;
        instance.renderNodeIndex = static_cast<uint32_t>(renderNodeIndex);
        instance.primitiveIndex = static_cast<uint32_t>(primitiveIndex);
        instance.materialIndex = static_cast<uint32_t>(std::max(renderNode.materialIndex, 0));
        instance.visible = renderNode.visible ? 1u : 0u;
        copyMatrix(renderNode.worldMatrix, instance.worldMatrix);
        instances.push_back(instance);
    }
    if (instances.empty()) {
        for (uint32_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
            MeshletStreamInstanceInfo instance;
            instance.primitiveIndex = primitiveIndex;
            instance.materialIndex = primitives[primitiveIndex].materialIndex;
            instance.visible = 1;
            copyMatrix(float4x4::Identity(), instance.worldMatrix);
            instances.push_back(instance);
        }
    }

    if (!alignStream(stream, kFileAlignment)) {
        reason = "streamasset directory alignment failed";
        return false;
    }
    header.primitiveOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, primitives) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset primitive directory write failed";
        return false;
    }
    header.instanceOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, instances) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset instance directory write failed";
        return false;
    }
    header.geometryOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, geometries) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset geometry directory write failed";
        return false;
    }
    header.lodLevelOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, lodLevels) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset LOD directory write failed";
        return false;
    }
    header.pageInfoOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, pages) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset page directory write failed";
        return false;
    }
    header.pageOffsetTableOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, pageOffsets)) {
        reason = "streamasset page offset table write failed";
        return false;
    }

    header.primitiveCount = static_cast<uint32_t>(primitives.size());
    header.instanceCount = static_cast<uint32_t>(instances.size());
    header.geometryCount = static_cast<uint32_t>(geometries.size());
    header.lodLevelCount = static_cast<uint32_t>(lodLevels.size());
    header.pageCount = static_cast<uint32_t>(pages.size());
    header.fileSize = static_cast<uint64_t>(stream.tellp());
    stream.seekp(0);
    if (!writePod(stream, header)) {
        reason = "streamasset header patch write failed";
        return false;
    }

    return true;
}

} // namespace metallic::scene

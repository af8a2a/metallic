#include "Runtime/Scene/MeshletStreamAsset.h"

#include "json.hpp"
#include "tiny_gltf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <string_view>
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
constexpr std::array<char, 8> kMeshletStreamPartialMagic{'M', 'T', 'L', 'M', 'S', 'P', 'R', 'T'};
constexpr uint32_t kMeshletStreamVersion = 5;
constexpr uint32_t kMeshletStreamPartialVersion = 4;
constexpr uint32_t kMeshletStreamEndian = 0x01020304;
constexpr uint32_t kPayloadMagic = 0x4d535047u; // "GSPM"
constexpr uint32_t kPayloadVersion = 2;
constexpr uint64_t kFileAlignment = 16;
constexpr uint64_t kPageSlotAlignment = 256;
constexpr uint32_t kMeshletClusterMaxVertices = 128;
constexpr uint32_t kMeshletClusterMinTriangles = 32;
constexpr uint32_t kMeshletClusterMaxTriangles = 128;
constexpr uint32_t kMeshletLodGroupSize = 32;
constexpr uint32_t kMeshletStreamNodeWidth = 8;
constexpr float kMeshletClusterFillWeight = 0.5f;
constexpr float kMeshletLodErrorMergePrevious = 1.5f;
constexpr float kMeshletLodErrorMergeAdditive = 0.0f;
constexpr const char* kExtensionNodeVisibility = "KHR_node_visibility";

struct MeshletStreamFileHeader {
    char magic[8] = {};
    uint32_t version = 0;
    uint32_t endian = 0;
    uint64_t fileSize = 0;
    uint64_t sourceFileSize = 0;
    int64_t sourceWriteTime = 0;
    uint64_t sourceDependencyFingerprint = 0;
    uint32_t sourceDependencyCount = 0;
    uint32_t sourceDependencyPathByteCount = 0;
    uint32_t primitiveCount = 0;
    uint32_t instanceCount = 0;
    uint32_t geometryCount = 0;
    uint32_t reserved0 = 0;
    uint32_t lodLevelCount = 0;
    uint32_t groupCount = 0;
    uint32_t clusterRefCount = 0;
    uint32_t nodeCount = 0;
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
    uint64_t groupInfoOffset = 0;
    uint64_t clusterRefOffset = 0;
    uint64_t nodeInfoOffset = 0;
    uint64_t pageInfoOffset = 0;
    uint64_t pageOffsetTableOffset = 0;
    uint64_t sourceDependencyPathOffset = 0;
    float fillWeight = 0.0f;
    float lodErrorMergePrevious = 0.0f;
    float lodErrorMergeAdditive = 0.0f;
    uint32_t reserved1 = 0;
};

struct MeshletStreamPartialFileHeader {
    char magic[8] = {};
    uint32_t version = 0;
    uint32_t endian = 0;
    uint64_t sourceFileSize = 0;
    int64_t sourceWriteTime = 0;
    uint64_t sourceDependencyFingerprint = 0;
    uint64_t payloadWriteOffset = 0;
    uint32_t compressionMode = 0;
    uint32_t nextRenderPrimitiveIndex = 0;
    uint32_t primitiveCount = 0;
    uint32_t geometryCount = 0;
    uint32_t lodLevelCount = 0;
    uint32_t groupCount = 0;
    uint32_t clusterRefCount = 0;
    uint32_t nodeCount = 0;
    uint32_t pageCount = 0;
    uint32_t pageOffsetCount = 0;
    uint32_t geometryEntryCount = 0;
    uint32_t maxPagePayloadBytes = 0;
    uint32_t pagePayloadAlignment = 0;
    uint32_t maxVertices = 0;
    uint32_t minTriangles = 0;
    uint32_t maxTriangles = 0;
    uint32_t lodGroupSize = 0;
    uint32_t reserved = 0;
    float fillWeight = 0.0f;
    float lodErrorMergePrevious = 0.0f;
    float lodErrorMergeAdditive = 0.0f;
    uint32_t reserved1 = 0;
};

struct MeshletStreamPartialGeometryEntry {
    int32_t meshIndex = 0;
    int32_t primitiveIndex = 0;
    uint32_t renderPrimitiveIndex = 0;
    uint32_t streamPrimitiveIndex = 0;
};

static_assert(std::is_trivially_copyable_v<MeshletStreamFileHeader>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPartialFileHeader>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPartialGeometryEntry>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPrimitiveInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamInstanceInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamGeometryInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamLodLevelInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamGroupInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamNodeInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPageInfo>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPayloadHeader>);
static_assert(std::is_trivially_copyable_v<MeshletStreamPayloadCluster>);
static_assert(sizeof(MeshletStreamPayloadHeader) == 96);
static_assert(sizeof(MeshletStreamPayloadCluster) == 32);
static_assert(sizeof(MeshletStreamGroupInfo) == 64);
static_assert(sizeof(MeshletStreamNodeInfo) == 48);

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

bool meshletStreamPartialBuildParamsMatch(const MeshletStreamPartialFileHeader& header)
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

template <typename T>
bool readPod(std::istream& stream, T& value)
{
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return stream.good();
}

template <typename T>
bool readArray(std::istream& stream, uint32_t count, std::vector<T>& values)
{
    values.resize(count);
    if (values.empty()) {
        return true;
    }
    stream.read(
        reinterpret_cast<char*>(values.data()),
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

struct MeshletStreamBuildState {
    MeshletStreamFileHeader header;
    std::vector<MeshletStreamPrimitiveInfo> primitives;
    std::vector<MeshletStreamInstanceInfo> instances;
    std::vector<MeshletStreamGeometryInfo> geometries;
    std::vector<MeshletStreamLodLevelInfo> lodLevels;
    std::vector<MeshletStreamGroupInfo> groups;
    std::vector<uint32_t> clusterRefs;
    std::vector<MeshletStreamNodeInfo> nodes;
    std::vector<MeshletStreamPageInfo> pages;
    std::vector<uint64_t> pageOffsets;
    std::vector<char> sourceDependencyPaths;
    std::vector<MeshletStreamPartialGeometryEntry> partialGeometryEntries;
    std::vector<uint8_t> payload;
    std::vector<uint8_t> storedPayload;
    uint32_t sourceDependencyCount = 0;
    uint32_t nextRenderPrimitiveIndex = 0;
};

struct MeshletStreamPartialBuildContext {
    std::filesystem::path partialPath;
    uint32_t maxNewGeometriesPerInvocation = 0;
    uint32_t newGeometryCount = 0;
    bool paused = false;
};

MeshletStreamFileHeader makeStreamFileHeader(
    const std::filesystem::path& sourcePath,
    uint64_t sourceDependencyFingerprint)
{
    MeshletStreamFileHeader header;
    std::memcpy(header.magic, kMeshletStreamMagic.data(), kMeshletStreamMagic.size());
    header.version = kMeshletStreamVersion;
    header.endian = kMeshletStreamEndian;
    header.sourceFileSize = sourceFileSizeFor(sourcePath);
    header.sourceWriteTime = sourceWriteTimeFor(sourcePath);
    header.sourceDependencyFingerprint = sourceDependencyFingerprint;
    header.pagePayloadAlignment = static_cast<uint32_t>(kPageSlotAlignment);
    header.maxVertices = kMeshletClusterMaxVertices;
    header.minTriangles = kMeshletClusterMinTriangles;
    header.maxTriangles = kMeshletClusterMaxTriangles;
    header.lodGroupSize = kMeshletLodGroupSize;
    header.fillWeight = kMeshletClusterFillWeight;
    header.lodErrorMergePrevious = kMeshletLodErrorMergePrevious;
    header.lodErrorMergeAdditive = kMeshletLodErrorMergeAdditive;
    return header;
}

std::filesystem::path meshletStreamPartialPathFor(const std::filesystem::path& outputPath)
{
    std::filesystem::path path = outputPath;
    path += ".partial";
    return path;
}

bool openStreamAssetBuildFile(
    const std::filesystem::path& outputPath,
    std::ofstream& stream,
    std::string& reason)
{
    std::error_code createError;
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path(), createError);
        if (createError) {
            reason = createError.message();
            return false;
        }
    }

    stream.open(outputPath, std::ios::binary | std::ios::trunc);
    if (!stream) {
        reason = "streamasset output file cannot be opened";
        return false;
    }
    return true;
}

bool openStreamAssetBuildFile(
    const std::filesystem::path& outputPath,
    std::fstream& stream,
    std::string& reason)
{
    std::error_code createError;
    if (outputPath.has_parent_path()) {
        std::filesystem::create_directories(outputPath.parent_path(), createError);
        if (createError) {
            reason = createError.message();
            return false;
        }
    }

    stream.open(outputPath, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
    if (!stream) {
        reason = "streamasset output file cannot be opened";
        return false;
    }
    return true;
}

bool openStreamAssetResumeFile(
    const std::filesystem::path& outputPath,
    uint64_t payloadWriteOffset,
    std::fstream& stream,
    std::string& reason)
{
    if (payloadWriteOffset < sizeof(MeshletStreamFileHeader)) {
        reason = "streamasset partial cache payload offset is invalid";
        return false;
    }

    std::error_code resizeError;
    std::filesystem::resize_file(outputPath, payloadWriteOffset, resizeError);
    if (resizeError) {
        reason = "streamasset partial output truncate failed: " + resizeError.message();
        return false;
    }

    stream.open(outputPath, std::ios::binary | std::ios::in | std::ios::out);
    if (!stream) {
        reason = "streamasset partial output file cannot be reopened";
        return false;
    }
    stream.seekp(static_cast<std::streamoff>(payloadWriteOffset));
    if (!stream) {
        reason = "streamasset partial output seek failed";
        return false;
    }
    return true;
}

bool writeStreamAssetHeaderPlaceholder(
    std::ostream& stream,
    const MeshletStreamFileHeader& header,
    std::string& reason)
{
    if (!writePod(stream, header)) {
        reason = "streamasset header placeholder write failed";
        return false;
    }
    return true;
}

void sortStreamNodesSpatially(std::vector<MeshletStreamNodeInfo>& nodes)
{
    if (nodes.size() < 2) {
        return;
    }

    float minCenter[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    float maxCenter[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };
    for (const MeshletStreamNodeInfo& node : nodes) {
        for (uint32_t axis = 0; axis < 3; ++axis) {
            minCenter[axis] = std::min(minCenter[axis], node.boundsCenterRadius[axis]);
            maxCenter[axis] = std::max(maxCenter[axis], node.boundsCenterRadius[axis]);
        }
    }
    uint32_t sortAxis = 0;
    for (uint32_t axis = 1; axis < 3; ++axis) {
        if (maxCenter[axis] - minCenter[axis] > maxCenter[sortAxis] - minCenter[sortAxis]) {
            sortAxis = axis;
        }
    }
    std::stable_sort(
        nodes.begin(),
        nodes.end(),
        [sortAxis](const MeshletStreamNodeInfo& lhs, const MeshletStreamNodeInfo& rhs) {
            return lhs.boundsCenterRadius[sortAxis] < rhs.boundsCenterRadius[sortAxis];
        });
}

MeshletStreamNodeInfo makeStreamInteriorNode(
    uint32_t primitiveIndex,
    uint32_t lodLevel,
    uint32_t childOffset,
    std::span<const MeshletStreamNodeInfo> children)
{
    MeshletStreamNodeInfo node;
    node.primitiveIndex = primitiveIndex;
    node.childOffset = childOffset;
    node.childCount = static_cast<uint32_t>(children.size());
    node.groupIndex = kMeshletStreamInvalidGroupIndex;
    node.lodLevel = lodLevel;

    float minBounds[3] = {
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    float maxBounds[3] = {
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };
    for (const MeshletStreamNodeInfo& child : children) {
        const float radius = std::max(child.boundsCenterRadius[3], 0.0f);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            minBounds[axis] = std::min(minBounds[axis], child.boundsCenterRadius[axis] - radius);
            maxBounds[axis] = std::max(maxBounds[axis], child.boundsCenterRadius[axis] + radius);
        }
        node.maxQuadricError = std::max(node.maxQuadricError, child.maxQuadricError);
    }
    for (uint32_t axis = 0; axis < 3; ++axis) {
        node.boundsCenterRadius[axis] = (minBounds[axis] + maxBounds[axis]) * 0.5f;
    }
    float enclosingRadius = 0.0f;
    for (const MeshletStreamNodeInfo& child : children) {
        const float dx = child.boundsCenterRadius[0] - node.boundsCenterRadius[0];
        const float dy = child.boundsCenterRadius[1] - node.boundsCenterRadius[1];
        const float dz = child.boundsCenterRadius[2] - node.boundsCenterRadius[2];
        enclosingRadius = std::max(
            enclosingRadius,
            std::sqrt(dx * dx + dy * dy + dz * dz) + child.boundsCenterRadius[3]);
    }
    node.boundsCenterRadius[3] = enclosingRadius;
    return node;
}

bool buildStreamLodNodeTree(
    MeshletStreamBuildState& state,
    const MeshletStreamPrimitiveInfo& primitive,
    const MeshletStreamLodLevelInfo& lod,
    MeshletStreamNodeInfo& outRoot,
    std::string& reason)
{
    if (lod.pageCount == 0 ||
        lod.pageOffset < primitive.pageOffset ||
        lod.pageOffset - primitive.pageOffset > primitive.groupCount ||
        lod.pageCount > primitive.groupCount - (lod.pageOffset - primitive.pageOffset)) {
        reason = "streamasset LOD cannot be mapped to hierarchy groups";
        return false;
    }

    std::vector<MeshletStreamNodeInfo> current;
    current.reserve(lod.pageCount);
    const uint32_t firstGroup = primitive.groupOffset + (lod.pageOffset - primitive.pageOffset);
    for (uint32_t groupChild = 0; groupChild < lod.pageCount; ++groupChild) {
        const uint32_t groupIndex = firstGroup + groupChild;
        if (groupIndex >= state.groups.size()) {
            reason = "streamasset hierarchy group index is out of range";
            return false;
        }
        const MeshletStreamGroupInfo& group = state.groups[groupIndex];
        if (group.primitiveIndex != lod.primitiveIndex || group.lodLevel != lod.lodLevel) {
            reason = "streamasset hierarchy group does not match its LOD";
            return false;
        }
        MeshletStreamNodeInfo leaf;
        leaf.primitiveIndex = lod.primitiveIndex;
        leaf.groupIndex = groupIndex;
        std::copy(
            std::begin(group.boundsCenterRadius),
            std::end(group.boundsCenterRadius),
            std::begin(leaf.boundsCenterRadius));
        leaf.maxQuadricError = group.maxQuadricError;
        leaf.lodLevel = lod.lodLevel;
        current.push_back(leaf);
    }

    std::vector<MeshletStreamNodeInfo> localNodes;
    while (current.size() > 1) {
        sortStreamNodesSpatially(current);
        const uint32_t childOffset = static_cast<uint32_t>(localNodes.size());
        localNodes.insert(localNodes.end(), current.begin(), current.end());

        std::vector<MeshletStreamNodeInfo> parents;
        parents.reserve((current.size() + kMeshletStreamNodeWidth - 1u) / kMeshletStreamNodeWidth);
        for (uint32_t firstChild = 0; firstChild < current.size(); firstChild += kMeshletStreamNodeWidth) {
            const uint32_t childCount = std::min<uint32_t>(
                kMeshletStreamNodeWidth,
                static_cast<uint32_t>(current.size()) - firstChild);
            parents.push_back(makeStreamInteriorNode(
                lod.primitiveIndex,
                lod.lodLevel,
                childOffset + firstChild,
                std::span<const MeshletStreamNodeInfo>(current.data() + firstChild, childCount)));
        }
        current = std::move(parents);
    }

    const uint32_t globalNodeOffset = static_cast<uint32_t>(state.nodes.size());
    for (MeshletStreamNodeInfo& node : localNodes) {
        if (node.childCount != 0) {
            node.childOffset += globalNodeOffset;
        }
        state.nodes.push_back(node);
    }
    outRoot = current.front();
    if (outRoot.childCount != 0) {
        outRoot.childOffset += globalNodeOffset;
    }
    return true;
}

bool appendStreamPrimitiveHierarchy(
    MeshletStreamBuildState& state,
    MeshletStreamPrimitiveInfo& primitive,
    std::string& reason)
{
    if (primitive.lodLevelCount == 0 || primitive.lodLevelCount > 32) {
        reason = "streamasset primitive hierarchy requires between 1 and 32 LOD roots";
        return false;
    }

    primitive.nodeOffset = static_cast<uint32_t>(state.nodes.size());
    state.nodes.resize(state.nodes.size() + 1u + primitive.lodLevelCount);
    const uint32_t lodRootOffset = primitive.nodeOffset + 1u;
    for (uint32_t lodChild = 0; lodChild < primitive.lodLevelCount; ++lodChild) {
        const MeshletStreamLodLevelInfo& lod = state.lodLevels[primitive.lodLevelOffset + lodChild];
        MeshletStreamNodeInfo root;
        if (!buildStreamLodNodeTree(state, primitive, lod, root, reason)) {
            return false;
        }
        state.nodes[lodRootOffset + lodChild] = root;
    }

    MeshletStreamNodeInfo& root = state.nodes[primitive.nodeOffset];
    root = makeStreamInteriorNode(
        state.lodLevels[primitive.lodLevelOffset].primitiveIndex,
        kMeshletStreamInvalidNodeIndex,
        lodRootOffset,
        std::span<const MeshletStreamNodeInfo>(state.nodes.data() + lodRootOffset, primitive.lodLevelCount));
    primitive.nodeCount = static_cast<uint32_t>(state.nodes.size()) - primitive.nodeOffset;
    return true;
}

bool appendStreamPrimitivePages(
    std::ostream& stream,
    MeshletStreamBuildState& state,
    const RenderPrimitive& primitive,
    uint32_t renderPrimitiveIndex,
    MeshletStreamPayloadCompression compressionMode,
    int32_t& outPrimitiveIndex,
    std::string& reason)
{
    outPrimitiveIndex = kInvalidSceneIndex;
    if (primitive.mode != 4 || primitive.positions.empty()) {
        return true;
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
        return true;
    }

    const uint32_t primitiveIndex = static_cast<uint32_t>(state.primitives.size());
    MeshletStreamPrimitiveInfo primitiveInfo;
    primitiveInfo.renderPrimitiveIndex = renderPrimitiveIndex;
    primitiveInfo.materialIndex = static_cast<uint32_t>(std::max(primitive.materialIndex, 0));
    primitiveInfo.lodLevelOffset = static_cast<uint32_t>(state.lodLevels.size());
    primitiveInfo.pageOffset = static_cast<uint32_t>(state.pages.size());
    primitiveInfo.groupOffset = static_cast<uint32_t>(state.groups.size());
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
            lodInfo.pageOffset = static_cast<uint32_t>(state.pages.size());
            lodInfo.minBoundingSphereRadius = sourceLevel.minBoundingSphereRadius;
            lodInfo.minMaxQuadricError = sourceLevel.minMaxQuadricError;

            for (uint32_t groupChild = 0; groupChild < sourceLevel.groupCount; ++groupChild) {
                const uint32_t groupIndex = sourceLevel.groupOffset + groupChild;
                const MeshletLodGroup& group = primitive.meshletLodGroups[groupIndex];
                if (group.clusterCount == 0 ||
                    group.clusterCount > kMeshletLodGroupSize ||
                    group.clusterOffset > primitive.meshletLodClusters.size() ||
                    group.clusterCount > primitive.meshletLodClusters.size() - group.clusterOffset) {
                    reason = "primitive meshlet LOD group has an invalid cluster range";
                    return false;
                }
                const uint32_t pageIndex = static_cast<uint32_t>(state.pages.size());
                PagePayloadBuildInput pageInput{
                    .primitive = &primitive,
                    .clusters = &primitive.meshletLodClusters,
                    .vertices = &primitive.meshletLodVertices,
                    .triangles = &primitive.meshletLodTriangles,
                    .firstCluster = group.clusterOffset,
                    .clusterCount = group.clusterCount,
                    .primitiveIndex = primitiveIndex,
                    .lodLevel = lodLevelIndex,
                    .lodGroupIndex = primitiveInfo.groupOffset + groupIndex,
                    .materialIndex = primitiveInfo.materialIndex,
                    .bounds = makeBounds(makeStreamBounds(group.bounds)),
                    .maxQuadricError = group.maxQuadricError,
                };
                MeshletStreamPageInfo pageInfo;
                if (!buildPagePayload(pageInput, state.payload, pageInfo, reason)) {
                    return false;
                }
                if (!encodePayloadForStorage(
                        state.payload,
                        compressionMode,
                        state.storedPayload,
                        pageInfo,
                        reason)) {
                    return false;
                }
                uint64_t payloadOffset = 0;
                if (!writePayload(stream, state.storedPayload, payloadOffset)) {
                    reason = "streamasset page payload write failed";
                    return false;
                }
                pageInfo.payloadOffset = payloadOffset;
                state.header.maxPagePayloadBytes = std::max<uint32_t>(
                    state.header.maxPagePayloadBytes,
                    static_cast<uint32_t>(pageInfo.uncompressedSize));
                lodInfo.clusterCount += pageInfo.clusterCount;
                state.pages.push_back(pageInfo);
                state.pageOffsets.push_back(payloadOffset);

                MeshletStreamGroupInfo groupInfo;
                groupInfo.primitiveIndex = primitiveIndex;
                groupInfo.pageIndex = pageIndex;
                groupInfo.lodLevel = lodLevelIndex;
                groupInfo.clusterRefOffset = static_cast<uint32_t>(state.clusterRefs.size());
                groupInfo.clusterCount = group.clusterCount;
                groupInfo.boundsCenterRadius[0] = group.boundingSphereCenter.x;
                groupInfo.boundsCenterRadius[1] = group.boundingSphereCenter.y;
                groupInfo.boundsCenterRadius[2] = group.boundingSphereCenter.z;
                groupInfo.boundsCenterRadius[3] = group.boundingSphereRadius;
                groupInfo.maxQuadricError = group.maxQuadricError;
                for (uint32_t clusterChild = 0; clusterChild < group.clusterCount; ++clusterChild) {
                    const MeshletCluster& cluster =
                        primitive.meshletLodClusters[group.clusterOffset + clusterChild];
                    if (cluster.lodGroupIndex != static_cast<int32_t>(groupIndex) ||
                        cluster.lodGroupChildIndex != clusterChild ||
                        (cluster.refinedGroupIndex != kInvalidSceneIndex &&
                            (cluster.refinedGroupIndex < 0 ||
                                cluster.refinedGroupIndex >= static_cast<int32_t>(groupIndex) ||
                                static_cast<size_t>(cluster.refinedGroupIndex) >=
                                    primitive.meshletLodGroups.size()))) {
                        reason = "primitive meshlet LOD cluster has invalid DAG metadata";
                        return false;
                    }
                    state.clusterRefs.push_back(
                        cluster.refinedGroupIndex == kInvalidSceneIndex
                            ? kMeshletStreamInvalidGroupIndex
                            : primitiveInfo.groupOffset + static_cast<uint32_t>(cluster.refinedGroupIndex));
                }
                state.groups.push_back(groupInfo);
            }

            lodInfo.pageCount = static_cast<uint32_t>(state.pages.size()) - lodInfo.pageOffset;
            if (lodInfo.pageCount > 0 && lodInfo.pageCount < bestFallbackPageCount) {
                primitiveInfo.fallbackPageOffset = lodInfo.pageOffset;
                primitiveInfo.fallbackPageCount = lodInfo.pageCount;
                primitiveInfo.fallbackGroupOffset =
                    primitiveInfo.groupOffset + sourceLevel.groupOffset;
                primitiveInfo.fallbackGroupCount = sourceLevel.groupCount;
                bestFallbackPageCount = lodInfo.pageCount;
            }
            state.lodLevels.push_back(lodInfo);
        }
    } else {
        MeshletStreamLodLevelInfo lodInfo;
        lodInfo.primitiveIndex = primitiveIndex;
        lodInfo.lodLevel = 0;
        lodInfo.pageOffset = static_cast<uint32_t>(state.pages.size());

        for (uint32_t firstCluster = 0; firstCluster < primitive.meshletClusters.size();) {
            const uint32_t clusterCount = std::min<uint32_t>(
                kMeshletLodGroupSize,
                static_cast<uint32_t>(primitive.meshletClusters.size()) - firstCluster);
            const uint32_t pageIndex = static_cast<uint32_t>(state.pages.size());
            PagePayloadBuildInput pageInput{
                .primitive = &primitive,
                .clusters = &primitive.meshletClusters,
                .vertices = &primitive.meshletVertices,
                .triangles = &primitive.meshletTriangles,
                .firstCluster = firstCluster,
                .clusterCount = clusterCount,
                .primitiveIndex = primitiveIndex,
                .lodLevel = 0,
                .lodGroupIndex = static_cast<uint32_t>(state.groups.size()),
                .materialIndex = primitiveInfo.materialIndex,
                .bounds = primitive.localBounds,
            };
            MeshletStreamPageInfo pageInfo;
            if (!buildPagePayload(pageInput, state.payload, pageInfo, reason)) {
                return false;
            }
            if (!encodePayloadForStorage(
                    state.payload,
                    compressionMode,
                    state.storedPayload,
                    pageInfo,
                    reason)) {
                return false;
            }
            uint64_t payloadOffset = 0;
            if (!writePayload(stream, state.storedPayload, payloadOffset)) {
                reason = "streamasset fallback page payload write failed";
                return false;
            }
            pageInfo.payloadOffset = payloadOffset;
            state.header.maxPagePayloadBytes = std::max<uint32_t>(
                state.header.maxPagePayloadBytes,
                static_cast<uint32_t>(pageInfo.uncompressedSize));
            lodInfo.clusterCount += pageInfo.clusterCount;
            state.pages.push_back(pageInfo);
            state.pageOffsets.push_back(payloadOffset);

            MeshletStreamGroupInfo groupInfo;
            groupInfo.primitiveIndex = primitiveIndex;
            groupInfo.pageIndex = pageIndex;
            groupInfo.lodLevel = 0;
            groupInfo.clusterRefOffset = static_cast<uint32_t>(state.clusterRefs.size());
            groupInfo.clusterCount = clusterCount;
            const float3 center = primitive.localBounds.center();
            groupInfo.boundsCenterRadius[0] = center.x;
            groupInfo.boundsCenterRadius[1] = center.y;
            groupInfo.boundsCenterRadius[2] = center.z;
            groupInfo.boundsCenterRadius[3] = primitive.localBounds.radius();
            groupInfo.maxQuadricError = kMeshletStreamTerminalGroupError;
            state.clusterRefs.insert(
                state.clusterRefs.end(),
                clusterCount,
                kMeshletStreamInvalidGroupIndex);
            state.groups.push_back(groupInfo);
            firstCluster += clusterCount;
        }
        lodInfo.pageCount = static_cast<uint32_t>(state.pages.size()) - lodInfo.pageOffset;
        primitiveInfo.fallbackPageOffset = lodInfo.pageOffset;
        primitiveInfo.fallbackPageCount = lodInfo.pageCount;
        primitiveInfo.fallbackGroupOffset = primitiveInfo.groupOffset;
        primitiveInfo.fallbackGroupCount = lodInfo.pageCount;
        state.lodLevels.push_back(lodInfo);
    }

    primitiveInfo.lodLevelCount = static_cast<uint32_t>(state.lodLevels.size()) - primitiveInfo.lodLevelOffset;
    primitiveInfo.pageCount = static_cast<uint32_t>(state.pages.size()) - primitiveInfo.pageOffset;
    primitiveInfo.groupCount = static_cast<uint32_t>(state.groups.size()) - primitiveInfo.groupOffset;
    if (primitiveInfo.pageCount == 0 ||
        primitiveInfo.groupCount == 0 ||
        primitiveInfo.pageCount != primitiveInfo.groupCount ||
        primitiveInfo.fallbackPageCount == 0 ||
        primitiveInfo.fallbackGroupCount == 0) {
        reason = "primitive meshlet stream directories are incomplete";
        return false;
    }
    if (!appendStreamPrimitiveHierarchy(state, primitiveInfo, reason) || primitiveInfo.nodeCount == 0) {
        return false;
    }

    uint64_t payloadFileBegin = std::numeric_limits<uint64_t>::max();
    uint64_t payloadFileEnd = 0;
    for (uint32_t page = 0; page < primitiveInfo.pageCount; ++page) {
        const MeshletStreamPageInfo& pageInfo = state.pages[primitiveInfo.pageOffset + page];
        payloadFileBegin = std::min(payloadFileBegin, pageInfo.payloadOffset);
        payloadFileEnd = std::max(payloadFileEnd, pageInfo.payloadOffset + pageInfo.payloadSize);
    }
    state.geometries.push_back(MeshletStreamGeometryInfo{
        .primitiveIndex = primitiveIndex,
        .renderPrimitiveIndex = primitiveInfo.renderPrimitiveIndex,
        .pageOffset = primitiveInfo.pageOffset,
        .pageCount = primitiveInfo.pageCount,
        .pagePayloadOffsetTableOffset = primitiveInfo.pageOffset,
        .pagePayloadOffsetTableCount = primitiveInfo.pageCount,
        .payloadFileOffset = payloadFileBegin == std::numeric_limits<uint64_t>::max() ? 0u : payloadFileBegin,
        .payloadFileSize = payloadFileEnd > payloadFileBegin ? payloadFileEnd - payloadFileBegin : 0u,
    });
    state.primitives.push_back(primitiveInfo);
    outPrimitiveIndex = static_cast<int32_t>(primitiveIndex);
    return true;
}

bool finalizeStreamAssetBuild(std::ostream& stream, MeshletStreamBuildState& state, std::string& reason)
{
    if (state.primitives.empty() || state.pages.empty()) {
        reason = "scene contains no meshlet data suitable for streamasset";
        return false;
    }

    if (state.instances.empty()) {
        for (uint32_t primitiveIndex = 0; primitiveIndex < state.primitives.size(); ++primitiveIndex) {
            MeshletStreamInstanceInfo instance;
            instance.primitiveIndex = primitiveIndex;
            instance.materialIndex = state.primitives[primitiveIndex].materialIndex;
            instance.visible = 1;
            copyMatrix(float4x4::Identity(), instance.worldMatrix);
            state.instances.push_back(instance);
        }
    }

    if (!alignStream(stream, kFileAlignment)) {
        reason = "streamasset directory alignment failed";
        return false;
    }
    state.header.primitiveOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.primitives) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset primitive directory write failed";
        return false;
    }
    state.header.instanceOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.instances) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset instance directory write failed";
        return false;
    }
    state.header.geometryOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.geometries) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset geometry directory write failed";
        return false;
    }
    state.header.lodLevelOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.lodLevels) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset LOD directory write failed";
        return false;
    }
    state.header.groupInfoOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.groups) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset group directory write failed";
        return false;
    }
    state.header.clusterRefOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.clusterRefs) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset cluster-ref directory write failed";
        return false;
    }
    state.header.nodeInfoOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.nodes) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset hierarchy node directory write failed";
        return false;
    }
    state.header.pageInfoOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.pages) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset page directory write failed";
        return false;
    }
    state.header.pageOffsetTableOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.pageOffsets) || !alignStream(stream, kFileAlignment)) {
        reason = "streamasset page offset table write failed";
        return false;
    }
    state.header.sourceDependencyPathOffset = static_cast<uint64_t>(stream.tellp());
    if (!writeArray(stream, state.sourceDependencyPaths)) {
        reason = "streamasset source dependency directory write failed";
        return false;
    }

    state.header.sourceDependencyCount = state.sourceDependencyCount;
    state.header.sourceDependencyPathByteCount =
        static_cast<uint32_t>(state.sourceDependencyPaths.size());
    state.header.primitiveCount = static_cast<uint32_t>(state.primitives.size());
    state.header.instanceCount = static_cast<uint32_t>(state.instances.size());
    state.header.geometryCount = static_cast<uint32_t>(state.geometries.size());
    state.header.lodLevelCount = static_cast<uint32_t>(state.lodLevels.size());
    state.header.groupCount = static_cast<uint32_t>(state.groups.size());
    state.header.clusterRefCount = static_cast<uint32_t>(state.clusterRefs.size());
    state.header.nodeCount = static_cast<uint32_t>(state.nodes.size());
    state.header.pageCount = static_cast<uint32_t>(state.pages.size());
    state.header.fileSize = static_cast<uint64_t>(stream.tellp());
    stream.seekp(0);
    if (!writePod(stream, state.header)) {
        reason = "streamasset header patch write failed";
        return false;
    }
    return true;
}

bool validGltfIndex(int32_t index, size_t size)
{
    return index >= 0 && static_cast<size_t>(index) < size;
}

std::string lowerExtensionForStreamBuilder(const std::filesystem::path& path)
{
    std::string extension = path.extension().string();
    std::transform(
        extension.begin(),
        extension.end(),
        extension.begin(),
        [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension;
}

bool streamBuilderSupportsRequiredExtension(std::string_view extension)
{
    return extension == "KHR_lights_punctual" ||
        extension == kExtensionNodeVisibility ||
        extension == "KHR_materials_diffuse_transmission" ||
        extension == "KHR_materials_emissive_strength" ||
        extension == "KHR_materials_ior" ||
        extension == "KHR_materials_transmission" ||
        extension == "KHR_materials_volume" ||
        extension == "KHR_mesh_quantization" ||
        extension == "KHR_texture_transform";
}

float3 makeFloat3ForStreamBuilder(const std::vector<double>& values, const float3& fallback)
{
    if (values.size() < 3) {
        return fallback;
    }
    return float3(
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]));
}

float4 makeQuaternionForStreamBuilder(const std::vector<double>& values)
{
    if (values.size() < 4) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return float4(
        static_cast<float>(values[0]),
        static_cast<float>(values[1]),
        static_cast<float>(values[2]),
        static_cast<float>(values[3]));
}

float4x4 makeMatrixFromGltfForStreamBuilder(const std::vector<double>& values)
{
    if (values.size() != 16) {
        return float4x4::Identity();
    }

    float4x4 matrix;
    for (size_t index = 0; index < values.size(); ++index) {
        matrix.a[index] = static_cast<float>(values[index]);
    }
    return matrix;
}

float4x4 makeNodeLocalMatrixForStreamBuilder(const tinygltf::Node& node)
{
    if (node.matrix.size() == 16) {
        return makeMatrixFromGltfForStreamBuilder(node.matrix);
    }

    float4x4 translation;
    translation.SetupByTranslation(makeFloat3ForStreamBuilder(node.translation, float3(0.0f, 0.0f, 0.0f)));

    float4x4 rotation;
    rotation.SetupByQuaternion(makeQuaternionForStreamBuilder(node.rotation));

    float4x4 scale;
    scale.SetupByScale(makeFloat3ForStreamBuilder(node.scale, float3(1.0f, 1.0f, 1.0f)));

    return translation * rotation * scale;
}

bool readNodeVisibilityForStreamBuilder(const tinygltf::Node& node)
{
    const auto extension = node.extensions.find(kExtensionNodeVisibility);
    if (extension == node.extensions.end()) {
        return true;
    }

    const tinygltf::Value& value = extension->second;
    if (!value.IsObject() || !value.Has("visible") || !value.Get("visible").IsBool()) {
        return true;
    }
    return value.Get("visible").Get<bool>();
}

Bounds accessorBoundsForStreamBuilder(const tinygltf::Accessor& accessor)
{
    Bounds bounds;
    if (accessor.minValues.size() < 3 || accessor.maxValues.size() < 3) {
        return bounds;
    }

    bounds.include(makeFloat3ForStreamBuilder(accessor.minValues, float3(0.0f, 0.0f, 0.0f)));
    bounds.include(makeFloat3ForStreamBuilder(accessor.maxValues, float3(0.0f, 0.0f, 0.0f)));
    return bounds;
}

Bounds boundsFromPositionsForStreamBuilder(std::span<const float3> positions)
{
    Bounds bounds;
    for (const float3& position : positions) {
        bounds.include(position);
    }
    return bounds;
}

struct StreamGltfSource {
    tinygltf::Model model;
    std::filesystem::path directory;
    std::vector<uint64_t> bufferByteLengths;
    std::vector<std::string> externalBufferUris;
    std::vector<bool> nodeVisibility;
    MeshletStreamAssetOfflineBuildStats* stats = nullptr;
    bool rangeReadExternalBuffers = false;
};

std::filesystem::path pathFromUtf8ForStreamBuilder(std::string_view value)
{
#if defined(__cpp_char8_t)
    return std::filesystem::path(std::u8string(
        reinterpret_cast<const char8_t*>(value.data()),
        value.size()));
#else
    return std::filesystem::path(value);
#endif
}

bool addWithin(uint64_t lhs, uint64_t rhs, uint64_t limit, uint64_t& result)
{
    if (lhs > limit || rhs > limit - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool readAccessorRangeForStreamBuilder(
    const StreamGltfSource& source,
    const tinygltf::Accessor& accessor,
    size_t elementByteSize,
    std::vector<uint8_t>& outBytes,
    size_t& outStride,
    std::string& reason)
{
    outBytes.clear();
    outStride = 0;
    if (accessor.sparse.isSparse) {
        reason = "streamasset builder does not support sparse accessors";
        return false;
    }
    if (!validGltfIndex(accessor.bufferView, source.model.bufferViews.size())) {
        reason = "streamasset builder accessor has an invalid bufferView";
        return false;
    }
    const tinygltf::BufferView& bufferView =
        source.model.bufferViews[static_cast<size_t>(accessor.bufferView)];
    if (!validGltfIndex(bufferView.buffer, source.model.buffers.size()) ||
        static_cast<size_t>(bufferView.buffer) >= source.bufferByteLengths.size()) {
        reason = "streamasset builder bufferView has an invalid buffer";
        return false;
    }

    const int stride = accessor.ByteStride(bufferView);
    if (stride <= 0 || static_cast<size_t>(stride) < elementByteSize) {
        reason = "streamasset builder accessor has an invalid byte stride";
        return false;
    }
    if (accessor.count == 0) {
        reason = "streamasset builder accessor is empty";
        return false;
    }

    const uint64_t countMinusOne = static_cast<uint64_t>(accessor.count - 1u);
    if (countMinusOne > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(stride)) {
        reason = "streamasset builder accessor byte range overflows";
        return false;
    }
    const uint64_t stridedBytes = countMinusOne * static_cast<uint64_t>(stride);
    uint64_t rangeByteSize = 0;
    if (!addWithin(stridedBytes, elementByteSize, std::numeric_limits<uint64_t>::max(), rangeByteSize)) {
        reason = "streamasset builder accessor byte range overflows";
        return false;
    }
    uint64_t rangeOffset = 0;
    if (!addWithin(
            static_cast<uint64_t>(bufferView.byteOffset),
            static_cast<uint64_t>(accessor.byteOffset),
            std::numeric_limits<uint64_t>::max(),
            rangeOffset)) {
        reason = "streamasset builder accessor byte offset overflows";
        return false;
    }
    const uint64_t bufferByteLength = source.bufferByteLengths[static_cast<size_t>(bufferView.buffer)];
    if (rangeOffset > bufferByteLength || rangeByteSize > bufferByteLength - rangeOffset) {
        reason = "streamasset builder accessor byte range exceeds its buffer";
        return false;
    }
    uint64_t bufferViewEnd = 0;
    if (!addWithin(
            static_cast<uint64_t>(bufferView.byteOffset),
            static_cast<uint64_t>(bufferView.byteLength),
            std::numeric_limits<uint64_t>::max(),
            bufferViewEnd)) {
        reason = "streamasset builder bufferView byte range overflows";
        return false;
    }
    if (rangeOffset < bufferView.byteOffset ||
        rangeOffset > bufferViewEnd ||
        rangeByteSize > bufferViewEnd - rangeOffset) {
        reason = "streamasset builder accessor byte range exceeds its bufferView";
        return false;
    }
    if (rangeByteSize > std::numeric_limits<size_t>::max()) {
        reason = "streamasset builder accessor byte range exceeds host address space";
        return false;
    }

    outBytes.resize(static_cast<size_t>(rangeByteSize));
    const tinygltf::Buffer& buffer = source.model.buffers[static_cast<size_t>(bufferView.buffer)];
    if (!source.rangeReadExternalBuffers || !buffer.data.empty()) {
        if (rangeOffset > buffer.data.size() || rangeByteSize > buffer.data.size() - rangeOffset) {
            reason = "streamasset builder accessor byte range exceeds loaded buffer data";
            outBytes.clear();
            return false;
        }
        std::memcpy(outBytes.data(), buffer.data.data() + rangeOffset, outBytes.size());
    } else {
        std::string decodedUri;
        if (!tinygltf::URIDecode(buffer.uri, &decodedUri, nullptr) || decodedUri.empty()) {
            reason = "streamasset builder failed to decode external buffer URI";
            outBytes.clear();
            return false;
        }
        const std::filesystem::path bufferPath =
            source.directory / pathFromUtf8ForStreamBuilder(decodedUri);
        std::ifstream file(bufferPath, std::ios::binary);
        if (!file) {
            reason = "streamasset builder cannot open external buffer: " + bufferPath.string();
            outBytes.clear();
            return false;
        }
        if (rangeOffset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            reason = "streamasset builder external buffer offset exceeds stream limits";
            outBytes.clear();
            return false;
        }
        file.seekg(static_cast<std::streamoff>(rangeOffset));
        if (!file) {
            reason = "streamasset builder failed to seek external buffer: " + bufferPath.string();
            outBytes.clear();
            return false;
        }
        if (outBytes.size() > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
            reason = "streamasset builder accessor range exceeds stream read limits";
            outBytes.clear();
            return false;
        }
        file.read(reinterpret_cast<char*>(outBytes.data()), static_cast<std::streamsize>(outBytes.size()));
        if (!file || static_cast<size_t>(file.gcount()) != outBytes.size()) {
            reason = "streamasset builder failed to read accessor range from: " + bufferPath.string();
            outBytes.clear();
            return false;
        }
        if (source.stats != nullptr) {
            source.stats->accessorRangeReadBytes = rangeByteSize >
                    std::numeric_limits<uint64_t>::max() - source.stats->accessorRangeReadBytes
                ? std::numeric_limits<uint64_t>::max()
                : source.stats->accessorRangeReadBytes + rangeByteSize;
            source.stats->maxAccessorRangeReadBytes = std::max(
                source.stats->maxAccessorRangeReadBytes,
                rangeByteSize);
            ++source.stats->accessorRangeReadCount;
        }
    }

    outStride = static_cast<size_t>(stride);
    return true;
}

std::vector<float3> readFloat3AccessorForStreamBuilder(
    const StreamGltfSource& source,
    const tinygltf::Accessor& accessor,
    std::string& reason)
{
    std::vector<float3> values;
    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return values;
    }

    std::vector<uint8_t> bytes;
    size_t stride = 0;
    if (!readAccessorRangeForStreamBuilder(source, accessor, sizeof(float) * 3u, bytes, stride, reason)) {
        return values;
    }

    values.reserve(accessor.count);
    for (size_t index = 0; index < accessor.count; ++index) {
        float components[3] = {};
        std::memcpy(components, bytes.data() + stride * index, sizeof(components));
        values.emplace_back(components[0], components[1], components[2]);
    }
    return values;
}

std::vector<float2> readFloat2AccessorForStreamBuilder(
    const StreamGltfSource& source,
    const tinygltf::Accessor& accessor,
    std::string& reason)
{
    std::vector<float2> values;
    if (accessor.type != TINYGLTF_TYPE_VEC2 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
        return values;
    }

    std::vector<uint8_t> bytes;
    size_t stride = 0;
    if (!readAccessorRangeForStreamBuilder(source, accessor, sizeof(float) * 2u, bytes, stride, reason)) {
        return values;
    }

    values.reserve(accessor.count);
    for (size_t index = 0; index < accessor.count; ++index) {
        float components[2] = {};
        std::memcpy(components, bytes.data() + stride * index, sizeof(components));
        values.emplace_back(components[0], components[1]);
    }
    return values;
}

std::vector<uint32_t> readIndexAccessorForStreamBuilder(
    const StreamGltfSource& source,
    const tinygltf::Accessor& accessor,
    std::string& reason)
{
    std::vector<uint32_t> indices;
    if (accessor.type != TINYGLTF_TYPE_SCALAR) {
        return indices;
    }

    size_t componentSize = 0;
    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        componentSize = sizeof(uint8_t);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        componentSize = sizeof(uint16_t);
        break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        componentSize = sizeof(uint32_t);
        break;
    default:
        return indices;
    }

    std::vector<uint8_t> bytes;
    size_t stride = 0;
    if (!readAccessorRangeForStreamBuilder(source, accessor, componentSize, bytes, stride, reason)) {
        return indices;
    }

    indices.reserve(accessor.count);
    for (size_t index = 0; index < accessor.count; ++index) {
        const uint8_t* data = bytes.data() + stride * index;

        switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
            indices.push_back(*data);
            break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: {
            uint16_t value = 0;
            std::memcpy(&value, data, sizeof(value));
            indices.push_back(value);
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: {
            uint32_t value = 0;
            std::memcpy(&value, data, sizeof(value));
            indices.push_back(value);
            break;
        }
        default:
            break;
        }
    }
    return indices;
}

uint64_t triangleCountForStreamPrimitive(int32_t mode, uint64_t elementCount)
{
    switch (mode) {
    case TINYGLTF_MODE_TRIANGLES:
        return elementCount / 3u;
    case TINYGLTF_MODE_TRIANGLE_STRIP:
    case TINYGLTF_MODE_TRIANGLE_FAN:
        return elementCount >= 3u ? elementCount - 2u : 0u;
    default:
        return 0;
    }
}

std::vector<double> jsonNumberArrayForStreamBuilder(const nlohmann::json& value)
{
    std::vector<double> result;
    if (!value.is_array()) {
        return result;
    }
    result.reserve(value.size());
    for (const nlohmann::json& component : value) {
        if (!component.is_number()) {
            result.clear();
            return result;
        }
        result.push_back(component.get<double>());
    }
    return result;
}

std::vector<int> jsonIntArrayForStreamBuilder(const nlohmann::json& value)
{
    std::vector<int> result;
    if (!value.is_array()) {
        return result;
    }
    result.reserve(value.size());
    for (const nlohmann::json& component : value) {
        if (!component.is_number_integer()) {
            result.clear();
            return result;
        }
        result.push_back(component.get<int>());
    }
    return result;
}

int accessorTypeForStreamBuilder(std::string_view type)
{
    if (type == "SCALAR") {
        return TINYGLTF_TYPE_SCALAR;
    }
    if (type == "VEC2") {
        return TINYGLTF_TYPE_VEC2;
    }
    if (type == "VEC3") {
        return TINYGLTF_TYPE_VEC3;
    }
    if (type == "VEC4") {
        return TINYGLTF_TYPE_VEC4;
    }
    if (type == "MAT2") {
        return TINYGLTF_TYPE_MAT2;
    }
    if (type == "MAT3") {
        return TINYGLTF_TYPE_MAT3;
    }
    if (type == "MAT4") {
        return TINYGLTF_TYPE_MAT4;
    }
    return -1;
}

bool isDataUriForStreamBuilder(std::string_view uri)
{
    return uri.size() >= 5 &&
        std::tolower(static_cast<unsigned char>(uri[0])) == 'd' &&
        std::tolower(static_cast<unsigned char>(uri[1])) == 'a' &&
        std::tolower(static_cast<unsigned char>(uri[2])) == 't' &&
        std::tolower(static_cast<unsigned char>(uri[3])) == 'a' &&
        uri[4] == ':';
}

void appendFingerprintBytes(uint64_t& fingerprint, std::string_view bytes)
{
    constexpr uint64_t kFnvPrime = 1099511628211ull;
    for (const unsigned char byte : bytes) {
        fingerprint ^= byte;
        fingerprint *= kFnvPrime;
    }
}

void appendFingerprintUint64(uint64_t& fingerprint, uint64_t value)
{
    std::array<char, sizeof(value)> bytes{};
    for (size_t byteIndex = 0; byteIndex < bytes.size(); ++byteIndex) {
        bytes[byteIndex] = static_cast<char>((value >> (byteIndex * 8u)) & 0xffu);
    }
    appendFingerprintBytes(fingerprint, std::string_view(bytes.data(), bytes.size()));
}

bool appendFileMetadataFingerprint(
    uint64_t& fingerprint,
    const std::filesystem::path& path)
{
    std::error_code sizeError;
    const uint64_t byteSize = std::filesystem::file_size(path, sizeError);
    std::error_code timeError;
    const auto writeTime = std::filesystem::last_write_time(path, timeError);
    if (sizeError || timeError) {
        return false;
    }
    appendFingerprintUint64(fingerprint, byteSize);
    appendFingerprintUint64(
        fingerprint,
        static_cast<uint64_t>(writeTime.time_since_epoch().count()));
    return true;
}

template <typename UriRange>
uint64_t sourceDependencyFingerprintFor(
    const std::filesystem::path& sourcePath,
    const UriRange& externalBufferUris)
{
    constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ull;
    uint64_t fingerprint = kFnvOffsetBasis;
    if (!appendFileMetadataFingerprint(fingerprint, sourcePath)) {
        return 0;
    }
    appendFingerprintUint64(fingerprint, static_cast<uint64_t>(externalBufferUris.size()));
    for (const auto& uriValue : externalBufferUris) {
        const std::string_view decodedUri(uriValue);
        if (decodedUri.empty()) {
            return 0;
        }
        appendFingerprintUint64(fingerprint, static_cast<uint64_t>(decodedUri.size()));
        appendFingerprintBytes(fingerprint, decodedUri);
        if (!appendFileMetadataFingerprint(
                fingerprint,
                sourcePath.parent_path() / pathFromUtf8ForStreamBuilder(decodedUri))) {
            return 0;
        }
    }
    return fingerprint == 0 ? 1 : fingerprint;
}

bool setStreamSourceDependencies(
    MeshletStreamBuildState& state,
    const std::vector<std::string>& externalBufferUris,
    std::string& reason)
{
    if (externalBufferUris.size() > std::numeric_limits<uint32_t>::max()) {
        reason = "streamasset source dependency count exceeds the format limit";
        return false;
    }
    state.sourceDependencyPaths.clear();
    state.sourceDependencyCount = static_cast<uint32_t>(externalBufferUris.size());
    for (const std::string& uri : externalBufferUris) {
        if (uri.empty() ||
            uri.find('\0') != std::string::npos ||
            uri.size() >= std::numeric_limits<uint32_t>::max() ||
            state.sourceDependencyPaths.size() >
                std::numeric_limits<uint32_t>::max() - uri.size() - 1u) {
            reason = "streamasset source dependency path directory exceeds the format limit";
            return false;
        }
        state.sourceDependencyPaths.insert(
            state.sourceDependencyPaths.end(),
            uri.begin(),
            uri.end());
        state.sourceDependencyPaths.push_back('\0');
    }
    return true;
}

bool loadSourceDependencyUris(
    const std::filesystem::path& sourcePath,
    std::vector<std::string>& externalBufferUris,
    std::string& reason)
{
    externalBufferUris.clear();
    if (lowerExtensionForStreamBuilder(sourcePath) != ".gltf") {
        return true;
    }

    std::ifstream file(sourcePath, std::ios::binary);
    if (!file) {
        reason = "streamasset builder cannot open glTF dependency metadata";
        return false;
    }
    try {
        nlohmann::json root;
        file >> root;
        const nlohmann::json buffers = root.value("buffers", nlohmann::json::array());
        if (!buffers.is_array()) {
            reason = "streamasset builder glTF buffers are invalid";
            return false;
        }
        for (const nlohmann::json& bufferJson : buffers) {
            if (!bufferJson.is_object()) {
                reason = "streamasset builder glTF buffer entry is invalid";
                return false;
            }
            const std::string uri = bufferJson.value("uri", std::string{});
            if (uri.empty() || isDataUriForStreamBuilder(uri)) {
                continue;
            }
            std::string decodedUri;
            if (!tinygltf::URIDecode(uri, &decodedUri, nullptr) || decodedUri.empty()) {
                reason = "streamasset builder failed to decode external buffer URI";
                return false;
            }
            externalBufferUris.push_back(std::move(decodedUri));
        }
    } catch (const std::exception& exception) {
        reason = "streamasset builder failed to parse glTF dependency metadata: ";
        reason += exception.what();
        return false;
    }
    return true;
}

bool loadExternalGltfMetadataForStreamAssetBuilder(
    const std::filesystem::path& sourcePath,
    StreamGltfSource& source,
    bool& applicable,
    std::string& reason)
{
    applicable = false;
    std::ifstream file(sourcePath, std::ios::binary);
    if (!file) {
        reason = "streamasset builder cannot open glTF metadata";
        return false;
    }

    nlohmann::json root;
    try {
        file >> root;
        if (!root.is_object()) {
            reason = "streamasset builder glTF root is not an object";
            return false;
        }

        const nlohmann::json buffers = root.value("buffers", nlohmann::json::array());
        if (!buffers.is_array() || buffers.empty()) {
            reason = "streamasset builder glTF has no buffers";
            return false;
        }
        for (const nlohmann::json& bufferJson : buffers) {
            if (!bufferJson.is_object()) {
                reason = "streamasset builder glTF buffer entry is invalid";
                return false;
            }
            const std::string uri = bufferJson.value("uri", std::string{});
            if (uri.empty() || isDataUriForStreamBuilder(uri)) {
                return true;
            }
        }
        applicable = true;

        source = {};
        source.directory = sourcePath.parent_path();
        source.rangeReadExternalBuffers = true;
        tinygltf::Model& model = source.model;
        model.defaultScene = root.value("scene", -1);

        const nlohmann::json extensionsRequired =
            root.value("extensionsRequired", nlohmann::json::array());
        if (extensionsRequired.is_array()) {
            for (const nlohmann::json& extension : extensionsRequired) {
                if (extension.is_string()) {
                    model.extensionsRequired.push_back(extension.get<std::string>());
                }
            }
        }

        model.buffers.reserve(buffers.size());
        source.bufferByteLengths.reserve(buffers.size());
        for (const nlohmann::json& bufferJson : buffers) {
            if (!bufferJson.contains("byteLength") || !bufferJson["byteLength"].is_number_unsigned()) {
                reason = "streamasset builder glTF buffer byteLength is missing or invalid";
                return false;
            }
            const uint64_t byteLength = bufferJson["byteLength"].get<uint64_t>();
            tinygltf::Buffer buffer;
            buffer.name = bufferJson.value("name", std::string{});
            buffer.uri = bufferJson.value("uri", std::string{});

            std::string decodedUri;
            if (!tinygltf::URIDecode(buffer.uri, &decodedUri, nullptr) || decodedUri.empty()) {
                reason = "streamasset builder failed to decode external buffer URI";
                return false;
            }
            const std::filesystem::path bufferPath =
                source.directory / pathFromUtf8ForStreamBuilder(decodedUri);
            std::error_code sizeError;
            const uint64_t actualByteLength = std::filesystem::file_size(bufferPath, sizeError);
            if (sizeError) {
                reason = "streamasset builder cannot stat external buffer: " + bufferPath.string();
                return false;
            }
            if (actualByteLength < byteLength) {
                reason = "streamasset builder external buffer is smaller than its declared byteLength: " +
                    bufferPath.string();
                return false;
            }
            model.buffers.push_back(std::move(buffer));
            source.bufferByteLengths.push_back(byteLength);
            source.externalBufferUris.push_back(std::move(decodedUri));
        }

        const nlohmann::json bufferViews = root.value("bufferViews", nlohmann::json::array());
        if (!bufferViews.is_array()) {
            reason = "streamasset builder glTF bufferViews is invalid";
            return false;
        }
        model.bufferViews.reserve(bufferViews.size());
        for (const nlohmann::json& viewJson : bufferViews) {
            tinygltf::BufferView view;
            view.name = viewJson.value("name", std::string{});
            view.buffer = viewJson.value("buffer", -1);
            view.byteOffset = viewJson.value("byteOffset", size_t{0});
            view.byteLength = viewJson.value("byteLength", size_t{0});
            view.byteStride = viewJson.value("byteStride", size_t{0});
            view.target = viewJson.value("target", 0);
            model.bufferViews.push_back(std::move(view));
        }

        const nlohmann::json accessors = root.value("accessors", nlohmann::json::array());
        if (!accessors.is_array()) {
            reason = "streamasset builder glTF accessors is invalid";
            return false;
        }
        model.accessors.reserve(accessors.size());
        for (const nlohmann::json& accessorJson : accessors) {
            tinygltf::Accessor accessor;
            accessor.name = accessorJson.value("name", std::string{});
            accessor.bufferView = accessorJson.value("bufferView", -1);
            accessor.byteOffset = accessorJson.value("byteOffset", size_t{0});
            accessor.normalized = accessorJson.value("normalized", false);
            accessor.componentType = accessorJson.value("componentType", -1);
            accessor.count = accessorJson.value("count", size_t{0});
            accessor.type = accessorTypeForStreamBuilder(accessorJson.value("type", std::string{}));
            accessor.minValues = jsonNumberArrayForStreamBuilder(
                accessorJson.value("min", nlohmann::json::array()));
            accessor.maxValues = jsonNumberArrayForStreamBuilder(
                accessorJson.value("max", nlohmann::json::array()));
            accessor.sparse.isSparse = accessorJson.contains("sparse");
            model.accessors.push_back(std::move(accessor));
        }

        const nlohmann::json meshes = root.value("meshes", nlohmann::json::array());
        if (!meshes.is_array()) {
            reason = "streamasset builder glTF meshes is invalid";
            return false;
        }
        model.meshes.reserve(meshes.size());
        for (const nlohmann::json& meshJson : meshes) {
            tinygltf::Mesh mesh;
            mesh.name = meshJson.value("name", std::string{});
            const nlohmann::json primitives = meshJson.value("primitives", nlohmann::json::array());
            if (!primitives.is_array()) {
                reason = "streamasset builder glTF mesh primitives is invalid";
                return false;
            }
            mesh.primitives.reserve(primitives.size());
            for (const nlohmann::json& primitiveJson : primitives) {
                tinygltf::Primitive primitive;
                primitive.material = primitiveJson.value("material", -1);
                primitive.indices = primitiveJson.value("indices", -1);
                primitive.mode = primitiveJson.value("mode", TINYGLTF_MODE_TRIANGLES);
                const nlohmann::json attributes =
                    primitiveJson.value("attributes", nlohmann::json::object());
                if (!attributes.is_object()) {
                    reason = "streamasset builder glTF primitive attributes is invalid";
                    return false;
                }
                for (const auto& [name, accessorIndex] : attributes.items()) {
                    if (accessorIndex.is_number_integer()) {
                        primitive.attributes[name] = accessorIndex.get<int>();
                    }
                }
                mesh.primitives.push_back(std::move(primitive));
            }
            model.meshes.push_back(std::move(mesh));
        }

        const nlohmann::json nodes = root.value("nodes", nlohmann::json::array());
        if (!nodes.is_array()) {
            reason = "streamasset builder glTF nodes is invalid";
            return false;
        }
        model.nodes.reserve(nodes.size());
        source.nodeVisibility.reserve(nodes.size());
        for (const nlohmann::json& nodeJson : nodes) {
            tinygltf::Node node;
            node.name = nodeJson.value("name", std::string{});
            node.mesh = nodeJson.value("mesh", -1);
            node.children = jsonIntArrayForStreamBuilder(
                nodeJson.value("children", nlohmann::json::array()));
            node.rotation = jsonNumberArrayForStreamBuilder(
                nodeJson.value("rotation", nlohmann::json::array()));
            node.scale = jsonNumberArrayForStreamBuilder(
                nodeJson.value("scale", nlohmann::json::array()));
            node.translation = jsonNumberArrayForStreamBuilder(
                nodeJson.value("translation", nlohmann::json::array()));
            node.matrix = jsonNumberArrayForStreamBuilder(
                nodeJson.value("matrix", nlohmann::json::array()));

            bool visible = true;
            const auto extensions = nodeJson.find("extensions");
            if (extensions != nodeJson.end() && extensions->is_object()) {
                const auto visibility = extensions->find(kExtensionNodeVisibility);
                if (visibility != extensions->end() && visibility->is_object()) {
                    const auto visibleValue = visibility->find("visible");
                    if (visibleValue != visibility->end() && visibleValue->is_boolean()) {
                        visible = visibleValue->get<bool>();
                    }
                }
            }
            model.nodes.push_back(std::move(node));
            source.nodeVisibility.push_back(visible);
        }

        const nlohmann::json scenes = root.value("scenes", nlohmann::json::array());
        if (!scenes.is_array()) {
            reason = "streamasset builder glTF scenes is invalid";
            return false;
        }
        model.scenes.reserve(scenes.size());
        for (const nlohmann::json& sceneJson : scenes) {
            tinygltf::Scene scene;
            scene.name = sceneJson.value("name", std::string{});
            scene.nodes = jsonIntArrayForStreamBuilder(
                sceneJson.value("nodes", nlohmann::json::array()));
            model.scenes.push_back(std::move(scene));
        }
    } catch (const std::exception& exception) {
        reason = "streamasset builder failed to parse glTF metadata: ";
        reason += exception.what();
        return false;
    }
    return true;
}

bool loadGltfModelForStreamAssetBuilder(
    const std::filesystem::path& sourcePath,
    StreamGltfSource& source,
    std::string& reason)
{
    if (lowerExtensionForStreamBuilder(sourcePath) == ".gltf") {
        bool externalMetadataApplicable = false;
        if (!loadExternalGltfMetadataForStreamAssetBuilder(
                sourcePath,
                source,
                externalMetadataApplicable,
                reason)) {
            return false;
        }
        if (externalMetadataApplicable) {
            for (const std::string& extension : source.model.extensionsRequired) {
                if (!streamBuilderSupportsRequiredExtension(extension)) {
                    reason = "Required extension unsupported by meshstream builder: " + extension;
                    return false;
                }
            }
            return true;
        }
    }

    source = {};
    source.directory = sourcePath.parent_path();
    tinygltf::TinyGLTF loader;
    loader.SetImagesAsIs(true);
    loader.SetMaxExternalFileSize(static_cast<size_t>(-1));
    loader.SetImageLoader(
        [](tinygltf::Image* image,
           const int,
           std::string*,
           std::string*,
           int,
           int,
           const unsigned char*,
           int,
           void*) {
            if (image != nullptr) {
                image->image.clear();
                image->width = 0;
                image->height = 0;
                image->component = 0;
                image->bits = 8;
                image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
            }
            return true;
        },
        nullptr);

    std::string error;
    std::string warning;
    const std::string filenameString = sourcePath.string();
    const bool ok = lowerExtensionForStreamBuilder(sourcePath) == ".glb"
        ? loader.LoadBinaryFromFile(&source.model, &error, &warning, filenameString)
        : loader.LoadASCIIFromFile(&source.model, &error, &warning, filenameString);
    if (!ok) {
        reason = error.empty() ? "tinygltf failed to load streamasset source" : error;
        if (!warning.empty()) {
            reason += " warning: " + warning;
        }
        return false;
    }

    source.bufferByteLengths.reserve(source.model.buffers.size());
    for (const tinygltf::Buffer& buffer : source.model.buffers) {
        source.bufferByteLengths.push_back(buffer.data.size());
    }
    if (!loadSourceDependencyUris(sourcePath, source.externalBufferUris, reason)) {
        return false;
    }
    source.nodeVisibility.reserve(source.model.nodes.size());
    for (const tinygltf::Node& node : source.model.nodes) {
        source.nodeVisibility.push_back(readNodeVisibilityForStreamBuilder(node));
    }

    for (const std::string& extension : source.model.extensionsRequired) {
        if (!streamBuilderSupportsRequiredExtension(extension)) {
            reason = "Required extension unsupported by meshstream builder: " + extension;
            return false;
        }
    }
    return true;
}

bool loadRenderPrimitiveForStreamAssetBuilder(
    const StreamGltfSource& source,
    int32_t meshIndex,
    int32_t primitiveIndex,
    RenderPrimitive& outPrimitive,
    std::string& reason)
{
    outPrimitive = {};
    const tinygltf::Model& model = source.model;
    if (!validGltfIndex(meshIndex, model.meshes.size())) {
        reason = "streamasset builder mesh index is out of range";
        return false;
    }

    const tinygltf::Mesh& mesh = model.meshes[static_cast<size_t>(meshIndex)];
    if (!validGltfIndex(primitiveIndex, mesh.primitives.size())) {
        reason = "streamasset builder primitive index is out of range";
        return false;
    }

    const tinygltf::Primitive& gltfPrimitive = mesh.primitives[static_cast<size_t>(primitiveIndex)];
    outPrimitive.name = mesh.name.empty()
        ? "Primitive " + std::to_string(primitiveIndex)
        : mesh.name;
    outPrimitive.meshIndex = meshIndex;
    outPrimitive.primitiveIndex = primitiveIndex;
    outPrimitive.materialIndex = gltfPrimitive.material;
    outPrimitive.mode = gltfPrimitive.mode;
    if (outPrimitive.mode != TINYGLTF_MODE_TRIANGLES) {
        return true;
    }

    const auto positionAccessorIter = gltfPrimitive.attributes.find("POSITION");
    if (positionAccessorIter == gltfPrimitive.attributes.end() ||
        !validGltfIndex(positionAccessorIter->second, model.accessors.size())) {
        return true;
    }

    const tinygltf::Accessor& positionAccessor =
        model.accessors[static_cast<size_t>(positionAccessorIter->second)];
    outPrimitive.vertexCount = positionAccessor.count;
    if (outPrimitive.vertexCount > std::numeric_limits<uint32_t>::max()) {
        reason = "streamasset builder primitive vertex count exceeds 32-bit indices";
        return false;
    }
    outPrimitive.localBounds = accessorBoundsForStreamBuilder(positionAccessor);
    outPrimitive.positions = readFloat3AccessorForStreamBuilder(source, positionAccessor, reason);
    if (outPrimitive.positions.empty()) {
        reason = "streamasset builder failed to read primitive POSITION accessor";
        return false;
    }
    if (!outPrimitive.localBounds.valid) {
        outPrimitive.localBounds = boundsFromPositionsForStreamBuilder(outPrimitive.positions);
    }

    const auto normalAccessorIter = gltfPrimitive.attributes.find("NORMAL");
    if (normalAccessorIter != gltfPrimitive.attributes.end() &&
        validGltfIndex(normalAccessorIter->second, model.accessors.size())) {
        outPrimitive.normals = readFloat3AccessorForStreamBuilder(
            source,
            model.accessors[static_cast<size_t>(normalAccessorIter->second)],
            reason);
        if (outPrimitive.normals.size() != outPrimitive.positions.size()) {
            outPrimitive.normals.clear();
        } else {
            outPrimitive.hasAuthoredNormals = true;
        }
    }

    const auto texcoordAccessorIter = gltfPrimitive.attributes.find("TEXCOORD_0");
    if (texcoordAccessorIter != gltfPrimitive.attributes.end() &&
        validGltfIndex(texcoordAccessorIter->second, model.accessors.size())) {
        outPrimitive.texcoords0 = readFloat2AccessorForStreamBuilder(
            source,
            model.accessors[static_cast<size_t>(texcoordAccessorIter->second)],
            reason);
        if (outPrimitive.texcoords0.size() != outPrimitive.positions.size()) {
            outPrimitive.texcoords0.clear();
        }
    }

    if (validGltfIndex(gltfPrimitive.indices, model.accessors.size())) {
        const tinygltf::Accessor& indexAccessor =
            model.accessors[static_cast<size_t>(gltfPrimitive.indices)];
        outPrimitive.indexCount = indexAccessor.count;
        outPrimitive.indices = readIndexAccessorForStreamBuilder(source, indexAccessor, reason);
        if (outPrimitive.indices.empty()) {
            reason = "streamasset builder failed to read primitive index accessor";
            return false;
        }
    } else {
        outPrimitive.indexCount = outPrimitive.vertexCount;
        outPrimitive.indices.reserve(static_cast<size_t>(outPrimitive.vertexCount));
        for (uint64_t index = 0; index < outPrimitive.vertexCount; ++index) {
            outPrimitive.indices.push_back(static_cast<uint32_t>(index));
        }
    }

    outPrimitive.triangleCount = triangleCountForStreamPrimitive(outPrimitive.mode, outPrimitive.indexCount);
    return true;
}

uint64_t gltfPrimitiveKey(int32_t meshIndex, int32_t primitiveIndex)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(meshIndex)) << 32u) |
        static_cast<uint32_t>(primitiveIndex);
}

MeshletStreamPartialFileHeader makePartialBuildHeader(
    const MeshletStreamBuildState& state,
    MeshletStreamPayloadCompression compressionMode,
    uint64_t payloadWriteOffset)
{
    MeshletStreamPartialFileHeader header;
    std::memcpy(header.magic, kMeshletStreamPartialMagic.data(), kMeshletStreamPartialMagic.size());
    header.version = kMeshletStreamPartialVersion;
    header.endian = kMeshletStreamEndian;
    header.sourceFileSize = state.header.sourceFileSize;
    header.sourceWriteTime = state.header.sourceWriteTime;
    header.sourceDependencyFingerprint = state.header.sourceDependencyFingerprint;
    header.payloadWriteOffset = payloadWriteOffset;
    header.compressionMode = static_cast<uint32_t>(compressionMode);
    header.nextRenderPrimitiveIndex = state.nextRenderPrimitiveIndex;
    header.primitiveCount = static_cast<uint32_t>(state.primitives.size());
    header.geometryCount = static_cast<uint32_t>(state.geometries.size());
    header.lodLevelCount = static_cast<uint32_t>(state.lodLevels.size());
    header.groupCount = static_cast<uint32_t>(state.groups.size());
    header.clusterRefCount = static_cast<uint32_t>(state.clusterRefs.size());
    header.nodeCount = static_cast<uint32_t>(state.nodes.size());
    header.pageCount = static_cast<uint32_t>(state.pages.size());
    header.pageOffsetCount = static_cast<uint32_t>(state.pageOffsets.size());
    header.geometryEntryCount = static_cast<uint32_t>(state.partialGeometryEntries.size());
    header.maxPagePayloadBytes = state.header.maxPagePayloadBytes;
    header.pagePayloadAlignment = static_cast<uint32_t>(kPageSlotAlignment);
    header.maxVertices = kMeshletClusterMaxVertices;
    header.minTriangles = kMeshletClusterMinTriangles;
    header.maxTriangles = kMeshletClusterMaxTriangles;
    header.lodGroupSize = kMeshletLodGroupSize;
    header.fillWeight = kMeshletClusterFillWeight;
    header.lodErrorMergePrevious = kMeshletLodErrorMergePrevious;
    header.lodErrorMergeAdditive = kMeshletLodErrorMergeAdditive;
    return header;
}

bool directoryRangeValid(uint32_t offset, uint32_t count, size_t size)
{
    return offset <= size && count <= size - offset;
}

bool validStreamGroupMetric(const MeshletStreamGroupInfo& group)
{
    for (float value : group.boundsCenterRadius) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return group.boundsCenterRadius[3] >= 0.0f &&
        std::isfinite(group.maxQuadricError) &&
        group.maxQuadricError >= 0.0f;
}

bool validStreamNodeMetric(const MeshletStreamNodeInfo& node)
{
    for (float value : node.boundsCenterRadius) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return node.boundsCenterRadius[3] >= 0.0f &&
        std::isfinite(node.maxQuadricError) &&
        node.maxQuadricError >= 0.0f;
}

bool streamHierarchyValid(
    std::span<const MeshletStreamPrimitiveInfo> primitives,
    std::span<const MeshletStreamGroupInfo> groups,
    std::span<const MeshletStreamNodeInfo> nodes)
{
    uint32_t expectedNodeOffset = 0;
    std::vector<uint8_t> reached(nodes.size(), 0);
    std::vector<uint8_t> groupReached(groups.size(), 0);
    std::vector<uint32_t> stack;
    for (uint32_t primitiveIndex = 0; primitiveIndex < primitives.size(); ++primitiveIndex) {
        const MeshletStreamPrimitiveInfo& primitive = primitives[primitiveIndex];
        if (primitive.lodLevelCount == 0 ||
            primitive.nodeCount <= primitive.lodLevelCount ||
            primitive.nodeOffset != expectedNodeOffset ||
            !directoryRangeValid(primitive.nodeOffset, primitive.nodeCount, nodes.size())) {
            return false;
        }
        expectedNodeOffset += primitive.nodeCount;

        const MeshletStreamNodeInfo& root = nodes[primitive.nodeOffset];
        if (root.primitiveIndex != primitiveIndex ||
            root.groupIndex != kMeshletStreamInvalidGroupIndex ||
            root.childCount != primitive.lodLevelCount ||
            root.childOffset != primitive.nodeOffset + 1u ||
            root.lodLevel != kMeshletStreamInvalidNodeIndex) {
            return false;
        }
        for (uint32_t lodChild = 0; lodChild < root.childCount; ++lodChild) {
            if (nodes[root.childOffset + lodChild].lodLevel != lodChild) {
                return false;
            }
        }

        stack.clear();
        stack.push_back(primitive.nodeOffset);
        uint32_t reachedCount = 0;
        while (!stack.empty()) {
            const uint32_t nodeIndex = stack.back();
            stack.pop_back();
            if (nodeIndex < primitive.nodeOffset ||
                nodeIndex >= primitive.nodeOffset + primitive.nodeCount ||
                reached[nodeIndex] != 0) {
                return false;
            }
            reached[nodeIndex] = 1;
            ++reachedCount;

            const MeshletStreamNodeInfo& node = nodes[nodeIndex];
            if (node.primitiveIndex != primitiveIndex || !validStreamNodeMetric(node)) {
                return false;
            }
            if (node.childCount == 0) {
                if (node.groupIndex < primitive.groupOffset ||
                    node.groupIndex >= primitive.groupOffset + primitive.groupCount ||
                    node.groupIndex >= groups.size() ||
                    groupReached[node.groupIndex] != 0 ||
                    groups[node.groupIndex].primitiveIndex != primitiveIndex ||
                    groups[node.groupIndex].lodLevel != node.lodLevel) {
                    return false;
                }
                groupReached[node.groupIndex] = 1;
                continue;
            }
            if (node.childCount > 32 ||
                node.groupIndex != kMeshletStreamInvalidGroupIndex ||
                !directoryRangeValid(node.childOffset, node.childCount, nodes.size()) ||
                node.childOffset < primitive.nodeOffset ||
                node.childOffset + node.childCount > primitive.nodeOffset + primitive.nodeCount) {
                return false;
            }
            for (uint32_t child = 0; child < node.childCount; ++child) {
                stack.push_back(node.childOffset + child);
            }
        }
        if (reachedCount != primitive.nodeCount) {
            return false;
        }
        for (uint32_t groupChild = 0; groupChild < primitive.groupCount; ++groupChild) {
            if (groupReached[primitive.groupOffset + groupChild] == 0) {
                return false;
            }
        }
    }
    return expectedNodeOffset == nodes.size();
}

bool addArrayByteSize(uint64_t count, uint64_t elementSize, uint64_t& inOutByteSize)
{
    if (count != 0 && elementSize > std::numeric_limits<uint64_t>::max() / count) {
        return false;
    }
    const uint64_t byteSize = count * elementSize;
    if (byteSize > std::numeric_limits<uint64_t>::max() - inOutByteSize) {
        return false;
    }
    inOutByteSize += byteSize;
    return true;
}

bool partialBuildStateRangesValid(
    const MeshletStreamBuildState& state,
    uint64_t payloadWriteOffset)
{
    if (state.primitives.size() != state.geometries.size() ||
        state.pages.size() != state.groups.size() ||
        state.pages.size() != state.pageOffsets.size() ||
        state.partialGeometryEntries.size() != state.primitives.size() ||
        payloadWriteOffset < sizeof(MeshletStreamFileHeader)) {
        return false;
    }

    for (uint32_t primitiveIndex = 0; primitiveIndex < state.primitives.size(); ++primitiveIndex) {
        const MeshletStreamPrimitiveInfo& primitive = state.primitives[primitiveIndex];
        if (primitive.pageCount == 0 ||
            primitive.groupCount == 0 ||
            primitive.pageCount != primitive.groupCount ||
            primitive.fallbackPageCount == 0 ||
            primitive.fallbackGroupCount == 0 ||
            !directoryRangeValid(primitive.lodLevelOffset, primitive.lodLevelCount, state.lodLevels.size()) ||
            !directoryRangeValid(primitive.pageOffset, primitive.pageCount, state.pages.size()) ||
            !directoryRangeValid(primitive.fallbackPageOffset, primitive.fallbackPageCount, state.pages.size()) ||
            !directoryRangeValid(primitive.groupOffset, primitive.groupCount, state.groups.size()) ||
            !directoryRangeValid(
                primitive.fallbackGroupOffset,
                primitive.fallbackGroupCount,
                state.groups.size()) ||
            !directoryRangeValid(primitive.nodeOffset, primitive.nodeCount, state.nodes.size())) {
            return false;
        }
    }

    std::vector<uint8_t> primitiveHasTerminalGroup(state.primitives.size(), 0);
    for (uint32_t groupIndex = 0; groupIndex < state.groups.size(); ++groupIndex) {
        const MeshletStreamGroupInfo& group = state.groups[groupIndex];
        if (group.primitiveIndex >= state.primitives.size() ||
            group.pageIndex >= state.pages.size() ||
            group.clusterCount == 0 ||
            group.clusterCount > kMeshletLodGroupSize ||
            !validStreamGroupMetric(group) ||
            !directoryRangeValid(
                group.clusterRefOffset,
                group.clusterCount,
                state.clusterRefs.size())) {
            return false;
        }
        const MeshletStreamPrimitiveInfo& primitive = state.primitives[group.primitiveIndex];
        const MeshletStreamPageInfo& page = state.pages[group.pageIndex];
        if (groupIndex < primitive.groupOffset ||
            groupIndex >= primitive.groupOffset + primitive.groupCount ||
            group.pageIndex != primitive.pageOffset + (groupIndex - primitive.groupOffset) ||
            page.primitiveIndex != group.primitiveIndex ||
            page.lodLevel != group.lodLevel ||
            page.lodGroupIndex != groupIndex ||
            page.clusterCount != group.clusterCount) {
            return false;
        }
        for (uint32_t child = 0; child < group.clusterCount; ++child) {
            const uint32_t refinedGroup = state.clusterRefs[group.clusterRefOffset + child];
            if (refinedGroup != kMeshletStreamInvalidGroupIndex &&
                (refinedGroup < primitive.groupOffset ||
                    refinedGroup >= groupIndex)) {
                return false;
            }
        }
        if (group.maxQuadricError == kMeshletStreamTerminalGroupError) {
            primitiveHasTerminalGroup[group.primitiveIndex] = 1;
        }
    }
    if (std::find(primitiveHasTerminalGroup.begin(), primitiveHasTerminalGroup.end(), 0) !=
        primitiveHasTerminalGroup.end()) {
        return false;
    }
    if (!streamHierarchyValid(state.primitives, state.groups, state.nodes)) {
        return false;
    }

    for (uint32_t geometryIndex = 0; geometryIndex < state.geometries.size(); ++geometryIndex) {
        const MeshletStreamGeometryInfo& geometry = state.geometries[geometryIndex];
        if (geometry.primitiveIndex >= state.primitives.size() ||
            geometry.pageCount == 0 ||
            !directoryRangeValid(geometry.pageOffset, geometry.pageCount, state.pages.size()) ||
            !directoryRangeValid(
                geometry.pagePayloadOffsetTableOffset,
                geometry.pagePayloadOffsetTableCount,
                state.pageOffsets.size()) ||
            geometry.pagePayloadOffsetTableCount != geometry.pageCount) {
            return false;
        }

        const MeshletStreamPrimitiveInfo& primitive = state.primitives[geometry.primitiveIndex];
        if (geometry.pageOffset != primitive.pageOffset ||
            geometry.pageCount != primitive.pageCount ||
            geometry.renderPrimitiveIndex != primitive.renderPrimitiveIndex ||
            geometry.pagePayloadOffsetTableOffset != primitive.pageOffset) {
            return false;
        }
    }

    for (uint32_t pageIndex = 0; pageIndex < state.pages.size(); ++pageIndex) {
        const MeshletStreamPageInfo& page = state.pages[pageIndex];
        if (page.payloadOffset != state.pageOffsets[pageIndex] ||
            page.payloadOffset % kFileAlignment != 0 ||
            page.payloadSize == 0 ||
            page.payloadOffset > payloadWriteOffset ||
            page.payloadSize > payloadWriteOffset - page.payloadOffset) {
            return false;
        }
    }

    for (const MeshletStreamPartialGeometryEntry& entry : state.partialGeometryEntries) {
        if (entry.streamPrimitiveIndex >= state.primitives.size() ||
            entry.renderPrimitiveIndex >= state.nextRenderPrimitiveIndex) {
            return false;
        }
    }

    return true;
}

bool savePartialBuildState(
    const MeshletStreamPartialBuildContext& context,
    const MeshletStreamBuildState& state,
    MeshletStreamPayloadCompression compressionMode,
    uint64_t payloadWriteOffset,
    std::string& reason)
{
    if (!partialBuildStateRangesValid(state, payloadWriteOffset)) {
        reason = "streamasset partial cache state is invalid";
        return false;
    }

    std::error_code createError;
    if (context.partialPath.has_parent_path()) {
        std::filesystem::create_directories(context.partialPath.parent_path(), createError);
        if (createError) {
            reason = "streamasset partial cache directory create failed: " + createError.message();
            return false;
        }
    }

    std::filesystem::path tempPath = context.partialPath;
    tempPath += ".tmp";
    std::ofstream file(tempPath, std::ios::binary | std::ios::trunc);
    if (!file) {
        reason = "streamasset partial cache file cannot be opened";
        return false;
    }

    const MeshletStreamPartialFileHeader header =
        makePartialBuildHeader(state, compressionMode, payloadWriteOffset);
    if (!writePod(file, header) ||
        !writeArray(file, state.primitives) ||
        !writeArray(file, state.geometries) ||
        !writeArray(file, state.lodLevels) ||
        !writeArray(file, state.groups) ||
        !writeArray(file, state.clusterRefs) ||
        !writeArray(file, state.nodes) ||
        !writeArray(file, state.pages) ||
        !writeArray(file, state.pageOffsets) ||
        !writeArray(file, state.partialGeometryEntries)) {
        reason = "streamasset partial cache write failed";
        return false;
    }
    file.close();
    if (!file) {
        reason = "streamasset partial cache close failed";
        return false;
    }

    std::error_code removeError;
    std::filesystem::remove(context.partialPath, removeError);
    std::error_code renameError;
    std::filesystem::rename(tempPath, context.partialPath, renameError);
    if (renameError) {
        std::error_code cleanupError;
        std::filesystem::remove(tempPath, cleanupError);
        reason = "streamasset partial cache publish failed: " + renameError.message();
        return false;
    }

    return true;
}

bool loadPartialBuildState(
    const std::filesystem::path& partialPath,
    const std::filesystem::path& outputPath,
    const std::filesystem::path& sourcePath,
    uint64_t sourceDependencyFingerprint,
    MeshletStreamPayloadCompression compressionMode,
    MeshletStreamBuildState& state,
    std::unordered_map<uint64_t, uint32_t>& primitiveMap,
    uint64_t& payloadWriteOffset)
{
    std::error_code existsError;
    if (!std::filesystem::exists(partialPath, existsError) || existsError) {
        return false;
    }

    std::ifstream file(partialPath, std::ios::binary);
    if (!file) {
        return false;
    }

    MeshletStreamPartialFileHeader partialHeader;
    if (!readPod(file, partialHeader)) {
        return false;
    }
    if (std::memcmp(partialHeader.magic, kMeshletStreamPartialMagic.data(), kMeshletStreamPartialMagic.size()) != 0 ||
        partialHeader.version != kMeshletStreamPartialVersion ||
        partialHeader.endian != kMeshletStreamEndian ||
        partialHeader.sourceFileSize != sourceFileSizeFor(sourcePath) ||
        partialHeader.sourceWriteTime != sourceWriteTimeFor(sourcePath) ||
        partialHeader.sourceDependencyFingerprint != sourceDependencyFingerprint ||
        partialHeader.compressionMode != static_cast<uint32_t>(compressionMode) ||
        !meshletStreamPartialBuildParamsMatch(partialHeader) ||
        partialHeader.maxPagePayloadBytes == 0 ||
        partialHeader.payloadWriteOffset < sizeof(MeshletStreamFileHeader)) {
        return false;
    }

    std::error_code sizeError;
    const uint64_t outputFileSize = std::filesystem::file_size(outputPath, sizeError);
    if (sizeError || outputFileSize < partialHeader.payloadWriteOffset) {
        return false;
    }

    std::error_code partialSizeError;
    const uint64_t partialFileSize = std::filesystem::file_size(partialPath, partialSizeError);
    if (partialSizeError) {
        return false;
    }
    uint64_t requiredPartialFileSize = sizeof(MeshletStreamPartialFileHeader);
    if (!addArrayByteSize(partialHeader.primitiveCount, sizeof(MeshletStreamPrimitiveInfo), requiredPartialFileSize) ||
        !addArrayByteSize(partialHeader.geometryCount, sizeof(MeshletStreamGeometryInfo), requiredPartialFileSize) ||
        !addArrayByteSize(partialHeader.lodLevelCount, sizeof(MeshletStreamLodLevelInfo), requiredPartialFileSize) ||
        !addArrayByteSize(partialHeader.groupCount, sizeof(MeshletStreamGroupInfo), requiredPartialFileSize) ||
        !addArrayByteSize(partialHeader.clusterRefCount, sizeof(uint32_t), requiredPartialFileSize) ||
        !addArrayByteSize(partialHeader.nodeCount, sizeof(MeshletStreamNodeInfo), requiredPartialFileSize) ||
        !addArrayByteSize(partialHeader.pageCount, sizeof(MeshletStreamPageInfo), requiredPartialFileSize) ||
        !addArrayByteSize(partialHeader.pageOffsetCount, sizeof(uint64_t), requiredPartialFileSize) ||
        !addArrayByteSize(
            partialHeader.geometryEntryCount,
            sizeof(MeshletStreamPartialGeometryEntry),
            requiredPartialFileSize) ||
        requiredPartialFileSize > partialFileSize) {
        return false;
    }

    MeshletStreamBuildState loaded;
    loaded.header = makeStreamFileHeader(sourcePath, sourceDependencyFingerprint);
    loaded.header.maxPagePayloadBytes = partialHeader.maxPagePayloadBytes;
    loaded.nextRenderPrimitiveIndex = partialHeader.nextRenderPrimitiveIndex;
    if (!readArray(file, partialHeader.primitiveCount, loaded.primitives) ||
        !readArray(file, partialHeader.geometryCount, loaded.geometries) ||
        !readArray(file, partialHeader.lodLevelCount, loaded.lodLevels) ||
        !readArray(file, partialHeader.groupCount, loaded.groups) ||
        !readArray(file, partialHeader.clusterRefCount, loaded.clusterRefs) ||
        !readArray(file, partialHeader.nodeCount, loaded.nodes) ||
        !readArray(file, partialHeader.pageCount, loaded.pages) ||
        !readArray(file, partialHeader.pageOffsetCount, loaded.pageOffsets) ||
        !readArray(file, partialHeader.geometryEntryCount, loaded.partialGeometryEntries)) {
        return false;
    }

    if (!partialBuildStateRangesValid(loaded, partialHeader.payloadWriteOffset)) {
        return false;
    }

    primitiveMap.clear();
    primitiveMap.reserve(loaded.partialGeometryEntries.size() * 2u);
    for (const MeshletStreamPartialGeometryEntry& entry : loaded.partialGeometryEntries) {
        primitiveMap[gltfPrimitiveKey(entry.meshIndex, entry.primitiveIndex)] = entry.streamPrimitiveIndex;
    }

    state = std::move(loaded);
    payloadWriteOffset = partialHeader.payloadWriteOffset;
    return true;
}

bool buildStreamAssetGeometryPayloadsFromGltf(
    std::ostream& stream,
    MeshletStreamBuildState& state,
    const StreamGltfSource& source,
    MeshletStreamPayloadCompression compressionMode,
    std::unordered_map<uint64_t, uint32_t>& primitiveMap,
    MeshletStreamPartialBuildContext* partialContext,
    std::string& reason)
{
    const tinygltf::Model& model = source.model;
    uint32_t renderPrimitiveIndex = 0;
    for (size_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
        const tinygltf::Mesh& mesh = model.meshes[meshIndex];
        for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex) {
            const uint32_t sourceRenderPrimitiveIndex = renderPrimitiveIndex++;
            if (sourceRenderPrimitiveIndex < state.nextRenderPrimitiveIndex) {
                continue;
            }
            if (partialContext != nullptr &&
                partialContext->maxNewGeometriesPerInvocation != 0 &&
                partialContext->newGeometryCount >= partialContext->maxNewGeometriesPerInvocation) {
                partialContext->paused = true;
                return true;
            }

            RenderPrimitive primitive;
            if (!loadRenderPrimitiveForStreamAssetBuilder(
                    source,
                    static_cast<int32_t>(meshIndex),
                    static_cast<int32_t>(primitiveIndex),
                    primitive,
                    reason)) {
                return false;
            }

            if (primitive.positions.empty()) {
                continue;
            }

            buildMeshletsForPrimitive(primitive);
            int32_t streamPrimitiveIndex = kInvalidSceneIndex;
            if (!appendStreamPrimitivePages(
                    stream,
                    state,
                    primitive,
                    sourceRenderPrimitiveIndex,
                    compressionMode,
                    streamPrimitiveIndex,
                    reason)) {
                return false;
            }
            if (streamPrimitiveIndex >= 0) {
                primitiveMap[gltfPrimitiveKey(static_cast<int32_t>(meshIndex), static_cast<int32_t>(primitiveIndex))] =
                    static_cast<uint32_t>(streamPrimitiveIndex);
                state.partialGeometryEntries.push_back(MeshletStreamPartialGeometryEntry{
                    .meshIndex = static_cast<int32_t>(meshIndex),
                    .primitiveIndex = static_cast<int32_t>(primitiveIndex),
                    .renderPrimitiveIndex = sourceRenderPrimitiveIndex,
                    .streamPrimitiveIndex = static_cast<uint32_t>(streamPrimitiveIndex),
                });
                state.nextRenderPrimitiveIndex = sourceRenderPrimitiveIndex + 1u;
                if (partialContext != nullptr) {
                    const std::streampos payloadWritePosition = stream.tellp();
                    if (payloadWritePosition == std::streampos(-1)) {
                        reason = "streamasset partial payload offset query failed";
                        return false;
                    }
                    if (!savePartialBuildState(
                            *partialContext,
                            state,
                            compressionMode,
                            static_cast<uint64_t>(payloadWritePosition),
                            reason)) {
                        return false;
                    }
                    ++partialContext->newGeometryCount;
                }
            }
        }
    }
    return true;
}

bool appendStreamAssetInstancesFromGltf(
    MeshletStreamBuildState& state,
    const StreamGltfSource& source,
    int32_t sceneIndex,
    const std::unordered_map<uint64_t, uint32_t>& primitiveMap,
    std::string& reason)
{
    const tinygltf::Model& model = source.model;
    if (!validGltfIndex(sceneIndex, model.scenes.size())) {
        reason = "streamasset builder default scene index is out of range";
        return false;
    }

    uint32_t renderNodeIndex = 0;
    std::function<bool(int32_t, const float4x4&, bool)> traverseNode;
    traverseNode = [&](int32_t nodeIndex, const float4x4& parentWorld, bool parentVisible) -> bool {
        if (!validGltfIndex(nodeIndex, model.nodes.size())) {
            reason = "streamasset builder scene references an out-of-range node";
            return false;
        }

        const tinygltf::Node& node = model.nodes[static_cast<size_t>(nodeIndex)];
        const float4x4 worldMatrix = parentWorld * makeNodeLocalMatrixForStreamBuilder(node);
        const bool nodeVisible = static_cast<size_t>(nodeIndex) < source.nodeVisibility.size()
            ? source.nodeVisibility[static_cast<size_t>(nodeIndex)]
            : readNodeVisibilityForStreamBuilder(node);
        const bool visible = parentVisible && nodeVisible;
        if (validGltfIndex(node.mesh, model.meshes.size())) {
            const tinygltf::Mesh& mesh = model.meshes[static_cast<size_t>(node.mesh)];
            for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex) {
                const auto mappedPrimitive = primitiveMap.find(
                    gltfPrimitiveKey(node.mesh, static_cast<int32_t>(primitiveIndex)));
                if (mappedPrimitive == primitiveMap.end()) {
                    continue;
                }

                const uint32_t streamPrimitiveIndex = mappedPrimitive->second;
                if (streamPrimitiveIndex >= state.primitives.size()) {
                    reason = "streamasset builder instance references an invalid primitive";
                    return false;
                }

                const tinygltf::Primitive& gltfPrimitive = mesh.primitives[primitiveIndex];
                MeshletStreamInstanceInfo instance;
                instance.renderNodeIndex = renderNodeIndex++;
                instance.primitiveIndex = streamPrimitiveIndex;
                instance.materialIndex = static_cast<uint32_t>(
                    gltfPrimitive.material >= 0
                        ? gltfPrimitive.material
                        : static_cast<int32_t>(state.primitives[streamPrimitiveIndex].materialIndex));
                instance.visible = visible ? 1u : 0u;
                copyMatrix(worldMatrix, instance.worldMatrix);
                state.instances.push_back(instance);
            }
        }

        for (const int32_t childIndex : node.children) {
            if (!traverseNode(childIndex, worldMatrix, visible)) {
                return false;
            }
        }
        return true;
    };

    const tinygltf::Scene& scene = model.scenes[static_cast<size_t>(sceneIndex)];
    for (const int32_t rootNodeIndex : scene.nodes) {
        if (!traverseNode(rootNodeIndex, float4x4::Identity(), true)) {
            return false;
        }
    }
    return true;
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
    std::span<const MeshletStreamGroupInfo> groups;
    std::span<const uint32_t> clusterRefs;
    std::span<const MeshletStreamNodeInfo> nodes;
    std::span<const MeshletStreamPageInfo> pages;
    std::span<const uint64_t> pageOffsets;
    std::span<const char> sourceDependencyPaths;
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
        impl->header.sourceDependencyFingerprint == 0 ||
        impl->header.maxPagePayloadBytes == 0 ||
        impl->header.groupCount == 0 ||
        impl->header.groupCount != impl->header.pageCount ||
        impl->header.clusterRefCount == 0 ||
        impl->header.nodeCount == 0) {
        reason = "streamasset header metadata is invalid";
        return false;
    }
    if (!rangeWithin<MeshletStreamPrimitiveInfo>(impl->dataSize, impl->header.primitiveOffset, impl->header.primitiveCount) ||
        !rangeWithin<MeshletStreamInstanceInfo>(impl->dataSize, impl->header.instanceOffset, impl->header.instanceCount) ||
        !rangeWithin<MeshletStreamGeometryInfo>(impl->dataSize, impl->header.geometryOffset, impl->header.geometryCount) ||
        !rangeWithin<MeshletStreamLodLevelInfo>(impl->dataSize, impl->header.lodLevelOffset, impl->header.lodLevelCount) ||
        !rangeWithin<MeshletStreamGroupInfo>(impl->dataSize, impl->header.groupInfoOffset, impl->header.groupCount) ||
        !rangeWithin<uint32_t>(impl->dataSize, impl->header.clusterRefOffset, impl->header.clusterRefCount) ||
        !rangeWithin<MeshletStreamNodeInfo>(impl->dataSize, impl->header.nodeInfoOffset, impl->header.nodeCount) ||
        !rangeWithin<MeshletStreamPageInfo>(impl->dataSize, impl->header.pageInfoOffset, impl->header.pageCount) ||
        !rangeWithin<uint64_t>(impl->dataSize, impl->header.pageOffsetTableOffset, impl->header.pageCount) ||
        !rangeWithin<char>(
            impl->dataSize,
            impl->header.sourceDependencyPathOffset,
            impl->header.sourceDependencyPathByteCount)) {
        reason = "streamasset directory exceeds file bounds";
        return false;
    }

    impl->primitives = makeSpan<MeshletStreamPrimitiveInfo>(impl->data, impl->header.primitiveOffset, impl->header.primitiveCount);
    impl->instances = makeSpan<MeshletStreamInstanceInfo>(impl->data, impl->header.instanceOffset, impl->header.instanceCount);
    impl->geometries = makeSpan<MeshletStreamGeometryInfo>(impl->data, impl->header.geometryOffset, impl->header.geometryCount);
    impl->lodLevels = makeSpan<MeshletStreamLodLevelInfo>(impl->data, impl->header.lodLevelOffset, impl->header.lodLevelCount);
    impl->groups = makeSpan<MeshletStreamGroupInfo>(impl->data, impl->header.groupInfoOffset, impl->header.groupCount);
    impl->clusterRefs = makeSpan<uint32_t>(impl->data, impl->header.clusterRefOffset, impl->header.clusterRefCount);
    impl->nodes = makeSpan<MeshletStreamNodeInfo>(impl->data, impl->header.nodeInfoOffset, impl->header.nodeCount);
    impl->pages = makeSpan<MeshletStreamPageInfo>(impl->data, impl->header.pageInfoOffset, impl->header.pageCount);
    impl->pageOffsets = makeSpan<uint64_t>(impl->data, impl->header.pageOffsetTableOffset, impl->header.pageCount);
    impl->sourceDependencyPaths = makeSpan<char>(
        impl->data,
        impl->header.sourceDependencyPathOffset,
        impl->header.sourceDependencyPathByteCount);

    size_t dependencyPathOffset = 0;
    for (uint32_t dependencyIndex = 0;
         dependencyIndex < impl->header.sourceDependencyCount;
         ++dependencyIndex) {
        if (dependencyPathOffset >= impl->sourceDependencyPaths.size()) {
            reason = "streamasset source dependency directory is truncated";
            return false;
        }
        const char* pathBegin = impl->sourceDependencyPaths.data() + dependencyPathOffset;
        const size_t remainingBytes = impl->sourceDependencyPaths.size() - dependencyPathOffset;
        const void* terminator = std::memchr(pathBegin, '\0', remainingBytes);
        if (terminator == nullptr || terminator == pathBegin) {
            reason = "streamasset source dependency path is invalid";
            return false;
        }
        dependencyPathOffset =
            static_cast<const char*>(terminator) - impl->sourceDependencyPaths.data() + 1u;
    }
    if (dependencyPathOffset != impl->sourceDependencyPaths.size()) {
        reason = "streamasset source dependency directory size is invalid";
        return false;
    }

    for (uint32_t primitiveIndex = 0; primitiveIndex < impl->primitives.size(); ++primitiveIndex) {
        const MeshletStreamPrimitiveInfo& primitive = impl->primitives[primitiveIndex];
        if (primitive.lodLevelCount == 0 ||
            primitive.pageCount == 0 ||
            primitive.groupCount == 0 ||
            primitive.pageCount != primitive.groupCount ||
            primitive.fallbackPageCount == 0 ||
            primitive.fallbackGroupCount == 0 ||
            primitive.lodLevelOffset > impl->lodLevels.size() ||
            primitive.lodLevelCount > impl->lodLevels.size() - primitive.lodLevelOffset ||
            primitive.pageOffset > impl->pages.size() ||
            primitive.pageCount > impl->pages.size() - primitive.pageOffset ||
            primitive.fallbackPageOffset > impl->pages.size() ||
            primitive.fallbackPageCount > impl->pages.size() - primitive.fallbackPageOffset ||
            primitive.groupOffset > impl->groups.size() ||
            primitive.groupCount > impl->groups.size() - primitive.groupOffset ||
            primitive.fallbackGroupOffset > impl->groups.size() ||
            primitive.fallbackGroupCount > impl->groups.size() - primitive.fallbackGroupOffset ||
            primitive.nodeOffset > impl->nodes.size() ||
            primitive.nodeCount > impl->nodes.size() - primitive.nodeOffset) {
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

    std::vector<uint8_t> primitiveHasTerminalGroup(impl->primitives.size(), 0);
    for (uint32_t groupIndex = 0; groupIndex < impl->groups.size(); ++groupIndex) {
        const MeshletStreamGroupInfo& group = impl->groups[groupIndex];
        if (group.primitiveIndex >= impl->primitives.size() ||
            group.pageIndex >= impl->pages.size() ||
            group.clusterCount == 0 ||
            group.clusterCount > kMeshletLodGroupSize ||
            !validStreamGroupMetric(group) ||
            group.clusterRefOffset > impl->clusterRefs.size() ||
            group.clusterCount > impl->clusterRefs.size() - group.clusterRefOffset) {
            reason = "streamasset group directory contains invalid ranges";
            return false;
        }
        const MeshletStreamPrimitiveInfo& primitive = impl->primitives[group.primitiveIndex];
        const MeshletStreamPageInfo& page = impl->pages[group.pageIndex];
        if (groupIndex < primitive.groupOffset ||
            groupIndex >= primitive.groupOffset + primitive.groupCount ||
            group.pageIndex != primitive.pageOffset + (groupIndex - primitive.groupOffset) ||
            page.primitiveIndex != group.primitiveIndex ||
            page.lodLevel != group.lodLevel ||
            page.lodGroupIndex != groupIndex ||
            page.clusterCount != group.clusterCount) {
            reason = "streamasset group directory does not match primitive/page directories";
            return false;
        }
        for (uint32_t clusterChild = 0; clusterChild < group.clusterCount; ++clusterChild) {
            const uint32_t refinedGroup = impl->clusterRefs[group.clusterRefOffset + clusterChild];
            if (refinedGroup != kMeshletStreamInvalidGroupIndex &&
                (refinedGroup < primitive.groupOffset ||
                    refinedGroup >= groupIndex)) {
                reason = "streamasset group DAG references an invalid refined group";
                return false;
            }
        }
        if (group.maxQuadricError == kMeshletStreamTerminalGroupError) {
            primitiveHasTerminalGroup[group.primitiveIndex] = 1;
        }
    }
    if (std::find(primitiveHasTerminalGroup.begin(), primitiveHasTerminalGroup.end(), 0) !=
        primitiveHasTerminalGroup.end()) {
        reason = "streamasset primitive has no terminal fallback group";
        return false;
    }
    if (!streamHierarchyValid(impl->primitives, impl->groups, impl->nodes)) {
        reason = "streamasset hierarchy node directory is invalid";
        return false;
    }

    for (uint32_t pageIndex = 0; pageIndex < impl->header.pageCount; ++pageIndex) {
        const MeshletStreamPageInfo& page = impl->pages[pageIndex];
        if (page.primitiveIndex >= impl->primitives.size() ||
            page.payloadOffset != impl->pageOffsets[pageIndex] ||
            page.payloadOffset % kFileAlignment != 0 ||
            page.payloadOffset > impl->dataSize ||
            page.payloadSize > impl->dataSize - page.payloadOffset ||
            page.uncompressedSize == 0 ||
            page.payloadSize == 0 ||
            page.payloadSize > std::numeric_limits<uint32_t>::max() ||
            !meshletStreamCompressionSupported(page.compressionMode) ||
            (page.compressionMode == static_cast<uint32_t>(MeshletStreamPayloadCompression::None) &&
                page.payloadSize != page.uncompressedSize) ||
            (page.compressionMode != static_cast<uint32_t>(MeshletStreamPayloadCompression::None) &&
                page.uncompressedSize < sizeof(MeshletStreamPayloadHeader)) ||
            page.uncompressedSize > impl->header.maxPagePayloadBytes) {
            reason = "streamasset page payload exceeds file bounds";
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
    if (!valid()) {
        return false;
    }
    std::vector<std::string_view> externalBufferUris;
    externalBufferUris.reserve(impl_->header.sourceDependencyCount);
    size_t dependencyPathOffset = 0;
    while (dependencyPathOffset < impl_->sourceDependencyPaths.size()) {
        const char* path = impl_->sourceDependencyPaths.data() + dependencyPathOffset;
        const size_t pathByteSize = std::strlen(path);
        externalBufferUris.emplace_back(path, pathByteSize);
        dependencyPathOffset += pathByteSize + 1u;
    }
    const uint64_t sourceDependencyFingerprint =
        sourceDependencyFingerprintFor(sourcePath, externalBufferUris);
    return
        sourceDependencyFingerprint != 0 &&
        impl_->header.sourceFileSize == sourceFileSizeFor(sourcePath) &&
        impl_->header.sourceWriteTime == sourceWriteTimeFor(sourcePath) &&
        impl_->header.sourceDependencyFingerprint == sourceDependencyFingerprint &&
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

uint32_t MeshletStreamAsset::groupCount() const
{
    return valid() ? impl_->header.groupCount : 0;
}

uint32_t MeshletStreamAsset::clusterRefCount() const
{
    return valid() ? impl_->header.clusterRefCount : 0;
}

uint32_t MeshletStreamAsset::nodeCount() const
{
    return valid() ? impl_->header.nodeCount : 0;
}

uint32_t MeshletStreamAsset::pageCount() const
{
    return valid() ? impl_->header.pageCount : 0;
}

uint32_t MeshletStreamAsset::maxPagePayloadBytes() const
{
    return valid() ? impl_->header.maxPagePayloadBytes : 0;
}

uint32_t MeshletStreamAsset::maxClusterVertices() const
{
    return valid() ? impl_->header.maxVertices : 0;
}

uint32_t MeshletStreamAsset::maxClusterTriangles() const
{
    return valid() ? impl_->header.maxTriangles : 0;
}

uint32_t MeshletStreamAsset::maxPageClusters() const
{
    return valid() ? impl_->header.lodGroupSize : 0;
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

std::span<const MeshletStreamGroupInfo> MeshletStreamAsset::groups() const
{
    return valid() ? impl_->groups : std::span<const MeshletStreamGroupInfo>{};
}

std::span<const uint32_t> MeshletStreamAsset::clusterRefinedGroups() const
{
    return valid() ? impl_->clusterRefs : std::span<const uint32_t>{};
}

std::span<const uint32_t> MeshletStreamAsset::groupClusterRefinedGroups(uint32_t groupIndex) const
{
    if (!valid() || groupIndex >= impl_->groups.size()) {
        return {};
    }
    const MeshletStreamGroupInfo& group = impl_->groups[groupIndex];
    return std::span<const uint32_t>(
        impl_->clusterRefs.data() + group.clusterRefOffset,
        group.clusterCount);
}

std::span<const MeshletStreamNodeInfo> MeshletStreamAsset::nodes() const
{
    return valid() ? impl_->nodes : std::span<const MeshletStreamNodeInfo>{};
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

    MeshletStreamPageInfo localPage = page;
    localPage.payloadOffset = 0;
    if (!validatePayloadHeader(
            storedPayload.data(),
            storedPayload.size(),
            localPage,
            0,
            reason)) {
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
    std::vector<std::string> externalBufferUris;
    if (!loadSourceDependencyUris(desc.sourcePath, externalBufferUris, reason)) {
        return false;
    }
    const uint64_t sourceDependencyFingerprint =
        sourceDependencyFingerprintFor(desc.sourcePath, externalBufferUris);
    if (sourceDependencyFingerprint == 0) {
        reason = "streamasset build cannot fingerprint source dependencies";
        return false;
    }
    std::ofstream stream;
    if (!openStreamAssetBuildFile(outputPath, stream, reason)) {
        return false;
    }

    MeshletStreamBuildState state;
    state.header = makeStreamFileHeader(desc.sourcePath, sourceDependencyFingerprint);
    if (!setStreamSourceDependencies(state, externalBufferUris, reason)) {
        return false;
    }
    if (!writeStreamAssetHeaderPlaceholder(stream, state.header, reason)) {
        return false;
    }

    const std::vector<RenderPrimitive>& renderPrimitives = desc.scene->renderPrimitives();
    std::vector<int32_t> primitiveMap(renderPrimitives.size(), kInvalidSceneIndex);
    for (size_t renderPrimitiveIndex = 0; renderPrimitiveIndex < renderPrimitives.size(); ++renderPrimitiveIndex) {
        int32_t primitiveIndex = kInvalidSceneIndex;
        if (!appendStreamPrimitivePages(
                stream,
                state,
                renderPrimitives[renderPrimitiveIndex],
                static_cast<uint32_t>(renderPrimitiveIndex),
                desc.compressionMode,
                primitiveIndex,
                reason)) {
            return false;
        }
        primitiveMap[renderPrimitiveIndex] = primitiveIndex;
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
        state.instances.push_back(instance);
    }

    if (sourceDependencyFingerprintFor(desc.sourcePath, externalBufferUris) !=
        sourceDependencyFingerprint) {
        reason = "streamasset source dependencies changed during build";
        return false;
    }
    return finalizeStreamAssetBuild(stream, state, reason);
}

bool buildMeshletStreamAssetOffline(const MeshletStreamAssetOfflineBuildDesc& desc, std::string& reason)
{
    reason.clear();
    if (desc.stats != nullptr) {
        *desc.stats = {};
    }
    if (desc.sourcePath.empty()) {
        reason = "streamasset offline build requires a source path";
        return false;
    }

    StreamGltfSource source;
    if (!loadGltfModelForStreamAssetBuilder(desc.sourcePath, source, reason)) {
        return false;
    }
    const uint64_t sourceDependencyFingerprint =
        sourceDependencyFingerprintFor(desc.sourcePath, source.externalBufferUris);
    if (sourceDependencyFingerprint == 0) {
        reason = "streamasset offline build cannot fingerprint source dependencies";
        return false;
    }
    source.stats = desc.stats;
    if (desc.stats != nullptr && source.rangeReadExternalBuffers) {
        desc.stats->usedExternalBufferRangeReads = 1;
        for (const uint64_t byteLength : source.bufferByteLengths) {
            if (byteLength > std::numeric_limits<uint64_t>::max() -
                    desc.stats->externalBufferDeclaredBytes) {
                desc.stats->externalBufferDeclaredBytes = std::numeric_limits<uint64_t>::max();
                break;
            }
            desc.stats->externalBufferDeclaredBytes += byteLength;
        }
    }
    const tinygltf::Model& model = source.model;
    if (model.scenes.empty()) {
        reason = "glTF model contains no scenes";
        return false;
    }

    const int32_t sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (!validGltfIndex(sceneIndex, model.scenes.size())) {
        reason = "glTF default scene index is out of range";
        return false;
    }

    const std::filesystem::path outputPath = desc.outputPath.empty()
        ? meshletStreamAssetPathFor(desc.sourcePath)
        : desc.outputPath;
    const std::filesystem::path partialPath = meshletStreamPartialPathFor(outputPath);
    {
        MeshletStreamAsset existingAsset;
        std::string existingReason;
        if (existingAsset.open(outputPath, existingReason) &&
            existingAsset.isCurrentForSource(desc.sourcePath)) {
            bool compressionMatches = true;
            for (const MeshletStreamPageInfo& page : existingAsset.pages()) {
                if (page.compressionMode != static_cast<uint32_t>(desc.compressionMode)) {
                    compressionMatches = false;
                    break;
                }
            }
            if (compressionMatches) {
                std::error_code removeError;
                std::filesystem::remove(partialPath, removeError);
                return true;
            }
        }
    }

    std::fstream stream;
    MeshletStreamBuildState state;
    std::unordered_map<uint64_t, uint32_t> primitiveMap;
    primitiveMap.reserve(model.meshes.size() * 2u);
    uint64_t payloadWriteOffset = 0;
    const bool resumedPartial = loadPartialBuildState(
        partialPath,
        outputPath,
        desc.sourcePath,
        sourceDependencyFingerprint,
        desc.compressionMode,
        state,
        primitiveMap,
        payloadWriteOffset);
    if (resumedPartial) {
        if (!openStreamAssetResumeFile(outputPath, payloadWriteOffset, stream, reason)) {
            return false;
        }
    } else {
        std::error_code removeError;
        std::filesystem::remove(partialPath, removeError);
        state.header = makeStreamFileHeader(desc.sourcePath, sourceDependencyFingerprint);
        if (!openStreamAssetBuildFile(outputPath, stream, reason)) {
            return false;
        }
        if (!writeStreamAssetHeaderPlaceholder(stream, state.header, reason)) {
            return false;
        }
    }
    if (!setStreamSourceDependencies(state, source.externalBufferUris, reason)) {
        return false;
    }

    MeshletStreamPartialBuildContext partialContext{
        .partialPath = partialPath,
        .maxNewGeometriesPerInvocation = desc.maxNewGeometriesPerInvocation,
    };
    if (!buildStreamAssetGeometryPayloadsFromGltf(
            stream,
            state,
            source,
            desc.compressionMode,
            primitiveMap,
            &partialContext,
            reason)) {
        return false;
    }
    if (partialContext.paused) {
        reason = "streamasset offline build paused after geometry budget; rerun the same build to resume";
        return false;
    }
    if (!appendStreamAssetInstancesFromGltf(state, source, sceneIndex, primitiveMap, reason)) {
        return false;
    }

    if (sourceDependencyFingerprintFor(desc.sourcePath, source.externalBufferUris) !=
        sourceDependencyFingerprint) {
        reason = "streamasset source dependencies changed during offline build";
        return false;
    }

    if (!finalizeStreamAssetBuild(stream, state, reason)) {
        return false;
    }
    std::error_code removeError;
    std::filesystem::remove(partialPath, removeError);
    return true;
}

} // namespace metallic::scene

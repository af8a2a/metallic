#pragma once

#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace metallic::scene {

inline constexpr const char* kMeshletStreamAssetSuffix = ".meshstream.bin";
inline constexpr uint32_t kMeshletStreamInvalidGroupIndex = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kMeshletStreamInvalidNodeIndex = std::numeric_limits<uint32_t>::max();
inline constexpr float kMeshletStreamTerminalGroupError = std::numeric_limits<float>::max();
inline constexpr uint32_t kMeshletStreamGroupTerminal = 1u;

inline constexpr uint32_t kMeshletStreamPayloadAttributePosition = 1u << 0u;
inline constexpr uint32_t kMeshletStreamPayloadAttributeNormal = 1u << 1u;
inline constexpr uint32_t kMeshletStreamPayloadAttributeTexcoord0 = 1u << 2u;
inline constexpr uint32_t kMeshletStreamPayloadAttributeMaterial = 1u << 3u;
inline constexpr uint32_t kMeshletStreamPayloadAttributeTangent = 1u << 4u;

enum class MeshletStreamPayloadCompression : uint32_t {
    None = 0,
    ByteRle = 1,
    GpuTiles = 2,
};

enum class MeshletStreamPayloadFormat : uint32_t {
    Unknown = 0,
    Float32x2 = 1,
    Float32x4 = 2,
    Uint32 = 3,
    Float32x3 = 4,
};

struct MeshletStreamBounds {
    float min[3] = {};
    float max[3] = {};
    uint32_t valid = 0;
};

struct MeshletStreamPrimitiveInfo {
    uint32_t renderPrimitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t lodLevelOffset = 0;
    uint32_t lodLevelCount = 0;
    uint32_t pageOffset = 0;
    uint32_t pageCount = 0;
    uint32_t fallbackPageOffset = 0;
    uint32_t fallbackPageCount = 0;
    uint32_t groupOffset = 0;
    uint32_t groupCount = 0;
    uint32_t fallbackGroupOffset = 0;
    uint32_t fallbackGroupCount = 0;
    uint32_t nodeOffset = 0;
    uint32_t nodeCount = 0;
    MeshletStreamBounds bounds;
};

struct MeshletStreamInstanceInfo {
    uint32_t renderNodeIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t visible = 1;
    float worldMatrix[16] = {};
};

struct MeshletStreamGeometryInfo {
    uint32_t primitiveIndex = 0;
    uint32_t renderPrimitiveIndex = 0;
    uint32_t pageOffset = 0;
    uint32_t pageCount = 0;
    uint32_t pagePayloadOffsetTableOffset = 0;
    uint32_t pagePayloadOffsetTableCount = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint64_t payloadFileOffset = 0;
    uint64_t payloadFileSize = 0;
};

struct MeshletStreamLodLevelInfo {
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t pageOffset = 0;
    uint32_t pageCount = 0;
    uint32_t clusterCount = 0;
    float minBoundingSphereRadius = 0.0f;
    float minMaxQuadricError = 0.0f;
};

struct MeshletStreamGroupInfo {
    uint32_t primitiveIndex = 0;
    uint32_t pageIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t clusterCount = 0;
    float boundsCenterRadius[4] = {};
    float maxQuadricError = 0.0f;
    uint32_t clusterRefinedOffset = 0;
    uint32_t flags = 0;
};

struct MeshletStreamNodeInfo {
    uint32_t primitiveIndex = 0;
    uint32_t childOffset = 0;
    uint32_t childCount = 0;
    uint32_t groupIndex = kMeshletStreamInvalidGroupIndex;
    float boundsCenterRadius[4] = {};
    float maxQuadricError = 0.0f;
    uint32_t lodLevel = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

struct MeshletStreamPageInfo {
    uint64_t payloadOffset = 0;
    uint64_t payloadSize = 0;
    uint64_t uncompressedSize = 0;
    uint32_t primitiveIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t lodGroupIndex = 0;
    uint32_t clusterCount = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleIndexCount = 0;
    uint32_t materialIndex = 0;
    uint32_t attributeFlags = 0;
    uint32_t compressionMode = static_cast<uint32_t>(MeshletStreamPayloadCompression::None);
    uint32_t payloadFlags = 0;
    uint32_t primitiveGroupOffset = 0;
    MeshletStreamBounds bounds;
    float maxQuadricError = 0.0f;
    // Former tail padding in the unchanged 104-byte disk record. Initialize it
    // explicitly so independent/resumed builds produce identical file bytes.
    uint32_t reserved0 = 0;
};

// Disk metadata keeps the original uncompressed size. Legacy float4 pages are
// compacted at decode time, without changing their on-disk cache or XYZ bits.
inline constexpr uint32_t kMeshletStreamPayloadCompactPositions = 1u;
inline constexpr uint32_t meshletStreamPositionStride(uint32_t format)
{
    return format == static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x3) ? 12u :
        format == static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x4) ? 16u : 0u;
}
inline constexpr uint64_t meshletStreamDevicePayloadSize(const MeshletStreamPageInfo& page)
{
    const uint64_t savings = (page.payloadFlags & kMeshletStreamPayloadCompactPositions) != 0u
        ? 0u : (uint64_t(page.vertexCount) * 4u & ~uint64_t(15));
    return savings <= page.uncompressedSize ? page.uncompressedSize - savings : 0u;
}

struct MeshletStreamPayloadHeader {
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t clusterCount = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleIndexCount = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t lodGroupIndex = 0;
    uint32_t clusterOffsetBytes = 0;
    uint32_t positionOffsetBytes = 0;
    uint32_t triangleOffsetBytes = 0;
    uint32_t payloadByteSize = 0;
    uint32_t uncompressedPayloadByteSize = 0;
    uint32_t attributeFlags = 0;
    uint32_t compressionMode = static_cast<uint32_t>(MeshletStreamPayloadCompression::None);
    uint32_t normalOffsetBytes = 0;
    uint32_t texcoord0OffsetBytes = 0;
    uint32_t materialOffsetBytes = 0;
    uint32_t materialCount = 0;
    uint32_t positionFormat = static_cast<uint32_t>(MeshletStreamPayloadFormat::Float32x4);
    uint32_t normalFormat = static_cast<uint32_t>(MeshletStreamPayloadFormat::Unknown);
    uint32_t texcoord0Format = static_cast<uint32_t>(MeshletStreamPayloadFormat::Unknown);
    uint32_t materialFormat = static_cast<uint32_t>(MeshletStreamPayloadFormat::Uint32);
    uint32_t tangentOffsetBytes = 0;
    uint32_t tangentFormat = static_cast<uint32_t>(MeshletStreamPayloadFormat::Unknown);
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

struct MeshletStreamPayloadCluster {
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleOffset = 0;
    uint32_t triangleCount = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t lodLevel = 0;
    uint32_t lodGroupIndex = 0;
    uint32_t refinedGroupIndex = kMeshletStreamInvalidGroupIndex;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
    float boundingSphere[4] = {};
    float coneApexCutoff[4] = {};
    float coneAxisLodError[4] = {};
};

class MeshletStreamAsset {
public:
    MeshletStreamAsset();
    ~MeshletStreamAsset();

    MeshletStreamAsset(MeshletStreamAsset&&) noexcept;
    MeshletStreamAsset& operator=(MeshletStreamAsset&&) noexcept;

    MeshletStreamAsset(const MeshletStreamAsset&) = delete;
    MeshletStreamAsset& operator=(const MeshletStreamAsset&) = delete;

    bool open(const std::filesystem::path& path, std::string& reason);
    void close();

    bool valid() const;
    const std::filesystem::path& path() const { return path_; }
    bool isCurrentForSource(const std::filesystem::path& sourcePath) const;
    // Runtime may reuse revision 0 position-only assets. Offline cooking still
    // requires isCurrentForSource so attributed assets get the latest rules.
    bool isRuntimeCompatibleForSource(const std::filesystem::path& sourcePath, std::string& reason) const;
    uint32_t cookRevision() const;

    uint32_t primitiveCount() const;
    uint32_t instanceCount() const;
    uint32_t geometryCount() const;
    uint32_t lodLevelCount() const;
    uint32_t groupCount() const;
    uint32_t nodeCount() const;
    uint32_t pageCount() const;
    uint32_t maxPagePayloadBytes() const;
    uint32_t maxClusterVertices() const;
    uint32_t maxClusterTriangles() const;
    uint32_t maxPageClusters() const;
    uint64_t sourceFileSize() const;
    int64_t sourceWriteTime() const;

    std::span<const MeshletStreamPrimitiveInfo> primitives() const;
    std::span<const MeshletStreamInstanceInfo> instances() const;
    std::span<const MeshletStreamGeometryInfo> geometries() const;
    std::span<const MeshletStreamLodLevelInfo> lodLevels() const;
    std::span<const MeshletStreamGroupInfo> groups() const;
    // One global refined group index per cluster, independent of page residency.
    std::span<const uint32_t> refinedGroups() const;
    // Roots of every DAG branch, including branches that terminate at a finer LOD.
    std::span<const uint32_t> terminalGroups() const;
    std::span<const uint32_t> primitiveTerminalGroups(uint32_t primitiveIndex) const;
    std::span<const MeshletStreamNodeInfo> nodes() const;
    std::span<const MeshletStreamPageInfo> pages() const;
    std::span<const uint64_t> pagePayloadOffsets() const;
    std::span<const uint64_t> geometryPagePayloadOffsets(uint32_t geometryIndex) const;
    std::span<const uint8_t> pagePayload(uint32_t pageIndex) const;

private:
    struct Impl;

    bool validateSourceDependencies(const std::filesystem::path& sourcePath, std::string& reason) const;

    std::filesystem::path path_;
    Impl* impl_ = nullptr;
};

struct MeshletStreamAssetBuildDesc {
    const Scene* scene = nullptr;
    std::filesystem::path sourcePath;
    std::filesystem::path outputPath;
    MeshletStreamPayloadCompression compressionMode = MeshletStreamPayloadCompression::None;
};

struct MeshletStreamAssetOfflineBuildStats {
    uint64_t externalBufferDeclaredBytes = 0;
    uint64_t accessorRangeReadBytes = 0;
    uint64_t maxAccessorRangeReadBytes = 0;
    uint32_t accessorRangeReadCount = 0;
    uint32_t usedExternalBufferRangeReads = 0;
    uint32_t partialCheckpointCount = 0;
};

struct MeshletStreamCookProgress {
    const char* phase = "";
    uint32_t sourcePrimitiveIndex = 0;
    uint32_t meshIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t completedGeometries = 0;
    uint64_t vertices = 0;
    uint64_t triangles = 0;
    uint64_t payloadBytes = 0;
    uint64_t clusters = 0;
    uint32_t groups = 0;
    double decodeSeconds = 0;
    double buildSeconds = 0;
    double encodeSeconds = 0;
};

struct MeshletStreamAssetOfflineBuildDesc {
    std::filesystem::path sourcePath;
    std::filesystem::path outputPath;
    MeshletStreamPayloadCompression compressionMode = MeshletStreamPayloadCompression::None;
    uint32_t maxNewGeometriesPerInvocation = 0;
    uint32_t partialCheckpointGeometryInterval = 64;
    MeshletStreamAssetOfflineBuildStats* stats = nullptr;
    MeshletBuildOptions meshletOptions;
    std::function<void(const MeshletStreamCookProgress&)> progress;
};

bool decodeMeshletStreamPayloadForDevice(
    const MeshletStreamPageInfo& page,
    std::span<const uint8_t> storedPayload,
    std::vector<uint8_t>& scratchPayload,
    std::span<const uint8_t>& outDevicePayload,
    std::string& reason);
bool validateMeshletStreamDeviceHeader(const MeshletStreamPayloadHeader& header,
    const MeshletStreamPageInfo& page, std::string& reason);
bool buildMeshletStreamAsset(const MeshletStreamAssetBuildDesc& desc, std::string& reason);
bool buildMeshletStreamAssetOffline(const MeshletStreamAssetOfflineBuildDesc& desc, std::string& reason);

struct MeshletStreamAttributeValidation {
    uint32_t sourcePrimitive = 0;
    uint64_t sourceVertices = 0;
    uint64_t preparedVertices = 0;
    uint64_t lod0Triangles = 0;
    uint64_t verticesAllLods = 0;
    uint64_t positionBytes = 0;
    uint64_t normalBytes = 0;
    uint64_t uvBytes = 0;
    uint64_t tangentBytes = 0;
    uint64_t triangleBytes = 0;
    uint64_t clusterBytes = 0;
    uint64_t negativeTangentVertices = 0;
};

// Offline validation: range-decode one source primitive at a time. All-LOD
// vertex tuples must match source attributes exactly; LOD0 preserves the full
// triangle multiset and winding. Does not load images or verify pixel shading.
bool validateMeshletStreamAttributes(const MeshletStreamAsset& asset,
    const std::filesystem::path& sourcePath, std::vector<MeshletStreamAttributeValidation>& results,
    std::string& reason);
std::filesystem::path meshletStreamAssetPathFor(const std::filesystem::path& sourcePath);

} // namespace metallic::scene

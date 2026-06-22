#pragma once

#include "Runtime/Scene/Scene.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metallic::scene {

inline constexpr const char* kMeshletStreamAssetSuffix = ".meshstream.bin";

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
    MeshletStreamBounds bounds;
};

struct MeshletStreamInstanceInfo {
    uint32_t renderNodeIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t materialIndex = 0;
    uint32_t visible = 1;
    float worldMatrix[16] = {};
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
    uint32_t reserved = 0;
    MeshletStreamBounds bounds;
    float maxQuadricError = 0.0f;
};

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
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t reserved2 = 0;
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

    uint32_t primitiveCount() const;
    uint32_t instanceCount() const;
    uint32_t lodLevelCount() const;
    uint32_t pageCount() const;
    uint32_t maxPagePayloadBytes() const;
    uint64_t sourceFileSize() const;
    int64_t sourceWriteTime() const;

    std::span<const MeshletStreamPrimitiveInfo> primitives() const;
    std::span<const MeshletStreamInstanceInfo> instances() const;
    std::span<const MeshletStreamLodLevelInfo> lodLevels() const;
    std::span<const MeshletStreamPageInfo> pages() const;
    std::span<const uint64_t> pagePayloadOffsets() const;
    std::span<const uint8_t> pagePayload(uint32_t pageIndex) const;

private:
    struct Impl;

    std::filesystem::path path_;
    Impl* impl_ = nullptr;
};

struct MeshletStreamAssetBuildDesc {
    const Scene* scene = nullptr;
    std::filesystem::path sourcePath;
    std::filesystem::path outputPath;
};

bool buildMeshletStreamAsset(const MeshletStreamAssetBuildDesc& desc, std::string& reason);
std::filesystem::path meshletStreamAssetPathFor(const std::filesystem::path& sourcePath);

} // namespace metallic::scene

#pragma once

#include "Runtime/Scene/MeshletStreamAsset.h"

namespace metallic::scene {

// The transport envelope is separate from the final GPU payload. Offsets are
// relative to the stored envelope or decoded page, never device addresses.
struct MeshletStreamGpuTile {
    uint32_t sourceOffset = 0;
    uint32_t storedBytes = 0;
    uint32_t destinationOffset = 0;
    uint32_t decodedBytes = 0;
    uint32_t codec = 0; // 0: raw; 1: GDeflate 1.0.
    uint32_t checksum = 0;
};

struct MeshletStreamGpuCluster {
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleOffset = 0;
    uint32_t triangleCount = 0;
    uint32_t materialIndex = 0;
    uint32_t reserved = 0;
};

struct MeshletStreamGpuPage {
    MeshletStreamPayloadHeader header;
    std::vector<MeshletStreamGpuTile> tiles;
    std::vector<MeshletStreamGpuCluster> clusters;
};

bool encodeMeshletStreamGpuPage(std::span<const uint8_t> decoded, bool compress,
    std::vector<uint8_t>& stored, std::string& reason);
bool inspectMeshletStreamGpuPage(const MeshletStreamPageInfo& page,
    std::span<const uint8_t> stored, MeshletStreamGpuPage& result, std::string& reason);
bool decodeMeshletStreamGpuPage(const MeshletStreamPageInfo& page,
    std::span<const uint8_t> stored, std::vector<uint8_t>& decoded, std::string& reason);

// Preserves topology/IDs and writes a new file; never overwrites the source.
bool transcodeMeshletStreamAsset(const std::filesystem::path& source,
    const std::filesystem::path& destination, bool compress, std::string& reason,
    const std::function<void(uint32_t, uint32_t)>& progress = {});

} // namespace metallic::scene

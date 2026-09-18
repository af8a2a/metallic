#pragma once
#include "Runtime/Render/Streamer/MeshletStreamClas.h"
#include "Runtime/Scene/MeshletStreamGpuCodec.h"
#include "Runtime/Scene/Scene.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace metallic::tests {
inline void requireGpuPage(bool value, const std::string& reason)
{
    if (!value) { throw std::runtime_error(reason); }
}

inline uint32_t referenceGpuPageCrc32c(std::span<const uint8_t> bytes)
{
    uint32_t crc = ~0u;
    for (const uint8_t byte : bytes) {
        crc ^= byte;
        for (uint32_t bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82f63b78u : 0u);
        }
    }
    return ~crc;
}

inline std::vector<uint8_t> makeMixedTileGpuPagePayload()
{
    // Keep the geometry wholly inside the first tile. The second tile is
    // incompressible padding; the third exercises a short final tile.
    std::vector<uint8_t> decoded(131088, 0);
    scene::MeshletStreamPayloadHeader header{.magic = 0x4d535047u, .version = 4, .clusterCount = 1,
        .vertexCount = 3, .triangleIndexCount = 3, .clusterOffsetBytes = 112, .positionOffsetBytes = 208,
        .triangleOffsetBytes = 256, .payloadByteSize = uint32_t(decoded.size()), .uncompressedPayloadByteSize = uint32_t(decoded.size()),
        .attributeFlags = scene::kMeshletStreamPayloadAttributePosition, .positionFormat = uint32_t(scene::MeshletStreamPayloadFormat::Float32x3)};
    scene::MeshletStreamPayloadCluster cluster{.vertexCount = 3, .triangleCount = 1};
    std::memcpy(decoded.data(), &header, sizeof(header));
    std::memcpy(decoded.data() + header.clusterOffsetBytes, &cluster, sizeof(cluster));
    decoded[header.triangleOffsetBytes + 1] = 1;
    decoded[header.triangleOffsetBytes + 2] = 2;
    uint32_t random = 12345;
    for (size_t i = 65536; i < 131072; ++i) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        decoded[i] = uint8_t(random);
    }
    return decoded;
}

inline void checkGpuPageCodec(const std::filesystem::path& directory)
{
    const auto source = std::filesystem::path(PROJECT_SOURCE_DIR) / "Asset/StandfordBunny/scene.gltf";
    const auto originalPath = directory / "gpu_page_source.meshstream.bin";
    scene::Scene model;
    requireGpuPage(model.load(source), model.lastLoadResult().error);
    std::string reason;
    requireGpuPage(scene::buildMeshletStreamAsset({.scene = &model, .sourcePath = source,
        .outputPath = originalPath}, reason), reason);
    scene::MeshletStreamAsset original;
    requireGpuPage(original.open(originalPath, reason), reason);
    uint64_t compressedTiles = 0;
    for (bool compress : {false, true}) {
        const auto path = directory / (compress ? "gpu_page_compressed.meshstream.bin" : "gpu_page_raw.meshstream.bin");
        std::filesystem::remove(path);
        requireGpuPage(scene::transcodeMeshletStreamAsset(originalPath, path, compress, reason), reason);
        requireGpuPage(!scene::transcodeMeshletStreamAsset(originalPath, path, compress, reason), "Transcoder overwrote existing file");
        scene::MeshletStreamAsset encoded;
        requireGpuPage(encoded.open(path, reason), reason);
        requireGpuPage(encoded.pageCount() == original.pageCount() && encoded.groupCount() == original.groupCount() &&
            encoded.nodeCount() == original.nodeCount(), "Transcoding changed topology counts");
        requireGpuPage(std::memcmp(encoded.groups().data(), original.groups().data(), original.groups().size_bytes()) == 0 &&
            std::memcmp(encoded.nodes().data(), original.nodes().data(), original.nodes().size_bytes()) == 0 &&
            std::ranges::equal(encoded.refinedGroups(), original.refinedGroups()), "Transcoding changed topology bytes");
        for (uint32_t page = 0; page < original.pageCount(); ++page) {
            std::vector<uint8_t> scratch, decoded;
            std::span<const uint8_t> reference, output;
            requireGpuPage(scene::decodeMeshletStreamPayloadForDevice(original.pages()[page], original.pagePayload(page), scratch, reference, reason), reason);
            requireGpuPage(scene::decodeMeshletStreamPayloadForDevice(encoded.pages()[page], encoded.pagePayload(page), decoded, output, reason), reason);
            requireGpuPage(std::ranges::equal(reference, output), "CPU transcode round trip differs");
            scene::MeshletStreamGpuPage metadata;
            requireGpuPage(scene::inspectMeshletStreamGpuPage(encoded.pages()[page], encoded.pagePayload(page), metadata, reason), reason);
            for (const auto& tile : metadata.tiles) {
                compressedTiles += tile.codec;
                requireGpuPage(referenceGpuPageCrc32c(encoded.pagePayload(page).subspan(tile.sourceOffset, tile.storedBytes)) == tile.checksum,
                    "Stored CRC32C differs from independent bitwise oracle");
            }
            render::MeshletStreamClasPagePlan cpu, gpu;
            requireGpuPage(render::buildMeshletStreamClasPagePlan(original.pages()[page], reference, page, page * original.maxPageClusters(), cpu, reason), reason);
            requireGpuPage(render::buildMeshletStreamClasGpuPagePlan(metadata, page, page * original.maxPageClusters(), gpu, reason), reason);
            requireGpuPage(cpu.clusters.size() == gpu.clusters.size() && std::memcmp(cpu.clusters.data(), gpu.clusters.data(),
                cpu.clusters.size() * sizeof(render::MeshletStreamClasClusterInput)) == 0, "CLAS sideband differs from decoded plan");
            auto corrupt = std::vector<uint8_t>(encoded.pagePayload(page).begin(), encoded.pagePayload(page).end());
            corrupt.back() ^= 1;
            requireGpuPage(!scene::inspectMeshletStreamGpuPage(encoded.pages()[page], corrupt, metadata, reason), "Corrupt tile accepted");
            corrupt.pop_back();
            requireGpuPage(!scene::inspectMeshletStreamGpuPage(encoded.pages()[page], corrupt, metadata, reason), "Truncated envelope accepted");
        }
    }
    requireGpuPage(compressedTiles > 0, "Fixture did not exercise GDeflate");
}
} // namespace metallic::tests

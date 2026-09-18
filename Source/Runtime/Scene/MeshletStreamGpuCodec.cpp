#include "Runtime/Scene/MeshletStreamGpuCodec.h"

#include <libdeflate.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <memory>

namespace metallic::scene {
namespace {

constexpr uint32_t kMagic = 0x4750534du;
constexpr uint32_t kTileBytes = 65536;
struct Envelope {
    uint32_t magic = kMagic;
    uint32_t version = 1;
    uint32_t storedBytes = 0;
    uint32_t tileCount = 0;
    uint32_t metadataChecksum = 0;
    uint32_t reserved = 0;
    MeshletStreamPayloadHeader payload;
};
static_assert(sizeof(Envelope) == 136);
static_assert(sizeof(MeshletStreamGpuTile) == 24);
static_assert(sizeof(MeshletStreamGpuCluster) == 24);

uint32_t checksum(std::span<const uint8_t> data)
{
    // CRC32C slicing-by-eight. Validate the exact same stored bytes without a
    // serial table dependency per byte on the streaming worker's hot path.
    static const auto table = [] {
        std::array<std::array<uint32_t, 256>, 8> values{};
        for (uint32_t i = 0; i < values[0].size(); ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) { value = (value >> 1) ^ ((value & 1) ? 0x82f63b78u : 0u); }
            values[0][i] = value;
        }
        for (uint32_t slice = 1; slice < values.size(); ++slice) {
            for (uint32_t i = 0; i < values[slice].size(); ++i) {
                const uint32_t previous = values[slice - 1][i];
                values[slice][i] = values[0][previous & 255] ^ (previous >> 8);
            }
        }
        return values;
    }();
    uint32_t value = ~0u;
    size_t offset = 0;
    for (; data.size() - offset >= sizeof(uint64_t); offset += sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, data.data() + offset, sizeof(word));
        if constexpr (std::endian::native == std::endian::big) { word = std::byteswap(word); }
        word ^= value;
        value = table[7][word & 255] ^ table[6][(word >> 8) & 255] ^
            table[5][(word >> 16) & 255] ^ table[4][(word >> 24) & 255] ^
            table[3][(word >> 32) & 255] ^ table[2][(word >> 40) & 255] ^
            table[1][(word >> 48) & 255] ^ table[0][word >> 56];
    }
    for (uint8_t byte : data.subspan(offset)) { value = table[0][(value ^ byte) & 255] ^ (value >> 8); }
    return ~value;
}

bool range(uint64_t offset, uint64_t size, uint64_t capacity)
{
    return offset <= capacity && size <= capacity - offset;
}

using Compressor = std::unique_ptr<libdeflate_gdeflate_compressor, decltype(&libdeflate_free_gdeflate_compressor)>;
using Decompressor = std::unique_ptr<libdeflate_gdeflate_decompressor, decltype(&libdeflate_free_gdeflate_decompressor)>;

} // namespace

bool encodeMeshletStreamGpuPage(std::span<const uint8_t> decoded, bool compress,
    std::vector<uint8_t>& stored, std::string& reason)
{
    stored.clear();
    if (decoded.size() < sizeof(MeshletStreamPayloadHeader) || decoded.size() > UINT32_MAX || decoded.size() % 4) {
        reason = "GPU page size is invalid"; return false;
    }
    Envelope envelope;
    std::memcpy(&envelope.payload, decoded.data(), sizeof(envelope.payload));
    const auto& header = envelope.payload;
    MeshletStreamPageInfo page;
    page.payloadSize = page.uncompressedSize = decoded.size();
    page.payloadFlags = kMeshletStreamPayloadCompactPositions;
    page.clusterCount = header.clusterCount;
    page.vertexCount = header.vertexCount;
    page.triangleIndexCount = header.triangleIndexCount;
    page.primitiveIndex = header.primitiveIndex;
    page.materialIndex = header.materialIndex;
    page.lodLevel = header.lodLevel;
    page.lodGroupIndex = header.lodGroupIndex;
    page.attributeFlags = header.attributeFlags;
    std::vector<uint8_t> scratch;
    std::span<const uint8_t> validated;
    if (!decodeMeshletStreamPayloadForDevice(page, decoded, scratch, validated, reason)) { return false; }
    if (header.positionFormat != uint32_t(MeshletStreamPayloadFormat::Float32x3) ||
        header.payloadByteSize != decoded.size() || header.compressionMode != 0 ||
        !range(header.clusterOffsetBytes, uint64_t(header.clusterCount) * sizeof(MeshletStreamPayloadCluster), decoded.size())) {
        reason = "GPU page requires canonical compact payload"; return false;
    }
    envelope.tileCount = uint32_t((decoded.size() + kTileBytes - 1) / kTileBytes);
    const uint64_t clusterOffset = sizeof(Envelope) + uint64_t(envelope.tileCount) * sizeof(MeshletStreamGpuTile);
    const uint64_t metadataSize = clusterOffset + uint64_t(header.clusterCount) * sizeof(MeshletStreamGpuCluster);
    if (metadataSize + decoded.size() + uint64_t(envelope.tileCount) * 15 > UINT32_MAX) {
        reason = "GPU page metadata is too large"; return false;
    }
    stored.resize(size_t(metadataSize));
    for (uint32_t i = 0; i < header.clusterCount; ++i) {
        MeshletStreamPayloadCluster cluster;
        std::memcpy(&cluster, decoded.data() + header.clusterOffsetBytes + uint64_t(i) * sizeof(cluster), sizeof(cluster));
        const MeshletStreamGpuCluster input{cluster.vertexOffset, cluster.vertexCount,
            cluster.triangleOffset, cluster.triangleCount, cluster.materialIndex};
        std::memcpy(stored.data() + clusterOffset + uint64_t(i) * sizeof(input), &input, sizeof(input));
    }
    thread_local Compressor compressor(libdeflate_alloc_gdeflate_compressor(1), libdeflate_free_gdeflate_compressor);
    if (compress && !compressor) { reason = "GDeflate compressor allocation failed"; return false; }
    for (uint32_t i = 0; i < envelope.tileCount; ++i) {
        const auto input = decoded.subspan(uint64_t(i) * kTileBytes,
            std::min<size_t>(kTileBytes, decoded.size() - uint64_t(i) * kTileBytes));
        std::vector<uint8_t> compressed;
        bool useCompressed = false;
        if (compress) {
            size_t pageCount = 0;
            compressed.resize(libdeflate_gdeflate_compress_bound(compressor.get(), input.size(), &pageCount));
            libdeflate_gdeflate_out_page output{compressed.data(), compressed.size()};
            if (pageCount != 1 || !libdeflate_gdeflate_compress(compressor.get(), input.data(), input.size(), &output, 1)) {
                reason = "GDeflate compression failed"; return false;
            }
            compressed.resize(output.nbytes);
            useCompressed = compressed.size() < input.size();
        }
        const std::span<const uint8_t> bytes = useCompressed ? std::span<const uint8_t>(compressed) : input;
        stored.resize((stored.size() + 15) & ~size_t(15), 0);
        MeshletStreamGpuTile tile{uint32_t(stored.size()), uint32_t(bytes.size()),
            i * kTileBytes, uint32_t(input.size()), useCompressed ? 1u : 0u, checksum(bytes)};
        stored.insert(stored.end(), bytes.begin(), bytes.end());
        std::memcpy(stored.data() + sizeof(Envelope) + uint64_t(i) * sizeof(tile), &tile, sizeof(tile));
    }
    envelope.storedBytes = uint32_t(stored.size());
    std::memcpy(stored.data(), &envelope, sizeof(envelope));
    envelope.metadataChecksum = checksum(std::span(stored).first(size_t(metadataSize)));
    std::memcpy(stored.data(), &envelope, sizeof(envelope));
    return true;
}

bool inspectMeshletStreamGpuPage(const MeshletStreamPageInfo& page,
    std::span<const uint8_t> stored, MeshletStreamGpuPage& result, std::string& reason)
{
    result = {};
    const auto fail = [&] { reason = "GPU page envelope, checksum or range is invalid"; return false; };
    if (stored.size() < sizeof(Envelope) || stored.size() != page.payloadSize ||
        !(page.payloadFlags & kMeshletStreamPayloadCompactPositions)) { return fail(); }
    Envelope envelope;
    std::memcpy(&envelope, stored.data(), sizeof(envelope));
    const auto& header = envelope.payload;
    if (envelope.magic != kMagic || envelope.version != 1 || envelope.reserved != 0 ||
        envelope.storedBytes != stored.size() || page.uncompressedSize == 0 || page.uncompressedSize > UINT32_MAX ||
        envelope.tileCount != (page.uncompressedSize + kTileBytes - 1) / kTileBytes ||
        header.magic != 0x4d535047u || header.version != 4 ||
        header.clusterCount != page.clusterCount || header.vertexCount != page.vertexCount ||
        header.triangleIndexCount != page.triangleIndexCount || header.attributeFlags != page.attributeFlags ||
        header.primitiveIndex != page.primitiveIndex || header.materialIndex != page.materialIndex ||
        header.lodLevel != page.lodLevel || header.lodGroupIndex != page.lodGroupIndex ||
        header.payloadByteSize != page.uncompressedSize || header.uncompressedPayloadByteSize != page.uncompressedSize ||
        header.compressionMode != 0 || header.positionFormat != uint32_t(MeshletStreamPayloadFormat::Float32x3) ||
        !range(header.positionOffsetBytes, uint64_t(header.vertexCount) * 12, page.uncompressedSize) ||
        !range(header.triangleOffsetBytes, header.triangleIndexCount, page.uncompressedSize)) { return fail(); }
    const uint64_t clusterOffset = sizeof(Envelope) + uint64_t(envelope.tileCount) * sizeof(MeshletStreamGpuTile);
    const uint64_t metadataSize = clusterOffset + uint64_t(header.clusterCount) * sizeof(MeshletStreamGpuCluster);
    if (metadataSize > stored.size() || !validateMeshletStreamDeviceHeader(header, page, reason)) { return fail(); }
    std::vector<uint8_t> metadata(stored.begin(), stored.begin() + size_t(metadataSize));
    const uint32_t expected = envelope.metadataChecksum;
    envelope.metadataChecksum = 0;
    std::memcpy(metadata.data(), &envelope, sizeof(envelope));
    if (checksum(metadata) != expected) { return fail(); }
    result.header = header;
    result.tiles.resize(envelope.tileCount);
    result.clusters.resize(header.clusterCount);
    std::memcpy(result.tiles.data(), stored.data() + sizeof(Envelope), result.tiles.size() * sizeof(MeshletStreamGpuTile));
    std::memcpy(result.clusters.data(), stored.data() + clusterOffset, result.clusters.size() * sizeof(MeshletStreamGpuCluster));
    uint64_t decodedEnd = 0, storedEnd = metadataSize;
    for (const auto& tile : result.tiles) {
        if (tile.sourceOffset % 4 || tile.sourceOffset < storedEnd || tile.destinationOffset != decodedEnd ||
            tile.destinationOffset % 4 || tile.codec > 1 || tile.storedBytes == 0 || tile.decodedBytes == 0 ||
            tile.decodedBytes > kTileBytes || !range(tile.sourceOffset, tile.storedBytes, stored.size()) ||
            !range(tile.destinationOffset, tile.decodedBytes, page.uncompressedSize) ||
            (!tile.codec && (tile.storedBytes != tile.decodedBytes || tile.storedBytes % 4)) ||
            checksum(stored.subspan(tile.sourceOffset, tile.storedBytes)) != tile.checksum) { return fail(); }
        decodedEnd += tile.decodedBytes;
        storedEnd = uint64_t(tile.sourceOffset) + tile.storedBytes;
    }
    if (decodedEnd != page.uncompressedSize || storedEnd != stored.size()) { return fail(); }
    for (const auto& cluster : result.clusters) {
        if (cluster.reserved || !cluster.vertexCount || !cluster.triangleCount ||
            cluster.vertexCount > 128u || cluster.triangleCount > 128u ||
            cluster.materialIndex != header.materialIndex ||
            !range(cluster.vertexOffset, cluster.vertexCount, header.vertexCount) ||
            !range(cluster.triangleOffset, uint64_t(cluster.triangleCount) * 3, header.triangleIndexCount)) { return fail(); }
    }
    return true;
}

bool decodeMeshletStreamGpuPage(const MeshletStreamPageInfo& page,
    std::span<const uint8_t> stored, std::vector<uint8_t>& decoded, std::string& reason)
{
    MeshletStreamGpuPage info;
    if (!inspectMeshletStreamGpuPage(page, stored, info, reason)) { return false; }
    thread_local Decompressor decoder(libdeflate_alloc_gdeflate_decompressor(), libdeflate_free_gdeflate_decompressor);
    if (!decoder) { reason = "GDeflate decoder allocation failed"; return false; }
    decoded.resize(size_t(page.uncompressedSize));
    for (const auto& tile : info.tiles) {
        const uint8_t* source = stored.data() + tile.sourceOffset;
        uint8_t* destination = decoded.data() + tile.destinationOffset;
        if (!tile.codec) { std::memcpy(destination, source, tile.decodedBytes); continue; }
        libdeflate_gdeflate_in_page input{source, tile.storedBytes};
        size_t outputBytes = 0;
        if (libdeflate_gdeflate_decompress(decoder.get(), &input, 1, destination, tile.decodedBytes, &outputBytes) != LIBDEFLATE_SUCCESS ||
            outputBytes != tile.decodedBytes) { reason = "GDeflate decode failed"; return false; }
    }
    if (std::memcmp(decoded.data(), &info.header, sizeof(info.header)) != 0) {
        reason = "GPU page decoded header differs from sideband"; return false;
    }
    for (uint32_t i = 0; i < info.header.clusterCount; ++i) {
        MeshletStreamPayloadCluster cluster;
        std::memcpy(&cluster, decoded.data() + info.header.clusterOffsetBytes + uint64_t(i) * sizeof(cluster), sizeof(cluster));
        const MeshletStreamGpuCluster sideband{cluster.vertexOffset, cluster.vertexCount,
            cluster.triangleOffset, cluster.triangleCount, cluster.materialIndex};
        if (std::memcmp(&sideband, &info.clusters[i], sizeof(sideband)) != 0) {
            reason = "GPU page decoded cluster differs from sideband"; return false;
        }
    }
    return true;
}

} // namespace metallic::scene

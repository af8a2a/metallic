/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Normal/tangent encoding and arithmetic float-bit coding are adapted from
 * nvpro-samples/vk_lod_clusters. The checked page envelope is Metallic's.
 */

#include "Runtime/Scene/MeshletStreamReferenceCodec.h"
#include "Runtime/Scene/MeshletStreamAsset.h"

#include <libdeflate.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>

namespace metallic::scene {
namespace {

constexpr uint32_t kMagic = 0x5246534du;
constexpr uint32_t kCompression = 3;
constexpr uint32_t kTileBytes = 65536;
constexpr float kPi = 3.14159265358979323846f;
struct Envelope {
    uint32_t magic = kMagic;
    uint32_t version = 1;
    uint32_t structuredBytes = 0;
    uint32_t residualBytes = 0;
    uint32_t tileCount = 0;
    uint32_t metadataChecksum = 0;
    uint32_t discardedPositionBits = 7;
    uint32_t discardedTexcoordBits = 7;
};
struct Tile {
    uint32_t sourceOffset = 0;
    uint32_t storedBytes = 0;
    uint32_t destinationOffset = 0;
    uint32_t decodedBytes = 0;
    uint32_t codec = 0;
    uint32_t checksum = 0;
};
struct AttributeRange { uint32_t offset = 0; uint32_t bytes = 0; };
struct Vec3 { float x, y, z; };

uint32_t checksum(std::span<const uint8_t> data)
{
    static const auto table = [] {
        std::array<std::array<uint32_t, 256>, 8> result{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (uint32_t b = 0; b < 8; ++b) { value = (value >> 1) ^ ((value & 1) ? 0x82f63b78u : 0); }
            result[0][i] = value;
        }
        for (uint32_t s = 1; s < 8; ++s) {
            for (uint32_t i = 0; i < 256; ++i) {
                const uint32_t value = result[s - 1][i];
                result[s][i] = result[0][value & 255] ^ (value >> 8);
            }
        }
        return result;
    }();
    uint32_t crc = ~0u;
    size_t offset = 0;
    for (; data.size() - offset >= 8; offset += 8) {
        uint64_t word;
        std::memcpy(&word, data.data() + offset, 8);
        if constexpr (std::endian::native == std::endian::big) { word = std::byteswap(word); }
        word ^= crc;
        crc = table[7][word & 255] ^ table[6][(word >> 8) & 255] ^
            table[5][(word >> 16) & 255] ^ table[4][(word >> 24) & 255] ^
            table[3][(word >> 32) & 255] ^ table[2][(word >> 40) & 255] ^
            table[1][(word >> 48) & 255] ^ table[0][word >> 56];
    }
    for (uint8_t byte : data.subspan(offset)) { crc = table[0][(crc ^ byte) & 255] ^ (crc >> 8); }
    return ~crc;
}

bool range(uint64_t offset, uint64_t bytes, uint64_t capacity)
{
    return offset <= capacity && bytes <= capacity - offset;
}

Vec3 normalize(Vec3 v)
{
    const float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return length > 0 && std::isfinite(length) ? Vec3{v.x / length, v.y / length, v.z / length} : Vec3{0, 0, 1};
}
float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
bool valid(Vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && dot(v, v) > 0; }

Vec3 octToVector(float x, float y)
{
    Vec3 v{x, y, 1 - std::abs(x) - std::abs(y)};
    if (v.z < 0) {
        v.x = (1 - std::abs(y)) * (x >= 0 ? 1 : -1);
        v.y = (1 - std::abs(x)) * (y >= 0 ? 1 : -1);
    }
    return normalize(v);
}

uint32_t packNormal(Vec3 n)
{
    n = normalize(n);
    const float inverse = 1 / (std::abs(n.x) + std::abs(n.y) + std::abs(n.z));
    float x = n.x * inverse, y = n.y * inverse;
    if (n.z <= 0) {
        const float oldX = x;
        x = (1 - std::abs(y)) * (oldX >= 0 ? 1 : -1);
        y = (1 - std::abs(oldX)) * (y >= 0 ? 1 : -1);
    }
    constexpr float m = 1023;
    x = std::floor(std::clamp(x, -1.0f, 1.0f) * m) / m;
    y = std::floor(std::clamp(y, -1.0f, 1.0f) * m) / m;
    float bestX = x, bestY = y, bestCosine = dot(octToVector(x, y), n);
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 2; ++j) {
            const float candidateX = x + float(i) / m, candidateY = y + float(j) / m;
            const float cosine = dot(octToVector(candidateX, candidateY), n);
            if (cosine > bestCosine) { bestX = candidateX; bestY = candidateY; bestCosine = cosine; }
        }
    }
    const uint32_t px = uint32_t((bestX + 1) * 0.5f * 2047 + 0.5f) & 2047;
    const uint32_t py = uint32_t((bestY + 1) * 0.5f * 2047 + 0.5f) & 2047;
    return px | (py << 11);
}

Vec3 unpackNormal(uint32_t word)
{
    return octToVector(float(word & 2047) / 2047 * 2 - 1,
        float((word >> 11) & 2047) / 2047 * 2 - 1);
}

void basis(Vec3 n, Vec3& tangent, Vec3& bitangent)
{
    if (n.z < -0.99998796f) { tangent = {0, -1, 0}; }
    else {
        const float a = 1 / (1 + n.z), b = -n.x * n.y * a;
        tangent = {1 - n.x * n.x * a, b, -n.x};
    }
    // The closed-form basis loses orthogonality near its south-pole
    // singularity with float arithmetic. Reorthogonalize the shared basis so
    // the angular quantizer keeps its 0.353-degree error bound there too.
    const float along = dot(n, tangent);
    tangent = normalize({tangent.x - along * n.x, tangent.y - along * n.y, tangent.z - along * n.z});
    bitangent = {n.y * tangent.z - n.z * tangent.y,
        n.z * tangent.x - n.x * tangent.z, n.x * tangent.y - n.y * tangent.x};
}

uint32_t packTangent(Vec3 normal, const float* tangent)
{
    Vec3 t, b;
    basis(normal, t, b);
    const Vec3 source{tangent[0], tangent[1], tangent[2]};
    const float angle = std::atan2(dot(t, source), dot(b, source)) / kPi;
    const uint32_t angleBits = uint32_t(std::clamp((angle + 1) * 0.5f, 0.0f, 1.0f) * 511 + 0.5f);
    return (angleBits << 1) | (tangent[3] > 0 ? 1u : 0u);
}

void unpackTangent(Vec3 normal, uint32_t word, float* output)
{
    Vec3 t, b;
    basis(normal, t, b);
    const float angle = (float((word >> 1) & 511) / 511 * 2 - 1) * kPi;
    const float c = std::cos(angle), s = std::sin(angle);
    output[0] = c * b.x + s * t.x;
    output[1] = c * b.y + s * t.y;
    output[2] = c * b.z + s * t.z;
    output[3] = (word & 1) ? 1.0f : -1.0f;
}

uint32_t quantizedBits(float value, uint32_t discarded)
{
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return std::isfinite(value) ? bits & (~0u << discarded) : bits;
}

template <class T> void append(std::vector<uint8_t>& bytes, const T& value)
{
    const size_t offset = bytes.size();
    bytes.resize(offset + sizeof(T));
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}
void appendBytes(std::vector<uint8_t>& bytes, std::span<const uint8_t> value)
{
    bytes.insert(bytes.end(), value.begin(), value.end());
}
struct Reader {
    std::span<const uint8_t> bytes;
    size_t offset = 0;
    template <class T> bool read(T& value)
    {
        if (!range(offset, sizeof(T), bytes.size())) { return false; }
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return true;
    }
    bool copy(uint8_t* output, size_t count)
    {
        if (!range(offset, count, bytes.size())) { return false; }
        std::memcpy(output, bytes.data() + offset, count);
        offset += count;
        return true;
    }
};

bool layout(const MeshletStreamPayloadHeader& h, uint64_t bytes,
    std::vector<AttributeRange>& attributes, std::string& reason)
{
    attributes.clear();
    const auto fail = [&] { reason = "Reference page decoded layout is invalid"; return false; };
    if (h.magic != 0x4d535047u || h.version != 4 || h.compressionMode != 0 ||
        h.payloadByteSize != bytes || h.uncompressedPayloadByteSize != bytes ||
        bytes > UINT32_MAX || h.vertexCount == 0 || h.clusterCount == 0 ||
        h.clusterCount > 65536 || h.vertexCount > uint64_t(h.clusterCount) * 128 ||
        (h.attributeFlags & ~31u) || !(h.attributeFlags & kMeshletStreamPayloadAttributePosition) ||
        !range(h.clusterOffsetBytes, uint64_t(h.clusterCount) * sizeof(MeshletStreamPayloadCluster), bytes) ||
        !range(h.triangleOffsetBytes, h.triangleIndexCount, bytes) ||
        !range(h.materialOffsetBytes, uint64_t(h.materialCount) * 4, bytes)) { return fail(); }
    const uint32_t positionStride = meshletStreamPositionStride(h.positionFormat);
    if (!positionStride) { return fail(); }
    attributes.push_back({h.positionOffsetBytes, h.vertexCount * positionStride});
    if (h.attributeFlags & kMeshletStreamPayloadAttributeNormal) {
        if (h.normalFormat != uint32_t(MeshletStreamPayloadFormat::Float32x4)) { return fail(); }
        attributes.push_back({h.normalOffsetBytes, h.vertexCount * 16});
    }
    if (h.attributeFlags & kMeshletStreamPayloadAttributeTangent) {
        if (h.tangentFormat != uint32_t(MeshletStreamPayloadFormat::Float32x4)) { return fail(); }
        attributes.push_back({h.tangentOffsetBytes, h.vertexCount * 16});
    }
    if (h.attributeFlags & kMeshletStreamPayloadAttributeTexcoord0) {
        if (h.texcoord0Format != uint32_t(MeshletStreamPayloadFormat::Float32x2)) { return fail(); }
        attributes.push_back({h.texcoord0OffsetBytes, h.vertexCount * 8});
    }
    std::sort(attributes.begin(), attributes.end(), [](auto a, auto b) { return a.offset < b.offset; });
    uint64_t end = sizeof(h);
    for (const auto& a : attributes) {
        if (a.offset < end || !range(a.offset, a.bytes, bytes)) { return fail(); }
        end = uint64_t(a.offset) + a.bytes;
    }
    const std::array fixed = {AttributeRange{0, sizeof(h)},
        AttributeRange{h.clusterOffsetBytes, h.clusterCount * uint32_t(sizeof(MeshletStreamPayloadCluster))},
        AttributeRange{h.triangleOffsetBytes, h.triangleIndexCount},
        AttributeRange{h.materialOffsetBytes, h.materialCount * 4}};
    for (const auto& a : attributes) {
        for (const auto& f : fixed) {
            if (a.offset < uint64_t(f.offset) + f.bytes && f.offset < uint64_t(a.offset) + a.bytes) { return fail(); }
        }
    }
    return true;
}

bool clusters(const MeshletStreamPayloadHeader& h, std::span<const uint8_t> bytes,
    std::vector<MeshletStreamPayloadCluster>& output, std::string& reason)
{
    output.resize(h.clusterCount);
    std::memcpy(output.data(), bytes.data() + h.clusterOffsetBytes, output.size() * sizeof(output[0]));
    uint64_t vertexEnd = 0, triangleEnd = 0;
    for (const auto& c : output) {
        if (c.vertexOffset != vertexEnd || c.triangleOffset != triangleEnd || !c.vertexCount || c.vertexCount > 128 ||
            !c.triangleCount || c.triangleCount > 128 ||
            !range(c.vertexOffset, c.vertexCount, h.vertexCount) ||
            !range(c.triangleOffset, uint64_t(c.triangleCount) * 3, h.triangleIndexCount)) {
            reason = "Reference page cluster ranges are invalid"; return false;
        }
        for (uint32_t i = 0; i < c.triangleCount * 3; ++i) {
            if (bytes[h.triangleOffsetBytes + c.triangleOffset + i] >= c.vertexCount) {
                reason = "Reference page triangle index is invalid"; return false;
            }
        }
        vertexEnd += c.vertexCount;
        triangleEnd += uint64_t(c.triangleCount) * 3;
    }
    if (vertexEnd != h.vertexCount || triangleEnd != h.triangleIndexCount) {
        reason = "Reference page clusters do not cover the attribute streams"; return false;
    }
    return true;
}

void writeFloatBlock(std::vector<uint8_t>& output, const uint8_t* input,
    uint32_t count, uint32_t dimensions, uint32_t stride, uint32_t discarded)
{
    std::array<uint32_t, 3> low{~0u, ~0u, ~0u}, high{}, mask{}, shifts{}, widths{};
    std::array<std::array<uint32_t, 3>, 128> values{};
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dimensions; ++d) {
            float value;
            std::memcpy(&value, input + uint64_t(i) * stride + d * 4, 4);
            values[i][d] = quantizedBits(value, discarded);
            low[d] = std::min(low[d], values[i][d]);
            high[d] = std::max(high[d], values[i][d]);
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dimensions; ++d) { mask[d] |= values[i][d] - low[d]; }
    }
    uint32_t bitsPerVertex = 0, packedWidths = 0, packedShifts = 0;
    for (uint32_t d = 0; d < dimensions; ++d) {
        shifts[d] = mask[d] ? uint32_t(std::countr_zero(mask[d])) : 31;
        widths[d] = std::max(1u, uint32_t(std::bit_width((high[d] - low[d]) >> shifts[d])));
        packedWidths |= (widths[d] - 1) << (d * 5);
        packedShifts |= shifts[d] << (d * 5);
        bitsPerVertex += widths[d];
    }
    const uint32_t packedBytes = 4 + dimensions * 4 + ((uint64_t(bitsPerVertex) * count + 31) / 32) * 4;
    const uint32_t rawBytes = count * dimensions * 4;
    if (packedBytes >= rawBytes) {
        append(output, rawBytes << 1);
        for (uint32_t i = 0; i < count; ++i) {
            for (uint32_t d = 0; d < dimensions; ++d) { append(output, values[i][d]); }
        }
        return;
    }
    append(output, (packedBytes << 1) | 1);
    append(output, packedShifts | (packedWidths << 16));
    for (uint32_t d = 0; d < dimensions; ++d) { append(output, low[d]); }
    const size_t start = output.size();
    output.resize(start + packedBytes - 4 - dimensions * 4, 0);
    uint64_t bit = 0;
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dimensions; ++d) {
            const uint32_t value = (values[i][d] - low[d]) >> shifts[d];
            const uint32_t shift = uint32_t(bit & 31);
            const size_t offset = start + size_t(bit / 32) * 4;
            uint32_t word;
            std::memcpy(&word, output.data() + offset, 4);
            word |= value << shift;
            std::memcpy(output.data() + offset, &word, 4);
            if (shift + widths[d] > 32) {
                word = value >> (32 - shift);
                std::memcpy(output.data() + offset + 4, &word, 4);
            }
            bit += widths[d];
        }
    }
}

bool readFloatBlock(Reader& input, uint8_t* output, uint32_t count, uint32_t dimensions, uint32_t stride)
{
    uint32_t tag;
    if (!input.read(tag)) { return false; }
    const uint32_t bytes = tag >> 1;
    if (!range(input.offset, bytes, input.bytes.size())) { return false; }
    Reader block{input.bytes.subspan(input.offset, bytes)};
    input.offset += bytes;
    if (!(tag & 1)) {
        if (bytes != count * dimensions * 4) { return false; }
        for (uint32_t i = 0; i < count; ++i) {
            if (!block.copy(output + uint64_t(i) * stride, dimensions * 4)) { return false; }
        }
        return true;
    }
    uint32_t parameters;
    std::array<uint32_t, 3> low{}, shifts{}, widths{};
    if (!block.read(parameters) || (parameters & 0x80008000u)) { return false; }
    uint64_t totalBits = 0;
    for (uint32_t d = 0; d < dimensions; ++d) {
        shifts[d] = (parameters >> (d * 5)) & 31;
        widths[d] = ((parameters >> (16 + d * 5)) & 31) + 1;
        if (!block.read(low[d]) || shifts[d] + widths[d] > 32) { return false; }
        totalBits += uint64_t(count) * widths[d];
    }
    if (bytes != 4 + dimensions * 4 + ((totalBits + 31) / 32) * 4) { return false; }
    const size_t start = block.offset;
    uint64_t bit = 0;
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dimensions; ++d) {
            uint32_t word, upper = 0;
            const uint32_t shift = uint32_t(bit & 31);
            const size_t offset = start + size_t(bit / 32) * 4;
            std::memcpy(&word, block.bytes.data() + offset, 4);
            if (shift + widths[d] > 32) { std::memcpy(&upper, block.bytes.data() + offset + 4, 4); }
            uint32_t delta = uint32_t((uint64_t(upper) << 32 | word) >> shift);
            if (widths[d] < 32) { delta &= (1u << widths[d]) - 1; }
            if (uint64_t(delta) << shifts[d] > uint64_t(UINT32_MAX) - low[d]) { return false; }
            const uint32_t value = low[d] + (delta << shifts[d]);
            std::memcpy(output + uint64_t(i) * stride + d * 4, &value, 4);
            bit += widths[d];
        }
    }
    return true;
}

using Compressor = std::unique_ptr<libdeflate_gdeflate_compressor, decltype(&libdeflate_free_gdeflate_compressor)>;
using Decompressor = std::unique_ptr<libdeflate_gdeflate_decompressor, decltype(&libdeflate_free_gdeflate_decompressor)>;

} // namespace

void canonicalizeMeshletStreamReferenceVertex(float* position3, float* normal3,
    float* texcoord2, float* tangent4, const MeshletStreamReferenceCodecOptions& options)
{
    const uint32_t positionBits = std::min(options.discardedPositionBits, 23u);
    const uint32_t texcoordBits = std::min(options.discardedTexcoordBits, 23u);
    if (position3) {
        for (uint32_t d = 0; d < 3; ++d) { position3[d] = std::bit_cast<float>(quantizedBits(position3[d], positionBits)); }
    }
    if (texcoord2) {
        for (uint32_t d = 0; d < 2; ++d) { texcoord2[d] = std::bit_cast<float>(quantizedBits(texcoord2[d], texcoordBits)); }
    }
    const Vec3 originalNormal = normal3 ? Vec3{normal3[0], normal3[1], normal3[2]} : Vec3{0, 0, 1};
    if ((normal3 && !valid(originalNormal)) || (tangent4 &&
        (!valid({tangent4[0], tangent4[1], tangent4[2]}) || !std::isfinite(tangent4[3]) || tangent4[3] == 0))) { return; }
    const Vec3 decodedNormal = normal3 && valid(originalNormal) ? unpackNormal(packNormal(originalNormal)) : Vec3{0, 0, 0};
    if (tangent4 && valid({tangent4[0], tangent4[1], tangent4[2]}) && tangent4[3] != 0) {
        if (normal3 && valid(originalNormal)) { unpackTangent(decodedNormal, packTangent(decodedNormal, tangent4), tangent4); }
        else {
            const Vec3 t = unpackNormal(packNormal({tangent4[0], tangent4[1], tangent4[2]}));
            tangent4[0] = t.x; tangent4[1] = t.y; tangent4[2] = t.z;
            tangent4[3] = tangent4[3] > 0 ? 1.0f : -1.0f;
        }
    }
    if (normal3) { normal3[0] = decodedNormal.x; normal3[1] = decodedNormal.y; normal3[2] = decodedNormal.z; }
}

bool encodeMeshletStreamReferencePage(std::span<const uint8_t> decoded,
    std::vector<uint8_t>& stored, std::string& reason, const MeshletStreamReferenceCodecOptions& options)
{
    stored.clear();
    if (decoded.size() < sizeof(MeshletStreamPayloadHeader) || options.discardedPositionBits > 23 ||
        options.discardedTexcoordBits > 23) { reason = "Reference page size or precision is invalid"; return false; }
    MeshletStreamPayloadHeader header;
    std::memcpy(&header, decoded.data(), sizeof(header));
    std::vector<AttributeRange> attributes;
    std::vector<MeshletStreamPayloadCluster> clusterRecords;
    if (!layout(header, decoded.size(), attributes, reason) || !clusters(header, decoded, clusterRecords, reason)) { return false; }
    Envelope envelope;
    envelope.discardedPositionBits = options.discardedPositionBits;
    envelope.discardedTexcoordBits = options.discardedTexcoordBits;
    std::vector<uint8_t> structured;
    structured.reserve(decoded.size() / 3);
    size_t cursor = 0;
    for (const auto& a : attributes) {
        appendBytes(structured, decoded.subspan(cursor, a.offset - cursor));
        cursor = uint64_t(a.offset) + a.bytes;
    }
    appendBytes(structured, decoded.subspan(cursor));
    envelope.residualBytes = uint32_t(structured.size());
    const bool hasNormal = header.attributeFlags & kMeshletStreamPayloadAttributeNormal;
    const bool hasTangent = header.attributeFlags & kMeshletStreamPayloadAttributeTangent;
    const bool hasTexcoord = header.attributeFlags & kMeshletStreamPayloadAttributeTexcoord0;
    const uint32_t positionStride = meshletStreamPositionStride(header.positionFormat);
    for (const auto& c : clusterRecords) {
        writeFloatBlock(structured, decoded.data() + header.positionOffsetBytes + uint64_t(c.vertexOffset) * positionStride,
            c.vertexCount, 3, positionStride, options.discardedPositionBits);
        if (positionStride == 16) {
            for (uint32_t i = 0; i < c.vertexCount; ++i) {
                uint32_t w;
                std::memcpy(&w, decoded.data() + header.positionOffsetBytes + uint64_t(c.vertexOffset + i) * 16 + 12, 4);
                append(structured, w);
            }
        }
        if (hasNormal || hasTangent) {
            // Invalid shading vectors remain exact instead of acquiring a
            // made-up tangent basis. The bitmap keeps every other vertex at
            // reference precision, independent of neighboring invalid data.
            std::array<uint32_t, 4> rawMask{};
            bool rawShading = false;
            for (uint32_t i = 0; i < c.vertexCount; ++i) {
                float n[4] = {}, t[4] = {};
                if (hasNormal) { std::memcpy(n, decoded.data() + header.normalOffsetBytes + uint64_t(c.vertexOffset + i) * 16, 16); }
                if (hasTangent) { std::memcpy(t, decoded.data() + header.tangentOffsetBytes + uint64_t(c.vertexOffset + i) * 16, 16); }
                const bool rawVertex = (hasNormal && (!valid({n[0], n[1], n[2]}) || n[3] != 0)) ||
                    (hasTangent && (!valid({t[0], t[1], t[2]}) || !std::isfinite(t[3]) || t[3] == 0));
                if (rawVertex) { rawMask[i / 32] |= 1u << (i % 32); rawShading = true; }
            }
            append(structured, rawShading ? 1u : 0u);
            if (rawShading) {
                for (uint32_t i = 0; i < (c.vertexCount + 31) / 32; ++i) { append(structured, rawMask[i]); }
            }
            for (uint32_t i = 0; i < c.vertexCount; ++i) {
                float n[4] = {}, t[4] = {};
                if (hasNormal) { std::memcpy(n, decoded.data() + header.normalOffsetBytes + uint64_t(c.vertexOffset + i) * 16, 16); }
                if (hasTangent) { std::memcpy(t, decoded.data() + header.tangentOffsetBytes + uint64_t(c.vertexOffset + i) * 16, 16); }
                if (rawMask[i / 32] & (1u << (i % 32))) {
                    if (hasNormal) { append(structured, n); }
                    if (hasTangent) { append(structured, t); }
                } else if (hasNormal) {
                    const Vec3 normal{n[0], n[1], n[2]};
                    const uint32_t normalBits = packNormal(normal);
                    // Use the decoded normal for both sides of the tangent
                    // angle transform. Quantization can cross the basis's
                    // south-pole singularity; using the authored normal here
                    // then the decoded normal on read rotates such tangents.
                    append(structured, normalBits | (hasTangent ? packTangent(unpackNormal(normalBits), t) << 22 : 0));
                } else {
                    append(structured, packNormal({t[0], t[1], t[2]}) | (t[3] > 0 ? 1u << 22 : 0));
                }
            }
        }
        if (hasTexcoord) {
            writeFloatBlock(structured, decoded.data() + header.texcoord0OffsetBytes + uint64_t(c.vertexOffset) * 8,
                c.vertexCount, 2, 8, options.discardedTexcoordBits);
        }
    }
    if (structured.size() > UINT32_MAX) { reason = "Reference page structured data is too large"; return false; }
    envelope.structuredBytes = uint32_t(structured.size());
    envelope.tileCount = uint32_t((structured.size() + kTileBytes - 1) / kTileBytes);
    const size_t metadataBytes = sizeof(header) + sizeof(envelope) + envelope.tileCount * sizeof(Tile);
    stored.resize(metadataBytes);
    thread_local Compressor compressor(libdeflate_alloc_gdeflate_compressor(1), libdeflate_free_gdeflate_compressor);
    if (options.generalCompression && !compressor) { reason = "Reference page GDeflate allocation failed"; return false; }
    std::vector<uint8_t> compressed;
    for (uint32_t i = 0; i < envelope.tileCount; ++i) {
        const auto input = std::span(structured).subspan(uint64_t(i) * kTileBytes,
            std::min<size_t>(kTileBytes, structured.size() - uint64_t(i) * kTileBytes));
        bool useCompressed = false;
        if (options.generalCompression) {
            size_t pageCount = 0;
            compressed.resize(libdeflate_gdeflate_compress_bound(compressor.get(), input.size(), &pageCount));
            libdeflate_gdeflate_out_page output{compressed.data(), compressed.size()};
            if (pageCount != 1 || !libdeflate_gdeflate_compress(compressor.get(), input.data(), input.size(), &output, 1)) {
                reason = "Reference page GDeflate compression failed"; return false;
            }
            compressed.resize(output.nbytes);
            useCompressed = compressed.size() < input.size();
        }
        const std::span<const uint8_t> bytes = useCompressed ? std::span<const uint8_t>(compressed) : input;
        const Tile tile{uint32_t(stored.size()), uint32_t(bytes.size()), i * kTileBytes,
            uint32_t(input.size()), useCompressed ? 1u : 0u, checksum(bytes)};
        appendBytes(stored, bytes);
        std::memcpy(stored.data() + sizeof(header) + sizeof(envelope) + uint64_t(i) * sizeof(tile), &tile, sizeof(tile));
    }
    header.compressionMode = kCompression;
    header.payloadByteSize = uint32_t(stored.size());
    std::memcpy(stored.data(), &header, sizeof(header));
    std::memcpy(stored.data() + sizeof(header), &envelope, sizeof(envelope));
    envelope.metadataChecksum = checksum(std::span(stored).first(metadataBytes));
    std::memcpy(stored.data() + sizeof(header), &envelope, sizeof(envelope));
    return true;
}

bool decodeMeshletStreamReferencePage(std::span<const uint8_t> stored,
    std::vector<uint8_t>& decoded, std::string& reason)
{
    decoded.clear();
    const auto fail = [&] { decoded.clear(); reason = "Reference page envelope, checksum or data is invalid"; return false; };
    MeshletStreamPayloadHeader header;
    Envelope envelope;
    if (stored.size() < sizeof(header) + sizeof(envelope)) { return fail(); }
    std::memcpy(&header, stored.data(), sizeof(header));
    std::memcpy(&envelope, stored.data() + sizeof(header), sizeof(envelope));
    if (header.compressionMode != kCompression || header.payloadByteSize != stored.size() ||
        envelope.magic != kMagic || envelope.version != 1 ||
        !envelope.structuredBytes || envelope.structuredBytes > header.uncompressedPayloadByteSize * uint64_t(2) ||
        envelope.discardedPositionBits > 23 || envelope.discardedTexcoordBits > 23 ||
        envelope.tileCount != (uint64_t(envelope.structuredBytes) + kTileBytes - 1) / kTileBytes ||
        envelope.residualBytes < sizeof(header) || envelope.residualBytes > envelope.structuredBytes) { return fail(); }
    const uint64_t metadataBytes = sizeof(header) + sizeof(envelope) + uint64_t(envelope.tileCount) * sizeof(Tile);
    if (metadataBytes > stored.size()) { return fail(); }
    std::vector<uint8_t> metadata(stored.begin(), stored.begin() + size_t(metadataBytes));
    const uint32_t expectedChecksum = envelope.metadataChecksum;
    envelope.metadataChecksum = 0;
    std::memcpy(metadata.data() + sizeof(header), &envelope, sizeof(envelope));
    if (checksum(metadata) != expectedChecksum) { return fail(); }
    header.compressionMode = 0;
    header.payloadByteSize = header.uncompressedPayloadByteSize;
    std::vector<AttributeRange> attributes;
    if (!layout(header, header.payloadByteSize, attributes, reason)) { return false; }
    uint64_t expectedResidual = header.payloadByteSize;
    for (const auto& a : attributes) { expectedResidual -= a.bytes; }
    if (expectedResidual != envelope.residualBytes) { return fail(); }
    std::vector<uint8_t> structured(envelope.structuredBytes);
    uint64_t sourceEnd = metadataBytes, destinationEnd = 0;
    thread_local Decompressor decoder(libdeflate_alloc_gdeflate_decompressor(), libdeflate_free_gdeflate_decompressor);
    if (!decoder) { reason = "Reference page GDeflate decoder allocation failed"; return false; }
    for (uint32_t i = 0; i < envelope.tileCount; ++i) {
        Tile tile;
        std::memcpy(&tile, stored.data() + sizeof(header) + sizeof(envelope) + uint64_t(i) * sizeof(tile), sizeof(tile));
        if (tile.sourceOffset != sourceEnd || tile.destinationOffset != destinationEnd || tile.codec > 1 ||
            !tile.storedBytes || !tile.decodedBytes || tile.decodedBytes > kTileBytes ||
            !range(tile.sourceOffset, tile.storedBytes, stored.size()) ||
            !range(tile.destinationOffset, tile.decodedBytes, structured.size()) ||
            (!tile.codec && tile.storedBytes != tile.decodedBytes) ||
            checksum(stored.subspan(tile.sourceOffset, tile.storedBytes)) != tile.checksum) { return fail(); }
        if (!tile.codec) { std::memcpy(structured.data() + tile.destinationOffset, stored.data() + tile.sourceOffset, tile.decodedBytes); }
        else {
            libdeflate_gdeflate_in_page input{stored.data() + tile.sourceOffset, tile.storedBytes};
            size_t outputBytes = 0;
            if (libdeflate_gdeflate_decompress(decoder.get(), &input, 1, structured.data() + tile.destinationOffset,
                tile.decodedBytes, &outputBytes) != LIBDEFLATE_SUCCESS || outputBytes != tile.decodedBytes) { return fail(); }
        }
        sourceEnd += tile.storedBytes;
        destinationEnd += tile.decodedBytes;
    }
    if (sourceEnd != stored.size() || destinationEnd != structured.size()) { return fail(); }
    decoded.resize(header.payloadByteSize, 0);
    Reader reader{structured};
    size_t cursor = 0;
    for (const auto& a : attributes) {
        if (!reader.copy(decoded.data() + cursor, a.offset - cursor)) { return fail(); }
        cursor = uint64_t(a.offset) + a.bytes;
    }
    if (!reader.copy(decoded.data() + cursor, decoded.size() - cursor) || reader.offset != envelope.residualBytes ||
        std::memcmp(decoded.data(), &header, sizeof(header))) { return fail(); }
    std::vector<MeshletStreamPayloadCluster> clusterRecords;
    if (!clusters(header, decoded, clusterRecords, reason)) { decoded.clear(); return false; }
    const bool hasNormal = header.attributeFlags & kMeshletStreamPayloadAttributeNormal;
    const bool hasTangent = header.attributeFlags & kMeshletStreamPayloadAttributeTangent;
    const bool hasTexcoord = header.attributeFlags & kMeshletStreamPayloadAttributeTexcoord0;
    const uint32_t positionStride = meshletStreamPositionStride(header.positionFormat);
    for (const auto& c : clusterRecords) {
        uint8_t* positions = decoded.data() + header.positionOffsetBytes + uint64_t(c.vertexOffset) * positionStride;
        if (!readFloatBlock(reader, positions, c.vertexCount, 3, positionStride)) { return fail(); }
        if (positionStride == 16) {
            for (uint32_t i = 0; i < c.vertexCount; ++i) {
                if (!reader.copy(positions + uint64_t(i) * 16 + 12, 4)) { return fail(); }
            }
        }
        if (hasNormal || hasTangent) {
            uint32_t rawShading;
            if (!reader.read(rawShading) || rawShading > 1) { return fail(); }
            std::array<uint32_t, 4> rawMask{};
            if (rawShading) {
                const uint32_t words = (c.vertexCount + 31) / 32;
                for (uint32_t i = 0; i < words; ++i) { if (!reader.read(rawMask[i])) { return fail(); } }
                if ((c.vertexCount % 32) && (rawMask[words - 1] >> (c.vertexCount % 32))) { return fail(); }
            }
            for (uint32_t i = 0; i < c.vertexCount; ++i) {
                uint8_t* n = hasNormal ? decoded.data() + header.normalOffsetBytes + uint64_t(c.vertexOffset + i) * 16 : nullptr;
                uint8_t* t = hasTangent ? decoded.data() + header.tangentOffsetBytes + uint64_t(c.vertexOffset + i) * 16 : nullptr;
                if (rawMask[i / 32] & (1u << (i % 32))) {
                    if ((hasNormal && !reader.copy(n, 16)) || (hasTangent && !reader.copy(t, 16))) { return fail(); }
                    continue;
                }
                uint32_t word;
                if (!reader.read(word)) { return fail(); }
                const Vec3 normal = unpackNormal(word);
                if (hasNormal) {
                    const float value[4] = {normal.x, normal.y, normal.z, 0};
                    std::memcpy(n, value, 16);
                }
                if (hasTangent) {
                    float value[4];
                    if (hasNormal) { unpackTangent(normal, word >> 22, value); }
                    else { value[0] = normal.x; value[1] = normal.y; value[2] = normal.z; value[3] = (word & (1u << 22)) ? 1.0f : -1.0f; }
                    std::memcpy(t, value, 16);
                }
            }
        }
        if (hasTexcoord && !readFloatBlock(reader, decoded.data() + header.texcoord0OffsetBytes + uint64_t(c.vertexOffset) * 8,
            c.vertexCount, 2, 8)) { return fail(); }
    }
    if (reader.offset != structured.size()) { return fail(); }
    return true;
}

} // namespace metallic::scene

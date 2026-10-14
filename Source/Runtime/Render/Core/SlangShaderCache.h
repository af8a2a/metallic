#pragma once

#include "ShaderNamespaceProof.h"
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>

namespace metallic::render {
namespace slang_shader_cache {

inline constexpr uint32_t kVersion = 3, kHeaderSize = 48;
inline constexpr uint64_t kMaximumFileBytes = 512ull * 1024 * 1024;
inline constexpr uint64_t kMaximumProofBytes = 16ull * 1024 * 1024;
using Bytes = std::vector<uint8_t>;

inline uint64_t checksum(std::span<const uint8_t> data)
{
    uint64_t value = 14695981039346656037ull;
    for (const auto byte : data) { value = (value ^ byte) * 1099511628211ull; }
    return value;
}

inline void integer(Bytes& output, uint64_t value, size_t count)
{
    for (size_t byte = 0; byte < count; ++byte) { output.push_back(static_cast<uint8_t>(value >> (byte * 8))); }
}
inline void text(Bytes& output, const std::string& value)
{
    integer(output, value.size(), 4);
    output.insert(output.end(), value.begin(), value.end());
}

struct Cursor {
    std::span<const uint8_t> bytes;
    size_t offset = 0;
    bool integer(uint64_t& value, size_t count)
    {
        if (count > 8 || offset > bytes.size() || count > bytes.size() - offset) { return false; }
        value = 0;
        for (size_t byte = 0; byte < count; ++byte) { value |= uint64_t(bytes[offset++]) << (byte * 8); }
        return true;
    }
    bool text(std::string& value, bool allowEmpty = false)
    {
        uint64_t count;
        if (!integer(count, 4) || count > 32768 || (!allowEmpty && !count) || count > bytes.size() - offset) { return false; }
        value.assign(reinterpret_cast<const char*>(bytes.data() + offset), static_cast<size_t>(count));
        offset += static_cast<size_t>(count);
        return value.find('\0') == std::string::npos;
    }
};

inline Bytes encodeProof(const ShaderNamespaceProof& proof)
{
    if (!isWellFormedNamespaceProof(proof)) { return {}; }
    Bytes bytes;
    integer(bytes, proof.schemaVersion, 4); integer(bytes, proof.compilerProducerIdentity, 8);
    integer(bytes, proof.roots.size(), 4);
    for (const auto& root : proof.roots) {
        text(bytes, root.absolutePath); integer(bytes, root.exists ? 1 : 0, 4); integer(bytes, root.recursive ? 1 : 0, 4); text(bytes, root.existingWatchedAncestor);
        integer(bytes, root.entries.size(), 4);
        for (const auto& entry : root.entries) { text(bytes, entry.relativePath); integer(bytes, static_cast<uint32_t>(entry.kind), 4); }
    }
    integer(bytes, proof.actualReads.size(), 4);
    for (const auto& read : proof.actualReads) { text(bytes, read.canonicalPath); integer(bytes, read.byteSize, 8); integer(bytes, read.contentHash, 8); }
    integer(bytes, proof.negativeLookupPaths.size(), 4);
    for (const auto& path : proof.negativeLookupPaths) { text(bytes, path); }
    return bytes.size() <= kMaximumProofBytes ? bytes : Bytes{};
}

inline bool decodeProof(std::span<const uint8_t> bytes, ShaderNamespaceProof& output)
{
    output = {};
    if (bytes.empty() || bytes.size() > kMaximumProofBytes) { return false; }
    Cursor cursor{bytes};
    uint64_t schema, producer, count;
    if (!cursor.integer(schema, 4) || schema != 2 || !cursor.integer(producer, 8) || !producer ||
        !cursor.integer(count, 4) || !count || count > 128) { return false; }
    ShaderNamespaceProof proof;
    proof.schemaVersion = static_cast<uint32_t>(schema); proof.compilerProducerIdentity = producer;
    for (size_t index = 0; index < count; ++index) {
        ShaderNamespaceRootProof root;
        uint64_t exists, recursive, entries;
        if (!cursor.text(root.absolutePath) || !cursor.integer(exists, 4) || exists > 1 ||
            !cursor.integer(recursive, 4) || recursive > 1 ||
            !cursor.text(root.existingWatchedAncestor, true) || !cursor.integer(entries, 4) || entries > 1000000) { return false; }
        root.exists = exists != 0;
        root.recursive = recursive != 0;
        for (size_t entry = 0; entry < entries; ++entry) {
            ShaderNamespaceEntry item;
            uint64_t kind;
            if (!cursor.text(item.relativePath) || !cursor.integer(kind, 4) || kind < 1 || kind > 2) { return false; }
            item.kind = static_cast<ShaderNamespaceEntryKind>(kind);
            root.entries.push_back(std::move(item));
        }
        proof.roots.push_back(std::move(root));
    }
    if (!cursor.integer(count, 4) || !count || count > 4096) { return false; }
    for (size_t index = 0; index < count; ++index) {
        ShaderSourceReadProof read;
        if (!cursor.text(read.canonicalPath) || !cursor.integer(read.byteSize, 8) || !cursor.integer(read.contentHash, 8)) { return false; }
        proof.actualReads.push_back(std::move(read));
    }
    if (!cursor.integer(count, 4) || count > 65536) { return false; }
    for (size_t index = 0; index < count; ++index) {
        std::string path;
        if (!cursor.text(path)) { return false; }
        proof.negativeLookupPaths.push_back(std::move(path));
    }
    if (cursor.offset != bytes.size() || !isWellFormedNamespaceProof(proof)) { return false; }
    output = std::move(proof);
    return true;
}

inline Bytes encode(uint64_t requestHash, const ShaderNamespaceProof& proof, std::span<const uint32_t> spirv)
{
    if (spirv.size() < 5 || spirv[0] != 0x07230203 || spirv[4] != 0 || spirv.size_bytes() > kMaximumFileBytes) { return {}; }
    auto proofBytes = encodeProof(proof);
    if (proofBytes.empty() || proofBytes.size() + spirv.size_bytes() > kMaximumFileBytes - kHeaderSize) { return {}; }
    Bytes output{'M', 'T', 'L', 'S', 'P', 'V', '0', '1'};
    output.reserve(kHeaderSize + proofBytes.size() + spirv.size_bytes());
    integer(output, kVersion, 4); integer(output, kHeaderSize, 4); integer(output, requestHash, 8);
    integer(output, proofBytes.size(), 8); integer(output, spirv.size_bytes(), 8); integer(output, 0, 8);
    output.insert(output.end(), proofBytes.begin(), proofBytes.end());
    const auto* begin = reinterpret_cast<const uint8_t*>(spirv.data());
    output.insert(output.end(), begin, begin + spirv.size_bytes());
    const auto bodyChecksum = checksum(std::span<const uint8_t>(output).subspan(kHeaderSize));
    for (size_t byte = 0; byte < 8; ++byte) { output[40 + byte] = static_cast<uint8_t>(bodyChecksum >> (byte * 8)); }
    return output;
}

struct CachedPayload {
    ShaderNamespaceProof proof;
    std::vector<uint32_t> spirv;
};

// Decoder never falls back to V2. Its caller constructs current proof through
// a validated namespace epoch and actual current dependency byte reads. The
// same function serves ordinary runtime lookup and warmup snapshot readback.
inline bool decode(std::span<const uint8_t> bytes, uint64_t requestHash,
    const std::function<bool(const ShaderNamespaceProof&, ShaderNamespaceProof&)>& currentProof,
    CachedPayload& output)
{
    output = {};
    if (bytes.size() < kHeaderSize || bytes.size() > kMaximumFileBytes || std::memcmp(bytes.data(), "MTLSPV01", 8)) { return false; }
    Cursor cursor{bytes, 8};
    uint64_t version, headerSize, request, proofSize, spirvSize, payloadChecksum;
    if (!cursor.integer(version, 4) || version != kVersion || !cursor.integer(headerSize, 4) || headerSize != kHeaderSize ||
        !cursor.integer(request, 8) || request != requestHash || !cursor.integer(proofSize, 8) || !proofSize || proofSize > kMaximumProofBytes ||
        !cursor.integer(spirvSize, 8) || spirvSize < 20 || spirvSize % 4 ||
        !cursor.integer(payloadChecksum, 8) || proofSize > bytes.size() - kHeaderSize ||
        spirvSize != bytes.size() - kHeaderSize - proofSize || checksum(bytes.subspan(kHeaderSize)) != payloadChecksum) { return false; }
    ShaderNamespaceProof proof, current;
    if (!decodeProof(bytes.subspan(kHeaderSize, static_cast<size_t>(proofSize)), proof) || !currentProof) { return false; }
    try {
        if (!currentProof(proof, current) || !namespaceProofMatches(proof, current)) { return false; }
    } catch (...) { return false; } // Unavailable/failed live proof never qualifies a hit.
    std::vector<uint32_t> spirv(static_cast<size_t>(spirvSize / 4));
    std::memcpy(spirv.data(), bytes.data() + kHeaderSize + proofSize, static_cast<size_t>(spirvSize));
    if (spirv[0] != 0x07230203 || spirv[4] != 0) { return false; }
    output.proof = std::move(proof); output.spirv = std::move(spirv);
    return true;
}

} // namespace slang_shader_cache
} // namespace metallic::render

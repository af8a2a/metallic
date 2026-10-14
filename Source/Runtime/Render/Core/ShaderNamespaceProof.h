#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace metallic::render {

enum class ShaderNamespaceEntryKind : uint8_t { File = 1, Directory = 2 };

struct ShaderNamespaceEntry {
    std::string relativePath;
    ShaderNamespaceEntryKind kind = ShaderNamespaceEntryKind::File;
    bool operator==(const ShaderNamespaceEntry&) const = default;
};

struct ShaderNamespaceRootProof {
    std::string absolutePath;
    // A missing directory is qualified by a watched, existing ancestor and
    // its namespace; simply hashing current positive files is insufficient.
    bool exists = true;
    bool recursive = true;
    std::string existingWatchedAncestor;
    std::vector<ShaderNamespaceEntry> entries;
    bool operator==(const ShaderNamespaceRootProof&) const = default;
};

struct ShaderSourceReadProof {
    std::string canonicalPath;
    uint64_t byteSize = 0, contentHash = 0;
    // Source bytes came from SlangSnapshotFileSystem.loadFile, not a later live hash.
    bool operator==(const ShaderSourceReadProof&) const = default;
};

struct ShaderNamespaceProof {
    uint32_t schemaVersion = 2;
    uint64_t compilerProducerIdentity = 0;
    // Canonical monitored graph roots, including SDK/vendor/header namespaces.
    // Ordered effective search paths remain in the unchanged request identity.
    std::vector<ShaderNamespaceRootProof> roots;
    std::vector<ShaderSourceReadProof> actualReads;
    // Misses are distinct from zero-length successful files.
    std::vector<std::string> negativeLookupPaths;
    bool operator==(const ShaderNamespaceProof&) const = default;
};

inline bool isSafeProofPath(std::string_view text, bool relative)
{
    if (text.empty() || text.size() > 32768 || text.find('\0') != std::string_view::npos) { return false; }
    if (!relative && text.front() != '/' && text.front() != '\\' &&
        !(text.size() >= 3 && ((text[0] >= 'A' && text[0] <= 'Z') || (text[0] >= 'a' && text[0] <= 'z')) &&
            text[1] == ':' && (text[2] == '/' || text[2] == '\\'))) { return false; }
    if (relative && (text.front() == '/' || text.front() == '\\' || text.find(':') != std::string_view::npos)) { return false; }
    if (relative) {
        size_t begin = 0;
        while (begin <= text.size()) {
            const auto end = text.find_first_of("/\\", begin);
            const auto component = text.substr(begin, end == std::string_view::npos ? text.size() - begin : end - begin);
            if (component.empty() || component == "." || component == "..") { return false; }
            if (end == std::string_view::npos) { break; }
            begin = end + 1;
        }
    }
    return true;
}

inline bool isWellFormedNamespaceProof(const ShaderNamespaceProof& proof)
{
    if (proof.schemaVersion != 2 || proof.compilerProducerIdentity == 0 || proof.roots.empty() || proof.roots.size() > 128 ||
        proof.actualReads.empty() || proof.actualReads.size() > 4096 || proof.negativeLookupPaths.size() > 65536) { return false; }
    std::vector<std::string> rootNames;
    for (const auto& root : proof.roots) {
        if (!isSafeProofPath(root.absolutePath, false) || (!root.exists &&
            (!isSafeProofPath(root.existingWatchedAncestor, false) || !root.entries.empty())) || root.entries.size() > 1000000) { return false; }
        if (std::find(rootNames.begin(), rootNames.end(), root.absolutePath) != rootNames.end()) { return false; }
        rootNames.push_back(root.absolutePath);
        std::string previous;
        for (const auto& entry : root.entries) {
            if (!isSafeProofPath(entry.relativePath, true) || (!previous.empty() && entry.relativePath <= previous) ||
                (entry.kind != ShaderNamespaceEntryKind::File && entry.kind != ShaderNamespaceEntryKind::Directory)) { return false; }
            previous = entry.relativePath;
        }
    }
    std::string previous;
    for (const auto& read : proof.actualReads) {
        if (!isSafeProofPath(read.canonicalPath, false) || (!previous.empty() && read.canonicalPath <= previous) || read.byteSize > 512ull * 1024 * 1024) { return false; }
        previous = read.canonicalPath;
    }
    previous.clear();
    for (const auto& path : proof.negativeLookupPaths) {
        if (!isSafeProofPath(path, false) || (!previous.empty() && path <= previous)) { return false; }
        previous = path;
    }
    return true;
}

// The current proof is produced from an independently complete validated
// immutable epoch. In particular, a newly created higher-precedence file
// changes a namespace before old positive dependency hashes can qualify it.
inline bool namespaceProofMatches(const ShaderNamespaceProof& cached, const ShaderNamespaceProof& current)
{
    return isWellFormedNamespaceProof(cached) && isWellFormedNamespaceProof(current) && cached == current;
}

} // namespace metallic::render

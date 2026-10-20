#pragma once

// Immutable source bytes and resolver namespaces for one compilation epoch.
// Live notifications and complete boundary checks invalidate edited epochs.
// No binary Slang module/IR inputs or source-writer locks are introduced.
#include "SlangShaderCache.h"
#include <slang-com-ptr.h>
#include <slang.h>
#include <slang-tag-version.h>
#include <json.hpp>
#include "Runtime/Render/GAPI/Vulkan/NativeDescriptorHeapSPIRV.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace metallic::render::slang_snapshot {
using Json = nlohmann::json;
namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;
using Clock = std::chrono::steady_clock;

inline uint64_t hash(const void* data, size_t size)
{
    uint64_t value = 14695981039346656037ull;
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) { value = (value ^ bytes[index]) * 1099511628211ull; }
    return value;
}
inline std::string hex(uint64_t value)
{
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}
inline Bytes read(const fs::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() > 512ull * 1024 * 1024) {
        throw std::runtime_error("Invalid input file: " + path.string());
    }
    Bytes bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!file) { throw std::runtime_error("Cannot read: " + path.string()); }
    return bytes;
}
inline std::string pathText(const fs::path& path)
{
    const auto text = fs::absolute(path).lexically_normal().generic_u8string();
    return {reinterpret_cast<const char*>(text.data()), text.size()};
}
inline std::string pathKey(const fs::path& path)
{
    auto text = pathText(path);
#ifdef _WIN32
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    return text;
}
inline bool childOf(const std::string& child, const std::string& parent)
{
    return child == parent || (child.size() > parent.size() && child.compare(0, parent.size(), parent) == 0 && child[parent.size()] == '/');
}

class Blob final : public ISlangBlob {
public:
    explicit Blob(Bytes bytes) : bytes_(std::move(bytes)) {}
    explicit Blob(const std::string& text) : bytes_(text.begin(), text.end()) { bytes_.push_back(0); }
    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid, void** object) override
    {
        if (!object) { return SLANG_E_INVALID_ARG; }
        *object = nullptr;
        if (uuid == ISlangUnknown::getTypeGuid() || uuid == ISlangBlob::getTypeGuid()) {
            *object = static_cast<ISlangBlob*>(this); addRef(); return SLANG_OK;
        }
        return SLANG_E_NO_INTERFACE;
    }
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return ++refs_; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override
    {
        const uint32_t count = --refs_; if (!count) { delete this; } return count;
    }
    SLANG_NO_THROW const void* SLANG_MCALL getBufferPointer() override { return bytes_.data(); }
    SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() override { return bytes_.size(); }
private:
    std::atomic<uint32_t> refs_{1};
    Bytes bytes_;
};

struct Mapping {
    fs::path logical, physical;
    bool monitorOriginal;
    bool recursive = true, namespaceOnly = false;
};
struct FileImage {
    std::string logical;
    fs::path physical;
    Bytes bytes;
    fs::file_time_type writeTime{};
    const uint64_t contentHash;
    FileImage(std::string logicalPath, fs::path physicalPath, Bytes capturedBytes, fs::file_time_type capturedWriteTime)
        : logical(std::move(logicalPath)), physical(std::move(physicalPath)), bytes(std::move(capturedBytes)),
        writeTime(capturedWriteTime), contentHash(hash(bytes.data(), bytes.size())) {}
};

// Full immutable bytes for one epoch, while strict live validation prevents
// publishing/reusing these bytes after an edit. Metadata/mtime are not trusted.
// All path methods implement the SDK's complete ISlangFileSystemExt interface.
class SlangSnapshotFileSystem final : public ISlangFileSystemExt {
public:
    SlangSnapshotFileSystem(const Json& manifest, size_t epoch) : epoch_(epoch), cwdKey_(pathKey(fs::current_path()))
    {
        for (const auto& item : manifest.at("mappings")) {
            mappings_.push_back({fs::u8path(item.at("logical").get<std::string>()), fs::u8path(item.at("physical").get<std::string>()),
                item.value("monitor_original", false), item.value("recursive", true), item.value("namespace_only", false)});
        }
        // Install watchers before capture: byte equality alone could miss an
        // edit-and-restore between checkpoints. Windows notifications advance
        // the epoch even when the final bytes/mtime have been restored.
        try {
            for (const auto& mapping : mappings_) {
                rejectReparse(mapping.physical);
                watch(mapping.physical, mapping.recursive);
                watch(mapping.physical.parent_path(), false);
                if (mapping.monitorOriginal && pathKey(mapping.logical) != pathKey(mapping.physical)) {
                    rejectReparse(mapping.logical);
                    watch(mapping.logical, mapping.recursive);
                    watch(mapping.logical.parent_path(), false);
                }
            }
            capture();
            std::string reason;
            if (!validate(reason)) { throw std::runtime_error("Unstable snapshot capture: " + reason); }
        } catch (...) {
            closeWatches();
            throw;
        }
    }
    ~SlangSnapshotFileSystem()
    {
        closeWatches();
    }
    // Sessions with different root modules share captured bytes and the epoch's
    // complete live-validation boundary, but never accumulate each other's
    // delivered reads, resolver misses or accepted-cache provenance.
    std::unique_ptr<SlangSnapshotFileSystem> createProvenanceView()
    {
        return std::unique_ptr<SlangSnapshotFileSystem>(new SlangSnapshotFileSystem(this));
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID& uuid, void** object) override
    {
        if (!object) { return SLANG_E_INVALID_ARG; }
        *object = castAs(uuid);
        if (!*object) { return SLANG_E_NO_INTERFACE; }
        addRef(); return SLANG_OK;
    }
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return ++refs_; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override
    {
        const uint32_t count = --refs_; if (!count) { delete this; } return count;
    }
    SLANG_NO_THROW void* SLANG_MCALL castAs(const SlangUUID& uuid) override
    {
        if (uuid == ISlangFileSystemExt::getTypeGuid()) { return static_cast<ISlangFileSystemExt*>(this); }
        if (uuid == ISlangFileSystem::getTypeGuid()) { return static_cast<ISlangFileSystem*>(this); }
        if (uuid == ISlangCastable::getTypeGuid()) { return static_cast<ISlangCastable*>(this); }
        if (uuid == ISlangUnknown::getTypeGuid()) { return static_cast<ISlangUnknown*>(this); }
        return nullptr;
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(const char* path, ISlangBlob** out) override
    {
        if (!path || !out) { return SLANG_E_INVALID_ARG; }
        *out = nullptr;
        try {
            std::scoped_lock provenanceLock(provenanceMutex_);
            const auto key = pathKey(fs::u8path(path));
            const auto found = files_.find(key);
            if (found == files_.end()) { misses_.insert(pathText(fs::u8path(path))); return SLANG_E_NOT_FOUND; }
            served_.insert(key);
            *out = new Blob(found->second.bytes);
            return SLANG_OK;
        } catch (...) { return SLANG_FAIL; }
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL getFileUniqueIdentity(const char* path, ISlangBlob** out) override
    {
        if (!path || !out) { return SLANG_E_INVALID_ARG; }
        *out = nullptr;
        try {
            std::scoped_lock provenanceLock(provenanceMutex_);
            const auto key = pathKey(fs::u8path(path));
            if (!files_.contains(key) && !directories_.contains(key)) { misses_.insert(pathText(fs::u8path(path))); return SLANG_E_NOT_FOUND; }
            // Canonical logical identity within immutable epoch; same identity
            // always refers to exactly the same bytes. No content-hash collision
            // can cause distinct paths/files to share a pragma-once identity.
            const auto identity = std::to_string(epoch_) + ":" + key;
            issuedIdentities_.insert_or_assign(identity, key);
            *out = new Blob(identity);
            return SLANG_OK;
        } catch (...) { return SLANG_FAIL; }
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL calcCombinedPath(SlangPathType fromType, const char* from,
        const char* path, ISlangBlob** out) override
    {
        if (!from || !path || !out) { return SLANG_E_INVALID_ARG; }
        *out = nullptr;
        try {
            fs::path base = fs::u8path(from);
            if (fromType == SLANG_PATH_TYPE_FILE) { base = base.parent_path(); }
            else if (fromType != SLANG_PATH_TYPE_DIRECTORY) { return SLANG_E_INVALID_ARG; }
            *out = new Blob(pathText(base / fs::u8path(path)));
            return SLANG_OK;
        } catch (...) { return SLANG_FAIL; }
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL getPathType(const char* path, SlangPathType* out) override
    {
        if (!path || !out) { return SLANG_E_INVALID_ARG; }
        try {
            std::scoped_lock provenanceLock(provenanceMutex_);
            const auto key = pathKey(fs::u8path(path));
            if (files_.contains(key)) { *out = SLANG_PATH_TYPE_FILE; return SLANG_OK; }
            if (directories_.contains(key)) { *out = SLANG_PATH_TYPE_DIRECTORY; return SLANG_OK; }
            misses_.insert(pathText(fs::u8path(path))); return SLANG_E_NOT_FOUND;
        } catch (...) { return SLANG_FAIL; }
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL getPath(PathKind kind, const char* path, ISlangBlob** out) override
    {
        if (!path || !out) { return SLANG_E_INVALID_ARG; }
        *out = nullptr;
        try {
            std::scoped_lock provenanceLock(provenanceMutex_);
            const auto key = pathKey(fs::u8path(path));
            if (kind == PathKind::OperatingSystem) {
                // A frozen logical source path does not expose its backing file
                // for out-of-band compiler reads; all source must use loadFile.
                return SLANG_E_NOT_IMPLEMENTED;
            }
            if (kind == PathKind::Canonical && !files_.contains(key) && !directories_.contains(key)) {
                misses_.insert(pathText(fs::u8path(path))); return SLANG_E_NOT_FOUND;
            }
            if (kind != PathKind::Canonical && kind != PathKind::Simplified && kind != PathKind::Display) {
                return SLANG_E_INVALID_ARG;
            }
            auto found = files_.find(key);
            *out = new Blob(found != files_.end() ? found->second.logical : pathText(fs::u8path(path)));
            return SLANG_OK;
        } catch (...) { return SLANG_FAIL; }
    }
    SLANG_NO_THROW void SLANG_MCALL clearCache() override
    {
        // Immutable epoch has no mutable path cache. Provenance is retained.
    }
    SLANG_NO_THROW SlangResult SLANG_MCALL enumeratePathContents(const char* path,
        FileSystemContentsCallBack callback, void* user) override
    {
        if (!path || !callback) { return SLANG_E_INVALID_ARG; }
        try {
            if (!directories_.contains(pathKey(fs::u8path(path)))) {
                std::scoped_lock provenanceLock(provenanceMutex_);
                misses_.insert(pathText(fs::u8path(path)));
                return SLANG_E_NOT_FOUND;
            }
            std::map<std::string, SlangPathType> children;
            for (const auto& [key, file] : files_) {
                if (pathKey(fs::u8path(file.logical).parent_path()) == pathKey(fs::u8path(path))) {
                    const auto name = fs::u8path(file.logical).filename().u8string();
                    children[{reinterpret_cast<const char*>(name.data()), name.size()}] = SLANG_PATH_TYPE_FILE;
                }
            }
            for (const auto& directory : directories_) {
                if (pathKey(fs::u8path(directory).parent_path()) == pathKey(fs::u8path(path))) {
                    const auto name = fs::u8path(directory).filename().u8string();
                    children[{reinterpret_cast<const char*>(name.data()), name.size()}] = SLANG_PATH_TYPE_DIRECTORY;
                }
            }
            for (const auto& [name, type] : children) { callback(type, name.c_str(), user); }
            return SLANG_OK;
        } catch (...) { return SLANG_FAIL; }
    }
    SLANG_NO_THROW OSPathKind SLANG_MCALL getOSPathKind() override { return OSPathKind::None; }

    bool validateNotifications(std::string& reason) const
    {
        if (epochOwner_) { return epochOwner_->validateNotifications(reason); }
        std::scoped_lock notificationLock(notificationMutex_);
        if (pathKey(fs::current_path()) != cwdKey_) { reason = "process source lookup cwd changed during epoch"; return false; }
#ifdef _WIN32
        if (watches_.empty()) { reason = "source epoch notifications unavailable"; return false; }
        for (const auto& watcher : watches_) {
            if (!watcher->unchanged(reason)) { return false; }
        }
        return true;
#else
        reason = "Windows source epoch notifications unavailable";
        return false;
#endif
    }
    bool validate(std::string& reason) const
    {
        if (epochOwner_) { return epochOwner_->validate(reason); }
        return validateSource(reason, nullptr);
    }
    bool validateCacheHit(std::string& reason) const
    {
        // A positive cache envelope contributes its qualified reads to this
        // invocation. Keep the complete resolver namespace, but reread only
        // these dependencies and any source already delivered to this view.
        // Forwarding to the owner here would discard per-invocation provenance.
        try {
            std::set<std::string> reads;
            {
                std::scoped_lock provenanceLock(provenanceMutex_);
                reads = served_;
                reads.insert(qualifiedCacheReads_.begin(), qualifiedCacheReads_.end());
            }
            return validateSource(reason, &reads);
        } catch (const std::exception& error) { reason = error.what(); return false; }
    }
    Json provenance() const
    {
        std::scoped_lock provenanceLock(provenanceMutex_);
        Json output{{"epoch", epoch_}, {"graph_file_count", files_.size()}, {"served", Json::array()}, {"negative_lookups", misses_}};
        for (const auto& key : served_) {
            const auto& file = files_.at(key);
            output["served"].push_back({{"logical", file.logical}, {"physical", pathText(file.physical)},
                {"bytes", file.bytes.size()}, {"read_bytes_fnv1a64", hex(file.contentHash)}});
        }
        output["namespace"] = namespaces_;
#ifdef _WIN32
        output["notification_watchers"] = Json::array();
        for (const auto& watcher : watches_) {
            output["notification_watchers"].push_back({{"root", watcher->rootText}, {"recursive", watcher->recursive},
                {"api", "ReadDirectoryChangesW-overlapped-no-rearm"}, {"buffer_bytes", sizeof(watcher->buffer)}});
        }
#endif
        return output;
    }

    metallic::render::ShaderNamespaceProof sourceProof(uint64_t producerIdentity) const
    {
        std::scoped_lock provenanceLock(provenanceMutex_);
        metallic::render::ShaderNamespaceProof proof;
        proof.compilerProducerIdentity = producerIdentity;
        for (const auto& mapping : mappings_) {
            metallic::render::ShaderNamespaceRootProof root;
            root.absolutePath = pathText(mapping.logical);
            root.recursive = mapping.recursive;
            for (const auto& [relative, kind] : namespaces_.at(pathKey(mapping.logical))) {
                root.entries.push_back({relative, kind == "file" ? metallic::render::ShaderNamespaceEntryKind::File :
                    metallic::render::ShaderNamespaceEntryKind::Directory});
            }
            proof.roots.push_back(std::move(root));
        }
        const auto allReads = [&] { auto reads = served_; reads.insert(qualifiedCacheReads_.begin(), qualifiedCacheReads_.end()); return reads; }();
        for (const auto& key : allReads) {
            const auto& image = files_.at(key);
            proof.actualReads.push_back({image.logical, image.bytes.size(), image.contentHash});
        }
        std::sort(proof.actualReads.begin(), proof.actualReads.end(), [](const auto& left, const auto& right) {
            return left.canonicalPath < right.canonicalPath;
        });
        const auto allMisses = [&] { auto misses = misses_; misses.insert(qualifiedCacheMisses_.begin(), qualifiedCacheMisses_.end()); return misses; }();
        for (const auto& path : allMisses) {
            // A bare-cwd or missing-parent probe outside the captured search
            // graph must expand the graph or bypass reuse. Never omit it from
            // persistent proof merely because no positive file was loaded.
            if (!covers(fs::u8path(path))) {
                throw std::runtime_error("Negative lookup outside monitored namespace; extend snapshot or reject reuse: " + path);
            }
            proof.negativeLookupPaths.push_back(path);
        }
        if (!metallic::render::isWellFormedNamespaceProof(proof)) { throw std::runtime_error("Snapshot cannot provide a complete V3 source proof"); }
        return proof;
    }

    bool qualifyCachedProof(const metallic::render::ShaderNamespaceProof& cached,
        uint64_t producerIdentity, metallic::render::ShaderNamespaceProof& current, std::string& reason) const
    {
        current = {};
        if (!validateNotifications(reason) || !metallic::render::isWellFormedNamespaceProof(cached) ||
            cached.compilerProducerIdentity != producerIdentity) { return false; }
        std::scoped_lock provenanceLock(provenanceMutex_);
        current.schemaVersion = 2; current.compilerProducerIdentity = producerIdentity;
        for (const auto& mapping : mappings_) {
            metallic::render::ShaderNamespaceRootProof root;
            root.absolutePath = pathText(mapping.logical);
            root.recursive = mapping.recursive;
            for (const auto& [relative, kind] : namespaces_.at(pathKey(mapping.logical))) {
                root.entries.push_back({relative, kind == "file" ? metallic::render::ShaderNamespaceEntryKind::File :
                    metallic::render::ShaderNamespaceEntryKind::Directory});
            }
            current.roots.push_back(std::move(root));
        }
        for (const auto& dependency : cached.actualReads) {
            const auto found = files_.find(pathKey(fs::u8path(dependency.canonicalPath)));
            if (found == files_.end()) { reason = "V3 positive source missing in current snapshot"; return false; }
            const auto& image = found->second;
            current.actualReads.push_back({image.logical, image.bytes.size(), image.contentHash});
        }
        for (const auto& missing : cached.negativeLookupPaths) {
            if (!covers(fs::u8path(missing)) || files_.contains(pathKey(fs::u8path(missing)))) {
                reason = "V3 negative lookup now resolves or is not namespace-qualified"; return false;
            }
            current.negativeLookupPaths.push_back(missing);
        }
        if (!metallic::render::namespaceProofMatches(cached, current)) { reason = "V3 namespace/source proof changed"; return false; }
        return true;
    }

    // One accepted-hit operation. Keep the exact current proof supplied to
    // the decoder, and merge only after all envelope and SPIR-V gates pass.
    // Callers cannot accept a bare old proof before payload validation.
    bool decodeAndAcceptCache(std::span<const uint8_t> bytes, uint64_t requestHash,
        uint64_t producerIdentity, metallic::render::slang_shader_cache::CachedPayload& output, std::string& reason)
    {
        metallic::render::ShaderNamespaceProof current;
        const auto qualify = [&](const metallic::render::ShaderNamespaceProof& cached,
            metallic::render::ShaderNamespaceProof& qualified) {
            if (!qualifyCachedProof(cached, producerIdentity, qualified, reason)) { return false; }
            current = qualified; return true;
        };
        if (!metallic::render::slang_shader_cache::decode(bytes, requestHash, qualify, output)) { return false; }
        if (!validateNotifications(reason)) { output = {}; return false; }
        std::scoped_lock provenanceLock(provenanceMutex_);
        for (const auto& dependency : current.actualReads) {
            qualifiedCacheReads_.insert(pathKey(fs::u8path(dependency.canonicalPath)));
        }
        qualifiedCacheMisses_.insert(current.negativeLookupPaths.begin(), current.negativeLookupPaths.end());
        return true;
    }

    uint64_t immutableIdentity() const
    {
        Json identity{{"namespace", namespaces_}, {"files", Json::array()}, {"epoch", epoch_}};
        for (const auto& [key, image] : files_) {
            identity["files"].push_back({{"path", key}, {"bytes", image.bytes.size()},
                {"fnv1a64", hex(image.contentHash)}});
        }
        const auto text = identity.dump();
        return hash(text.data(), text.size());
    }
    bool monitorsOriginal() const
    {
        return std::any_of(mappings_.begin(), mappings_.end(), [](const auto& mapping) { return mapping.monitorOriginal; });
    }
    bool actualDependency(const fs::path& path, metallic::render::ShaderSourceReadProof& output,
        fs::file_time_type& writeTime) const
    {
        std::scoped_lock provenanceLock(provenanceMutex_);
        const auto key = pathKey(path);
        const auto found = files_.find(key);
        if (found == files_.end() || (!served_.contains(key) && !qualifiedCacheReads_.contains(key))) { return false; }
        const auto& image = found->second;
        output = {image.logical, image.bytes.size(), image.contentHash};
        writeTime = image.writeTime;
        return true;
    }
    bool actualIssuedDependency(const char* identity, metallic::render::ShaderSourceReadProof& output,
        fs::file_time_type& writeTime) const
    {
        if (!identity) { return false; }
        std::scoped_lock provenanceLock(provenanceMutex_);
        // Exact tokens emitted by this epoch only. No prefix stripping and
        // no filesystem normalization of opaque SDK dependency identities.
        const auto issued = issuedIdentities_.find(identity);
        if (issued == issuedIdentities_.end()) { return false; }
        const auto found = files_.find(issued->second);
        if (found == files_.end() || (!served_.contains(issued->second) && !qualifiedCacheReads_.contains(issued->second))) { return false; }
        const auto& image = found->second;
        output = {image.logical, image.bytes.size(), image.contentHash};
        writeTime = image.writeTime; return true;
    }
    bool hasUnqualifiedLookups() const
    {
        std::scoped_lock provenanceLock(provenanceMutex_);
        return std::ranges::any_of(misses_, [&](const auto& path) { return !covers(fs::u8path(path)); });
    }
    bool covers(const fs::path& path) const
    {
        const auto key = pathKey(path);
        return std::any_of(mappings_.begin(), mappings_.end(), [&](const auto& mapping) {
            const auto root = pathKey(mapping.logical);
            if (!childOf(key, root)) { return false; }
            if (mapping.recursive || key == root) { return true; }
            // Root-only cwd names prove a missing first path component. A
            // lookup inside an existing untracked child directory is unknown.
            const auto relative = key.substr(root.size() + 1);
            const auto separator = relative.find('/');
            const auto first = relative.substr(0, separator);
            const auto& names = namespaces_.at(root);
            const auto found = std::find_if(names.begin(), names.end(), [&](const auto& entry) {
                auto name = entry.first;
                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return name == first;
            });
            if (found == names.end()) { return true; }
            return separator == std::string::npos && files_.contains(key);
        });
    }
private:
    bool validateSource(std::string& reason, const std::set<std::string>* reads) const
    {
        try {
            if (!validateNotifications(reason)) { return false; }
            for (const auto& mapping : mappings_) {
                const auto current = namespaceFor(mapping.physical, mapping.recursive);
                if (current != namespaces_.at(pathKey(mapping.logical))) {
                    reason = "physical namespace changed: " + pathText(mapping.logical); return false;
                }
                if (mapping.monitorOriginal && namespaceFor(mapping.logical, mapping.recursive) != current) {
                    reason = "original namespace changed: " + pathText(mapping.logical); return false;
                }
            }
            const auto validateFile = [&](const std::string& key, const FileImage& file) {
                if (read(file.physical) != file.bytes) {
                    reason = "physical bytes changed: " + file.logical; return false;
                }
                for (const auto& mapping : mappings_) {
                    if (mapping.monitorOriginal && childOf(key, pathKey(mapping.logical)) && read(fs::u8path(file.logical)) != file.bytes) {
                        reason = "original bytes changed: " + file.logical; return false;
                    }
                }
                return true;
            };
            if (reads) {
                for (const auto& key : *reads) {
                    if (!validateFile(key, files_.at(key))) { return false; }
                }
            } else {
                for (const auto& [key, file] : files_) {
                    if (!validateFile(key, file)) { return false; }
                }
            }
            return validateNotifications(reason);
        } catch (const std::exception& error) { reason = error.what(); return false; }
    }
    explicit SlangSnapshotFileSystem(SlangSnapshotFileSystem* source)
        : epoch_(source->epoch_), cwdKey_(source->cwdKey_),
          sourceImage_(source->sourceImage_), epochOwner_(source->epochOwner_ ? source->epochOwner_ : source) {}

#ifdef _WIN32
    struct DirectoryWatch {
        std::string rootText;
        bool recursive;
        HANDLE directory = INVALID_HANDLE_VALUE, completionPort = nullptr;
        OVERLAPPED overlapped{};
        // DWORD-aligned 64KiB buffer; exactly one request, never re-armed.
        std::array<DWORD, 16384> buffer{};
        bool queued = false;
        mutable bool completed = false;
        mutable DWORD completionBytes = 0, completionError = ERROR_SUCCESS;
        explicit DirectoryWatch(const fs::path& root, bool watchSubtree) : rootText(pathText(root)), recursive(watchSubtree)
        {
            if (GetDriveTypeW(root.root_path().c_str()) == DRIVE_REMOTE) {
                throw std::runtime_error("Remote source notification semantics unsupported");
            }
            try {
                directory = CreateFileW(root.c_str(), FILE_LIST_DIRECTORY,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OVERLAPPED, nullptr);
                if (directory == INVALID_HANDLE_VALUE) { throw std::runtime_error("Cannot open source watcher root: " + root.string()); }
                // Port association makes the pending request independent of
                // the capture thread, which may exit before this epoch is used.
                // Callers poll from arbitrary threads; the notification mutex
                // serializes state, so the port must not throttle those threads.
                completionPort = CreateIoCompletionPort(directory, nullptr, 0, MAXDWORD);
                if (!completionPort) { throw std::runtime_error("Cannot create source watcher completion port"); }
                const DWORD filter = FILE_NOTIFY_CHANGE_FILE_NAME | FILE_NOTIFY_CHANGE_DIR_NAME |
                    FILE_NOTIFY_CHANGE_ATTRIBUTES | FILE_NOTIFY_CHANGE_SECURITY |
                    (recursive ? FILE_NOTIFY_CHANGE_SIZE | FILE_NOTIFY_CHANGE_LAST_WRITE : 0);
                const BOOL started = ReadDirectoryChangesW(directory, buffer.data(), static_cast<DWORD>(sizeof(buffer)),
                    recursive ? TRUE : FALSE, filter, nullptr, &overlapped, nullptr);
                if (!started) { throw std::runtime_error("Source filesystem cannot queue notifications: " + root.string()); }
                queued = true;
            } catch (...) {
                closeHandles();
                throw;
            }
        }
        ~DirectoryWatch()
        {
            if (queued && !completed) {
                CancelIoEx(directory, &overlapped);
                DWORD bytes = 0;
                ULONG_PTR completionKey = 0;
                OVERLAPPED* packet = nullptr;
                // Keep OVERLAPPED/buffer alive until cancellation/completion.
                GetQueuedCompletionStatus(completionPort, &bytes, &completionKey, &packet, INFINITE);
                assert(packet == &overlapped);
            }
            closeHandles();
        }
        void closeHandles() noexcept
        {
            if (directory != INVALID_HANDLE_VALUE) { CloseHandle(directory); }
            if (completionPort) { CloseHandle(completionPort); }
        }
        DirectoryWatch(const DirectoryWatch&) = delete;
        DirectoryWatch& operator=(const DirectoryWatch&) = delete;
        bool unchanged(std::string& reason) const
        {
            if (!completed) {
                ULONG_PTR completionKey = 0;
                OVERLAPPED* packet = nullptr;
                const BOOL succeeded = GetQueuedCompletionStatus(completionPort, &completionBytes, &completionKey, &packet, 0);
                const DWORD error = succeeded ? ERROR_SUCCESS : GetLastError();
                if (!packet) {
                    if (error == WAIT_TIMEOUT) { return true; }
                    reason = "source epoch notification poll failed " + std::to_string(error) + ": " + rootText;
                    return false;
                }
                assert(packet == &overlapped);
                completed = true;
                completionError = error;
            }
            // Consuming the single completion packet must never make a later
            // validation treat an already advanced epoch as unchanged.
            if (completionError == ERROR_SUCCESS) {
                reason = (completionBytes ? "source epoch notification advanced: " : "source epoch notification overflow/zero-byte completion: ") + rootText;
            } else {
                reason = (completionError == ERROR_NOTIFY_ENUM_DIR ? "source epoch notification overflow: " :
                    "source epoch notification failed " + std::to_string(completionError) + ": ") + rootText;
            }
            return false;
        }
    };
#endif
    void closeWatches()
    {
#ifdef _WIN32
        watches_.clear();
        watchedRoots_.clear();
#endif
    }
    static void rejectReparse(const fs::path& path)
    {
#ifdef _WIN32
        const DWORD attributes = GetFileAttributesW(path.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES) { throw std::runtime_error("Cannot inspect source namespace: " + path.string()); }
        if (attributes & FILE_ATTRIBUTE_REPARSE_POINT) {
            throw std::runtime_error("Source reparse points, junctions and mount points are unsupported: " + path.string());
        }
#else
        if (fs::is_symlink(path)) { throw std::runtime_error("Source symlinks are unsupported: " + path.string()); }
#endif
    }
    void watch(const fs::path& root, bool recursive = true)
    {
#ifdef _WIN32
        rejectReparse(root);
        const auto key = std::make_pair(pathKey(root), recursive);
        if (watchedRoots_.contains(key)) { return; }
        watches_.push_back(std::make_unique<DirectoryWatch>(root, recursive));
        watchedRoots_.insert(key);
#else
        (void)root; (void)recursive;
        throw std::runtime_error("Source epoch notifications require Windows");
#endif
    }
    static std::map<std::string, std::string> namespaceFor(const fs::path& root, bool recursive = true)
    {
        std::map<std::string, std::string> names;
#ifdef _WIN32
        // Enumeration already returns each entry's attributes. Querying them
        // again per file dominates ordinary hit validation on Windows.
        std::vector<fs::path> pending{fs::path{}};
        while (!pending.empty()) {
            const auto relativeDirectory = std::move(pending.back());
            pending.pop_back();
            const auto directory = root / relativeDirectory;
            const auto checkDirectory = [&] {
                const DWORD attributes = GetFileAttributesW(directory.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || attributes & FILE_ATTRIBUTE_REPARSE_POINT ||
                    !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    throw std::runtime_error("Source namespace directory unavailable or replaced: " + directory.string());
                }
            };
            checkDirectory();
            WIN32_FIND_DATAW entry{};
            struct Search {
                HANDLE handle;
                ~Search() { if (handle != INVALID_HANDLE_VALUE) { FindClose(handle); } }
            } search{FindFirstFileExW((directory / L"*").c_str(), FindExInfoBasic, &entry, FindExSearchNameMatch, nullptr, 0)};
            if (search.handle == INVALID_HANDLE_VALUE) {
                const DWORD error = GetLastError();
                if (error == ERROR_FILE_NOT_FOUND) { checkDirectory(); continue; }
                throw std::runtime_error("Cannot enumerate source namespace (" + std::to_string(error) + "): " + directory.string());
            }
            do {
                const std::wstring_view name(entry.cFileName);
                if (name == L"." || name == L"..") { continue; }
                if (entry.dwFileAttributes & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DEVICE)) {
                    throw std::runtime_error("Unsupported source filesystem object: " + (directory / name).string());
                }
                const auto relative = relativeDirectory / name;
                const auto text = relative.generic_u8string();
                const std::string key(reinterpret_cast<const char*>(text.data()), text.size());
                const bool isDirectory = (entry.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
                names.emplace(key, isDirectory ? "directory" : "file");
                if (recursive && isDirectory) { pending.push_back(relative); }
            } while (FindNextFileW(search.handle, &entry));
            const DWORD error = GetLastError();
            if (error != ERROR_NO_MORE_FILES) {
                throw std::runtime_error("Source namespace enumeration failed (" + std::to_string(error) + "): " + directory.string());
            }
        }
#else
        rejectReparse(root);
        if (!fs::is_directory(root)) { throw std::runtime_error("Namespace root is absent: " + root.string()); }
        const auto add = [&](const auto& item) {
            // Supporting symlink aliases requires canonical backing identity and
            // target namespace tracking. Reject unsupported namespace aliases.
            rejectReparse(item.path());
            if (item.is_symlink()) { throw std::runtime_error("Symlinks are not supported: " + item.path().string()); }
            const auto text = item.path().lexically_relative(root).generic_u8string();
            const std::string relative(reinterpret_cast<const char*>(text.data()), text.size());
            if (item.is_directory()) { names.emplace(relative, "directory"); }
            else if (item.is_regular_file()) { names.emplace(relative, "file"); }
            else { throw std::runtime_error("Unsupported source filesystem object"); }
        };
        if (recursive) { for (const auto& item : fs::recursive_directory_iterator(root)) { add(item); } }
        else { for (const auto& item : fs::directory_iterator(root)) { add(item); } }
#endif
        return names;
    }
    void capture()
    {
        for (const auto& mapping : mappings_) {
            const auto names = namespaceFor(mapping.physical, mapping.recursive);
            if (!namespaces_.emplace(pathKey(mapping.logical), names).second) { throw std::runtime_error("Duplicate logical mapping"); }
            directories_.insert(pathKey(mapping.logical));
            for (const auto& [relative, type] : names) {
                const auto logical = mapping.logical / fs::u8path(relative);
                if (type == "directory") { directories_.insert(pathKey(logical)); }
                else {
                    auto extension = logical.extension().string();
                    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    if (extension == ".slang-module" || extension == ".slang-lib") {
                        throw std::runtime_error("Disk Slang module/IR inputs are forbidden");
                    }
                    // cwd namespaces need complete root names, not all editor
                    // binaries/cache trees. Unknown present files cannot be
                    // qualified as negative lookups by covers() above.
                    if (mapping.namespaceOnly && extension != ".slang" && extension != ".h" && extension != ".hpp" &&
                        extension != ".hlsl" && extension != ".hlsli" && extension != ".glsl" && extension != ".inc") { continue; }
                    auto bytes = read(mapping.physical / fs::u8path(relative));
                    if (!files_.emplace(pathKey(logical), FileImage{pathText(logical), mapping.physical / fs::u8path(relative),
                        std::move(bytes), fs::last_write_time(mapping.physical / fs::u8path(relative))}).second) {
                        throw std::runtime_error("Overlapping mapping or case alias");
                    }
                }
            }
        }
    }
    mutable std::mutex provenanceMutex_, notificationMutex_;
    std::atomic<uint32_t> refs_{1};
    size_t epoch_;
    std::string cwdKey_;
    // Only the owning epoch's constructor populates this image. Views retain
    // it without copying source bytes; their owner outlives every session/view.
    struct SourceImage {
        std::vector<Mapping> mappings;
        std::map<std::string, FileImage> files;
        std::set<std::string> directories;
        std::map<std::string, std::map<std::string, std::string>> namespaces;
    };
    std::shared_ptr<SourceImage> sourceImage_ = std::make_shared<SourceImage>();
    SlangSnapshotFileSystem* epochOwner_ = nullptr;
    std::vector<Mapping>& mappings_ = sourceImage_->mappings;
    std::map<std::string, FileImage>& files_ = sourceImage_->files;
    std::set<std::string>& directories_ = sourceImage_->directories;
    std::map<std::string, std::map<std::string, std::string>>& namespaces_ = sourceImage_->namespaces;
    std::set<std::string> served_, misses_, qualifiedCacheReads_, qualifiedCacheMisses_;
    std::map<std::string, std::string> issuedIdentities_;
#ifdef _WIN32
    std::vector<std::unique_ptr<DirectoryWatch>> watches_;
    std::set<std::pair<std::string, bool>> watchedRoots_;
#endif
};

} // namespace metallic::render::slang_snapshot

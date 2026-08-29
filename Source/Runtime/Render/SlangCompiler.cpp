#include "Runtime/Render/SlangCompiler.h"

#include <slang-com-ptr.h>
#include <slang-tag-version.h>
#include <slang.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace metallic::render {
namespace {

// Versioned independently from Slang so malformed or stale cache files fail closed.
constexpr std::array<char, 8> kShaderCacheMagic{'M', 'T', 'L', 'S', 'P', 'V', '0', '1'};
constexpr uint32_t kShaderCacheVersion = 2;
constexpr uint32_t kShaderCacheRequestVersion = 3;
constexpr uint32_t kMaxShaderDependencyCount = 4096;
constexpr uint32_t kMaxShaderDependencyPathSize = 32768;
constexpr uint64_t kMaxShaderCacheFileSize = 512ull * 1024ull * 1024ull;
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr uint32_t kSpirvMagic = 0x07230203u;
constexpr auto kShaderDependencyHashInterval = std::chrono::milliseconds(500);

std::atomic<SlangShaderDebugMode> gSlangShaderDebugMode{SlangShaderDebugMode::Disabled};

struct ShaderDependencyStamp {
    bool exists = false;
    uintmax_t fileSize = 0;
    std::filesystem::file_time_type writeTime{};
    uint64_t contentHash = 0;

    bool operator==(const ShaderDependencyStamp&) const = default;
};

struct ShaderDependencySnapshot {
    std::filesystem::path path;
    ShaderDependencyStamp stamp;
};

struct TrackedShaderDependency {
    std::filesystem::path path;
    ShaderDependencyStamp acceptedStamp;
    std::optional<ShaderDependencyStamp> pendingStamp;
    std::chrono::steady_clock::time_point pendingSince{};
    ShaderDependencyStamp reportedStamp;
    std::chrono::steady_clock::time_point lastReportedAt{};
    std::chrono::steady_clock::time_point nextContentHashAt{};
    bool dirty = false;
};

std::mutex gShaderDependencyMutex;
std::unordered_map<std::string, TrackedShaderDependency> gShaderDependencies;

struct ShaderCacheHeader {
    std::array<char, 8> magic{};
    uint32_t version = 0;
    uint32_t headerSize = 0;
    uint64_t requestHash = 0;
    uint32_t dependencyCount = 0;
    uint32_t reserved = 0;
    uint64_t spirvByteSize = 0;
    uint64_t payloadHash = 0;
};

struct ShaderCacheDependencyHeader {
    uint32_t pathSize = 0;
    uint32_t reserved = 0;
    uint64_t contentHash = 0;
};

static_assert(std::is_trivially_copyable_v<ShaderCacheHeader>);
static_assert(std::is_trivially_copyable_v<ShaderCacheDependencyHeader>);
static_assert(sizeof(ShaderCacheHeader) == 48);
static_assert(sizeof(ShaderCacheDependencyHeader) == 16);

uint64_t hashBytes(uint64_t hash, const void* data, size_t byteSize)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < byteSize; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

template <typename T>
uint64_t hashValue(uint64_t hash, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    return hashBytes(hash, &value, sizeof(T));
}

uint64_t hashText(uint64_t hash, const char* value)
{
    const char* text = value != nullptr ? value : "";
    const uint64_t length = std::strlen(text);
    hash = hashValue(hash, length);
    return hashBytes(hash, text, static_cast<size_t>(length));
}

std::filesystem::path normalizedAbsolutePath(
    const std::filesystem::path& path,
    const std::filesystem::path& relativeBase = {})
{
    std::error_code pathError;
    std::filesystem::path resolved = path;
    if (resolved.is_relative() && !relativeBase.empty()) {
        resolved = relativeBase / resolved;
    }
    resolved = std::filesystem::absolute(resolved, pathError);
    return pathError ? path.lexically_normal() : resolved.lexically_normal();
}

bool hashFileContents(const std::filesystem::path& path, uint64_t& outHash);

bool sameShaderDependencyMetadata(
    const ShaderDependencyStamp& left,
    const ShaderDependencyStamp& right)
{
    return left.exists == right.exists &&
        left.fileSize == right.fileSize &&
        left.writeTime == right.writeTime;
}

ShaderDependencyStamp shaderDependencyMetadata(const std::filesystem::path& path)
{
    std::error_code fileError;
    const std::filesystem::file_time_type writeTime =
        std::filesystem::last_write_time(path, fileError);
    if (fileError) {
        return {};
    }
    const uintmax_t fileSize = std::filesystem::file_size(path, fileError);
    if (fileError) {
        return {};
    }
    return ShaderDependencyStamp{
        .exists = true,
        .fileSize = fileSize,
        .writeTime = writeTime,
    };
}

ShaderDependencyStamp shaderDependencyStamp(const std::filesystem::path& path)
{
    ShaderDependencyStamp stamp = shaderDependencyMetadata(path);
    if (!stamp.exists || !hashFileContents(path, stamp.contentHash)) {
        return {};
    }
    return stamp;
}

ShaderDependencyStamp currentTrackedDependencyStamp(
    TrackedShaderDependency& tracked,
    std::chrono::steady_clock::time_point now,
    bool forceContentHash)
{
    ShaderDependencyStamp current = shaderDependencyMetadata(tracked.path);
    if (!current.exists) {
        return current;
    }

    const ShaderDependencyStamp* reusable = nullptr;
    if (tracked.pendingStamp.has_value() &&
        sameShaderDependencyMetadata(current, *tracked.pendingStamp)) {
        reusable = &*tracked.pendingStamp;
    } else if (tracked.dirty &&
        sameShaderDependencyMetadata(current, tracked.reportedStamp)) {
        reusable = &tracked.reportedStamp;
    } else if (sameShaderDependencyMetadata(current, tracked.acceptedStamp)) {
        reusable = &tracked.acceptedStamp;
    }

    if (!forceContentHash && reusable != nullptr && now < tracked.nextContentHashAt) {
        current.contentHash = reusable->contentHash;
        return current;
    }
    if (!hashFileContents(tracked.path, current.contentHash)) {
        return {};
    }
    tracked.nextContentHashAt = now + kShaderDependencyHashInterval;
    return current;
}

bool snapshotShaderDependencies(
    std::span<const std::filesystem::path> dependencies,
    std::vector<ShaderDependencySnapshot>& outSnapshots)
{
    outSnapshots.clear();
    outSnapshots.reserve(dependencies.size());
    for (const std::filesystem::path& dependency : dependencies) {
        const std::filesystem::path normalized = normalizedAbsolutePath(dependency);
        ShaderDependencyStamp stamp = shaderDependencyStamp(normalized);
        if (!stamp.exists) {
            outSnapshots.clear();
            return false;
        }
        outSnapshots.push_back(ShaderDependencySnapshot{
            .path = normalized,
            .stamp = stamp,
        });
    }
    return true;
}

bool shaderDependencySnapshotsMatch(std::span<const ShaderDependencySnapshot> snapshots)
{
    return std::all_of(
        snapshots.begin(),
        snapshots.end(),
        [](const ShaderDependencySnapshot& snapshot) {
            return shaderDependencyStamp(snapshot.path) == snapshot.stamp;
        });
}

void registerShaderDependencies(std::span<const ShaderDependencySnapshot> dependencies)
{
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock(gShaderDependencyMutex);
    for (const ShaderDependencySnapshot& dependency : dependencies) {
        const std::string key = dependency.path.generic_string();
        if (key.empty()) {
            continue;
        }
        auto [iter, inserted] = gShaderDependencies.try_emplace(key);
        if (inserted) {
            iter->second.path = dependency.path;
            iter->second.acceptedStamp = dependency.stamp;
            iter->second.nextContentHashAt = now + kShaderDependencyHashInterval;
        }
    }
}

void publishShaderDependencies(
    std::span<const ShaderDependencySnapshot> dependencies,
    ShaderCompileResult& outResult)
{
    outResult.dependencies.clear();
    outResult.dependencies.reserve(dependencies.size());
    for (const ShaderDependencySnapshot& dependency : dependencies) {
        outResult.dependencies.push_back(dependency.path.generic_string());
    }
    registerShaderDependencies(dependencies);
}

uint64_t shaderRequestHash(const SlangShaderDesc& desc, SlangShaderDebugMode debugMode)
{
    uint64_t hash = kFnvOffset;
    hash = hashValue(hash, kShaderCacheRequestVersion);
    hash = hashText(hash, SLANG_VERSION_NUMERIC);
    hash = hashText(hash, desc.moduleName);
    hash = hashText(hash, desc.entryPointName);
    hash = hashText(hash, desc.profileName != nullptr ? desc.profileName : kDefaultSlangProfileName);
    hash = hashValue(hash, static_cast<uint8_t>(debugMode));
    const std::string searchPath = normalizedAbsolutePath(desc.searchPath).generic_string();
    hash = hashText(hash, searchPath.c_str());
    hash = hashValue(hash, desc.additionalSearchPathCount);
    for (uint32_t index = 0; index < desc.additionalSearchPathCount; ++index) {
        const std::string additionalSearchPath = normalizedAbsolutePath(
            desc.additionalSearchPaths[index]).generic_string();
        hash = hashText(hash, additionalSearchPath.c_str());
    }
    hash = hashValue(hash, desc.capabilityCount);
    for (uint32_t index = 0; index < desc.capabilityCount; ++index) {
        hash = hashText(hash, desc.capabilities != nullptr ? desc.capabilities[index] : nullptr);
    }
    hash = hashValue(hash, desc.macroDefineCount);
    for (uint32_t index = 0; index < desc.macroDefineCount; ++index) {
        const SlangMacroDefine* macro = desc.macroDefines != nullptr
            ? &desc.macroDefines[index]
            : nullptr;
        hash = hashText(hash, macro != nullptr ? macro->name : nullptr);
        hash = hashText(
            hash,
            macro != nullptr && macro->value != nullptr ? macro->value : "1");
    }
    return hash;
}

std::filesystem::path shaderCachePath(
    const SlangShaderCacheOptions& cacheOptions,
    uint64_t requestHash)
{
    const std::filesystem::path directory =
        cacheOptions.cacheDirectory != nullptr && cacheOptions.cacheDirectory[0] != '\0'
        ? std::filesystem::path(cacheOptions.cacheDirectory)
        : std::filesystem::path(PROJECT_SOURCE_DIR) / ".cache" / "shaders" / "spirv";
    std::array<char, 32> fileName{};
    std::snprintf(
        fileName.data(),
        fileName.size(),
        "%016llx.spv",
        static_cast<unsigned long long>(requestHash));
    return directory / fileName.data();
}

bool hashFileContents(const std::filesystem::path& path, uint64_t& outHash)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }

    uint64_t hash = kFnvOffset;
    std::array<char, 64 * 1024> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize readSize = stream.gcount();
        if (readSize > 0) {
            hash = hashBytes(hash, buffer.data(), static_cast<size_t>(readSize));
        }
    }
    if (!stream.eof()) {
        return false;
    }
    outHash = hash;
    return true;
}

template <typename T>
void appendPayloadValue(std::vector<uint8_t>& payload, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const size_t offset = payload.size();
    payload.resize(offset + sizeof(T));
    std::memcpy(payload.data() + offset, &value, sizeof(T));
}

void appendPayloadBytes(std::vector<uint8_t>& payload, const void* data, size_t byteSize)
{
    const size_t offset = payload.size();
    payload.resize(offset + byteSize);
    if (byteSize > 0) {
        std::memcpy(payload.data() + offset, data, byteSize);
    }
}

bool readPayloadBytes(
    std::span<const uint8_t> payload,
    size_t& offset,
    void* destination,
    size_t byteSize)
{
    if (offset > payload.size() || byteSize > payload.size() - offset) {
        return false;
    }
    if (byteSize > 0) {
        std::memcpy(destination, payload.data() + offset, byteSize);
    }
    offset += byteSize;
    return true;
}

bool loadCachedShader(
    const std::filesystem::path& path,
    uint64_t requestHash,
    std::vector<uint32_t>& outSpirv,
    std::vector<ShaderDependencySnapshot>& outDependencies)
{
    outSpirv.clear();
    outDependencies.clear();
    std::error_code fileError;
    const uint64_t fileSize = std::filesystem::file_size(path, fileError);
    if (fileError || fileSize < sizeof(ShaderCacheHeader) ||
        fileSize > kMaxShaderCacheFileSize || fileSize > std::numeric_limits<size_t>::max()) {
        return false;
    }

    std::vector<uint8_t> fileBytes(static_cast<size_t>(fileSize));
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(
            reinterpret_cast<char*>(fileBytes.data()),
            static_cast<std::streamsize>(fileBytes.size()))) {
        return false;
    }

    ShaderCacheHeader header;
    std::memcpy(&header, fileBytes.data(), sizeof(header));
    if (header.magic != kShaderCacheMagic ||
        header.version != kShaderCacheVersion ||
        header.headerSize != sizeof(ShaderCacheHeader) ||
        header.requestHash != requestHash ||
        header.reserved != 0 ||
        header.dependencyCount == 0 ||
        header.dependencyCount > kMaxShaderDependencyCount ||
        header.spirvByteSize < 5u * sizeof(uint32_t) ||
        (header.spirvByteSize % sizeof(uint32_t)) != 0 ||
        header.spirvByteSize > fileSize - sizeof(ShaderCacheHeader)) {
        return false;
    }

    const std::span<const uint8_t> payload(
        fileBytes.data() + sizeof(ShaderCacheHeader),
        fileBytes.size() - sizeof(ShaderCacheHeader));
    if (hashBytes(kFnvOffset, payload.data(), payload.size()) != header.payloadHash) {
        return false;
    }

    size_t offset = 0;
    std::vector<ShaderDependencySnapshot> dependencies;
    dependencies.reserve(header.dependencyCount);
    for (uint32_t dependencyIndex = 0;
         dependencyIndex < header.dependencyCount;
         ++dependencyIndex) {
        ShaderCacheDependencyHeader dependencyHeader;
        if (!readPayloadBytes(payload, offset, &dependencyHeader, sizeof(dependencyHeader)) ||
            dependencyHeader.reserved != 0 || dependencyHeader.pathSize == 0 ||
            dependencyHeader.pathSize > kMaxShaderDependencyPathSize ||
            dependencyHeader.pathSize > payload.size() - offset) {
            return false;
        }
        std::string dependencyPath(dependencyHeader.pathSize, '\0');
        if (!readPayloadBytes(
                payload,
                offset,
                dependencyPath.data(),
                dependencyPath.size()) ||
            dependencyPath.find('\0') != std::string::npos) {
            return false;
        }
        const std::filesystem::path dependency(dependencyPath);
        ShaderDependencyStamp stamp = shaderDependencyStamp(dependency);
        if (!stamp.exists || stamp.contentHash != dependencyHeader.contentHash) {
            return false;
        }
        dependencies.push_back(ShaderDependencySnapshot{
            .path = dependency,
            .stamp = stamp,
        });
    }

    if (header.spirvByteSize != payload.size() - offset) {
        return false;
    }
    outSpirv.resize(static_cast<size_t>(header.spirvByteSize / sizeof(uint32_t)));
    if (!readPayloadBytes(
            payload,
            offset,
            outSpirv.data(),
            static_cast<size_t>(header.spirvByteSize)) ||
        offset != payload.size() || outSpirv[0] != kSpirvMagic) {
        outSpirv.clear();
        return false;
    }
    if (!shaderDependencySnapshotsMatch(dependencies)) {
        outSpirv.clear();
        return false;
    }
    outDependencies = std::move(dependencies);
    return true;
}

std::filesystem::path temporaryShaderCachePath(const std::filesystem::path& path)
{
    std::filesystem::path temporary = path;
    temporary += ".tmp.";
    temporary += std::to_string(
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    temporary += ".";
    temporary += std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return temporary;
}

bool replaceShaderCacheFile(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination)
{
#if defined(_WIN32)
    return MoveFileExW(
        temporary.c_str(),
        destination.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
#else
    std::error_code renameError;
    std::filesystem::rename(temporary, destination, renameError);
    return !renameError;
#endif
}

std::vector<std::filesystem::path> collectShaderDependencies(
    slang::IModule& module,
    std::span<const std::filesystem::path> searchPaths)
{
    std::vector<std::filesystem::path> dependencies;
    const SlangInt32 dependencyCount = module.getDependencyFileCount();
    dependencies.reserve(std::max<SlangInt32>(dependencyCount, 0));
    for (SlangInt32 index = 0; index < dependencyCount; ++index) {
        const char* dependency = module.getDependencyFilePath(index);
        if (dependency == nullptr || dependency[0] == '\0') {
            continue;
        }
        std::filesystem::path dependencyPath(dependency);
        if (dependencyPath.is_relative()) {
            for (const std::filesystem::path& searchPath : searchPaths) {
                const std::filesystem::path searchRelative = searchPath / dependencyPath;
                std::error_code existsError;
                if (std::filesystem::exists(searchRelative, existsError) && !existsError) {
                    dependencyPath = searchRelative;
                    break;
                }
            }
        }
        dependencies.push_back(normalizedAbsolutePath(dependencyPath));
    }
    std::sort(dependencies.begin(), dependencies.end());
    dependencies.erase(
        std::unique(dependencies.begin(), dependencies.end()),
        dependencies.end());
    return dependencies;
}

bool saveCachedShader(
    const std::filesystem::path& path,
    uint64_t requestHash,
    std::span<const ShaderDependencySnapshot> dependencies,
    std::span<const uint32_t> spirv)
{
    if (dependencies.empty() || dependencies.size() > kMaxShaderDependencyCount ||
        spirv.size() < 5 || spirv[0] != kSpirvMagic ||
        spirv.size_bytes() > kMaxShaderCacheFileSize) {
        return false;
    }

    std::vector<uint8_t> payload;
    for (const ShaderDependencySnapshot& dependency : dependencies) {
        const std::string dependencyPath = dependency.path.generic_string();
        if (dependencyPath.empty() || dependencyPath.size() > kMaxShaderDependencyPathSize ||
            !dependency.stamp.exists) {
            return false;
        }
        const ShaderCacheDependencyHeader dependencyHeader{
            .pathSize = static_cast<uint32_t>(dependencyPath.size()),
            .contentHash = dependency.stamp.contentHash,
        };
        appendPayloadValue(payload, dependencyHeader);
        appendPayloadBytes(payload, dependencyPath.data(), dependencyPath.size());
    }
    appendPayloadBytes(payload, spirv.data(), spirv.size_bytes());
    if (payload.size() + sizeof(ShaderCacheHeader) > kMaxShaderCacheFileSize) {
        return false;
    }

    const ShaderCacheHeader header{
        .magic = kShaderCacheMagic,
        .version = kShaderCacheVersion,
        .headerSize = sizeof(ShaderCacheHeader),
        .requestHash = requestHash,
        .dependencyCount = static_cast<uint32_t>(dependencies.size()),
        .spirvByteSize = spirv.size_bytes(),
        .payloadHash = hashBytes(kFnvOffset, payload.data(), payload.size()),
    };

    std::error_code directoryError;
    std::filesystem::create_directories(path.parent_path(), directoryError);
    if (directoryError) {
        return false;
    }

    const std::filesystem::path temporary = temporaryShaderCachePath(path);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        const bool wrote = stream &&
            static_cast<bool>(stream.write(
                reinterpret_cast<const char*>(&header),
                sizeof(header))) &&
            static_cast<bool>(stream.write(
                reinterpret_cast<const char*>(payload.data()),
                static_cast<std::streamsize>(payload.size())));
        stream.flush();
        if (!wrote || !stream) {
            stream.close();
            std::error_code removeError;
            std::filesystem::remove(temporary, removeError);
            return false;
        }
    }
    if (!replaceShaderCacheFile(temporary, path)) {
        std::error_code removeError;
        std::filesystem::remove(temporary, removeError);
        return false;
    }
    return true;
}

void appendDiagnostics(slang::IBlob* diagnostics, std::string& outDiagnostics)
{
    if (diagnostics == nullptr || diagnostics->getBufferPointer() == nullptr || diagnostics->getBufferSize() == 0) {
        return;
    }

    const char* text = static_cast<const char*>(diagnostics->getBufferPointer());
    size_t size = diagnostics->getBufferSize();
    if (size > 0 && text[size - 1] == '\0') {
        --size;
    }

    outDiagnostics.append(text, size);
}

} // namespace

void setSlangShaderDebugMode(SlangShaderDebugMode mode) noexcept
{
    gSlangShaderDebugMode.store(mode, std::memory_order_relaxed);
}

SlangShaderDebugMode slangShaderDebugMode() noexcept
{
    return gSlangShaderDebugMode.load(std::memory_order_relaxed);
}

std::vector<std::string> pollSlangShaderChanges(
    uint32_t debounceMilliseconds,
    uint32_t retryMilliseconds)
{
    const auto now = std::chrono::steady_clock::now();
    const auto debounce = std::chrono::milliseconds(debounceMilliseconds);
    const auto retry = std::chrono::milliseconds(retryMilliseconds);
    std::vector<std::string> changedFiles;

    std::scoped_lock lock(gShaderDependencyMutex);
    for (auto& [path, tracked] : gShaderDependencies) {
        const ShaderDependencyStamp currentStamp = currentTrackedDependencyStamp(
            tracked,
            now,
            debounceMilliseconds == 0 || tracked.pendingStamp.has_value());
        if (!tracked.dirty && currentStamp == tracked.acceptedStamp) {
            tracked.pendingStamp.reset();
            continue;
        }

        if (tracked.dirty) {
            if (currentStamp == tracked.acceptedStamp) {
                tracked.dirty = false;
                tracked.pendingStamp.reset();
                continue;
            }
            if (currentStamp == tracked.reportedStamp) {
                tracked.pendingStamp.reset();
                if (now - tracked.lastReportedAt >= retry) {
                    tracked.lastReportedAt = now;
                    changedFiles.push_back(path);
                }
                continue;
            }
        }

        if (!tracked.pendingStamp.has_value() || *tracked.pendingStamp != currentStamp) {
            tracked.pendingStamp = currentStamp;
            tracked.pendingSince = now;
        }
        if (now - tracked.pendingSince < debounce) {
            continue;
        }
        tracked.dirty = true;
        tracked.reportedStamp = currentStamp;
        tracked.lastReportedAt = now;
        tracked.pendingStamp.reset();
        changedFiles.push_back(path);
    }
    std::sort(changedFiles.begin(), changedFiles.end());
    return changedFiles;
}

void acknowledgeSlangShaderChanges()
{
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock(gShaderDependencyMutex);
    for (auto& [path, tracked] : gShaderDependencies) {
        (void)path;
        if (!tracked.dirty) {
            continue;
        }

        const ShaderDependencyStamp currentStamp = currentTrackedDependencyStamp(
            tracked,
            now,
            true);
        tracked.acceptedStamp = tracked.reportedStamp;
        tracked.dirty = false;
        tracked.pendingStamp.reset();
        if (currentStamp != tracked.acceptedStamp) {
            tracked.pendingStamp = currentStamp;
            tracked.pendingSince = now;
        }
    }
}

void resetSlangShaderHotReloadTracking()
{
    std::scoped_lock lock(gShaderDependencyMutex);
    gShaderDependencies.clear();
}

Result compileSlangShaderToSpirv(const SlangShaderDesc& desc, ShaderCompileResult& outResult)
{
    return compileSlangShaderToSpirv(desc, SlangShaderCacheOptions{}, outResult);
}

Result compileSlangShaderToSpirv(
    const SlangShaderDesc& desc,
    const SlangShaderCacheOptions& cacheOptions,
    ShaderCompileResult& outResult)
{
    outResult = {};
    if (cacheOptions.outCacheHit != nullptr) {
        *cacheOptions.outCacheHit = false;
    }

    if (desc.moduleName == nullptr || desc.entryPointName == nullptr || desc.searchPath == nullptr ||
        (desc.additionalSearchPathCount > 0 && desc.additionalSearchPaths == nullptr)) {
        return makeError(Error::InvalidArgument);
    }
    for (uint32_t index = 0; index < desc.additionalSearchPathCount; ++index) {
        if (desc.additionalSearchPaths[index] == nullptr ||
            desc.additionalSearchPaths[index][0] == '\0') {
            return makeError(Error::InvalidArgument);
        }
    }

    const SlangShaderDebugMode debugMode = slangShaderDebugMode();
    const uint64_t requestHash = shaderRequestHash(desc, debugMode);
    const std::filesystem::path cachePath = shaderCachePath(cacheOptions, requestHash);
    std::vector<ShaderDependencySnapshot> cachedDependencies;
    if (cacheOptions.enableDiskCache &&
        loadCachedShader(cachePath, requestHash, outResult.spirv, cachedDependencies)) {
        if (cacheOptions.outCacheHit != nullptr) {
            *cacheOptions.outCacheHit = true;
        }
        publishShaderDependencies(cachedDependencies, outResult);
        spdlog::info(
            "[ShaderCache] Loaded {}.{} from '{}'",
            desc.moduleName,
            desc.entryPointName,
            cachePath.string());
        return {};
    }

    Slang::ComPtr<slang::IGlobalSession> globalSession;
    if (SLANG_FAILED(slang::createGlobalSession(globalSession.writeRef())) || globalSession == nullptr) {
        return makeError(Error::Failure);
    }

    slang::TargetDesc targetDesc{};
    targetDesc.format = SLANG_SPIRV;
    const char* profileName = desc.profileName != nullptr ? desc.profileName : kDefaultSlangProfileName;
    targetDesc.profile = globalSession->findProfile(profileName);
    if (targetDesc.profile == SLANG_PROFILE_UNKNOWN) {
        outResult.diagnostics = "Unknown Slang target profile: ";
        outResult.diagnostics += profileName;
        outResult.diagnostics += '\n';
        return makeError(Error::InvalidArgument);
    }

    std::vector<std::filesystem::path> normalizedSearchPaths;
    normalizedSearchPaths.reserve(1u + desc.additionalSearchPathCount);
    normalizedSearchPaths.push_back(normalizedAbsolutePath(desc.searchPath));
    for (uint32_t index = 0; index < desc.additionalSearchPathCount; ++index) {
        normalizedSearchPaths.push_back(
            normalizedAbsolutePath(desc.additionalSearchPaths[index]));
    }
    std::vector<std::string> searchPathStorage;
    searchPathStorage.reserve(normalizedSearchPaths.size());
    for (const std::filesystem::path& searchPath : normalizedSearchPaths) {
        searchPathStorage.push_back(searchPath.string());
    }
    std::vector<const char*> searchPaths;
    searchPaths.reserve(searchPathStorage.size());
    for (const std::string& searchPath : searchPathStorage) {
        searchPaths.push_back(searchPath.c_str());
    }
    std::vector<slang::CompilerOptionEntry> compilerOptions;
    compilerOptions.reserve(desc.capabilityCount + desc.macroDefineCount + 3u);
    if (debugMode != SlangShaderDebugMode::Disabled) {
        compilerOptions.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::EmitSpirvDirectly,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::Int,
                .intValue0 = 1,
            },
        });
        compilerOptions.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::DebugInformation,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::Int,
                .intValue0 = static_cast<int32_t>(SLANG_DEBUG_INFO_LEVEL_STANDARD),
            },
        });
        if (debugMode == SlangShaderDebugMode::ShaderDebug) {
            compilerOptions.push_back(slang::CompilerOptionEntry{
                .name = slang::CompilerOptionName::Optimization,
                .value = slang::CompilerOptionValue{
                    .kind = slang::CompilerOptionValueKind::Int,
                    .intValue0 = static_cast<int32_t>(SLANG_OPTIMIZATION_LEVEL_NONE),
                },
            });
        }
    }
    for (uint32_t capabilityIndex = 0;
         desc.capabilities != nullptr && capabilityIndex < desc.capabilityCount;
         ++capabilityIndex) {
        const char* capability = desc.capabilities[capabilityIndex];
        if (capability == nullptr || capability[0] == '\0') {
            continue;
        }
        compilerOptions.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::Capability,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::String,
                .stringValue0 = capability,
            },
        });
    }
    for (uint32_t macroIndex = 0;
         desc.macroDefines != nullptr && macroIndex < desc.macroDefineCount;
         ++macroIndex) {
        const SlangMacroDefine& macro = desc.macroDefines[macroIndex];
        if (macro.name == nullptr || macro.name[0] == '\0') {
            continue;
        }
        compilerOptions.push_back(slang::CompilerOptionEntry{
            .name = slang::CompilerOptionName::MacroDefine,
            .value = slang::CompilerOptionValue{
                .kind = slang::CompilerOptionValueKind::String,
                .stringValue0 = macro.name,
                .stringValue1 = macro.value != nullptr ? macro.value : "1",
            },
        });
    }

    slang::SessionDesc sessionDesc{};
    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;
    sessionDesc.searchPaths = searchPaths.data();
    sessionDesc.searchPathCount = static_cast<SlangInt>(searchPaths.size());
    sessionDesc.defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    sessionDesc.compilerOptionEntries = compilerOptions.empty() ? nullptr : compilerOptions.data();
    sessionDesc.compilerOptionEntryCount = compilerOptions.size();

    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef())) || session == nullptr) {
        return makeError(Error::Failure);
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> module(session->loadModule(desc.moduleName, diagnostics.writeRef()));
    appendDiagnostics(diagnostics, outResult.diagnostics);
    if (module == nullptr) {
        return makeError(Error::Failure);
    }
    const std::vector<std::filesystem::path> dependencies = collectShaderDependencies(
        *module,
        normalizedSearchPaths);
    std::vector<ShaderDependencySnapshot> dependencySnapshots;
    if (!snapshotShaderDependencies(dependencies, dependencySnapshots)) {
        outResult.diagnostics += "Could not snapshot every Slang source dependency.\n";
        return makeError(Error::Failure);
    }
    // Register the discovered files before later compilation stages. This lets
    // a fix to an included file trigger another attempt even when linking or
    // code generation fails.
    registerShaderDependencies(dependencySnapshots);

    diagnostics.setNull();
    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    if (SLANG_FAILED(module->findEntryPointByName(desc.entryPointName, entryPoint.writeRef())) || entryPoint == nullptr) {
        outResult.diagnostics += "Slang entry point not found: ";
        outResult.diagnostics += desc.entryPointName;
        outResult.diagnostics += '\n';
        return makeError(Error::Failure);
    }

    slang::IComponentType* componentTypes[] = {module, entryPoint};
    Slang::ComPtr<slang::IComponentType> program;
    if (SLANG_FAILED(session->createCompositeComponentType(componentTypes, 2, program.writeRef(), diagnostics.writeRef()))
        || program == nullptr) {
        appendDiagnostics(diagnostics, outResult.diagnostics);
        return makeError(Error::Failure);
    }
    appendDiagnostics(diagnostics, outResult.diagnostics);

    diagnostics.setNull();
    Slang::ComPtr<slang::IComponentType> linkedProgram;
    if (SLANG_FAILED(program->link(linkedProgram.writeRef(), diagnostics.writeRef())) || linkedProgram == nullptr) {
        appendDiagnostics(diagnostics, outResult.diagnostics);
        return makeError(Error::Failure);
    }
    appendDiagnostics(diagnostics, outResult.diagnostics);

    diagnostics.setNull();
    Slang::ComPtr<slang::IBlob> shaderCode;
    if (SLANG_FAILED(linkedProgram->getEntryPointCode(0, 0, shaderCode.writeRef(), diagnostics.writeRef()))
        || shaderCode == nullptr) {
        appendDiagnostics(diagnostics, outResult.diagnostics);
        return makeError(Error::Failure);
    }
    appendDiagnostics(diagnostics, outResult.diagnostics);

    const size_t byteSize = shaderCode->getBufferSize();
    if (byteSize == 0 || (byteSize % sizeof(uint32_t)) != 0) {
        outResult.diagnostics += "Slang produced invalid SPIR-V bytecode size.\n";
        return makeError(Error::Failure);
    }

    outResult.spirv.resize(byteSize / sizeof(uint32_t));
    std::memcpy(outResult.spirv.data(), shaderCode->getBufferPointer(), byteSize);
    if (!shaderDependencySnapshotsMatch(dependencySnapshots)) {
        outResult.spirv.clear();
        outResult.diagnostics +=
            "A Slang source dependency changed during compilation; retry after the save completes.\n";
        return makeError(Error::Failure);
    }
    publishShaderDependencies(dependencySnapshots, outResult);
    if (cacheOptions.enableDiskCache) {
        if (!saveCachedShader(cachePath, requestHash, dependencySnapshots, outResult.spirv)) {
            spdlog::debug(
                "[ShaderCache] Could not persist {}.{} to '{}'",
                desc.moduleName,
                desc.entryPointName,
                cachePath.string());
        }
    }
    return {};
}

} // namespace metallic::render

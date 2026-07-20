#include "Runtime/Render/GAPI/PipelineCacheFile.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <thread>
#include <type_traits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace metallic::render::detail {
namespace {

constexpr std::array<char, 8> kPipelineCacheMagic{'M', 'T', 'L', 'P', 'S', 'O', '0', '1'};
constexpr uint32_t kPipelineCacheFileVersion = 1;
constexpr uint64_t kMaxPsoHashCount = 1'048'576;
constexpr uint64_t kMaxBackendDataSize = 1ull << 32u;
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

struct PipelineCacheFileHeader {
    std::array<char, 8> magic{};
    uint32_t version = 0;
    uint32_t backendTag = 0;
    uint32_t headerSize = 0;
    uint32_t reserved = 0;
    uint64_t psoHashCount = 0;
    uint64_t backendDataSize = 0;
    uint64_t payloadHash = 0;
    std::array<uint8_t, 32> compatibilityKey{};
};

static_assert(std::is_trivially_copyable_v<PipelineCacheFileHeader>);
static_assert(sizeof(PipelineCacheFileHeader) == 80);

uint64_t hashBytes(uint64_t hash, const void* data, size_t byteSize)
{
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < byteSize; ++index) {
        hash ^= bytes[index];
        hash *= kFnvPrime;
    }
    return hash;
}

uint64_t payloadHash(
    std::span<const uint64_t> psoHashes,
    std::span<const uint8_t> backendData)
{
    uint64_t hash = kFnvOffset;
    if (!psoHashes.empty()) {
        hash = hashBytes(hash, psoHashes.data(), psoHashes.size_bytes());
    }
    if (!backendData.empty()) {
        hash = hashBytes(hash, backendData.data(), backendData.size_bytes());
    }
    return hash;
}

std::filesystem::path temporaryPathFor(const std::filesystem::path& path)
{
    std::filesystem::path temporary = path;
    temporary += ".tmp.";
    temporary += std::to_string(
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    temporary += ".";
    temporary += std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return temporary;
}

bool replaceFile(
    const std::filesystem::path& temporary,
    const std::filesystem::path& destination,
    std::string& reason)
{
#if defined(_WIN32)
    if (MoveFileExW(
            temporary.c_str(),
            destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) == FALSE) {
        reason = "atomic .pso replacement failed with Win32 error " +
            std::to_string(GetLastError());
        return false;
    }
#else
    std::error_code renameError;
    std::filesystem::rename(temporary, destination, renameError);
    if (renameError) {
        reason = "atomic .pso replacement failed: " + renameError.message();
        return false;
    }
#endif
    return true;
}

} // namespace

bool isPipelineCacheFilePath(const std::filesystem::path& path)
{
    return !path.empty() && path.extension() == ".pso";
}

PipelineCacheFileLoadStatus loadPipelineCacheFile(
    const std::filesystem::path& path,
    const PipelineCacheFileIdentity& identity,
    PipelineCacheFileData& outData,
    std::string& reason)
{
    outData = {};
    reason.clear();
    if (!isPipelineCacheFilePath(path)) {
        reason = "pipeline cache path must use the .pso extension";
        return PipelineCacheFileLoadStatus::Invalid;
    }

    std::error_code fileError;
    const uint64_t fileSize = std::filesystem::file_size(path, fileError);
    if (fileError) {
        std::error_code existsError;
        if (!std::filesystem::exists(path, existsError) && !existsError) {
            return PipelineCacheFileLoadStatus::NotFound;
        }
        reason = "cannot query .pso file size: " + fileError.message();
        return PipelineCacheFileLoadStatus::Invalid;
    }
    if (fileSize < sizeof(PipelineCacheFileHeader)) {
        reason = ".pso file is smaller than its header";
        return PipelineCacheFileLoadStatus::Invalid;
    }

    std::ifstream stream(path, std::ios::binary);
    PipelineCacheFileHeader header;
    if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        reason = "cannot read .pso header";
        return PipelineCacheFileLoadStatus::Invalid;
    }
    if (header.magic != kPipelineCacheMagic ||
        header.version != kPipelineCacheFileVersion ||
        header.headerSize != sizeof(PipelineCacheFileHeader) ||
        header.reserved != 0) {
        reason = ".pso header magic or version is invalid";
        return PipelineCacheFileLoadStatus::Invalid;
    }
    if (header.backendTag != identity.backendTag ||
        header.compatibilityKey != identity.compatibilityKey) {
        reason = ".pso backend or device compatibility key changed";
        return PipelineCacheFileLoadStatus::Incompatible;
    }
    if (header.psoHashCount > kMaxPsoHashCount ||
        header.backendDataSize > kMaxBackendDataSize ||
        header.psoHashCount > std::numeric_limits<uint64_t>::max() / sizeof(uint64_t)) {
        reason = ".pso payload sizes exceed supported limits";
        return PipelineCacheFileLoadStatus::Invalid;
    }

    const uint64_t hashByteSize = header.psoHashCount * sizeof(uint64_t);
    const uint64_t expectedFileSize = sizeof(PipelineCacheFileHeader) +
        hashByteSize + header.backendDataSize;
    if (expectedFileSize != fileSize ||
        header.psoHashCount > std::numeric_limits<size_t>::max() ||
        header.backendDataSize > std::numeric_limits<size_t>::max()) {
        reason = ".pso payload sizes do not match the file";
        return PipelineCacheFileLoadStatus::Invalid;
    }

    outData.psoHashes.resize(static_cast<size_t>(header.psoHashCount));
    outData.backendData.resize(static_cast<size_t>(header.backendDataSize));
    if ((!outData.psoHashes.empty() &&
         !stream.read(
             reinterpret_cast<char*>(outData.psoHashes.data()),
             static_cast<std::streamsize>(hashByteSize))) ||
        (!outData.backendData.empty() &&
         !stream.read(
             reinterpret_cast<char*>(outData.backendData.data()),
             static_cast<std::streamsize>(outData.backendData.size())))) {
        outData = {};
        reason = "cannot read .pso payload";
        return PipelineCacheFileLoadStatus::Invalid;
    }
    if (!std::is_sorted(outData.psoHashes.begin(), outData.psoHashes.end()) ||
        std::adjacent_find(outData.psoHashes.begin(), outData.psoHashes.end()) !=
            outData.psoHashes.end() ||
        payloadHash(outData.psoHashes, outData.backendData) != header.payloadHash) {
        outData = {};
        reason = ".pso payload hash or PSO hash table is invalid";
        return PipelineCacheFileLoadStatus::Invalid;
    }
    return PipelineCacheFileLoadStatus::Loaded;
}

bool savePipelineCacheFile(
    const std::filesystem::path& path,
    const PipelineCacheFileIdentity& identity,
    std::span<const uint64_t> psoHashes,
    std::span<const uint8_t> backendData,
    std::string& reason)
{
    reason.clear();
    if (!isPipelineCacheFilePath(path)) {
        reason = "pipeline cache path must use the .pso extension";
        return false;
    }
    if (psoHashes.size() > kMaxPsoHashCount || backendData.size() > kMaxBackendDataSize) {
        reason = ".pso payload exceeds supported limits";
        return false;
    }

    std::vector<uint64_t> sortedHashes(psoHashes.begin(), psoHashes.end());
    std::sort(sortedHashes.begin(), sortedHashes.end());
    sortedHashes.erase(
        std::unique(sortedHashes.begin(), sortedHashes.end()),
        sortedHashes.end());

    const PipelineCacheFileHeader header{
        .magic = kPipelineCacheMagic,
        .version = kPipelineCacheFileVersion,
        .backendTag = identity.backendTag,
        .headerSize = sizeof(PipelineCacheFileHeader),
        .psoHashCount = sortedHashes.size(),
        .backendDataSize = backendData.size(),
        .payloadHash = payloadHash(sortedHashes, backendData),
        .compatibilityKey = identity.compatibilityKey,
    };

    std::error_code directoryError;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), directoryError);
    }
    if (directoryError) {
        reason = "cannot create .pso directory: " + directoryError.message();
        return false;
    }

    const std::filesystem::path temporary = temporaryPathFor(path);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        const bool wrote = stream &&
            static_cast<bool>(stream.write(reinterpret_cast<const char*>(&header), sizeof(header))) &&
            (sortedHashes.empty() ||
             static_cast<bool>(stream.write(
                 reinterpret_cast<const char*>(sortedHashes.data()),
                 static_cast<std::streamsize>(sortedHashes.size() * sizeof(uint64_t))))) &&
            (backendData.empty() ||
             static_cast<bool>(stream.write(
                 reinterpret_cast<const char*>(backendData.data()),
                 static_cast<std::streamsize>(backendData.size()))));
        if (!wrote) {
            reason = "cannot write temporary .pso file";
            stream.close();
            std::error_code removeError;
            std::filesystem::remove(temporary, removeError);
            return false;
        }
        stream.flush();
        if (!stream) {
            reason = "cannot flush temporary .pso file";
            stream.close();
            std::error_code removeError;
            std::filesystem::remove(temporary, removeError);
            return false;
        }
    }

    if (!replaceFile(temporary, path, reason)) {
        std::error_code removeError;
        std::filesystem::remove(temporary, removeError);
        return false;
    }
    return true;
}

} // namespace metallic::render::detail

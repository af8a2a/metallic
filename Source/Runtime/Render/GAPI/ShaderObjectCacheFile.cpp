#include "Runtime/Render/GAPI/ShaderObjectCacheFile.h"
#include "Runtime/Render/GAPI/Hash.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

namespace metallic::render::detail {
namespace {

constexpr std::array<char, 8> kShaderObjectCacheMagic{'M', 'T', 'L', 'S', 'H', 'B', '0', '1'};
constexpr uint32_t kShaderObjectCacheFileVersion = 1;
constexpr uint64_t kMaxShaderBinarySize = 64ull << 20u;

struct ShaderObjectCacheFileHeader {
    std::array<char, 8> magic{};
    uint32_t version = 0;
    uint32_t headerSize = 0;
    uint32_t binaryVersion = 0;
    uint32_t reserved = 0;
    uint64_t programHash = 0;
    std::array<uint8_t, 16> binaryUUID{};
    std::array<uint64_t, 2> binarySizes{};
    uint64_t payloadHash = 0;
};

static_assert(std::is_trivially_copyable_v<ShaderObjectCacheFileHeader>);
static_assert(sizeof(ShaderObjectCacheFileHeader) == 72);

uint64_t payloadHash(const ShaderObjectCacheFileData& data)
{
    uint64_t hash = kFnvOffset;
    for (const auto& binary : data.binaries) {
        const uint64_t byteSize = binary.size();
        hash = hashBytes(hash, &byteSize, sizeof(byteSize));
        hash = hashBytes(hash, binary.data(), binary.size());
    }
    return hash;
}

std::filesystem::path temporaryPathFor(const std::filesystem::path& path)
{
    static std::atomic<uint64_t> sequence{0};
#if defined(_WIN32)
    const uint64_t processID = GetCurrentProcessId();
#else
    const uint64_t processID = static_cast<uint64_t>(getpid());
#endif
    std::filesystem::path temporary = path;
    temporary += ".tmp." + std::to_string(processID) + ".";
    temporary += std::to_string(
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
    temporary += "." + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    temporary += "." + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
    return temporary;
}

void removeTemporaryFile(const std::filesystem::path& path)
{
    std::error_code removeError;
    std::filesystem::remove(path, removeError);
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
        reason = "atomic .shaderbin replacement failed with Win32 error " +
            std::to_string(GetLastError());
        return false;
    }
#else
    std::error_code renameError;
    std::filesystem::rename(temporary, destination, renameError);
    if (renameError) {
        reason = "atomic .shaderbin replacement failed: " + renameError.message();
        return false;
    }
#endif
    return true;
}

bool isShaderObjectCacheFilePath(const std::filesystem::path& path)
{
    return !path.empty() && path.extension() == ".shaderbin";
}

} // namespace

ShaderObjectCacheFileLoadStatus loadShaderObjectCacheFile(
    const std::filesystem::path& path,
    const ShaderObjectCacheFileIdentity& identity,
    ShaderObjectCacheFileData& outData,
    std::string& reason)
{
    outData = {};
    reason.clear();
    if (!isShaderObjectCacheFilePath(path)) {
        reason = "shader object cache path must use the .shaderbin extension";
        return ShaderObjectCacheFileLoadStatus::Invalid;
    }

    // Query the opened file so atomic replacement cannot mix an old size with a new payload.
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        std::error_code existsError;
        if (!std::filesystem::exists(path, existsError) && !existsError) {
            return ShaderObjectCacheFileLoadStatus::NotFound;
        }
        reason = "cannot open .shaderbin file";
        return ShaderObjectCacheFileLoadStatus::Invalid;
    }
    const auto end = stream.tellg();
    if (end < std::streampos(0) ||
        static_cast<uint64_t>(end) < sizeof(ShaderObjectCacheFileHeader)) {
        reason = ".shaderbin file is smaller than its header";
        return ShaderObjectCacheFileLoadStatus::Invalid;
    }
    const uint64_t fileSize = static_cast<uint64_t>(end);
    stream.seekg(0, std::ios::beg);
    ShaderObjectCacheFileHeader header;
    if (!stream.read(reinterpret_cast<char*>(&header), sizeof(header))) {
        reason = "cannot read .shaderbin header";
        return ShaderObjectCacheFileLoadStatus::Invalid;
    }
    if (header.magic != kShaderObjectCacheMagic ||
        header.version != kShaderObjectCacheFileVersion ||
        header.headerSize != sizeof(ShaderObjectCacheFileHeader) ||
        header.reserved != 0) {
        reason = ".shaderbin header magic or version is invalid";
        return ShaderObjectCacheFileLoadStatus::Invalid;
    }
    for (const uint64_t size : header.binarySizes) {
        if (size == 0 || size > kMaxShaderBinarySize ||
            size > std::numeric_limits<size_t>::max() ||
            size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            reason = ".shaderbin stage sizes are empty or exceed supported limits";
            return ShaderObjectCacheFileLoadStatus::Invalid;
        }
    }
    const uint64_t expectedFileSize = sizeof(ShaderObjectCacheFileHeader) +
        header.binarySizes[0] + header.binarySizes[1];
    if (expectedFileSize != fileSize) {
        reason = ".shaderbin payload sizes do not match the file";
        return ShaderObjectCacheFileLoadStatus::Invalid;
    }
    if (header.binaryUUID != identity.binaryUUID ||
        header.binaryVersion > identity.binaryVersion ||
        header.programHash != identity.programHash) {
        reason = ".shaderbin binary compatibility or program identity changed";
        return ShaderObjectCacheFileLoadStatus::Incompatible;
    }

    ShaderObjectCacheFileData loaded;
    for (size_t stage = 0; stage < loaded.binaries.size(); ++stage) {
        auto& binary = loaded.binaries[stage];
        binary.resize(static_cast<size_t>(header.binarySizes[stage]));
        if (!stream.read(
                reinterpret_cast<char*>(binary.data()),
                static_cast<std::streamsize>(binary.size()))) {
            reason = "cannot read .shaderbin payload";
            return ShaderObjectCacheFileLoadStatus::Invalid;
        }
    }
    if (payloadHash(loaded) != header.payloadHash || stream.peek() != std::char_traits<char>::eof()) {
        reason = ".shaderbin payload checksum or length is invalid";
        return ShaderObjectCacheFileLoadStatus::Invalid;
    }
    outData = std::move(loaded);
    return ShaderObjectCacheFileLoadStatus::Loaded;
}

bool saveShaderObjectCacheFile(
    const std::filesystem::path& path,
    const ShaderObjectCacheFileIdentity& identity,
    const ShaderObjectCacheFileData& data,
    std::string& reason)
{
    reason.clear();
    if (!isShaderObjectCacheFilePath(path)) {
        reason = "shader object cache path must use the .shaderbin extension";
        return false;
    }
    for (const auto& binary : data.binaries) {
        if (binary.empty() || binary.size() > kMaxShaderBinarySize ||
            binary.size() > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            reason = ".shaderbin stage sizes are empty or exceed supported limits";
            return false;
        }
    }
    const ShaderObjectCacheFileHeader header{
        .magic = kShaderObjectCacheMagic,
        .version = kShaderObjectCacheFileVersion,
        .headerSize = sizeof(ShaderObjectCacheFileHeader),
        .binaryVersion = identity.binaryVersion,
        .programHash = identity.programHash,
        .binaryUUID = identity.binaryUUID,
        .binarySizes = {data.binaries[0].size(), data.binaries[1].size()},
        .payloadHash = payloadHash(data),
    };

    std::error_code directoryError;
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path(), directoryError);
    }
    if (directoryError) {
        reason = "cannot create .shaderbin directory: " + directoryError.message();
        return false;
    }

    const std::filesystem::path temporary = temporaryPathFor(path);
    {
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        bool wrote = stream &&
            static_cast<bool>(stream.write(reinterpret_cast<const char*>(&header), sizeof(header)));
        for (const auto& binary : data.binaries) {
            wrote = wrote && static_cast<bool>(stream.write(
                reinterpret_cast<const char*>(binary.data()),
                static_cast<std::streamsize>(binary.size())));
        }
        if (!wrote) {
            reason = "cannot write temporary .shaderbin file";
            stream.close();
            removeTemporaryFile(temporary);
            return false;
        }
        stream.flush();
        stream.close();
        if (!stream) {
            reason = "cannot flush or close temporary .shaderbin file";
            removeTemporaryFile(temporary);
            return false;
        }
    }
    if (!replaceFile(temporary, path, reason)) {
        removeTemporaryFile(temporary);
        return false;
    }
    return true;
}

} // namespace metallic::render::detail

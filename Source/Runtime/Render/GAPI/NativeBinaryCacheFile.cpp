#include "Runtime/Render/GAPI/NativeBinaryCacheFile.h"
#include "Runtime/Render/GAPI/Hash.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <type_traits>

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

constexpr std::array<uint8_t, 8> kMagic{'M', 'T', 'L', 'N', 'B', 'I', 'N', '1'};
constexpr uint64_t kMaxFileSize = 256ull << 20u;
constexpr uint32_t kMaxEntries = 64;
constexpr uint32_t kMaxKeySize = 256;

template<class T>
void append(std::vector<uint8_t>& bytes, T value)
{
    static_assert(std::is_unsigned_v<T>);
    for (size_t i = 0; i < sizeof(T); ++i) { bytes.push_back(static_cast<uint8_t>(value >> (8 * i))); }
}

template<class T>
T read(std::span<const uint8_t>& bytes)
{
    static_assert(std::is_unsigned_v<T>);
    if (bytes.size() < sizeof(T)) { throw std::runtime_error("truncated native binary header"); }
    T value = 0;
    for (size_t i = 0; i < sizeof(T); ++i) { value |= static_cast<T>(bytes[i]) << (8 * i); }
    bytes = bytes.subspan(sizeof(T));
    return value;
}

std::vector<uint8_t> readBytes(std::span<const uint8_t>& bytes, uint64_t size)
{
    if (size > bytes.size()) { throw std::runtime_error("truncated native binary payload"); }
    std::vector<uint8_t> result(bytes.begin(), bytes.begin() + static_cast<size_t>(size));
    bytes = bytes.subspan(static_cast<size_t>(size));
    return result;
}

void validate(const NativeBinaryCacheData& data)
{
    if (data.compatibilityKey.empty() || data.compatibilityKey.size() > kMaxKeySize ||
        data.entries.empty() || data.entries.size() > kMaxEntries) {
        throw std::runtime_error("invalid native binary identity or entry count");
    }
    uint64_t total = 40 + data.compatibilityKey.size();
    for (const auto& entry : data.entries) {
        if (entry.key.size() > kMaxKeySize || entry.data.empty() || entry.data.size() > kMaxFileSize) {
            throw std::runtime_error("invalid native binary key or data size");
        }
        total += 12 + entry.key.size() + entry.data.size();
        if (total > kMaxFileSize) { throw std::runtime_error("native binary file exceeds its size limit"); }
    }
}

} // namespace

NativeBinaryCacheLoadStatus loadNativeBinaryCacheFile(const std::filesystem::path& path,
    const NativeBinaryCacheData& identity, NativeBinaryCacheData& data, std::string& reason)
{
    data = {};
    reason.clear();
    try {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            std::error_code error;
            if (!std::filesystem::exists(path, error) && !error) { return NativeBinaryCacheLoadStatus::NotFound; }
            throw std::runtime_error("cannot open native binary file");
        }
        const auto size = input.tellg();
        if (size < std::streampos(40) || static_cast<uint64_t>(size) > kMaxFileSize) {
            throw std::runtime_error("invalid native binary file size");
        }
        std::vector<uint8_t> storage(static_cast<size_t>(size));
        input.seekg(0);
        if (!input.read(reinterpret_cast<char*>(storage.data()), static_cast<std::streamsize>(storage.size()))) {
            throw std::runtime_error("cannot read native binary file");
        }
        auto checksumBytes = std::span<const uint8_t>(storage).last(8);
        const uint64_t checksum = read<uint64_t>(checksumBytes);
        auto bytes = std::span<const uint8_t>(storage).first(storage.size() - 8);
        if (hashBytes(kFnvOffset, bytes.data(), bytes.size()) != checksum ||
            !std::equal(kMagic.begin(), kMagic.end(), bytes.begin())) {
            throw std::runtime_error("native binary checksum or magic mismatch");
        }
        bytes = bytes.subspan(kMagic.size());
        if (read<uint32_t>(bytes) != 1) { throw std::runtime_error("unsupported native binary version"); }
        NativeBinaryCacheData loaded;
        loaded.kind = read<uint32_t>(bytes);
        loaded.programHash = read<uint64_t>(bytes);
        const uint32_t keySize = read<uint32_t>(bytes);
        const uint32_t count = read<uint32_t>(bytes);
        if (keySize == 0 || keySize > kMaxKeySize || count == 0 || count > kMaxEntries) {
            throw std::runtime_error("invalid native binary compatibility key or count");
        }
        loaded.compatibilityKey = readBytes(bytes, keySize);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t entryKeySize = read<uint32_t>(bytes);
            const uint64_t entrySize = read<uint64_t>(bytes);
            if (entryKeySize > kMaxKeySize || entrySize == 0 || entrySize > kMaxFileSize) {
                throw std::runtime_error("invalid native binary entry size");
            }
            NativeBinaryCacheEntry entry;
            entry.key = readBytes(bytes, entryKeySize);
            entry.data = readBytes(bytes, entrySize);
            loaded.entries.push_back(std::move(entry));
        }
        if (!bytes.empty()) { throw std::runtime_error("native binary file has trailing data"); }
        if (loaded.kind != identity.kind || loaded.programHash != identity.programHash ||
            loaded.compatibilityKey != identity.compatibilityKey) {
            reason = "native binary compatibility key or program identity changed";
            return NativeBinaryCacheLoadStatus::Incompatible;
        }
        data = std::move(loaded);
        return NativeBinaryCacheLoadStatus::Loaded;
    } catch (const std::exception& error) {
        reason = error.what();
        return NativeBinaryCacheLoadStatus::Invalid;
    }
}

bool saveNativeBinaryCacheFile(const std::filesystem::path& path,
    const NativeBinaryCacheData& data, std::string& reason)
{
    std::filesystem::path temporary;
    reason.clear();
    try {
        validate(data);
        std::vector<uint8_t> bytes(kMagic.begin(), kMagic.end());
        append(bytes, uint32_t{1}); append(bytes, data.kind); append(bytes, data.programHash);
        append(bytes, static_cast<uint32_t>(data.compatibilityKey.size()));
        append(bytes, static_cast<uint32_t>(data.entries.size()));
        bytes.insert(bytes.end(), data.compatibilityKey.begin(), data.compatibilityKey.end());
        for (const auto& entry : data.entries) {
            append(bytes, static_cast<uint32_t>(entry.key.size()));
            append(bytes, static_cast<uint64_t>(entry.data.size()));
            bytes.insert(bytes.end(), entry.key.begin(), entry.key.end());
            bytes.insert(bytes.end(), entry.data.begin(), entry.data.end());
        }
        append(bytes, hashBytes(kFnvOffset, bytes.data(), bytes.size()));
        if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
        static std::atomic<uint64_t> sequence{0};
        temporary = path;
#if defined(_WIN32)
        const auto processId = GetCurrentProcessId();
#else
        const auto processId = getpid();
#endif
        temporary += ".tmp-" + std::to_string(processId) + "-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" + std::to_string(sequence.fetch_add(1));
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.close();
        if (!output) { throw std::runtime_error("cannot write native binary file"); }
#if defined(_WIN32)
        if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error("cannot atomically replace native binary file");
        }
#else
        std::filesystem::rename(temporary, path);
#endif
        return true;
    } catch (const std::exception& error) {
        reason = error.what();
        if (!temporary.empty()) { std::error_code errorCode; std::filesystem::remove(temporary, errorCode); }
        return false;
    }
}

} // namespace metallic::render::detail

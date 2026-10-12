#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace metallic::render::detail {

// Shader objects and pipeline binaries have distinct compatibility keys and
// identities, but share this bounded, checksummed native-data container.
struct NativeBinaryCacheEntry {
    std::vector<uint8_t> key;
    std::vector<uint8_t> data;
};

struct NativeBinaryCacheData {
    std::vector<uint8_t> compatibilityKey;
    uint64_t programHash = 0;
    uint32_t kind = 0;
    std::vector<NativeBinaryCacheEntry> entries;
};

enum class NativeBinaryCacheLoadStatus : uint8_t { NotFound, Loaded, Invalid, Incompatible };

NativeBinaryCacheLoadStatus loadNativeBinaryCacheFile(const std::filesystem::path& path,
    const NativeBinaryCacheData& identity, NativeBinaryCacheData& data, std::string& reason);
bool saveNativeBinaryCacheFile(const std::filesystem::path& path,
    const NativeBinaryCacheData& data, std::string& reason);

} // namespace metallic::render::detail

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace metallic::render::detail {

struct ShaderObjectCacheFileIdentity {
    std::array<uint8_t, 16> binaryUUID{};
    uint32_t binaryVersion = 0;
    uint64_t programHash = 0;
};

struct ShaderObjectCacheFileData {
    std::array<std::vector<uint8_t>, 2> binaries;
};

enum class ShaderObjectCacheFileLoadStatus : uint8_t {
    NotFound,
    Loaded,
    Invalid,
    Incompatible,
};

ShaderObjectCacheFileLoadStatus loadShaderObjectCacheFile(
    const std::filesystem::path& path,
    const ShaderObjectCacheFileIdentity& identity,
    ShaderObjectCacheFileData& outData,
    std::string& reason);

bool saveShaderObjectCacheFile(
    const std::filesystem::path& path,
    const ShaderObjectCacheFileIdentity& identity,
    const ShaderObjectCacheFileData& data,
    std::string& reason);

} // namespace metallic::render::detail

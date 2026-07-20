#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace metallic::render::detail {

struct PipelineCacheFileIdentity {
    uint32_t backendTag = 0;
    std::array<uint8_t, 32> compatibilityKey{};
};

enum class PipelineCacheFileLoadStatus : uint8_t {
    NotFound,
    Loaded,
    Invalid,
    Incompatible,
};

struct PipelineCacheFileData {
    std::vector<uint64_t> psoHashes;
    std::vector<uint8_t> backendData;
};

bool isPipelineCacheFilePath(const std::filesystem::path& path);

PipelineCacheFileLoadStatus loadPipelineCacheFile(
    const std::filesystem::path& path,
    const PipelineCacheFileIdentity& identity,
    PipelineCacheFileData& outData,
    std::string& reason);

bool savePipelineCacheFile(
    const std::filesystem::path& path,
    const PipelineCacheFileIdentity& identity,
    std::span<const uint64_t> psoHashes,
    std::span<const uint8_t> backendData,
    std::string& reason);

} // namespace metallic::render::detail

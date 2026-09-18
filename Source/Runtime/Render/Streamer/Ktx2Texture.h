#pragma once
#include "Runtime/Render/GAPI/TextureFormat.h"
#include <filesystem>
#include <string>
#include <vector>

namespace metallic::render {
struct Ktx2Level {
    uint64_t offset = 0, storedBytes = 0, decodedBytes = 0;
};
struct Ktx2TextureInfo {
    std::filesystem::path path;
    Format format = Format::Unknown;
    uint32_t width = 0, height = 0, supercompression = 0;
    std::array<TextureViewDesc::Component, 4> swizzle{};
    std::vector<Ktx2Level> levels;
    uint32_t firstMipForDimension(uint32_t maxDimension) const;
    uint64_t tailBytes(uint32_t firstMip) const;
    TextureDesc textureDesc(uint32_t firstMip) const;
};

bool isKtx2File(const std::filesystem::path& path);
// Restricted to static 2D BC4/BC5/BC7, raw or independent Zstd levels.
// Header inspection never reads image payload or relies on extension/MIME.
bool readKtx2TextureInfo(const std::filesystem::path& path, Ktx2TextureInfo& info, std::string& reason);
bool decodeKtx2Mip(const Ktx2TextureInfo& info, uint32_t mip, std::vector<uint8_t>& bytes,
                   std::string& reason);
} // namespace metallic::render

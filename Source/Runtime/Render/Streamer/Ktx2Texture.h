#pragma once
#include "Runtime/Render/GAPI/TextureFormat.h"
#include <filesystem>
#include <memory>
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

struct Ktx2ReadStats {
    uint64_t fileOpens = 0, decodedMips = 0, storedBytes = 0, decodedBytes = 0, decoderCreations = 0;
    double openMs = 0, readMs = 0, decodeMs = 0;
};

// One reader per loading worker. Reuse its input scratch and decoder across
// textures; open once for all selected mips. Not thread safe.
class Ktx2MipReader {
public:
    Ktx2MipReader();
    ~Ktx2MipReader();
    Ktx2MipReader(const Ktx2MipReader&) = delete;
    Ktx2MipReader& operator=(const Ktx2MipReader&) = delete;
    bool open(const Ktx2TextureInfo& info, std::string& reason);
    bool decode(uint32_t mip, std::vector<uint8_t>& bytes, std::string& reason);
    void close();
    void releaseScratch();
    void reserveScratch(uint64_t bytes);
    const Ktx2ReadStats& stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Ktx2PrefetchRequest {
    Ktx2TextureInfo info;
    uint32_t firstMip = 0;
};
struct Ktx2PrefetchResult {
    std::vector<std::vector<uint8_t>> mips;
    Ktx2ReadStats stats;
    std::string error;
    double workerMs = 0;
};
struct Ktx2PrefetchStats {
    uint32_t workers = 0, peakJobs = 0;
    uint64_t byteLimit = 0, peakBytes = 0;
};

// CPU-only, ordered consumption. Credits cover decoded tails and compressed
// input scratch for running AND completed jobs. pop() releases the credits.
class Ktx2TexturePrefetch {
public:
    Ktx2TexturePrefetch(std::vector<Ktx2PrefetchRequest> requests,
        uint32_t workers, uint64_t byteLimit);
    ~Ktx2TexturePrefetch();
    Ktx2TexturePrefetch(const Ktx2TexturePrefetch&) = delete;
    Ktx2TexturePrefetch& operator=(const Ktx2TexturePrefetch&) = delete;
    static uint64_t requiredBytes(const Ktx2TextureInfo& info, uint32_t firstMip);
    const Ktx2PrefetchResult* front() const;
    void wait();
    void pop();
    Ktx2PrefetchStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace metallic::render

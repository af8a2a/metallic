#include "Runtime/Render/Streamer/Ktx2Texture.h"
#include <zstd.h>
#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>

namespace metallic::render {
namespace {
constexpr std::array<uint8_t, 12> kIdentifier{0xab, 0x4b, 0x54, 0x58, 0x20, 0x32,
                                              0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a};
template <typename T> T little(const uint8_t* p)
{
    T value;
    std::memcpy(&value, p, sizeof(value));
    if constexpr (std::endian::native == std::endian::big) {
        return std::byteswap(value);
    }
    return value;
}
bool range(uint64_t offset, uint64_t size, uint64_t total)
{
    return offset <= total && size <= total - offset;
}
bool read(std::ifstream& file, uint64_t offset, std::span<uint8_t> bytes)
{
    file.seekg(offset);
    return bool(file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()));
}
} // namespace

bool isKtx2File(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::array<uint8_t, 12> magic{};
    return read(file, 0, magic) && magic == kIdentifier;
}

uint32_t Ktx2TextureInfo::firstMipForDimension(uint32_t maxDimension) const
{
    uint32_t mip = 0;
    while (mip + 1 < levels.size() && std::max(width >> mip, height >> mip) > maxDimension) {
        ++mip;
    }
    return mip;
}

uint64_t Ktx2TextureInfo::tailBytes(uint32_t firstMip) const
{
    uint64_t bytes = 0;
    for (size_t i = firstMip; i < levels.size(); ++i) {
        bytes += levels[i].decodedBytes;
    }
    return bytes;
}

TextureDesc Ktx2TextureInfo::textureDesc(uint32_t firstMip) const
{
    return {.usage = TextureUsageBits::Sampled | TextureUsageBits::TransferDestination,
            .format = format,
            .width = std::max(width >> firstMip, 1u),
            .height = std::max(height >> firstMip, 1u),
            .mipCount = uint32_t(levels.size()) - firstMip,
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy};
}

bool readKtx2TextureInfo(const std::filesystem::path& path, Ktx2TextureInfo& info, std::string& reason)
{
    info = {};
    info.path = path;
    const auto fail = [&](const char* message) {
        reason = path.string() + ": " + message;
        return false;
    };
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return fail("cannot open KTX2");
    }
    const auto end = file.tellg();
    if (end < 80) {
        return fail("truncated KTX2 header");
    }
    const uint64_t size = uint64_t(end);
    std::array<uint8_t, 80> h;
    if (!read(file, 0, h) || !std::equal(kIdentifier.begin(), kIdentifier.end(), h.begin())) {
        return fail("invalid KTX2 magic");
    }
    const auto u32 = [&](size_t offset) { return little<uint32_t>(h.data() + offset); };
    switch (u32(12)) {
    case 139:
        info.format = Format::Bc4Unorm;
        break;
    case 141:
        info.format = Format::Bc5Unorm;
        break;
    case 145:
        info.format = Format::Bc7Unorm;
        break;
    case 146:
        info.format = Format::Bc7Srgb;
        break;
    default:
        return fail("unsupported KTX2 format (expected BC4/BC5/BC7)");
    }
    info.width = u32(20);
    info.height = u32(24);
    info.supercompression = u32(44);
    const uint32_t count = u32(40), dfdOffset = u32(48), dfdSize = u32(52), kvOffset = u32(56),
                   kvSize = u32(60);
    const uint64_t indexEnd = 80 + uint64_t(count) * 24;
    if (u32(16) != 1 || !info.width || !info.height || info.width > 32768 || info.height > 32768 ||
        u32(28) != 0 || u32(32) != 0 || u32(36) != 1 || !count ||
        count > std::bit_width(std::max(info.width, info.height)) ||
        (info.supercompression != 0 && info.supercompression != 2) || little<uint64_t>(h.data() + 72) != 0) {
        return fail("unsupported KTX2 shape, mip count, or supercompression");
    }
    if (!range(80, count * 24, size) || dfdOffset < indexEnd || dfdSize < 28 || dfdSize > 65536 ||
        !range(dfdOffset, dfdSize, size) || kvSize > 16 * 1024 * 1024 ||
        (kvSize && (kvOffset < indexEnd || !range(kvOffset, kvSize, size)))) {
        return fail("invalid KTX2 metadata range");
    }
    std::vector<uint8_t> dfd(dfdSize);
    if (!read(file, dfdOffset, dfd) || little<uint32_t>(dfd.data()) != dfdSize) {
        return fail("invalid KTX2 DFD");
    }
    std::vector<std::pair<uint64_t, uint64_t>> ranges{{0, indexEnd},
                                                      {dfdOffset, uint64_t(dfdOffset) + dfdSize}};
    if (kvSize) {
        ranges.emplace_back(kvOffset, uint64_t(kvOffset) + kvSize);
    }
    std::vector<uint8_t> indices(count * 24);
    if (!read(file, 80, indices)) {
        return fail("truncated KTX2 mip index");
    }
    for (uint32_t mip = 0; mip < count; ++mip) {
        const auto* p = indices.data() + mip * 24;
        Ktx2Level level{little<uint64_t>(p), little<uint64_t>(p + 8), little<uint64_t>(p + 16)};
        const auto expected =
            bcMipBytes(info.format, std::max(info.width >> mip, 1u), std::max(info.height >> mip, 1u));
        if (level.decodedBytes != expected || !level.storedBytes ||
            level.storedBytes > 512ull * 1024 * 1024 || !range(level.offset, level.storedBytes, size) ||
            (info.supercompression == 0 && level.storedBytes != expected)) {
            return fail("KTX2 mip length does not match BC blocks");
        }
        ranges.emplace_back(level.offset, level.offset + level.storedBytes);
        info.levels.push_back(level);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
            return fail("overlapping KTX2 ranges");
        }
    }
    if (kvSize) {
        std::vector<uint8_t> kv(kvSize);
        if (!read(file, kvOffset, kv)) {
            return fail("truncated KTX2 metadata");
        }
        for (size_t offset = 0; offset < kv.size();) {
            if (kv.size() - offset < 4) {
                return fail("invalid KTX2 key length");
            }
            const size_t length = little<uint32_t>(kv.data() + offset);
            offset += 4;
            if (!length || length > kv.size() - offset) {
                return fail("invalid KTX2 key range");
            }
            const char* begin = reinterpret_cast<const char*>(kv.data() + offset);
            const char* separator = static_cast<const char*>(std::memchr(begin, 0, length));
            if (!separator) {
                return fail("unterminated KTX2 key");
            }
            const std::string key(begin, separator);
            std::string value(separator + 1, begin + length);
            if (!value.empty() && value.back() == '\0') {
                value.pop_back();
            }
            if (key == "KTXswizzle") {
                if (value.size() != 4) {
                    return fail("invalid KTXswizzle");
                }
                const std::string components = "01rgba";
                for (size_t c = 0; c < 4; ++c) {
                    const auto index = components.find(value[c]);
                    if (index == std::string::npos) {
                        return fail("unsupported KTXswizzle");
                    }
                    info.swizzle[c] = static_cast<TextureViewDesc::Component>(index + 1);
                }
            }
            if (key == "KTXorientation" && value != "rd") {
                return fail("unsupported KTXorientation");
            }
            offset += (length + 3) & ~size_t(3);
            if (offset > kv.size()) {
                return fail("truncated KTX2 key padding");
            }
        }
    }
    reason.clear();
    return true;
}

bool decodeKtx2Mip(const Ktx2TextureInfo& info, uint32_t mip, std::vector<uint8_t>& bytes,
                   std::string& reason)
{
    bytes.clear();
    if (mip >= info.levels.size()) {
        reason = "KTX2 mip out of range";
        return false;
    }
    const auto& level = info.levels[mip];
    std::ifstream file(info.path, std::ios::binary);
    std::vector<uint8_t> stored(level.storedBytes);
    if (!read(file, level.offset, stored)) {
        reason = "KTX2 mip read failed: " + info.path.string();
        return false;
    }
    if (info.supercompression == 0) {
        bytes = std::move(stored);
        return true;
    }
    bytes.resize(level.decodedBytes);
    auto* context = ZSTD_createDCtx();
    if (!context) {
        reason = "Cannot allocate Zstd decoder";
        return false;
    }
    ZSTD_DCtx_setParameter(context, ZSTD_d_windowLogMax, 27);
    const auto result =
        ZSTD_decompressDCtx(context, bytes.data(), bytes.size(), stored.data(), stored.size());
    ZSTD_freeDCtx(context);
    if (ZSTD_isError(result) || result != bytes.size()) {
        reason = "KTX2 Zstd mip decode failed: " + info.path.string();
        bytes.clear();
        return false;
    }
    return true;
}
} // namespace metallic::render

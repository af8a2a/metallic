#include "Runtime/Render/Streamer/Ktx2Texture.h"
#include <zstd.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdexcept>

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
            .queueAccess = QueueAccessBits::Graphics | QueueAccessBits::Copy,
            .memoryDomain = MemoryBudgetDomain::MaterialTextures};
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

struct Ktx2MipReader::Impl {
    std::ifstream file;
    Ktx2TextureInfo info;
    uint64_t fileSize = 0;
    std::vector<uint8_t> stored;
    ZSTD_DCtx* context = nullptr;
    Ktx2ReadStats stats;
    ~Impl() { ZSTD_freeDCtx(context); }
};

namespace {
struct ReadTimer {
    double& elapsed;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ~ReadTimer() { elapsed += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(); }
};
} // namespace

Ktx2MipReader::Ktx2MipReader() : impl_(std::make_unique<Impl>()) {}
Ktx2MipReader::~Ktx2MipReader() = default;

void Ktx2MipReader::close()
{
    if (impl_->file.is_open()) { impl_->file.close(); }
    impl_->file.clear();
    impl_->fileSize = 0;
}

void Ktx2MipReader::releaseScratch()
{
    std::vector<uint8_t>().swap(impl_->stored);
}

void Ktx2MipReader::reserveScratch(uint64_t bytes)
{
    impl_->stored.reserve(size_t(bytes));
}

const Ktx2ReadStats& Ktx2MipReader::stats() const { return impl_->stats; }

bool Ktx2MipReader::open(const Ktx2TextureInfo& info, std::string& reason)
{
    ReadTimer timer{impl_->stats.openMs};
    close();
    impl_->info = info;
    ++impl_->stats.fileOpens;
    impl_->file.open(info.path, std::ios::binary | std::ios::ate);
    const auto size = impl_->file.tellg();
    if (!impl_->file || size < 0) {
        reason = "Cannot open KTX2 payload: " + info.path.string();
        close();
        return false;
    }
    impl_->fileSize = uint64_t(size);
    reason.clear();
    return true;
}

bool Ktx2MipReader::decode(uint32_t mip, std::vector<uint8_t>& bytes, std::string& reason)
{
    bytes.clear();
    auto& p = *impl_;
    const auto fail = [&](const char* message) {
        bytes.clear();
        reason = std::string(message) + ": " + p.info.path.string();
        return false;
    };
    if (!p.file.is_open() || mip >= p.info.levels.size()) { return fail("KTX2 reader closed or mip out of range"); }
    const auto& level = p.info.levels[mip];
    if (!range(level.offset, level.storedBytes, p.fileSize) ||
        level.storedBytes > uint64_t(std::numeric_limits<std::streamsize>::max()) ||
        level.storedBytes > SIZE_MAX || level.decodedBytes > SIZE_MAX ||
        (p.info.supercompression != 0 && p.info.supercompression != 2)) {
        return fail("Invalid KTX2 mip payload range or compression");
    }
    {
        ReadTimer timer{p.stats.readMs};
        p.stored.resize(size_t(level.storedBytes));
        p.file.clear();
        if (!read(p.file, level.offset, p.stored)) { return fail("KTX2 mip read failed"); }
        p.stats.storedBytes += level.storedBytes;
    }
    {
        ReadTimer timer{p.stats.decodeMs};
        bytes.resize(size_t(level.decodedBytes));
        if (p.info.supercompression == 0) {
            if (level.storedBytes != level.decodedBytes) { return fail("KTX2 raw mip size mismatch"); }
            std::memcpy(bytes.data(), p.stored.data(), bytes.size());
        } else {
            if (!p.context) {
                p.context = ZSTD_createDCtx();
                if (!p.context) { return fail("Cannot allocate Zstd decoder"); }
                ++p.stats.decoderCreations;
                ZSTD_DCtx_setParameter(p.context, ZSTD_d_windowLogMax, 27);
            }
            const auto result = ZSTD_decompressDCtx(p.context, bytes.data(), bytes.size(), p.stored.data(), p.stored.size());
            if (ZSTD_isError(result) || result != bytes.size()) { return fail("KTX2 Zstd mip decode failed"); }
        }
    }
    ++p.stats.decodedMips;
    p.stats.decodedBytes += bytes.size();
    reason.clear();
    return true;
}

struct Ktx2TexturePrefetch::Impl {
    struct Job {
        Ktx2PrefetchRequest request;
        Ktx2PrefetchResult result;
        uint64_t bytes = 0;
        bool ready = false;
    };
    std::vector<Job> jobs;
    mutable std::mutex mutex;
    std::condition_variable_any condition;
    size_t next = 0, consumed = 0;
    uint64_t reservedBytes = 0;
    Ktx2PrefetchStats stats;
    std::vector<std::jthread> threads;

    void stop()
    {
        for (auto& thread : threads) { thread.request_stop(); }
        condition.notify_all();
        threads.clear(); // Join before any job storage is destroyed.
    }
    ~Impl() { stop(); }

    void run(std::stop_token stop)
    {
        Ktx2MipReader reader;
        for (;;) {
            size_t index;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, stop, [&] {
                    return next == jobs.size() || (next - consumed < stats.workers * 2 &&
                        jobs[next].bytes <= stats.byteLimit - reservedBytes);
                });
                if (stop.stop_requested() || next == jobs.size()) { return; }
                index = next++;
                reservedBytes += jobs[index].bytes;
                stats.peakBytes = std::max(stats.peakBytes, reservedBytes);
                stats.peakJobs = std::max(stats.peakJobs, uint32_t(next - consumed));
            }
            auto& job = jobs[index];
            auto& result = job.result;
            const auto before = reader.stats();
            const auto begin = std::chrono::steady_clock::now();
            try {
                reader.reserveScratch(job.bytes - job.request.info.tailBytes(job.request.firstMip));
                if (reader.open(job.request.info, result.error)) {
                    const auto first = job.request.firstMip;
                    result.mips.resize(job.request.info.levels.size() - first);
                    for (uint32_t mip = first; mip < job.request.info.levels.size(); ++mip) {
                        if (stop.stop_requested()) { break; }
                        if (!reader.decode(mip, result.mips[mip - first], result.error)) { break; }
                    }
                }
            } catch (const std::exception& exception) {
                result.error = exception.what();
            } catch (...) {
                result.error = "KTX2 prefetch failed with an unknown exception";
            }
            reader.close();
            reader.releaseScratch(); // Retained input capacity must not escape byte credits.
            const auto& after = reader.stats();
            result.stats = {after.fileOpens - before.fileOpens, after.decodedMips - before.decodedMips,
                after.storedBytes - before.storedBytes, after.decodedBytes - before.decodedBytes,
                after.decoderCreations - before.decoderCreations, after.openMs - before.openMs,
                after.readMs - before.readMs, after.decodeMs - before.decodeMs};
            result.workerMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            {
                std::lock_guard lock(mutex);
                job.ready = true;
            }
            condition.notify_all();
        }
    }
};

uint64_t Ktx2TexturePrefetch::requiredBytes(const Ktx2TextureInfo& info, uint32_t firstMip)
{
    uint64_t stored = 0;
    for (size_t mip = firstMip; mip < info.levels.size(); ++mip) {
        stored = std::max(stored, info.levels[mip].storedBytes);
    }
    return info.tailBytes(firstMip) + stored;
}

Ktx2TexturePrefetch::Ktx2TexturePrefetch(std::vector<Ktx2PrefetchRequest> requests,
    uint32_t workers, uint64_t byteLimit) : impl_(std::make_unique<Impl>())
{
    impl_->stats.workers = std::min(uint32_t(requests.size()), std::clamp(workers, 1u, 8u));
    impl_->stats.byteLimit = byteLimit;
    for (auto& request : requests) {
        const uint64_t bytes = requiredBytes(request.info, request.firstMip);
        if (request.firstMip >= request.info.levels.size() || bytes > byteLimit) {
            throw std::invalid_argument("KTX2 prefetch request exceeds byte limit or has no selected mip");
        }
        impl_->jobs.push_back({std::move(request), {}, bytes});
    }
    // Impl's destructor also joins partially constructed pools if thread creation fails.
    for (uint32_t i = 0; i < impl_->stats.workers; ++i) {
        impl_->threads.emplace_back([p = impl_.get()](std::stop_token stop) { p->run(stop); });
    }
}
Ktx2TexturePrefetch::~Ktx2TexturePrefetch() = default;

const Ktx2PrefetchResult* Ktx2TexturePrefetch::front() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->consumed < impl_->jobs.size() && impl_->jobs[impl_->consumed].ready
        ? &impl_->jobs[impl_->consumed].result : nullptr;
}
void Ktx2TexturePrefetch::wait()
{
    std::unique_lock lock(impl_->mutex);
    impl_->condition.wait(lock, [&] {
        return impl_->consumed == impl_->jobs.size() || impl_->jobs[impl_->consumed].ready;
    });
}
void Ktx2TexturePrefetch::pop()
{
    {
        std::lock_guard lock(impl_->mutex);
        auto& job = impl_->jobs.at(impl_->consumed);
        if (!job.ready) { throw std::logic_error("KTX2 prefetch result is not ready"); }
        job.result = {}; // Free payload BEFORE returning credits.
        impl_->reservedBytes -= job.bytes;
        ++impl_->consumed;
    }
    impl_->condition.notify_all();
}
Ktx2PrefetchStats Ktx2TexturePrefetch::stats() const
{
    std::lock_guard lock(impl_->mutex);
    return impl_->stats;
}

bool decodeKtx2Mip(const Ktx2TextureInfo& info, uint32_t mip, std::vector<uint8_t>& bytes, std::string& reason)
{
    Ktx2MipReader reader;
    bytes.clear();
    return reader.open(info, reason) && reader.decode(mip, bytes, reason);
}
} // namespace metallic::render

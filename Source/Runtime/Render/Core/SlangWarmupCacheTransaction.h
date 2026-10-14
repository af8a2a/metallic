#pragma once

#include "SlangShaderCache.h"
#include <filesystem>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>

namespace metallic::render {

// Complete-batch publisher using the caller's atomic cache writer;
// it does not replace filesystem/rename policy or register testing artifacts.
template<typename Snapshot>
class SlangWarmupCacheTransaction {
public:
    struct Timings {
        uint64_t proofEncodeNanoseconds = 0;
        uint64_t atomicWriteNanoseconds = 0;
        uint64_t fileReadDecodeNanoseconds = 0;
        uint64_t fullSourceValidationNanoseconds = 0;
    };
    using AtomicWrite = std::function<bool(const std::filesystem::path&, std::span<const uint8_t>)>;
    using NormalRegistration = std::function<void(size_t)>;
    struct Item {
        size_t requestIndex = 0;
        uint64_t requestHash = 0;
        std::vector<uint32_t> finalizedSpirv;
    };

    SlangWarmupCacheTransaction(Snapshot& snapshot, uint64_t compilerProducerIdentity,
        std::filesystem::path cacheDirectory) : snapshot_(snapshot), producerIdentity_(compilerProducerIdentity),
        cacheDirectory_(std::move(cacheDirectory)) {}

    bool stage(Item item, std::string& log)
    {
        if (published_ || failed_ || item.finalizedSpirv.empty()) { log = "Invalid warmup cache transaction state"; return false; }
        for (const auto& previous : items_) {
            if (previous.requestIndex == item.requestIndex || previous.requestHash == item.requestHash) {
                log = "Duplicate warmup request identity"; failed_ = true; return false;
            }
        }
        items_.push_back(std::move(item));
        return true;
    }

    bool publish(size_t expectedRequestCount, const AtomicWrite& atomicWrite,
        const NormalRegistration& registerNormal, std::string& log)
    {
        if (failed_ || published_ || !expectedRequestCount || items_.size() != expectedRequestCount || !atomicWrite || !registerNormal) {
            log = "Incomplete or invalid complete warmup publication"; return false;
        }
        // All stage results stay private until the complete source epoch has
        // passed. Proof hashes are taken from delivered immutable bytes.
        const auto validateSource = [&] {
            const auto begin = std::chrono::steady_clock::now();
            const bool valid = snapshot_.validate(log);
            timings_.fullSourceValidationNanoseconds += elapsed(begin);
            return valid;
        };
        if (!validateSource()) { return discard(log); }
        const auto encodeStart = std::chrono::steady_clock::now();
        const auto proof = snapshot_.sourceProof(producerIdentity_);
        const auto qualify = [&](const ShaderNamespaceProof& cached, ShaderNamespaceProof& current) {
            return snapshot_.qualifyCachedProof(cached, producerIdentity_, current, log);
        };
        std::vector<slang_shader_cache::Bytes> encoded;
        for (const auto& item : items_) {
            auto bytes = slang_shader_cache::encode(item.requestHash, proof, item.finalizedSpirv);
            // Encoding validates header/proof shape. The one acceptance gate
            // is the actual persisted file's complete decoder below: current
            // source proof, checksum, SPIR-V and exact expected bytes all stay
            // mandatory. No completed result/registry registration precedes it.
            if (bytes.empty()) { log = "Pending V3 warmup encode failed"; return discard(log); }
            encoded.push_back(std::move(bytes));
        }
        timings_.proofEncodeNanoseconds = elapsed(encodeStart);
        if (!validateSource()) { return discard(log); }
        for (size_t index = 0; index < items_.size(); ++index) {
            const auto& item = items_[index];
            std::ostringstream name;
            name << std::hex << std::setw(16) << std::setfill('0') << item.requestHash << ".spv";
            const auto path = cacheDirectory_ / name.str();
            const auto writeStart = std::chrono::steady_clock::now();
            const bool written = atomicWrite(path, encoded[index]);
            timings_.atomicWriteNanoseconds += elapsed(writeStart);
            if (!written) { log = "Atomic V3 warmup cache persistence failed"; return discard(log); }
            const auto readStart = std::chrono::steady_clock::now();
            std::error_code fileError;
            const auto size = std::filesystem::file_size(path, fileError);
            if (fileError || size != encoded[index].size() || size > slang_shader_cache::kMaximumFileBytes) {
                log = "Actual V3 warmup file size differs from expected envelope"; return discard(log);
            }
            std::ifstream stream(path, std::ios::binary);
            if (!stream) { log = "Actual V3 warmup file cannot be opened"; return discard(log); }
            slang_shader_cache::Bytes real(static_cast<size_t>(size));
            stream.read(reinterpret_cast<char*>(real.data()), static_cast<std::streamsize>(real.size()));
            if (stream.gcount() != static_cast<std::streamsize>(real.size()) || stream.bad() ||
                stream.peek() != std::char_traits<char>::eof()) {
                log = "Actual V3 warmup file short read or trailing bytes"; return discard(log);
            }
            slang_shader_cache::CachedPayload decoded;
            if (stream.bad() || !slang_shader_cache::decode(real, item.requestHash, qualify, decoded) ||
                decoded.spirv != item.finalizedSpirv || real != encoded[index]) {
                log = "Actual V3 warmup file readback failed"; return discard(log);
            }
            timings_.fileReadDecodeNanoseconds += elapsed(readStart);
        }
        if (!validateSource()) { return discard(log); }
        // The callback is ShaderRegistry's normal identity helper, preserving
        // compilation metrics and genuine cache-hit status for each result.
        for (const auto& item : items_) { registerNormal(item.requestIndex); }
        published_ = true;
        return true;
    }

    bool published() const noexcept { return published_; }
    const Timings& timings() const noexcept { return timings_; }
private:
    static uint64_t elapsed(std::chrono::steady_clock::time_point begin)
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count());
    }
    bool discard(std::string& log)
    {
        failed_ = true; items_.clear();
        // Already persisted V3 files remain individually namespace-qualified;
        // no completed batch/result manifest or registry identities are added.
        // A later real source change cannot hit them with the old namespace.
        if (log.empty()) { log = "Warmup immutable source epoch was rejected"; }
        return false;
    }
    Snapshot& snapshot_;
    uint64_t producerIdentity_;
    std::filesystem::path cacheDirectory_;
    std::vector<Item> items_;
    bool published_ = false, failed_ = false;
    Timings timings_;
};

} // namespace metallic::render

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace metallic::render::detail {

// Device/driver-specific cold creation costs. The identity belongs to the
// caller; PSO hashes already include the complete pipeline state and bytecode.
class PipelineCreationHistory {
public:
    PipelineCreationHistory(std::filesystem::path path, std::array<uint8_t, 32> identity);

    // Load during initialization, before admitting pipeline observations.
    // Missing files are an empty history; rejected files leave no entries.
    bool load(std::string& reason);
    std::optional<uint64_t> duration(uint64_t psoHash) const;
    // Call only after successful native creation. Cache hits cannot reduce a
    // remembered cold cost. Returns a changed revision, or zero for no change.
    uint64_t observe(uint64_t psoHash, uint64_t nativeNanoseconds,
        bool feedbackValid, bool nativeCacheHit);
    // Concurrent observations continue while the snapshot is written. Calls
    // are serialized so an older snapshot cannot replace a newer save.
    bool save(std::string& reason);

private:
    std::filesystem::path path_;
    std::array<uint8_t, 32> identity_;
    mutable std::mutex mutex_;
    std::mutex ioMutex_;
    std::unordered_map<uint64_t, uint64_t> durations_;
    uint64_t revision_ = 0;
};

} // namespace metallic::render::detail

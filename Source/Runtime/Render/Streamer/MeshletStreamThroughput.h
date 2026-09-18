#pragma once

#include <cstdint>
#include <deque>

namespace metallic::render {

struct MeshletStreamTraffic {
    // Successful worker loads, including re-loads. File payload bytes touched,
    // not physical disk traffic (the file may already be mapped/cached).
    uint64_t loadedPages = 0;
    uint64_t loadedStoredBytes = 0;
    uint64_t preparedDeviceBytes = 0;
    // Accepted copy payload, excluding GPU envelope/CPU sideband bytes.
    uint64_t transferPayloadBytes = 0;
    // Completed upload receipts observed by residency, not actual draw/RT use.
    uint64_t geometryReadyPages = 0;
    uint64_t geometryReadyBytes = 0;
    uint64_t smallBatchCpuPages = 0;
};

struct MeshletStreamThroughput {
    MeshletStreamTraffic totals;
    double windowSeconds = 0;
    double loadedPagesPerSecond = 0;
    double loadedStoredMiBPerSecond = 0;
    double preparedMiBPerSecond = 0;
    double transferMiBPerSecond = 0;
    double geometryReadyPagesPerSecond = 0;
    double geometryReadyMiBPerSecond = 0;
};

// Wall-clock rolling window, sampled at most every 50 ms. Idle frames age the
// samples too; a generation reset clears both history and cumulative traffic.
class MeshletStreamThroughputTracker {
public:
    void sample(uint64_t now, const MeshletStreamTraffic& totals)
    {
        if (!samples_.empty() && (now < samples_.back().time ||
            totals.loadedPages < samples_.back().traffic.loadedPages ||
            totals.geometryReadyBytes < samples_.back().traffic.geometryReadyBytes)) {
            samples_.clear();
        }
        while (samples_.size() > 1 && now >= samples_[1].time && now - samples_[1].time >= 1'000'000) {
            samples_.pop_front();
        }
        if (samples_.empty() || now - samples_.back().time >= 50'000) { samples_.push_back({now, totals}); }
    }

    MeshletStreamThroughput snapshot(uint64_t now, const MeshletStreamTraffic& totals) const
    {
        MeshletStreamThroughput result{.totals = totals};
        if (samples_.empty() || now <= samples_.front().time) { return result; }
        result.windowSeconds = double(now - samples_.front().time) / 1e6;
        // Avoid showing enormous startup rates from a nearly zero denominator.
        if (result.windowSeconds < 0.05) { return result; }
        const auto rate = [&](uint64_t current, uint64_t previous) {
            return current >= previous ? double(current - previous) / result.windowSeconds : 0.0;
        };
        constexpr double mib = 1024.0 * 1024.0;
        const auto& before = samples_.front().traffic;
        result.loadedPagesPerSecond = rate(totals.loadedPages, before.loadedPages);
        result.loadedStoredMiBPerSecond = rate(totals.loadedStoredBytes, before.loadedStoredBytes) / mib;
        result.preparedMiBPerSecond = rate(totals.preparedDeviceBytes, before.preparedDeviceBytes) / mib;
        result.transferMiBPerSecond = rate(totals.transferPayloadBytes, before.transferPayloadBytes) / mib;
        result.geometryReadyPagesPerSecond = rate(totals.geometryReadyPages, before.geometryReadyPages);
        result.geometryReadyMiBPerSecond = rate(totals.geometryReadyBytes, before.geometryReadyBytes) / mib;
        return result;
    }

private:
    struct Sample { uint64_t time; MeshletStreamTraffic traffic; };
    std::deque<Sample> samples_;
};

} // namespace metallic::render

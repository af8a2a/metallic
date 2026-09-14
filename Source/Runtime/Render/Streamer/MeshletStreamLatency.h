#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <unordered_map>

namespace metallic::render {

inline uint64_t meshletStreamTimeMicroseconds()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

struct MeshletStreamLatencySummary {
    uint64_t count = 0;
    double p50 = 0, p95 = 0, p99 = 0, maximum = 0, mean = 0;
};

// Cumulative, bounded memory; percentiles are upper bounds in 1 ms bins.
// The overflow bin uses the observed maximum, so long stalls stay visible.
struct MeshletStreamLatencyHistogram {
    std::array<uint64_t, 2049> bins{};
    uint64_t count = 0, totalMicroseconds = 0, maximumMicroseconds = 0;

    void observe(uint64_t microseconds)
    {
        ++bins[std::min<uint64_t>((microseconds + 999) / 1000, bins.size() - 1)];
        ++count;
        totalMicroseconds += microseconds;
        maximumMicroseconds = std::max(maximumMicroseconds, microseconds);
    }

    MeshletStreamLatencySummary summary() const
    {
        MeshletStreamLatencySummary result{.count = count};
        if (count == 0) { return result; }
        const auto quantile = [&](uint64_t percent) {
            const uint64_t rank = (count * percent + 99) / 100;
            uint64_t sum = 0;
            for (size_t i = 0; i < bins.size(); ++i) {
                sum += bins[i];
                if (sum >= rank) { return i + 1 == bins.size() ? double(maximumMicroseconds) / 1000.0 : double(i); }
            }
            return 0.0;
        };
        result.p50 = quantile(50); result.p95 = quantile(95); result.p99 = quantile(99);
        result.maximum = double(maximumMicroseconds) / 1000.0;
        result.mean = double(totalMicroseconds) / double(count) / 1000.0;
        return result;
    }
};

enum class MeshletStreamLatencyStage : uint32_t {
    Feedback, Admission, IoQueue, Decode, ReadyToUpload, UploadToDrawable,
    DemandToDrawable, PrefetchToDrawable, Count
};

struct MeshletStreamLatencySnapshot {
    bool enabled = false;
    std::array<MeshletStreamLatencySummary, size_t(MeshletStreamLatencyStage::Count)> milliseconds{};
    MeshletStreamLatencySummary demandFrames;
    uint32_t pendingDemand = 0, pendingPrefetch = 0;
    uint64_t abandonedDemand = 0, abandonedPrefetch = 0;
    double oldestPendingDemandMilliseconds = 0;
};

// Side table covers admission failures too; PageEntry only exists after admission.
// Source-frame starts approximate GPU request time by at most that frame's work.
class MeshletStreamLatencyTracker {
public:
    struct Request {
        uint64_t firstTime = 0, feedbackTime = 0, lastSeenFrame = 0;
        uint64_t demandTime = 0, demandFrame = 0;
        uint64_t admissionTime = 0, enqueueTime = 0, uploadTime = 0;
    };
    std::unordered_map<uint32_t, Request> pending;
    std::array<MeshletStreamLatencyHistogram, size_t(MeshletStreamLatencyStage::Count)> stages{};
    MeshletStreamLatencyHistogram demandFrames;
    std::array<std::array<uint64_t, 2>, 256> frameTimes{};
    uint64_t abandonedDemand = 0, abandonedPrefetch = 0;

    void observe(MeshletStreamLatencyStage stage, uint64_t start, uint64_t end)
    {
        if (start != 0 && end >= start) { stages[size_t(stage)].observe(end - start); }
    }

    void request(uint32_t page, uint64_t sourceFrame, uint64_t cpuFrame, bool prefetch)
    {
        const uint64_t now = meshletStreamTimeMicroseconds();
        const auto& stamp = frameTimes[sourceFrame % frameTimes.size()];
        const uint64_t sourceTime = stamp[0] == sourceFrame && stamp[1] != 0 ? stamp[1] : now;
        auto& request = pending[page];
        if (request.firstTime == 0) {
            request.firstTime = sourceTime; request.feedbackTime = now;
            observe(MeshletStreamLatencyStage::Feedback, sourceTime, now);
        }
        request.lastSeenFrame = cpuFrame;
        if (!prefetch && request.demandTime == 0) {
            request.demandTime = sourceTime; request.demandFrame = std::min(sourceFrame, cpuFrame);
        }
    }

    void abandon(uint32_t page)
    {
        const auto found = pending.find(page);
        if (found == pending.end()) { return; }
        if (found->second.demandTime != 0) { ++abandonedDemand; } else { ++abandonedPrefetch; }
        pending.erase(found);
    }

    void complete(uint32_t page, uint64_t frame)
    {
        const auto found = pending.find(page);
        if (found == pending.end()) { return; }
        const auto& request = found->second;
        const uint64_t now = meshletStreamTimeMicroseconds();
        observe(MeshletStreamLatencyStage::UploadToDrawable, request.uploadTime, now);
        if (request.demandTime != 0) {
            observe(MeshletStreamLatencyStage::DemandToDrawable, request.demandTime, now);
            demandFrames.observe((frame - request.demandFrame) * 1000);
        } else {
            observe(MeshletStreamLatencyStage::PrefetchToDrawable, request.firstTime, now);
        }
        pending.erase(found);
    }

    MeshletStreamLatencySnapshot snapshot() const
    {
        MeshletStreamLatencySnapshot result{.enabled = true};
        for (size_t i = 0; i < stages.size(); ++i) { result.milliseconds[i] = stages[i].summary(); }
        result.demandFrames = demandFrames.summary();
        result.abandonedDemand = abandonedDemand; result.abandonedPrefetch = abandonedPrefetch;
        const uint64_t now = meshletStreamTimeMicroseconds();
        for (const auto& [page, request] : pending) {
            if (request.demandTime != 0) {
                ++result.pendingDemand;
                result.oldestPendingDemandMilliseconds = std::max(result.oldestPendingDemandMilliseconds,
                    double(now - request.demandTime) / 1000.0);
            } else { ++result.pendingPrefetch; }
        }
        return result;
    }
};

} // namespace metallic::render

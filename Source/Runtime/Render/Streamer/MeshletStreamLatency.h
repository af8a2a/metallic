#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

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
    std::array<MeshletStreamLatencyHistogram, size_t(MeshletStreamLatencyStage::Count)> stages{};
    MeshletStreamLatencyHistogram demandFrames;
    std::array<std::array<uint64_t, 2>, 256> frameTimes{};
    uint64_t abandonedDemand = 0, abandonedPrefetch = 0;

    void observe(MeshletStreamLatencyStage stage, uint64_t start, uint64_t end)
    {
        if (start != 0 && end >= start) { stages[size_t(stage)].observe(end - start); }
    }

    explicit MeshletStreamLatencyTracker(uint64_t maxAgeFrames = 8) : maxAgeFrames_(maxAgeFrames)
    {
        expiryHeads_.fill(kInvalid);
        expiryTails_.fill(kInvalid);
    }

    Request* find(uint32_t page)
    {
        const uint32_t slot = findSlot(page);
        return slot == kInvalid ? nullptr : &entries_[slot].request;
    }

    // One timestamp per feedback batch. Callers without a batch retain the old API.
    void request(uint32_t page, uint64_t sourceFrame, uint64_t cpuFrame, bool prefetch,
        uint64_t now = meshletStreamTimeMicroseconds())
    {
        const auto& stamp = frameTimes[sourceFrame % frameTimes.size()];
        const uint64_t sourceTime = stamp[0] == sourceFrame && stamp[1] != 0 ? stamp[1] : now;
        uint32_t slot = findSlot(page);
        if (slot == kInvalid) {
            const size_t block = page / kIndexBlockSize;
            if (block >= indexBlocks_.size()) { indexBlocks_.resize(block + 1); }
            if (!indexBlocks_[block]) {
                indexBlocks_[block] = std::make_unique<IndexBlock>();
                indexBlocks_[block]->fill(kInvalid);
            }
            if (freeSlots_.empty()) {
                slot = static_cast<uint32_t>(entries_.size());
                entries_.emplace_back();
            } else {
                slot = freeSlots_.back();
                freeSlots_.pop_back();
                entries_[slot] = {};
            }
            entries_[slot].active = true;
            entries_[slot].page = page;
            (*indexBlocks_[block])[page % kIndexBlockSize] = slot;
        }
        auto& entry = entries_[slot];
        auto& request = entry.request;
        if (request.firstTime == 0) {
            request.firstTime = sourceTime; request.feedbackTime = now;
            observe(MeshletStreamLatencyStage::Feedback, sourceTime, now);
        }
        request.lastSeenFrame = cpuFrame;
        if (!prefetch && request.demandTime == 0) {
            request.demandTime = sourceTime; request.demandFrame = std::min(sourceFrame, cpuFrame);
        }
        const uint64_t due = cpuFrame + maxAgeFrames_ + 1;
        // Renew lazily when the existing event becomes due; repeated demand only
        // updates lastSeenFrame instead of relinking thousands of entries per frame.
        if (!entry.scheduled) { schedule(slot, due); }
    }

    // Visit only due buckets. A protected overdue request is checked next frame,
    // preserving cancellation/expiry behavior without scanning every pending ID.
    // The predicate must not mutate this tracker.
    template<class IsInFlight>
    void expire(uint64_t frame, IsInFlight&& isInFlight)
    {
        if (frame <= expiryFrame_) { return; }
        const uint64_t steps = std::min<uint64_t>(frame - expiryFrame_, kExpiryBuckets);
        for (uint64_t step = 0; step < steps; ++step) {
            const uint32_t bucket = static_cast<uint32_t>((frame - steps + 1 + step) % kExpiryBuckets);
            // Requeued entries may land in this same bucket after a wheel wrap.
            // Only process its original members, once each.
            uint32_t count = expiryCounts_[bucket];
            while (count-- != 0) {
                const uint32_t slot = expiryHeads_[bucket];
                auto& entry = entries_[slot];
                const uint64_t due = std::max(entry.dueFrame, entry.request.lastSeenFrame + maxAgeFrames_ + 1);
                unlink(slot);
                if (due > frame) { schedule(slot, due); }
                else if (isInFlight(entry.page)) { schedule(slot, frame + 1); }
                else { abandon(entry.page); }
            }
        }
        expiryFrame_ = frame;
    }

    void abandon(uint32_t page)
    {
        const uint32_t slot = findSlot(page);
        if (slot == kInvalid) { return; }
        if (entries_[slot].request.demandTime != 0) { ++abandonedDemand; } else { ++abandonedPrefetch; }
        retire(slot);
    }

    void complete(uint32_t page, uint64_t frame, uint64_t now = meshletStreamTimeMicroseconds())
    {
        const uint32_t slot = findSlot(page);
        if (slot == kInvalid) { return; }
        const auto& request = entries_[slot].request;
        observe(MeshletStreamLatencyStage::UploadToDrawable, request.uploadTime, now);
        if (request.demandTime != 0) {
            observe(MeshletStreamLatencyStage::DemandToDrawable, request.demandTime, now);
            demandFrames.observe((frame - request.demandFrame) * 1000);
        } else {
            observe(MeshletStreamLatencyStage::PrefetchToDrawable, request.firstTime, now);
        }
        retire(slot);
    }

    MeshletStreamLatencySnapshot snapshot(uint64_t now = meshletStreamTimeMicroseconds()) const
    {
        MeshletStreamLatencySnapshot result{.enabled = true};
        for (size_t i = 0; i < stages.size(); ++i) { result.milliseconds[i] = stages[i].summary(); }
        result.demandFrames = demandFrames.summary();
        result.abandonedDemand = abandonedDemand; result.abandonedPrefetch = abandonedPrefetch;
        for (const auto& entry : entries_) {
            if (!entry.active) { continue; }
            const auto& request = entry.request;
            if (request.demandTime != 0) {
                ++result.pendingDemand;
                result.oldestPendingDemandMilliseconds = std::max(result.oldestPendingDemandMilliseconds,
                    double(now - request.demandTime) / 1000.0);
            } else { ++result.pendingPrefetch; }
        }
        return result;
    }

private:
    static constexpr uint32_t kInvalid = UINT32_MAX;
    static constexpr uint32_t kIndexBlockSize = 1024;
    static constexpr uint32_t kExpiryBuckets = 256;
    using IndexBlock = std::array<uint32_t, kIndexBlockSize>;
    struct Entry {
        Request request;
        uint64_t dueFrame = 0;
        uint32_t page = kInvalid, previous = kInvalid, next = kInvalid;
        bool active = false, scheduled = false;
    };
    std::vector<std::unique_ptr<IndexBlock>> indexBlocks_;
    std::vector<Entry> entries_;
    std::vector<uint32_t> freeSlots_;
    std::array<uint32_t, kExpiryBuckets> expiryHeads_, expiryTails_, expiryCounts_{};
    uint64_t expiryFrame_ = 0, maxAgeFrames_ = 8;

    uint32_t findSlot(uint32_t page) const
    {
        const size_t block = page / kIndexBlockSize;
        if (block >= indexBlocks_.size() || !indexBlocks_[block]) { return kInvalid; }
        return (*indexBlocks_[block])[page % kIndexBlockSize];
    }

    void unlink(uint32_t slot)
    {
        auto& entry = entries_[slot];
        if (!entry.scheduled) { return; }
        const uint32_t bucket = static_cast<uint32_t>(entry.dueFrame % kExpiryBuckets);
        if (entry.previous != kInvalid) { entries_[entry.previous].next = entry.next; }
        else { expiryHeads_[bucket] = entry.next; }
        if (entry.next != kInvalid) { entries_[entry.next].previous = entry.previous; }
        else { expiryTails_[bucket] = entry.previous; }
        --expiryCounts_[bucket];
        entry.previous = entry.next = kInvalid;
        entry.scheduled = false;
    }

    void schedule(uint32_t slot, uint64_t due)
    {
        unlink(slot);
        auto& entry = entries_[slot];
        const uint32_t bucket = static_cast<uint32_t>(due % kExpiryBuckets);
        entry.dueFrame = due;
        entry.previous = expiryTails_[bucket];
        if (entry.previous != kInvalid) { entries_[entry.previous].next = slot; }
        else { expiryHeads_[bucket] = slot; }
        expiryTails_[bucket] = slot;
        ++expiryCounts_[bucket];
        entry.scheduled = true;
    }

    void retire(uint32_t slot)
    {
        auto& entry = entries_[slot];
        // Intrusive removal cancels the only expiry event before slot reuse.
        // No stale event can address a subsequent request at this slot/page ID.
        unlink(slot);
        (*indexBlocks_[entry.page / kIndexBlockSize])[entry.page % kIndexBlockSize] = kInvalid;
        entry.active = false;
        freeSlots_.push_back(slot);
    }

};

} // namespace metallic::render

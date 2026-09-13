#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace metallic::render::profiling {

// Opt-in CPU attribution for a bounded sequence of frames. No file I/O or
// clocks on the disabled path; a caller serializes the events after recording.
struct CpuPhaseTrace {
    using Clock = std::chrono::steady_clock;
    struct Event {
        const char* name; // Static labels; valid until the caller serializes.
        uint64_t frame;
        uint64_t startNanoseconds;
        uint64_t durationNanoseconds;
        uint64_t value;
        uint32_t depth;
    };
    struct GpuSpan {
        uint64_t executionId;
        uint64_t beginTimestamp;
        uint64_t endTimestamp;
        uint64_t calibrationTimestamp;
        double periodNanoseconds;
        uint64_t hostReferenceNanoseconds;
        uint64_t calibrationCallNanoseconds;
        uint64_t maxDeviationNanoseconds;
    };
    static constexpr size_t kMaxEvents = 65536;
    static constexpr size_t kMaxFrames = 1024;
    Clock::time_point origin = Clock::now();
    std::vector<Event> events;
    std::vector<GpuSpan> gpuSpans;
    uint64_t frame = 0;
    uint32_t depth = 0;
    uint32_t dropped = 0;
    CpuPhaseTrace()
    {
        events.reserve(kMaxEvents);
        gpuSpans.reserve(kMaxFrames);
    }
    static inline thread_local CpuPhaseTrace* active = nullptr;
};

class CpuPhaseTraceFrame {
public:
    CpuPhaseTraceFrame(CpuPhaseTrace* trace, uint64_t frame) : previous_(CpuPhaseTrace::active)
    {
        CpuPhaseTrace::active = trace;
        if (trace) { trace->frame = frame; }
    }
    ~CpuPhaseTraceFrame() { CpuPhaseTrace::active = previous_; }
    CpuPhaseTraceFrame(const CpuPhaseTraceFrame&) = delete;
    CpuPhaseTraceFrame& operator=(const CpuPhaseTraceFrame&) = delete;
private:
    CpuPhaseTrace* previous_;
};

class CpuPhase {
public:
    explicit CpuPhase(const char* name, uint64_t value = 0)
        : trace_(CpuPhaseTrace::active), name_(name), value_(value)
    {
        if (trace_) { depth_ = trace_->depth++; start_ = CpuPhaseTrace::Clock::now(); }
    }
    ~CpuPhase()
    {
        if (trace_) { finish(CpuPhaseTrace::Clock::now()); --trace_->depth; }
    }
    void next(const char* name, uint64_t value = 0)
    {
        if (trace_) {
            const auto now = CpuPhaseTrace::Clock::now();
            finish(now);
            start_ = now;
            name_ = name;
            value_ = value;
        }
    }
    CpuPhase(const CpuPhase&) = delete;
    CpuPhase& operator=(const CpuPhase&) = delete;
private:
    void finish(CpuPhaseTrace::Clock::time_point end)
    {
        if (trace_->events.size() == CpuPhaseTrace::kMaxEvents) { ++trace_->dropped; return; }
        trace_->events.push_back({name_, trace_->frame,
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(start_ - trace_->origin).count()),
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count()),
            value_, depth_});
    }
    CpuPhaseTrace* trace_;
    const char* name_;
    uint64_t value_;
    uint32_t depth_ = 0;
    CpuPhaseTrace::Clock::time_point start_;
};

} // namespace metallic::render::profiling

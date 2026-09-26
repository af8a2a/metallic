#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <json.hpp>

namespace metallic::render::profiling {

// Opt-in CPU wall-clock attribution. Nested metrics must not be added together;
// worker totals are sums of parallel work, not critical-path duration.
struct SchedulingMetrics {
    bool enabled = false;
    uint64_t executeNs = 0, prepareNs = 0, scenePrepareNs = 0, frameWaitNs = 0, serialExecuteNs = 0, workerExecuteNs = 0;
    uint64_t waitNs = 0, sealNs = 0, submitNs = 0, nativeSubmitNs = 0;
    uint64_t firstSubmitNs = 0, firstPassSubmitNs = 0, recordingEndNs = 0;
    uint64_t readyDelayNs = 0, maxReadyDelayNs = 0, nativeSubmits = 0;
    uint64_t renderingScopes = 0, renderingNs = 0, maxRenderingNs = 0;
    uint64_t drawCalls = 0, dispatchCalls = 0, maxScopeDrawCalls = 0, invalidScopes = 0;

    void mergeRecording(const SchedulingMetrics& other)
    {
        workerExecuteNs += other.workerExecuteNs;
        renderingScopes += other.renderingScopes;
        renderingNs += other.renderingNs;
        maxRenderingNs = std::max(maxRenderingNs, other.maxRenderingNs);
        drawCalls += other.drawCalls;
        dispatchCalls += other.dispatchCalls;
        maxScopeDrawCalls = std::max(maxScopeDrawCalls, other.maxScopeDrawCalls);
        invalidScopes += other.invalidScopes;
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SchedulingMetrics, enabled, executeNs, prepareNs, scenePrepareNs, frameWaitNs, serialExecuteNs, workerExecuteNs,
    waitNs, sealNs, submitNs, nativeSubmitNs, firstSubmitNs, firstPassSubmitNs, recordingEndNs, readyDelayNs,
    maxReadyDelayNs, nativeSubmits, renderingScopes, renderingNs, maxRenderingNs, drawCalls, dispatchCalls,
    maxScopeDrawCalls, invalidScopes)

class SchedulingCapture {
public:
    using Clock = std::chrono::steady_clock;
    static inline thread_local SchedulingCapture* active = nullptr;
    static bool requested()
    {
        static const bool enabled = [] {
            const char* value = std::getenv("METALLIC_RENDER_SCHEDULING_DIAGNOSTICS");
            return value && std::string_view(value) == "1";
        }();
        return enabled;
    }
    explicit SchedulingCapture(SchedulingMetrics* output, Clock::time_point origin = {})
        : metrics(output), previous_(active), origin_(origin)
    {
        if (metrics) {
            metrics->enabled = true;
            if (origin_ == Clock::time_point{}) { origin_ = Clock::now(); }
        }
        active = metrics ? this : nullptr;
    }
    ~SchedulingCapture()
    {
        if (metrics && renderingCommand_) { ++metrics->invalidScopes; }
        active = previous_;
    }
    SchedulingCapture(const SchedulingCapture&) = delete;
    SchedulingCapture& operator=(const SchedulingCapture&) = delete;
    uint64_t elapsed() const
    {
        return metrics ? uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - origin_).count()) : 0;
    }
    Clock::time_point origin() const { return origin_; }
    void beginRendering(const void* command)
    {
        if (renderingCommand_) { ++metrics->invalidScopes; }
        renderingCommand_ = command;
        scopeDraws_ = 0;
        renderingBegin_ = elapsed();
    }
    void endRendering(const void* command)
    {
        if (renderingCommand_ != command) { ++metrics->invalidScopes; return; }
        const uint64_t duration = elapsed() - renderingBegin_;
        ++metrics->renderingScopes;
        metrics->renderingNs += duration;
        metrics->maxRenderingNs = std::max(metrics->maxRenderingNs, duration);
        metrics->maxScopeDrawCalls = std::max(metrics->maxScopeDrawCalls, scopeDraws_);
        renderingCommand_ = nullptr;
    }
    void draw(const void* command)
    {
        ++metrics->drawCalls;
        if (renderingCommand_ == command) { ++scopeDraws_; }
        else { ++metrics->invalidScopes; }
    }
    SchedulingMetrics* metrics;
private:
    SchedulingCapture* previous_;
    Clock::time_point origin_;
    const void* renderingCommand_ = nullptr;
    uint64_t renderingBegin_ = 0, scopeDraws_ = 0;
};

class SchedulingPhase {
public:
    explicit SchedulingPhase(uint64_t SchedulingMetrics::* field)
        : capture_(SchedulingCapture::active), field_(field), begin_(capture_ ? capture_->elapsed() : 0) {}
    ~SchedulingPhase()
    {
        if (capture_) { capture_->metrics->*field_ += capture_->elapsed() - begin_; }
    }
    SchedulingPhase(const SchedulingPhase&) = delete;
    SchedulingPhase& operator=(const SchedulingPhase&) = delete;
private:
    SchedulingCapture* capture_;
    uint64_t SchedulingMetrics::* field_;
    uint64_t begin_;
};

} // namespace metallic::render::profiling

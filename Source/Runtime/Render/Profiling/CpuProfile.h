#pragma once

#include "Runtime/Render/Profiling/RenderGraphProfile.h"
#include <chrono>
#include <string_view>

namespace metallic::render {

// Collect CPU work without command recording, GPU queries or per-page scopes.
// Publish the completed tree under the enclosing render graph scope.
struct CpuProfileRecorder {
    std::vector<RenderGraphProfileSection> sections;
    uint32_t parent = UINT32_MAX;
    void reset() { sections.clear(); parent = UINT32_MAX; }
};

class CpuProfileScope {
public:
    CpuProfileScope(CpuProfileRecorder* recorder, std::string_view name)
        : recorder_(recorder), parent_(recorder ? recorder->parent : UINT32_MAX)
    {
        next(name);
    }
    ~CpuProfileScope() { end(); }
    CpuProfileScope(const CpuProfileScope&) = delete;
    CpuProfileScope& operator=(const CpuProfileScope&) = delete;

    void end()
    {
        if (!recorder_ || index_ == UINT32_MAX) { return; }
        recorder_->sections[index_].cpuMilliseconds = std::chrono::duration<double, std::milli>(Clock::now() - begin_).count();
        recorder_->parent = parent_;
        index_ = UINT32_MAX;
    }
    void next(std::string_view name)
    {
        end();
        if (!recorder_) { return; }
        index_ = static_cast<uint32_t>(recorder_->sections.size());
        recorder_->sections.push_back({.name = std::string(name), .parent = parent_, .cpuOnly = true});
        recorder_->parent = index_;
        begin_ = Clock::now();
    }
private:
    using Clock = std::chrono::steady_clock;
    CpuProfileRecorder* recorder_ = nullptr;
    uint32_t parent_ = UINT32_MAX;
    uint32_t index_ = UINT32_MAX;
    Clock::time_point begin_;
};

} // namespace metallic::render

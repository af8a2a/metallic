#pragma once

#include <chrono>
#include <vector>

namespace metallic::render::profiling {

// Install the process-lifetime CPU marker/pacing adapter. Safe to call repeatedly.
void initializeRHIProfiling();

struct RHICPUInterval {
    const char* name;
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;
};

// Capture CPU operations on the calling thread through the installed adapter.
// Reserve the destination's capacity before capture and keep it alive until all
// operations within the capture have completed.
class RHICPUProfileCapture {
public:
    explicit RHICPUProfileCapture(std::vector<RHICPUInterval>& intervals) noexcept;
    ~RHICPUProfileCapture();
    RHICPUProfileCapture(const RHICPUProfileCapture&) = delete;
    RHICPUProfileCapture& operator=(const RHICPUProfileCapture&) = delete;

private:
    std::vector<RHICPUInterval>* previous_;
};

} // namespace metallic::render::profiling

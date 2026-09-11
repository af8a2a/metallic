#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <source_location>
#include <string>
#include <vector>

#if METALLIC_HAS_TRACY
#include <tracy/Tracy.hpp>
#define METALLIC_TRACY_CPU_SCOPE(name) ZoneScoped; ZoneName(name, std::char_traits<char>::length(name))
#define METALLIC_TRACY_FRAME_MARK() FrameMark
#else
#define METALLIC_TRACY_CPU_SCOPE(name) ((void)0)
#define METALLIC_TRACY_FRAME_MARK() ((void)0)
#endif

namespace metallic::render::profiling {

// Recording metadata stays engine-owned until its submission completes. Emitting
// complete frames together prevents cancelled recordings from leaving open Tracy
// zones, and lets the adapter reuse the editor's timestamp queries.
struct GpuProfileZone {
    std::string name;
    std::source_location location;
    int64_t cpuBegin = 0;
    int64_t cpuEnd = 0;
};

struct GpuProfileFrame {
    GpuClockCalibration calibration;
    int64_t calibrationCpuTime = 0;
    int64_t cpuBegin = 0;
    int64_t cpuEnd = 0;
    uint64_t thread = 0;
    uint64_t connection = 0;
    bool active = false;
    bool calibrated = false;
    std::vector<GpuProfileZone> zones;
};

class TracyGpuProfiler {
public:
    void beginFrame(Queue& queue, GpuProfileFrame& frame);
    void beginZone(GpuProfileFrame& frame, std::string_view name,
        std::source_location location = std::source_location::current());
    void endZone(GpuProfileFrame& frame);
    void endFrame(GpuProfileFrame& frame);
    // Query layout: frame begin/end, followed by a begin/end pair per pass.
    void publish(const GpuProfileFrame& frame, std::span<const TimestampQueryResult> timestamps,
        double timestampPeriodNanoseconds);

private:
    bool initialized_ = false;
    bool calibrated_ = false;
    bool exhausted_ = false;
    uint8_t context_ = 0;
    uint64_t connection_ = 0;
    GpuClockCalibration initialCalibration_;
    GpuClockCalibration previousCalibration_;
};

} // namespace metallic::render::profiling

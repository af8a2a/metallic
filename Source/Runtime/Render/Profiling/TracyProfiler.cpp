#include "Runtime/Render/Profiling/TracyProfiler.h"

#if METALLIC_HAS_TRACY
#include <tracy/TracyC.h>
#include <spdlog/spdlog.h>

namespace metallic::render::profiling {
namespace {

bool captureActive()
{
#ifdef TRACY_ON_DEMAND
    return tracy::GetProfiler().IsConnected();
#else
    return true;
#endif
}

uint64_t connectionId()
{
#ifdef TRACY_ON_DEMAND
    return tracy::GetProfiler().ConnectionId();
#else
    return 0;
#endif
}

void emitTime(uint8_t context, uint16_t query, uint64_t timestamp)
{
    ___tracy_emit_gpu_time_serial({static_cast<int64_t>(timestamp), query, context});
}

// Tracy's public C GPU entry points stamp events at call time. Delayed resolve
// needs the original recording time/thread, so keep the small protocol adapter
// here, isolated from the RHI and RenderGraph (see the pinned Tracy Vulkan backend).
void emitBegin(uint8_t context, uint16_t query, int64_t cpuTime, uint64_t thread,
    std::string_view name, std::source_location location)
{
    const auto source = tracy::Profiler::AllocSourceLocation(location.line(), location.file_name(),
        location.function_name(), name.data(), name.size());
    auto item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginAllocSrcLocSerial);
    tracy::MemWrite(&item->gpuZoneBegin.cpuTime, cpuTime);
    tracy::MemWrite(&item->gpuZoneBegin.thread, thread);
    tracy::MemWrite(&item->gpuZoneBegin.srcloc, source);
    tracy::MemWrite(&item->gpuZoneBegin.queryId, query);
    tracy::MemWrite(&item->gpuZoneBegin.context, context);
    tracy::Profiler::QueueSerialFinish();
}

void emitEnd(uint8_t context, uint16_t query, int64_t cpuTime, uint64_t thread)
{
    auto item = tracy::Profiler::QueueSerial();
    tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
    tracy::MemWrite(&item->gpuZoneEnd.cpuTime, cpuTime);
    tracy::MemWrite(&item->gpuZoneEnd.thread, thread);
    tracy::MemWrite(&item->gpuZoneEnd.queryId, query);
    tracy::MemWrite(&item->gpuZoneEnd.context, context);
    tracy::Profiler::QueueSerialFinish();
}

} // namespace

void TracyGpuProfiler::beginFrame(Queue& queue, GpuProfileFrame& frame)
{
    frame = {};
    if (!captureActive() || exhausted_) { return; }
    frame.active = true;
    frame.connection = connectionId();
    frame.thread = tracy::GetThreadHandle();
    frame.calibrated = queue.calibrateTimestamps(frame.calibration).has_value();
    frame.calibrationCpuTime = tracy::Profiler::GetTime();
    frame.cpuBegin = frame.calibrationCpuTime;
}

void TracyGpuProfiler::beginZone(GpuProfileFrame& frame, std::string_view name,
    std::source_location location)
{
    if (frame.active) {
        frame.zones.push_back({std::string(name), location, tracy::Profiler::GetTime(), 0});
    }
}

void TracyGpuProfiler::endZone(GpuProfileFrame& frame)
{
    if (frame.active && !frame.zones.empty()) { frame.zones.back().cpuEnd = tracy::Profiler::GetTime(); }
}

void TracyGpuProfiler::endFrame(GpuProfileFrame& frame)
{
    if (frame.active) { frame.cpuEnd = tracy::Profiler::GetTime(); }
}

void TracyGpuProfiler::publish(const GpuProfileFrame& frame,
    std::span<const TimestampQueryResult> timestamps, double timestampPeriodNanoseconds)
{
    if (!frame.active || !captureActive() || frame.connection != connectionId() || exhausted_ ||
        timestamps.size() != (frame.zones.size() + 1) * 2 || timestampPeriodNanoseconds <= 0.0) {
        return;
    }
    for (const auto& timestamp : timestamps) { if (!timestamp.available) { return; } }

    if (!initialized_) {
        // Tracy context IDs live for the process lifetime. Do not wrap after many
        // preview executors, and keep this context across graph recompiles.
        auto& counter = tracy::GetGpuCtxCounter();
        auto id = counter.load(std::memory_order_relaxed);
        do {
            if (id == UINT8_MAX) { exhausted_ = true; return; }
        } while (!counter.compare_exchange_weak(id, static_cast<uint8_t>(id + 1)));
        context_ = id;
        calibrated_ = frame.calibrated;
        auto item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuNewContext);
        tracy::MemWrite(&item->gpuNewContext.cpuTime,
            calibrated_ ? frame.calibrationCpuTime : frame.cpuBegin);
        tracy::MemWrite(&item->gpuNewContext.gpuTime, static_cast<int64_t>(
            calibrated_ ? frame.calibration.gpuTimestamp : timestamps[0].value));
        tracy::MemWrite(&item->gpuNewContext.thread, uint64_t{0});
        tracy::MemWrite(&item->gpuNewContext.period, static_cast<float>(timestampPeriodNanoseconds));
        tracy::MemWrite(&item->gpuNewContext.context, context_);
        tracy::MemWrite(&item->gpuNewContext.flags,
            static_cast<uint8_t>(calibrated_ ? tracy::GpuContextCalibration : 0));
        tracy::MemWrite(&item->gpuNewContext.type, tracy::GpuContextType::Vulkan);
#ifdef TRACY_ON_DEMAND
        tracy::GetProfiler().DeferItem(*item);
#endif
        tracy::Profiler::QueueSerialFinish();
        const std::string_view name = calibrated_
            ? "Metallic Graphics / RenderGraph" : "Metallic Graphics / RenderGraph (uncalibrated)";
        ___tracy_emit_gpu_context_name_serial({context_, name.data(), static_cast<uint16_t>(name.size())});
        previousCalibration_ = frame.calibration;
        initialCalibration_ = frame.calibration;
        connection_ = frame.connection;
        initialized_ = true;
        spdlog::info("[Tracy] GPU capture enabled ({})", calibrated_ ? "calibrated" : "uncalibrated; CPU/GPU alignment approximate");
    }
    // On reconnect the viewer receives the deferred initial context again, not
    // the calibrations from the previous capture.
    if (connection_ != frame.connection) {
        previousCalibration_ = initialCalibration_;
        connection_ = frame.connection;
    }
    if (calibrated_ && frame.calibrated &&
        frame.calibration.gpuTimestamp > previousCalibration_.gpuTimestamp &&
        frame.calibration.cpuNanoseconds > previousCalibration_.cpuNanoseconds) {
        auto item = tracy::Profiler::QueueSerial();
        tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuCalibration);
        tracy::MemWrite(&item->gpuCalibration.cpuTime, frame.calibrationCpuTime);
        tracy::MemWrite(&item->gpuCalibration.gpuTime, static_cast<int64_t>(frame.calibration.gpuTimestamp));
        tracy::MemWrite(&item->gpuCalibration.cpuDelta,
            static_cast<int64_t>(frame.calibration.cpuNanoseconds - previousCalibration_.cpuNanoseconds));
        tracy::MemWrite(&item->gpuCalibration.context, context_);
        tracy::Profiler::QueueSerialFinish();
        previousCalibration_ = frame.calibration;
    }

    emitBegin(context_, 0, frame.cpuBegin, frame.thread, "RenderGraph Frame", std::source_location::current());
    emitTime(context_, 0, timestamps[0].value);
    for (size_t index = 0; index < frame.zones.size(); ++index) {
        const auto& zone = frame.zones[index];
        // Resolve each pair immediately, so Tracy IDs are independent of native
        // query indices and cannot wrap on large graphs or reused frame slots.
        emitBegin(context_, 2, zone.cpuBegin, frame.thread, zone.name, zone.location);
        emitTime(context_, 2, timestamps[2 + index * 2].value);
        emitEnd(context_, 3, zone.cpuEnd, frame.thread);
        emitTime(context_, 3, timestamps[3 + index * 2].value);
    }
    emitEnd(context_, 1, frame.cpuEnd, frame.thread);
    emitTime(context_, 1, timestamps[1].value);
}

} // namespace metallic::render::profiling

#else
namespace metallic::render::profiling {
void TracyGpuProfiler::beginFrame(Queue&, GpuProfileFrame&) {}
void TracyGpuProfiler::beginZone(GpuProfileFrame&, std::string_view, std::source_location) {}
void TracyGpuProfiler::endZone(GpuProfileFrame&) {}
void TracyGpuProfiler::endFrame(GpuProfileFrame&) {}
void TracyGpuProfiler::publish(const GpuProfileFrame&, std::span<const TimestampQueryResult>, double) {}
} // namespace metallic::render::profiling
#endif

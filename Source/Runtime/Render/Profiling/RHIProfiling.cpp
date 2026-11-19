#include "RHIProfiling.h"
#include "NsightEvents.h"
#include "PacingTrace.h"
#include "TracyProfiler.h"
#include "Runtime/Render/GAPI/RHIEvents.h"
#include <memory>
#include <optional>
#include <utility>

namespace metallic::render::profiling {
namespace {

using Clock = std::chrono::steady_clock;
thread_local std::vector<RHICPUInterval>* activeIntervals = nullptr;

const char* operationName(RHIOperation operation)
{
    switch (operation) {
        case RHIOperation::Submit: return "Queue Submit";
        case RHIOperation::FenceWait: return "Fence Wait";
        case RHIOperation::TimelineWait: return "Timeline Wait";
        case RHIOperation::NativeSubmit: return "Native Queue Submit";
    }
    std::unreachable();
}

struct Marker {
    NsightProfileRange nvtx;
#if METALLIC_HAS_TRACY
    tracy::ScopedZone tracy;
    static const tracy::SourceLocationData* location(RHIOperation operation)
    {
        static constexpr tracy::SourceLocationData locations[] = {
            {"Queue Submit", "Queue::submit", __FILE__, __LINE__, 0},
            {"Fence Wait", "Fence::wait", __FILE__, __LINE__, 0},
            {"Timeline Wait", "Semaphore::wait", __FILE__, __LINE__, 0}};
        return &locations[static_cast<size_t>(operation)];
    }
#endif
    Marker(RHIOperation operation, uint64_t payload)
        : nvtx(NsightDomain::Render, operation == RHIOperation::Submit ? "Submit" :
            operation == RHIOperation::FenceWait ? "Fence Wait" : "Semaphore Wait",
            operation == RHIOperation::Submit ? NsightCategory::QueueSubmit : NsightCategory::FenceWait, payload)
#if METALLIC_HAS_TRACY
        , tracy(location(operation), true)
#endif
    {}
};

struct OperationState {
    OperationState(RHIOperation operation, uint64_t payload)
    {
        if (operation == RHIOperation::NativeSubmit) {
            queueFamily = static_cast<uint32_t>(payload);
            pacingTrace("QueueSubmitBegin", UINT64_MAX, queueFamily);
        } else {
            marker.emplace(operation, payload);
        }
        if (activeIntervals && activeIntervals->size() < activeIntervals->capacity()) {
            intervals = activeIntervals;
            begin = Clock::now();
        }
    }

    void finish(RHIOperation operation)
    {
        // Nested operations may have filled the reserved capacity during this scope.
        const bool capture = intervals && intervals->size() < intervals->capacity();
        const auto end = capture ? Clock::now() : Clock::time_point{};
        if (operation == RHIOperation::NativeSubmit) {
            pacingTrace("QueueSubmitEnd", UINT64_MAX, queueFamily);
        }
        marker.reset();
        if (capture) { intervals->push_back({operationName(operation), begin, end}); }
    }

    std::optional<Marker> marker;
    std::vector<RHICPUInterval>* intervals = nullptr;
    Clock::time_point begin;
    uint32_t queueFamily = 0;
};

static_assert(sizeof(OperationState) <= RHIOperationScope::storageSize);
static_assert(alignof(OperationState) <= alignof(std::max_align_t));
const RHIOperationObserver observer{
    [](RHIOperation operation, uint64_t payload, void* storage) {
        std::construct_at(static_cast<OperationState*>(storage), operation, payload);
    },
    [](RHIOperation operation, void* storage) {
        auto* state = static_cast<OperationState*>(storage);
        state->finish(operation);
        std::destroy_at(state);
    },
    pacingTrace
};
} // namespace

void initializeRHIProfiling()
{
    rhiOperationObserver.store(&observer, std::memory_order_release);
}

RHICPUProfileCapture::RHICPUProfileCapture(std::vector<RHICPUInterval>& intervals) noexcept
    : previous_(std::exchange(activeIntervals, &intervals))
{
}

RHICPUProfileCapture::~RHICPUProfileCapture()
{
    activeIntervals = previous_;
}

} // namespace metallic::render::profiling

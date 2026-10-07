#include "RHIProfiling.h"
#include "NsightEvents.h"
#include "PacingTrace.h"
#include "TracyProfiler.h"
#include "Runtime/Render/GAPI/RHIEvents.h"
#include <memory>

namespace metallic::render::profiling {
namespace {
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
static_assert(sizeof(Marker) <= RHIOperationScope::storageSize);
static_assert(alignof(Marker) <= alignof(std::max_align_t));
const RHIOperationObserver observer{
    [](RHIOperation operation, uint64_t payload, void* storage) {
        if (operation == RHIOperation::NativeSubmit) {
            std::construct_at(static_cast<uint32_t*>(storage), static_cast<uint32_t>(payload));
            pacingTrace("QueueSubmitBegin", UINT64_MAX, static_cast<uint32_t>(payload));
        } else { std::construct_at(static_cast<Marker*>(storage), operation, payload); }
    },
    [](RHIOperation operation, void* storage) {
        if (operation == RHIOperation::NativeSubmit) {
            pacingTrace("QueueSubmitEnd", UINT64_MAX, *static_cast<uint32_t*>(storage));
        } else { std::destroy_at(static_cast<Marker*>(storage)); }
    },
    pacingTrace
};
} // namespace
void initializeRHIProfiling()
{
    rhiOperationObserver.store(&observer, std::memory_order_release);
}
} // namespace metallic::render::profiling

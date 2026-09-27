#include "VulkanTrace.h"
#include <atomic>

namespace metallic::render::vulkan {
#if METALLIC_RHI_DIAGNOSTICS
namespace { std::atomic<const TraceSink*> activeSink{nullptr}; }
#endif
bool traceCompiled()
{
    return METALLIC_RHI_DIAGNOSTICS != 0;
}
bool installTraceSink(const TraceSink* sink)
{
#if METALLIC_RHI_DIAGNOSTICS
    if (!sink || !sink->callback || !sink->device) { return false; }
    const TraceSink* empty = nullptr;
    return activeSink.compare_exchange_strong(empty, sink, std::memory_order_acq_rel);
#else
    return false;
#endif
}
void removeTraceSink(const TraceSink* sink)
{
#if METALLIC_RHI_DIAGNOSTICS
    activeSink.compare_exchange_strong(sink, nullptr, std::memory_order_acq_rel);
#endif
}
#if METALLIC_RHI_DIAGNOSTICS
void emitTrace(const TraceEvent& event) noexcept
{
    const auto* sink = activeSink.load(std::memory_order_acquire);
    if (sink && sink->device == event.device) { sink->callback(sink->context, event); }
}
#endif
} // namespace metallic::render::vulkan

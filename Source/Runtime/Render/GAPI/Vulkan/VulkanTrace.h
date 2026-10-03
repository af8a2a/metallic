#pragma once
#include "Runtime/Render/GAPI/RHI.h"
#include <volk.h>

namespace metallic::render::vulkan {

enum class TraceKind { CommandBegin, Barrier, Submit, Retire };
// Borrowed native arguments, valid only during the synchronous noexcept callback.
// The observer must deep-copy data and replace native handles with logical IDs.
struct TraceEvent {
    TraceKind kind;
    VkDevice device = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    const VkDependencyInfo* dependency = nullptr;
    const BarrierDesc* requested = nullptr;
    const VkSubmitInfo2* submit = nullptr;
    VkResult result = VK_SUCCESS;
    VkObjectType objectType = VK_OBJECT_TYPE_UNKNOWN;
    uint64_t object = 0;
};
struct TraceSink {
    VkDevice device = VK_NULL_HANDLE;
    void (*callback)(void*, const TraceEvent&) noexcept = nullptr;
    void* context = nullptr;
};
// One diagnostic session per process. Install/remove only with recording/submit
// workers quiescent. The caller owns the sink through all callbacks and removal.
bool traceCompiled();
bool installTraceSink(const TraceSink* sink);
void removeTraceSink(const TraceSink* sink);
#if METALLIC_RHI_DIAGNOSTICS
void emitTrace(const TraceEvent& event) noexcept;
#else
inline void emitTrace(const TraceEvent&) noexcept {}
#endif
inline void forgetTraceObject(VkDevice device, VkObjectType type, uint64_t object) noexcept
{
    emitTrace({.kind = TraceKind::Retire, .device = device, .objectType = type, .object = object});
}
inline void recordBarrier(const VolkDeviceTable& functions, VkDevice device, VkCommandBuffer command, const VkDependencyInfo& dependency,
    const BarrierDesc* requested = nullptr)
{
    functions.vkCmdPipelineBarrier2(command, &dependency);
    emitTrace({.kind = TraceKind::Barrier, .device = device, .command = command,
        .dependency = &dependency, .requested = requested});
}

} // namespace metallic::render::vulkan

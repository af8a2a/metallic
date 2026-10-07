#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace metallic::render {

// Optional observation only: callbacks must not throw, submit, or change RHI
// state. The observer and its code must outlive all scopes using it.
enum class RHIOperation { Submit, FenceWait, TimelineWait, NativeSubmit };
struct RHIOperationObserver {
    void (*begin)(RHIOperation, uint64_t payload, void* storage);
    void (*end)(RHIOperation, void* storage);
    void (*trace)(const char* event, uint64_t frame, uint32_t value) = nullptr;
};
inline std::atomic<const RHIOperationObserver*> rhiOperationObserver{nullptr};

inline void observeRHITrace(const char* event, uint64_t frame, uint32_t value)
{
    if (const auto* observer = rhiOperationObserver.load(std::memory_order_acquire); observer && observer->trace) {
        observer->trace(event, frame, value);
    }
}

// Per-thread recording observer. Owners install/restore this with their capture
// lifetime; disabled recording does not read process-wide profiling state.
enum class RHICommandEvent { BeginRendering, EndRendering, Draw, Dispatch, SubmitBegin, SubmitEnd };
struct RHICommandObserver {
    void* context = nullptr;
    void (*notify)(void*, RHICommandEvent, const void*) = nullptr;
};
inline thread_local RHICommandObserver rhiCommandObserver;
inline void observeRHICommand(RHICommandEvent event, const void* command = nullptr)
{
    if (const auto observer = rhiCommandObserver; observer.notify) {
        observer.notify(observer.context, event, command);
    }
}

class RHIOperationScope {
public:
    static constexpr size_t storageSize = 128;
    explicit RHIOperationScope(RHIOperation operation, uint64_t payload = 0)
        : observer_(rhiOperationObserver.load(std::memory_order_acquire)), operation_(operation)
    {
        if (observer_) { observer_->begin(operation_, payload, storage_); }
    }
    ~RHIOperationScope()
    {
        if (observer_) { observer_->end(operation_, storage_); }
    }
    RHIOperationScope(const RHIOperationScope&) = delete;
    RHIOperationScope& operator=(const RHIOperationScope&) = delete;
private:
    const RHIOperationObserver* observer_;
    RHIOperation operation_;
    alignas(std::max_align_t) std::byte storage_[storageSize];
};
} // namespace metallic::render

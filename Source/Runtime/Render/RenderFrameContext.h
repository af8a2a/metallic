#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <memory>
#include <vector>

namespace metallic::render {

// Copies refer to the same one-shot recording/batch. A batch becomes waitable
// only after it is sealed, and completes when every contributing queue finishes.
class GpuCompletionPoint {
public:
    bool valid() const { return state_ != nullptr; }
    bool isSubmitted() const;
    bool isCancelled() const;
    bool isComplete() const;
    bool sameSubmission(const GpuCompletionPoint& other) const { return state_ == other.state_; }
    uint64_t value() const;
    Result wait(uint64_t timeoutNanoseconds = UINT64_MAX) const;
    // Append/coalesce timeline waits. Keep this point alive until the waiting
    // submission completes. value() is zero for a point covering multiple queues.
    Result appendWaits(std::vector<SemaphoreSubmitDesc>& waits) const;

private:
    struct State;
    std::shared_ptr<State> state_;
    friend class RenderFrameContext;
    friend class QueueSubmissionTracker;
    friend class CommandBuffer;
};

// CPU recording lifetime and GPU submission lifetime are deliberately separate.
// Device and Queue must outlive their contexts, completion points and resources.
class RenderFrameContext {
public:
    explicit RenderFrameContext(uint32_t slotIndex = 0) : slotIndex_(slotIndex) {}
    ~RenderFrameContext();
    RenderFrameContext(const RenderFrameContext&) = delete;
    RenderFrameContext& operator=(const RenderFrameContext&) = delete;

    Result begin(uint64_t frameIndex, uint64_t timeoutNanoseconds = UINT64_MAX);
    Result wait(uint64_t timeoutNanoseconds = UINT64_MAX) const;
    // Reset/discard recorded command buffers before cancelling a frame that will
    // not be submitted. Cancelled recordings must never subsequently be submitted.
    void cancel();
    // Seal a batch after its final successful segment, including partial failure.
    Result finishSubmission();
    Result reset();
    bool recording() const;
    void retain(std::shared_ptr<void> resource);
    Result addDependency(GpuCompletionPoint completion);
    uint64_t frameIndex() const { return frameIndex_; }
    uint32_t slotIndex() const { return slotIndex_; }
    const GpuCompletionPoint& completion() const { return completion_; }

private:
    uint32_t slotIndex_ = 0;
    uint64_t frameIndex_ = 0;
    GpuCompletionPoint completion_;
    std::vector<std::shared_ptr<void>> resources_;
    std::vector<GpuCompletionPoint> dependencies_;
    friend class QueueSubmissionTracker;
};

// One tracker per submitting queue. Calls are externally serialized, just like
// Queue::submit. Graphics completion does not imply presentation completion.
class QueueSubmissionTracker {
public:
    QueueSubmissionTracker() = default;
    ~QueueSubmissionTracker();
    QueueSubmissionTracker(const QueueSubmissionTracker&) = delete;
    QueueSubmissionTracker& operator=(const QueueSubmissionTracker&) = delete;
    Result initialize(Device& device, Queue& queue);
    Result submit(const QueueSubmitDesc& desc, RenderFrameContext& frame);
    // All command buffers must be recorded before the first segment is submitted.
    // The returned point covers this segment; frame.completion() covers the batch.
    Result submitSegment(const QueueSubmitDesc& desc, RenderFrameContext& frame,
        GpuCompletionPoint& completion);
    Result wait(uint64_t timeoutNanoseconds = UINT64_MAX) const;
    Result reset();

private:
    Queue* queue_ = nullptr;
    std::shared_ptr<Semaphore> timeline_;
    uint64_t nextValue_ = 1;
    GpuCompletionPoint lastSubmission_;
};

class DeferredReleaseQueue {
public:
    ~DeferredReleaseQueue();
    void retire(GpuCompletionPoint completion, std::shared_ptr<void> resource);
    void collect();
    Result drain();
    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        GpuCompletionPoint completion;
        std::shared_ptr<void> resource;
    };
    std::vector<Entry> entries_;
};

} // namespace metallic::render

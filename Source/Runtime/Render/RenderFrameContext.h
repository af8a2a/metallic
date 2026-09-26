#pragma once

#include "Runtime/Render/GAPI/Rhi.h"

#include <functional>
#include <atomic>
#include <memory>
#include <vector>

namespace metallic::render {

// A CPU publication made while recording. Submission means the queue accepted
// the commands, not that the GPU finished. Callbacks must not throw or reenter
// submission/frame lifecycle. Status reads may be concurrent; cancellation and
// queue acceptance must be serialized by their coordinator.
class SubmissionTransaction {
public:
    SubmissionTransaction(std::function<void()> submitted, std::function<void()> cancelled);
    ~SubmissionTransaction();
    SubmissionTransaction(const SubmissionTransaction&) = delete;
    SubmissionTransaction& operator=(const SubmissionTransaction&) = delete;
    bool resolved() const { return status_ != Status::Pending; }
    bool cancelled() const { return status_ == Status::Cancelled; }
    void cancel() noexcept;

private:
    enum class Status { Pending, Submitted, Cancelled };
    std::atomic<Status> status_ = Status::Pending;
    bool attached_ = false;
    std::function<void()> submitted_;
    std::function<void()> cancelled_;
    void submit() noexcept;
    friend struct detail::CommandSubmissionState;
    friend class CommandBuffer;
};

namespace detail {
struct CommandSubmissionState {
    ~CommandSubmissionState();
    bool canSubmit() const;
    void submit() noexcept;
    void cancel() noexcept;
    std::atomic<bool> submitted = false;
    std::atomic<bool> cancelled = false;
    std::atomic<bool> finished = false;
    bool sealed = false; // Coordinator only, after finished publication.
    CommandBuffer* owner = nullptr; // Invalidated by wrapper destruction/move.
    std::vector<std::shared_ptr<SubmissionTransaction>> transactions;
    std::vector<std::shared_ptr<void>> resources;
};

struct CommandSubmissionRegistry {
    void add(const std::shared_ptr<CommandSubmissionState>& recording);
    void cancel() noexcept;
    std::vector<std::weak_ptr<CommandSubmissionState>> recordings;
};
} // namespace detail

// Immutable identity during recording; a published point becomes waitable.
// A frame point is published only after its submission window closes.
class GpuCompletionPoint {
public:
    bool valid() const { return state_ != nullptr; }
    bool isSubmitted() const;
    bool isCancelled() const;
    bool isComplete() const;
    bool sameSubmission(const GpuCompletionPoint& other) const { return state_ == other.state_; }
    uint64_t value() const;
    Result<> wait(uint64_t timeoutNanoseconds = UINT64_MAX) const;
    // Append/coalesce timeline waits. Keep this point alive until the waiting
    // submission completes. value() is zero for a point covering multiple queues.
    Result<> appendWaits(std::vector<SemaphoreSubmitDesc>& waits) const;

private:
    struct State;
    std::shared_ptr<State> state_;
    friend class RenderFrameContext;
    friend class QueueSubmissionTracker;
    friend class CommandBuffer;
    friend class CommandRecordingContext;
    friend class RecordedBatch;
};

enum class FrameSubmissionMode { Joined, Pipelined };

// CPU hand-off only: seal neither submits commands nor closes the frame.
// Transaction cancellation is revalidated at acceptance, not during CPU seal.
// Command buffers/pools must remain alive and untouched through GPU completion.
class RecordedBatch {
public:
    Result<> seal(RenderFrameContext& frame, std::span<CommandBuffer* const> commands);
    bool valid() const;

private:
    RenderFrameContext* frame_ = nullptr;
    GpuCompletionPoint generation_;
    std::vector<CommandBuffer*> commands_;
    std::vector<std::shared_ptr<detail::CommandSubmissionState>> states_;
    friend class QueueSubmissionTracker;
};

// Queue acceptance is distinct from GPU completion. An empty receipt never
// exposes a reserved timeline value, including after a failed submission.
class SubmissionReceipt {
public:
    bool accepted() const { return completion_.isSubmitted(); }
    const GpuCompletionPoint& completion() const { return completion_; }
private:
    GpuCompletionPoint completion_;
    friend class QueueSubmissionTracker;
};

// CPU recording lifetime and GPU submission lifetime are deliberately separate.
// Device and Queue must outlive their contexts, completion points and resources.
// Lifecycle/retain calls belong to the coordinator. Join all recording workers
// before cancel/reset/frame seal; a sealed batch can submit while other contexts
// record. Workers retain through their CommandBuffer only.
class RenderFrameContext {
public:
    explicit RenderFrameContext(uint32_t slotIndex = 0) : slotIndex_(slotIndex) {}
    ~RenderFrameContext();
    RenderFrameContext(const RenderFrameContext&) = delete;
    RenderFrameContext& operator=(const RenderFrameContext&) = delete;

    Result<> begin(uint64_t frameIndex, uint64_t timeoutNanoseconds = UINT64_MAX,
        FrameSubmissionMode mode = FrameSubmissionMode::Joined);
    Result<> wait(uint64_t timeoutNanoseconds = UINT64_MAX) const;
    // Roll back unsubmitted recordings in reverse order. They become invalid for
    // submission and must be reset before reuse. Accepted segments remain alive.
    void cancel();
    // Close admission after all recording workers join. Already sealed batches
    // may still submit; this does not publish the aggregate GPU completion.
    Result<> sealRecording();
    // Close admission, cancel any unaccepted tail, then publish all accepted work.
    Result<> finishSubmission();
    Result<> reset();
    bool recording() const;
    // Coordinator only. Joined mode requires this before any submission.
    bool recordingsFinished() const;
    bool hasAcceptedWork() const;
    FrameSubmissionMode submissionMode() const { return submissionMode_; }
    void retain(std::shared_ptr<void> resource);
    Result<> addDependency(GpuCompletionPoint completion);
    uint64_t frameIndex() const { return frameIndex_; }
    uint32_t slotIndex() const { return slotIndex_; }
    const GpuCompletionPoint& completion() const { return completion_; }

private:
    uint32_t slotIndex_ = 0;
    uint64_t frameIndex_ = 0;
    GpuCompletionPoint completion_;
    std::atomic<bool> recordingOpen_ = false;
    FrameSubmissionMode submissionMode_ = FrameSubmissionMode::Joined;
    std::vector<std::shared_ptr<void>> resources_;
    std::vector<GpuCompletionPoint> dependencies_;
    detail::CommandSubmissionRegistry recordings_;
    friend class CommandBuffer;
    friend class QueueSubmissionTracker;
    friend class Queue;
};

// One pool per frame slot / queue / recording lane. The coordinator prepares
// command buffers in graph order, then hands the entire context to one worker.
// record() rejects overlapping ownership; reset never waits implicitly.
class CommandRecordingContext {
public:
    ~CommandRecordingContext();
    Result<> initialize(Device& device, Queue& queue);
    Result<CommandBuffer*> prepare(RenderFrameContext& frame);
    Result<> record(const std::function<Result<>()>& callback);
    Result<> reset();
    Queue* queue() const { return queue_; }

private:
    std::atomic_flag ownership_ = ATOMIC_FLAG_INIT;
    Queue* queue_ = nullptr;
    GpuCompletionPoint completion_;
    std::unique_ptr<CommandPool> pool_;
    std::vector<std::unique_ptr<CommandBuffer>> commands_;
};

// One tracker per submitting queue. Calls are externally serialized, just like
// Queue::submit. Graphics completion does not imply presentation completion.
class QueueSubmissionTracker {
public:
    QueueSubmissionTracker() = default;
    ~QueueSubmissionTracker();
    QueueSubmissionTracker(const QueueSubmissionTracker&) = delete;
    QueueSubmissionTracker& operator=(const QueueSubmissionTracker&) = delete;
    Result<> initialize(Device& device, Queue& queue);
    Result<> submit(const QueueSubmitDesc& desc, RenderFrameContext& frame);
    // synchronization contains waits/signals only. Receipts cover exactly this
    // batch; frame.completion() covers every accepted batch on every queue.
    Result<> submitBatch(const RecordedBatch& batch, const QueueSubmitDesc& synchronization,
        RenderFrameContext& frame, SubmissionReceipt& receipt);
    // Compatibility adapter: seals these commands and returns their GPU point.
    Result<> submitSegment(const QueueSubmitDesc& desc, RenderFrameContext& frame,
        GpuCompletionPoint& completion);
    Result<> wait(uint64_t timeoutNanoseconds = UINT64_MAX) const;
    Result<> reset();

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
    Result<> drain();
    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        GpuCompletionPoint completion;
        std::shared_ptr<void> resource;
    };
    std::vector<Entry> entries_;
};

} // namespace metallic::render

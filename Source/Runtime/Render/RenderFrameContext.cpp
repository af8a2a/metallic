#include "Runtime/Render/RenderFrameContext.h"

#include <algorithm>
#include <utility>

namespace metallic::render {

struct GpuCompletionPoint::State {
    enum class Status { Recording, Submitted, Cancelled };
    Status status = Status::Recording;
    std::shared_ptr<Semaphore> timeline;
    uint64_t value = 0;
};

bool GpuCompletionPoint::isSubmitted() const
{
    return state_ != nullptr && state_->status == State::Status::Submitted;
}

bool GpuCompletionPoint::isCancelled() const
{
    return state_ != nullptr && state_->status == State::Status::Cancelled;
}

bool GpuCompletionPoint::isComplete() const
{
    return state_ == nullptr || isCancelled() ||
        (isSubmitted() && state_->timeline->currentValue() >= state_->value);
}

uint64_t GpuCompletionPoint::value() const
{
    return state_ != nullptr ? state_->value : 0;
}

Result GpuCompletionPoint::wait(uint64_t timeoutNanoseconds) const
{
    if (state_ == nullptr || isCancelled()) {
        return {};
    }
    if (!isSubmitted()) {
        return makeError(Error::InvalidArgument);
    }
    return state_->timeline->wait(state_->value, timeoutNanoseconds);
}

RenderFrameContext::~RenderFrameContext()
{
    (void)reset();
}

Result RenderFrameContext::begin(uint64_t frameIndex, uint64_t timeoutNanoseconds)
{
    if (recording()) {
        return makeError(Error::InvalidArgument);
    }
    Result result = wait(timeoutNanoseconds);
    if (!result) {
        return result;
    }
    resources_.clear();
    completion_.state_ = std::make_shared<GpuCompletionPoint::State>();
    frameIndex_ = frameIndex;
    return {};
}

Result RenderFrameContext::wait(uint64_t timeoutNanoseconds) const
{
    return completion_.wait(timeoutNanoseconds);
}

bool RenderFrameContext::recording() const
{
    return completion_.state_ != nullptr &&
        completion_.state_->status == GpuCompletionPoint::State::Status::Recording;
}

void RenderFrameContext::cancel()
{
    if (recording()) {
        completion_.state_->status = GpuCompletionPoint::State::Status::Cancelled;
        resources_.clear();
    }
}

Result RenderFrameContext::reset()
{
    cancel();
    Result result = wait();
    if (!result) {
        return result;
    }
    resources_.clear();
    completion_ = {};
    return {};
}

void RenderFrameContext::retain(std::shared_ptr<void> resource)
{
    if (resource != nullptr && recording()) {
        resources_.push_back(std::move(resource));
    }
}

QueueSubmissionTracker::~QueueSubmissionTracker()
{
    (void)reset();
}

Result QueueSubmissionTracker::initialize(Device& device, Queue& queue)
{
    Result result = reset();
    if (!result) {
        return result;
    }
    std::unique_ptr<Semaphore> timeline;
    result = device.createSemaphore(timeline);
    if (result) {
        timeline_ = std::move(timeline);
        queue_ = &queue;
    }
    return result;
}

Result QueueSubmissionTracker::submit(const QueueSubmitDesc& desc, RenderFrameContext& frame)
{
    if (queue_ == nullptr || timeline_ == nullptr || !frame.recording() ||
        nextValue_ == UINT64_MAX ||
        (desc.signalSemaphoreCount != 0 && desc.signalSemaphores == nullptr) ||
        (desc.commandBufferCount != 0 && desc.commandBuffers == nullptr)) {
        return makeError(Error::InvalidArgument);
    }
    for (uint32_t index = 0; index < desc.commandBufferCount; ++index) {
        if (desc.commandBuffers[index] == nullptr || desc.commandBuffers[index]->frameContext() != &frame ||
            desc.commandBuffers[index]->frameRecording_ != frame.completion_.state_) {
            return makeError(Error::InvalidArgument);
        }
    }
    std::vector<SemaphoreSubmitDesc> signals;
    if (desc.signalSemaphoreCount != 0) {
        signals.assign(desc.signalSemaphores, desc.signalSemaphores + desc.signalSemaphoreCount);
    }
    signals.push_back(SemaphoreSubmitDesc{
        .semaphore = timeline_.get(),
        .value = nextValue_,
        .stages = PipelineStageBits::AllCommands,
    });
    QueueSubmitDesc submission = desc;
    submission.signalSemaphores = signals.data();
    submission.signalSemaphoreCount = static_cast<uint32_t>(signals.size());
    Result result = queue_->submit(submission);
    if (!result) {
        return result;
    }
    auto& state = *frame.completion_.state_;
    state.timeline = timeline_;
    state.value = nextValue_++;
    state.status = GpuCompletionPoint::State::Status::Submitted;
    lastSubmission_ = frame.completion_;
    return {};
}

Result QueueSubmissionTracker::wait(uint64_t timeoutNanoseconds) const
{
    return lastSubmission_.wait(timeoutNanoseconds);
}

Result QueueSubmissionTracker::reset()
{
    Result result = wait();
    if (result) {
        lastSubmission_ = {};
        timeline_.reset();
        queue_ = nullptr;
        nextValue_ = 1;
    }
    return result;
}

DeferredReleaseQueue::~DeferredReleaseQueue()
{
    (void)drain();
}

void DeferredReleaseQueue::retire(GpuCompletionPoint completion, std::shared_ptr<void> resource)
{
    if (resource != nullptr && !completion.isComplete()) {
        entries_.push_back(Entry{std::move(completion), std::move(resource)});
    }
}

void DeferredReleaseQueue::collect()
{
    std::erase_if(entries_, [](const Entry& entry) { return entry.completion.isComplete(); });
}

Result DeferredReleaseQueue::drain()
{
    for (const Entry& entry : entries_) {
        Result result = entry.completion.wait();
        if (!result) {
            return result;
        }
    }
    entries_.clear();
    return {};
}

} // namespace metallic::render

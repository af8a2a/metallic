#include "Runtime/Render/RenderFrameContext.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace metallic::render {

SubmissionTransaction::SubmissionTransaction(std::function<void()> submitted, std::function<void()> cancelled)
    : submitted_(std::move(submitted)), cancelled_(std::move(cancelled))
{
}

SubmissionTransaction::~SubmissionTransaction()
{
    cancel();
}

void SubmissionTransaction::submit() noexcept
{
    if (resolved()) { return; }
    status_ = Status::Submitted;
    auto callback = std::move(submitted_);
    cancelled_ = {};
    if (callback) { callback(); }
}

void SubmissionTransaction::cancel() noexcept
{
    if (resolved()) { return; }
    status_ = Status::Cancelled;
    auto callback = std::move(cancelled_);
    submitted_ = {};
    if (callback) { callback(); }
}

detail::CommandSubmissionState::~CommandSubmissionState()
{
    cancel();
}

bool detail::CommandSubmissionState::canSubmit() const
{
    return !submitted && !cancelled && std::none_of(transactions.begin(), transactions.end(),
        [](const auto& transaction) { return transaction->cancelled(); });
}

void detail::CommandSubmissionState::submit() noexcept
{
    submitted = true;
    for (const auto& transaction : transactions) { transaction->submit(); }
}

void detail::CommandSubmissionState::cancel() noexcept
{
    if (submitted || cancelled) { return; }
    cancelled = true;
    for (auto iter = transactions.rbegin(); iter != transactions.rend(); ++iter) { (*iter)->cancel(); }
}

void detail::CommandSubmissionRegistry::add(const std::shared_ptr<CommandSubmissionState>& recording)
{
    std::erase_if(recordings, [](const auto& entry) {
        const auto state = entry.lock();
        return state == nullptr || state->submitted || state->cancelled;
    });
    recordings.push_back(recording);
}

void detail::CommandSubmissionRegistry::cancel() noexcept
{
    for (auto iter = recordings.rbegin(); iter != recordings.rend(); ++iter) {
        if (auto recording = iter->lock()) { recording->cancel(); }
    }
    recordings.clear();
}

Result CommandBuffer::addSubmissionTransaction(std::shared_ptr<SubmissionTransaction> transaction)
{
    if (!recording_ || submission_ == nullptr || !submission_->canSubmit() ||
        transaction == nullptr || transaction->resolved() || transaction->attached_) {
        return makeError(Error::InvalidArgument);
    }
    transaction->attached_ = true;
    submission_->transactions.push_back(std::move(transaction));
    return {};
}

struct GpuCompletionPoint::State {
    enum class Status { Recording, Submitting, Submitted, Cancelled };
    Status status = Status::Recording;
    struct Signal {
        std::shared_ptr<Semaphore> timeline;
        uint64_t value = 0;
    };
    std::vector<Signal> signals;
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
        (isSubmitted() && std::all_of(state_->signals.begin(), state_->signals.end(),
            [](const auto& signal) { return signal.timeline->currentValue() >= signal.value; }));
}

uint64_t GpuCompletionPoint::value() const
{
    return state_ != nullptr && state_->signals.size() == 1 ? state_->signals.front().value : 0;
}

Result GpuCompletionPoint::wait(uint64_t timeoutNanoseconds) const
{
    if (state_ == nullptr || isCancelled()) {
        return {};
    }
    if (!isSubmitted()) {
        return makeError(Error::InvalidArgument);
    }
    const auto begin = std::chrono::steady_clock::now();
    for (const auto& signal : state_->signals) {
        uint64_t remaining = timeoutNanoseconds;
        if (remaining != UINT64_MAX) {
            const auto elapsed = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - begin).count());
            remaining -= std::min(remaining, elapsed);
        }
        Result result = signal.timeline->wait(signal.value, remaining);
        if (!result) { return result; }
    }
    return {};
}

Result GpuCompletionPoint::appendWaits(std::vector<SemaphoreSubmitDesc>& waits) const
{
    if (state_ == nullptr || isCancelled()) { return {}; }
    if (!isSubmitted()) { return makeError(Error::InvalidArgument); }
    for (const auto& signal : state_->signals) {
        const auto existing = std::find_if(waits.begin(), waits.end(),
            [&](const auto& wait) { return wait.semaphore == signal.timeline.get(); });
        if (existing == waits.end()) {
            waits.push_back({.semaphore = signal.timeline.get(), .value = signal.value,
                .stages = PipelineStageBits::AllCommands});
        } else {
            existing->value = std::max(existing->value, signal.value);
            existing->stages = PipelineStageBits::AllCommands;
        }
    }
    return {};
}

Result CommandBuffer::addDependency(const GpuCompletionPoint& completion)
{
    if (!recording_) { return makeError(Error::InvalidArgument); }
    Result result = completion.appendWaits(dependencyWaits_);
    if (result && completion.valid() &&
        std::find(dependencyLifetimes_.begin(), dependencyLifetimes_.end(), completion.state_) == dependencyLifetimes_.end()) {
        dependencyLifetimes_.push_back(completion.state_);
    }
    return result;
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
    dependencies_.clear();
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
    recordings_.cancel();
    if (recording()) {
        completion_.state_->status = GpuCompletionPoint::State::Status::Cancelled;
        resources_.clear();
        dependencies_.clear();
    } else if (completion_.state_ != nullptr &&
        completion_.state_->status == GpuCompletionPoint::State::Status::Submitting) {
        // Successful segments cannot be cancelled. Preserve their resources and
        // make the partial batch waitable before returning an error to the caller.
        (void)finishSubmission();
    }
}

Result RenderFrameContext::finishSubmission()
{
    if (completion_.state_ == nullptr ||
        completion_.state_->status != GpuCompletionPoint::State::Status::Submitting) {
        return makeError(Error::InvalidArgument);
    }
    recordings_.cancel();
    completion_.state_->status = GpuCompletionPoint::State::Status::Submitted;
    return {};
}

Result RenderFrameContext::reset()
{
    cancel();
    Result result = wait();
    if (!result && !hasError(result, Error::DeviceLost)) {
        return result;
    }
    // Device loss is terminal: release while Device still exists instead of
    // retrying these waits from a destructor after the owner tears Device down.
    resources_.clear();
    dependencies_.clear();
    completion_ = {};
    return result;
}

void RenderFrameContext::retain(std::shared_ptr<void> resource)
{
    if (resource != nullptr && recording()) {
        resources_.push_back(std::move(resource));
    }
}

Result RenderFrameContext::addDependency(GpuCompletionPoint completion)
{
    if (!recording() || (completion.valid() && !completion.isSubmitted() && !completion.isCancelled())) {
        return makeError(Error::InvalidArgument);
    }
    if (completion.valid() && std::none_of(dependencies_.begin(), dependencies_.end(),
            [&](const auto& point) { return point.sameSubmission(completion); })) {
        dependencies_.push_back(std::move(completion));
    }
    return {};
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
    if (!frame.recording()) { return makeError(Error::InvalidArgument); }
    GpuCompletionPoint completion;
    Result result = submitSegment(desc, frame, completion);
    return result ? frame.finishSubmission() : result;
}

Result QueueSubmissionTracker::submitSegment(const QueueSubmitDesc& desc, RenderFrameContext& frame,
    GpuCompletionPoint& completion)
{
    using State = GpuCompletionPoint::State;
    if (queue_ == nullptr || timeline_ == nullptr || frame.completion_.state_ == nullptr ||
        (frame.completion_.state_->status != State::Status::Recording &&
            frame.completion_.state_->status != State::Status::Submitting) ||
        nextValue_ == UINT64_MAX ||
        (desc.waitSemaphoreCount != 0 && desc.waitSemaphores == nullptr) ||
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
    std::vector<SemaphoreSubmitDesc> waits;
    if (desc.waitSemaphoreCount != 0) {
        waits.assign(desc.waitSemaphores, desc.waitSemaphores + desc.waitSemaphoreCount);
    }
    for (const auto& point : frame.dependencies_) {
        Result result = point.appendWaits(waits);
        if (!result) { return result; }
    }
    QueueSubmitDesc submission = desc;
    submission.waitSemaphores = waits.data();
    submission.waitSemaphoreCount = static_cast<uint32_t>(waits.size());
    submission.signalSemaphores = signals.data();
    submission.signalSemaphoreCount = static_cast<uint32_t>(signals.size());
    auto segment = std::make_shared<State>();
    segment->signals.push_back({timeline_, nextValue_});
    auto& state = *frame.completion_.state_;
    state.signals.reserve(state.signals.size() + 1);
    Result result = queue_->submit(submission);
    if (!result) {
        return result;
    }
    const auto existing = std::find_if(state.signals.begin(), state.signals.end(),
        [&](const auto& signal) { return signal.timeline == timeline_; });
    if (existing == state.signals.end()) {
        state.signals.push_back({timeline_, nextValue_});
    } else {
        existing->value = nextValue_;
    }
    ++nextValue_;
    state.status = State::Status::Submitting;
    segment->status = State::Status::Submitted;
    completion.state_ = std::move(segment);
    lastSubmission_ = completion;
    return {};
}

Result QueueSubmissionTracker::wait(uint64_t timeoutNanoseconds) const
{
    return lastSubmission_.wait(timeoutNanoseconds);
}

Result QueueSubmissionTracker::reset()
{
    Result result = wait();
    if (result || hasError(result, Error::DeviceLost)) {
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
    Result result;
    for (const Entry& entry : entries_) {
        result = entry.completion.wait();
        if (hasError(result, Error::DeviceLost)) {
            break;
        }
        if (!result) {
            return result;
        }
    }
    entries_.clear();
    return result;
}

} // namespace metallic::render

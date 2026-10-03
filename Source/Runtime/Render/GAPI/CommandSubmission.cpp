#include "Runtime/Render/GAPI/CommandSubmission.h"

#include <algorithm>
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
    auto pending = Status::Pending;
    if (!status_.compare_exchange_strong(pending, Status::Submitted)) { return; }
    auto callback = std::move(submitted_);
    cancelled_ = {};
    if (callback) { callback(); }
}

void SubmissionTransaction::cancel() noexcept
{
    auto pending = Status::Pending;
    if (!status_.compare_exchange_strong(pending, Status::Cancelled)) { return; }
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
    resources.clear();
    finished.store(true, std::memory_order_release);
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

Result<> CommandBuffer::addSubmissionTransaction(std::shared_ptr<SubmissionTransaction> transaction)
{
    if (!recording_ || submission_ == nullptr || !submission_->canSubmit() ||
        transaction == nullptr || transaction->resolved() || transaction->attached_) {
        return makeError(Error::InvalidArgument);
    }
    transaction->attached_ = true;
    submission_->transactions.push_back(std::move(transaction));
    return {};
}

Result<> CommandBuffer::retainResource(std::shared_ptr<void> resource)
{
    if (!recording_ || !submission_ || submission_->submitted || submission_->cancelled ||
        (submissionContext_ && !submissionContext_->recording()) || !resource) {
        return makeError(Error::InvalidArgument);
    }
    submission_->resources.push_back(std::move(resource));
    return {};
}

Result<> CommandBuffer::addDependency(std::span<const SemaphoreSubmitDesc> waits, std::shared_ptr<const void> lifetime)
{
    if (!recording_ || !submission_ || !submission_->canSubmit()) { return makeError(Error::InvalidArgument); }
    for (const auto& wait : waits) {
        if (!wait.semaphore) { return makeError(Error::InvalidArgument); }
    }
    for (const auto& wait : waits) {
        const auto existing = std::find_if(dependencyWaits_.begin(), dependencyWaits_.end(),
            [&](const auto& entry) { return entry.semaphore == wait.semaphore; });
        if (existing == dependencyWaits_.end()) {
            dependencyWaits_.push_back(wait);
        } else {
            existing->value = std::max(existing->value, wait.value);
            existing->stages = PipelineStageBits::AllCommands;
        }
    }
    if (lifetime && std::find(dependencyLifetimes_.begin(), dependencyLifetimes_.end(), lifetime) == dependencyLifetimes_.end()) {
        dependencyLifetimes_.push_back(std::move(lifetime));
    }
    return {};
}

} // namespace metallic::render

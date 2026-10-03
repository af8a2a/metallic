#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include <atomic>

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

// One context per recording generation, registered by the lifetime owner.
// Lifecycle calls are externally serialized. RHI invokes reserveResources before
// native submission; acceptance must not allocate, throw or reenter submission.
// Acceptance means queued, not GPU-complete. The owner supplies completion waits.
class CommandSubmissionContext {
public:
    virtual ~CommandSubmissionContext() = default;
    virtual bool recording() const = 0;
    virtual bool canSubmit(bool tracked, bool sealed) const = 0;
    virtual void registerRecording(const std::shared_ptr<detail::CommandSubmissionState>& state) = 0;
    virtual void reserveResources(size_t count) = 0;
    virtual void acceptResources(std::vector<std::shared_ptr<void>>& resources) noexcept = 0;
};

} // namespace metallic::render

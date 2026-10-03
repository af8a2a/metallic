#pragma once

#include <mutex>
#include <shared_mutex>

namespace metallic::render {

// Excludes submissions from other threads across all RHI queues. The owning
// thread can still submit, and isolation scopes may nest on that same thread.
// This only coordinates CPU submission; callers must drain GPU work separately.
// Acquire outside submit/publication callbacks; shared-to-exclusive upgrades
// are rejected. Scopes are thread-affine and cannot be moved.
class QueueSubmissionIsolation {
public:
    QueueSubmissionIsolation();
    ~QueueSubmissionIsolation();
    QueueSubmissionIsolation(const QueueSubmissionIsolation&) = delete;
    QueueSubmissionIsolation& operator=(const QueueSubmissionIsolation&) = delete;
private:
    std::unique_lock<std::shared_mutex> lock_;
};

namespace detail {

// Backend scope covering validation, native submission and ownership handoff.
// Ordinary submissions share access; this does not synchronize a VkQueue's
// own external-synchronization requirement.
class QueueSubmissionAccess {
public:
    QueueSubmissionAccess();
    ~QueueSubmissionAccess();
    QueueSubmissionAccess(const QueueSubmissionAccess&) = delete;
    QueueSubmissionAccess& operator=(const QueueSubmissionAccess&) = delete;
private:
    std::shared_lock<std::shared_mutex> lock_;
};

} // namespace detail
} // namespace metallic::render

#pragma once

#include "Runtime/Render/RenderFrameContext.h"

#include <utility>
#include <algorithm>

namespace metallic::render {

// A receipt for the next flush of a completion-tracked Streamer. The batch may
// have accepted an earlier segment while the actual copy recording was cancelled,
// so both copy submission and GPU completion are required before publication.
class StreamUploadCompletion {
public:
    bool isComplete() const
    {
        return submission_->resolved() && !submission_->cancelled() &&
            completion_.isSubmitted() && completion_.isComplete();
    }
    bool isCancelled() const
    {
        return submission_->cancelled() || completion_.isCancelled();
    }

    // Only commands recorded after this receipt's copies in the SAME recording
    // may consume them without a CPU completion round trip. Queue acceptance or
    // sharing a frame context alone is insufficient (including cancelled tails).
    bool isRecordedBefore(const CommandBuffer& commands) const
    {
        return !isCancelled() && commands.recording_ && commands.submission_ &&
            commands.submission_->canSubmit() && commands.frameContext_ &&
            completion_.sameSubmission(commands.frameContext_->completion()) &&
            std::find(commands.submission_->transactions.begin(), commands.submission_->transactions.end(),
                submission_) != commands.submission_->transactions.end();
    }

private:
    explicit StreamUploadCompletion(GpuCompletionPoint completion)
        : completion_(std::move(completion)),
          submission_(std::make_shared<SubmissionTransaction>(nullptr, nullptr))
    {
    }

    GpuCompletionPoint completion_;
    std::shared_ptr<SubmissionTransaction> submission_;
    friend struct detail::StreamerImpl;
};

} // namespace metallic::render

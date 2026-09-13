#pragma once

#include "Runtime/Render/RenderFrameContext.h"

#include <utility>

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

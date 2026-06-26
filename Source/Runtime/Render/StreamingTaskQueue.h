#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <queue>

namespace metallic::render {

static constexpr uint32_t kStreamingMaxActiveTasks = 3;
static constexpr uint32_t kInvalidStreamingTaskIndex = std::numeric_limits<uint32_t>::max();

class StreamingTaskQueue {
public:
    struct Stats {
        uint32_t availableTaskCount = kStreamingMaxActiveTasks;
        uint32_t queuedTaskCount = 0;
        uint32_t acquiredTaskCount = 0;
        uint32_t frontTaskIndex = kInvalidStreamingTaskIndex;
        uint32_t frontDependentIndex = kInvalidStreamingTaskIndex;
        uint64_t frontCompletionFrameIndex = 0;
        bool acquisitionBlocked = false;
    };

    StreamingTaskQueue() = default;

    void reset()
    {
        availableTaskBits_ = (1u << kStreamingMaxActiveTasks) - 1u;
        tasks_ = {};
    }

    uint32_t acquireTaskIndex()
    {
        for (uint32_t index = 0; index < kStreamingMaxActiveTasks; ++index) {
            const uint32_t bit = 1u << index;
            if ((availableTaskBits_ & bit) != 0) {
                availableTaskBits_ &= ~bit;
                return index;
            }
        }
        return kInvalidStreamingTaskIndex;
    }

    void releaseTaskIndex(uint32_t index)
    {
        assert(index < kStreamingMaxActiveTasks);
        const uint32_t bit = 1u << index;
        assert((availableTaskBits_ & bit) == 0);
        availableTaskBits_ |= bit;
    }

    bool canPop(uint64_t completedFrameIndex, bool ensureAcquisition) const
    {
        if (tasks_.empty()) {
            return false;
        }

        // The reference implementation may block on a timeline semaphore when
        // ensureAcquisition is true. Metallic's prototype uses frame indices, so
        // it reports readiness without blocking the render thread.
        (void)ensureAcquisition;
        return tasks_.front().completionFrameIndex <= completedFrameIndex;
    }

    void push(
        uint32_t taskIndex,
        uint64_t completionFrameIndex,
        uint32_t dependentIndex = kInvalidStreamingTaskIndex)
    {
        assert(taskIndex < kStreamingMaxActiveTasks);
        tasks_.push(Task{
            .completionFrameIndex = completionFrameIndex,
            .taskIndex = taskIndex,
            .dependentIndex = dependentIndex,
        });
    }

    uint32_t pop()
    {
        uint32_t dependentIndex = kInvalidStreamingTaskIndex;
        return popWithDependent(dependentIndex);
    }

    uint32_t popWithDependent(uint32_t& dependentIndex)
    {
        assert(!tasks_.empty());
        const Task task = tasks_.front();
        tasks_.pop();
        dependentIndex = task.dependentIndex;
        assert(task.taskIndex != kInvalidStreamingTaskIndex);
        return task.taskIndex;
    }

    bool empty() const { return tasks_.empty(); }
    uint32_t queuedTaskCount() const { return static_cast<uint32_t>(tasks_.size()); }
    uint32_t availableTaskCount() const { return countAvailableTasks(); }
    uint32_t acquiredTaskCount() const { return kStreamingMaxActiveTasks - countAvailableTasks(); }
    bool acquisitionBlocked() const { return availableTaskBits_ == 0 && !tasks_.empty(); }
    uint32_t frontTaskIndex() const { return tasks_.empty() ? kInvalidStreamingTaskIndex : tasks_.front().taskIndex; }
    uint32_t frontDependentIndex() const
    {
        return tasks_.empty() ? kInvalidStreamingTaskIndex : tasks_.front().dependentIndex;
    }
    uint64_t frontCompletionFrameIndex() const
    {
        return tasks_.empty() ? 0 : tasks_.front().completionFrameIndex;
    }

    Stats stats() const
    {
        return Stats{
            .availableTaskCount = availableTaskCount(),
            .queuedTaskCount = queuedTaskCount(),
            .acquiredTaskCount = acquiredTaskCount(),
            .frontTaskIndex = frontTaskIndex(),
            .frontDependentIndex = frontDependentIndex(),
            .frontCompletionFrameIndex = frontCompletionFrameIndex(),
            .acquisitionBlocked = acquisitionBlocked(),
        };
    }

private:
    struct Task {
        uint64_t completionFrameIndex = 0;
        uint32_t taskIndex = kInvalidStreamingTaskIndex;
        uint32_t dependentIndex = kInvalidStreamingTaskIndex;
    };

    uint32_t countAvailableTasks() const
    {
        uint32_t count = 0;
        for (uint32_t index = 0; index < kStreamingMaxActiveTasks; ++index) {
            if ((availableTaskBits_ & (1u << index)) != 0) {
                ++count;
            }
        }
        return count;
    }

    uint32_t availableTaskBits_ = (1u << kStreamingMaxActiveTasks) - 1u;
    std::queue<Task> tasks_;
};

} // namespace metallic::render

#pragma once

#include "Runtime/Task/TaskGraph.h"

#include <cstdint>
#include <expected>
#include <memory>

namespace metallic::task {

struct TaskSystemDesc {
    uint32_t workerCount = 0;
};

struct TaskNodeEvent {
    uint64_t executionId = 0;
    uint64_t graphId = 0;
    TaskNodeSnapshot node;
    TaskClock::time_point timestamp{};
};

class ITaskEventSink {
public:
    virtual ~ITaskEventSink() = default;

    virtual void onGraphSubmitted(const TaskGraphSnapshot& snapshot) = 0;
    virtual void onTaskStateChanged(const TaskNodeEvent& event) = 0;
    virtual void onGraphCompleted(const TaskGraphSnapshot& snapshot) = 0;
};

using TaskEventSinkToken = uint64_t;

class TaskGraphRun {
public:
    TaskGraphRun() = default;

    bool valid() const;
    bool isComplete() const;
    bool requestStop();
    TaskGraphSnapshot snapshot() const;
    std::expected<TaskGraphSnapshot, TaskError> wait() const;

private:
    struct State;
    explicit TaskGraphRun(std::shared_ptr<State> state);

    std::shared_ptr<State> state_;

    friend class TaskSystem;
};

class TaskSystem {
public:
    ~TaskSystem();

    TaskSystem(const TaskSystem&) = delete;
    TaskSystem& operator=(const TaskSystem&) = delete;
    TaskSystem(TaskSystem&&) noexcept = delete;
    TaskSystem& operator=(TaskSystem&&) noexcept = delete;

    std::expected<TaskGraphRun, TaskError> submit(TaskGraph&& graph);

    TaskEventSinkToken subscribe(std::shared_ptr<ITaskEventSink> sink);
    bool unsubscribe(TaskEventSinkToken token);

    uint32_t workerCount() const;
    bool acceptingTasks() const;

private:
    explicit TaskSystem(const TaskSystemDesc& desc);

    void shutdown();

    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend std::expected<void, TaskError> initializeTaskSystem(const TaskSystemDesc& desc);
    friend void shutdownTaskSystem();
};

std::expected<void, TaskError> initializeTaskSystem(const TaskSystemDesc& desc = {});
TaskSystem* tryGetTaskSystem() noexcept;
TaskSystem& taskSystem();
void shutdownTaskSystem();

namespace detail {

std::shared_ptr<TaskSystem> tryAcquireTaskSystem() noexcept;

} // namespace detail

} // namespace metallic::task

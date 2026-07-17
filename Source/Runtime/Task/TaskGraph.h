#pragma once

#include <chrono>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace metallic::task {

using TaskClock = std::chrono::steady_clock;
using TaskOutcome = std::expected<void, std::string>;

enum class TaskState : uint8_t {
    Pending,
    Scheduled,
    Running,
    Succeeded,
    Failed,
    Skipped,
    Cancelled,
};

enum class TaskGraphStatus : uint8_t {
    Running,
    Succeeded,
    Failed,
    Cancelled,
};

enum class TaskErrorCode : uint8_t {
    InvalidGraph,
    NotInitialized,
    AlreadyInitialized,
    ShuttingDown,
    WaitFromTask,
    InternalFailure,
};

struct TaskError {
    TaskErrorCode code = TaskErrorCode::InternalFailure;
    std::string message;
};

struct TaskNodeHandle {
    uint64_t graphId = 0;
    uint32_t nodeIndex = UINT32_MAX;

    bool valid() const { return graphId != 0 && nodeIndex != UINT32_MAX; }
    bool operator==(const TaskNodeHandle&) const = default;
};

struct TaskDesc {
    std::string name;
    std::string category;
    uint32_t color = 0;
    uint64_t userTag = 0;
};

struct TaskGraphEdge {
    TaskNodeHandle prerequisite;
    TaskNodeHandle dependent;
};

struct TaskNodeSnapshot {
    TaskNodeHandle handle;
    TaskDesc desc;
    TaskState state = TaskState::Pending;
    TaskClock::time_point scheduledTime{};
    TaskClock::time_point startTime{};
    TaskClock::time_point finishTime{};
    uint64_t workerThreadId = 0;
    std::string error;
};

struct TaskGraphSnapshot {
    uint64_t executionId = 0;
    uint64_t graphId = 0;
    std::string name;
    TaskGraphStatus status = TaskGraphStatus::Running;
    TaskClock::time_point submittedTime{};
    TaskClock::time_point completedTime{};
    std::vector<TaskNodeSnapshot> nodes;
    std::vector<TaskGraphEdge> edges;
};

class TaskContext {
public:
    bool stopRequested() const;
    uint64_t executionId() const;
    TaskNodeHandle node() const;

private:
    struct Impl;
    explicit TaskContext(Impl* impl);

    Impl* impl_ = nullptr;

    friend struct detail_TaskExecutionState;
};

class TaskGraph {
public:
    explicit TaskGraph(std::string name);
    ~TaskGraph();

    TaskGraph(TaskGraph&&) noexcept;
    TaskGraph& operator=(TaskGraph&&) noexcept;

    TaskGraph(const TaskGraph&) = delete;
    TaskGraph& operator=(const TaskGraph&) = delete;

    template <typename Callback>
    TaskNodeHandle addTask(TaskDesc desc, Callback&& callback)
    {
        using CallbackType = std::decay_t<Callback>;
        auto wrapper = [callable = CallbackType(std::forward<Callback>(callback))](TaskContext& context) mutable
            -> TaskOutcome {
            if constexpr (std::is_invocable_v<CallbackType&, TaskContext&>) {
                using Result = std::invoke_result_t<CallbackType&, TaskContext&>;
                if constexpr (std::is_same_v<Result, TaskOutcome>) {
                    return std::invoke(callable, context);
                } else {
                    static_assert(std::is_same_v<Result, void>,
                        "Task callbacks must return void or metallic::task::TaskOutcome");
                    std::invoke(callable, context);
                    return {};
                }
            } else {
                static_assert(std::is_invocable_v<CallbackType&>,
                    "Task callbacks must be invocable with no arguments or TaskContext&");
                using Result = std::invoke_result_t<CallbackType&>;
                if constexpr (std::is_same_v<Result, TaskOutcome>) {
                    return std::invoke(callable);
                } else {
                    static_assert(std::is_same_v<Result, void>,
                        "Task callbacks must return void or metallic::task::TaskOutcome");
                    std::invoke(callable);
                    return {};
                }
            }
        };
        return addTaskImpl(std::move(desc), std::move(wrapper));
    }

    std::expected<void, TaskError> addDependency(
        TaskNodeHandle prerequisite,
        TaskNodeHandle dependent);

    uint64_t id() const;
    const std::string& name() const;

private:
    using TaskFunction = std::move_only_function<TaskOutcome(TaskContext&)>;

    TaskNodeHandle addTaskImpl(TaskDesc desc, TaskFunction callback);

    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend class TaskSystem;
};

const char* taskStateName(TaskState state);
const char* taskGraphStatusName(TaskGraphStatus status);

} // namespace metallic::task

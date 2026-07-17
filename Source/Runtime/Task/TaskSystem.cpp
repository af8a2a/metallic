#include "Runtime/Task/TaskSystem.h"

#include <exec/async_scope.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace metallic::task {
namespace {

std::atomic_uint64_t gNextGraphId{1};
std::mutex gTaskSystemMutex;
std::shared_ptr<TaskSystem> gTaskSystem;
bool gTaskSystemShuttingDown = false;
thread_local bool gInsideTaskCallback = false;

TaskError makeTaskError(TaskErrorCode code, std::string message)
{
    return TaskError{
        .code = code,
        .message = std::move(message),
    };
}

uint64_t currentThreadId()
{
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

bool terminalState(TaskState state)
{
    return state == TaskState::Succeeded ||
        state == TaskState::Failed ||
        state == TaskState::Skipped ||
        state == TaskState::Cancelled;
}

class TaskCallbackScope {
public:
    TaskCallbackScope()
        : previous_(gInsideTaskCallback)
    {
        gInsideTaskCallback = true;
    }

    ~TaskCallbackScope()
    {
        gInsideTaskCallback = previous_;
    }

private:
    bool previous_ = false;
};

} // namespace

struct TaskContext::Impl {
    std::stop_token stopToken;
    uint64_t executionId = 0;
    TaskNodeHandle node;
};

struct TaskGraph::Impl {
    struct Node {
        TaskDesc desc;
        TaskFunction callback;
    };

    uint64_t id = gNextGraphId.fetch_add(1, std::memory_order_relaxed);
    std::string name;
    std::vector<Node> nodes;
    std::vector<TaskGraphEdge> edges;
};

struct TaskSystemBackend;

struct RuntimeTaskNode {
    RuntimeTaskNode(
        TaskNodeHandle handle,
        TaskDesc desc,
        std::move_only_function<TaskOutcome(TaskContext&)> callback,
        uint32_t dependencyCount)
        : callback(std::move(callback))
        , remainingDependencies(dependencyCount)
    {
        snapshot.handle = handle;
        snapshot.desc = std::move(desc);
    }

    TaskNodeSnapshot snapshot;
    std::move_only_function<TaskOutcome(TaskContext&)> callback;
    std::vector<uint32_t> successors;
    std::atomic_uint32_t remainingDependencies{0};
    std::atomic_uint32_t blockedDependencies{0};
};

struct detail_TaskExecutionState : std::enable_shared_from_this<detail_TaskExecutionState> {
    detail_TaskExecutionState(
        TaskSystemBackend& backend,
        uint64_t executionId,
        uint64_t graphId,
        std::string graphName,
        std::vector<std::shared_ptr<RuntimeTaskNode>> nodes,
        std::vector<TaskGraphEdge> edges);

    void start();
    void scheduleNode(uint32_t nodeIndex);
    void runNode(uint32_t nodeIndex) noexcept;
    void finishNode(uint32_t nodeIndex, TaskState state, std::string error = {});
    void finishGraph();
    bool requestStop();
    bool isComplete() const;
    void waitBlocking() const;
    TaskGraphSnapshot snapshot() const;

    TaskSystemBackend* backend = nullptr;
    uint64_t executionId = 0;
    uint64_t graphId = 0;
    std::string graphName;
    std::vector<std::shared_ptr<RuntimeTaskNode>> nodes;
    std::vector<TaskGraphEdge> edges;
    std::stop_source stopSource;
    std::atomic_uint32_t remainingNodes{0};
    std::atomic_uint32_t failedNodes{0};
    std::atomic_uint32_t cancelledNodes{0};
    std::atomic_bool complete{false};
    mutable std::mutex mutex;
    mutable std::condition_variable condition;
    TaskGraphStatus status = TaskGraphStatus::Running;
    TaskClock::time_point submittedTime = TaskClock::now();
    TaskClock::time_point completedTime{};
};

struct TaskGraphRun::State final : detail_TaskExecutionState {
    using detail_TaskExecutionState::detail_TaskExecutionState;
};

struct TaskSystemBackend {
    explicit TaskSystemBackend(uint32_t requestedWorkerCount)
    {
        const uint32_t hardwareThreads = std::max(std::thread::hardware_concurrency(), 1u);
        workerCount = requestedWorkerCount == 0
            ? std::max(hardwareThreads - 1u, 1u)
            : requestedWorkerCount;
        pool = std::make_unique<exec::static_thread_pool>(workerCount);
    }

    bool schedule(const std::shared_ptr<detail_TaskExecutionState>& state, uint32_t nodeIndex)
    {
        if (!accepting.load(std::memory_order_acquire) || pool == nullptr) {
            return false;
        }

        try {
            const uint32_t workerIndex = nextWorkerIndex.fetch_add(1, std::memory_order_relaxed) % workerCount;
            auto sender = stdexec::schedule(pool->get_scheduler_on_thread(workerIndex)) |
                stdexec::then([state, nodeIndex]() noexcept {
                    state->runNode(nodeIndex);
                });
            scope.spawn(std::move(sender));
            return true;
        } catch (...) {
            return false;
        }
    }

    void addExecution(const std::shared_ptr<detail_TaskExecutionState>& state)
    {
        std::lock_guard lock(activeMutex);
        activeExecutions.emplace(state->executionId, state);
    }

    void removeExecution(uint64_t executionId)
    {
        std::lock_guard lock(activeMutex);
        activeExecutions.erase(executionId);
    }

    std::vector<std::shared_ptr<detail_TaskExecutionState>> activeSnapshot()
    {
        std::vector<std::shared_ptr<detail_TaskExecutionState>> result;
        std::lock_guard lock(activeMutex);
        result.reserve(activeExecutions.size());
        for (const auto& [id, execution] : activeExecutions) {
            (void)id;
            result.push_back(execution);
        }
        return result;
    }

    std::vector<std::shared_ptr<ITaskEventSink>> sinkSnapshot()
    {
        std::vector<std::shared_ptr<ITaskEventSink>> result;
        std::lock_guard lock(sinkMutex);
        result.reserve(sinks.size());
        for (const auto& [token, sink] : sinks) {
            (void)token;
            result.push_back(sink);
        }
        return result;
    }

    void emitGraphSubmitted(const TaskGraphSnapshot& snapshot)
    {
        for (const std::shared_ptr<ITaskEventSink>& sink : sinkSnapshot()) {
            try {
                sink->onGraphSubmitted(snapshot);
            } catch (...) {
            }
        }
    }

    void emitTaskStateChanged(const TaskNodeEvent& event)
    {
        for (const std::shared_ptr<ITaskEventSink>& sink : sinkSnapshot()) {
            try {
                sink->onTaskStateChanged(event);
            } catch (...) {
            }
        }
    }

    void emitGraphCompleted(const TaskGraphSnapshot& snapshot)
    {
        for (const std::shared_ptr<ITaskEventSink>& sink : sinkSnapshot()) {
            try {
                sink->onGraphCompleted(snapshot);
            } catch (...) {
            }
        }
    }

    void shutdown()
    {
        if (!accepting.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        const std::vector<std::shared_ptr<detail_TaskExecutionState>> executions = activeSnapshot();
        for (const std::shared_ptr<detail_TaskExecutionState>& execution : executions) {
            execution->requestStop();
        }
        for (const std::shared_ptr<detail_TaskExecutionState>& execution : executions) {
            execution->waitBlocking();
        }

        stdexec::sync_wait(scope.on_empty());
        pool.reset();

        std::lock_guard lock(sinkMutex);
        sinks.clear();
    }

    uint32_t workerCount = 0;
    std::unique_ptr<exec::static_thread_pool> pool;
    exec::async_scope scope;
    std::atomic_bool accepting{true};
    std::atomic_uint64_t nextExecutionId{1};
    std::atomic_uint64_t nextSinkToken{1};
    std::atomic_uint32_t nextWorkerIndex{0};
    std::mutex activeMutex;
    std::unordered_map<uint64_t, std::shared_ptr<detail_TaskExecutionState>> activeExecutions;
    std::mutex sinkMutex;
    std::unordered_map<TaskEventSinkToken, std::shared_ptr<ITaskEventSink>> sinks;
};

struct TaskSystem::Impl {
    explicit Impl(uint32_t workerCount)
        : backend(workerCount)
    {
    }

    TaskSystemBackend backend;
};

detail_TaskExecutionState::detail_TaskExecutionState(
    TaskSystemBackend& taskBackend,
    uint64_t newExecutionId,
    uint64_t newGraphId,
    std::string newGraphName,
    std::vector<std::shared_ptr<RuntimeTaskNode>> newNodes,
    std::vector<TaskGraphEdge> newEdges)
    : backend(&taskBackend)
    , executionId(newExecutionId)
    , graphId(newGraphId)
    , graphName(std::move(newGraphName))
    , nodes(std::move(newNodes))
    , edges(std::move(newEdges))
    , remainingNodes(static_cast<uint32_t>(nodes.size()))
{
}

void detail_TaskExecutionState::start()
{
    backend->emitGraphSubmitted(snapshot());

    if (nodes.empty()) {
        finishGraph();
        return;
    }

    for (uint32_t nodeIndex = 0; nodeIndex < nodes.size(); ++nodeIndex) {
        if (nodes[nodeIndex]->remainingDependencies.load(std::memory_order_relaxed) == 0) {
            scheduleNode(nodeIndex);
        }
    }
}

void detail_TaskExecutionState::scheduleNode(uint32_t nodeIndex)
{
    const std::shared_ptr<RuntimeTaskNode>& node = nodes[nodeIndex];
    if (stopSource.stop_requested()) {
        finishNode(nodeIndex, TaskState::Cancelled, "task graph stop requested");
        return;
    }
    if (node->blockedDependencies.load(std::memory_order_acquire) != 0) {
        finishNode(nodeIndex, TaskState::Skipped, "prerequisite task did not succeed");
        return;
    }

    TaskNodeEvent event;
    {
        std::lock_guard lock(mutex);
        if (node->snapshot.state != TaskState::Pending) {
            return;
        }
        node->snapshot.state = TaskState::Scheduled;
        node->snapshot.scheduledTime = TaskClock::now();
        event = TaskNodeEvent{
            .executionId = executionId,
            .graphId = graphId,
            .node = node->snapshot,
            .timestamp = node->snapshot.scheduledTime,
        };
    }
    backend->emitTaskStateChanged(event);

    if (!backend->schedule(shared_from_this(), nodeIndex)) {
        finishNode(nodeIndex, TaskState::Failed, "TaskSystem rejected scheduled task");
    }
}

void detail_TaskExecutionState::runNode(uint32_t nodeIndex) noexcept
{
    const std::shared_ptr<RuntimeTaskNode>& node = nodes[nodeIndex];
    if (stopSource.stop_requested()) {
        finishNode(nodeIndex, TaskState::Cancelled, "task graph stop requested");
        return;
    }

    TaskNodeEvent event;
    {
        std::lock_guard lock(mutex);
        if (node->snapshot.state != TaskState::Scheduled) {
            return;
        }
        node->snapshot.state = TaskState::Running;
        node->snapshot.startTime = TaskClock::now();
        node->snapshot.workerThreadId = currentThreadId();
        event = TaskNodeEvent{
            .executionId = executionId,
            .graphId = graphId,
            .node = node->snapshot,
            .timestamp = node->snapshot.startTime,
        };
    }
    backend->emitTaskStateChanged(event);

    TaskState finalState = TaskState::Succeeded;
    std::string error;
    TaskContext::Impl contextImpl{
        .stopToken = stopSource.get_token(),
        .executionId = executionId,
        .node = node->snapshot.handle,
    };
    TaskContext context(&contextImpl);
    try {
        TaskCallbackScope callbackScope;
        TaskOutcome outcome = node->callback(context);
        if (!outcome) {
            finalState = TaskState::Failed;
            error = std::move(outcome.error());
        }
    } catch (const std::exception& exception) {
        finalState = TaskState::Failed;
        error = exception.what();
    } catch (...) {
        finalState = TaskState::Failed;
        error = "task callback threw an unknown exception";
    }

    node->callback = nullptr;
    finishNode(nodeIndex, finalState, std::move(error));
}

void detail_TaskExecutionState::finishNode(uint32_t nodeIndex, TaskState finalState, std::string error)
{
    assert(terminalState(finalState));
    const std::shared_ptr<RuntimeTaskNode>& node = nodes[nodeIndex];

    TaskNodeEvent event;
    {
        std::lock_guard lock(mutex);
        if (terminalState(node->snapshot.state)) {
            return;
        }
        node->snapshot.state = finalState;
        node->snapshot.finishTime = TaskClock::now();
        if (node->snapshot.workerThreadId == 0 && finalState != TaskState::Skipped) {
            node->snapshot.workerThreadId = currentThreadId();
        }
        node->snapshot.error = std::move(error);
        event = TaskNodeEvent{
            .executionId = executionId,
            .graphId = graphId,
            .node = node->snapshot,
            .timestamp = node->snapshot.finishTime,
        };
    }

    if (finalState == TaskState::Failed) {
        failedNodes.fetch_add(1, std::memory_order_relaxed);
    } else if (finalState == TaskState::Cancelled) {
        cancelledNodes.fetch_add(1, std::memory_order_relaxed);
    }
    backend->emitTaskStateChanged(event);

    const bool blocksDependents = finalState != TaskState::Succeeded;
    for (uint32_t successorIndex : node->successors) {
        const std::shared_ptr<RuntimeTaskNode>& successor = nodes[successorIndex];
        if (blocksDependents) {
            successor->blockedDependencies.fetch_add(1, std::memory_order_relaxed);
        }
        if (successor->remainingDependencies.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            scheduleNode(successorIndex);
        }
    }

    if (remainingNodes.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        finishGraph();
    }
}

void detail_TaskExecutionState::finishGraph()
{
    TaskGraphSnapshot completedSnapshot;
    {
        std::lock_guard lock(mutex);
        if (complete.load(std::memory_order_relaxed)) {
            return;
        }
        if (failedNodes.load(std::memory_order_relaxed) != 0) {
            status = TaskGraphStatus::Failed;
        } else if (stopSource.stop_requested() ||
                   cancelledNodes.load(std::memory_order_relaxed) != 0) {
            status = TaskGraphStatus::Cancelled;
        } else {
            status = TaskGraphStatus::Succeeded;
        }
        completedTime = TaskClock::now();
        complete.store(true, std::memory_order_release);

        completedSnapshot.executionId = executionId;
        completedSnapshot.graphId = graphId;
        completedSnapshot.name = graphName;
        completedSnapshot.status = status;
        completedSnapshot.submittedTime = submittedTime;
        completedSnapshot.completedTime = completedTime;
        completedSnapshot.edges = edges;
        completedSnapshot.nodes.reserve(nodes.size());
        for (const std::shared_ptr<RuntimeTaskNode>& node : nodes) {
            completedSnapshot.nodes.push_back(node->snapshot);
        }
    }

    condition.notify_all();
    backend->emitGraphCompleted(completedSnapshot);
    backend->removeExecution(executionId);
}

bool detail_TaskExecutionState::requestStop()
{
    if (complete.load(std::memory_order_acquire)) {
        return false;
    }
    return stopSource.request_stop();
}

bool detail_TaskExecutionState::isComplete() const
{
    return complete.load(std::memory_order_acquire);
}

void detail_TaskExecutionState::waitBlocking() const
{
    std::unique_lock lock(mutex);
    condition.wait(lock, [this] {
        return complete.load(std::memory_order_acquire);
    });
}

TaskGraphSnapshot detail_TaskExecutionState::snapshot() const
{
    std::lock_guard lock(mutex);
    TaskGraphSnapshot result{
        .executionId = executionId,
        .graphId = graphId,
        .name = graphName,
        .status = status,
        .submittedTime = submittedTime,
        .completedTime = completedTime,
        .edges = edges,
    };
    result.nodes.reserve(nodes.size());
    for (const std::shared_ptr<RuntimeTaskNode>& node : nodes) {
        result.nodes.push_back(node->snapshot);
    }
    return result;
}

TaskContext::TaskContext(Impl* impl)
    : impl_(impl)
{
}

bool TaskContext::stopRequested() const
{
    return impl_ != nullptr && impl_->stopToken.stop_requested();
}

uint64_t TaskContext::executionId() const
{
    return impl_ != nullptr ? impl_->executionId : 0;
}

TaskNodeHandle TaskContext::node() const
{
    return impl_ != nullptr ? impl_->node : TaskNodeHandle{};
}

TaskGraph::TaskGraph(std::string name)
    : impl_(std::make_unique<Impl>())
{
    impl_->name = std::move(name);
}

TaskGraph::~TaskGraph() = default;
TaskGraph::TaskGraph(TaskGraph&&) noexcept = default;
TaskGraph& TaskGraph::operator=(TaskGraph&&) noexcept = default;

TaskNodeHandle TaskGraph::addTaskImpl(TaskDesc desc, TaskFunction callback)
{
    if (impl_ == nullptr) {
        return {};
    }
    const uint32_t nodeIndex = static_cast<uint32_t>(impl_->nodes.size());
    impl_->nodes.push_back(Impl::Node{
        .desc = std::move(desc),
        .callback = std::move(callback),
    });
    return TaskNodeHandle{
        .graphId = impl_->id,
        .nodeIndex = nodeIndex,
    };
}

std::expected<void, TaskError> TaskGraph::addDependency(
    TaskNodeHandle prerequisite,
    TaskNodeHandle dependent)
{
    if (impl_ == nullptr ||
        prerequisite.graphId != impl_->id ||
        dependent.graphId != impl_->id ||
        prerequisite.nodeIndex >= impl_->nodes.size() ||
        dependent.nodeIndex >= impl_->nodes.size()) {
        return std::unexpected(makeTaskError(
            TaskErrorCode::InvalidGraph,
            "task dependency contains an invalid or cross-graph node handle"));
    }
    if (prerequisite == dependent) {
        return std::unexpected(makeTaskError(
            TaskErrorCode::InvalidGraph,
            "task cannot depend on itself"));
    }
    const auto duplicate = std::find_if(
        impl_->edges.begin(),
        impl_->edges.end(),
        [prerequisite, dependent](const TaskGraphEdge& edge) {
            return edge.prerequisite == prerequisite && edge.dependent == dependent;
        });
    if (duplicate != impl_->edges.end()) {
        return std::unexpected(makeTaskError(
            TaskErrorCode::InvalidGraph,
            "duplicate task dependency"));
    }
    impl_->edges.push_back(TaskGraphEdge{
        .prerequisite = prerequisite,
        .dependent = dependent,
    });
    return {};
}

uint64_t TaskGraph::id() const
{
    return impl_ != nullptr ? impl_->id : 0;
}

const std::string& TaskGraph::name() const
{
    static const std::string kEmpty;
    return impl_ != nullptr ? impl_->name : kEmpty;
}

const char* taskStateName(TaskState state)
{
    switch (state) {
    case TaskState::Pending: return "Pending";
    case TaskState::Scheduled: return "Scheduled";
    case TaskState::Running: return "Running";
    case TaskState::Succeeded: return "Succeeded";
    case TaskState::Failed: return "Failed";
    case TaskState::Skipped: return "Skipped";
    case TaskState::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

const char* taskGraphStatusName(TaskGraphStatus status)
{
    switch (status) {
    case TaskGraphStatus::Running: return "Running";
    case TaskGraphStatus::Succeeded: return "Succeeded";
    case TaskGraphStatus::Failed: return "Failed";
    case TaskGraphStatus::Cancelled: return "Cancelled";
    }
    return "Unknown";
}

TaskGraphRun::TaskGraphRun(std::shared_ptr<State> state)
    : state_(std::move(state))
{
}

bool TaskGraphRun::valid() const
{
    return state_ != nullptr;
}

bool TaskGraphRun::isComplete() const
{
    return state_ != nullptr && state_->isComplete();
}

bool TaskGraphRun::requestStop()
{
    return state_ != nullptr && state_->requestStop();
}

TaskGraphSnapshot TaskGraphRun::snapshot() const
{
    return state_ != nullptr ? state_->snapshot() : TaskGraphSnapshot{};
}

std::expected<TaskGraphSnapshot, TaskError> TaskGraphRun::wait() const
{
    if (state_ == nullptr) {
        return std::unexpected(makeTaskError(TaskErrorCode::InvalidGraph, "TaskGraphRun is invalid"));
    }
    if (gInsideTaskCallback) {
#ifndef NDEBUG
        assert(false && "TaskGraphRun::wait cannot be called from a TaskSystem worker callback");
#endif
        return std::unexpected(makeTaskError(
            TaskErrorCode::WaitFromTask,
            "TaskGraphRun::wait cannot be called from a TaskSystem worker callback"));
    }
    state_->waitBlocking();
    return state_->snapshot();
}

TaskSystem::TaskSystem(const TaskSystemDesc& desc)
    : impl_(std::make_unique<Impl>(desc.workerCount))
{
}

TaskSystem::~TaskSystem()
{
    shutdown();
}

std::expected<TaskGraphRun, TaskError> TaskSystem::submit(TaskGraph&& graph)
{
    if (impl_ == nullptr || !impl_->backend.accepting.load(std::memory_order_acquire)) {
        return std::unexpected(makeTaskError(TaskErrorCode::ShuttingDown, "TaskSystem is shutting down"));
    }
    if (graph.impl_ == nullptr || graph.impl_->name.empty()) {
        return std::unexpected(makeTaskError(TaskErrorCode::InvalidGraph, "TaskGraph requires a non-empty name"));
    }
    if (graph.impl_->nodes.size() > UINT32_MAX) {
        return std::unexpected(makeTaskError(TaskErrorCode::InvalidGraph, "TaskGraph has too many nodes"));
    }
    for (const TaskGraph::Impl::Node& node : graph.impl_->nodes) {
        if (node.desc.name.empty()) {
            return std::unexpected(makeTaskError(
                TaskErrorCode::InvalidGraph,
                "TaskGraph contains a task with an empty name"));
        }
    }

    const size_t nodeCount = graph.impl_->nodes.size();
    std::vector<uint32_t> indegree(nodeCount, 0);
    std::vector<std::vector<uint32_t>> successors(nodeCount);
    for (const TaskGraphEdge& edge : graph.impl_->edges) {
        if (edge.prerequisite.graphId != graph.impl_->id ||
            edge.dependent.graphId != graph.impl_->id ||
            edge.prerequisite.nodeIndex >= nodeCount ||
            edge.dependent.nodeIndex >= nodeCount) {
            return std::unexpected(makeTaskError(TaskErrorCode::InvalidGraph, "TaskGraph contains an invalid edge"));
        }
        successors[edge.prerequisite.nodeIndex].push_back(edge.dependent.nodeIndex);
        ++indegree[edge.dependent.nodeIndex];
    }

    std::queue<uint32_t> ready;
    std::vector<uint32_t> remainingIndegree = indegree;
    for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        if (remainingIndegree[nodeIndex] == 0) {
            ready.push(nodeIndex);
        }
    }
    size_t visited = 0;
    while (!ready.empty()) {
        const uint32_t nodeIndex = ready.front();
        ready.pop();
        ++visited;
        for (uint32_t successor : successors[nodeIndex]) {
            if (--remainingIndegree[successor] == 0) {
                ready.push(successor);
            }
        }
    }
    if (visited != nodeCount) {
        return std::unexpected(makeTaskError(TaskErrorCode::InvalidGraph, "TaskGraph contains a cycle"));
    }

    std::vector<std::shared_ptr<RuntimeTaskNode>> runtimeNodes;
    runtimeNodes.reserve(nodeCount);
    for (uint32_t nodeIndex = 0; nodeIndex < nodeCount; ++nodeIndex) {
        TaskGraph::Impl::Node& source = graph.impl_->nodes[nodeIndex];
        auto runtimeNode = std::make_shared<RuntimeTaskNode>(
            TaskNodeHandle{
                .graphId = graph.impl_->id,
                .nodeIndex = nodeIndex,
            },
            std::move(source.desc),
            std::move(source.callback),
            indegree[nodeIndex]);
        runtimeNode->successors = std::move(successors[nodeIndex]);
        runtimeNodes.push_back(std::move(runtimeNode));
    }

    const uint64_t executionId = impl_->backend.nextExecutionId.fetch_add(1, std::memory_order_relaxed);
    auto state = std::make_shared<TaskGraphRun::State>(
        impl_->backend,
        executionId,
        graph.impl_->id,
        std::move(graph.impl_->name),
        std::move(runtimeNodes),
        std::move(graph.impl_->edges));
    impl_->backend.addExecution(state);
    state->start();
    return TaskGraphRun(std::move(state));
}

TaskEventSinkToken TaskSystem::subscribe(std::shared_ptr<ITaskEventSink> sink)
{
    if (impl_ == nullptr || sink == nullptr) {
        return 0;
    }
    const TaskEventSinkToken token = impl_->backend.nextSinkToken.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(impl_->backend.sinkMutex);
    impl_->backend.sinks.emplace(token, std::move(sink));
    return token;
}

bool TaskSystem::unsubscribe(TaskEventSinkToken token)
{
    if (impl_ == nullptr || token == 0) {
        return false;
    }
    std::lock_guard lock(impl_->backend.sinkMutex);
    return impl_->backend.sinks.erase(token) != 0;
}

uint32_t TaskSystem::workerCount() const
{
    return impl_ != nullptr ? impl_->backend.workerCount : 0;
}

bool TaskSystem::acceptingTasks() const
{
    return impl_ != nullptr && impl_->backend.accepting.load(std::memory_order_acquire);
}

void TaskSystem::shutdown()
{
    if (impl_ != nullptr) {
        impl_->backend.shutdown();
    }
}

std::expected<void, TaskError> initializeTaskSystem(const TaskSystemDesc& desc)
{
    std::lock_guard lock(gTaskSystemMutex);
    if (gTaskSystemShuttingDown) {
        return std::unexpected(makeTaskError(
            TaskErrorCode::ShuttingDown,
            "TaskSystem shutdown is still in progress"));
    }
    if (gTaskSystem != nullptr) {
        return std::unexpected(makeTaskError(
            TaskErrorCode::AlreadyInitialized,
            "TaskSystem is already initialized"));
    }
    try {
        gTaskSystem = std::shared_ptr<TaskSystem>(new TaskSystem(desc));
    } catch (const std::exception& exception) {
        return std::unexpected(makeTaskError(TaskErrorCode::InternalFailure, exception.what()));
    } catch (...) {
        return std::unexpected(makeTaskError(
            TaskErrorCode::InternalFailure,
            "TaskSystem initialization failed with an unknown exception"));
    }
    return {};
}

TaskSystem* tryGetTaskSystem() noexcept
{
    std::lock_guard lock(gTaskSystemMutex);
    return gTaskSystem.get();
}

TaskSystem& taskSystem()
{
    TaskSystem* system = tryGetTaskSystem();
    assert(system != nullptr && "TaskSystem has not been initialized");
    return *system;
}

void shutdownTaskSystem()
{
    if (gInsideTaskCallback) {
        assert(false && "TaskSystem cannot be shut down from a task callback");
        return;
    }
    std::shared_ptr<TaskSystem> system;
    {
        std::lock_guard lock(gTaskSystemMutex);
        if (gTaskSystem == nullptr) {
            return;
        }
        gTaskSystemShuttingDown = true;
        system = std::move(gTaskSystem);
    }
    system->shutdown();
    {
        std::lock_guard lock(gTaskSystemMutex);
        gTaskSystemShuttingDown = false;
    }
}

namespace detail {

std::shared_ptr<TaskSystem> tryAcquireTaskSystem() noexcept
{
    std::lock_guard lock(gTaskSystemMutex);
    return gTaskSystem;
}

} // namespace detail

} // namespace metallic::task

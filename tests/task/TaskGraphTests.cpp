#include "Runtime/Task/TaskSystem.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <latch>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace metallic::task {
namespace {

using namespace std::chrono_literals;

class TaskSystemTest : public testing::Test {
protected:
    void SetUp() override
    {
        shutdownTaskSystem();
        ASSERT_TRUE(initializeTaskSystem(TaskSystemDesc{.workerCount = 4}).has_value());
    }

    void TearDown() override
    {
        shutdownTaskSystem();
    }
};

const TaskNodeSnapshot& nodeAt(const TaskGraphSnapshot& snapshot, TaskNodeHandle handle)
{
    return snapshot.nodes.at(handle.nodeIndex);
}

TEST(TaskSystemLifecycle, ExplicitInitializationShutdownAndRestart)
{
    shutdownTaskSystem();
    EXPECT_EQ(tryGetTaskSystem(), nullptr);

    ASSERT_TRUE(initializeTaskSystem(TaskSystemDesc{.workerCount = 2}).has_value());
    ASSERT_NE(tryGetTaskSystem(), nullptr);
    EXPECT_EQ(taskSystem().workerCount(), 2u);

    const auto duplicate = initializeTaskSystem();
    ASSERT_FALSE(duplicate.has_value());
    EXPECT_EQ(duplicate.error().code, TaskErrorCode::AlreadyInitialized);

    shutdownTaskSystem();
    EXPECT_EQ(tryGetTaskSystem(), nullptr);

    ASSERT_TRUE(initializeTaskSystem(TaskSystemDesc{.workerCount = 1}).has_value());
    EXPECT_EQ(taskSystem().workerCount(), 1u);
    shutdownTaskSystem();
}

TEST(TaskSystemLifecycle, RejectsReinitializationUntilShutdownHasDrained)
{
    shutdownTaskSystem();
    ASSERT_TRUE(initializeTaskSystem(TaskSystemDesc{.workerCount = 1}).has_value());

    std::latch started(1);
    std::latch shutdownIsDraining(1);
    std::latch allowCompletion(1);
    TaskGraph graph("ShutdownDrain");
    graph.addTask({.name = "CooperativeStop"}, [&](TaskContext& context) {
        started.count_down();
        while (!context.stopRequested()) {
            std::this_thread::yield();
        }
        shutdownIsDraining.count_down();
        allowCompletion.wait();
    });
    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    started.wait();

    std::jthread shutdownThread([] {
        shutdownTaskSystem();
    });
    shutdownIsDraining.wait();
    const auto whileDraining = initializeTaskSystem();
    ASSERT_FALSE(whileDraining.has_value());
    EXPECT_EQ(whileDraining.error().code, TaskErrorCode::ShuttingDown);

    allowCompletion.count_down();
    shutdownThread.join();
    ASSERT_TRUE(initializeTaskSystem(TaskSystemDesc{.workerCount = 1}).has_value());
    shutdownTaskSystem();
}

TEST_F(TaskSystemTest, EmptyGraphCompletesImmediately)
{
    auto submitted = taskSystem().submit(TaskGraph("Empty"));
    ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
    EXPECT_TRUE(submitted->isComplete());
    const auto completed = submitted->wait();
    ASSERT_TRUE(completed.has_value());
    EXPECT_EQ(completed->status, TaskGraphStatus::Succeeded);
    EXPECT_TRUE(completed->nodes.empty());
}

TEST_F(TaskSystemTest, RejectsEmptyGraphAndTaskNamesBeforeExecution)
{
    const auto emptyGraphName = taskSystem().submit(TaskGraph(""));
    ASSERT_FALSE(emptyGraphName.has_value());
    EXPECT_EQ(emptyGraphName.error().code, TaskErrorCode::InvalidGraph);

    std::atomic_bool ran = false;
    TaskGraph graph("InvalidName");
    graph.addTask({}, [&] { ran = true; });
    const auto emptyTaskName = taskSystem().submit(std::move(graph));
    ASSERT_FALSE(emptyTaskName.has_value());
    EXPECT_EQ(emptyTaskName.error().code, TaskErrorCode::InvalidGraph);
    EXPECT_FALSE(ran.load());
}

TEST_F(TaskSystemTest, LinearChainHonorsDependencies)
{
    TaskGraph graph("Linear");
    std::atomic_int sequence = 0;
    const auto a = graph.addTask({.name = "A"}, [&] { EXPECT_EQ(sequence.fetch_add(1), 0); });
    const auto b = graph.addTask({.name = "B"}, [&] { EXPECT_EQ(sequence.fetch_add(1), 1); });
    const auto c = graph.addTask({.name = "C"}, [&] { EXPECT_EQ(sequence.fetch_add(1), 2); });
    ASSERT_TRUE(graph.addDependency(a, b).has_value());
    ASSERT_TRUE(graph.addDependency(b, c).has_value());

    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    ASSERT_TRUE(submitted->wait().has_value());
    EXPECT_EQ(sequence.load(), 3);
}

TEST_F(TaskSystemTest, IndependentRootsAndDiamondBranchesRunInParallel)
{
    TaskGraph graph("Diamond");
    std::atomic_int branchesFinished = 0;
    std::atomic_int activeBranches = 0;
    std::atomic_int maxActiveBranches = 0;
    std::atomic_bool rootFinished = false;

    const auto root = graph.addTask({.name = "Root"}, [&] { rootFinished = true; });
    const auto left = graph.addTask({.name = "Left"}, [&] {
        EXPECT_TRUE(rootFinished.load());
        const int active = activeBranches.fetch_add(1) + 1;
        maxActiveBranches.store(std::max(maxActiveBranches.load(), active));
        std::this_thread::sleep_for(50ms);
        --activeBranches;
        ++branchesFinished;
    });
    const auto right = graph.addTask({.name = "Right"}, [&] {
        EXPECT_TRUE(rootFinished.load());
        const int active = activeBranches.fetch_add(1) + 1;
        maxActiveBranches.store(std::max(maxActiveBranches.load(), active));
        std::this_thread::sleep_for(50ms);
        --activeBranches;
        ++branchesFinished;
    });
    const auto join = graph.addTask({.name = "Join"}, [&] {
        EXPECT_EQ(branchesFinished.load(), 2);
    });
    ASSERT_TRUE(graph.addDependency(root, left).has_value());
    ASSERT_TRUE(graph.addDependency(root, right).has_value());
    ASSERT_TRUE(graph.addDependency(left, join).has_value());
    ASSERT_TRUE(graph.addDependency(right, join).has_value());

    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    const auto completed = submitted->wait();
    ASSERT_TRUE(completed.has_value());
    EXPECT_EQ(completed->status, TaskGraphStatus::Succeeded);
    EXPECT_GE(maxActiveBranches.load(), 2);
}

TEST_F(TaskSystemTest, RejectsInvalidEdgesAndCyclesBeforeStartingTasks)
{
    TaskGraph graph("InvalidEdges");
    TaskGraph other("Other");
    const auto a = graph.addTask({.name = "A"}, [] {});
    const auto b = graph.addTask({.name = "B"}, [] {});
    const auto foreign = other.addTask({.name = "Foreign"}, [] {});

    EXPECT_FALSE(graph.addDependency({}, b).has_value());
    EXPECT_FALSE(graph.addDependency(a, a).has_value());
    EXPECT_FALSE(graph.addDependency(a, foreign).has_value());
    ASSERT_TRUE(graph.addDependency(a, b).has_value());
    EXPECT_FALSE(graph.addDependency(a, b).has_value());

    std::atomic_int starts = 0;
    TaskGraph cycle("Cycle");
    const auto cycleA = cycle.addTask({.name = "A"}, [&] { ++starts; });
    const auto cycleB = cycle.addTask({.name = "B"}, [&] { ++starts; });
    ASSERT_TRUE(cycle.addDependency(cycleA, cycleB).has_value());
    ASSERT_TRUE(cycle.addDependency(cycleB, cycleA).has_value());
    const auto submitted = taskSystem().submit(std::move(cycle));
    ASSERT_FALSE(submitted.has_value());
    EXPECT_EQ(submitted.error().code, TaskErrorCode::InvalidGraph);
    EXPECT_EQ(starts.load(), 0);
}

TEST_F(TaskSystemTest, HoldsMoveOnlyCallbacksUntilExecution)
{
    TaskGraph graph("MoveOnly");
    auto value = std::make_unique<int>(42);
    std::atomic_int observed = 0;
    graph.addTask({.name = "Consume"}, [owned = std::move(value), &observed] {
        observed = *owned;
    });

    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    ASSERT_TRUE(submitted->wait().has_value());
    EXPECT_EQ(observed.load(), 42);
}

TEST_F(TaskSystemTest, ConvertsExceptionsAndKeepsIndependentBranchesRunning)
{
    TaskGraph graph("FailureIsolation");
    std::atomic_bool dependentRan = false;
    std::atomic_bool independentRan = false;
    const auto failure = graph.addTask({.name = "Failure"}, []() -> TaskOutcome {
        return std::unexpected("decode failed");
    });
    const auto dependent = graph.addTask({.name = "Dependent"}, [&] { dependentRan = true; });
    const auto throwing = graph.addTask({.name = "Throwing"}, [] {
        throw std::runtime_error("callback exploded");
    });
    const auto independent = graph.addTask({.name = "Independent"}, [&] { independentRan = true; });
    ASSERT_TRUE(graph.addDependency(failure, dependent).has_value());

    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    const auto completed = submitted->wait();
    ASSERT_TRUE(completed.has_value());
    EXPECT_EQ(completed->status, TaskGraphStatus::Failed);
    EXPECT_EQ(nodeAt(*completed, failure).state, TaskState::Failed);
    EXPECT_EQ(nodeAt(*completed, failure).error, "decode failed");
    EXPECT_EQ(nodeAt(*completed, dependent).state, TaskState::Skipped);
    EXPECT_FALSE(nodeAt(*completed, dependent).error.empty());
    EXPECT_EQ(nodeAt(*completed, throwing).state, TaskState::Failed);
    EXPECT_EQ(nodeAt(*completed, throwing).error, "callback exploded");
    EXPECT_EQ(nodeAt(*completed, independent).state, TaskState::Succeeded);
    EXPECT_FALSE(dependentRan.load());
    EXPECT_TRUE(independentRan.load());
}

TEST_F(TaskSystemTest, RunningTaskObservesCooperativeCancellation)
{
    TaskGraph graph("Cancellation");
    std::latch started(1);
    std::atomic_bool observedStop = false;
    const auto running = graph.addTask({.name = "Running"}, [&](TaskContext& context) {
        started.count_down();
        while (!context.stopRequested()) {
            std::this_thread::yield();
        }
        observedStop = true;
    });
    const auto dependent = graph.addTask({.name = "Dependent"}, [] {});
    ASSERT_TRUE(graph.addDependency(running, dependent).has_value());

    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    started.wait();
    EXPECT_TRUE(submitted->requestStop());
    const auto completed = submitted->wait();
    ASSERT_TRUE(completed.has_value());
    EXPECT_EQ(completed->status, TaskGraphStatus::Cancelled);
    EXPECT_TRUE(observedStop.load());
    EXPECT_EQ(nodeAt(*completed, running).state, TaskState::Succeeded);
    EXPECT_EQ(nodeAt(*completed, dependent).state, TaskState::Cancelled);
}

TEST(TaskSystemCancellation, CancelsScheduledWorkBeforeItRuns)
{
    shutdownTaskSystem();
    ASSERT_TRUE(initializeTaskSystem(TaskSystemDesc{.workerCount = 1}).has_value());

    std::latch blockerStarted(1);
    std::latch releaseBlocker(1);
    TaskGraph blocker("Blocker");
    blocker.addTask({.name = "Blocker"}, [&] {
        blockerStarted.count_down();
        releaseBlocker.wait();
    });
    auto blockerRun = taskSystem().submit(std::move(blocker));
    ASSERT_TRUE(blockerRun.has_value());
    blockerStarted.wait();

    std::atomic_bool ran = false;
    TaskGraph graph("PreRunCancellation");
    const auto node = graph.addTask({.name = "Pending"}, [&] { ran = true; });
    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    EXPECT_TRUE(submitted->requestStop());
    releaseBlocker.count_down();

    ASSERT_TRUE(blockerRun->wait().has_value());
    const auto completed = submitted->wait();
    ASSERT_TRUE(completed.has_value());
    EXPECT_FALSE(ran.load());
    EXPECT_EQ(nodeAt(*completed, node).state, TaskState::Cancelled);
    shutdownTaskSystem();
}

class RecordingSink final : public ITaskEventSink {
public:
    void onGraphSubmitted(const TaskGraphSnapshot& snapshot) override
    {
        std::lock_guard lock(mutex);
        submitted.push_back(snapshot);
    }

    void onTaskStateChanged(const TaskNodeEvent& event) override
    {
        std::lock_guard lock(mutex);
        events.push_back(event);
    }

    void onGraphCompleted(const TaskGraphSnapshot& snapshot) override
    {
        std::lock_guard lock(mutex);
        completed.push_back(snapshot);
    }

    std::mutex mutex;
    std::vector<TaskGraphSnapshot> submitted;
    std::vector<TaskNodeEvent> events;
    std::vector<TaskGraphSnapshot> completed;
};

TEST_F(TaskSystemTest, ObserverAndSnapshotContainStableGraphMetadata)
{
    auto sink = std::make_shared<RecordingSink>();
    const TaskEventSinkToken token = taskSystem().subscribe(sink);
    ASSERT_NE(token, 0u);

    TaskGraph graph("Observed");
    const auto a = graph.addTask({.name = "A", .category = "Tests", .color = 0xff00ffu, .userTag = 7}, [] {});
    const auto b = graph.addTask({.name = "B", .category = "Tests"}, [] {});
    ASSERT_TRUE(graph.addDependency(a, b).has_value());
    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted.has_value());
    const auto finalSnapshot = submitted->wait();
    ASSERT_TRUE(finalSnapshot.has_value());
    EXPECT_TRUE(taskSystem().unsubscribe(token));

    std::lock_guard lock(sink->mutex);
    ASSERT_EQ(sink->submitted.size(), 1u);
    ASSERT_EQ(sink->completed.size(), 1u);
    EXPECT_EQ(sink->events.size(), 6u);
    EXPECT_EQ(sink->submitted.front().edges.size(), 1u);
    EXPECT_EQ(sink->completed.front().executionId, finalSnapshot->executionId);
    EXPECT_EQ(sink->completed.front().graphId, finalSnapshot->graphId);
    EXPECT_EQ(sink->completed.front().status, finalSnapshot->status);
    for (const TaskNodeSnapshot& node : finalSnapshot->nodes) {
        EXPECT_EQ(node.state, TaskState::Succeeded);
        EXPECT_NE(node.workerThreadId, 0u);
        EXPECT_LE(node.scheduledTime, node.startTime);
        EXPECT_LE(node.startTime, node.finishTime);
    }
}

TEST(TaskSystemShutdown, DrainsAbandonedFireAndForgetRuns)
{
    shutdownTaskSystem();
    ASSERT_TRUE(initializeTaskSystem(TaskSystemDesc{.workerCount = 2}).has_value());
    std::latch started(1);
    std::atomic_bool stopped = false;
    {
        TaskGraph graph("FireAndForget");
        graph.addTask({.name = "LongRunning"}, [&](TaskContext& context) {
            started.count_down();
            while (!context.stopRequested()) {
                std::this_thread::yield();
            }
            stopped = true;
        });
        auto submitted = taskSystem().submit(std::move(graph));
        ASSERT_TRUE(submitted.has_value());
    }
    started.wait();
    shutdownTaskSystem();
    EXPECT_TRUE(stopped.load());
    EXPECT_EQ(tryGetTaskSystem(), nullptr);
}

#ifdef NDEBUG
TEST_F(TaskSystemTest, WorkerWaitReturnsAnExplicitErrorInReleaseBuilds)
{
    TaskGraph inner("Inner");
    inner.addTask({.name = "Inner"}, [] {});
    auto innerRun = taskSystem().submit(std::move(inner));
    ASSERT_TRUE(innerRun.has_value());
    std::atomic_bool rejected = false;

    TaskGraph outer("Outer");
    outer.addTask({.name = "Outer"}, [run = std::move(*innerRun), &rejected]() mutable {
        const auto result = run.wait();
        rejected = !result.has_value() && result.error().code == TaskErrorCode::WaitFromTask;
    });
    auto outerRun = taskSystem().submit(std::move(outer));
    ASSERT_TRUE(outerRun.has_value());
    ASSERT_TRUE(outerRun->wait().has_value());
    EXPECT_TRUE(rejected.load());
}
#endif

} // namespace
} // namespace metallic::task

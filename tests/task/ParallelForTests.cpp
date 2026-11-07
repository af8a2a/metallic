#include "Runtime/Task/ParallelFor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace metallic::task {
namespace {

constexpr std::array kStrategies{
    PartitionStrategy::Static,
    PartitionStrategy::Dynamic,
    PartitionStrategy::AdaptiveHalfSteal,
};

class ParallelForTest : public testing::Test {
protected:
    void SetUp() override
    {
        shutdownTaskSystem();
        ASSERT_TRUE(initializeTaskSystem({.workerCount = 4}));
    }

    void TearDown() override
    {
        shutdownTaskSystem();
    }
};

template <typename Predicate>
bool waitUntil(Predicate predicate)
{
    const auto deadline = TaskClock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (TaskClock::now() >= deadline) { return false; }
        std::this_thread::yield();
    }
    return true;
}

TEST_F(ParallelForTest, CoversEveryElementOnceAcrossStrategiesGrainsAndWorkerLimits)
{
    for (uint32_t workers : {1u, 2u, 4u, 8u}) {
        shutdownTaskSystem();
        ASSERT_TRUE(initializeTaskSystem({.workerCount = workers}));
        for (const auto strategy : kStrategies) {
            for (uint32_t count : {0u, 1u, 2u, 31u, 257u, 4099u}) {
                for (uint32_t grain : {1u, 7u, 256u}) {
                    SCOPED_TRACE(testing::Message() << workers << "/" << int(strategy)
                        << "/" << count << "/" << grain);
                    std::vector<std::atomic_uint32_t> visits(count);
                    std::array<std::atomic_uint32_t, 8> active{};
                    ParallelForDiagnostics diagnostics;
                    const auto result = parallelFor(11, 11 + count,
                        {.minGrain = grain, .maxWorkers = workers, .strategy = strategy,
                         .diagnostics = &diagnostics},
                        [&](uint32_t begin, uint32_t end, WorkerContext& worker) {
                            EXPECT_LT(begin, end);
                            EXPECT_GE(begin, 11u);
                            EXPECT_LE(end, 11u + count);
                            EXPECT_LE(uint64_t(end) - begin, grain);
                            EXPECT_GT(worker.workerCount, 0u);
                            EXPECT_LE(worker.workerCount, workers);
                            ASSERT_LT(worker.workerIndex, worker.workerCount);
                            EXPECT_EQ(active[worker.workerIndex].fetch_add(1), 0u);
                            for (uint32_t index = begin; index < end; ++index) {
                                ++visits[index - 11];
                            }
                            EXPECT_EQ(active[worker.workerIndex].fetch_sub(1), 1u);
                        });
                    ASSERT_TRUE(result) << result.error();
                    for (const auto& visit : visits) { EXPECT_EQ(visit.load(), 1u); }
                    uint64_t elements = 0;
                    for (const auto& worker : diagnostics.workers) { elements += worker.elements; }
                    EXPECT_EQ(elements, count);
                    for (const auto& slot : active) { EXPECT_EQ(slot.load(), 0u); }
                }
            }
        }
    }
}

TEST_F(ParallelForTest, HandlesTailsNearMaximumEndpointWithoutWraparound)
{
    constexpr uint32_t begin = UINT32_MAX - 1031;
    for (const auto strategy : kStrategies) {
        for (uint32_t grain : {1u, 17u, 1024u, UINT32_MAX}) {
            std::array<std::atomic_uint32_t, 1031> visits{};
            const auto result = parallelFor(begin, UINT32_MAX,
                {.minGrain = grain, .maxWorkers = 4, .strategy = strategy},
                [&](uint32_t first, uint32_t last, WorkerContext&) {
                    ASSERT_GE(first, begin);
                    ASSERT_LE(last, UINT32_MAX);
                    ASSERT_LT(first, last);
                    for (uint32_t index = first; index != last; ++index) { ++visits[index - begin]; }
                });
            ASSERT_TRUE(result) << result.error();
            for (const auto& count : visits) { EXPECT_EQ(count.load(), 1u); }
        }
    }
}

TEST_F(ParallelForTest, HalfStealTransfersTheBusyOwnersRemainingIntervalWithoutDuplicateClaims)
{
    std::array<std::atomic_uint32_t, 4096> visits{};
    std::atomic_bool stolenFromFirstLane{false};
    bool ownerWaited = false; // Only lane zero accesses this value.
    ParallelForDiagnostics diagnostics;
    const auto result = parallelFor(0, uint32_t(visits.size()),
        {.minGrain = 1, .maxWorkers = 4, .strategy = PartitionStrategy::AdaptiveHalfSteal,
         .diagnostics = &diagnostics},
        [&](uint32_t first, uint32_t last, WorkerContext& worker) {
            // Lane zero starts with [0,1024). Its first callback is held while
            // other lanes finish their own work and claim its unprocessed tail.
            // The wait is bounded and does not require every lane to start.
            if (worker.workerIndex != 0 && first < 1024) { stolenFromFirstLane = true; }
            if (worker.workerIndex == 0 && !ownerWaited) {
                ownerWaited = true;
                EXPECT_TRUE(waitUntil([&] { return stolenFromFirstLane.load(); }));
            }
            for (uint32_t index = first; index < last; ++index) { ++visits[index]; }
        });
    ASSERT_TRUE(result) << result.error();
    EXPECT_TRUE(stolenFromFirstLane.load());
    uint64_t steals = 0;
    for (const auto& worker : diagnostics.workers) { steals += worker.stealSuccesses; }
    EXPECT_GT(steals, 0u);
    for (const auto& visit : visits) { EXPECT_EQ(visit.load(), 1u); }
}

TEST_F(ParallelForTest, CoversHugeRangesWithCoarseBatchesAndNoEndpointOverflow)
{
    for (const auto strategy : kStrategies) {
        for (uint32_t grain : {1u << 28u, UINT32_MAX}) {
            std::vector<std::pair<uint32_t, uint32_t>> intervals;
            std::mutex mutex;
            const auto result = parallelFor(0, UINT32_MAX,
                {.minGrain = grain, .maxWorkers = 4, .strategy = strategy},
                [&](uint32_t first, uint32_t last, WorkerContext&) {
                    EXPECT_LT(first, last);
                    EXPECT_LE(uint64_t(last) - first, grain);
                    std::lock_guard lock(mutex);
                    intervals.emplace_back(first, last);
                });
            ASSERT_TRUE(result) << result.error();
            std::sort(intervals.begin(), intervals.end());
            uint64_t next = 0;
            for (const auto& [first, last] : intervals) {
                EXPECT_EQ(first, next);
                next = last;
            }
            EXPECT_EQ(next, UINT32_MAX);
        }
    }
}

TEST_F(ParallelForTest, RejectsInvalidBoundaryOptionsCallbacksAndNamesBeforeAddingNodes)
{
    const ParallelForCallback callback = [](uint32_t, uint32_t, WorkerContext&) {};
    EXPECT_FALSE(parallelFor(2, 1, {}, callback));
    EXPECT_FALSE(parallelFor(0, 1, {.minGrain = 0}, callback));
    EXPECT_FALSE(parallelFor(0, 1, {.strategy = static_cast<PartitionStrategy>(255)}, callback));
    EXPECT_FALSE(parallelFor(0, 1, {}, {}));
    TaskGraph graph("InvalidRanges");
    const auto invalidRange = addParallelFor(graph, {.name = "InvalidRange"}, 10, 9, {}, callback);
    const auto invalidGrain = addParallelFor(graph, {.name = "InvalidGrain"}, 0, 1, {.minGrain = 0}, callback);
    const auto invalidName = addParallelFor(graph, {}, 0, 1, {}, callback);
    const auto invalidCallback = addParallelFor(graph, {.name = "InvalidCallback"}, 0, 1, {}, {});
    for (const auto* error : {&invalidRange, &invalidGrain, &invalidName, &invalidCallback}) {
        ASSERT_FALSE(*error);
        EXPECT_EQ(error->error().code, TaskErrorCode::InvalidGraph);
    }
    graph.addTask({.name = "OnlyValidNode"}, [] {});
    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted);
    auto joined = submitted->wait();
    ASSERT_TRUE(joined);
    EXPECT_EQ(joined->status, TaskGraphStatus::Succeeded);
    EXPECT_EQ(joined->nodes.size(), 1u);
}

TEST_F(ParallelForTest, JoinsAllStartedCallbacksAfterExceptionsAndRemainsReusable)
{
    for (const auto strategy : kStrategies) {
        for (bool standardException : {false, true}) {
            std::atomic_uint32_t active{0};
            const auto result = parallelFor(0, 1024,
                {.minGrain = 1, .maxWorkers = 4, .strategy = strategy},
                [&](uint32_t first, uint32_t, WorkerContext&) {
                    ++active;
                    if (first == 0) {
                        --active;
                        if (standardException) { throw std::runtime_error("range failure"); }
                        throw 42;
                    }
                    std::this_thread::yield();
                    --active;
                });
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error(), standardException ? "range failure" : "task callback threw an unknown exception");
            EXPECT_EQ(active.load(), 0u);
            std::atomic_uint32_t elements{0};
            const auto reused = parallelFor(0, 1031,
                {.minGrain = 7, .maxWorkers = 4, .strategy = strategy},
                [&](uint32_t first, uint32_t last, WorkerContext&) { elements += last - first; });
            ASSERT_TRUE(reused) << reused.error();
            EXPECT_EQ(elements.load(), 1031u);
        }
    }
}

TEST_F(ParallelForTest, HonorsExternalStopBeforeAndDuringExecution)
{
    for (const auto strategy : kStrategies) {
        std::stop_source stopped;
        stopped.request_stop();
        std::atomic_uint32_t calls{0};
        const auto before = parallelFor(0, 100,
            {.minGrain = 1, .maxWorkers = 4, .strategy = strategy, .stopToken = stopped.get_token()},
            [&](uint32_t, uint32_t, WorkerContext&) { ++calls; });
        EXPECT_FALSE(before);
        EXPECT_EQ(calls.load(), 0u);

        std::stop_source during;
        const auto result = parallelFor(0, 10000,
            {.minGrain = 1, .maxWorkers = 4, .strategy = strategy, .stopToken = during.get_token()},
            [&](uint32_t, uint32_t, WorkerContext&) {
                ++calls;
                during.request_stop();
            });
        EXPECT_FALSE(result);
        EXPECT_GT(calls.load(), 0u);
        EXPECT_LT(calls.load(), 10000u);
    }
}

TEST_F(ParallelForTest, ConnectsAsyncInputsAndOutputsThroughEntryAndCompletion)
{
    for (const auto strategy : kStrategies) {
        TaskGraph graph("AsyncRanges");
        const uint64_t graphId = graph.id();
        std::vector<uint32_t> values(1031);
        std::atomic_bool consumed{false};
        const auto producer = graph.addTask({.name = "Producer"}, [&] { std::fill(values.begin(), values.end(), 7u); });
        const auto range = addParallelFor(graph, {.name = "Transform", .category = "Test", .userTag = 73},
            0, uint32_t(values.size()), {.minGrain = 13, .maxWorkers = 4, .strategy = strategy},
            [&](uint32_t first, uint32_t last, WorkerContext& worker) {
                EXPECT_NE(worker.executionId, 0u);
                EXPECT_EQ(worker.node.graphId, graphId);
                EXPECT_TRUE(worker.node.valid());
                for (uint32_t index = first; index < last; ++index) { values[index] *= 3; }
            });
        ASSERT_TRUE(range);
        const auto consumer = graph.addTask({.name = "Consumer"}, [&] {
            for (const auto value : values) { EXPECT_EQ(value, 21u); }
            consumed = true;
        });
        ASSERT_TRUE(graph.addDependency(producer, range->entry));
        ASSERT_TRUE(graph.addDependency(range->completion, consumer));
        auto submitted = taskSystem().submit(std::move(graph));
        ASSERT_TRUE(submitted);
        const auto joined = submitted->wait();
        ASSERT_TRUE(joined);
        EXPECT_EQ(joined->graphId, graphId);
        EXPECT_EQ(joined->status, TaskGraphStatus::Succeeded);
        EXPECT_TRUE(consumed.load());
        EXPECT_EQ(joined->nodes[range->completion.nodeIndex].state, TaskState::Succeeded);
    }
}

TEST_F(ParallelForTest, DiagnosticsAssociateOperationsAndLanesWithTheContainingGraphSnapshot)
{
    TaskGraph graph("ObservedRanges");
    std::array<ParallelForDiagnostics, 2> diagnostics;
    std::array<ParallelForNodes, 2> nodes;
    std::atomic_uint32_t elements{0};
    for (uint32_t operation = 0; operation < diagnostics.size(); ++operation) {
        std::string backend = "test-kernel-" + std::to_string(operation);
        auto added = addParallelFor(graph, {.name = "Operation" + std::to_string(operation)},
            7, 1038, {.minGrain = 8, .maxWorkers = 4, .strategy = kStrategies[operation],
                .backendLabel = backend, .diagnostics = &diagnostics[operation]},
            [&](uint32_t begin, uint32_t end, WorkerContext& worker) {
                EXPECT_TRUE(worker.node.valid());
                elements += end - begin;
            });
        ASSERT_TRUE(added);
        nodes[operation] = *added;
        backend.assign(backend.size(), 'x');
    }
    ASSERT_TRUE(graph.addDependency(nodes[0].completion, nodes[1].entry));
    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted);
    const auto snapshot = submitted->wait();
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(snapshot->status, TaskGraphStatus::Succeeded);
    EXPECT_EQ(elements.load(), 2u * 1031u);
    EXPECT_NE(diagnostics[0].operationId, 0u);
    EXPECT_NE(diagnostics[0].operationId, diagnostics[1].operationId);
    for (uint32_t operation = 0; operation < diagnostics.size(); ++operation) {
        const auto& observed = diagnostics[operation];
        EXPECT_EQ(observed.operationName, "Operation" + std::to_string(operation));
        EXPECT_EQ(observed.backendLabel, "test-kernel-" + std::to_string(operation));
        EXPECT_EQ(observed.graphId, snapshot->graphId);
        EXPECT_EQ(observed.begin, 7u);
        EXPECT_EQ(observed.end, 1038u);
        EXPECT_EQ(observed.minGrain, 8u);
        EXPECT_EQ(observed.strategy, kStrategies[operation]);
        EXPECT_LE(observed.createdTime, snapshot->submittedTime);
        for (uint32_t lane = 0; lane < observed.workers.size(); ++lane) {
            const auto& worker = observed.workers[lane];
            ASSERT_TRUE(worker.node.valid());
            ASSERT_LT(worker.node.nodeIndex, snapshot->nodes.size());
            const auto& task = snapshot->nodes[worker.node.nodeIndex];
            EXPECT_EQ(worker.workerIndex, lane);
            EXPECT_EQ(worker.executionId, snapshot->executionId);
            EXPECT_EQ(worker.node, task.handle);
            EXPECT_EQ(worker.workerThreadId, task.workerThreadId);
            EXPECT_EQ(worker.node.graphId, snapshot->graphId);
            EXPECT_TRUE(task.desc.name.starts_with(observed.operationName + ".range."));
            EXPECT_FALSE(worker.inlineExecution);
            EXPECT_LE(task.scheduledTime, task.startTime);
            EXPECT_LE(task.startTime, worker.startTime);
            EXPECT_LE(worker.startTime, worker.finishTime);
            EXPECT_LE(worker.finishTime, task.finishTime);
            EXPECT_LE(worker.activeNanoseconds,
                uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(worker.finishTime - worker.startTime).count()));
        }
    }
}

TEST_F(ParallelForTest, NestedDiagnosticsBorrowTheParentIdentityAndFinishOnException)
{
    TaskGraph graph("ObservedNestedRanges");
    ParallelForDiagnostics failed, succeeded;
    TaskNodeHandle parent;
    parent = graph.addTask({.name = "Parent"}, [&](TaskContext& context) {
        const auto result = parallelFor(0, 100, {.minGrain = 8, .maxWorkers = 4,
            .backendLabel = "nested-failure", .diagnostics = &failed},
            [&](uint32_t, uint32_t, WorkerContext& worker) {
                EXPECT_EQ(worker.node, context.node());
                EXPECT_EQ(worker.executionId, context.executionId());
                throw std::runtime_error("observed nested failure");
            });
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), "observed nested failure");
        EXPECT_EQ(detail::taskCallbackIdentity().node, context.node());
        EXPECT_EQ(detail::taskCallbackIdentity().executionId, context.executionId());
        EXPECT_TRUE(parallelFor(0, 1, {.diagnostics = &succeeded},
            [&](uint32_t, uint32_t, WorkerContext& worker) {
                EXPECT_EQ(worker.node, context.node());
                EXPECT_EQ(worker.executionId, context.executionId());
            }));
    });
    auto submitted = taskSystem().submit(std::move(graph));
    ASSERT_TRUE(submitted);
    const auto snapshot = submitted->wait();
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(snapshot->status, TaskGraphStatus::Succeeded);
    EXPECT_NE(failed.operationId, succeeded.operationId);
    const auto& task = snapshot->nodes[parent.nodeIndex];
    for (const auto* observed : {&failed, &succeeded}) {
        EXPECT_EQ(observed->graphId, snapshot->graphId);
        ASSERT_EQ(observed->workers.size(), 1u);
        const auto& worker = observed->workers.front();
        EXPECT_TRUE(worker.inlineExecution);
        EXPECT_EQ(worker.node, parent);
        EXPECT_EQ(worker.executionId, snapshot->executionId);
        EXPECT_EQ(worker.workerThreadId, task.workerThreadId);
        EXPECT_LE(task.startTime, worker.startTime);
        EXPECT_LE(worker.startTime, worker.finishTime);
        EXPECT_LE(worker.finishTime, task.finishTime);
    }
    EXPECT_FALSE(detail::taskCallbackIdentity().node.valid());
    EXPECT_EQ(detail::taskCallbackIdentity().executionId, 0u);
}

TEST_F(ParallelForTest, ExternalInlineDiagnosticsDoNotInventGraphOrTaskIdentities)
{
    ParallelForDiagnostics diagnostics;
    EXPECT_TRUE(parallelFor(17, 18, {.maxWorkers = 1, .diagnostics = &diagnostics},
        [](uint32_t, uint32_t, WorkerContext& worker) {
            EXPECT_FALSE(worker.node.valid());
            EXPECT_EQ(worker.executionId, 0u);
        }));
    EXPECT_EQ(diagnostics.graphId, 0u);
    EXPECT_NE(diagnostics.operationId, 0u);
    ASSERT_EQ(diagnostics.workers.size(), 1u);
    const auto& worker = diagnostics.workers.front();
    EXPECT_TRUE(worker.inlineExecution);
    EXPECT_FALSE(worker.node.valid());
    EXPECT_EQ(worker.executionId, 0u);
    EXPECT_LE(diagnostics.createdTime, worker.startTime);
    EXPECT_LE(worker.startTime, worker.finishTime);
}

TEST_F(ParallelForTest, AsyncFailureAndExternalStopSkipDependentPublication)
{
    for (bool cancellation : {false, true}) {
        std::stop_source stop;
        if (cancellation) { stop.request_stop(); }
        TaskGraph graph("FailedAsyncRanges");
        std::atomic_bool published{false};
        std::atomic_bool independent{false};
        const auto ranges = addParallelFor(graph, {.name = "Ranges"}, 0, 1024,
            {.minGrain = 1, .maxWorkers = 4, .stopToken = stop.get_token()},
            [](uint32_t first, uint32_t, WorkerContext&) {
                if (first == 0) { throw std::runtime_error("async range failure"); }
            });
        ASSERT_TRUE(ranges);
        const auto publication = graph.addTask({.name = "Publish"}, [&] { published = true; });
        graph.addTask({.name = "Independent"}, [&] { independent = true; });
        ASSERT_TRUE(graph.addDependency(ranges->completion, publication));
        auto submitted = taskSystem().submit(std::move(graph));
        ASSERT_TRUE(submitted);
        auto joined = submitted->wait();
        ASSERT_TRUE(joined);
        EXPECT_EQ(joined->status, TaskGraphStatus::Failed);
        EXPECT_EQ(joined->nodes[publication.nodeIndex].state, TaskState::Skipped);
        EXPECT_FALSE(published.load());
        EXPECT_TRUE(independent.load());
    }
}

TEST_F(ParallelForTest, GraphStopCancelsUnclaimedRangesAndSkipsPublication)
{
    for (const auto strategy : kStrategies) {
        TaskGraph graph("StoppedAsyncRanges");
        std::atomic_bool started{false};
        std::atomic_bool release{false};
        std::atomic_bool published{false};
        std::atomic_uint32_t callbacks{0};
        const auto ranges = addParallelFor(graph, {.name = "Ranges"}, 0, 10000,
            {.minGrain = 1, .maxWorkers = 4, .strategy = strategy},
            [&](uint32_t, uint32_t, WorkerContext&) {
                ++callbacks;
                started = true;
                EXPECT_TRUE(waitUntil([&] { return release.load(); }));
            });
        ASSERT_TRUE(ranges);
        const auto publication = graph.addTask({.name = "Publish"}, [&] { published = true; });
        ASSERT_TRUE(graph.addDependency(ranges->completion, publication));
        auto submitted = taskSystem().submit(std::move(graph));
        ASSERT_TRUE(submitted);
        EXPECT_TRUE(waitUntil([&] { return started.load(); }));
        EXPECT_TRUE(submitted->requestStop());
        release = true;
        const auto joined = submitted->wait();
        ASSERT_TRUE(joined);
        EXPECT_EQ(joined->status, TaskGraphStatus::Cancelled);
        EXPECT_FALSE(published.load());
        EXPECT_GT(callbacks.load(), 0u);
        EXPECT_LT(callbacks.load(), 10000u);
    }
}

TEST_F(ParallelForTest, NestedRangesRunInlineOnEveryParentCallbackWithoutWaitingForWorkers)
{
    shutdownTaskSystem();
    ASSERT_TRUE(initializeTaskSystem({.workerCount = 8}));
    for (const auto strategy : kStrategies) {
        TaskGraph graph("NestedRanges");
        std::atomic_uint32_t visits{0};
        for (uint32_t parent = 0; parent < 8; ++parent) {
            graph.addTask({.name = "Parent" + std::to_string(parent)}, [&] {
                const auto parentThread = std::this_thread::get_id();
                const auto nested = parallelFor(0, 33,
                    {.minGrain = 7, .maxWorkers = 8, .strategy = strategy},
                    [&](uint32_t first, uint32_t last, WorkerContext& worker) {
                        EXPECT_EQ(std::this_thread::get_id(), parentThread);
                        EXPECT_EQ(worker.workerIndex, 0u);
                        EXPECT_EQ(worker.workerCount, 1u);
                        const auto inner = parallelFor(first, last, {.minGrain = 1, .maxWorkers = 8},
                            [&](uint32_t, uint32_t, WorkerContext&) { ++visits; });
                        EXPECT_TRUE(inner);
                    });
                EXPECT_TRUE(nested);
            });
        }
        auto submitted = taskSystem().submit(std::move(graph));
        ASSERT_TRUE(submitted);
        const auto joined = submitted->wait();
        ASSERT_TRUE(joined);
        EXPECT_EQ(joined->status, TaskGraphStatus::Succeeded);
        EXPECT_EQ(visits.load(), 8u * 33u);
    }
}

TEST_F(ParallelForTest, NestedInlineRangeObservesTheParentGraphsStopToken)
{
    for (const auto strategy : kStrategies) {
        TaskGraph graph("NestedRangeCancellation");
        std::atomic_bool started{false};
        std::atomic_bool release{false};
        std::atomic_bool nestedStopped{false};
        std::atomic_uint32_t callbacks{0};
        graph.addTask({.name = "Parent"}, [&] {
            const auto result = parallelFor(0, 10000,
                {.minGrain = 1, .maxWorkers = 4, .strategy = strategy},
                [&](uint32_t, uint32_t, WorkerContext&) {
                    ++callbacks;
                    started = true;
                    EXPECT_TRUE(waitUntil([&] { return release.load(); }));
                });
            nestedStopped = !result;
        });
        auto submitted = taskSystem().submit(std::move(graph));
        ASSERT_TRUE(submitted);
        EXPECT_TRUE(waitUntil([&] { return started.load(); }));
        EXPECT_TRUE(submitted->requestStop());
        release = true;
        const auto joined = submitted->wait();
        ASSERT_TRUE(joined);
        EXPECT_EQ(joined->status, TaskGraphStatus::Cancelled);
        EXPECT_TRUE(nestedStopped.load());
        EXPECT_EQ(callbacks.load(), 1u);
    }
}

TEST_F(ParallelForTest, ConcurrentSubmissionsOwnIndependentRangesAndScratchSlots)
{
    std::atomic_uint32_t failures{0};
    std::vector<std::jthread> callers;
    for (uint32_t caller = 0; caller < 4; ++caller) {
        callers.emplace_back([&, caller] {
            for (uint32_t iteration = 0; iteration < 12; ++iteration) {
                std::array<std::atomic_uint32_t, 1031> visits{};
                std::array<std::atomic_uint32_t, 4> active{};
                const auto result = parallelFor(0, uint32_t(visits.size()),
                    {.minGrain = 7, .maxWorkers = 4, .strategy = kStrategies[(caller + iteration) % kStrategies.size()]},
                    [&](uint32_t first, uint32_t last, WorkerContext& worker) {
                        if (worker.workerIndex >= active.size()) { ++failures; return; }
                        if (active[worker.workerIndex].fetch_add(1) != 0) { ++failures; }
                        for (uint32_t index = first; index < last; ++index) { ++visits[index]; }
                        if (active[worker.workerIndex].fetch_sub(1) != 1) { ++failures; }
                    });
                if (!result) { ++failures; }
                for (const auto& visit : visits) { if (visit.load() != 1) { ++failures; } }
            }
        });
    }
    for (auto& caller : callers) { caller.join(); }
    EXPECT_EQ(failures.load(), 0u);
}

TEST_F(ParallelForTest, SerialFallbackNeedsNoRuntimeAndStillValidatesAndJoins)
{
    shutdownTaskSystem();
    std::atomic_uint32_t elements{0};
    for (const auto strategy : kStrategies) {
        const auto result = parallelFor(0, 1031, {.minGrain = 7, .maxWorkers = 8, .strategy = strategy},
            [&](uint32_t first, uint32_t last, WorkerContext& worker) {
                EXPECT_EQ(worker.workerIndex, 0u);
                EXPECT_EQ(worker.workerCount, 1u);
                elements += last - first;
            });
        ASSERT_TRUE(result) << result.error();
    }
    EXPECT_EQ(elements.load(), 3u * 1031u);
    const auto error = parallelFor(0, 1, {}, [](uint32_t, uint32_t, WorkerContext&) {
        throw std::runtime_error("inline failure");
    });
    ASSERT_FALSE(error);
    EXPECT_EQ(error.error(), "inline failure");
    EXPECT_EQ(tryGetTaskSystem(), nullptr);
}

TEST_F(ParallelForTest, ShutdownStopsRangesAndDrainsEveryStartedCallback)
{
    auto system = detail::tryAcquireTaskSystem();
    ASSERT_TRUE(system);
    TaskGraph graph("RangeShutdownDrain");
    std::atomic_bool started{false};
    std::atomic_bool release{false};
    std::atomic_uint32_t active{0};
    const auto ranges = addParallelFor(graph, {.name = "Ranges"}, 0, 10000,
        {.minGrain = 1, .maxWorkers = 4, .strategy = PartitionStrategy::AdaptiveHalfSteal},
        [&](uint32_t, uint32_t, WorkerContext&) {
            ++active;
            started = true;
            EXPECT_TRUE(waitUntil([&] { return release.load(); }));
            --active;
        });
    ASSERT_TRUE(ranges);
    auto submitted = system->submit(std::move(graph));
    ASSERT_TRUE(submitted);
    EXPECT_TRUE(waitUntil([&] { return started.load(); }));
    std::jthread shutdown([] { shutdownTaskSystem(); });
    EXPECT_TRUE(waitUntil([&] { return !system->acceptingTasks(); }));
    release = true;
    shutdown.join();
    EXPECT_EQ(active.load(), 0u);
    EXPECT_TRUE(submitted->isComplete());
    const auto joined = submitted->wait();
    ASSERT_TRUE(joined);
    EXPECT_EQ(joined->status, TaskGraphStatus::Cancelled);
}

TEST_F(ParallelForTest, SubmissionRacingShutdownIsRejectedOrFullyDrained)
{
    for (uint32_t round = 0; round < 12; ++round) {
        shutdownTaskSystem();
        ASSERT_TRUE(initializeTaskSystem({.workerCount = round % 2 == 0 ? 1u : 4u}));
        auto system = detail::tryAcquireTaskSystem();
        ASSERT_TRUE(system);
        std::atomic_bool ready{false};
        std::atomic_uint32_t errors{0};
        std::mutex runsMutex;
        std::vector<TaskGraphRun> runs;
        std::vector<std::jthread> callers;
        for (uint32_t caller = 0; caller < 2; ++caller) {
            callers.emplace_back([&] {
                EXPECT_TRUE(waitUntil([&] { return ready.load(); }));
                for (uint32_t iteration = 0; iteration < 32; ++iteration) {
                    TaskGraph graph("ShutdownRace");
                    graph.addTask({.name = "Work"}, [] { std::this_thread::yield(); });
                    auto submitted = system->submit(std::move(graph));
                    if (submitted) {
                        std::lock_guard lock(runsMutex);
                        runs.push_back(std::move(*submitted));
                    } else if (submitted.error().code != TaskErrorCode::ShuttingDown) {
                        ++errors;
                    }
                }
            });
        }
        ready = true;
        std::this_thread::yield();
        shutdownTaskSystem();
        for (auto& caller : callers) { caller.join(); }
        for (const auto& run : runs) {
            EXPECT_TRUE(run.isComplete());
            auto joined = run.wait();
            ASSERT_TRUE(joined);
            EXPECT_NE(joined->status, TaskGraphStatus::Running);
        }
        EXPECT_EQ(errors.load(), 0u);
        EXPECT_FALSE(system->acceptingTasks());
        TaskGraph after("AfterShutdown");
        after.addTask({.name = "Rejected"}, [] {});
        const auto rejected = system->submit(std::move(after));
        ASSERT_FALSE(rejected);
        EXPECT_EQ(rejected.error().code, TaskErrorCode::ShuttingDown);
    }
}

} // namespace
} // namespace metallic::task

#pragma once

#include "Runtime/Task/TaskSystem.h"

#include <functional>
#include <stop_token>
#include <vector>

namespace metallic::task {

enum class PartitionStrategy : uint8_t {
    Static,
    Dynamic,
    AdaptiveHalfSteal,
};

struct WorkerContext {
    // A dense, exclusive scratch slot for this invocation, not a thread affinity.
    uint32_t workerIndex = 0;
    uint32_t workerCount = 1;
    uint64_t executionId = 0;
    TaskNodeHandle node;
};

struct ParallelForWorkerMetrics {
    uint64_t batches = 0;
    uint64_t elements = 0;
    uint64_t activeNanoseconds = 0;
    uint64_t claimAttempts = 0;
    uint64_t stealAttempts = 0;
    uint64_t stealSuccesses = 0;
};

struct ParallelForDiagnostics {
    // Exclusive to one in-flight operation, one writer per lane. Read only
    // after the containing graph has completed/joined.
    std::vector<ParallelForWorkerMetrics> workers;
};

struct ParallelForOptions {
    // Compatibility default, not a performance recommendation. Calibrate per kernel.
    uint32_t minGrain = 1;
    uint32_t maxWorkers = 0;
    PartitionStrategy strategy = PartitionStrategy::Dynamic;
    std::stop_token stopToken;
    ParallelForDiagnostics* diagnostics = nullptr;
};

using ParallelForCallback = std::function<void(uint32_t begin, uint32_t end, WorkerContext& worker)>;

struct ParallelForNodes {
    TaskNodeHandle entry;
    TaskNodeHandle completion;
};

// Add dependencies into entry and out of completion. Captured inputs, outputs and
// optional diagnostics must live until the containing graph completes. Callbacks
// must not wait for other ranges: Static/HalfSteal do not promise execution order.
// An external stop token fails completion (and skips dependents); graph requestStop
// retains the TaskGraph's Cancelled semantics. Already running batches may finish.
// Lane count is fixed when this function is called, using the current runtime.
std::expected<ParallelForNodes, TaskError> addParallelFor(
    TaskGraph& graph, TaskDesc desc, uint32_t begin, uint32_t end,
    const ParallelForOptions& options, ParallelForCallback callback);

// Joins on an external caller. Nested task callbacks and uninitialized runtimes
// execute inline, without blocking a worker or creating another pool. Returns an
// error after all started callbacks have drained on failure/cancellation.
TaskOutcome parallelFor(uint32_t begin, uint32_t end,
    const ParallelForOptions& options, const ParallelForCallback& callback);

} // namespace metallic::task

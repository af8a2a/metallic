#include "Runtime/Task/ParallelFor.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <exception>
#include <memory>

namespace metallic::task {
namespace {

uint64_t packRange(uint32_t begin, uint32_t end)
{
    return (uint64_t(end) << 32) | begin;
}

struct alignas(64) RangeSlot {
    std::atomic_uint64_t range{0};
};

struct RangeState {
    RangeState(uint32_t first, uint32_t last, uint32_t lanes,
        ParallelForOptions settings, ParallelForCallback function)
        : begin(first), end(last), laneCount(lanes), options(settings), callback(std::move(function)),
          slots(std::make_unique<RangeSlot[]>(lanes)), next(first)
    {
        const uint64_t count = uint64_t(end) - begin;
        const uint64_t batches = (count + options.minGrain - 1) / options.minGrain;
        for (uint32_t lane = 0; lane < laneCount; ++lane) {
            const uint64_t firstBatch = batches / lanes * lane + std::min<uint64_t>(lane, batches % lanes);
            const uint64_t lastBatch = firstBatch + batches / lanes + (lane < batches % lanes);
            const auto lower = static_cast<uint32_t>(begin + std::min(count, firstBatch * options.minGrain));
            const auto upper = static_cast<uint32_t>(begin + std::min(count, lastBatch * options.minGrain));
            slots[lane].range.store(packRange(lower, upper), std::memory_order_relaxed);
        }
        if (options.diagnostics) { options.diagnostics->workers.assign(lanes, {}); }
    }

    bool stopped(TaskContext* context) const
    {
        return failed.load(std::memory_order_relaxed) || options.stopToken.stop_requested() ||
            (context ? context->stopRequested() : detail::taskCallbackStopToken().stop_requested());
    }

    bool claim(std::atomic_uint64_t& range, bool steal, uint32_t& first, uint32_t& last,
        ParallelForWorkerMetrics& metrics)
    {
        auto value = range.load(std::memory_order_relaxed);
        for (;;) {
            const auto lower = static_cast<uint32_t>(value);
            const auto upper = static_cast<uint32_t>(value >> 32);
            const uint64_t remaining = uint64_t(upper) - lower;
            if (remaining == 0) { return false; }
            uint64_t size = std::min<uint64_t>(options.minGrain, remaining);
            if (steal) {
                const uint64_t batches = (remaining + options.minGrain - 1) / options.minGrain;
                if (batches < 2) { return false; }
                // Keep the split at a grain boundary from the owner end, even
                // when the uppermost batch is a short, masked SIMD tail.
                size = remaining - (batches / 2) * options.minGrain;
                if (size < options.minGrain) { return false; }
            }
            first = steal ? static_cast<uint32_t>(upper - size) : lower;
            last = steal ? upper : static_cast<uint32_t>(lower + size);
            const uint64_t replacement = steal ? packRange(lower, first) : packRange(last, upper);
            if (options.diagnostics) { ++metrics.claimAttempts; }
            // Both endpoints share one CAS linearization point. No input/output
            // publication relies on this atomic; TaskGraph owns those lifetimes.
            if (range.compare_exchange_weak(value, replacement, std::memory_order_relaxed)) { return true; }
        }
    }

    TaskOutcome run(uint32_t lane, TaskContext* context)
    {
        WorkerContext worker{.workerIndex = lane, .workerCount = laneCount,
            .executionId = context ? context->executionId() : 0,
            .node = context ? context->node() : TaskNodeHandle{}};
        ParallelForWorkerMetrics metrics;
        auto invoke = [&](uint32_t first, uint32_t last) {
            if (options.diagnostics) {
                const auto start = TaskClock::now();
                callback(first, last, worker);
                metrics.activeNanoseconds += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    TaskClock::now() - start).count();
                ++metrics.batches;
                metrics.elements += uint64_t(last) - first;
            } else {
                callback(first, last, worker);
            }
        };
        try {
            if (options.strategy == PartitionStrategy::Static) {
                const auto value = slots[lane].range.load(std::memory_order_relaxed);
                uint32_t first = static_cast<uint32_t>(value);
                const auto upper = static_cast<uint32_t>(value >> 32);
                while (first < upper && !stopped(context)) {
                    const auto last = static_cast<uint32_t>(first + std::min<uint64_t>(options.minGrain, uint64_t(upper) - first));
                    invoke(first, last);
                    first = last;
                }
            } else {
                while (!stopped(context)) {
                    uint32_t first, last;
                    if (options.strategy == PartitionStrategy::Dynamic) {
                        // CAS rather than fetch_add: even the last claim cannot
                        // wrap at UINT32_MAX or execute a rounded dispatch tail.
                        auto lower = next.load(std::memory_order_relaxed);
                        do {
                            if (lower == end) { break; }
                            last = static_cast<uint32_t>(lower + std::min<uint64_t>(options.minGrain, uint64_t(end) - lower));
                            if (options.diagnostics) { ++metrics.claimAttempts; }
                        } while (!next.compare_exchange_weak(lower, last, std::memory_order_relaxed));
                        if (lower == end) { break; }
                        first = lower;
                    } else if (!claim(slots[lane].range, false, first, last, metrics)) {
                        bool found = false;
                        for (uint32_t offset = 1; offset < laneCount && !stopped(context); ++offset) {
                            if (options.diagnostics) { ++metrics.stealAttempts; }
                            if (claim(slots[(lane + offset) % laneCount].range, true, first, last, metrics)) {
                                if (options.diagnostics) { ++metrics.stealSuccesses; }
                                slots[lane].range.store(packRange(first, last), std::memory_order_relaxed);
                                found = true;
                                break;
                            }
                        }
                        if (!found) { break; }
                        continue;
                    }
                    invoke(first, last);
                }
            }
        } catch (...) {
            failed.store(true, std::memory_order_relaxed);
            if (options.diagnostics) { options.diagnostics->workers[lane] = metrics; }
            throw;
        }
        if (options.diagnostics) { options.diagnostics->workers[lane] = metrics; }
        if (options.stopToken.stop_requested() ||
            (!context && detail::taskCallbackStopToken().stop_requested())) {
            return std::unexpected("parallelFor stop requested");
        }
        return {};
    }

    uint32_t begin, end, laneCount;
    ParallelForOptions options;
    ParallelForCallback callback;
    std::unique_ptr<RangeSlot[]> slots;
    std::atomic_uint32_t next;
    std::atomic_bool failed{false};
};

TaskOutcome validate(uint32_t begin, uint32_t end, const ParallelForOptions& options,
    const ParallelForCallback& callback)
{
    if (begin > end) { return std::unexpected("parallelFor requires begin <= end"); }
    if (options.minGrain == 0) { return std::unexpected("parallelFor minGrain must be nonzero"); }
    if (!callback) { return std::unexpected("parallelFor requires a callback"); }
    switch (options.strategy) {
    case PartitionStrategy::Static:
    case PartitionStrategy::Dynamic:
    case PartitionStrategy::AdaptiveHalfSteal: return {};
    }
    return std::unexpected("parallelFor partition strategy is invalid");
}

uint32_t lanesFor(uint32_t begin, uint32_t end, const ParallelForOptions& options, uint32_t available)
{
    const uint64_t batches = (uint64_t(end) - begin + options.minGrain - 1) / options.minGrain;
    const uint32_t limit = options.maxWorkers == 0 ? available : std::min(available, options.maxWorkers);
    return static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(limit, batches)));
}

} // namespace

std::expected<ParallelForNodes, TaskError> addParallelFor(
    TaskGraph& graph, TaskDesc desc, uint32_t begin, uint32_t end,
    const ParallelForOptions& options, ParallelForCallback callback)
{
    auto validated = validate(begin, end, options, callback);
    if (!validated || desc.name.empty() || graph.id() == 0) {
        return std::unexpected(TaskError{TaskErrorCode::InvalidGraph,
            validated ? "parallelFor requires a valid graph and task name" : validated.error()});
    }
    auto system = detail::tryAcquireTaskSystem();
    const uint32_t lanes = lanesFor(begin, end, options, system ? system->workerCount() : 1);
    auto state = std::make_shared<RangeState>(begin, end, lanes, options, std::move(callback));
    const std::string name = desc.name;
    desc.name = name + ".entry";
    const auto entry = graph.addTask(desc, [] {});
    desc.name = name + ".completion";
    const auto completion = graph.addTask(desc, [state]() -> TaskOutcome {
        if (state->options.stopToken.stop_requested()) { return std::unexpected("parallelFor stop requested"); }
        return {};
    });
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        desc.name = name + ".range." + std::to_string(lane);
        const auto node = graph.addTask(desc, [state, lane](TaskContext& context) { return state->run(lane, &context); });
        // Newly created handles belong to this graph and every edge is unique.
        const auto in = graph.addDependency(entry, node);
        const auto out = graph.addDependency(node, completion);
        assert(in && out);
    }
    return ParallelForNodes{entry, completion};
}

TaskOutcome parallelFor(uint32_t begin, uint32_t end,
    const ParallelForOptions& options, const ParallelForCallback& callback)
{
    if (auto validated = validate(begin, end, options, callback); !validated) { return validated; }
    auto system = detail::tryAcquireTaskSystem();
    if (isInsideTaskCallback() || !system || options.maxWorkers == 1 || begin == end) {
        RangeState state(begin, end, 1, options, callback);
        try { return state.run(0, nullptr); }
        catch (const std::exception& error) { return std::unexpected(error.what()); }
        catch (...) { return std::unexpected("parallelFor callback threw an unknown exception"); }
    }
    TaskGraph graph("ParallelFor");
    auto nodes = addParallelFor(graph, {.name = "ParallelFor", .category = "CPU range"}, begin, end, options, callback);
    if (!nodes) { return std::unexpected(nodes.error().message); }
    auto submitted = system->submit(std::move(graph));
    if (!submitted) { return std::unexpected(submitted.error().message); }
    const auto joined = submitted->wait();
    if (!joined) { return std::unexpected(joined.error().message); }
    if (joined->status == TaskGraphStatus::Succeeded) { return {}; }
    for (const auto& node : joined->nodes) {
        if (node.state == TaskState::Failed) { return std::unexpected(node.error); }
    }
    return std::unexpected("parallelFor task graph cancelled");
}

} // namespace metallic::task

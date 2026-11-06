#include "Runtime/Task/TaskSystem.h"
#include "LegacyMeshletBuildParallel.h"
#if METALLIC_BENCH_HAS_PARALLEL_FOR
#include "Runtime/Task/ParallelFor.h"
#endif

#include <exec/async_scope.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>
#if METALLIC_BENCH_HAS_TBB
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;
using namespace metallic::task;
bool gDiagnostics = false;
double gTaskRuntimeInitNs = 0;
double gRawPoolInitNs = 0;
double gLegacyInitNs = 0;

double elapsedNs(Clock::time_point begin, Clock::time_point end)
{
    return std::chrono::duration<double, std::nano>(end - begin).count();
}

double processNs()
{
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) {
        throw std::runtime_error("GetProcessTimes failed");
    }
    const auto ticks = [](FILETIME value) { return (uint64_t(value.dwHighDateTime) << 32) | value.dwLowDateTime; };
    return double(ticks(kernel) + ticks(user)) * 100.;
#else
    return double(std::clock()) * (1.e9 / CLOCKS_PER_SEC);
#endif
}

uint32_t mix(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

struct Workload {
    Workload(std::string name, uint32_t count) : name(std::move(name)), input(count), output(count)
    {
        for (uint32_t index = 0; index < count; ++index) { input[index] = mix(index + 73u); }
    }

    void run(uint32_t begin, uint32_t end)
    {
        if (name == "bandwidth") {
            for (uint32_t index = begin; index < end; ++index) { output[index] = input[index] * 3u + 7u; }
            return;
        }
        if (name == "locality") {
            for (uint32_t index = begin; index < end; ++index) {
                uint32_t value = input[index];
                for (uint32_t offset = 1; offset < 16; ++offset) {
                    value += input[(index + offset) % input.size()];
                }
                output[index] = value;
            }
            return;
        }
        for (uint32_t index = begin; index < end; ++index) {
            uint32_t value = input[index];
            const uint32_t rounds = name == "random" ? 1u + (value & 127u)
                : name == "long_tail" ? ((index % 1024u < 8u) ? 8192u : 8u) : 64u;
            for (uint32_t round = 0; round < rounds; ++round) { value = mix(value + round); }
            output[index] = value;
        }
    }

    uint64_t checksum() const
    {
        return std::accumulate(output.begin(), output.end(), uint64_t{0});
    }

    std::string name;
    std::vector<uint32_t> input;
    std::vector<uint32_t> output;
};

struct Sample {
    double wallNs = 0;
    double cpuNs = 0;
    std::vector<double> queueNs;
    std::map<uint64_t, double> workerActiveNs;
    uint64_t checksum = 0;
    uint64_t batches = 0;
    int64_t claimAttempts = -1;
    int64_t stealAttempts = -1;
    int64_t stealSuccesses = -1;
};

template<class Function>
Sample measure(Function&& function)
{
    const double cpuBegin = processNs();
    const auto begin = Clock::now();
    Sample sample = function();
    const auto end = Clock::now();
    sample.wallNs = elapsedNs(begin, end);
    sample.cpuNs = processNs() - cpuBegin;
    return sample;
}

void requireOutcome(const TaskOutcome& outcome)
{
    if (!outcome) { throw std::runtime_error(outcome.error()); }
}

Sample runTaskGraph(Workload& work, uint32_t tasks, bool fanIn)
{
    TaskGraph graph("CPUJobBench");
    TaskNodeHandle root{}, join{};
    if (fanIn) {
        root = graph.addTask({.name = "root", .category = "bench"}, [] {});
        join = graph.addTask({.name = "join", .category = "bench"}, [] {});
    }
    for (uint32_t index = 0; index < tasks; ++index) {
        const uint32_t begin = uint64_t(work.input.size()) * index / tasks;
        const uint32_t end = uint64_t(work.input.size()) * (index + 1u) / tasks;
        const auto node = graph.addTask({.name = "range", .category = "bench"}, [&, begin, end] { work.run(begin, end); });
        if (fanIn && (!graph.addDependency(root, node) || !graph.addDependency(node, join))) {
            throw std::runtime_error("DAG dependency rejected");
        }
    }
    TaskGraphSnapshot snapshot;
    Sample sample = measure([&] {
        auto run = taskSystem().submit(std::move(graph));
        if (!run) { throw std::runtime_error(run.error().message); }
        auto completion = run->wait();
        if (!completion || completion->status != TaskGraphStatus::Succeeded) {
            throw std::runtime_error("TaskGraph benchmark failed");
        }
        snapshot = std::move(*completion);
        return Sample{};
    });
    for (const auto& node : snapshot.nodes) {
        sample.queueNs.push_back(elapsedNs(node.scheduledTime, node.startTime));
        sample.workerActiveNs[node.workerThreadId] += elapsedNs(node.startTime, node.finishTime);
    }
    sample.batches = snapshot.nodes.size();
    sample.checksum = work.checksum();
    return sample;
}

// Deliberately separate from TaskGraph: isolates upstream scheduler submission
// policy without attributing graph validation/snapshots to the worker deque.
Sample runRawPool(exec::static_thread_pool& pool, uint32_t workers, Workload& work,
    uint32_t tasks, bool directed, bool workerFanout)
{
    struct Timing {
        Clock::time_point scheduled{}, begin{}, end{};
        uint64_t thread = 0;
    };
    std::vector<Timing> timings(tasks);
    exec::async_scope scope;
    const auto enqueue = [&](uint32_t index) {
        auto& timing = timings[index];
        if (gDiagnostics) { timing.scheduled = Clock::now(); }
        const auto scheduler = directed ? pool.get_scheduler_on_thread(index % workers) : pool.get_scheduler();
        scope.spawn(stdexec::schedule(scheduler) | stdexec::then([&, index] {
            auto& item = timings[index];
            if (gDiagnostics) {
                item.begin = Clock::now();
                item.thread = std::hash<std::thread::id>{}(std::this_thread::get_id());
            }
            work.run(uint64_t(work.input.size()) * index / tasks, uint64_t(work.input.size()) * (index + 1u) / tasks);
            if (gDiagnostics) { item.end = Clock::now(); }
        }));
    };
    Sample sample = measure([&] {
        if (workerFanout) {
            scope.spawn(stdexec::schedule(pool.get_scheduler_on_thread(0)) | stdexec::then([&] {
                for (uint32_t index = 0; index < tasks; ++index) { enqueue(index); }
            }));
        } else {
            for (uint32_t index = 0; index < tasks; ++index) { enqueue(index); }
        }
        stdexec::sync_wait(scope.on_empty());
        return Sample{};
    });
    if (gDiagnostics) {
        for (const auto& item : timings) {
            sample.queueNs.push_back(elapsedNs(item.scheduled, item.begin));
            sample.workerActiveNs[item.thread] += elapsedNs(item.begin, item.end);
        }
    }
    sample.batches = tasks;
    sample.checksum = work.checksum();
    return sample;
}

Sample runLegacy(metallic::bench::MeshletBuildParallel& pool, Workload& work, uint32_t grain)
{
    const uint32_t batches = (work.input.size() + grain - 1u) / grain;
    std::vector<double> active(pool.workerCount());
    Sample sample = measure([&] {
        pool.forEach(batches, [&](size_t batch, size_t worker) {
            const auto begin = gDiagnostics ? Clock::now() : Clock::time_point{};
            const uint32_t first = uint64_t(batch) * grain;
            work.run(first, std::min<uint64_t>(work.input.size(), uint64_t(first) + grain));
            if (gDiagnostics) { active[worker] += elapsedNs(begin, Clock::now()); }
        });
        return Sample{};
    });
    if (gDiagnostics) {
        for (size_t worker = 0; worker < active.size(); ++worker) { sample.workerActiveNs[worker] = active[worker]; }
    }
    sample.batches = batches;
    sample.checksum = work.checksum();
    return sample;
}

#if METALLIC_BENCH_HAS_PARALLEL_FOR
Sample runRange(Workload& work, uint32_t workers, uint32_t grain, PartitionStrategy strategy)
{
    ParallelForDiagnostics diagnostics;
    Sample sample = measure([&] {
        requireOutcome(parallelFor(0, static_cast<uint32_t>(work.input.size()),
            {.minGrain = grain, .maxWorkers = workers, .strategy = strategy,
                .diagnostics = gDiagnostics ? &diagnostics : nullptr},
            [&](uint32_t begin, uint32_t end, WorkerContext& worker) {
                (void)worker;
                work.run(begin, end);
            }));
        return Sample{};
    });
    for (size_t worker = 0; worker < diagnostics.workers.size(); ++worker) {
        sample.workerActiveNs[worker] = diagnostics.workers[worker].activeNanoseconds;
        sample.batches += diagnostics.workers[worker].batches;
    }
    if (gDiagnostics) {
        sample.claimAttempts = sample.stealAttempts = sample.stealSuccesses = 0;
        for (const auto& worker : diagnostics.workers) {
            sample.claimAttempts += worker.claimAttempts;
            sample.stealAttempts += worker.stealAttempts;
            sample.stealSuccesses += worker.stealSuccesses;
        }
    }
    sample.checksum = work.checksum();
    return sample;
}
#endif

#if METALLIC_BENCH_HAS_TBB
Sample runTBB(Workload& work, uint32_t grain)
{
    Sample sample = measure([&] {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<uint32_t>(0, work.input.size(), grain),
            [&](const auto& range) { work.run(range.begin(), range.end()); });
        return Sample{};
    });
    sample.checksum = work.checksum();
    return sample;
}
#endif

double percentile(std::vector<double> values, double fraction)
{
    if (values.empty()) { return 0; }
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(fraction * (values.size() - 1))];
}

void writeSample(const std::string& domain, const std::string& backend, const Workload& work,
    uint32_t workers, uint32_t grain, int repeat, const Sample& sample)
{
    double active = 0;
    for (const auto& [worker, ns] : sample.workerActiveNs) { active += ns; }
    std::cout << std::setprecision(12) << "{\"domain\":\"" << domain << "\",\"backend\":\"" << backend
        << "\",\"workload\":\"" << work.name << "\",\"workers\":" << workers
        << ",\"grain\":" << grain << ",\"elements\":" << work.input.size()
        << ",\"repeat\":" << repeat << ",\"cache\":\"" << (repeat < 0 ? "first_call" : "warm")
        << "\",\"diagnostics\":" << (gDiagnostics ? "true" : "false")
        << ",\"wallNs\":" << sample.wallNs << ",\"processCpuNs\":" << sample.cpuNs
        << ",\"runtimeConstructionNs\":" << gTaskRuntimeInitNs
        << ",\"rawPoolConstructionNs\":" << gRawPoolInitNs
        << ",\"legacyPoolConstructionNs\":" << gLegacyInitNs
        << ",\"batches\":";
    if (sample.batches) { std::cout << sample.batches; } else { std::cout << "null"; }
    std::cout << ",\"checksum\":" << sample.checksum << ",\"activeNs\":";
    if (sample.workerActiveNs.empty()) { std::cout << "null"; } else { std::cout << active; }
    const auto optionalCounter = [](const char* key, int64_t value) {
        std::cout << ",\"" << key << "\":";
        if (value < 0) { std::cout << "null"; } else { std::cout << value; }
    };
    optionalCounter("claimAttempts", sample.claimAttempts);
    optionalCounter("stealAttempts", sample.stealAttempts);
    optionalCounter("stealSuccesses", sample.stealSuccesses);
    std::cout << ",\"queueP50Ns\":";
    if (sample.queueNs.empty()) { std::cout << "null,\"queueP95Ns\":null,\"queueP99Ns\":null"; }
    else {
        std::cout << percentile(sample.queueNs, .5) << ",\"queueP95Ns\":" << percentile(sample.queueNs, .95)
            << ",\"queueP99Ns\":" << percentile(sample.queueNs, .99);
    }
    std::cout << ",\"workerActiveNs\":{";
    bool first = true;
    for (const auto& [worker, ns] : sample.workerActiveNs) {
        if (!first) { std::cout << ','; }
        std::cout << '\"' << worker << "\":" << ns;
        first = false;
    }
    std::cout << "}}" << std::endl;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        uint32_t workers = 4, repeats = 12, count = 65536, grain = 64, tasks = 256;
        std::string mode = "all";
        std::string selectedWorkload = "all";
        for (int index = 1; index < argc; index += 2) {
            if (index + 1 == argc) { throw std::invalid_argument("missing argument value"); }
            const std::string key = argv[index], value = argv[index + 1];
            if (key == "--workers") { workers = std::stoul(value); }
            else if (key == "--repeats") { repeats = std::stoul(value); }
            else if (key == "--count") { count = std::stoul(value); }
            else if (key == "--grain") { grain = std::stoul(value); }
            else if (key == "--tasks") { tasks = std::stoul(value); }
            else if (key == "--mode") { mode = value; }
            else if (key == "--workload") { selectedWorkload = value; }
            else if (key == "--diagnostics") { gDiagnostics = value == "on"; }
            else { throw std::invalid_argument("unknown option " + key); }
        }
        if (!workers || !grain || !count || !tasks || !repeats) { throw std::invalid_argument("arguments must be positive"); }
        if (mode != "all" && mode != "baseline" && mode != "range") { throw std::invalid_argument("invalid mode"); }
        if (selectedWorkload != "all" && selectedWorkload != "uniform" && selectedWorkload != "random" &&
            selectedWorkload != "long_tail" && selectedWorkload != "bandwidth" && selectedWorkload != "locality") {
            throw std::invalid_argument("invalid workload");
        }
        auto constructionBegin = Clock::now();
        auto initialized = initializeTaskSystem({.workerCount = workers});
        gTaskRuntimeInitNs = elapsedNs(constructionBegin, Clock::now());
        if (!initialized) { throw std::runtime_error(initialized.error().message); }
        struct RuntimeScope { ~RuntimeScope() { shutdownTaskSystem(); } } runtimeScope;
        constructionBegin = Clock::now();
        exec::static_thread_pool rawPool(workers);
        gRawPoolInitNs = elapsedNs(constructionBegin, Clock::now());
        constructionBegin = Clock::now();
        metallic::bench::MeshletBuildParallel legacy(workers);
        gLegacyInitNs = elapsedNs(constructionBegin, Clock::now());
#if METALLIC_BENCH_HAS_TBB
        oneapi::tbb::global_control control(oneapi::tbb::global_control::max_allowed_parallelism, workers);
#endif
        for (const auto name : {"uniform", "random", "long_tail", "bandwidth", "locality"}) {
            if (selectedWorkload != "all" && selectedWorkload != name) { continue; }
            // Bandwidth uses a 128 MiB input + output working set, independent
            // from compute counts; this deliberately exceeds small CPU caches.
            const uint32_t workloadCount = std::string(name) == "bandwidth" ? std::max(count, 16u * 1024u * 1024u) : count;
            Workload work(name, workloadCount);
            work.run(0, workloadCount);
            const uint64_t expectedChecksum = work.checksum();
            struct Case { std::string domain, backend; std::function<Sample()> run; };
            std::vector<Case> cases;
            if (mode != "range") {
                for (bool fanIn : {false, true}) {
                    const std::string domain = fanIn ? "fanout_join" : "flat";
                    cases.push_back({domain, "task_graph_directed", [&, fanIn] { return runTaskGraph(work, tasks, fanIn); }});
                    for (bool directed : {false, true}) {
                        cases.push_back({domain, directed ? "stdexec_directed" : "stdexec_default",
                            [&, directed, fanIn] { return runRawPool(rawPool, workers, work, tasks, directed, fanIn); }});
                    }
                }
            }
            cases.push_back({"range", "legacy_atomic", [&] { return runLegacy(legacy, work, grain); }});
#if METALLIC_BENCH_HAS_TBB
            cases.push_back({"range", "oneTBB", [&] { return runTBB(work, grain); }});
#endif
#if METALLIC_BENCH_HAS_PARALLEL_FOR
            if (mode != "baseline") {
                cases.push_back({"range", "static", [&] { return runRange(work, workers, grain, PartitionStrategy::Static); }});
                cases.push_back({"range", "dynamic", [&] { return runRange(work, workers, grain, PartitionStrategy::Dynamic); }});
                cases.push_back({"range", "adaptive_half_steal", [&] { return runRange(work, workers, grain, PartitionStrategy::AdaptiveHalfSteal); }});
            }
#endif
            for (int repeat = -1; repeat < static_cast<int>(repeats); ++repeat) {
                // Rotate order deterministically; every round contains every
                // candidate, preventing all-A-then-all-B thermal drift.
                for (size_t offset = 0; offset < cases.size(); ++offset) {
                    const auto& candidate = cases[(offset + repeat + 1) % cases.size()];
                    std::fill(work.output.begin(), work.output.end(), 0u);
                    const Sample sample = candidate.run();
                    if (sample.checksum != expectedChecksum) { throw std::runtime_error("result checksum differs from serial reference"); }
                    writeSample(candidate.domain, candidate.backend, work, workers, grain, repeat, sample);
                }
            }
        }
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
    return 0;
}

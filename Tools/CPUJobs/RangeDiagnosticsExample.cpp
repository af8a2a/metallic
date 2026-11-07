#include "Runtime/Task/ParallelFor.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace metallic::task;

uint64_t ticks(TaskClock::time_point value)
{
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count());
}

void writeOperation(std::ostream& out, const ParallelForDiagnostics& diagnostics, const TaskGraphSnapshot& graph)
{
    out << "{\"operationId\":" << diagnostics.operationId << ",\"graphId\":" << diagnostics.graphId
        << ",\"name\":\"" << diagnostics.operationName << "\",\"backend\":\"" << diagnostics.backendLabel
        << "\",\"begin\":" << diagnostics.begin << ",\"end\":" << diagnostics.end << ",\"minGrain\":"
        << diagnostics.minGrain << ",\"strategy\":" << unsigned(diagnostics.strategy)
        << ",\"createdNs\":" << ticks(diagnostics.createdTime) << ",\"lanes\":[";
    bool first = true;
    for (const auto& lane : diagnostics.workers) {
        if (!lane.node.valid() || lane.executionId != graph.executionId || lane.node.graphId != graph.graphId) {
            throw std::runtime_error("Range identity did not match containing graph");
        }
        const auto& node = graph.nodes.at(lane.node.nodeIndex);
        if (lane.workerThreadId != node.workerThreadId || lane.startTime < node.startTime ||
            lane.finishTime < lane.startTime || lane.finishTime > node.finishTime) {
            throw std::runtime_error("Range identity/timing did not lie inside its owning node");
        }
        if (!first) { out << ','; }
        first = false;
        out << "{\"lane\":" << lane.workerIndex << ",\"executionId\":" << lane.executionId
            << ",\"nodeIndex\":" << lane.node.nodeIndex << ",\"threadId\":" << lane.workerThreadId
            << ",\"inline\":" << (lane.inlineExecution ? "true" : "false")
            << ",\"startNs\":" << ticks(lane.startTime) << ",\"finishNs\":" << ticks(lane.finishTime)
            << ",\"queueDelayNs\":";
        if (lane.inlineExecution) { out << "null"; }
        else { out << ticks(node.startTime) - ticks(node.scheduledTime); }
        out << ",\"elements\":" << lane.elements << ",\"batches\":" << lane.batches
            << ",\"activeNs\":" << lane.activeNanoseconds << ",\"claims\":" << lane.claimAttempts
            << ",\"stealAttempts\":" << lane.stealAttempts << ",\"stealSuccesses\":" << lane.stealSuccesses << '}';
    }
    out << "]}";
}

int main(int argc, char** argv)
{
    if (argc != 2) { return 2; }
    try {
        auto initialized = initializeTaskSystem({.workerCount = 4});
        if (!initialized) { throw std::runtime_error(initialized.error().message); }
        struct Scope { ~Scope() { shutdownTaskSystem(); } } scope;
        ParallelForDiagnostics outer, nested;
        std::vector<uint32_t> output(1031);
        TaskGraph graph("Range diagnostics example");
        auto ranges = addParallelFor(graph, {.name = "Uniform values"}, 0, uint32_t(output.size()),
            {.minGrain = 8, .maxWorkers = 4, .strategy = PartitionStrategy::AdaptiveHalfSteal,
             .backendLabel = "cpp-integer-demo", .diagnostics = &outer},
            [&](uint32_t begin, uint32_t end, WorkerContext&) {
                for (uint32_t index = begin; index < end; ++index) { output[index] = index ^ 73u; }
            });
        if (!ranges) { throw std::runtime_error(ranges.error().message); }
        const auto consumer = graph.addTask({.name = "Nested consumer"}, [&] {
            auto result = parallelFor(0, uint32_t(output.size()),
                {.minGrain = 32, .maxWorkers = 4, .backendLabel = "cpp-inline-check", .diagnostics = &nested},
                [&](uint32_t begin, uint32_t end, WorkerContext&) {
                    for (uint32_t index = begin; index < end; ++index) {
                        if (output[index] != (index ^ 73u)) { throw std::runtime_error("Incorrect publication"); }
                    }
                });
            if (!result) { throw std::runtime_error(result.error()); }
        });
        if (!graph.addDependency(ranges->completion, consumer)) { throw std::runtime_error("Dependency failed"); }
        auto submitted = taskSystem().submit(std::move(graph));
        if (!submitted) { throw std::runtime_error(submitted.error().message); }
        auto completed = submitted->wait();
        if (!completed || completed->status != TaskGraphStatus::Succeeded) { throw std::runtime_error("Graph failed"); }
        std::ofstream out(argv[1], std::ios::binary);
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out << "{\"schema\":1,\"scope\":\"Correctness/identity diagnostic example, not a performance sample\","
            "\"clock\":\"TaskClock steady_clock epoch nanoseconds\","
            "\"association\":\"executionId + graphId/nodeIndex; inline intervals borrow parent node\","
            "\"executionId\":" << completed->executionId << ",\"graphId\":" << completed->graphId << ",\"tasks\":[";
        bool first = true;
        for (const auto& node : completed->nodes) {
            if (!first) { out << ','; }
            first = false;
            out << "{\"nodeIndex\":" << node.handle.nodeIndex << ",\"name\":\"" << node.desc.name
                << "\",\"threadId\":" << node.workerThreadId << ",\"scheduledNs\":" << ticks(node.scheduledTime)
                << ",\"startNs\":" << ticks(node.startTime) << ",\"finishNs\":" << ticks(node.finishTime) << '}';
        }
        out << "],\"operations\":[";
        writeOperation(out, outer, *completed);
        out << ',';
        writeOperation(out, nested, *completed);
        out << "],\"correct\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

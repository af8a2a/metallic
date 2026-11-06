#include "AABBTestData.h"
#include "Runtime/Task/ParallelFor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace {

TEST(CPUKernels, RangeSchedulingKeepsKernelOutputEquivalent)
{
    using namespace metallic;
    struct RuntimeScope {
        ~RuntimeScope() { task::shutdownTaskSystem(); }
    } runtime;
    ASSERT_EQ(task::tryGetTaskSystem(), nullptr);
    constexpr uint32_t kCount = 1031;
    cpu::test::AABBTestData data(kCount);
    cpu::transformAABBs(data.input(), data.output(), 0, kCount, cpu::CPUKernelBackend::Scalar);
    const auto reference = data.outputs;
    for (uint32_t workers : {1u, 2u, 4u, 8u}) {
        task::shutdownTaskSystem();
        ASSERT_TRUE(task::initializeTaskSystem({.workerCount = workers}));
        for (auto strategy : {task::PartitionStrategy::Static, task::PartitionStrategy::Dynamic,
                 task::PartitionStrategy::AdaptiveHalfSteal}) {
            for (auto backend : {cpu::CPUKernelBackend::Scalar, cpu::CPUKernelBackend::AutoVectorized,
                     cpu::CPUKernelBackend::ISPC, cpu::CPUKernelBackend::Automatic}) {
                SCOPED_TRACE(cpu::cpuKernelBackendName(backend));
                task::TaskGraph graph("CPU kernel composition");
                const auto producer = graph.addTask({.name = "prepare outputs"}, [&] {
                    for (auto& stream : data.outputs) { std::fill(stream.begin(), stream.end(), -12345.0f); }
                });
                auto ranges = task::addParallelFor(graph, {.name = "transform AABBs"}, 0, kCount,
                    {.minGrain = 7, .maxWorkers = workers, .strategy = strategy},
                    [&](uint32_t begin, uint32_t end, task::WorkerContext&) {
                        cpu::transformAABBs(data.input(), data.output(), begin, end, backend);
                    });
                ASSERT_TRUE(ranges);
                ASSERT_TRUE(graph.addDependency(producer, ranges->entry));
                bool consumerRan = false, consumerSawCompleteOutput = true;
                const auto consumer = graph.addTask({.name = "consume bounds"}, [&] {
                    for (uint32_t axis = 0; axis < 6; ++axis) {
                        for (uint32_t index = 0; index < kCount; ++index) {
                            if (std::abs(data.outputs[axis][index] - reference[axis][index]) >
                                1e-4f * (1.0f + std::abs(reference[axis][index]))) {
                                consumerSawCompleteOutput = false;
                            }
                        }
                    }
                    consumerRan = true;
                });
                ASSERT_TRUE(graph.addDependency(ranges->completion, consumer));
                auto run = task::taskSystem().submit(std::move(graph));
                ASSERT_TRUE(run);
                auto snapshot = run->wait();
                ASSERT_TRUE(snapshot);
                ASSERT_EQ(snapshot->status, task::TaskGraphStatus::Succeeded);
                EXPECT_TRUE(consumerRan);
                EXPECT_TRUE(consumerSawCompleteOutput);
                for (uint32_t axis = 0; axis < 6; ++axis) {
                    for (uint32_t index = 0; index < kCount; ++index) {
                        // The independent double oracle is exercised separately;
                        // this test checks composition, publication and range tails.
                        EXPECT_NEAR(data.outputs[axis][index], reference[axis][index],
                            1e-4f * (1.0f + std::abs(reference[axis][index])));
                    }
                }
            }
        }
    }
}

} // namespace

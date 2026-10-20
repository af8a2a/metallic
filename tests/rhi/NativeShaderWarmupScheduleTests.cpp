#include "Runtime/Render/Core/NativeShaderWarmupSchedule.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <queue>
#include <vector>

namespace metallic::tests {
namespace {

using render::detail::NativeShaderWarmupPriority;
using render::detail::nativeShaderWarmupPriorityBefore;

TEST(NativeShaderWarmupSchedule, HistoricalDurationOutranksModuleSize)
{
    const NativeShaderWarmupPriority slowSmall{80'000'000'000, 100};
    const NativeShaderWarmupPriority fastLarge{1'000'000'000, 100'000};
    EXPECT_TRUE(nativeShaderWarmupPriorityBefore(slowSmall, fastLarge));
    EXPECT_FALSE(nativeShaderWarmupPriorityBefore(fastLarge, slowSmall));
}

TEST(NativeShaderWarmupSchedule, UnknownHistoryRetainsLargestModuleFirst)
{
    const NativeShaderWarmupPriority small{0, 100};
    const NativeShaderWarmupPriority large{0, 100'000};
    EXPECT_TRUE(nativeShaderWarmupPriorityBefore(large, small));
    EXPECT_FALSE(nativeShaderWarmupPriorityBefore(small, large));
    EXPECT_FALSE(nativeShaderWarmupPriorityBefore(large, large));
}

TEST(NativeShaderWarmupSchedule, EqualDurationsUseModuleSize)
{
    const NativeShaderWarmupPriority small{1'000'000, 100};
    const NativeShaderWarmupPriority large{1'000'000, 100'000};
    EXPECT_TRUE(nativeShaderWarmupPriorityBefore(large, small));
    EXPECT_FALSE(nativeShaderWarmupPriorityBefore(small, large));
}

TEST(NativeShaderWarmupSchedule, MixedHistoryHasStrictWeakOrdering)
{
    const std::array priorities{
        NativeShaderWarmupPriority{0, 100'000},
        NativeShaderWarmupPriority{1, 1},
        NativeShaderWarmupPriority{5, 50},
        NativeShaderWarmupPriority{5, 100},
        NativeShaderWarmupPriority{0, 10},
        NativeShaderWarmupPriority{5, 100},
    };
    // Even the shortest measured task precedes any unmeasured task. Duration
    // and SPIR-V size cannot be compared as if they shared a unit.
    EXPECT_TRUE(nativeShaderWarmupPriorityBefore(priorities[1], priorities[0]));
    for (const auto& left : priorities) {
        EXPECT_FALSE(nativeShaderWarmupPriorityBefore(left, left));
        for (const auto& right : priorities) {
            const bool before = nativeShaderWarmupPriorityBefore(left, right);
            const bool after = nativeShaderWarmupPriorityBefore(right, left);
            EXPECT_FALSE(before && after);
            for (const auto& last : priorities) {
                if (before && nativeShaderWarmupPriorityBefore(right, last)) {
                    EXPECT_TRUE(nativeShaderWarmupPriorityBefore(left, last));
                }
                const bool equivalent = !before && !after;
                const bool rightEquivalentLast = !nativeShaderWarmupPriorityBefore(right, last) &&
                    !nativeShaderWarmupPriorityBefore(last, right);
                if (equivalent && rightEquivalentLast) {
                    EXPECT_FALSE(nativeShaderWarmupPriorityBefore(left, last));
                    EXPECT_FALSE(nativeShaderWarmupPriorityBefore(last, left));
                }
            }
        }
    }
}

TEST(NativeShaderWarmupSchedule, BatchAndReadyQueuePreserveSameStableOrder)
{
    struct Task {
        NativeShaderWarmupPriority priority;
        size_t sequence;
    };
    const std::vector<Task> tasks{
        {{0, 100'000}, 0},
        {{1, 1}, 1},
        {{80'000'000'000, 10}, 2},
        {{80'000'000'000, 10}, 3},
        {{0, 100'000}, 4},
        {{80'000'000'000, 100}, 5},
        {{0, 10}, 6},
    };
    auto sorted = tasks;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Task& left, const Task& right) {
        return nativeShaderWarmupPriorityBefore(left.priority, right.priority);
    });
    const auto queueOrder = [](const Task& left, const Task& right) {
        if (nativeShaderWarmupPriorityBefore(left.priority, right.priority)) { return false; }
        if (nativeShaderWarmupPriorityBefore(right.priority, left.priority)) { return true; }
        return left.sequence > right.sequence;
    };
    std::priority_queue<Task, std::vector<Task>, decltype(queueOrder)> ready(queueOrder);
    for (const auto& task : tasks) { ready.push(task); }
    const std::array<size_t, 7> expected{5, 2, 3, 1, 0, 4, 6};
    for (size_t index = 0; index < expected.size(); ++index) {
        ASSERT_FALSE(ready.empty());
        EXPECT_EQ(sorted[index].sequence, expected[index]);
        EXPECT_EQ(ready.top().sequence, expected[index]);
        ready.pop();
    }
    EXPECT_TRUE(ready.empty());
}

} // namespace
} // namespace metallic::tests

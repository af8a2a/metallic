#include "RHITest.h"
#include "harness/Evidence.h"
#include "Runtime/Render/Shadow/VirtualShadowMapBudget.h"

#include <array>
#include <numeric>
#include <stdexcept>

namespace metallic::tests {
namespace {
using namespace render;

void check(bool condition, const char* message)
{
    if (!condition) { throw std::runtime_error(message); }
}

// Independent policy reference: grant one page per unsatisfied face per round.
std::vector<uint32_t> serialBudgets(std::span<const uint32_t> targets, uint32_t budget)
{
    std::vector<uint32_t> result(targets.size(), 1u);
    uint32_t remaining = budget - uint32_t(targets.size());
    while (remaining != 0) {
        bool granted = false;
        for (size_t index = 0; index < targets.size() && remaining != 0; ++index) {
            if (result[index] - 1u < targets[index]) {
                ++result[index]; --remaining; granted = true;
            }
        }
        if (!granted) { break; }
    }
    return result;
}

void checkBudgets(std::span<const uint32_t> targets, uint32_t budget,
    std::span<const uint32_t> result)
{
    check(result.size() == targets.size(), "Budget count changed face identity");
    uint64_t desired = targets.size(), allocated = 0;
    for (size_t index = 0; index < targets.size(); ++index) {
        desired += targets[index]; allocated += result[index];
        check(result[index] >= 1 && result[index] <= targets[index] + 1u,
            "Coarse minimum or fine demand cap was violated");
        if (result[index] - 1u == targets[index]) { continue; }
        for (size_t other = 0; other < targets.size(); ++other) {
            check(result[other] <= result[index] + 1u, "Unsatisfied face violates max-min fairness");
            if (index < other) {
                check(result[other] <= result[index], "Fine-page remainder did not follow stable input order");
            }
        }
    }
    check(allocated == std::min(uint64_t(budget), desired), "Budget was lost or assigned beyond demand");
}

class VirtualShadowMapBudgetPolicyTest final : public RHITest {
public:
    VirtualShadowMapBudgetPolicyTest() { type = RHITestType::Resource; name = "virtual_shadow_map_budget_policy"; }
    std::optional<bench::Metadata> metadata() const override
    {
        return bench::Metadata{.suite = "contract", .layer = bench::Layer::Core,
            .requirements = {.requiresDevice = false, .validation = bench::Validation::Off, .queues = {}},
            .coverage = {"vsm.budget.waterfill", "vsm.budget.demand-history", "vsm.budget.retain-without-pressure"},
            .artifacts = {"VirtualShadowMapBudgetPolicy.json"}};
    }
    RHITestResult run(RHITestContext&) override { return execute(nullptr); }
    RHITestResult runCpu(bench::Evidence& evidence) override { return execute(&evidence); }

    RHITestResult execute(bench::Evidence* evidence)
    {
        bench::Json report{{"gpuExecuted", false}, {"passed", false}};
        uint32_t enumerated = 0, boundaryCases = 0, retentionCases = 0;
        try {
            check(allocateVirtualShadowMapPageBudgets({}, 0).empty(), "Empty zero-budget input failed");
            check(allocateVirtualShadowMapPageBudgets({}, UINT32_MAX).empty(), "Empty unused budget failed");
            // Enumerate every target vector of length 1..5 with values 0..4,
            // including all budgets through complete demand plus unused pages.
            for (uint32_t count = 1, combinations = 5; count <= 5; ++count, combinations *= 5) {
                for (uint32_t encoded = 0; encoded < combinations; ++encoded) {
                    std::vector<uint32_t> targets(count);
                    uint32_t digits = encoded;
                    for (auto& target : targets) { target = digits % 5; digits /= 5; }
                    const uint32_t desired = count + std::accumulate(targets.begin(), targets.end(), 0u);
                    std::vector<uint32_t> previous(count, 1u);
                    for (uint32_t budget = count; budget <= desired + 3; ++budget) {
                        const auto result = allocateVirtualShadowMapPageBudgets(targets, budget);
                        checkBudgets(targets, budget, result);
                        check(result == serialBudgets(targets, budget), "Water filling differs from serial policy");
                        for (uint32_t index = 0; index < count; ++index) {
                            check(result[index] >= previous[index], "Increasing global budget reduced a face grant");
                        }
                        previous = result;
                        ++enumerated;
                    }
                }
            }
            const std::array<std::vector<uint32_t>, 6> boundaries{{
                {0}, {4095}, {0, 4095}, {8, 4095, 8}, {4095, 0, 4095, 1, 0, 4094},
                std::vector<uint32_t>(384, 4095)}};
            for (const auto& targets : boundaries) {
                const uint32_t count = uint32_t(targets.size());
                const uint32_t desired = count + std::accumulate(targets.begin(), targets.end(), 0u);
                const std::array budgets{count, count + 1u, count + 7u, count + 8u,
                    count + 4095u, desired, desired + 1u, UINT32_MAX};
                for (uint32_t budget : budgets) {
                    const auto result = allocateVirtualShadowMapPageBudgets(targets, budget);
                    checkBudgets(targets, budget, result);
                    check(result == serialBudgets(targets, budget), "Boundary budget differs from serial policy");
                    ++boundaryCases;
                }
            }
            // Unknown faces must reproduce the old coarse-first equal split,
            // including stable remainders and the existing 4096-page cap.
            for (uint32_t count : {1u, 2u, 6u, 8u, 64u, 384u}) {
                const std::vector<uint32_t> targets(count, VirtualShadowMapDemandHistory{}.targetFine);
                for (uint32_t budget : {count, count + 1u, count + 17u, 4096u, 262144u, UINT32_MAX}) {
                    const auto result = allocateVirtualShadowMapPageBudgets(targets, budget);
                    const uint32_t extra = (budget - count) / count, remainder = (budget - count) % count;
                    for (uint32_t index = 0; index < count; ++index) {
                        check(result[index] == std::min(4096u, 1u + extra + (index < remainder ? 1u : 0u)),
                            "Unknown demand no longer matches equal-budget startup");
                    }
                    ++boundaryCases;
                }
            }

            const std::array<std::array<uint32_t, 2>, 10> targetCases{{
                {0, 0}, {1, 8}, {7, 8}, {8, 16}, {16, 24}, {64, 72},
                {128, 144}, {3640, 4095}, {4095, 4095}, {UINT32_MAX, 4095}}};
            for (const auto& item : targetCases) {
                VirtualShadowMapDemandHistory history;
                check(!history.known && history.targetFine == 4095, "Unknown demand has no equal-split prior");
                history.observe(item[0]);
                check(history.known && history.observedFine == item[0] && history.targetFine == item[1] &&
                    history.lowerSamples == 0, "First demand sample, headroom, quantization or cap is incorrect");
            }
            VirtualShadowMapDemandHistory history;
            history.observe(128);
            for (uint32_t sample = 0; sample < 64; ++sample) {
                history.observe(sample % 2 ? 121 : 127);
                check(history.targetFine == 144 && history.lowerSamples == 0, "Small demand jitter resized the face");
            }
            for (uint32_t sample = 1; sample <= 7; ++sample) {
                history.observe(32);
                check(history.targetFine == 144 && history.lowerSamples == sample, "Demand shrank before eight samples");
            }
            history.observe(80);
            check(history.targetFine == 144 && history.lowerSamples == 0 && history.lowerTargetFine == 0,
                "Demand recovery did not interrupt shrink window");
            for (uint32_t sample = 0; sample < 8; ++sample) { history.observe(32); }
            check(history.targetFine == 40 && history.lowerSamples == 0, "Sustained low demand did not shrink");
            history.observe(200);
            check(history.targetFine == 232 && history.lowerSamples == 0, "Demand growth did not expand immediately");
            for (uint32_t sample = 1; sample <= 7; ++sample) {
                history.observe(0);
                check(history.targetFine == 232 && history.lowerSamples == sample, "Transient zero demand discarded capacity");
            }
            history.observe(0);
            check(history.targetFine == 0 && history.lowerSamples == 0, "Stable zero demand retained fine budget");
            history.observe(1);
            check(history.targetFine == 8 && history.lowerSamples == 0, "Dormant face did not grow immediately");
            history.observe(128);
            history.observe(73);
            check(history.targetFine == 144 && history.lowerSamples == 0, "Above-half demand entered the shrink window");
            for (uint32_t sample = 0; sample < 8; ++sample) { history.observe(72); }
            check(history.targetFine == 88 && history.lowerSamples == 0, "Exactly-half demand did not enter the shrink window");
            history.observe(128);
            for (uint32_t sample = 0; sample < 8; ++sample) {
                history.observe(sample % 2 == 0 ? 64 : 8);
                check(history.targetFine == (sample == 7 ? 72u : 144u),
                    "Alternating low demand discarded the shrink window maximum");
            }
            check(history.lowerSamples == 0 && history.lowerTargetFine == 0, "Completed shrink retained stale window state");
            history.observe(64);
            check(history.targetFine == 72, "Low-demand oscillation regrew immediately after shrink");
            struct RetentionCase
            {
                std::vector<VirtualShadowMapBudgetRetentionFace> faces;
                uint32_t budget;
                bool retain;
            };
            const std::array retention{
                RetentionCase{{}, 0, true},
                RetentionCase{{}, UINT32_MAX, true},
                RetentionCase{{{0, 0, false}}, 0, false},
                RetentionCase{{{0, 0, true}}, 1, false},
                RetentionCase{{{0, 1, false}}, 1, false},
                RetentionCase{{{0, 1, true}}, 0, false},
                RetentionCase{{{0, 1, true}}, 1, true},
                RetentionCase{{{0, 8, true}}, 8, true},
                RetentionCase{{{7, 8, true}, {2, 3, true}}, 11, true},
                RetentionCase{{{7, 8, true}, {2, 3, true}}, 10, false},
                RetentionCase{{{8, 8, true}, {2, 3, true}}, 11, false},
                RetentionCase{{{7, 8, true}, {2, 3, false}}, 11, false},
                RetentionCase{{{3, 8, true}, {0, 16, true}}, 64, true},
                RetentionCase{{{3, 8, true}, {0, 16, true}}, 23, false},
                RetentionCase{{{UINT32_MAX - 1u, UINT32_MAX, true}}, UINT32_MAX, true},
                RetentionCase{{{0, UINT32_MAX, true}, {0, 1, true}}, UINT32_MAX, false},
                RetentionCase{{{0, UINT32_MAX, true}, {0, UINT32_MAX, true}}, UINT32_MAX, false},
                RetentionCase{{{UINT32_MAX, UINT32_MAX, true}}, UINT32_MAX, false}};
            for (const auto& item : retention) {
                check(canRetainVirtualShadowMapPageBudgets(item.faces, item.budget) == item.retain,
                    "Capacity retention ignored pressure, unknown faces, coarse coverage or the global limit");
                ++retentionCases;
            }
            report.update({{"enumeratedAllocationCases", enumerated}, {"boundaryAllocationCases", boundaryCases},
                {"targetCases", targetCases.size()}, {"retentionCases", retentionCases}, {"passed", true}});
            if (evidence) { evidence->json("VirtualShadowMapBudgetPolicy.json", report); }
            return RHITestResult::pass("CPU budget max-min fairness, stable remainders, demand hysteresis and pressure-driven retention verified");
        } catch (const std::exception& error) {
            report.update({{"enumeratedAllocationCases", enumerated}, {"boundaryAllocationCases", boundaryCases}, {"retentionCases", retentionCases},
                {"error", error.what()}});
            if (evidence) { evidence->json("VirtualShadowMapBudgetPolicy.json", report); }
            return RHITestResult::fail(error.what());
        }
    }
};

METALLIC_REGISTER_RHI_TEST(VirtualShadowMapBudgetPolicyTest);
} // namespace
} // namespace metallic::tests

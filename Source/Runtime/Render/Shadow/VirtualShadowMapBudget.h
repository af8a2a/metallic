#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>

namespace metallic::render {

inline constexpr uint32_t kVirtualShadowMapMaxFinePageBudget = 4095;

struct VirtualShadowMapDemandHistory
{
    // Keep the untruncated GPU request for diagnostics. Unknown faces request
    // the maximum, which reproduces equal division until feedback arrives.
    uint32_t observedFine = 0;
    uint32_t targetFine = kVirtualShadowMapMaxFinePageBudget;
    uint32_t lowerSamples = 0;
    uint32_t lowerTargetFine = 0;
    bool known = false;

    void observe(uint32_t requestedFine)
    {
        observedFine = requestedFine;
        const uint32_t bounded = std::min(requestedFine, kVirtualShadowMapMaxFinePageBudget);
        // One eighth headroom, rounded upward to eight pages. Zero stays zero;
        // the caller separately reserves complete coarse coverage for each face.
        const uint32_t padded = bounded + (bounded + 7u) / 8u;
        const uint32_t desired = std::min((padded + 7u) / 8u * 8u, kVirtualShadowMapMaxFinePageBudget);
        if (!known || desired > targetFine) {
            targetFine = desired;
            lowerSamples = 0;
            lowerTargetFine = 0;
            known = true;
        } else if (requestedFine <= targetFine / 2u && desired < targetFine) {
            // Changing a face capacity invalidates its atlas cache. Demand must
            // remain substantially lower for eight completed observations before
            // shrinking; retain the largest target in that window so alternating
            // low samples do not immediately regrow after its final low sample.
            lowerTargetFine = std::max(lowerTargetFine, desired);
            if (++lowerSamples == 8u) {
                targetFine = lowerTargetFine;
                lowerSamples = 0;
                lowerTargetFine = 0;
            }
        } else {
            lowerSamples = 0;
            lowerTargetFine = 0;
        }
    }
};

struct VirtualShadowMapBudgetRetentionFace
{
    uint32_t observedFine = 0;
    uint32_t previousPages = 0;
    bool known = false;
};

// Reclaim idle capacity only when another face needs it or the global limit
// shrinks. If every actual request already fits, preserving the existing atlas
// layout avoids cache invalidation and capacity-dependent raster roundoff.
// Padding in targetFine is growth headroom, not an unmet receiver request.
inline bool canRetainVirtualShadowMapPageBudgets(
    std::span<const VirtualShadowMapBudgetRetentionFace> faces, uint32_t totalBudget)
{
    uint64_t assigned = 0;
    for (const auto& face : faces) {
        if (!face.known || face.previousPages == 0 || face.observedFine > face.previousPages - 1u) { return false; }
        assigned += face.previousPages;
    }
    return assigned <= totalBudget;
}

// The manager has admitted complete lights and reserved one coarse page per
// face. Inputs are in stable light-ID/face order, with targets in [0,4095].
// Raise a common fine-page water level, then award its remainder in that order.
// Unneeded budget remains free rather than being assigned beyond known demand.
inline std::vector<uint32_t> allocateVirtualShadowMapPageBudgets(
    std::span<const uint32_t> fineTargets, uint32_t totalBudget)
{
    assert(fineTargets.size() <= totalBudget);
    assert(std::all_of(fineTargets.begin(), fineTargets.end(),
        [](uint32_t target) { return target <= kVirtualShadowMapMaxFinePageBudget; }));
    std::vector<uint32_t> budgets(fineTargets.size(), 1u);
    if (fineTargets.empty()) { return budgets; }

    const uint64_t available = totalBudget - fineTargets.size();
    uint32_t lower = 0, upper = kVirtualShadowMapMaxFinePageBudget;
    while (lower < upper) {
        const uint32_t level = lower + (upper - lower + 1u) / 2u;
        uint64_t needed = 0;
        for (uint32_t target : fineTargets) { needed += std::min(target, level); }
        if (needed <= available) { lower = level; }
        else { upper = level - 1u; }
    }
    uint64_t remainder = available;
    for (size_t index = 0; index < fineTargets.size(); ++index) {
        const uint32_t fine = std::min(fineTargets[index], lower);
        budgets[index] += fine;
        remainder -= fine;
    }
    for (size_t index = 0; index < fineTargets.size() && remainder != 0; ++index) {
        if (fineTargets[index] > lower) { ++budgets[index]; --remainder; }
    }
    return budgets;
}

} // namespace metallic::render

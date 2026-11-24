#pragma once

#include "Runtime/Render/Shadow/VirtualShadowMaps.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace metallic::render {

class VirtualShadowMapRasterBudget
{
public:
    void update(float budgetMs, float maxBias, std::optional<VirtualShadowMapRasterSample> sample = {})
    {
        if (budgetMs == 0 || maxBias == 0) {
            *this = {};
            return;
        }
        if (budgetMs_ != budgetMs) { belowTargetSamples_ = 0; }
        budgetMs_ = budgetMs;
        if (sample) {
            const float targetMs = budgetMs * 0.95f;
            const float cost = sample->milliseconds;
            const float target = cost > 0 ?
                std::clamp(sample->appliedBias + std::log2(cost / targetMs), 0.f, maxBias) : 0.f;
            if (cost > budgetMs) {
                belowTargetSamples_ = 0;
                bias_ = std::lerp(bias_, std::max(bias_, target), 0.5f);
            } else if (cost <= targetMs) {
                belowTargetSamples_ = std::min(belowTargetSamples_ + 1, 11u);
                if (belowTargetSamples_ > 10) { bias_ = std::lerp(bias_, std::min(bias_, target), 0.1f); }
            } else {
                belowTargetSamples_ = 0;
            }
        }
        bias_ = std::min(bias_, maxBias);
    }

    float bias() const { return bias_; }
    uint32_t belowTargetSamples() const { return belowTargetSamples_; }

private:
    float budgetMs_ = 0;
    float bias_ = 0;
    uint32_t belowTargetSamples_ = 0;
};

// Each controller follows one physical pool across its submitted frames.
class VirtualShadowMapResolutionController
{
public:
    static constexpr float kDefaultLoadFactor = 0.85f;
    static constexpr float kDefaultMaxBias = 2.0f;
    using Diagnostics = std::array<uint32_t, 8>;

    Result<Diagnostics> update(float loadFactor, float maxBias, bool hasActiveFaces,
        float rasterTimeBudgetMs = 0, bool timestampSupported = false)
    {
        if (!std::isfinite(loadFactor) || loadFactor < 0 || loadFactor > 1 ||
            !std::isfinite(maxBias) || maxBias < 0 || maxBias > 12 ||
            !std::isfinite(rasterTimeBudgetMs) || rasterTimeBudgetMs < 0) {
            return makeError(Error::InvalidArgument);
        }
        if (!hasActiveFaces || maxBias == 0) {
            *this = {};
            rasterDiagnostics_[7] = std::bit_cast<uint32_t>(rasterTimeBudgetMs);
            return Diagnostics{1};
        }

        rasterEnabled_ = timestampSupported && rasterTimeBudgetMs > 0;
        std::optional<VirtualShadowMapRasterSample> rasterSample;
        if (rasterEnabled_) {
            auto completed = rasterFeedback_.poll();
            if (!completed) { return makeError(completed.error()); }
            rasterSample = *completed;
            if (rasterSample) { lastRaster_ = *rasterSample; }
        } else {
            rasterFeedback_ = {};
            lastRaster_ = {};
        }
        rasterBudget_.update(rasterEnabled_ ? rasterTimeBudgetMs : 0.f, maxBias, rasterSample);
        rasterDiagnostics_ = {1, uint32_t(rasterEnabled_), uint32_t(rasterSample.has_value()),
            std::bit_cast<uint32_t>(lastRaster_.milliseconds), std::bit_cast<uint32_t>(lastRaster_.appliedBias),
            std::bit_cast<uint32_t>(rasterBudget_.bias()), rasterBudget_.belowTargetSamples(),
            std::bit_cast<uint32_t>(rasterTimeBudgetMs)};

        pageEnabled_ = loadFactor > 0;
        bool consumedPage = false;
        if (pageEnabled_) {
            auto completed = feedback_.poll();
            if (!completed) { return makeError(completed.error()); }
            consumedPage = completed->has_value();
            if (consumedPage) {
                last_ = **completed;
                const float load = float(last_.requestedPages) / float(last_.capacity);
                float target = load > 0 ? std::max(0.0f, last_.appliedBias + std::log2(load / loadFactor)) : 0.0f;
                if (load > loadFactor) {
                    belowTargetSamples_ = 0;
                    if (rasterEnabled_) { target = std::max(pageBias_, target); }
                    pageBias_ = std::lerp(pageBias_, target, 0.5f);
                } else if (last_.motionBiased) {
                    // Temporary per-light coarsening hides demand that can return when the light stops.
                    belowTargetSamples_ = 0;
                } else {
                    belowTargetSamples_ = std::min(belowTargetSamples_ + 1, 11u);
                    if (rasterEnabled_) { target = std::min(pageBias_, target); }
                    if (belowTargetSamples_ > 10) { pageBias_ = std::lerp(pageBias_, target, 0.1f); }
                }
            }
        } else {
            feedback_ = {};
            last_ = {};
            pageBias_ = 0;
            belowTargetSamples_ = 0;
        }
        pageBias_ = std::clamp(pageBias_, 0.0f, maxBias);
        return Diagnostics{1, uint32_t(consumedPage), last_.requestedPages, last_.capacity,
            std::bit_cast<uint32_t>(last_.appliedBias), std::bit_cast<uint32_t>(bias()), belowTargetSamples_, 0};
    }

    float bias() const { return std::max(pageBias_, rasterBudget_.bias()); }
    const Diagnostics& rasterDiagnostics() const { return rasterDiagnostics_; }

    Result<> record(Device& device, CommandBuffer& commands, std::span<const VirtualShadowMapResult> faces,
        uint32_t capacity, const VirtualShadowMapPhysicalPool* pool = nullptr, bool motionBiased = false)
    {
        if (!pageEnabled_) { return {}; }
        return feedback_.record(device, commands, faces, capacity, bias(), pool, motionBiased);
    }

    Result<VirtualShadowMapRasterFeedback*> prepareRaster(Device& device, CommandBuffer& commands, uint32_t rangeCount)
    {
        if (!rasterEnabled_) { return nullptr; }
        if (auto r = rasterFeedback_.prepare(device, commands, rangeCount, bias()); !r) { return std::unexpected(r.error()); }
        return &rasterFeedback_;
    }

private:
    VirtualShadowMapPressureFeedback feedback_;
    VirtualShadowMapPressureSample last_;
    VirtualShadowMapRasterFeedback rasterFeedback_;
    VirtualShadowMapRasterBudget rasterBudget_;
    VirtualShadowMapRasterSample lastRaster_;
    Diagnostics rasterDiagnostics_{1};
    float pageBias_ = 0;
    uint32_t belowTargetSamples_ = 0;
    bool pageEnabled_ = false;
    bool rasterEnabled_ = false;
};

} // namespace metallic::render

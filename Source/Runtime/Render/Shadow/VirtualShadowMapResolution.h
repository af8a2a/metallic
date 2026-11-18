#pragma once

#include "Runtime/Render/Shadow/VirtualShadowMaps.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>

namespace metallic::render {

// Each controller follows one physical pool across its submitted frames.
class VirtualShadowMapResolutionController
{
public:
    static constexpr float kDefaultLoadFactor = 0.85f;
    static constexpr float kDefaultMaxBias = 2.0f;
    using Diagnostics = std::array<uint32_t, 8>;

    Result<Diagnostics> update(float loadFactor, float maxBias, bool hasActiveFaces)
    {
        if (!std::isfinite(loadFactor) || loadFactor < 0 || loadFactor > 1 ||
            !std::isfinite(maxBias) || maxBias < 0 || maxBias > 12) {
            return makeError(Error::InvalidArgument);
        }
        if (!hasActiveFaces || loadFactor == 0 || maxBias == 0) {
            *this = {};
            return Diagnostics{1};
        }
        enabled_ = true;
        auto completed = feedback_.poll();
        if (!completed) { return makeError(completed.error()); }
        if (*completed) {
            last_ = **completed;
            const float load = float(last_.requestedPages) / float(last_.capacity);
            const float target = load > 0 ? std::max(0.0f, last_.appliedBias + std::log2(load / loadFactor)) : 0.0f;
            if (load > loadFactor) {
                belowTargetSamples_ = 0;
                bias_ = std::lerp(bias_, target, 0.5f);
            } else {
                belowTargetSamples_ = std::min(belowTargetSamples_ + 1, 11u);
                if (belowTargetSamples_ > 10) { bias_ = std::lerp(bias_, target, 0.1f); }
            }
        }
        bias_ = std::clamp(bias_, 0.0f, maxBias);
        return Diagnostics{1, uint32_t(completed->has_value()), last_.requestedPages, last_.capacity,
            std::bit_cast<uint32_t>(last_.appliedBias), std::bit_cast<uint32_t>(bias_), belowTargetSamples_, 0};
    }

    float bias() const { return bias_; }

    Result<> record(Device& device, CommandBuffer& commands, std::span<const VirtualShadowMapResult> faces,
        uint32_t capacity, const VirtualShadowMapPhysicalPool* pool = nullptr)
    {
        if (!enabled_) { return {}; }
        return feedback_.record(device, commands, faces, capacity, bias_, pool);
    }

private:
    VirtualShadowMapPressureFeedback feedback_;
    VirtualShadowMapPressureSample last_;
    float bias_ = 0;
    uint32_t belowTargetSamples_ = 0;
    bool enabled_ = false;
};

} // namespace metallic::render

#pragma once

#include <algorithm>
#include <cstdint>

namespace metallic::render {

// A finer replacement tail already contains every mip of the previous tail.
// Restrict sampling to that visible source LOD first, then reveal the new mips.
// The caller supplies monotonic active time so publication freezes can pause it.
class TextureMipTransition {
public:
    static constexpr double kDurationSeconds = 0.15;

    void reset(uint32_t firstMip)
    {
        firstMip_ = firstMip;
        initialFloor_ = 0.0f;
        startedSeconds_ = 0.0;
    }

    float samplingLodFloor(double activeSeconds) const
    {
        const double progress = std::clamp((activeSeconds - startedSeconds_) / kDurationSeconds, 0.0, 1.0);
        return initialFloor_ * static_cast<float>(1.0 - progress);
    }

    void publish(uint32_t firstMip, double activeSeconds)
    {
        const float visibleSourceLod = static_cast<float>(firstMip_) + samplingLodFloor(activeSeconds);
        initialFloor_ = std::max(visibleSourceLod - static_cast<float>(firstMip), 0.0f);
        firstMip_ = firstMip;
        startedSeconds_ = activeSeconds;
    }

private:
    uint32_t firstMip_ = 0;
    float initialFloor_ = 0.0f;
    double startedSeconds_ = 0.0;
};

} // namespace metallic::render

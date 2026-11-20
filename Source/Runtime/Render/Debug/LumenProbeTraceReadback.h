#pragma once

#include <array>
#include <cstdint>

namespace metallic::render {

inline constexpr char kLumenProbeTraceReadbackLayout[] = "LumenProbeTraceReadback";

// Matches the records written by LumenScreenProbeGather.slang.
struct LumenProbeTraceReadbackRecord {
    float screenEnd, detailEnd, gdfMin, gdfMax;
    std::array<float, 3> wrcRadiance;
    float gdfHitDistance;
    uint32_t gdfClassification, gdfHitClipmap, gdfSteps, flags;
};
static_assert(sizeof(LumenProbeTraceReadbackRecord) == 48);

} // namespace metallic::render

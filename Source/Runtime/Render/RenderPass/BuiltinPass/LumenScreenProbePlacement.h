#pragma once

#include "Runtime/Render/Core/ResourceRegistry.h"

namespace metallic::render::builtin_pass {

inline constexpr uint64_t kScreenProbeFilterABI = 0x4c554d454e010014ull;
inline constexpr uint32_t kLumenScreenProbeSlotsPerTile = 9;
inline constexpr uint32_t kLumenScreenProbeDownsampleFactor = 16;

inline constexpr uint32_t lumenScreenProbeTiles(uint32_t pixels)
{
    return (pixels + kLumenScreenProbeDownsampleFactor - 1) / kLumenScreenProbeDownsampleFactor;
}

// Shared by placement in the cache and the legacy gather permutation, and by screen probe filtering.
struct LumenScreenProbeFilterParams {
    ShaderSampledImage position, normal;
    ShaderBuffer probeOutput, probeInput, inputRadiance, inputDistance, outputRadiance;
    uint32_t width, height, probeWidth, probeHeight, frameIndex;
    ShaderSampledImage blueNoiseVec2;
    ShaderBuffer outputDistance;
    ShaderStorageImage irradianceOutput, shDirectionalOutput;
    uint32_t rayDirectionFrameIndex;
};
static_assert(sizeof(LumenScreenProbeFilterParams) == 68 && offsetof(LumenScreenProbeFilterParams, irradianceOutput) == 56);

} // namespace metallic::render::builtin_pass

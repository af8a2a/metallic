#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using FootprintProbeOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias FootprintProbeOutput = DataSpan<float2>;
#endif
struct TextureFootprintProbeInput
{
    uint32_t width;
    uint32_t height;
    uint32_t index;
    uint32_t orthographic;
    float aspect;
    float verticalFovRadians;
    float orthographicHeight;
    float reserved;
};
struct TextureFootprintProbeParameters
{
    FootprintProbeOutput output;
    TextureFootprintProbeInput input;
};
#ifdef __cplusplus
inline constexpr uint64_t kFootprintProbeABI = 0x465450524f420001ull;
static_assert(sizeof(TextureFootprintProbeInput) == 32);
static_assert(sizeof(TextureFootprintProbeParameters) == 48);
} // namespace metallic::tests
#endif

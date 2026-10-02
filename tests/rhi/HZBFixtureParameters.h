#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using HZBFixtureDepth = render::ShaderStorageImage;
using HZBFixtureCounter = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias HZBFixtureDepth = DescriptorHandle<RWTexture2D<float>>;
typealias HZBFixtureCounter = DataSpan<uint>;
#endif
struct HZBFixtureParameters
{
    HZBFixtureDepth depth;
    HZBFixtureCounter counter;
    uint32_t width, height, seed, reversedZ;
};
#ifdef __cplusplus
inline constexpr uint64_t kHZBFixtureABI = 0x485a424649580001ull;
static_assert(sizeof(HZBFixtureParameters) == 40);
} // namespace metallic::tests
#endif

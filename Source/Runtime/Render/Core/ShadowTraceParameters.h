#pragma once
#include "PathTraceParameters.h"
#include "StreamSceneParameters.h"
#ifdef __cplusplus
namespace metallic::render {
using ShadowTraceSettings = ShaderDataSpan;
using ShadowTraceScene = uint64_t;
using ShadowTraceDepth = ShaderSampledImage;
using ShadowTraceScalar = ShaderStorageImage;
using ShadowTraceVector = ShaderStorageImage;
#else
namespace Metallic {
typealias ShadowTraceSettings = DataSpan<ShadowParameters>;
typealias ShadowTraceScene = PathTraceParameters*;
typealias ShadowTraceDepth = DescriptorHandle<Texture2D<float>>;
typealias ShadowTraceScalar = DescriptorHandle<RWTexture2D<float>>;
typealias ShadowTraceVector = DescriptorHandle<RWTexture2D<float4>>;
#endif
// Fixed-size inline root; geometry and settings are immutable typed BDA data.
struct ShadowTraceParameters
{
    ShadowTraceSettings settings;
    ShadowTraceScene scene;
    StreamSceneAddress streamScene;
    ShadowTraceDepth depth;
    ShadowTraceScalar penumbra;
    ShadowTraceVector normal;
    ShadowTraceScalar viewZ;
    ShadowTraceVector motion;
    ShadowTraceScalar shadow;
    uint32_t materialTextureCount;
    uint32_t ntcTextureSetCount;
};
#ifdef __cplusplus
inline constexpr uint64_t kShadowTraceABI = 0x5348445452430001ull;
static_assert(sizeof(ShadowTraceParameters) == 88);
static_assert(offsetof(ShadowTraceParameters, scene) == 16);
static_assert(offsetof(ShadowTraceParameters, depth) == 32);
static_assert(offsetof(ShadowTraceParameters, materialTextureCount) == 80);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)

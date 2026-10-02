#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::render {
using ImageSampleSource = ShaderSampledImage;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias ImageSampleSource = DescriptorHandle<Texture2D<float4>>;
#endif
struct ImageSampleParams
{
    ImageSampleSource source;
};
#ifdef __cplusplus
inline constexpr uint64_t kImageSampleABI = 0x494d4753414d0001ull;
static_assert(sizeof(ImageSampleParams) == 8);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)

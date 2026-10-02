#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using UploadProbeImage = render::ShaderSampledImage;
using UploadProbeOutput = render::ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
typealias UploadProbeImage = DescriptorHandle<Texture2D<float4>>;
typealias UploadProbeOutput = DataSpan<uint4>;
#endif
struct SceneUploadProbeParameters
{
    UploadProbeImage image;
    UploadProbeOutput output;
    uint32_t outputOffset;
    uint32_t mipCount;
};
#ifdef __cplusplus
inline constexpr uint64_t kSceneUploadProbeABI = 0x55504c5052420001ull;
static_assert(sizeof(SceneUploadProbeParameters) == 32);
} // namespace metallic::tests
#endif

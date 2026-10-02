#pragma once
#include "PathTraceParameters.h"
#ifdef __cplusplus
namespace metallic::render {
using RTXDITraceData = uint64_t;
using RTXDITraceScene = uint64_t;
using RTXDITraceImage = ShaderStorageImage;
using RTXDITraceReservoir = ShaderStorageImage;
using RTXDITraceScalar = ShaderStorageImage;
#else
namespace Metallic {
typealias RTXDITraceData = uint*;
typealias RTXDITraceScene = PathTraceParameters*;
typealias RTXDITraceImage = DescriptorHandle<RWTexture2D<float4>>;
typealias RTXDITraceReservoir = DescriptorHandle<RWTexture2D<uint4>>;
typealias RTXDITraceScalar = DescriptorHandle<RWTexture2D<float>>;
#endif
struct RTXDITraceParameters
{
    RTXDITraceData settings;
    RTXDITraceScene scene;
    RTXDITraceImage output;
    RTXDITraceReservoir reservoirCurrent;
    RTXDITraceReservoir reservoirPrevious;
    RTXDITraceImage positionCurrent;
    RTXDITraceImage positionPrevious;
    RTXDITraceImage normalCurrent;
    RTXDITraceImage normalPrevious;
    RTXDITraceImage noisyDiffuse;
    RTXDITraceImage noisySpecular;
    RTXDITraceImage normalRoughness;
    RTXDITraceImage motionVectors;
    RTXDITraceScalar viewZ;
    RTXDITraceImage baseColorMetalness;
    RTXDITraceImage emissive;
};
#ifdef __cplusplus
inline constexpr uint64_t kRTXDITraceABI = 0x5254584449540001ull;
static_assert(sizeof(RTXDITraceParameters) == 128);
static_assert(offsetof(RTXDITraceParameters, output) == 16);
static_assert(offsetof(RTXDITraceParameters, emissive) == 120);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)

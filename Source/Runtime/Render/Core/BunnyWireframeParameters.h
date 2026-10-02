#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using BunnyFloat4 = float[4];
using BunnyPositions = ShaderDataSpan;
using BunnyTransforms = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
struct BunnyTransform
{
    float4 world0, world1, world2, world3;
};
typealias BunnyFloat4 = float4;
typealias BunnyPositions = DataSpan<float4>;
typealias BunnyTransforms = DataSpan<BunnyTransform>;
#endif
struct BunnyWireframeGPUParams
{
    BunnyFloat4 eye;
    BunnyFloat4 center;
    BunnyFloat4 upProjection;
    BunnyFloat4 viewport;
    BunnyFloat4 clipOrtho;
    BunnyFloat4 clearColor;
    BunnyFloat4 wireColor;
    BunnyFloat4 settings;
};
struct BunnyWireframeParameters
{
    BunnyPositions positions;
    BunnyTransforms transforms;
    BunnyWireframeGPUParams settings;
};
#ifdef __cplusplus
inline constexpr uint64_t kBunnyWireframeABI = 0x42554e4e59570001ull;
static_assert(sizeof(BunnyWireframeGPUParams) == 128);
static_assert(sizeof(BunnyWireframeParameters) == 160);
static_assert(offsetof(BunnyWireframeParameters, settings) == 32);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)

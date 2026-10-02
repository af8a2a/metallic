#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using MaterialRasterFloat4 = float[4];
using MaterialRasterVectors = ShaderDataSpan;
using MaterialRasterIndices = ShaderDataSpan;
using MaterialRasterTransforms = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
struct MaterialRasterTransform { float4 world0, world1, world2, world3; };
typealias MaterialRasterFloat4 = float4;
typealias MaterialRasterVectors = DataSpan<float4>;
typealias MaterialRasterIndices = DataSpan<uint>;
typealias MaterialRasterTransforms = DataSpan<MaterialRasterTransform>;
#endif
struct MaterialShaderObjectGPUParams
{
    MaterialRasterFloat4 eye;
    MaterialRasterFloat4 center;
    MaterialRasterFloat4 upProjection;
    MaterialRasterFloat4 viewport;
    MaterialRasterFloat4 clipOrtho;
};
struct MaterialRasterParameters
{
    MaterialRasterVectors positions;
    MaterialRasterIndices materialIndices;
    MaterialRasterVectors materials;
    MaterialRasterTransforms transforms;
    MaterialShaderObjectGPUParams camera;
    uint32_t vertexOffset;
    uint32_t padding0;
    uint32_t padding1;
    uint32_t padding2;
};
#ifdef __cplusplus
inline constexpr uint64_t kMaterialRasterABI = 0x4d41545241530001ull;
static_assert(sizeof(MaterialShaderObjectGPUParams) == 80);
static_assert(sizeof(MaterialRasterParameters) == 160);
static_assert(offsetof(MaterialRasterParameters, camera) == 64);
static_assert(offsetof(MaterialRasterParameters, vertexOffset) == 144);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)

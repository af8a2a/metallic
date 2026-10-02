#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using StreamSceneWords = ShaderDataSpan;
using StreamScenePages = ShaderDataSpan;
using StreamSceneInstances = ShaderDataSpan;
#else
import ShaderCore;
using Metallic;
struct StreamRayInstance
{
    uint primitiveIndex, materialIndex, visible, gpuSceneInstanceIndex;
    float4 world0, world1, world2, world3;
    float4 bounds;
};
namespace Metallic {
typealias StreamSceneWords = DataSpan<uint>;
typealias StreamScenePages = DataSpan<uint2>;
typealias StreamSceneInstances = DataSpan<StreamRayInstance>;
#endif
struct StreamSceneParameters
{
    StreamSceneWords pages;
    StreamScenePages pageTable;
    StreamSceneInstances instances;
    StreamSceneWords header;
};
#ifdef __cplusplus
using StreamSceneAddress = uint64_t;
#else
typealias StreamSceneAddress = StreamSceneParameters*;
#endif
#ifdef __cplusplus
static_assert(sizeof(StreamSceneParameters) == 64);
static_assert(offsetof(StreamSceneParameters, pageTable) == 16);
static_assert(offsetof(StreamSceneParameters, instances) == 32);
static_assert(offsetof(StreamSceneParameters, header) == 48);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)

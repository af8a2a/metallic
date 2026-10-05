#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::render {
#define STRAND_FLOAT4(name)                                                                                            \
    float name[4]                                                                                                      \
    {                                                                                                                  \
    }
#define STRAND_UINT4(name)                                                                                             \
    uint32_t name[4]                                                                                                   \
    {                                                                                                                  \
    }
using StrandUInt = uint32_t;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
#define STRAND_FLOAT4(name) float4 name
#define STRAND_UINT4(name) uint4 name
typealias StrandUInt = uint;
#endif
struct StrandSegment
{
    STRAND_FLOAT4(a);
    STRAND_FLOAT4(b); // xyz position, radius
    STRAND_FLOAT4(previousA);
    STRAND_FLOAT4(previousB);
    STRAND_FLOAT4(normalU); // stable normal, root-to-tip parameter at a
    STRAND_FLOAT4(shape);   // parameter at b, opacity, stable LOD rank, unused
    STRAND_UINT4(identity); // strand, original segment, material, unused
};
struct StrandVisibilityRecord
{
    StrandUInt segment;
    float u, coverage, depth;
};
struct StrandCamera
{
    STRAND_FLOAT4(eyeNear);
    STRAND_FLOAT4(rightTan); // right vector, tan(FOV/2); negative = -orthographic half-height
    STRAND_FLOAT4(upAspect); // up vector, aspect
    STRAND_FLOAT4(forwardFar);
};
struct StrandFrame
{
    StrandCamera camera, previousCamera;
    StrandUInt width, height, capacity, segmentCount;
    float phase, previousPhase, density, amplitude;
    float previousAmplitude;
    StrandUInt historyValid, padding0, padding1;
};
#ifdef __cplusplus
using StrandData = GPUBufferSpan;
using StrandRecords = GPUBufferSpan;
using StrandMaterials = GPUBufferSpan;
using StrandFrames = GPUBufferSpan;
using StrandStorage = GPUResourceHandle<ResourceViewKind::StorageImage>;
using StrandCounts = StrandStorage;
using StrandSampled = GPUResourceHandle<ResourceViewKind::SampledImage>;
#else
import Material;
typealias StrandData = BufferSpan<StrandSegment>;
typealias StrandRecords = RWBufferSpan<StrandVisibilityRecord>;
typealias StrandMaterials = BufferSpan<PathTraceMaterial>;
typealias StrandFrames = BufferSpan<StrandFrame>;
typealias StrandStorage = ResourceHandle<RWTexture2D<float4>>;
typealias StrandCounts = ResourceHandle<RWTexture2D<uint4>>;
typealias StrandSampled = ResourceHandle<Texture2D<float4>>;
#endif
#ifdef __cplusplus
struct alignas(16) StrandParameters
#else
struct StrandParameters
#endif
{
    StrandData segments;
    StrandRecords records;
    StrandMaterials materials;
    StrandFrames frame;
    StrandCounts counts;
    StrandStorage color, motion;
    StrandCounts identity;
    StrandSampled opaqueColor, opaqueDepth;
    StrandUInt hasOpaqueColor, hasOpaqueDepth, overflowPolicy, shadows;
    StrandUInt padding0, padding1;
    STRAND_FLOAT4(light); // direction toward light, intensity
    STRAND_FLOAT4(background);
};
#ifdef __cplusplus
inline constexpr uint64_t kStrandABI = 0x535452414e440001ull;
static_assert(sizeof(StrandSegment) == 112 && sizeof(StrandVisibilityRecord) == 16);
static_assert(sizeof(StrandCamera) == 64 && sizeof(StrandFrame) == 176);
static_assert(sizeof(StrandParameters) == 128 && offsetof(StrandParameters, light) == 96);
#endif
#undef STRAND_FLOAT4
#undef STRAND_UINT4
}

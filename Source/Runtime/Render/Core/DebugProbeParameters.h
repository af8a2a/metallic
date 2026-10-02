#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
#include <cstddef>
namespace metallic::render {
using ProbeUInt = uint32_t;
#else
import ShaderCore;
using Metallic;
namespace Metallic {
typealias ProbeUInt = uint;
#endif
struct DebugProbePush
{
    ProbeUInt byteOffset;
    ProbeUInt stride;
    ProbeUInt fieldOffset;
    ProbeUInt count;
    ProbeUInt scalarType; // 0 u32, 1 i32, 2 f32
    ProbeUInt operation; // 0 count, 1 outOfBounds, 2 nonFinite, 3 minMax
    ProbeUInt predicate; // all, eq, ne, lt, le, gt, ge
    ProbeUInt lowerBits;
    ProbeUInt upperBits;
    ProbeUInt groupCount;
    ProbeUInt bitOffset;
    ProbeUInt bitMask;
    ProbeUInt scale;
};
struct DebugProbePartial
{
    ProbeUInt finiteCount;
    ProbeUInt matchedCount;
    ProbeUInt nanCount;
    ProbeUInt infCount;
    ProbeUInt minBits;
    ProbeUInt maxBits;
    ProbeUInt firstIndex;
    ProbeUInt firstBits;
};
#ifdef __cplusplus
using ProbeInput = ShaderBuffer;
using ProbeOutput = ShaderBuffer;
#else
typealias ProbeInput = DescriptorHandle<ByteAddressBuffer>;
typealias ProbeOutput = DescriptorHandle<RWStructuredBuffer<DebugProbePartial>>;
#endif
struct DebugProbeParams {
    ProbeInput source;
    ProbeOutput output;
    DebugProbePush settings;
    ProbeUInt padding;
};
#ifdef __cplusplus
inline constexpr uint64_t kDebugProbeABI = 0x44424750524f0001ull;
static_assert(sizeof(DebugProbePush) == 52 && offsetof(DebugProbePush, groupCount) == 36);
static_assert(sizeof(DebugProbePartial) == 32);
static_assert(sizeof(DebugProbeParams) == 72 && offsetof(DebugProbeParams, settings) == 16);
#endif
} // namespace metallic::render (C++) / Metallic (Slang)

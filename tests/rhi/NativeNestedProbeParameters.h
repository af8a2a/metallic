#pragma once
#ifdef __cplusplus
#include "Runtime/Render/Core/ResourceRegistry.h"
namespace metallic::tests {
using NativeProbeRecords = render::ShaderBuffer;
using NativeProbeReadRecords = render::ShaderBuffer;
using NativeProbeRaw = render::ShaderBuffer;
using NativeProbeOutput = render::ShaderBuffer;
#else
import ShaderCore;
using Metallic;
struct NestedChild { uint4 first; uint4 second; };
struct NestedRecord { uint4 prefix; NestedChild children[2]; uint4 suffix; };
typealias NativeProbeRecords = DescriptorHandle<RWStructuredBuffer<NestedRecord>>;
typealias NativeProbeReadRecords = DescriptorHandle<StructuredBuffer<NestedRecord>>;
typealias NativeProbeRaw = DescriptorHandle<ByteAddressBuffer>;
typealias NativeProbeOutput = DescriptorHandle<RWByteAddressBuffer>;
#endif
struct NativeNestedProbeParameters
{
    NativeProbeRecords records;
    NativeProbeReadRecords readRecords;
    NativeProbeRaw raw;
    NativeProbeOutput output;
};
#ifdef __cplusplus
inline constexpr uint64_t kNativeNestedProbeABI = 0x4e41544e45530001ull;
static_assert(sizeof(NativeNestedProbeParameters) == 32);
} // namespace metallic::tests
#endif

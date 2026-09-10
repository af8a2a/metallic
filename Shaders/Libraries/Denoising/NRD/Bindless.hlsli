#pragma once
#define NRD_INTERNAL 1

// Slots describe algorithm inputs, never Vulkan descriptor bindings. Values are
// Metallic BindlessHandle::shaderIndex and may be anywhere in the native heap.
struct NrdResourceIndices
{
    uint sampled[32];
    uint storage[16];
    uint samplers[2];
};

struct NrdPushData
{
    // The RHI prepends this header to every native bindless pipeline's user data.
    uint imageShaderIndexBase;
    uint bufferShaderIndexBase;
    uint* constants;
    NrdResourceIndices* resources;
};

[[vk::push_constant]] NrdPushData gNrdPush;

#define NRD_CONSTANTS_START(name) struct name {
#define NRD_TYPE_float4x4 column_major float4x4
#define NRD_TYPE_float4 float4
#define NRD_TYPE_float3 float3
#define NRD_TYPE_float2 float2
#define NRD_TYPE_float float
#define NRD_TYPE_uint2 uint2
#define NRD_TYPE_int2 int2
#define NRD_TYPE_uint uint
#define NRD_TYPE_int int
#define NRD_CONSTANT(type, name) NRD_TYPE_##type nrd_##name;
#define NRD_CONSTANTS_END };
#define NRD_INPUTS_START
#define NRD_INPUT(...)
#define NRD_INPUTS_END
#define NRD_OUTPUTS_START
#define NRD_OUTPUT(...)
#define NRD_OUTPUTS_END
#define NRD_SAMPLERS_START
#define NRD_SAMPLER(...)
#define NRD_SAMPLERS_END

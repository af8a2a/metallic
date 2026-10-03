#ifndef METALLIC_STREAM_ATTRIBUTE_DECODE
#define METALLIC_STREAM_ATTRIBUTE_DECODE
import Core;
using Metallic;

// MeshletStreamPayloadFormat; packed streams reuse resident shading encoding.
uint streamNormalStride(uint format) { return format == 2u ? 16u : (format == 5u ? 4u : 0u); }
uint streamTangentStride(uint format) { return format == 2u ? 16u : (format == 6u ? 4u : 0u); }
float3 streamLoadNormal(StructuredBuffer<uint> words, uint page, uint offset, uint vertex, uint format)
{
    uint n = page + offset / 4u + vertex * (streamNormalStride(format) / 4u);
    return format == 5u ? decodeSceneNormal(words[n]) : asfloat(uint3(words[n],words[n+1u],words[n+2u]));
}
float4 streamLoadTangent(StructuredBuffer<uint> words, uint page, uint offset, uint vertex, uint format)
{
    uint t = page + offset / 4u + vertex * (streamTangentStride(format) / 4u);
    return format == 6u ? decodeSceneTangent(words[t]) : asfloat(uint4(words[t],words[t+1u],words[t+2u],words[t+3u]));
}
#endif

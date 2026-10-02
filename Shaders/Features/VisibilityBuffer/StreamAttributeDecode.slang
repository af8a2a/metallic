#ifndef METALLIC_STREAM_ATTRIBUTE_DECODE
#define METALLIC_STREAM_ATTRIBUTE_DECODE
import ShaderCore;
using Metallic;

// Statically specialized readers let the same decoder consume legacy descriptors
// and bounded BDA spans. Bounds are checked by the triangle decoder before loads.
interface IStreamReadBuffer<T>
{
    T load(uint index);
    uint length();
};
struct StreamDescriptorBuffer<T> : IStreamReadBuffer<T>
{
    StructuredBuffer<T> buffer;
    T load(uint index) { return buffer[index]; }
    uint length() { uint count, stride; buffer.GetDimensions(count, stride); return count; }
};
struct StreamDataBuffer<T> : IStreamReadBuffer<T>
{
    DataSpan<T> span;
    T load(uint index) { return span.data[index]; }
    uint length() { return span.length(); }
};

// MeshletStreamPayloadFormat; packed streams reuse resident shading encoding.
uint streamNormalStride(uint format) { return format == 2u ? 16u : (format == 5u ? 4u : 0u); }
uint streamTangentStride(uint format) { return format == 2u ? 16u : (format == 6u ? 4u : 0u); }
float3 streamLoadNormal<Words : IStreamReadBuffer<uint>>(Words words, uint page, uint offset, uint vertex, uint format)
{
    uint n = page + offset / 4u + vertex * (streamNormalStride(format) / 4u);
    return format == 5u ? decodeSceneNormal(words.load(n)) : asfloat(uint3(words.load(n),words.load(n+1u),words.load(n+2u)));
}
float4 streamLoadTangent<Words : IStreamReadBuffer<uint>>(Words words, uint page, uint offset, uint vertex, uint format)
{
    uint t = page + offset / 4u + vertex * (streamTangentStride(format) / 4u);
    return format == 6u ? decodeSceneTangent(words.load(t)) : asfloat(uint4(words.load(t),words.load(t+1u),words.load(t+2u),words.load(t+3u)));
}
float3 streamLoadNormal(StructuredBuffer<uint> words, uint page, uint offset, uint vertex, uint format)
{
    return streamLoadNormal(StreamDescriptorBuffer<uint>(words), page, offset, vertex, format);
}
float4 streamLoadTangent(StructuredBuffer<uint> words, uint page, uint offset, uint vertex, uint format)
{
    return streamLoadTangent(StreamDescriptorBuffer<uint>(words), page, offset, vertex, format);
}
#endif

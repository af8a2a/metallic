module EnvironmentPrefilterSamplingProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Lighting;
using Metallic.Lighting;

static const uint kSamplesPerLayer = 256u;

int2 referenceTexel(int2 pixel)
{
    int width = int(kEnvironmentFilterWidth);
    return int2((pixel.x % width + width) % width,
        clamp(pixel.y, 0, int(kEnvironmentFilterHeight) - 1));
}

float3 referenceSample(StructuredBuffer<float4> coefficients, float2 uv, uint layer, out float3 contrast)
{
    // Independent copy of the former buffer bilinear algorithm. This oracle
    // intentionally does not call the production texture-sampling helper.
    float2 texel = uv * float2(kEnvironmentFilterWidth, kEnvironmentFilterHeight) - 0.5;
    int2 p = int2(floor(texel));
    float2 f = frac(texel);
    int2 a = referenceTexel(p), b = referenceTexel(p + int2(1, 0));
    int2 c = referenceTexel(p + int2(0, 1)), d = referenceTexel(p + 1);
    uint base = layer * kEnvironmentFilterWidth * kEnvironmentFilterHeight;
    float3 ca = coefficients[base + uint(a.y) * kEnvironmentFilterWidth + uint(a.x)].rgb;
    float3 cb = coefficients[base + uint(b.y) * kEnvironmentFilterWidth + uint(b.x)].rgb;
    float3 cc = coefficients[base + uint(c.y) * kEnvironmentFilterWidth + uint(c.x)].rgb;
    float3 cd = coefficients[base + uint(d.y) * kEnvironmentFilterWidth + uint(d.x)].rgb;
    contrast = max(max(ca, cb), max(cc, cd)) - min(min(ca, cb), min(cc, cd));
    return lerp(lerp(ca, cb, f.x), lerp(cc, cd, f.x), f.y);
}

float2 sampleUv(uint sample)
{
    float2 size = float2(kEnvironmentFilterWidth, kEnvironmentFilterHeight);
    switch (sample) {
    case 0u: return float2(0, 0);
    case 1u: return float2(1, 1);
    case 2u: return float2(0, 1);
    case 3u: return float2(1, 0);
    case 4u: return float2(-0.271 / size.x, 0.371);
    case 5u: return float2(0.271 / size.x, 0.371);
    case 6u: return float2(1.0 - 0.271 / size.x, 0.371);
    case 7u: return float2(1.0 + 0.271 / size.x, 0.371);
    case 8u: return float2(0.319, -0.139 / size.y);
    case 9u: return float2(0.319, 0.139 / size.y);
    case 10u: return float2(0.731, 1.0 - 0.139 / size.y);
    case 11u: return float2(0.731, 1.0 + 0.139 / size.y);
    // Non-dyadic subtexel offsets around every deliberately bright fixture texel.
    case 12u: return float2(0.771, 0.639) / size;
    case 13u: return float2(0.229, 0.361) / size;
    case 14u: return (size - float2(0.771, 0.639)) / size;
    case 15u: return (size - float2(0.229, 0.361)) / size;
    case 16u: return (size * 0.5 + float2(0.771, 0.639)) / size;
    case 17u: return (size * 0.5 + float2(0.229, 0.361)) / size;
    case 18u: return environmentFilterUv(float3(0, 1, 0));
    case 19u: return environmentFilterUv(float3(0, -1, 0));
    case 20u: return environmentFilterUv(normalize(float3(-1, 0.137, 0.00001)));
    case 21u: return environmentFilterUv(normalize(float3(-1, 0.137, -0.00001)));
    }
    uint x = sample * 1664525u + 1013904223u;
    uint y = x * 1664525u + 1013904223u;
    float2 uv = float2(x & 65535u, y & 65535u) / 65535.0;
    return (sample & 1u) != 0u ? environmentFilterUv(environmentFilterDirection(uv)) : uv;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void environmentPrefilterSamplingProbeMain(uint3 id : SV_DispatchThreadID)
{
    let resources = getResourceParameters<EnvironmentPrefilterSamplingProbeResources>();
    let output = resolveBuffer<RWStructuredBuffer<float4>>(resources.output);
    let coefficients = resolveBuffer<StructuredBuffer<float4>>(resources.coefficients);
    let prefilter = resolveUniform(resources.prefilter);
    let samplerState = resolveUniform(resources.sampler);
    uint texelsPerLayer = kEnvironmentFilterWidth * kEnvironmentFilterHeight;
    uint texelCount = texelsPerLayer * kEnvironmentFilterLayers;
    if (id.x < texelCount) {
        uint layer = id.x / texelsPerLayer;
        uint pixel = id.x % texelsPerLayer;
        uint2 xy = uint2(pixel % kEnvironmentFilterWidth, pixel / kEnvironmentFilterWidth);
        float4 expected = coefficients[id.x];
        float4 loaded = prefilter.Load(int4(xy, layer, 0));
        float2 uv = (float2(xy) + 0.5) / float2(kEnvironmentFilterWidth, kEnvironmentFilterHeight);
        float3 filtered = environmentFilterLayer(prefilter, samplerState, uv, layer);
        output[id.x] = float4(any(asuint(loaded) != asuint(expected)) ? 1.0 : 0.0,
            any(asuint(filtered) != asuint(expected.rgb)) ? 1.0 : 0.0,
            all(isfinite(expected)) && all(isfinite(filtered)) ? 0.0 : 1.0,
            max(expected.x, max(expected.y, expected.z)));
    } else if (id.x < texelCount + kSamplesPerLayer * kEnvironmentFilterLayers) {
        uint sample = id.x - texelCount;
        uint layer = sample / kSamplesPerLayer;
        float2 uv = sampleUv(sample % kSamplesPerLayer);
        float3 contrast;
        float3 expected = referenceSample(coefficients, uv, layer, contrast);
        float3 actual = environmentFilterLayer(prefilter, samplerState, uv, layer);
        uint base = texelCount + sample * 4u;
        output[base] = float4(actual, 1);
        output[base + 1u] = float4(expected, 1);
        output[base + 2u] = float4(contrast, 0);
        output[base + 3u] = float4(uv, float(layer), 0);
    }
}

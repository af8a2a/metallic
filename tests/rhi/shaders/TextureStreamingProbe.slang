module TextureStreamingProbe;

import Core;
using Metallic;
#include "../TestResourceParameters.h"
import Material;
import StreamTextureSampling;
using Metallic.Material;

struct Push { uint slot; uint visible; float nearSourceLod; float farSourceLod; };

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    Push push = getConstants<Push>();
    Texture2D<float4> texture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(getResourceParameters<TextureStreamingProbeResources>().textures.load(push.slot)));
    let resources = getResourceParameters<TextureStreamingProbeResources>();
    StructuredBuffer<TextureSamplingMetadata> metadata = resolveBuffer<StructuredBuffer<TextureSamplingMetadata>>(resources.samplingMetadata);
    RWStructuredBuffer<uint> demandMip = resolveBuffer<RWStructuredBuffer<uint>>(resources.demandMip);
    RWStructuredBuffer<uint> demandHits = resolveBuffer<RWStructuredBuffer<uint>>(resources.demandHits);
    RWStructuredBuffer<float4> output = resolveBuffer<RWStructuredBuffer<float4>>(resources.output);
    SamplerState sampler = resolveUniform(resources.sampler);
    TextureSamplingMetadata state = metadata[push.slot];
    uint firstMip = state.residentFirstMipAndFlags & 0x7fffffffu;
    float minimumLod = state.samplingFloor;
    float2 uv = float2(0.375, 0.625);
    float nearValue = sampleResidentMaterialTexture(texture, sampler, uv,
        push.nearSourceLod - float(firstMip), minimumLod).r;
    float farValue = sampleResidentMaterialTexture(texture, sampler, uv,
        push.farSourceLod - float(firstMip), minimumLod).r;
    if (push.visible != 0u && (state.residentFirstMipAndFlags & 0x80000000u) != 0u) {
        uint ignored;
        InterlockedMin(demandMip[push.slot], uint(max(floor(push.nearSourceLod), 0.0)), ignored);
        InterlockedAdd(demandHits[push.slot], 1000u, ignored);
    }
    uint width, height;
    texture.GetDimensions(width, height);
    output[0] = float4(nearValue, farValue, minimumLod, float(firstMip));
    output[1] = float4(float(demandMip[push.slot]), float(demandHits[push.slot]),
        float(state.residentWidth << firstMip), float(width));
}

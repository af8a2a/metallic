module TextureLodMetadataProbe;
import Core;
import Material;
import SceneSurface;
import ScenePathMath;
import StreamTextureSampling;
#include "../TestResourceParameters.h"
using Metallic;
using Metallic.Material;
using Metallic.SceneSurface;
using Metallic.ScenePath;
struct LodCase { uint4 dimensions; float4 constants; };
[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    let resources = getResourceParameters<TextureLodProbeResources>();
    StructuredBuffer<LodCase> cases = resolveBuffer<StructuredBuffer<LodCase>>(resources.cases);
    StructuredBuffer<TextureSamplingMetadata> metadata = resolveBuffer<StructuredBuffer<TextureSamplingMetadata>>(resources.samplingMetadata);
    RWStructuredBuffer<float4> output = resolveBuffer<RWStructuredBuffer<float4>>(resources.output);
    LodCase input = cases[id.y];
    TextureSamplingMetadata state = metadata[id.y];
    Texture2D<float4> texture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(resources.textures.load(id.y)));
    SamplerState sampler = resolveUniform(resources.sampler);
    MaterialTextureInfo info = (MaterialTextureInfo)0;
    info.sourceLodTerm = input.constants.x;
    info.sourceMipCount = input.dimensions.z;
    // Integer crossings include both neighboring representable-scale offsets.
    float delta = (int(id.x % 3u) - 1) * 0.00000095367431640625;
    float baseLod = float(id.x / 3u) - 7.0 - input.constants.y + delta;
    float uvBias = id.x % 5u == 0u ? -1.0 : (id.x % 5u == 1u ? 2.0 : 0.0);
    uint width, height, mipCount;
    texture.GetDimensions(0, width, height, mipCount);
    float oldResident = clamp((baseLod + 0.5 * safeLog2(float(width) * float(height))) + uvBias, 0.0, float(mipCount - 1u));
    float oldSource = clamp((baseLod + 0.5 * safeLog2(float(input.dimensions.x) * float(input.dimensions.y))) + uvBias, 0.0, float(info.sourceMipCount - 1u));
    float resident = materialResidentTextureLod(state, baseLod, uvBias);
    float source = materialSourceTextureLod(info, baseLod, uvBias);
    float oldSample = sampleResidentMaterialTexture(texture, sampler, float2(0.37, 0.63), oldResident, state.samplingFloor).r;
    float newSample = sampleResidentMaterialTexture(texture, sampler, float2(0.37, 0.63), resident, state.samplingFloor).r;
    float naive = clamp(baseLod + textureDimensionLod(info.sourceLodTerm) + uvBias - float(state.residentFirstMipAndFlags & 0x7fffffffu), 0.0, float(mipCount - 1u));
    uint index = (id.y * 64u + id.x) * 3u;
    output[index] = float4(oldResident, resident, oldSource, source);
    output[index + 1u] = float4(oldSample, newSample, naive, state.samplingFloor);
    output[index + 2u] = float4(float(width), float(height), float(mipCount), baseLod);
}

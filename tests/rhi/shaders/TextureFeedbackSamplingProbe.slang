module TextureFeedbackSamplingProbe;

import Core;
import Material;
import ScenePathTypes;
import SceneSurface;
using Metallic;
using Metallic.Material;
using Metallic.SceneSurface;

[shader("compute")]
[numthreads(8, 8, 1)]
void textureFeedbackSamplingMain(uint3 id : SV_DispatchThreadID)
{
    // 8 frame phases x capture enabled/disabled x regular/Mip0/alpha sampling.
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.sampleFrame = id.z & 7u;
    push.textureFeedbackEnabled = (id.z >> 3u) & 1u;
    push.materialTextureCount = 48u;
    push = withTextureFeedbackSampling(push, id.xy);
    MaterialTextureInfo info = (MaterialTextureInfo)0;
    info.textureIndex = id.z;
    info.sourceLodTerm = 3.0;
    info.sourceMipCount = 4u;
    info.transform0 = float4(1, 0, 0, 0);
    info.transform1 = float4(0, 1, 0, 0);
    HitInfo hit = (HitInfo)0;
    hit.geometryNormal = float3(0, 0, 1);
    RayCone cone = (RayCone)0;
    cone.width = 0.5;
    TextureFootprint footprint = (TextureFootprint)0;
    footprint.flags = kTextureFootprintIsotropic0;
    footprint.isotropicLod0 = -1.0;
    uint seed = 12345u;
    float4 sampled = float4(0);
    // Nine material slots may alias one hot texture. Each selected use contributes
    // exactly one demand hit, including the deterministic Mip0 path.
    for (uint slot = 0; slot < 9u; ++slot) {
        if (id.z >= 32u) {
            float alpha = loadMaterialAlphaCoverage(info, float2(0.25), push);
            sampled = float4(0, alpha, 0, alpha);
        } else {
            sampled = loadMaterialTextureForMaterial(info, float2(0.25), float4(-1),
                push, cone, hit, float3(0, 0, 1), id.z >= 16u, seed, footprint);
        }
    }
    uint index = (id.z * 4096u + id.y * 64u + id.x) * 2u;
    RWStructuredBuffer<float4> output = resolveBuffer<RWStructuredBuffer<float4>>(
        getResourceParameters<SceneResourceParameters>().probeOutput);
    output[index] = sampled;
    output[index + 1u] = float4(float(push.textureFeedbackEnabled),
        materialTextureSamplingFloor(id.z), seed == 12345u ? 1.0 : 0.0, 1.0);
}

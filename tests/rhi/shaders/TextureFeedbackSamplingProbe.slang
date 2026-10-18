module TextureFeedbackSamplingProbe;

import Core;
import Material;
import ScenePathTypes;
import SceneSurface;
using Metallic;
using Metallic.Material;
using Metallic.SceneSurface;

internal void probeResolvedMaterialFootprint(uint3 id)
{
    // Paired compatibility and resolved paths share the same inputs and RNG.
    // 4 cone cases x 4 initial footprint states x Mip0 x absent slot.
    uint test = (id.z - 48u) / 2u;
    bool resolvedPath = (id.z & 1u) != 0u;
    uint coneCase = test & 3u;
    uint footprintCase = (test >> 2u) & 3u;
    bool mip0 = (test & 16u) != 0u;
    bool absent = (test & 32u) != 0u;
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.materialTextureCount = 176u;
    push.textureFeedbackEnabled = 1u;
    push.sampleFrame = test & 7u;
    push = withTextureFeedbackSampling(push, id.xy);
    MaterialTextureInfo info = (MaterialTextureInfo)0;
    info.textureIndex = absent ? 0xffffffffu : id.z;
    info.sourceLodTerm = 3.0;
    info.sourceMipCount = 4u;
    // Anisotropic UV transform with determinant one: no scalar LOD bias.
    info.transform0 = float4(2, 0, 0.125, 0);
    info.transform1 = float4(0, 0.5, -0.25, 0);
    HitInfo hit = (HitInfo)0;
    hit.geometryNormal = float3(0, 0, 1);
    hit.rayConeLodConstant = 0.25;
    RayCone cone = (RayCone)0;
    cone.width = coneCase == 0u ? 0.125 : coneCase == 1u ? -0.5 : coneCase == 2u ? 0.0 : 2.0;
    float3 direction = coneCase < 2u ? float3(0, 0, 1) : float3(sqrt(1.0 - 0.0625 * 0.0625), 0, 0.0625);
    TextureFootprint footprint = (TextureFootprint)0;
    footprint.flags = (footprintCase & 1u) != 0u ? kTextureFootprintIsotropic0 : 0u;
    if ((footprintCase & 2u) != 0u) {
        footprint.flags |= kTextureFootprintGradients0 | kTextureFootprintGradients1 | kTextureFootprintIsotropic1;
    }
    footprint.isotropicLod0 = -0.75;
    footprint.isotropicLod1 = 2.5;
    footprint.uv0Dx = float2(0.25, 0.03125);
    footprint.uv0Dy = float2(-0.0625, 0.5);
    footprint.uv1Dx = float2(0.5, 0.125);
    footprint.uv1Dy = float2(-0.25, 0.75);
    TextureFootprint resolved = resolveRayConeTextureFootprint(hit, cone, direction, footprint);
    bool preserved = resolved.flags == (footprint.flags | kTextureFootprintIsotropic0)
        && all(resolved.uv0Dx == footprint.uv0Dx) && all(resolved.uv0Dy == footprint.uv0Dy)
        && all(resolved.uv1Dx == footprint.uv1Dx) && all(resolved.uv1Dy == footprint.uv1Dy)
        && resolved.isotropicLod1 == footprint.isotropicLod1;
    uint seed = 12345u + id.y * 64u + id.x;
    bool sampled = false;
    float4 value = float4(0);
    for (uint slot = 0u; slot < 9u; ++slot) {
        if (resolvedPath) {
            value = loadMaterialTextureForMaterialResolved(info, float2(0.25), float4(-1),
                push, mip0, seed, resolved, sampled);
        } else {
            value = loadMaterialTextureForMaterial(info, float2(0.25), float4(-1),
                push, cone, hit, direction, mip0, seed, footprint, sampled);
        }
    }
    uint index = (id.z * 4096u + id.y * 64u + id.x) * 3u;
    RWStructuredBuffer<float4> output = resolveBuffer<RWStructuredBuffer<float4>>(
        getResourceParameters<SceneResourceParameters>().probeOutput);
    output[index] = value;
    output[index + 1u] = float4(seed & 0xffffu, seed >> 16u, sampled ? 1.0 : 0.0, preserved ? 1.0 : 0.0);
    output[index + 2u] = float4(resolved.isotropicLod0, float(push.textureFeedbackEnabled), 0, 0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void textureFeedbackSamplingMain(uint3 id : SV_DispatchThreadID)
{
    if (id.z >= 48u) {
        probeResolvedMaterialFootprint(id);
        return;
    }
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
    uint index = (id.z * 4096u + id.y * 64u + id.x) * 3u;
    RWStructuredBuffer<float4> output = resolveBuffer<RWStructuredBuffer<float4>>(
        getResourceParameters<SceneResourceParameters>().probeOutput);
    output[index] = sampled;
    output[index + 1u] = float4(float(push.textureFeedbackEnabled),
        materialTextureSamplingFloor(id.z), seed == 12345u ? 1.0 : 0.0, 1.0);
    output[index + 2u] = float4(0);
}

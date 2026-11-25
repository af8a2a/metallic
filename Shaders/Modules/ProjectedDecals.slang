#language slang 2026
module ProjectedDecals;

import Core;
import Material;
import MaterialTextureDecode;
import ScenePathTypes;
import SceneSurface;
import StreamTextureSampling;
import WorkingColor;

namespace Metallic
{
using Metallic.Material;
using Metallic.SceneSurface;

#include "../../Source/Runtime/Render/Material/DeferredDecalData.h"

// Project the camera ray's footprint onto the geometric tangent plane, then
// onto the decal axes. This also works for non-neighboring material-bin lanes.
internal float2 decalUvFootprint(DeferredDecalData decal, float3 normal, float3 rayDirection, float rayWidth)
{
    float cosine = dot(normal, rayDirection);
    float inverseCosine = (cosine < 0.0 ? -1.0 : 1.0) / max(abs(cosine), 1e-6);
    float3 u = decal.worldToDecal[0].xyz;
    float3 v = decal.worldToDecal[1].xyz;
    return rayWidth * float2(length(u - normal * (dot(u, rayDirection) * inverseCosine)),
        length(v - normal * (dot(v, rayDirection) * inverseCosine)));
}

internal float4 sampleDecal(MaterialTextureInfo info, uint2 sourceExtent, SamplerHandle sampler, float2 uv,
    float2 uvFootprint, ScenePathTracePush push)
{
    if (!hasMaterialTexture(info, push)) { return float4(1.0); }
    TextureSamplingMetadata metadata = sceneTextureSamplingMetadata()[info.textureIndex];
    float residentLod = projectedTextureLod(uvFootprint, uint2(metadata.residentWidth, metadata.residentHeight));
    if (push.textureFeedbackEnabled != 0u && streamTextureDemandEnabled(metadata)) {
        uint wanted = uint(clamp(projectedTextureLod(uvFootprint, sourceExtent), 0.0, float(info.sourceMipCount - 1u)));
        recordTextureDemand(sceneTextureDemandMip(), sceneTextureDemandHits(), info.textureIndex, wanted);
    }
    return decodeMaterialTextureSample(sampleResidentMaterialTexture(sceneMaterialTexture(info.textureIndex),
        resolveUniform(sampler), uv, residentLod, max(metadata.samplingFloor, 0.0)), info.transform0.w);
}

public void applyProjectedDecals(uint2 pixel, uint sourceMaterialIndex, HitInfo hit, RayCone cone, float3 rayDirection,
    ScenePathTracePush push, inout float3 baseColor, inout float roughness)
{
    let resources = getResourceParameterField<SceneRealtimeResourceParameters, DecalResourceParameters>(kSceneDecalsOffset);
    let records = resolveBuffer<StructuredBuffer<DeferredDecalData>>(resources.records);
    bool tiled = resources.tileRanges.index != 0xffffffffu;
    uint first = 0u, count;
    if (tiled) {
        uint columns = (push.width + kDeferredDecalTileSize - 1u) / kDeferredDecalTileSize;
        uint2 tile = pixel / kDeferredDecalTileSize;
        uint2 range = resolveBuffer<StructuredBuffer<uint2>>(resources.tileRanges)[tile.y * columns + tile.x];
        first = range.x;
        count = range.y;
    } else {
        uint stride;
        records.GetDimensions(count, stride);
    }
    uint receiverMask = 1u;
    if (resources.receiverMasks.index != 0xffffffffu && sourceMaterialIndex != 0xffffffffu) {
        receiverMask = resolveBuffer<StructuredBuffer<uint>>(resources.receiverMasks)[sourceMaterialIndex];
    }
    for (uint i = 0u; i < count; ++i) {
        uint index = i;
        if (tiled) { index = resolveBuffer<StructuredBuffer<uint>>(resources.tileIndices)[first + i]; }
        DeferredDecalData decal = records[index];
        if ((decal.receiverMask & receiverMask) == 0u || dot(decal.worldToDecal[2].xyz, hit.geometryNormal) <= 0.0) { continue; }
        float4 position = float4(hit.position, 1.0);
        float3 local = float3(dot(decal.worldToDecal[0], position), dot(decal.worldToDecal[1], position),
            dot(decal.worldToDecal[2], position));
        if (any(abs(local) > 0.5)) { continue; }
        let material = resolveBuffer<StructuredBuffer<DeferredDecalMaterial>>(resources.materials)[decal.materialIndex];
        float2 uv = local.xy + 0.5;
        float2 footprint = decalUvFootprint(decal, hit.geometryNormal, rayDirection, cone.width);
        float4 colorSample = sampleDecal(material.baseColorTexture, material.baseColorSourceExtent,
            resources.sampler, uv, footprint, push);
        float3 tint = WorkingColor::fromLinearRec709(material.baseColor.rgb);
        float3 color = hasMaterialTexture(material.baseColorTexture, push)
            ? materialTextureModulate(tint, colorSample.rgb, material.baseColorTexture.transform0.w)
            : tint;
        baseColor = lerp(baseColor, saturate(color), colorSample.a * material.baseColor.a);
        if (material.roughnessOpacity > 0.0) {
            float roughnessSample = sampleDecal(material.roughnessTexture, material.roughnessSourceExtent,
                resources.sampler, uv, footprint, push).r;
            roughness = lerp(roughness, clamp(material.roughness * roughnessSample, 0.001, 1.0),
                colorSample.a * material.roughnessOpacity);
        }
    }
}
} // namespace Metallic

#language slang 2026
module VirtualShadowMapWorld;

import ShaderCore;
import VirtualShadowMapCommon;
using Metallic;

namespace Metallic.Shadow
{
#include "../../Source/Runtime/Render/Shadow/VirtualShadowMapWorld.h"

public typealias WorldShadowFaces = ResourceHandle<StructuredBuffer<VirtualShadowMapWorldFace>>;

public VirtualShadowMapWorldFace getDirectionalWorldShadowFace(WorldShadowFaces faces, uint celestialSlot)
{
    if (faces.index == 0xffffffffu) { return {}; }
    return resolveUniform(faces)[celestialSlot * 6u];
}

public VirtualShadowMapWorldFace getLocalWorldShadowFace(WorldShadowFaces faces, uint sourceSlot,
    bool isSpotLight, float3 lightToReceiver)
{
    if (faces.index == 0xffffffffu) { return {}; }
    // The first two source entries are the celestial lights. All entries own
    // six canonical face slots independently of the surface output packing.
    uint cubeFace = isSpotLight ? 0u : virtualShadowMapPointFace(lightToReceiver);
    return resolveUniform(faces)[(sourceSlot + 2u) * 6u + cubeFace];
}

float worldPageDepth(VirtualShadowMapWorldFace face, uint physicalPageIndex, uint2 texel)
{
    uint2 physicalPage = uint2(physicalPageIndex % face.physicalPagesX, physicalPageIndex / face.physicalPagesX);
    uint2 atlasTexel = physicalPage * face.pageSize + texel % face.pageSize;
    if (face.physicalPagesY != 0u) {
        uint layerHeight = face.physicalPagesY * face.pageSize;
        let atlas = resolveNonUniform(ResourceHandle<Texture2DArray<float>>(face.atlas));
        return atlas.Load(int4(atlasTexel.x, atlasTexel.y % layerHeight, atlasTexel.y / layerHeight, 0));
    }
    let atlas = resolveNonUniform(ResourceHandle<Texture2D<float>>(face.atlas));
    return atlas.Load(int3(atlasTexel, 0));
}

// UE SkyAtmosphere.usf calls SampleVirtualShadowMapDirectional with zero ray-start
// distance and no receiver-plane bias. The caller supplies a published face.
public float sampleDirectionalVirtualShadowMap(VirtualShadowMapWorldFace face, float3 worldPosition)
{
    let projections = resolveNonUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapProjection>>(face.projections));
    let pages = resolveNonUniform(ResourceHandle<StructuredBuffer<uint4>>(face.pages));
    let statistics = resolveNonUniform(ResourceHandle<StructuredBuffer<uint>>(face.statistics));
    uint firstLevel = statistics[5] != 0u ? face.clipmapCount : 0u;
    for (uint level = firstLevel; level <= face.clipmapCount; ++level) {
        VirtualShadowMapProjection projection = projections[level];
        float3 projected = virtualShadowMapProjectClip(worldPosition, projection).xyz;
        uint resolution = projection.page.y * face.pageSize;
        bool fine = level < face.clipmapCount;
        float margin = fine ? 1.0f / float(resolution) : 0.0f;
        if (any(projected.xy < -1.0f + margin) || any(projected.xy >= 1.0f - margin)) { continue; }
        uint2 texel = virtualShadowMapLocalTexel(projected.xy, resolution);
        uint2 page = texel / face.pageSize;
        uint index = projection.page.x + page.y * projection.page.y + page.x;
        uint4 entry = pages[index];
        if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 1u) == 0u) { continue; }
        // Layered SMRT may omit casters outside visible receiver subtiles.
        // Volume samples in those tiles must descend to a complete ancestor.
        if (fine && face.validReceiverMaskByteOffset != 0u) {
            uint2 tile = (texel % face.pageSize) * 8u / face.pageSize;
            uint bit = tile.y * 8u + tile.x;
            let table = resolveNonUniform(ResourceHandle<ByteAddressBuffer>(face.pages));
            uint mask = table.Load(face.validReceiverMaskByteOffset + index * 8u + (bit >> 5u) * 4u);
            if ((mask & (1u << (bit & 31u))) == 0u) { continue; }
        }
        // The projection's Z bounds enclose casters. Air behind the last caster
        // can still be in shadow, while an empty reversed-Z texel sees the sky.
        return worldPageDepth(face, entry.x, texel) > max(projected.z, 0.0f) ? 0.0f : 1.0f;
    }
    return 1.0f;
}

// UE VirtualShadowMapProjectionCommon.ush::SampleVirtualShadowMapLocal,
// called from VolumetricFog.usf with no surface normal or ray-start bias.
public float sampleLocalVirtualShadowMap(VirtualShadowMapWorldFace face, float3 worldPosition)
{
    let projections = resolveNonUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapProjection>>(face.projections));
    VirtualShadowMapProjection base = projections[0];
    float4 clip = virtualShadowMapProjectClip(worldPosition, base);
    if (clip.w <= base.originNear.w) { return 1.0f; }
    float3 projected = clip.xyz / clip.w;
    if (any(abs(projected.xy) > 1.0f)) { return 1.0f; }

    let pages = resolveNonUniform(ResourceHandle<StructuredBuffer<uint4>>(face.pages));
    let statistics = resolveNonUniform(ResourceHandle<StructuredBuffer<uint>>(face.statistics));
    // Work overflow invalidates all fine pages for this face. The final
    // projection has one page covering the whole light frustum.
    uint firstMip = statistics[5] != 0u ? face.clipmapCount : 0u;
    for (uint mip = firstMip; mip <= face.clipmapCount; ++mip) {
        uint4 layout = projections[mip].page;
        uint2 texel = virtualShadowMapLocalTexel(projected.xy, layout.y * face.pageSize);
        uint2 page = texel / face.pageSize;
        uint4 entry = pages[layout.x + page.y * layout.y + page.x];
        if (entry.x == 0xffffffffu || (entry.y & 2u) == 0u || (entry.w & 1u) == 0u) { continue; }

        // Metallic bounds an unlimited light's shadow far plane to its casters.
        // Beyond it, empty reversed-Z texels still see light and blockers still occlude.
        return worldPageDepth(face, entry.x, texel) > max(projected.z, 0.0f) ? 0.0f : 1.0f;
    }
    return 1.0f;
}
} // namespace Metallic.Shadow

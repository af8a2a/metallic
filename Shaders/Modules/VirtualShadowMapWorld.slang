#language slang 2026
module VirtualShadowMapWorld;

import ShaderCore;
import VirtualShadowMapCommon;
using Metallic;

namespace Metallic.Shadow
{
#include "../../Source/Runtime/Render/Shadow/VirtualShadowMapWorld.h"

public typealias WorldShadowFaces = ResourceHandle<StructuredBuffer<VirtualShadowMapWorldFace>>;

public VirtualShadowMapWorldFace getLocalWorldShadowFace(WorldShadowFaces faces, uint sourceSlot,
    bool isSpotLight, float3 lightToReceiver)
{
    if (faces.index == 0xffffffffu) { return {}; }
    // The first two source entries are the celestial lights. All entries own
    // six canonical face slots independently of the surface output packing.
    uint cubeFace = isSpotLight ? 0u : virtualShadowMapPointFace(lightToReceiver);
    return resolveUniform(faces)[(sourceSlot + 2u) * 6u + cubeFace];
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

        uint2 physicalPage = uint2(entry.x % face.physicalPagesX, entry.x / face.physicalPagesX);
        uint2 atlasTexel = physicalPage * face.pageSize + texel % face.pageSize;
        float depth;
        if (face.physicalPagesY != 0u) {
            uint layerHeight = face.physicalPagesY * face.pageSize;
            let atlas = resolveNonUniform(ResourceHandle<Texture2DArray<float>>(face.atlas));
            depth = atlas.Load(int4(atlasTexel.x, atlasTexel.y % layerHeight, atlasTexel.y / layerHeight, 0));
        } else {
            let atlas = resolveNonUniform(ResourceHandle<Texture2D<float>>(face.atlas));
            depth = atlas.Load(int3(atlasTexel, 0));
        }
        // Metallic bounds an unlimited light's shadow far plane to its casters.
        // Beyond it, empty reversed-Z texels still see light and blockers still occlude.
        return depth > max(projected.z, 0.0f) ? 0.0f : 1.0f;
    }
    return 1.0f;
}
} // namespace Metallic.Shadow

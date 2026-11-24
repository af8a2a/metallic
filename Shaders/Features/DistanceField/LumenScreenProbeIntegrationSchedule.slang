module LumenScreenProbeIntegrationSchedule;

import ShaderCore;
import ParameterRoot;
import LumenProbeIntegration;

using Metallic;
using Metallic.Lumen;

struct LumenProbeIntegrationScheduleParams
{
    ResourceHandle<Texture2D<float4>> position, specular;
    ResourceHandle<RWStructuredBuffer<uint>> tiles, indirectArgs;
    uint width, height, maxDispatchX;
};

[shader("compute")]
[numthreads(kProbeIntegrationModeCount, 1, 1)]
void lumenResetProbeIntegrationTilesMain(uint mode : SV_GroupIndex)
{
    let params = getParameters<LumenProbeIntegrationScheduleParams>();
    resolveUniform(params.tiles)[mode] = 0u;
}

groupshared uint tileMode;

// UE marks a tile with the union of its receiver requirements, then appends it to the
// most capable list. An opaque Metallic tile needs either preintegrated diffuse alone
// or the existing rough-specular loop. Empty tiles retain the normal diffuse writes.
[shader("compute")]
[numthreads(kProbeIntegrationTileSize, kProbeIntegrationTileSize, 1)]
void lumenClassifyProbeIntegrationTilesMain(uint3 group : SV_GroupID, uint3 thread : SV_DispatchThreadID,
    uint lane : SV_GroupIndex)
{
    let params = getParameters<LumenProbeIntegrationScheduleParams>();
    if (lane == 0u) { tileMode = kProbeIntegrationSimple; }
    GroupMemoryBarrierWithGroupSync();

    bool usesRoughSpecular = false;
    if (all(thread.xy < uint2(params.width, params.height)) &&
        resolveUniform(params.position).Load(int3(thread.xy, 0)).w > 0.0f) {
        const float roughness = resolveUniform(params.specular).Load(int3(thread.xy, 0)).w;
        usesRoughSpecular = lumenProbeRoughSpecularDiffuseLerp(roughness) < 1.0f;
    }
#if defined(METALLIC_LUMEN_INTEGRATION_WAVE_OPS)
    const uint roughLanes = WaveActiveCountBits(usesRoughSpecular);
    usesRoughSpecular = WaveIsFirstLane() && roughLanes != 0u;
#endif
    if (usesRoughSpecular) { InterlockedOr(tileMode, kProbeIntegrationRoughSpecular); }
    GroupMemoryBarrierWithGroupSync();

    if (lane == 0u) {
        let tiles = resolveUniform(params.tiles);
        const uint2 dimensions = lumenProbeIntegrationTileDimensions(uint2(params.width, params.height));
        const uint capacity = dimensions.x * dimensions.y;
        uint index;
        InterlockedAdd(tiles[tileMode], 1u, index);
        tiles[kProbeIntegrationModeCount + tileMode * capacity + index] = group.y * dimensions.x + group.x;
    }
}

[shader("compute")]
[numthreads(kProbeIntegrationModeCount, 1, 1)]
void lumenBuildProbeIntegrationIndirectArgsMain(uint mode : SV_GroupIndex)
{
    let params = getParameters<LumenProbeIntegrationScheduleParams>();
    const uint count = resolveUniform(params.tiles)[mode];
    let args = resolveUniform(params.indirectArgs);
    const uint offset = mode * 3u;
    args[offset] = min(count, params.maxDispatchX);
    args[offset + 1u] = max(1u, (count + params.maxDispatchX - 1u) / params.maxDispatchX);
    args[offset + 2u] = 1u;
}

#ifndef MESHLET_LOD_SLANG
#define MESHLET_LOD_SLANG

#include "GPUDrivenSceneCommon.slang"
#include "MeshletLodMetric.slang"

struct MeshletLodGroupRecord
{
    float4 sphere;
    float error;
    uint level;
    uint flags;
    uint reserved;
};

float meshletLodPixelError(MeshletLodGroupRecord group, GPUDrivenPreviewInstance instance,
    float4 eye, float4 forward, float4 projection)
{
    return meshletLodProjectedError(group.sphere, group.error,
        instance.world0, instance.world1, instance.world2, instance.world3,
        eye, forward, projection);
}

bool meshletLodNeedsFine(MeshletLodGroupRecord group, GPUDrivenPreviewInstance instance,
    float4 eye, float4 forward, float4 projection, uint manualLevel)
{
    return meshletLodGroupNeedsFine(group.sphere, group.error, group.level, group.flags,
        instance.world0, instance.world1, instance.world2, instance.world3,
        eye, forward, projection, manualLevel);
}

#endif

#language slang 2026
implementing GPUDriven;

namespace Metallic.GPUDriven
{
public struct MeshletLodGroupRecord
{
    public float4 sphere;
    public float error;
    public uint level;
    public uint flags;
    public uint reserved;
};

public float meshletLodPixelError(MeshletLodGroupRecord group, GPUDrivenPreviewInstance instance,
    float4 eye, float4 forward, float4 projection)
{
    return meshletLodProjectedError(group.sphere, group.error,
        instance.world0, instance.world1, instance.world2, instance.world3,
        eye, forward, projection);
}

public bool meshletLodNeedsFine(MeshletLodGroupRecord group, GPUDrivenPreviewInstance instance,
    float4 eye, float4 forward, float4 projection, uint manualLevel)
{
    return meshletLodGroupNeedsFine(group.sphere, group.error, group.level, group.flags,
        instance.world0, instance.world1, instance.world2, instance.world3,
        eye, forward, projection, manualLevel);
}
} // namespace Metallic.GPUDriven

#language slang 2026
module StreamRayHit;

namespace Metallic.StreamGeometry
{
// Raw committed triangle identity only. Surface reconstruction and all material
// transport stay in the ray-generation integrator. Negative t denotes a miss.
public struct StreamRayHit
{
    public float t;
    public uint instanceId;
    public uint clusterId;
    public uint primitiveId;
    public float2 barycentrics;
};
} // namespace Metallic.StreamGeometry

#language slang 2026
module AtmosphereVirtualShadowMap;

import Atmosphere;
import VirtualShadowMapWorld;
using Metallic.Shadow;

namespace Metallic.Atmosphere
{
public struct DirectionalAtmosphereShadow : IAtmosphereLightShadow
{
    private WorldShadowFaces faces;
    private uint lightMask;
    private float3 rayOriginMeters;
    private VirtualShadowMapWorldFace face;
    private float3 rayStartMeters;
    private float3 rayDirectionMetersPerKm;

    public static DirectionalAtmosphereShadow forRay(uint facesIndex, uint mask, float3 originMeters)
    {
        DirectionalAtmosphereShadow shadow;
        shadow.faces.index = facesIndex;
        shadow.lightMask = mask;
        shadow.rayOriginMeters = originMeters;
        shadow.face = {};
        shadow.rayStartMeters = originMeters;
        shadow.rayDirectionMetersPerKm = float3(0.0);
        return shadow;
    }

    [mutating] public void selectLight(uint slot, float3 direction, float startDepthKm)
    {
        face = {};
        if ((lightMask & (1u << slot)) != 0u) { face = getDirectionalWorldShadowFace(faces, slot); }
        // Keep the shadow receiver in world metres to preserve near-ground detail at planetary radii.
        rayDirectionMetersPerKm = direction * 1000.0;
        rayStartMeters = rayOriginMeters + startDepthKm * rayDirectionMetersPerKm;
    }

    public float visibility(float rayDistanceKm)
    {
        if (face.pageSize == 0u) { return 1.0; }
        return sampleDirectionalVirtualShadowMap(face, rayStartMeters + rayDistanceKm * rayDirectionMetersPerKm);
    }
};

} // namespace Metallic.Atmosphere

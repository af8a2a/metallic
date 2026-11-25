module RectLightSolidAngle;

import RectLightLTC;

namespace Metallic.Lighting
{

public struct RectLightSolidAngleSample
{
    public float3 direction;
    public float distance;
    public float solidAngle;
};

// Ureña, Fajardo and King, An Area-Preserving Parametrization for Spherical
// Rectangles (2013). The asin fit and small-angle branch match UE 5.7.4.
float sphericalRectAsin(float x)
{
    float a = saturate(abs(x));
    bool inner = a < 0.5f;
    float a2 = inner ? a * a : 0.5f - 0.5f * a;
    a = inner ? a : sqrt(a2);
    float r = 0.100323f;
    r = mad(r, a2, 0.163288f);
    r = mad(r, a2, 1.00011f) * a;
    r = inner ? r : (3.1415926535897932f / 2.0f) - 2.0f * r;
    return asfloat(asuint(r) ^ (asuint(x) & 0x80000000u));
}

// The caller handles the emitting plane and authored attenuation.
public RectLightSolidAngleSample sampleRectLightSolidAngle(GPURectLight light, float3 center, float2 random)
{
    const float2 extent = float2(length(light.halfWidth.xyz), length(light.halfHeight.xyz));
    const float3x3 axes = float3x3(light.halfWidth.xyz / extent.x, light.halfHeight.xyz / extent.y, light.direction.xyz);
    const float3 local = mul(axes, center);
    const float x0 = local.x - extent.x;
    const float x1 = local.x + extent.x;
    const float y0 = local.y - extent.y;
    const float y1 = local.y + extent.y;
    const float z0 = -abs(local.z);
    const float z0sq = local.z * local.z;
    const float b0 = -y0 * rsqrt(z0sq + y0 * y0);
    const float n1z = x1 * rsqrt(z0sq + x1 * x1);
    const float b1 = y1 * rsqrt(z0sq + y1 * y1);
    const float n3z = -x0 * rsqrt(z0sq + x0 * x0);
    const float g0g1 = sphericalRectAsin(b0 * n1z) + sphericalRectAsin(n1z * b1);
    const float k = sphericalRectAsin(b1 * n3z) + sphericalRectAsin(n3z * b0);

    RectLightSolidAngleSample result = {};
    result.solidAngle = g0g1 + k;
    if (!(result.solidAngle > 0.0f)) { return result; }

    float xu, yv;
    if (result.solidAngle > 1e-3f) {
        const float au = random.x * result.solidAngle - k;
        const float fu = (cos(au) * b0 + b1) / sin(au);
        const float cu = clamp(rsqrt(fu * fu + b0 * b0) * (fu > 0.0f ? 1.0f : -1.0f), -1.0f, 1.0f);
        xu = clamp(-(cu * z0) * rsqrt(1.0f - cu * cu), x0, x1);

        const float d2 = xu * xu + z0 * z0;
        const float h0 = y0 * rsqrt(d2 + y0 * y0);
        const float h1 = y1 * rsqrt(d2 + y1 * y1);
        const float hv = h0 + random.y * (h1 - h0);
        const float rv = 1.0f - hv * hv;
        yv = rv > 0.0f ? hv * d2 * rsqrt(rv * d2) : y1;
    } else {
        xu = lerp(x0, x1, random.x);
        yv = lerp(y0, y1, random.y);
    }

    const float distanceSquared = xu * xu + yv * yv + z0 * z0;
    const float inverseDistance = rsqrt(distanceSquared);
    result.distance = distanceSquared * inverseDistance;
    result.direction = mul(float3(xu, yv, z0), axes) * inverseDistance;
    return result;
}

} // namespace Metallic.Lighting

#language slang 2026
implementing Lighting;

import ShaderCore;
using Metallic;

namespace Metallic.Lighting
{
public struct GPUPunctualLight
{
    public float4 positionRange;
    public float4 directionType;
    public float4 colorIntensity;
    public float4 spot;
    public uint iesProfile;
    public float iesRollCos;
    public float iesRollSin;
    public float iesPadding;
};


public uint punctualLightCount(StructuredBuffer<GPUPunctualLight> lights) { return uint(lights[0].positionRange.x); }
public float lightingExposure(StructuredBuffer<GPUPunctualLight> lights) { return lights[0].positionRange.y; }

public struct PunctualLightSample
{
    public float3 direction;
    public float3 illuminance;
    public float distance;
};

// Environment-owned, fixed-slot wire contract. Angular radius and disk radiance
// are retained for finite-disk sampling; Checkpoint A evaluates delta direct light.
public struct GPUCelestialLight
{
    public float3 direction;
    public float angularRadius;
    public float3 irradiance;
    public uint flags;
    public float3 diskRadiance;
    public float shadowImportance;
};

public static const uint kCelestialLightCount = 2u;
public static const uint kCelestialLightEnabled = 1u;
public static const uint kCelestialLightCastsShadow = 2u;
public static const uint kCelestialSourceBit = 0x80000000u;

public uint celestialSourceIdentity(uint slot) { return kCelestialSourceBit | slot; }

public PunctualLightSample sampleCelestialLightData(GPUCelestialLight light)
{
    PunctualLightSample sample;
    sample.direction = -light.direction;
    sample.distance = 1e30;
    sample.illuminance = (light.flags & kCelestialLightEnabled) != 0u ? light.irradiance : float3(0.0);
    return sample;
}

public PunctualLightSample sampleCelestialLight(StructuredBuffer<GPUCelestialLight> lights, uint slot)
{
    uint count, stride;
    lights.GetDimensions(count, stride);
    // Unbound fallback descriptors and short test fixtures must stay dark.
    return slot < min(count, kCelestialLightCount)
        ? sampleCelestialLightData(lights[slot]) : PunctualLightSample();
}

public bool celestialLightCastsShadow(StructuredBuffer<GPUCelestialLight> lights, uint slot)
{
    uint count, stride;
    lights.GetDimensions(count, stride);
    return slot < min(count, kCelestialLightCount) &&
        (lights[slot].flags & (kCelestialLightEnabled | kCelestialLightCastsShadow)) ==
        (kCelestialLightEnabled | kCelestialLightCastsShadow);
}

internal float iesAngle(ByteAddressBuffer profile, uint anglesOffset, uint index)
{
    return asfloat(profile.Load(anglesOffset + index * 4u));
}

internal uint bracketIESAngle(ByteAddressBuffer profile, uint anglesOffset, uint angleCount,
    float angle, out float blend)
{
    uint lower = 0u;
    uint upper = angleCount - 1u;
    while (upper - lower > 1u) {
        uint middle = (lower + upper) / 2u;
        if (iesAngle(profile, anglesOffset, middle) <= angle) { lower = middle; }
        else { upper = middle; }
    }
    float a = iesAngle(profile, anglesOffset, lower);
    float b = iesAngle(profile, anglesOffset, upper);
    blend = (angle - a) / (b - a);
    return lower;
}

internal float sampleIESPlane(ByteAddressBuffer profile, uint rowOffset, uint verticalIndex, float verticalBlend)
{
    float2 intensities = asfloat(profile.Load2(rowOffset + verticalIndex * 4u));
    return lerp(intensities.x, intensities.y, verticalBlend);
}

public float applyPunctualIESProfile(GPUPunctualLight light, float3 lightToReceiver, float attenuation)
{
    if (light.iesProfile == 0xffffffffu || attenuation <= 0.0f) { return attenuation; }
    let profile = resolveNonUniform(ResourceHandle<ByteAddressBuffer>(light.iesProfile));
    uint2 counts = profile.Load2(0u);
    // The upload packs angle counts, vertical/horizontal angles in degrees,
    // then normalized candela rows with vertical angle varying fastest.
    const uint verticalOffset = 8u;
    uint horizontalOffset = verticalOffset + counts.x * 4u;
    uint candelaOffset = horizontalOffset + counts.y * 4u;
    float3 forward = light.directionType.xyz;
    const float radiansToDegrees = 57.29577951308232f;
    float transverse = length(cross(forward, lightToReceiver));
    float axial = dot(forward, lightToReceiver);
    float vertical = transverse > 0.0f ? atan2(transverse, axial) * radiansToDegrees
        : (axial < 0.0f ? 180.0f : 0.0f);
    if (vertical < iesAngle(profile, verticalOffset, 0u) ||
        vertical > iesAngle(profile, verticalOffset, counts.x - 1u)) { return 0.0f; }
    float verticalBlend;
    uint verticalIndex = bracketIESAngle(profile, verticalOffset, counts.x, vertical, verticalBlend);
    if (counts.y == 1u) {
        return attenuation * sampleIESPlane(profile, candelaOffset, verticalIndex, verticalBlend);
    }

    float3 referenceUp = abs(forward.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(0.0f, 0.0f, -1.0f);
    float3 right = normalize(cross(forward, referenceUp));
    float3 up = cross(right, forward);
    float x = dot(right, lightToReceiver);
    float y = dot(up, lightToReceiver);
    float horizontal = x == 0.0f && y == 0.0f ? 0.0f : atan2(
        y * light.iesRollCos + x * light.iesRollSin,
        x * light.iesRollCos - y * light.iesRollSin) * radiansToDegrees;
    if (horizontal < 0.0f) { horizontal += 360.0f; }
    float firstHorizontal = iesAngle(profile, horizontalOffset, 0u);
    float lastHorizontal = iesAngle(profile, horizontalOffset, counts.y - 1u);
    if (firstHorizontal == 90.0f) {
        if (horizontal < 90.0f) { horizontal = 180.0f - horizontal; }
        else if (horizontal > 270.0f) { horizontal = 540.0f - horizontal; }
    } else if (lastHorizontal <= 180.0f) {
        horizontal = min(horizontal, 360.0f - horizontal);
        if (lastHorizontal == 90.0f) { horizontal = min(horizontal, 180.0f - horizontal); }
    }
    float horizontalBlend;
    uint horizontalIndex = bracketIESAngle(profile, horizontalOffset, counts.y, horizontal, horizontalBlend);
    uint rowStride = counts.x * 4u;
    uint rowOffset = candelaOffset + horizontalIndex * rowStride;
    float a = sampleIESPlane(profile, rowOffset, verticalIndex, verticalBlend);
    float b = sampleIESPlane(profile, rowOffset + rowStride, verticalIndex, verticalBlend);
    return attenuation * lerp(a, b, horizontalBlend);
}

// Delta lights supply incident illuminance before the surface cosine/BSDF.
// Point/spot intensity is cd, distance is m: E = I / r^2 (lux).
public PunctualLightSample samplePunctualLightData(GPUPunctualLight light, float3 position)
{
    PunctualLightSample sample;
    sample.direction = float3(0.0);
    sample.distance = 0.0;
    float attenuation = 1.0;
    // Directional records are not part of the local light domain.
    if (light.directionType.w < 0.5) { return PunctualLightSample(); }
    if (light.directionType.w > 0.5) {
        float3 delta = light.positionRange.xyz - position;
        float distanceSquared = dot(delta, delta);
        sample.distance = sqrt(max(distanceSquared, 1e-12));
        sample.direction = delta / sample.distance;
        // Finite at the singularity only; no artistic attenuation exponent.
        attenuation = 1.0 / max(distanceSquared, 1e-6);
        if (light.positionRange.w > 0.0) {
            float ratio = sample.distance / light.positionRange.w;
            attenuation *= saturate(1.0 - ratio * ratio * ratio * ratio);
        }
        if (light.directionType.w > 1.5) {
            float cosine = dot(-sample.direction, light.directionType.xyz);
            float cone = saturate((cosine - light.spot.y) / max(light.spot.x - light.spot.y, 1e-7));
            attenuation *= cone * cone;
        }
        attenuation = applyPunctualIESProfile(light, -sample.direction, attenuation);
    }
    sample.illuminance = light.colorIntensity.rgb * (light.colorIntensity.w * attenuation);
    return sample;
}

public PunctualLightSample samplePunctualLight(StructuredBuffer<GPUPunctualLight> lights, uint index, float3 position)
{
    return samplePunctualLightData(lights[index + 1u], position);
}

// Isotropic GGX + energy-reduced Lambert for the standard material path.
public float3 punctualStandardBsdf(float3 normal, float3 view, float3 light,
    float3 baseColor, float3 f0, float metallic, float roughness, float transmission)
{
    const float pi = 3.14159265358979323846;
    float nl = saturate(dot(normal, light));
    float nv = saturate(dot(normal, view));
    if (nl <= 0.0 || nv <= 0.0) { return float3(0.0); }
    float3 h = normalize(view + light);
    float nh = saturate(dot(normal, h));
    float vh = saturate(dot(view, h));
    float a = max(roughness * roughness, 0.0004);
    float a2 = a * a;
    float denominator = nh * nh * (a2 - 1.0) + 1.0;
    float distribution = a2 / max(pi * denominator * denominator, 1e-12);
    float visibility = 0.5 / max(nl * sqrt(nv * nv * (1.0 - a2) + a2) +
        nv * sqrt(nl * nl * (1.0 - a2) + a2), 1e-8);
    float3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    float3 diffuse = (1.0 - fresnel) * baseColor * (1.0 - metallic) * (1.0 - transmission) / pi;
    return (diffuse + fresnel * distribution * visibility) * nl;
}
} // namespace Metallic.Lighting

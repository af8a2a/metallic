#language slang 2026
module Atmosphere;

__exported import AtmosphereTypes;
import ShaderCore;
import WorkingColor;
import Clouds;

namespace Metallic.Atmosphere
{
public static const float kPi = 3.14159265358979323846;
public static const uint kAerialSize = 32u;
public float aerialDistanceFromSlice(GPUAtmosphereParameters p, float fraction)
{
    // Logarithmic depth provides metre-scale nearby samples while covering
    // hundreds of kilometres without allocating a view-dependent frustum.
    return 0.01 * (exp(log(1.0 + p.settings.x / 0.01) * fraction) - 1.0);
}

// Original implementation of the transport integral and isotropic geometric
// closure described by Hillaire, EGSR 2020. No reference implementation code is
// copied. https://sebh.github.io/publications/egsr2020.pdf
public struct AtmosphereSegment
{
    public float3 radiance;
    // Working-space attenuation (equal-energy spectral weighting per channel).
    public float3 transmittance;
};
public struct SpectralSegment
{
    public float3 radiance;
    public float3 transmittance;
    public float3 scatteringResponse;
};
public struct Medium
{
    public float3 rayleigh;
    public float3 mie;
    public float3 extinction;
};

public GPUAtmosphereParameters loadParameters(RWBufferSpan<float4> data)
{
    GPUAtmosphereParameters p;
    p.observerPlanetBottom = data[0]; p.observerWorldTop = data[1];
    p.rayleighScaleHeight = data[2]; p.mieScatteringScaleHeight = data[3];
    p.mieExtinctionAnisotropy = data[4]; p.ozoneCenter = data[5];
    p.groundOzoneWidth = data[6]; p.sunDirectionRadius = data[7];
    p.sunIrradianceEnabled = data[8]; p.moonDirectionRadius = data[9];
    p.moonIrradianceEnabled = data[10]; p.settings = data[11];
    p.cloudLayer = data[12]; p.cloudOptics = data[13]; p.cloudAdvection = data[14];
    p.cloudShadowCentre = data[15]; p.cloudShadowBudget = data[16]; p.weatherComposition = data[17];
    p.moonToSun = data[18]; p.moonPhase = data[19];
    return p;
}
public float3 worldToAtmosphereKm(GPUAtmosphereParameters p, float3 worldPosition)
{
    return p.observerPlanetBottom.xyz + (worldPosition - p.observerWorldTop.xyz) * 0.001;
}
public float3 spectralToWorking(float3 spectrum)
{
    // Piecewise-linear reconstruction between 440/550/680 nm, constant tails
    // to 360/830 nm. 1 nm trapezoidal quadrature of Wyman/Sloan/Shirley 2013
    // analytic CIE 1931 CMFs (equation 4). This is a coarse spectral model,
    // not an RGB-to-RGB transform. Factor 683 converts to photometric units.
    float3 xyz = float3(
        dot(spectrum, float3(32.2813132251, 57.6281726163, 16.8049788713)),
        dot(spectrum, float3(18.3286150698, 76.9932864076, 11.6242660516)),
        dot(spectrum, float3(0.00697408432, 17.1338590149, 89.7146153395))) * 683.0;
    return WorkingColor::fromXYZ(xyz);
}
public float3 spectralTransmittanceToWorking(float3 transmittance)
{
    // Channel-wise attenuation is an approximation for arbitrary surface
    // spectra. Equal-energy normalization gives exactly T=1 for vacuum.
    return saturate(spectralToWorking(transmittance) / max(spectralToWorking(float3(1.0)), 1e-5));
}
public float2 sphereRoots(float3 position, float3 direction, float radius)
{
    float b = dot(position, direction);
    float r = length(position);
    float c = (r - radius) * (r + radius);
    float discriminant = b * b - c;
    if (discriminant < 0.0) { return float2(-1.0); }
    float root = sqrt(max(discriminant, 0.0));
    float nearRoot = -b - root;
    float farRoot = -b + root;
    // Avoid cancellation near a sphere boundary.
    if (b < 0.0 && abs(farRoot) > 1e-8) { nearRoot = c / farRoot; }
    if (b > 0.0 && abs(nearRoot) > 1e-8) { farRoot = c / nearRoot; }
    return float2(nearRoot, farRoot);
}
public float groundDistance(GPUAtmosphereParameters p, float3 position, float3 direction)
{
    float r = length(position);
    if (r < p.observerPlanetBottom.w - 0.001) { return 0.0; }
    if (r <= p.observerPlanetBottom.w + 0.001 && dot(position, direction) < 0.0) { return 0.0; }
    float2 roots = sphereRoots(position, direction, p.observerPlanetBottom.w);
    return roots.x > 0.0001 ? roots.x : -1.0;
}
public bool atmosphereInterval(GPUAtmosphereParameters p, float3 position, float3 direction,
    float maximumDistance, out float beginDistance, out float endDistance, out bool hitsGround)
{
    float2 roots = sphereRoots(position, direction, p.observerWorldTop.w);
    beginDistance = max(roots.x, 0.0);
    endDistance = min(roots.y, maximumDistance);
    float ground = groundDistance(p, position, direction);
    hitsGround = ground >= 0.0 && ground <= endDistance;
    if (hitsGround) { endDistance = min(endDistance, ground); }
    return roots.y >= 0.0 && endDistance > beginDistance;
}
public Medium sampleMedium(GPUAtmosphereParameters p, float3 position)
{
    float height = max(length(position) - p.observerPlanetBottom.w, 0.0);
    float rayleighDensity = exp(-height / max(p.rayleighScaleHeight.w, 0.001));
    float mieDensity = exp(-height / max(p.mieScatteringScaleHeight.w, 0.001));
    float ozoneDensity = saturate(1.0 - abs(height - p.ozoneCenter.w) / max(p.groundOzoneWidth.w, 0.001));
    Medium m;
    m.rayleigh = p.rayleighScaleHeight.xyz * rayleighDensity;
    m.mie = p.mieScatteringScaleHeight.xyz * mieDensity;
    m.extinction = m.rayleigh + p.mieExtinctionAnisotropy.xyz * mieDensity + p.ozoneCenter.xyz * ozoneDensity;
    return m;
}
public float3 integrateTransmittance(GPUAtmosphereParameters p, float3 position, float3 direction)
{
    float begin, end; bool ground;
    if (groundDistance(p, position, direction) >= 0.0) { return float3(0.0); }
    if (!atmosphereInterval(p, position, direction, 1e9, begin, end, ground)) { return float3(1.0); }
    float3 opticalDepth = float3(0.0);
    for (uint step = 0; step < 64; ++step) {
        float u0 = float(step) / 64.0, u1 = float(step + 1u) / 64.0;
        float t0 = lerp(begin, end, u0 * u0), t1 = lerp(begin, end, u1 * u1);
        opticalDepth += sampleMedium(p, position + direction * (0.5 * (t0 + t1))).extinction * (t1 - t0);
    }
    return exp(-min(opticalDepth, float3(80.0)));
}
public float4 sampleLinear(Texture2D<float4> image, float2 uv, bool wrapLongitude = false)
{
    uint width, height; image.GetDimensions(width, height);
    float2 pixel = float2(uv.x, saturate(uv.y)) * float2(width, height) - 0.5;
    int2 base = int2(floor(pixel)); float2 f = frac(pixel);
    float4 value = float4(0.0);
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int x = base.x + dx;
            x = wrapLongitude ? (x % int(width) + int(width)) % int(width) : clamp(x, 0, int(width) - 1);
            int y = clamp(base.y + dy, 0, int(height) - 1);
            float weight = (dx == 0 ? 1.0 - f.x : f.x) * (dy == 0 ? 1.0 - f.y : f.y);
            value += image.Load(int3(x, y, 0)) * weight;
        }
    }
    return value;
}
public float3 transmittanceToSource(GPUAtmosphereParameters p, Texture2D<float4> lut,
    float3 position, float3 direction)
{
    float height = length(position);
    if (groundDistance(p, position, direction) >= 0.0) { return float3(0.0); }
    if (height >= p.observerWorldTop.w) { return integrateTransmittance(p, position, direction); }
    float2 uv = float2(dot(position / max(height, 1e-6), direction) * 0.5 + 0.5,
        saturate((height - p.observerPlanetBottom.w) / (p.observerWorldTop.w - p.observerPlanetBottom.w)));
    uint width, heightCount; lut.GetDimensions(width, heightCount);
    uv = (uv * float2(width - 1u, heightCount - 1u) + 0.5) / float2(width, heightCount);
    return saturate(sampleLinear(lut, uv).xyz);
}
public float rayleighPhase(float cosine)
{
    return 3.0 * (1.0 + cosine * cosine) / (16.0 * kPi);
}
public float miePhase(float cosine, float g)
{
    float square = g * g;
    return (3.0 / (8.0 * kPi)) * (1.0 - square) * (1.0 + cosine * cosine) /
        ((2.0 + square) * pow(max(1.0 + square - 2.0 * g * cosine, 1e-3), 1.5));
}
public float3 scatteringIntegral(float3 extinction, float distance)
{
    float3 optical = extinction * distance;
    float3 integral = (float3(1.0) - exp(-min(optical, float3(80.0)))) / max(extinction, float3(1e-30));
    float3 thin = distance * (float3(1.0) - optical * 0.5 + optical * optical / 6.0);
    return float3(optical.x < 1e-3 ? thin.x : integral.x,
        optical.y < 1e-3 ? thin.y : integral.y, optical.z < 1e-3 ? thin.z : integral.z);
}
public float3 sampleMulti(GPUAtmosphereParameters p, Texture2D<float4> lut, float3 position, float3 direction)
{
    float radius = length(position);
    float2 uv = float2(dot(position / max(radius, 1e-6), direction) * 0.5 + 0.5,
        saturate((radius - p.observerPlanetBottom.w) / (p.observerWorldTop.w - p.observerPlanetBottom.w)));
    uint width, height; lut.GetDimensions(width, height);
    uv = (uv * float2(width - 1u, height - 1u) + 0.5) / float2(width, height);
    return max(sampleLinear(lut, uv).xyz, 0.0);
}
public SpectralSegment integrateSpectral(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    Texture2D<float4> multiScattering, float3 position, float3 direction, float distance,
    float3 sourceDirection, float3 sourceIrradiance, bool isotropic, bool useMultiple, bool groundBounce)
{
    SpectralSegment result;
    result.radiance = float3(0.0); result.transmittance = float3(1.0); result.scatteringResponse = float3(0.0);
    float begin, end; bool ground;
    if (!atmosphereInterval(p, position, direction, distance, begin, end, ground)) { return result; }
    float rayPhase = isotropic ? 1.0 / (4.0 * kPi) : rayleighPhase(dot(direction, sourceDirection));
    float aerosolPhase = isotropic ? 1.0 / (4.0 * kPi) : miePhase(dot(direction, sourceDirection), p.mieExtinctionAnisotropy.w);
    // The static isotropic LUT deliberately excludes dynamic cloud fields.
    bool cloud = !isotropic && useMultiple && Metallic.Clouds::enabled(p);
    float4 cloudRanges = cloud ? Metallic.Clouds::intervals(p, position, direction, end) : float4(-1.0);
    float boundaries[6] = {begin, end, end, end, end, end};
    uint regions = 1u;
    if (cloudRanges.x >= begin && cloudRanges.y > cloudRanges.x) {
        boundaries[1] = cloudRanges.x; boundaries[2] = cloudRanges.y;
        regions = 3u;
        if (cloudRanges.z >= cloudRanges.y && cloudRanges.w > cloudRanges.z) {
            boundaries[3] = cloudRanges.z; boundaries[4] = cloudRanges.w; boundaries[5] = end;
            regions = 5u;
        } else { boundaries[3] = end; }
    }
    for (uint region = 0u; region < regions; ++region) {
        float regionBegin = boundaries[region], regionEnd = boundaries[region + 1u];
        if (regionEnd <= regionBegin) { continue; }
        bool cloudRegion = regions > 1u && (region & 1u) != 0u;
        uint steps = cloudRegion ? 24u : 32u;
        for (uint step = 0; step < steps; ++step) {
            float u0 = float(step) / float(steps), u1 = float(step + 1u) / float(steps);
            float t0 = lerp(regionBegin, regionEnd, cloudRegion ? u0 : u0 * u0);
            float t1 = lerp(regionBegin, regionEnd, cloudRegion ? u1 : u1 * u1);
            float3 point = position + direction * (0.5 * (t0 + t1));
            Medium medium = sampleMedium(p, point);
            float cloudExtinction = cloud ? Metallic.Clouds::density(p, point) * p.cloudOptics.x : 0.0;
            float cloudScattering = cloudExtinction * p.cloudOptics.y;
            medium.extinction += cloudExtinction;
            float3 integral = scatteringIntegral(medium.extinction, t1 - t0);
            float3 scattered = medium.rayleigh + medium.mie;
            float cloudLightT = cloud && any(sourceIrradiance > 0.0) ?
                Metallic.Clouds::transmittance(p, point, sourceDirection, 1e9, 8u) : 1.0;
            float3 source = transmittanceToSource(p, transmittance, point, sourceDirection) *
                cloudLightT * (medium.rayleigh * rayPhase + medium.mie * aerosolPhase);
            if (cloudScattering > 0.0) {
                source += transmittanceToSource(p, transmittance, point, sourceDirection) * cloudScattering *
                    Metallic.Clouds::scatteringPhase(dot(direction, sourceDirection), p.cloudOptics.z, p.cloudOptics.y, cloudLightT);
            }
            if (useMultiple) { source += sampleMulti(p, multiScattering, point, sourceDirection) * scattered * cloudLightT; }
            result.radiance += result.transmittance * integral * source * sourceIrradiance;
            result.scatteringResponse += result.transmittance * integral * (scattered + cloudScattering);
            result.transmittance *= exp(-min(medium.extinction * (t1 - t0), float3(80.0)));
        }
    }
    if (ground && groundBounce) {
        float3 point = position + direction * end;
        float3 normal = normalize(point);
        float3 incident = transmittanceToSource(p, transmittance, point + normal * 0.002, sourceDirection) *
            max(dot(normal, sourceDirection), 0.0);
        if (cloud) { incident *= Metallic.Clouds::transmittance(p, point + normal * 0.002, sourceDirection); }
        result.radiance += result.transmittance * incident * p.groundOzoneWidth.xyz * sourceIrradiance / kPi;
        result.scatteringResponse += result.transmittance * p.groundOzoneWidth.xyz;
    }
    return result;
}
public SpectralSegment integrateEnvironment(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    Texture2D<float4> multiScattering, float3 position, float3 direction, float distance, bool groundBounce)
{
    SpectralSegment sun = integrateSpectral(p, transmittance, multiScattering, position, direction, distance,
        p.sunDirectionRadius.xyz, p.sunIrradianceEnabled.xyz * p.sunIrradianceEnabled.w, false, true, groundBounce);
    if (p.moonIrradianceEnabled.w <= 0.0 || all(p.moonIrradianceEnabled.xyz <= 0.0)) { return sun; }
    SpectralSegment moon = integrateSpectral(p, transmittance, multiScattering, position, direction, distance,
        p.moonDirectionRadius.xyz, p.moonIrradianceEnabled.xyz * p.moonIrradianceEnabled.w, false, true, groundBounce);
    sun.radiance += moon.radiance;
    return sun;
}
public AtmosphereSegment segmentIntegration(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    Texture2D<float4> multiScattering, float3 positionKm, float3 direction, float distanceKm)
{
    SpectralSegment integrated = integrateEnvironment(p, transmittance, multiScattering,
        positionKm, direction, max(distanceKm, 0.0), false);
    AtmosphereSegment result;
    result.radiance = spectralToWorking(integrated.radiance);
    result.transmittance = spectralTransmittanceToWorking(integrated.transmittance);
    return result;
}
public float3 solarIrradiance(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    float3 positionKm, uint slot, float cloudTransmittance)
{
    float4 source = slot == 0 ? p.sunIrradianceEnabled : p.moonIrradianceEnabled;
    float3 direction = slot == 0 ? p.sunDirectionRadius.xyz : p.moonDirectionRadius.xyz;
    return spectralToWorking(source.xyz * source.w * transmittanceToSource(p, transmittance, positionKm, direction)) *
        cloudTransmittance;
}
public float3 solarIrradiance(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    float3 positionKm, uint slot)
{
    float3 direction = slot == 0 ? p.sunDirectionRadius.xyz : p.moonDirectionRadius.xyz;
    return solarIrradiance(p, transmittance, positionKm, slot, Metallic.Clouds::transmittance(p, positionKm, direction));
}
public float cloudShadowTransmittance(GPUAtmosphereParameters p, Texture2D<float4> shadow,
    float3 positionKm, uint slot)
{
    if (!Metallic.Clouds::enabled(p)) { return 1.0; }
    float3 up, east, north; radialBasis(p, up, east, north);
    float3 sourceDirection = slot == 0u ? p.sunDirectionRadius.xyz : p.moonDirectionRadius.xyz;
    float radius = length(positionKm), altitude = radius - p.observerPlanetBottom.w;
    // Continue the receiver's source ray backwards onto the spherical ground.
    // Reusing its cloud column preserves shadow parallax on elevated terrain.
    float2 groundRoots = sphereRoots(positionKm, -sourceDirection, p.observerPlanetBottom.w + 0.002);
    float3 footprint = positionKm - sourceDirection * max(groundRoots.x, 0.0);
    float projectedHeight = dot(footprint, up);
    float2 tangentOffset = float2(dot(footprint, east), dot(footprint, north)) *
        (p.observerPlanetBottom.w + 0.002) / max(projectedHeight, 1e-6);
    float2 uv = tangentOffset / (2.0 * p.cloudShadowCentre.w) + 0.5;
    // The projected ground map is invalid for cloud-interior/aerial receivers.
    if (all(uv >= 0.0) && all(uv <= 1.0) && projectedHeight > 0.0 && groundRoots.y >= 0.0 &&
        altitude >= 0.0 && altitude < min(p.cloudLayer.x, 0.5)) {
        float4 sampled = sampleLinear(shadow, uv);
        if ((slot == 0u ? sampled.z : sampled.w) > 0.99) {
            return saturate(slot == 0u ? sampled.x : sampled.y);
        }
    }
    return Metallic.Clouds::transmittance(p, positionKm, sourceDirection);
}
public float3 latLongDirection(float2 uv)
{
    float theta = uv.y * kPi, phi = (uv.x - 0.5) * (2.0 * kPi);
    return float3(cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta));
}
public float2 directionLatLong(float3 direction)
{
    return float2(atan2(direction.z, direction.x) / (2.0 * kPi) + 0.5, acos(clamp(direction.y, -1.0, 1.0)) / kPi);
}
public float planetLimbZenith(GPUAtmosphereParameters p)
{
    float ratio = saturate(p.observerPlanetBottom.w / max(length(p.observerPlanetBottom.xyz), 1e-6));
    return acos(-sqrt(max(1.0 - ratio * ratio, 0.0)));
}
public void radialBasis(GPUAtmosphereParameters p, out float3 up, out float3 east, out float3 north)
{
    up = normalize(p.observerPlanetBottom.xyz);
    east = abs(up.y) < 0.999 ? normalize(cross(float3(0.0, 1.0, 0.0), up)) : float3(1.0, 0.0, 0.0);
    north = cross(up, east);
}
public float3 directionForZenithAzimuth(GPUAtmosphereParameters p, float zenith, float azimuth)
{
    float3 up, east, north; radialBasis(p, up, east, north);
    return cos(zenith) * up + sin(zenith) * (cos(azimuth) * east + sin(azimuth) * north);
}
public float quadraticEndpointDistribution(float u)
{
    float square = u * u, opposite = (1.0 - u) * (1.0 - u);
    return square / max(square + opposite, 1e-8);
}
public float inverseQuadraticEndpointDistribution(float fraction)
{
    float left = sqrt(saturate(fraction)), right = sqrt(saturate(1.0 - fraction));
    return left / max(left + right, 1e-8);
}
public float3 skyDirectionFromUV(GPUAtmosphereParameters p, float2 uv)
{
    // Two limb-centred angular intervals, each concentrated at both endpoints.
    // This retains fine zenith sampling as well as the ground/orbital limb.
    float horizon = planetLimbZenith(p);
    float zenith = uv.y < 0.5 ? horizon * quadraticEndpointDistribution(uv.y * 2.0) :
        horizon + (kPi - horizon) * quadraticEndpointDistribution((uv.y - 0.5) * 2.0);
    return directionForZenithAzimuth(p, zenith, (uv.x - 0.5) * 2.0 * kPi);
}
public float2 skyDirectionUV(GPUAtmosphereParameters p, float3 direction)
{
    float3 up, east, north; radialBasis(p, up, east, north);
    float zenith = acos(clamp(dot(direction, up), -1.0, 1.0));
    float horizon = planetLimbZenith(p);
    float v = zenith < horizon ? 0.5 * inverseQuadraticEndpointDistribution(zenith / max(horizon, 1e-6)) :
        0.5 + 0.5 * inverseQuadraticEndpointDistribution((zenith - horizon) / max(kPi - horizon, 1e-6));
    return float2(atan2(dot(direction, north), dot(direction, east)) / (2.0 * kPi) + 0.5, v);
}
public float3 sampleSkyView(GPUAtmosphereParameters p, Texture2D<float4> skyView, float3 direction)
{
    return sampleLinear(skyView, skyDirectionUV(p, direction), true).xyz;
}
public float3 sourceDiskIrradianceSample(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    uint slot, float3 positionKm, float3 worldDirection, bool assumeInDisk = false)
{
    float4 direction = slot == 0 ? p.sunDirectionRadius : p.moonDirectionRadius;
    direction.xyz = normalize(direction.xyz);
    worldDirection = normalize(worldDirection);
    float3 chord = worldDirection - direction.xyz;
    float sineHalf = sin(0.5 * direction.w);
    if (direction.w > 0.0 && (assumeInDisk || dot(chord, chord) <= 4.0 * sineHalf * sineHalf)) {
        // The finite disk crosses the planet limb per visible ray from
        // the actual primary origin, including orthographic ray offsets.
        float4 source = slot == 0 ? p.sunIrradianceEnabled : p.moonIrradianceEnabled;
        float3 disk = spectralToWorking(source.xyz * source.w *
            transmittanceToSource(p, transmittance, positionKm, worldDirection)) *
            Metallic.Clouds::transmittance(p, positionKm, worldDirection);
        if (slot == 1u && p.moonToSun.w > 0.5) {
            // The CPU TOA already includes integrated Lambert phase.
            // Normalize the resolved spherical disk by its projected-area
            // average, 2/3 times the Lambert phase function, preserving
            // that flux while exposing the terminator and limb shading.
            float3 view = -direction.xyz;
            float3 reference = abs(view.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
            float3 right = normalize(cross(reference, view)), top = cross(view, right);
            float2 diskCoordinate = float2(dot(worldDirection, right), dot(worldDirection, top)) / max(sin(direction.w), 1e-30);
            float3 normal = right * diskCoordinate.x + top * diskCoordinate.y +
                view * sqrt(max(1.0 - dot(diskCoordinate, diskCoordinate), 0.0));
            float average = (2.0 / 3.0) * p.moonPhase.z;
            disk *= average > 1e-30 ? max(dot(normal, p.moonToSun.xyz), 0.0) / average : 0.0;
        }
        return disk;
    }
    return float3(0.0);
}
public float3 sourceDiskRadiance(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    uint slot, float3 positionKm, float3 worldDirection)
{
    float radius = slot == 0u ? p.sunDirectionRadius.w : p.moonDirectionRadius.w;
    return sourceDiskIrradianceSample(p, transmittance, slot, positionKm, worldDirection) /
        max(kPi * sin(radius) * sin(radius), 1e-30);
}
public float3 sourceDiskSolidAngleEstimate(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    uint slot, float3 positionKm, float3 worldDirection)
{
    float radius = slot == 0u ? p.sunDirectionRadius.w : p.moonDirectionRadius.w;
    // L / pdf = L * solid angle. Cancel the small squared radius before
    // division so every positive finite angular radius preserves source flux.
    return sourceDiskIrradianceSample(p, transmittance, slot, positionKm, worldDirection, true) *
        (2.0 / max(1.0 + cos(radius), 1e-8));
}
public float3 visibleSourceDisks(GPUAtmosphereParameters p, Texture2D<float4> transmittance,
    float3 positionKm, float3 worldDirection)
{
    return sourceDiskRadiance(p, transmittance, 0u, positionKm, worldDirection) +
        sourceDiskRadiance(p, transmittance, 1u, positionKm, worldDirection);
}
public float3 visibleSky(GPUAtmosphereParameters p, Texture2D<float4> skyView,
    Texture2D<float4> transmittance, float3 worldDirection, bool includeDisks)
{
    float3 radiance = sampleSkyView(p, skyView, worldDirection);
    if (includeDisks) {
        radiance += visibleSourceDisks(p, transmittance, p.observerPlanetBottom.xyz, worldDirection);
    }
    return radiance;
}
public AtmosphereSegment aerialPerspective(GPUAtmosphereParameters p, BufferSpan<float4> aerial, float3 worldEndpoint)
{
    float3 delta = (worldEndpoint - p.observerWorldTop.xyz) * 0.001;
    float distance = length(delta);
    AtmosphereSegment result;
    result.radiance = float3(0.0); result.transmittance = float3(1.0);
    if (distance < 1e-7) { return result; }
    float2 uv = skyDirectionUV(p, delta / distance);
    float3 coordinates = float3(uv.x * 32.0 - 0.5, uv.y * 32.0 - 0.5,
        saturate(log(1.0 + distance / 0.01) / log(1.0 + p.settings.x / 0.01)) * 31.0);
    int3 base = int3(floor(coordinates)); float3 f = frac(coordinates);
    // Interpolate in travelled distance rather than the logarithmic index.
    // In particular, nearby haze must approach zero linearly.
    float lowerFraction = float(clamp(base.z, 0, 31)) / 31.0;
    float upperFraction = float(clamp(base.z + 1, 0, 31)) / 31.0;
    float lowerDistance = aerialDistanceFromSlice(p, lowerFraction);
    float upperDistance = aerialDistanceFromSlice(p, upperFraction);
    f.z = upperDistance > lowerDistance ? saturate((distance - lowerDistance) / (upperDistance - lowerDistance)) : 0.0;
    result.transmittance = float3(0.0);
    for (int z = 0; z < 2; ++z) {
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 2; ++x) {
                int ix = ((base.x + x) % 32 + 32) % 32;
                int iy = clamp(base.y + y, 0, 31), iz = clamp(base.z + z, 0, 31);
                uint index = uint((iz * 32 + iy) * 32 + ix) * 2u;
                float weight = (x == 0 ? 1.0 - f.x : f.x) * (y == 0 ? 1.0 - f.y : f.y) * (z == 0 ? 1.0 - f.z : f.z);
                result.radiance += aerial[index].xyz * weight;
                result.transmittance += aerial[index + 1u].xyz * weight;
            }
        }
    }
    return result;
}
} // namespace Metallic.Atmosphere

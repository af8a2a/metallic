module CloudShadow;

import Atmosphere;
import CloudShadowMap;
import Clouds;
using Metallic;
using Metallic.Atmosphere;
using Metallic.Clouds;

[[vk::push_constant]] ConstantBuffer<AtmospherePrecomputeParams> gParams;

// UE VolumetricCloud.usf SHADER_SHADOW_PS, with the default zero depth bias
// and unit shadow strength. Material extinction is already in inverse metres.
float4 traceColumn(GPUAtmosphereParameters p, CloudShadowProjection projection, float2 uv, uint steps)
{
    float farKm = 2.0f * projection.halfWidthKm;
    float3 origin = projection.nearPlanePosition(uv);
    float2 top = sphereRoots(origin, projection.forward, p.observerPlanetBottom.w + p.cloudLayer.y);
    if (all(top == -1.0f)) { return float4(farKm, 0, 0, 0); }
    // A positive-length cloud interval on the light side of this plane cannot be represented.
    // Returning a clipped integral would transmit sunlight through tall or grazing clouds.
    if (top.x < 0.0f) { return float4(0, 0, 0, 1); }
    float2 bottom = sphereRoots(origin, projection.forward, p.observerPlanetBottom.w + p.cloudLayer.x);
    float2 range = top;
    if (!all(bottom == -1.0f)) {
        if (!all(bottom > 0.0f)) { return 0.0f; }
        float firstTop = max(0.0f, min(top.x, top.y));
        float firstBottom = min(bottom.x, bottom.y);
        range = float2(min(firstBottom, firstTop), max(firstBottom, firstTop));
    }
    range = max(0.0f, range);
    float3 layerOrigin = origin + projection.forward * range.x;
    float lengthKm = range.y - range.x;
    float stepMeters = lengthKm * 1000.0f / float(steps);
    float extinctionSum = 0.0f, count = 0.0f, maxOpticalDepth = 0.0f;
    float nearDepth = farKm;
    float invSteps = 1.0f / float(steps);
    for (float s = 0.5f; s < float(steps); s += 1.0f) {
        float distanceKm = lengthKm * (s * invSteps);
        float extinction = cloudExtinctionPerMeter(p, layerOrigin + projection.forward * distanceKm);
        bool present = extinction > 0.0f;
        nearDepth = present ? min(nearDepth, distanceKm) : nearDepth;
        extinctionSum += extinction;
        maxOpticalDepth += extinction * stepMeters;
        count += present ? 1.0f : 0.0f;
    }
    float frontKm = nearDepth == farKm ? range.y : range.x + nearDepth;
    return float4(max(0.0f, frontKm), extinctionSum / max(1.0f, count), maxOpticalDepth, 0.0f);
}

[numthreads(8, 8, 1)]
void cloudShadowMain(uint3 id : SV_DispatchThreadID)
{
    if (any(id.xy >= gParams.dimensions.xy)) { return; }
    GPUAtmosphereParameters p = loadParameters(gParams.parameters);
    uint tileSize = gParams.dimensions.y;
    uint slot = id.x / tileSize;
    float4 result = 0.0f;
    if (Metallic.Clouds::enabled(p) && (uint(p.cloudShadowBudget.z) & (1u << slot)) != 0u) {
        CloudShadowProjection projection = CloudShadowProjection::forLight(p, slot);
        uint steps = uint(p.cloudShadowBudget[slot]);
        uint2 pixel = uint2(id.x - slot * tileSize, id.y);
        float2 origin = float2(2u * pixel) + 0.5f;
        float invSize = 1.0f / float(2u * tileSize);
        // Fuse the four source texels of UE's spatial downsample into the publishing dispatch.
        float4 a = traceColumn(p, projection, origin * invSize, steps);
        float4 b = traceColumn(p, projection, (origin + float2(1, 0)) * invSize, steps);
        float4 c = traceColumn(p, projection, (origin + float2(1, 1)) * invSize, steps);
        float4 d = traceColumn(p, projection, (origin + float2(0, 1)) * invSize, steps);
        float mean = a.x + b.x + c.x + d.x;
        float deviation = (abs(a.x - mean) + abs(b.x - mean) + abs(c.x - mean) + abs(d.x - mean)) * 0.25f;
        result = float4(mean - deviation, (a.yz + (b.yz + c.yz + d.yz)) * 0.25f,
            max(max(a.w, b.w), max(c.w, d.w)));
    }
    resolveUniform(gParams.output)[id.xy] = result;
}

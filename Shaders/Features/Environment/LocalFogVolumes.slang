implementing VolumetricFog;

import ShaderCore;
import Core;
import ParameterRoot;
import ScenePathMath;
using Metallic;
using Metallic.ScenePath;

// UE LocalFogVolumeTiledCulling.usf, USE_HZB=0. The graph supplies the raster
// camera and depth, but no late furthest HZB. Work in translated view space.
[shader("compute")]
[numthreads(8, 8, 1)]
void localFogTiledCulling(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<VolumetricFogParams>();
    const float2 pixels = p.current.viewport.yz;
    const uint2 tileResolution = (uint2(pixels) + 127u) / 128u;
    if (any(id.xy >= tileResolution)) { return; }
    float2 minLerp = float2(id.xy) / float2(tileResolution);
    float2 maxLerp = (float2(id.xy) + 1.0f) / float2(tileResolution);
    const float2 viewToTileSpaceRatio = float2(tileResolution * 128u) / pixels;
    minLerp *= viewToTileSpaceRatio;
    maxLerp = min(maxLerp * viewToTileSpaceRatio, 1.0f);

    float4 leftPlane, rightPlane, topPlane, bottomPlane;
    if (p.current.upProjection.w > 0.5f) {
        float halfY = p.current.clipOrtho.z * 50.0f;
        float halfX = halfY * p.current.viewport.x;
        leftPlane = float4(-1, 0, 0, halfX);
        rightPlane = float4(-1, 0, 0, -halfX);
        topPlane = float4(0, 1, 0, halfY);
        bottomPlane = float4(0, 1, 0, -halfY);
    } else {
        float tanY = tan(p.current.viewport.w * 0.5f);
        float tanX = tanY * p.current.viewport.x;
        leftPlane = float4(normalize(float3(-1, 0, -tanX)), 0);
        rightPlane = float4(normalize(float3(-1, 0, tanX)), 0);
        topPlane = float4(normalize(float3(0, 1, -tanY)), 0);
        bottomPlane = float4(normalize(float3(0, 1, tanY)), 0);
    }
    float4 planes[5] = {lerp(leftPlane, rightPlane, minLerp.x), -lerp(leftPlane, rightPlane, maxLerp.x),
        lerp(topPlane, bottomPlane, minLerp.y), -lerp(topPlane, bottomPlane, maxLerp.y),
        float4(0, 0, -1, -100.0f * p.current.clipOrtho.x)};
    float3 forward = cameraForwardFrom(p.current.eye, p.current.center);
    float3 right = cameraRightFrom(forward, p.current.upProjection.xyz);
    float3 up = cameraUpFrom(forward, right);
    uint count = 0u;
    for (uint i = 0; i < p.localFog.instances.count / 3u; ++i) {
        float4 sphere = asfloat(p.localFog.instances.load(i * 3u));
        float3 center = float3(dot(right, sphere.xyz), dot(up, sphere.xyz), dot(forward, sphere.xyz));
        bool outside = false;
        [unroll]
        for (uint j = 0; j < 5; ++j) {
            if (dot(planes[j].xyz, center) - planes[j].w > sphere.w) {
                outside = true;
                break;
            }
        }
        if (!outside) {
            resolveUniform(p.localFog.tilesOut)[uint3(id.xy, 1u + count)] = i;
            ++count;
        }
    }
    resolveUniform(p.localFog.tilesOut)[uint3(id.xy, 0)] = count;
}

float3 unpackFog111110(uint rgb)
{
    return f16tof32(uint3((rgb >> 17) & 0x7ff0, (rgb >> 6) & 0x7ff0, (rgb << 5) & 0x7fe0));
}

float distanceToFogCell(float3 translated, ViewCameraConstants camera)
{
    float distance = length(translated);
    if (camera.upProjection.w > 0.5f) {
        distance *= distance / dot(translated, cameraForwardFrom(camera.eye, camera.center));
    }
    return distance;
}

// UE VolumetricFog.usf MaterialSetupCS local volume injection.
void injectLocalFog(uint3 id, VolumetricFogParams p, inout float3 scattering, inout float extinction,
    inout float3 additionalEmissive)
{
    float3 position = computeCellTranslatedWorldPosition(id, 0.5f, p);
    float backDistance = distanceToFogCell(computeCellTranslatedWorldPosition(id, float3(0.5f, 0.5f, 1.0f), p), p.current);
    if (backDistance <= p.localFog.startDistanceCm) { return; }
    float frontDistance = distanceToFogCell(computeCellTranslatedWorldPosition(id, float3(0.5f, 0.5f, 0.0f), p), p.current);
    float softDensityScale = 1.0f;
    if (frontDistance < p.localFog.startDistanceCm && p.localFog.startDistanceCm < backDistance) {
        softDensityScale = (backDistance - p.localFog.startDistanceCm) / max(0.0001f, backDistance - frontDistance);
    }
    float2 pixels = p.current.viewport.yz;
    float2 svPosition = (float2(id.xy) + 0.5f) / float2(p.grid.xy) * pixels;
    uint2 tilePos = min(uint2(svPosition / 128.0f), (uint2(pixels) + 127u) / 128u - 1u);
    let tiles = resolveUniform(p.localFog.tiles);
    uint count = tiles.Load(int4(tilePos, 0, 0));
    for (uint index = 0u; index < count; ++index) {
        uint offset = tiles.Load(int4(tilePos, 1u + index, 0)) * 3u;
        float4 sphere = asfloat(p.localFog.instances.load(offset));
        uint4 packedRotation = p.localFog.instances.load(offset + 1u);
        uint4 packed = p.localFog.instances.load(offset + 2u);
        float3 x = f16tof32(uint3(packedRotation.x & 0xffffu, packedRotation.x >> 16u, packedRotation.y & 0xffffu));
        float3 y = f16tof32(uint3(packedRotation.y >> 16u, packedRotation.z & 0xffffu, packedRotation.z >> 16u));
        float3 z = cross(x, y);
        float inverseScale = 1.0f / sphere.w;
        float3 delta = position - sphere.xyz;
        float3 unitPos = mul(float3(delta.x, -delta.z, delta.y), float3x3(x * inverseScale, y * inverseScale, z * inverseScale));
        float sphereFade = saturate(1.0f - length(unitPos));
        float3 density = unpackFog111110(packed.x);
        float height = sphereFade <= 0.0f ? 0.0f : density.y * exp(-density.z * (unitPos.z - asfloat(packed.w)));
        float radial = sphereFade <= 0.0f ? 0.0f : density.x * pow(sphereFade, 0.82f);
        float a = exp(-height);
        float b = exp(-radial);
        float localExtinction = max(0.0f, -log(a - a * b + b));
        localExtinction *= (1.0f / 100.0f);
        localExtinction = min(0.01f, localExtinction) * softDensityScale;
        float3 albedo = float3(uint3(packed.z, packed.z >> 8u, packed.z >> 16u) & 0xffu) * (1.0f / 255.0f);
        float3 emissive = packed.y > 0u ? unpackFog111110(packed.y) : 0.0f;
        scattering += localExtinction * albedo;
        extinction += localExtinction;
        additionalEmissive += localExtinction * emissive;
    }
}

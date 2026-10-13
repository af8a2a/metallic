#language slang 2026
module FiberGuideProbe;

import Core;
using Metallic;

import OpenPBRSurface;
import SceneSurface;
import ScenePathTypes;
import ScenePathMath;
import SceneSurfaceMaterial;
import Material;
import MaterialProgram;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRMaterialParameters;
import SurfaceLighting;
using Metallic.SceneOpenPBR;
using Metallic.SceneSurface;
using Metallic.ScenePath;
using Metallic.Material;
using Metallic.OpenPBR;
using Metallic.Lighting;
import SceneStandardGuides;
using Metallic.SceneStandardGuides;

// Exercise the production guide dispatch against uploaded scene records, rather
// than calling the Fiber summary helper directly or constructing its outputs.
[shader("compute")]
[numthreads(8, 1, 1)]
void fiberGuideProbeMain(uint3 id : SV_DispatchThreadID)
{
    HitInfo hit = (HitInfo)0;
    hit.materialIndex = id.x;
    hit.normal = float3(0, 1, 0);
    hit.geometryNormal = hit.normal;
    hit.tangent = float4(1, 0, 0, 1);
    hit.bitangent = float3(0, 0, 1);
    hit.position = float3(0, 2, 0);
    hit.t = 2.0;
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.center = float4(0, 1, 0, 0);
    push.width = 8u;
    push.height = 1u;
    // Surface indices in the alternating inputs are in range, while this Fiber
    // probe intentionally supplies no Surface textures or feedback resources.
    push.materialTextureCount = 1u;
    push.clipOrtho = float4(0.001, 10, 1, 1);
    Ray ray;
    ray.origin = float3(0);
    ray.direction = float3(0, 1, 0);
    uint standardSeed = 97u;
    let standard = evaluateStandardDlssRrGuidesAtHit(ray, (RayCone)0,
        float2(0.5), push, standardSeed, hit);
    uint seed = 97u;
    let summary = evaluateSceneSurfaceGuides(hit, push,
        (RayCone)0, float3(0, 1, 0), seed);
    let output = resolveBuffer<RWStructuredBuffer<float4>>(
        getResourceParameters<SceneResourceParameters>().probeOutput);
    uint offset = id.x * 10u;
    output[offset] = float4(summary.baseColor, summary.roughness);
    output[offset + 1u] = float4(summary.albedo, summary.ior);
    output[offset + 2u] = float4(summary.guideSpecularReflectance, summary.occlusion);
    output[offset + 3u] = float4(hit.normal, float(seed));
    output[offset + 4u] = float4(hit.geometryNormal, summary.unlit ? 1.0 : 0.0);
    output[offset + 5u] = float4(summary.diffuseColor, summary.conductorFraction);
    output[offset + 6u] = float4(standard.albedo, standard.roughness);
    output[offset + 7u] = float4(standard.normal, float(standardSeed));
    output[offset + 8u] = float4(standard.specularAlbedo, standard.linearDepth);
    output[offset + 9u] = float4(standard.motionVector, 0, 0);
}

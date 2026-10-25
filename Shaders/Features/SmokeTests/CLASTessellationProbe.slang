#language slang 2026
module CLASTessellationProbe;
import Core;
using Metallic;

import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import Lighting;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.SceneSurface;
using Metallic.ScenePath;
import ScenePathTracing;
using Metallic.ScenePathTracing;
#include "../../Interop/MeshletStreamBLAS.h"

struct ProbePush
{
    float4 scaleTranslationX;
    float4 translationYZ;
};
#define gProbePush getConstants<ProbePush>()
#define gProbeOutput resolveBuffer<RWStructuredBuffer<float4>>(getResourceParameters<SceneResourceParameters>().probeOutput)

[shader("compute")]
[numthreads(8, 1, 1)]
void clasTessellationProbeMain(uint3 id : SV_DispatchThreadID)
{
    const float2 coordinates[8] = {
        float2(0.25, 0.25), float2(0.75, 0.25),
        float2(0.25, 0.75), float2(0.75, 0.75),
        float2(0.5, 0.5), float2(0.125, 0.375),
        float2(0.875, 0.625), float2(1.5, 0.5)
    };
    const float3 scale = gProbePush.scaleTranslationX.xyz;
    const float3 translation = float3(gProbePush.scaleTranslationX.w, gProbePush.translationYZ.xy);
    ScenePathTracePush push = {};
    push.clipOrtho = float4(0.001, 100.0, 0.0, 1.0);
    push.bitangentFlip = 1.0;
    if (id.x >= 8u) {
        // Oblique rays provide an inspectable surface image from the same
        // production TLAS and hit decoder as the precise acceptance probes.
        const uint pixel = id.x - 8u;
        const float2 screen = (float2(pixel % 128u, pixel / 128u) + 0.5) / 128.0 * 3.0 - 1.5;
        Ray imageRay;
        imageRay.origin = float3(screen.x, screen.y - 3.0, 4.0) * scale + translation;
        imageRay.direction = normalize(float3(0.0, 3.0, -4.0) * scale);
        HitInfo imageHit;
        const bool imageFound = traceClosest(imageRay, push, imageHit);
        float3 color = float3(0.035, 0.045, 0.065);
        if (imageFound) {
            const float light = 0.15 + 0.85 * saturate(dot(imageHit.geometryNormal, normalize(float3(-0.6, -0.3, 1.0))));
            color = lerp(float3(0.08, 0.35, 0.52), float3(0.9, 0.5, 0.14), imageHit.texcoord.x) * light;
        }
        gProbeOutput[8u * 7u + pixel] = float4(color, imageFound ? imageHit.t : 0.0);
        return;
    }
    const bool back = id.x == 6u;
    Ray ray;
    ray.origin = float3(coordinates[id.x] * 2.0 - 1.0, back ? -2.0 : 2.0) * scale + translation;
    ray.direction = float3(0.0, 0.0, back ? 1.0 : -1.0);
    HitInfo hit = {};
    const bool found = traceClosest(ray, push, hit);
    if (!found) { hit = (HitInfo)0; }
    const uint base = id.x * 7u;
    gProbeOutput[base + 0u] = float4(hit.position, hit.t);
    gProbeOutput[base + 1u] = float4(hit.normal, hit.frontFace ? 1.0 : 0.0);
    gProbeOutput[base + 2u] = float4(hit.geometryNormal, found ? 1.0 : 0.0);
    gProbeOutput[base + 3u] = hit.tangent;
    gProbeOutput[base + 4u] = float4(hit.bitangent, float(hit.triangleIndex));
    gProbeOutput[base + 5u] = float4(hit.texcoord, float(hit.materialIndex), float(hit.instanceIndex));
    let changes = resolveBuffer<StructuredBuffer<uint>>(getResourceParameters<SceneResourceParameters>().rayStreamChanges);
    gProbeOutput[base + 6u] = float4(float(changes[METALLIC_STREAM_BLAS_SEMANTIC_CHANGE_WORD]),
        float(changes[METALLIC_STREAM_BLAS_TESS_ADDRESS_WORD + 2u]),
        float(changes[METALLIC_STREAM_BLAS_TESS_ADDRESS_WORD + 3u]), 0.0);
}

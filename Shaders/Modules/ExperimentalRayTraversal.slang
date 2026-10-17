#language slang 2026
module ExperimentalRayTraversal;

import Core;
import Material;
import ScenePathTypes;
import SceneSurface;
import StreamRayGeometry;
import StreamRayHit;
#include "../Interop/MeshletStreamBLAS.h"

namespace Metallic.ExperimentalPath
{
using Metallic.Material;
using Metallic.SceneSurface;
using Metallic.StreamGeometry;

#if METALLIC_VKGLTF_SER
// Slang 2026.18.2 exposes the split HitObject API but not the fused EXT
// operation. Match its built-in TraceRay payload storage until it does.
[require(spirv, spvShaderInvocationReorderEXT)]
void traceReorderExecuteExperimental(out HitObject hitObject,
    RaytracingAccelerationStructure scene, RayDesc ray, inout StreamRayHit payload)
{
    [__vulkanRayPayload]
    static StreamRayHit rayPayload;
    rayPayload = payload;
    spirv_asm
    {
        OpExtension "SPV_EXT_shader_invocation_reorder";
        OpCapability ShaderInvocationReorderEXT;
        OpHitObjectTraceReorderExecuteEXT &hitObject $scene
            $(RAY_FLAG_FORCE_NON_OPAQUE) $(0xffu) $(0u) $(0u) $(0u)
            $(ray.Origin) $(ray.TMin) $(ray.Direction) $(ray.TMax) &rayPayload;
    };
    payload = rayPayload;
}
#endif

// The experimental BSDF never imports the standard or OpenPBR integrator.
// Coverage and geometry decoding deliberately retain the shared scene ABI.
public bool alphaAccepts(uint instance, uint cluster, uint triangle, float2 bary,
    Ray ray, float t, ScenePathTracePush push)
{
    uint materialIndex;
    if (!streamRayMaterialIndex(instance, materialIndex)) { return false; }
    uint count, stride;
    sceneMaterials().GetDimensions(count, stride);
    if (materialIndex >= count) { return false; }
    let material = sceneMaterialCoverageParameters(loadSceneMaterial(materialIndex));
    if (material.alphaMode < 0.5) { return true; }
    HitInfo hit;
    return streamRaySurface(instance, cluster, triangle, bary, ray, t, push.bitangentFlip, hit) &&
        evaluateSceneCoverage(materialIndex, material, hit.texcoord, push).accepted;
}

public bool traceClosest(Ray ray, float nearDistance, float farDistance,
    ScenePathTracePush push, out HitInfo hit)
{
    hit = (HitInfo)0;
    RayDesc desc;
    desc.Origin = ray.origin;
    desc.Direction = ray.direction;
    desc.TMin = nearDistance;
    desc.TMax = farDistance;
    StreamRayHit payload = (StreamRayHit)0;
    payload.t = -1.0;
    let scene = resolveAccelerationStructure(
        getResourceParameterField<SceneResourceParameters, uint64_t>(kSceneSceneOffset));
#if METALLIC_VKGLTF_SER
    HitObject hitObject;
    traceReorderExecuteExperimental(hitObject, scene, desc, payload);
#else
    TraceRay(scene, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, 0, 0, 0, desc, payload);
#endif
    return payload.t >= 0.0 && streamRaySurface(payload.instanceId, payload.clusterId,
        payload.primitiveId, payload.barycentrics, ray, payload.t, push.bitangentFlip, hit);
}

public float3 accumulate(uint2 pixel, float3 color, ScenePathTracePush push)
{
    let changes = resolveBuffer<StructuredBuffer<uint>>(
        getResourceParameterField<SceneResourceParameters, ResourceHandle<ByteAddressBuffer>>(kSceneRayStreamChangesOffset));
    float count = 1.0;
    if (push.hasHistory != 0u && changes[METALLIC_STREAM_BLAS_SEMANTIC_CHANGE_WORD] == 0u) {
        float4 previous = sceneHistoryPrevious()[pixel];
        count = min(previous.a + 1.0, 16777216.0);
        color = lerp(previous.rgb, color, rcp(count));
    }
    sceneHistoryCurrent()[pixel] = float4(color, count);
    return color;
}
} // namespace Metallic.ExperimentalPath

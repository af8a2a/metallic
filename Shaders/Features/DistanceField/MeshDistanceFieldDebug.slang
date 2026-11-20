module MeshDistanceFieldDebug;

import ShaderCore;
import ParameterRoot;
import MeshDistanceField;
using Metallic;
using Metallic.DistanceField;

struct MeshDistanceFieldDebugParams
{
    ResourceHandle<StructuredBuffer<MeshDistanceFieldInstance>> instances;
    ResourceHandle<StructuredBuffer<MeshDistanceFieldMip>> mips;
    ResourceHandle<StructuredBuffer<uint>> indirection;
    ResourceHandle<Texture3D<float>> atlas;
    SamplerHandle atlasSampler;
    ResourceHandle<RWTexture2D<float4>> color;
    ResourceHandle<RWTexture2D<float4>> trace;
    uint instanceCount;
    float4 clipToWorld0;
    float4 clipToWorld1;
    float4 clipToWorld2;
    float4 clipToWorld3;
    float4 cameraEyeAndOrthographic;
    uint bricksPerAxis;
    float minTraceDistance;
    float maxTraceDistance;
    float nearClipZ;
    float farClipZ;
    uint debugMode;
    uint padding0;
    uint padding1;
};

float3 unproject(MeshDistanceFieldDebugParams params, float2 ndc, float depth)
{
    float4 clip = float4(ndc, depth, 1.0f);
    float3 world = float3(dot(params.clipToWorld0, clip), dot(params.clipToWorld1, clip),
        dot(params.clipToWorld2, clip));
    return world / dot(params.clipToWorld3, clip);
}

float3 instanceColor(uint instanceID)
{
    uint hash = instanceID * 0x9e3779b9u;
    hash ^= hash >> 16u;
    return 0.2f + 0.8f * float3(hash & 255u, (hash >> 8u) & 255u, (hash >> 16u) & 255u) / 255.0f;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void meshDistanceFieldDebugMain(uint3 id : SV_DispatchThreadID)
{
    MeshDistanceFieldDebugParams params = getParameters<MeshDistanceFieldDebugParams>();
    RWTexture2D<float4> color = resolveUniform(params.color);
    RWTexture2D<float4> trace = resolveUniform(params.trace);
    uint width, height;
    color.GetDimensions(width, height);
    if (id.x >= width || id.y >= height) {
        return;
    }

    float2 uv = (float2(id.xy) + 0.5f) / float2(width, height);
    float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    float3 nearWorld = unproject(params, ndc, params.nearClipZ);
    bool orthographic = params.cameraEyeAndOrthographic.w != 0.0f;
    float3 rayStart = orthographic ? nearWorld : params.cameraEyeAndOrthographic.xyz;
    float3 rayEnd = orthographic ? unproject(params, ndc, params.farClipZ) : nearWorld;
    float3 rayDirection = normalize(rayEnd - rayStart);

    float hitDistance = params.maxTraceDistance;
    uint hitInstance = 0xffffffffu;
    uint instanceID = 0u;
    uint totalSteps = 0u;
    float3 display = float3(0.025f, 0.035f, 0.055f);
    if (params.instanceCount != 0u) {
        StructuredBuffer<MeshDistanceFieldInstance> instances = resolveUniform(params.instances);
        MeshDistanceFieldResources resources = {resolveUniform(params.mips), resolveUniform(params.indirection),
            resolveUniform(params.atlas), resolveUniform(params.atlasSampler), params.bricksPerAxis};
        for (uint index = 0u; index < params.instanceCount; ++index) {
            uint objectSteps;
            // LumenVisualize.usf uses distance-based surface expansion and an initial distance of zero.
            if (rayTraceSingleMeshSDF(resources, instances[index], rayStart, rayDirection, params.minTraceDistance,
                    params.maxTraceDistance, false, 0.0f, hitDistance, objectSteps)) {
                hitInstance = index;
            }
            totalSteps += objectSteps;
        }

        if (hitInstance != 0xffffffffu) {
            MeshDistanceFieldInstance instance = instances[hitInstance];
            instanceID = instance.indices.y;
            if (params.debugMode == 2u) {
                display = instanceColor(instanceID);
            } else if (params.debugMode == 0u) {
                float3 hitPosition = rayStart + rayDirection * hitDistance;
                float3 normal = calculateMeshSDFWorldNormal(resources, instance, hitPosition);
                float illumination = saturate(dot(normal, normalize(float3(0.4f, 0.8f, 0.6f))));
                display = (normal * 0.5f + 0.5f) * (0.35f + 0.65f * illumination);
            }
        }
    }

    if (params.debugMode == 1u && totalSteps != 0u) {
        float heat = saturate(log2(1.0f + float(totalSteps)) / log2(1.0f + float(kMeshDistanceFieldMaxTraceSteps)));
        display = lerp(float3(0.015f, 0.08f, 0.35f), float3(1.0f, 0.30f, 0.025f), heat);
    }
    bool hit = hitInstance != 0xffffffffu;
    color[id.xy] = float4(display, 1.0f);
    trace[id.xy] = float4(hit ? hitDistance : -1.0f, float(totalSteps), float(instanceID), hit ? 1.0f : 0.0f);
}

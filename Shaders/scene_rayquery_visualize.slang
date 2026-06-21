[[vk::binding(0, 0)]]
RaytracingAccelerationStructure gScene;

[[vk::binding(1, 0)]]
RWTexture2D<float4> gOutput;

#ifndef SCENE_RAYQUERY_ENABLE_CLUSTER_ID
#define SCENE_RAYQUERY_ENABLE_CLUSTER_ID 0
#endif

static const uint kRayQueryVisualizationModeInstance = 0u;
static const uint kRayQueryVisualizationModePrimitive = 1u;
static const uint kRayQueryVisualizationModeClusterId = 2u;

struct SceneRayQueryVisualizationPush
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
    uint mode;
    uint width;
    uint height;
    uint padding;
};

float3 safeNormalize(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 0.00000001 ? value * rsqrt(lengthSquared) : fallback;
}

float3 hashColor(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;

    float3 color = float3(
        float(value & 255u),
        float((value >> 8) & 255u),
        float((value >> 16) & 255u)) / 255.0;
    return lerp(float3(0.16, 0.20, 0.26), color, 0.82);
}

#if SCENE_RAYQUERY_ENABLE_CLUSTER_ID
int committedClusterId(inout RayQuery<RAY_FLAG_FORCE_OPAQUE> query)
{
    return spirv_asm {
        OpCapability RayTracingClusterAccelerationStructureNV;
        OpExtension "SPV_NV_cluster_acceleration_structure";
        OpRayQueryGetIntersectionClusterIdNV $$int result &query $(1);
    };
}
#else
int committedClusterId(inout RayQuery<RAY_FLAG_FORCE_OPAQUE> query)
{
    return -1;
}
#endif

float3 rayDirectionFromPixel(uint2 pixel, SceneRayQueryVisualizationPush push)
{
    float aspect = max(push.viewport.x, 0.001);
    float2 uv = (float2(pixel) + 0.5) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(push.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));

    if (push.upProjection.w > 0.5) {
        return forward;
    }

    float fovRadians = clamp(push.viewport.w, 0.017453292, 3.12413936);
    float halfHeight = tan(fovRadians * 0.5);
    return safeNormalize(
        forward + right * (ndc.x * aspect * halfHeight) + up * (ndc.y * halfHeight),
        forward);
}

float3 rayOriginFromPixel(uint2 pixel, SceneRayQueryVisualizationPush push)
{
    if (push.upProjection.w <= 0.5) {
        return push.eye.xyz;
    }

    float aspect = max(push.viewport.x, 0.001);
    float2 uv = (float2(pixel) + 0.5) / float2(max(push.width, 1u), max(push.height, 1u));
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    float3 forward = safeNormalize(push.center.xyz - push.eye.xyz, float3(0.0, 0.0, -1.0));
    float3 cameraUp = safeNormalize(push.upProjection.xyz, float3(0.0, 1.0, 0.0));
    float3 right = safeNormalize(cross(forward, cameraUp), float3(1.0, 0.0, 0.0));
    float3 up = safeNormalize(cross(right, forward), float3(0.0, 1.0, 0.0));
    float halfHeight = max(push.clipOrtho.z * 0.5, 0.0001);
    return push.eye.xyz + right * (ndc.x * halfHeight * aspect) + up * (ndc.y * halfHeight);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void sceneRayQueryVisualizeMain(uint3 dispatchThreadId : SV_DispatchThreadID,
                                uniform SceneRayQueryVisualizationPush push)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= push.width || pixel.y >= push.height) {
        return;
    }

    RayDesc ray;
    ray.Origin = rayOriginFromPixel(pixel, push);
    ray.Direction = rayDirectionFromPixel(pixel, push);
    ray.TMin = max(push.clipOrtho.x, 0.0001);
    ray.TMax = max(push.clipOrtho.y, ray.TMin + 0.0001);

    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(gScene, RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    while (query.Proceed()) {
    }

    float3 color = float3(0.010, 0.012, 0.016);
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        uint instanceIndex = query.CommittedInstanceIndex();
        uint primitiveIndex = query.CommittedPrimitiveIndex();
        uint colorId = 0u;
        if (push.mode == kRayQueryVisualizationModeInstance) {
            colorId = instanceIndex + 1u;
        } else if (push.mode == kRayQueryVisualizationModeClusterId) {
            int clusterId = committedClusterId(query);
            colorId = clusterId >= 0 ? uint(clusterId) + 1u : 0x7fffffffu;
        } else {
            colorId = ((instanceIndex + 1u) * 16777619u) ^ (primitiveIndex + 1u);
        }
        float distanceFade = saturate(1.0 - query.CommittedRayT() / max(push.clipOrtho.y, 0.0001));
        color = hashColor(colorId) * (0.55 + distanceFade * 0.45);
    }

    gOutput[pixel] = float4(color, 1.0);
}

#include "../../Libraries/Resources/ComputeResources.slang"

// Reuse scene alpha-mask and neural-texture evaluation for closest-hit shadow rays.
#include "../PathTracing/ScenePathTrace.slang"
#include "../../Libraries/Lighting/ScreenSpaceShadowCommon.slang"
#include "../../Libraries/Denoising/NRD.hlsli"

#define gShadowParameters METALLIC_RESOURCE(StructuredBuffer<ShadowParameters>, 80)
#define gDepth METALLIC_RESOURCE(Texture2D<float>, 81)
#define gPenumbra METALLIC_RESOURCE(RWTexture2D<float>, 82)
#define gNormal METALLIC_RESOURCE(RWTexture2D<float4>, 83)
#define gViewZ METALLIC_RESOURCE(RWTexture2D<float>, 84)
#define gMotion METALLIC_RESOURCE(RWTexture2D<float4>, 85)
#define gShadow METALLIC_RESOURCE(RWTexture2D<float>, 86)

struct ShadowGeometryPush
{
    uint materialTextureCount;
    uint ntcTextureSetCount;
};

float traceGeometryShadow(float3 origin, float3 direction, float maxDistance)
{
    ShadowGeometryPush geometry = METALLIC_CONSTANTS(ShadowGeometryPush);
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.materialTextureCount = geometry.materialTextureCount;
    push.ntcTextureSetCount = geometry.ntcTextureSetCount;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = max(maxDistance, ray.TMin);
    // SIGMA needs the closest blocker's distance, not the first traversal hit.
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(gScene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            alphaMaskAcceptsHit(query.CandidateInstanceIndex(), query.CandidatePrimitiveIndex(),
                query.CandidateTriangleBarycentrics(), push)) {
            query.CommitNonOpaqueTriangleHit();
        }
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? min(query.CommittedRayT(), NRD_FP16_MAX - 1.0) : NRD_FP16_MAX;
}

void shadowCameraBasis(ViewCameraConstants camera, out float3 forward, out float3 right, out float3 up)
{
    forward = normalize(camera.center.xyz - camera.eye.xyz);
    right = normalize(cross(forward, camera.upProjection.xyz));
    up = cross(right, forward);
}

float shadowLinearDepth(int2 pixel, ViewCameraConstants camera)
{
    float depth = gDepth.Load(int3(pixel, 0));
    float nearPlane = camera.clipOrtho.x, farPlane = camera.clipOrtho.y;
    bool reversed = camera.clipOrtho.w > 0.5;
    if (camera.upProjection.w > 0.5) { return reversed ? lerp(farPlane, nearPlane, depth) : lerp(nearPlane, farPlane, depth); }
    // Keep reversed-Z precision when the near/far ratio is very small.
    float denominator = reversed ? nearPlane + depth * (farPlane - nearPlane)
        : farPlane - depth * (farPlane - nearPlane);
    return nearPlane * farPlane / max(denominator, 1e-7);
}

float3 shadowWorldPosition(int2 pixel, ShadowParameters p)
{
    float3 forward, right, up;
    shadowCameraBasis(p.view.current, forward, right, up);
    float2 jitter = p.view.frame.z != 0u ? p.view.jitter.xy : float2(0.0);
    float2 uv = (float2(pixel) + 0.5 + jitter) / p.view.current.viewport.yz;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float z = shadowLinearDepth(pixel, p.view.current);
    float halfHeight = p.view.current.upProjection.w > 0.5 ? p.view.current.clipOrtho.z * 0.5
        : z * tan(p.view.current.viewport.w * 0.5);
    return p.view.current.eye.xyz + forward * z + halfHeight *
        (right * ndc.x * p.view.current.viewport.x + up * ndc.y);
}

float3 shadowProject(float3 position, ViewCameraConstants camera)
{
    float3 forward, right, up;
    shadowCameraBasis(camera, forward, right, up);
    float3 delta = position - camera.eye.xyz;
    float z = dot(delta, forward);
    float halfHeight = camera.upProjection.w > 0.5 ? camera.clipOrtho.z * 0.5
        : max(z, 1e-6) * tan(camera.viewport.w * 0.5);
    return float3(float2(dot(delta, right) / (halfHeight * camera.viewport.x),
        -dot(delta, up) / halfHeight) * 0.5 + 0.5, z);
}

float3 shadowDepthNormal(int2 pixel, ShadowParameters p, float3 position)
{
    int2 last = int2(p.view.current.viewport.yz) - 1;
    int2 left = max(pixel - int2(1, 0), int2(0));
    int2 right = min(pixel + int2(1, 0), last);
    int2 top = max(pixel - int2(0, 1), int2(0));
    int2 bottom = min(pixel + int2(0, 1), last);
    float centerZ = shadowLinearDepth(pixel, p.view.current);
    // Choose the neighbor on the same surface at depth discontinuities.
    bool useLeft = pixel.x == last.x || (pixel.x > 0 &&
        abs(shadowLinearDepth(left, p.view.current) - centerZ) < abs(shadowLinearDepth(right, p.view.current) - centerZ));
    bool useTop = pixel.y == last.y || (pixel.y > 0 &&
        abs(shadowLinearDepth(top, p.view.current) - centerZ) < abs(shadowLinearDepth(bottom, p.view.current) - centerZ));
    float3 dx = useLeft ? position - shadowWorldPosition(left, p) : shadowWorldPosition(right, p) - position;
    float3 dy = useTop ? position - shadowWorldPosition(top, p) : shadowWorldPosition(bottom, p) - position;
    float3 normal = cross(dy, dx);
    return dot(normal, normal) > 1e-20 ? normalize(normal)
        : normalize(p.view.current.eye.xyz - p.view.current.center.xyz);
}

uint shadowHash(uint seed)
{
    seed ^= seed >> 16;
    seed *= 0x7feb352du;
    seed ^= seed >> 15;
    seed *= 0x846ca68bu;
    return seed ^ (seed >> 16);
}

float shadowRandom(inout uint seed)
{
    seed = shadowHash(seed);
    return float(seed >> 8) * (1.0 / 16777216.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void rayTracedShadowsMain(uint3 dispatchId : SV_DispatchThreadID)
{
    ShadowParameters p = gShadowParameters[0];
    uint2 pixel = dispatchId.xy;
    uint2 size = uint2(p.view.current.viewport.yz);
    if (any(pixel >= size)) { return; }
    float viewZ = shadowLinearDepth(int2(pixel), p.view.current);
    bool background = viewZ >= p.view.current.clipOrtho.y * 0.9999;
    float3 position = shadowWorldPosition(int2(pixel), p);
    float3 normal = background ? float3(0, 0, 1) : shadowDepthNormal(int2(pixel), p, position);
    gViewZ[pixel] = background ? 1e6 : viewZ;
    gNormal[pixel] = NRD_FrontEnd_PackNormalAndRoughness(normal, 1.0, 0.0);
    float2 motion = float2(0.0);
    if (!background && p.view.frame.y != 0u) {
        float3 previous = shadowProject(position, p.view.previous);
        float2 currentUv = shadowProject(position, p.view.current).xy;
        if (previous.z > p.view.previous.clipOrtho.x) { motion = previous.xy - currentUv; }
    }
    gMotion[pixel] = float4(motion, 0.0, 0.0);
    float distance = NRD_FP16_MAX;
    float penumbra = NRD_FP16_MAX;
    if (!background && p.control.x != 0u) {
        PunctualLightSample light = samplePunctualLightData(p.light, position);
        if (any(light.illuminance > 0.0)) {
            uint seed = shadowHash(pixel.x + pixel.y * size.x + shadowHash(p.view.frame.x));
            float2 random = float2(shadowRandom(seed), shadowRandom(seed));
            float angle = random.x * 6.28318530718;
            float radius = sqrt(random.y);
            float3 tangent = normalize(cross(abs(light.direction.z) < 0.99 ? float3(0, 0, 1) : float3(0, 1, 0), light.direction));
            float3 bitangent = cross(light.direction, tangent);
            float angularRadius = p.light.directionType.w < 0.5 ? p.trace.w : p.shape.x / max(light.distance, 1e-4);
            // NRD-Sample/TraceOpaque.cs.hlsl: sample the source disk, query the
            // scene TLAS, then pack the blocker distance and tan(angular radius).
            float3 disk = radius * (cos(angle) * tangent + sin(angle) * bitangent);
            float3 origin = position + normal * p.trace.z;
            float3 direction;
            float rayLength = p.trace.x;
            if (p.light.directionType.w < 0.5) {
                direction = normalize(light.direction + angularRadius * disk);
            } else {
                // End at the sampled local emitter, accounting for the origin offset.
                float3 target = p.light.positionRange.xyz + p.shape.x * disk;
                float3 delta = target - origin;
                float targetDistance = length(delta);
                direction = delta / max(targetDistance, 1e-6);
                rayLength = min(rayLength, targetDistance - 0.001);
            }
            if (rayLength > 0.001) { distance = traceGeometryShadow(origin, direction, rayLength); }
            penumbra = p.light.directionType.w < 0.5 ? SIGMA_FrontEnd_PackPenumbra(distance, p.trace.w)
                : SIGMA_FrontEnd_PackPenumbra(distance, light.distance, p.shape.x * 2.0);
        }
    }
    gPenumbra[pixel] = penumbra;
    // SIGMA reads the previous encoded output during Copy; preserve it until denoising.
    if (p.shape.y == 0.0 || background) { gShadow[pixel] = distance >= NRD_FP16_MAX ? 1.0 : 0.0; }
}

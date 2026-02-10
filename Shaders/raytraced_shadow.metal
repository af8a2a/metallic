#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct ShadowUniforms {
    float4x4 invViewProj;
    float4   lightDir;       // world-space direction TO light (normalized)
    uint     screenWidth;
    uint     screenHeight;
    float    normalBias;
    float    maxRayDistance;
    uint     reversedZ;      // 1 if using reversed-Z, 0 otherwise
};

kernel void shadowRayMain(
    constant ShadowUniforms& uniforms  [[buffer(0)]],
    instance_acceleration_structure tlas [[buffer(1)]],
    texture2d<float, access::read> depthTex [[texture(0)]],
    texture2d<float, access::write> shadowMap [[texture(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    if (tid.x >= uniforms.screenWidth || tid.y >= uniforms.screenHeight)
        return;

    float depth = depthTex.read(tid).x;

    // Sky pixels — no shadow
    // Reversed-Z: sky = 0.0 (clear), near = 1.0
    // Normal-Z:   sky = 1.0 (clear), near = 0.0
    float skyClear = uniforms.reversedZ ? 0.0 : 1.0;
    if (depth == skyClear) {
        shadowMap.write(float4(1.0), tid);
        return;
    }

    // Reconstruct NDC
    float2 ndc;
    ndc.x = (float(tid.x) + 0.5) / float(uniforms.screenWidth)  *  2.0 - 1.0;
    ndc.y = 1.0 - (float(tid.y) + 0.5) / float(uniforms.screenHeight) * 2.0;

    // Reconstruct world position via inverse view-projection
    float4 clipPos = float4(ndc, depth, 1.0);
    float4 worldPos4 = uniforms.invViewProj * clipPos;
    float3 worldPos = worldPos4.xyz / worldPos4.w;

    // Offset ray origin along light direction to reduce self-shadowing
    float3 lightDir = normalize(uniforms.lightDir.xyz);
    float3 origin = worldPos + lightDir * uniforms.normalBias;

    // Trace shadow ray
    ray shadowRay;
    shadowRay.origin = origin;
    shadowRay.direction = lightDir;
    shadowRay.min_distance = 0.001;
    shadowRay.max_distance = uniforms.maxRayDistance;

    intersector<triangle_data, instancing> inter;
    inter.accept_any_intersection(true);

    auto result = inter.intersect(shadowRay, tlas);

    float shadow = (result.type == intersection_type::none) ? 1.0 : 0.0;
    shadowMap.write(float4(shadow), tid);
}

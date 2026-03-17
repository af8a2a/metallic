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

static inline bool isSkyDepth(float depth, constant ShadowUniforms& uniforms) {
    float skyClear = uniforms.reversedZ ? 0.0 : 1.0;
    return fabs(depth - skyClear) < 1e-6;
}

static inline float3 reconstructWorldPosition(uint2 pixel,
                                              float depth,
                                              constant ShadowUniforms& uniforms) {
    float2 ndc;
    ndc.x = (float(pixel.x) + 0.5) / float(uniforms.screenWidth)  *  2.0 - 1.0;
    ndc.y = 1.0 - (float(pixel.y) + 0.5) / float(uniforms.screenHeight) * 2.0;

    float4 clipPos = float4(ndc, depth, 1.0);
    float4 worldPos4 = uniforms.invViewProj * clipPos;
    return worldPos4.xyz / worldPos4.w;
}

static inline bool sampleWorldPosition(int2 pixel,
                                       constant ShadowUniforms& uniforms,
                                       texture2d<float, access::read> depthTex,
                                       thread float3& worldPos) {
    if (pixel.x < 0 || pixel.y < 0 ||
        pixel.x >= int(uniforms.screenWidth) ||
        pixel.y >= int(uniforms.screenHeight)) {
        worldPos = float3(0.0);
        return false;
    }

    float depth = depthTex.read(uint2(pixel)).x;
    if (isSkyDepth(depth, uniforms)) {
        worldPos = float3(0.0);
        return false;
    }

    worldPos = reconstructWorldPosition(uint2(pixel), depth, uniforms);
    return true;
}

static inline float3 estimateWorldNormal(uint2 pixel,
                                         float3 worldPos,
                                         float3 lightDir,
                                         constant ShadowUniforms& uniforms,
                                         texture2d<float, access::read> depthTex) {
    float3 leftPos;
    float3 rightPos;
    float3 upPos;
    float3 downPos;

    bool hasLeft = sampleWorldPosition(int2(pixel) + int2(-1, 0), uniforms, depthTex, leftPos);
    bool hasRight = sampleWorldPosition(int2(pixel) + int2(1, 0), uniforms, depthTex, rightPos);
    bool hasUp = sampleWorldPosition(int2(pixel) + int2(0, -1), uniforms, depthTex, upPos);
    bool hasDown = sampleWorldPosition(int2(pixel) + int2(0, 1), uniforms, depthTex, downPos);

    float3 dx = lightDir;
    bool hasDx = false;
    if (hasLeft && hasRight) {
        float3 leftDelta = worldPos - leftPos;
        float3 rightDelta = rightPos - worldPos;
        dx = dot(leftDelta, leftDelta) < dot(rightDelta, rightDelta) ? leftDelta : rightDelta;
        hasDx = true;
    } else if (hasLeft) {
        dx = worldPos - leftPos;
        hasDx = true;
    } else if (hasRight) {
        dx = rightPos - worldPos;
        hasDx = true;
    }

    float3 dy = lightDir;
    bool hasDy = false;
    if (hasUp && hasDown) {
        float3 upDelta = worldPos - upPos;
        float3 downDelta = downPos - worldPos;
        dy = dot(upDelta, upDelta) < dot(downDelta, downDelta) ? upDelta : downDelta;
        hasDy = true;
    } else if (hasUp) {
        dy = worldPos - upPos;
        hasDy = true;
    } else if (hasDown) {
        dy = downPos - worldPos;
        hasDy = true;
    }

    if (!hasDx || !hasDy) {
        return lightDir;
    }

    float3 normal = normalize(cross(dx, dy));
    if (dot(normal, lightDir) < 0.0) {
        normal = -normal;
    }
    return normal;
}

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

    if (isSkyDepth(depth, uniforms)) {
        shadowMap.write(float4(1.0), tid);
        return;
    }

    float3 lightDir = normalize(uniforms.lightDir.xyz);
    float3 worldPos = reconstructWorldPosition(tid, depth, uniforms);
    float3 surfaceNormal = estimateWorldNormal(tid, worldPos, lightDir, uniforms, depthTex);
    float3 origin = worldPos + surfaceNormal * uniforms.normalBias;

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

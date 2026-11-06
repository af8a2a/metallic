#language slang 2026
module VirtualShadowMapTemporal;

import ShaderCore;
import VirtualShadowMapCommon;
using Metallic;
using Metallic.Shadow;

#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapTemporal.h"
[[vk::push_constant]] ConstantBuffer<VirtualShadowMapTemporalParams> params;

bool temporalReceiver(int2 pixel, ViewConstants view, float depth, out float3 position, out float linearDepth)
{
    position = float3(0.0); linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= int2(view.current.viewport.yz))) { return false; }
    bool reversed = view.current.clipOrtho.w > 0.5;
    if (depth == (reversed ? 0.0 : 1.0)) { return false; }
    float nearPlane = view.current.clipOrtho.x, farPlane = view.current.clipOrtho.y;
    if (view.current.upProjection.w > 0.5) {
        linearDepth = reversed ? lerp(farPlane, nearPlane, depth) : lerp(nearPlane, farPlane, depth);
    } else {
        float denominator = reversed ? nearPlane + depth * (farPlane - nearPlane) : farPlane - depth * (farPlane - nearPlane);
        linearDepth = nearPlane * farPlane / denominator;
    }
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    float2 uv = (float2(pixel) + 0.5 + jitter) / view.current.viewport.yz;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float halfHeight = view.current.upProjection.w > 0.5 ? view.current.clipOrtho.z * 0.5 :
        linearDepth * tan(view.current.viewport.w * 0.5);
    position = view.current.eye.xyz + forward * linearDepth +
        halfHeight * (right * ndc.x * view.current.viewport.x + up * ndc.y);
    return true;
}

bool temporalSurface(int2 pixel, bool previous, ViewConstants view, out float3 position, out float linearDepth)
{
    position = float3(0.0); linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= int2(view.current.viewport.yz))) { return false; }
    float depth;
    if (previous) {
        uint2 stored = resolveUniform(params.previousHistory)[pixel];
        if ((stored.x >> 16u) == 0u) { return false; }
        depth = asfloat(stored.y);
    } else {
        depth = resolveUniform(params.depth).Load(int3(pixel, 0));
    }
    return temporalReceiver(pixel, view, depth, position, linearDepth);
}

float3 temporalNormal(int2 pixel, bool previous, ViewConstants view, float3 position, float linearDepth)
{
    float3 left, right, top, bottom;
    float zl, zr, zt, zb;
    bool hasLeft = temporalSurface(pixel - int2(1, 0), previous, view, left, zl);
    bool hasRight = temporalSurface(pixel + int2(1, 0), previous, view, right, zr);
    bool hasTop = temporalSurface(pixel - int2(0, 1), previous, view, top, zt);
    bool hasBottom = temporalSurface(pixel + int2(0, 1), previous, view, bottom, zb);
    bool useLeft = hasLeft && (!hasRight || abs(zl - linearDepth) < abs(zr - linearDepth));
    bool useTop = hasTop && (!hasBottom || abs(zt - linearDepth) < abs(zb - linearDepth));
    float3 dx = useLeft ? position - left : (hasRight ? right - position : float3(0.0));
    float3 dy = useTop ? position - top : (hasBottom ? bottom - position : float3(0.0));
    float3 normal = cross(dy, dx);
    float lengthSquared = dot(normal, normal);
    return lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);
}

bool temporalProject(float3 position, ViewConstants view, out float2 pixel)
{
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float3 delta = position - view.current.eye.xyz;
    float z = dot(delta, forward);
    pixel = float2(0.0);
    if (z <= view.current.clipOrtho.x || z >= view.current.clipOrtho.y) { return false; }
    float halfHeight = view.current.upProjection.w > 0.5 ? view.current.clipOrtho.z * 0.5 : z * tan(view.current.viewport.w * 0.5);
    float2 ndc = float2(dot(delta, right) / (halfHeight * view.current.viewport.x), dot(delta, up) / halfHeight);
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    pixel = uv * view.current.viewport.yz - 0.5 - jitter;
    return all(pixel >= -0.5) && all(pixel < view.current.viewport.yz - 0.5);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmTemporalMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height) { return; }
    ViewConstants current = params.views.load(0);
    ViewConstants previous = params.views.load(1);
    float rawDepth = resolveUniform(params.depth).Load(int3(id.xy, 0));
    float3 position;
    float linearDepth;
    if (!temporalReceiver(int2(id.xy), current, rawDepth, position, linearDepth)) {
        resolveUniform(params.currentHistory)[id.xy] = uint2(0u, asuint(rawDepth));
        return;
    }
    uint face = params.lightType == 2u ? virtualShadowMapPointFace(position - float3(params.lightX, params.lightY, params.lightZ)) : 0u;
    uint layer = params.baseLayer + (params.compact != 0u ? 0u : face);
    float raw = resolveUniform(params.shadows)[uint3(id.xy, layer)];
    float result = raw;
    uint count = 1u;
    float2 previousPixel;
    if (params.hasHistory != 0u && temporalProject(position, previous, previousPixel)) {
        float3 normal = temporalNormal(int2(id.xy), false, current, position, linearDepth);
        float pixelSize = current.current.upProjection.w > 0.5 ? current.current.clipOrtho.z / float(params.height) :
            2.0 * linearDepth * tan(current.current.viewport.w * 0.5) / float(params.height);
        float tolerance = max(1e-4, pixelSize * 0.01 + linearDepth * 1e-5);
        int2 base = int2(floor(previousPixel));
        float2 fraction = frac(previousPixel);
        float sum = 0.0, weightSum = 0.0, sampleCount = 0.0;
        [unroll]
        for (uint tap = 0u; tap < 4u; ++tap) {
            int2 offset = int2(tap & 1u, tap >> 1u);
            int2 pixel = base + offset;
            float2 weights = lerp(1.0 - fraction, fraction, float2(offset));
            float weight = weights.x * weights.y;
            if (weight <= 0.0) { continue; }
            float3 oldPosition;
            float oldDepth;
            if (!temporalSurface(pixel, true, previous, oldPosition, oldDepth)) { continue; }
            float3 oldNormal = temporalNormal(pixel, true, previous, oldPosition, oldDepth);
            if (dot(normal, oldNormal) < 0.9 || abs(dot(oldPosition - position, normal)) > tolerance ||
                abs(dot(oldPosition - position, oldNormal)) > tolerance) { continue; }
            uint2 history = resolveUniform(params.previousHistory)[pixel];
            sum += weight * f16tof32(history.x & 0xffffu);
            sampleCount += weight * float(history.x >> 16u);
            weightSum += weight;
        }
        // Reject footprints mostly outside the old surface instead of borrowing
        // a tiny valid corner across a disocclusion or silhouette.
        if (weightSum >= 0.5) {
            uint priorCount = min(uint(sampleCount / weightSum + 0.5), params.maxFrames - 1u);
            count = priorCount + 1u;
            result = (sum / weightSum * float(priorCount) + raw) / float(count);
        }
    }
    resolveUniform(params.currentHistory)[id.xy] = uint2((count << 16u) | (f32tof16(result) & 0xffffu), asuint(rawDepth));
    // Each invocation touches only its own raw pixel. A point light resolves
    // once after all six faces, preserving every noncanonical legacy layer.
    resolveUniform(params.shadows)[uint3(id.xy, layer)] = result;
}

#language slang 2026
module VirtualShadowMapTemporal;

import ShaderCore;
import VirtualShadowMapCommon;
using Metallic;
using Metallic.Shadow;

#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapTemporal.h"
[[vk::push_constant]] ConstantBuffer<VirtualShadowMapTemporalParams> params;

struct TemporalViewBasis
{
    float3 eye;
    float3 forward;
    float3 right;
    float3 up;
    float2 viewportExtent;
    float2 invViewport;
    float viewportAspect;
    float tanHalfFov;
    float halfOrthoHeight;
    float nearPlane;
    float farPlane;
    float farMinusNear;
    float2 jitter;
    bool isOrtho;
    bool reversed;
};

TemporalViewBasis createTemporalViewBasis(ViewConstants view)
{
    TemporalViewBasis b;
    b.eye = view.current.eye.xyz;
    b.forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    b.right = normalize(cross(b.forward, view.current.upProjection.xyz));
    b.up = cross(b.right, b.forward);
    b.viewportExtent = view.current.viewport.yz;
    b.invViewport = 1.0 / view.current.viewport.yz;
    b.viewportAspect = view.current.viewport.x;
    b.tanHalfFov = tan(view.current.viewport.w * 0.5);
    b.isOrtho = view.current.upProjection.w > 0.5;
    b.halfOrthoHeight = view.current.clipOrtho.z * 0.5;
    b.nearPlane = view.current.clipOrtho.x;
    b.farPlane = view.current.clipOrtho.y;
    b.farMinusNear = view.current.clipOrtho.y - view.current.clipOrtho.x;
    b.reversed = view.current.clipOrtho.w > 0.5;
    b.jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    return b;
}

bool temporalReceiver(int2 pixel, TemporalViewBasis basis, float depth, out float3 position, out float linearDepth)
{
    position = float3(0.0); linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= int2(basis.viewportExtent))) { return false; }
    if (depth == (basis.reversed ? 0.0 : 1.0)) { return false; }
    if (basis.isOrtho) {
        linearDepth = basis.reversed ? lerp(basis.farPlane, basis.nearPlane, depth) : lerp(basis.nearPlane, basis.farPlane, depth);
    } else {
        float denominator = basis.reversed ? basis.nearPlane + depth * basis.farMinusNear : basis.farPlane - depth * basis.farMinusNear;
        linearDepth = basis.nearPlane * basis.farPlane / denominator;
    }
    float2 uv = (float2(pixel) + 0.5 + basis.jitter) * basis.invViewport;
    float2 ndc = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    float halfHeight = basis.isOrtho ? basis.halfOrthoHeight : linearDepth * basis.tanHalfFov;
    position = basis.eye + basis.forward * linearDepth +
        halfHeight * (basis.right * (ndc.x * basis.viewportAspect) + basis.up * ndc.y);
    return true;
}

struct HistorySample
{
    bool valid;
    uint2 history;
    float3 position;
    float linearDepth;
};

HistorySample fetchHistorySample(int2 pixel, TemporalViewBasis basis, RWTexture2D<uint2> prevHistory)
{
    HistorySample s;
    s.valid = false; s.history = uint2(0u); s.position = float3(0.0); s.linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= int2(basis.viewportExtent))) { return s; }
    uint2 stored = prevHistory[pixel];
    if ((stored.x >> 16u) == 0u) { return s; }
    float depth = asfloat(stored.y);
    s.valid = temporalReceiver(pixel, basis, depth, s.position, s.linearDepth);
    s.history = stored;
    return s;
}

float3 temporalNormalFromNeighbors(HistorySample left, HistorySample right, HistorySample top, HistorySample bottom, float3 position, float linearDepth)
{
    bool useLeft = left.valid && (!right.valid || abs(left.linearDepth - linearDepth) < abs(right.linearDepth - linearDepth));
    bool useTop = top.valid && (!bottom.valid || abs(top.linearDepth - linearDepth) < abs(bottom.linearDepth - linearDepth));
    float3 dx = useLeft ? position - left.position : (right.valid ? right.position - position : float3(0.0));
    float3 dy = useTop ? position - top.position : (bottom.valid ? bottom.position - position : float3(0.0));
    float3 normal = cross(dy, dx);
    float lengthSquared = dot(normal, normal);
    return lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);
}

float3 temporalCurrentNormal(int2 pixel, TemporalViewBasis basis, Texture2D<float> depthTex, float3 position, float linearDepth)
{
    float3 left, right, top, bottom;
    float zl, zr, zt, zb;
    bool hasLeft = false, hasRight = false, hasTop = false, hasBottom = false;

    int2 pLeft = pixel - int2(1, 0);
    if (!any(pLeft < 0) && !any(pLeft >= int2(basis.viewportExtent))) {
        hasLeft = temporalReceiver(pLeft, basis, depthTex.Load(int3(pLeft, 0)), left, zl);
    }
    int2 pRight = pixel + int2(1, 0);
    if (!any(pRight < 0) && !any(pRight >= int2(basis.viewportExtent))) {
        hasRight = temporalReceiver(pRight, basis, depthTex.Load(int3(pRight, 0)), right, zr);
    }
    int2 pTop = pixel - int2(0, 1);
    if (!any(pTop < 0) && !any(pTop >= int2(basis.viewportExtent))) {
        hasTop = temporalReceiver(pTop, basis, depthTex.Load(int3(pTop, 0)), top, zt);
    }
    int2 pBottom = pixel + int2(0, 1);
    if (!any(pBottom < 0) && !any(pBottom >= int2(basis.viewportExtent))) {
        hasBottom = temporalReceiver(pBottom, basis, depthTex.Load(int3(pBottom, 0)), bottom, zb);
    }

    bool useLeft = hasLeft && (!hasRight || abs(zl - linearDepth) < abs(zr - linearDepth));
    bool useTop = hasTop && (!hasBottom || abs(zt - linearDepth) < abs(zb - linearDepth));
    float3 dx = useLeft ? position - left : (hasRight ? right - position : float3(0.0));
    float3 dy = useTop ? position - top : (hasBottom ? bottom - position : float3(0.0));
    float3 normal = cross(dy, dx);
    float lengthSquared = dot(normal, normal);
    return lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);
}

bool temporalProject(float3 position, TemporalViewBasis basis, out float2 pixel)
{
    float3 delta = position - basis.eye;
    float z = dot(delta, basis.forward);
    pixel = float2(0.0);
    if (z <= basis.nearPlane || z >= basis.farPlane) { return false; }
    float halfHeight = basis.isOrtho ? basis.halfOrthoHeight : z * basis.tanHalfFov;
    float2 ndc = float2(dot(delta, basis.right) / (halfHeight * basis.viewportAspect), dot(delta, basis.up) / halfHeight);
    float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    pixel = uv * basis.viewportExtent - 0.5 - basis.jitter;
    return all(pixel >= -0.5) && all(pixel < basis.viewportExtent - 0.5);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmTemporalMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height) { return; }
    ViewConstants currentView = params.views.load(0);
    ViewConstants previousView = params.views.load(1);
    TemporalViewBasis current = createTemporalViewBasis(currentView);
    TemporalViewBasis previous = createTemporalViewBasis(previousView);

    Texture2D<float> depthTex = resolveUniform(params.depth);
    RWTexture2D<uint2> prevHistory = resolveUniform(params.previousHistory);
    RWTexture2D<uint2> currHistory = resolveUniform(params.currentHistory);
    RWTexture2DArray<float> shadowTex = resolveUniform(params.shadows);

    float rawDepth = depthTex.Load(int3(id.xy, 0));
    float3 position;
    float linearDepth;
    if (!temporalReceiver(int2(id.xy), current, rawDepth, position, linearDepth)) {
        currHistory[id.xy] = uint2(0u, asuint(rawDepth));
        return;
    }
    uint face = params.lightType == 2u ? virtualShadowMapPointFace(position - float3(params.lightX, params.lightY, params.lightZ)) : 0u;
    uint layer = params.baseLayer + (params.compact != 0u ? 0u : face);
    float raw = shadowTex[uint3(id.xy, layer)];
    float result = raw;
    uint count = 1u;
    float2 previousPixel;
    if (params.hasHistory != 0u && temporalProject(position, previous, previousPixel)) {
        float3 normal = temporalCurrentNormal(int2(id.xy), current, depthTex, position, linearDepth);
        float pixelSize = current.isOrtho ? current.halfOrthoHeight * 2.0 / float(params.height) :
            2.0 * linearDepth * current.tanHalfFov / float(params.height);
        float tolerance = max(1e-4, pixelSize * 0.01 + linearDepth * 1e-5);
        int2 base = int2(floor(previousPixel));
        float2 fraction = frac(previousPixel);

        // Pre-fetch the 4 core tap samples; core[tap] corresponds to offset (tap & 1, tap >> 1)
        HistorySample core[4];
        core[0] = fetchHistorySample(base + int2(0, 0), previous, prevHistory);
        core[1] = fetchHistorySample(base + int2(1, 0), previous, prevHistory);
        core[2] = fetchHistorySample(base + int2(0, 1), previous, prevHistory);
        core[3] = fetchHistorySample(base + int2(1, 1), previous, prevHistory);

        float sum = 0.0, weightSum = 0.0, sampleCount = 0.0;
        [unroll]
        for (uint tap = 0u; tap < 4u; ++tap) {
            int2 offset = int2(tap & 1u, tap >> 1u);
            float2 weights = lerp(1.0 - fraction, fraction, float2(offset));
            float weight = weights.x * weights.y;
            if (weight <= 0.0) { continue; }
            HistorySample sample = core[tap];
            if (!sample.valid) { continue; }

            int2 pixel = base + offset;
            HistorySample sLeft, sRight, sTop, sBottom;
            if (tap == 0u) {
                sLeft = fetchHistorySample(pixel - int2(1, 0), previous, prevHistory);
                sRight = core[1];
                sTop = fetchHistorySample(pixel - int2(0, 1), previous, prevHistory);
                sBottom = core[2];
            } else if (tap == 1u) {
                sLeft = core[0];
                sRight = fetchHistorySample(pixel + int2(1, 0), previous, prevHistory);
                sTop = fetchHistorySample(pixel - int2(0, 1), previous, prevHistory);
                sBottom = core[3];
            } else if (tap == 2u) {
                sLeft = fetchHistorySample(pixel - int2(1, 0), previous, prevHistory);
                sRight = core[3];
                sTop = core[0];
                sBottom = fetchHistorySample(pixel + int2(0, 1), previous, prevHistory);
            } else {
                sLeft = core[2];
                sRight = fetchHistorySample(pixel + int2(1, 0), previous, prevHistory);
                sTop = core[1];
                sBottom = fetchHistorySample(pixel + int2(0, 1), previous, prevHistory);
            }

            float3 oldNormal = temporalNormalFromNeighbors(sLeft, sRight, sTop, sBottom, sample.position, sample.linearDepth);
            if (dot(normal, oldNormal) < 0.9 || abs(dot(sample.position - position, normal)) > tolerance ||
                abs(dot(sample.position - position, oldNormal)) > tolerance) { continue; }

            sum += weight * f16tof32(sample.history.x & 0xffffu);
            sampleCount += weight * float(sample.history.x >> 16u);
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
    currHistory[id.xy] = uint2((count << 16u) | (f32tof16(result) & 0xffffu), asuint(rawDepth));
    shadowTex[uint3(id.xy, layer)] = result;
}

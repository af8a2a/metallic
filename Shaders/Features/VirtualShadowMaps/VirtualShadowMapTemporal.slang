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
    float3 rightAspect;
    float2 viewportExtent;
    float2 invViewport;
    float viewportAspect;
    float tanHalfFov;
    float halfOrthoHeight;
    float nearPlane;
    float farPlane;
    float farMinusNear;
    float nearTimesFar;
    float2 jitter;
    float toleranceScale;
    float orthoTolerance;
    float2 ndcScale;
    float2 ndcBias;
    uint isOrtho;
    uint reversed;
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
    b.rightAspect = b.right * view.current.viewport.x;
    b.tanHalfFov = tan(view.current.viewport.w * 0.5);
    b.isOrtho = view.current.upProjection.w > 0.5 ? 1u : 0u;
    b.halfOrthoHeight = view.current.clipOrtho.z * 0.5;
    b.nearPlane = view.current.clipOrtho.x;
    b.farPlane = view.current.clipOrtho.y;
    b.farMinusNear = view.current.clipOrtho.y - view.current.clipOrtho.x;
    b.nearTimesFar = b.nearPlane * b.farPlane;
    b.toleranceScale = (b.isOrtho != 0u) ? 1e-5 : ((2.0 * b.tanHalfFov / b.viewportExtent.y) * 0.01 + 1e-5);
    b.orthoTolerance = (b.isOrtho != 0u) ? (b.halfOrthoHeight * 2.0 / b.viewportExtent.y * 0.01) : 0.0;
    b.ndcScale = float2(2.0, -2.0) * b.invViewport;
    b.reversed = view.current.clipOrtho.w > 0.5 ? 1u : 0u;
    b.jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    b.ndcBias = fma(0.5 + b.jitter, b.ndcScale, float2(-1.0, 1.0));
    return b;
}

bool temporalReceiver(int2 pixel, TemporalViewBasis basis, float depth, out float3 position, out float linearDepth)
{
    position = float3(0.0); linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= int2(basis.viewportExtent))) { return false; }
    if (depth == (basis.reversed != 0u ? 0.0 : 1.0)) { return false; }
    if (basis.isOrtho != 0u) {
        linearDepth = basis.reversed != 0u ? lerp(basis.farPlane, basis.nearPlane, depth) : lerp(basis.nearPlane, basis.farPlane, depth);
    } else {
        float denominator = basis.reversed != 0u ? fma(depth, basis.farMinusNear, basis.nearPlane) : fma(-depth, basis.farMinusNear, basis.farPlane);
        linearDepth = basis.nearTimesFar / denominator;
    }
    float2 ndc = fma(float2(pixel), basis.ndcScale, basis.ndcBias);
    float halfHeight = basis.isOrtho != 0u ? basis.halfOrthoHeight : linearDepth * basis.tanHalfFov;
    position = basis.eye + basis.forward * linearDepth +
        halfHeight * (basis.rightAspect * ndc.x + basis.up * ndc.y);
    return true;
}

struct HistorySample
{
    bool valid;
    uint2 history;
    float3 position;
    float linearDepth;
};

HistorySample reconstructHistorySample(int2 pixel, uint2 stored, bool inBounds, TemporalViewBasis basis)
{
    HistorySample s;
    s.valid = false; s.history = uint2(0u); s.position = float3(0.0); s.linearDepth = 0.0;
    if (!inBounds || (stored.x >> 16u) == 0u) { return s; }
    float depth = asfloat(stored.y);
    s.valid = temporalReceiver(pixel, basis, depth, s.position, s.linearDepth);
    s.history = stored;
    return s;
}

HistorySample fetchHistorySample(int2 pixel, TemporalViewBasis basis, RWTexture2D<uint2> prevHistory)
{
    bool inBounds = all(pixel >= 0) && all(pixel < int2(basis.viewportExtent));
    uint2 stored = inBounds ? prevHistory[pixel] : uint2(0u);
    return reconstructHistorySample(pixel, stored, inBounds, basis);
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

bool temporalProject(float3 position, TemporalViewBasis basis, out float2 pixel)
{
    float3 delta = position - basis.eye;
    float z = dot(delta, basis.forward);
    pixel = float2(0.0);
    if (z <= basis.nearPlane || z >= basis.farPlane) { return false; }
    float halfHeight = basis.isOrtho != 0u ? basis.halfOrthoHeight : z * basis.tanHalfFov;
    float invHalfHeight = 1.0 / halfHeight;
    float2 ndc = float2(dot(delta, basis.right) * (invHalfHeight / basis.viewportAspect), dot(delta, basis.up) * invHalfHeight);
    float2 uv = float2(fma(ndc.x, 0.5, 0.5), fma(-ndc.y, 0.5, 0.5));
    pixel = uv * basis.viewportExtent - 0.5 - basis.jitter;
    return all(pixel >= -0.5) && all(pixel < basis.viewportExtent - 0.5);
}

groupshared float4 gTemporalPositions[10][10];
groupshared float gTemporalRawDepth[10][10];

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmTemporalMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    TemporalViewBasis current = createTemporalViewBasis(params.views.load(0));
    Texture2D<float> depthTex = resolveUniform(params.depth);

    int2 tileOrigin = int2(group.xy * 8u) - int2(1, 1);
    for (uint i = lane; i < 100u; i += 64u) {
        uint ky = (i * 205u) >> 11u;
        uint kx = i - ky * 10u;
        int2 pixel = tileOrigin + int2(kx, ky);
        float depth = 1.0;
        float3 pos = float3(0.0);
        float z = -1.0;
        if (all(pixel >= 0) && all(pixel < int2(current.viewportExtent))) {
            depth = depthTex.Load(int3(pixel, 0));
            float lz;
            if (temporalReceiver(pixel, current, depth, pos, lz)) {
                z = lz;
            }
        }
        gTemporalPositions[ky][kx] = float4(pos, z);
        gTemporalRawDepth[ky][kx] = depth;
    }
    GroupMemoryBarrierWithGroupSync();

    if (id.x >= params.width || id.y >= params.height) { return; }

    uint localX = lane & 7u;
    uint localY = lane >> 3u;
    float4 mySurface = gTemporalPositions[localY + 1u][localX + 1u];
    float rawDepth = gTemporalRawDepth[localY + 1u][localX + 1u];
    RWTexture2D<uint2> currHistory = resolveUniform(params.currentHistory);
    RWTexture2DArray<float> shadowTex = resolveUniform(params.shadows);

    if (mySurface.w < 0.0) {
        currHistory[id.xy] = uint2(0u, asuint(rawDepth));
        return;
    }

    float3 position = mySurface.xyz;
    float linearDepth = mySurface.w;

    uint face = params.lightType == 2u ? virtualShadowMapPointFace(position - float3(params.lightX, params.lightY, params.lightZ)) : 0u;
    uint layer = params.baseLayer + (params.compact != 0u ? 0u : face);
    float raw = shadowTex[uint3(id.xy, layer)];
    float result = raw;
    uint count = 1u;


    if (params.hasHistory != 0u) {
        TemporalViewBasis previous = createTemporalViewBasis(params.views.load(1));
        float2 previousPixel;
        if (temporalProject(position, previous, previousPixel)) {
        float4 sLeft = gTemporalPositions[localY + 1u][localX];
        float4 sRight = gTemporalPositions[localY + 1u][localX + 2u];
        float4 sTop = gTemporalPositions[localY][localX + 1u];
        float4 sBottom = gTemporalPositions[localY + 2u][localX + 1u];

        bool hasLeft = sLeft.w >= 0.0;
        bool hasRight = sRight.w >= 0.0;
        bool hasTop = sTop.w >= 0.0;
        bool hasBottom = sBottom.w >= 0.0;

        bool useLeft = hasLeft && (!hasRight || abs(sLeft.w - linearDepth) < abs(sRight.w - linearDepth));
        bool useTop = hasTop && (!hasBottom || abs(sTop.w - linearDepth) < abs(sBottom.w - linearDepth));
        float3 dx = useLeft ? position - sLeft.xyz : (hasRight ? sRight.xyz - position : float3(0.0));
        float3 dy = useTop ? position - sTop.xyz : (hasBottom ? sBottom.xyz - position : float3(0.0));
        float3 normal = cross(dy, dx);
        float lengthSquared = dot(normal, normal);
        normal = lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);

        float tolerance = max(1e-4, linearDepth * current.toleranceScale + current.orthoTolerance);
        int2 base = int2(floor(previousPixel));
        float2 fraction = frac(previousPixel);

        RWTexture2D<uint2> prevHistory = resolveUniform(params.previousHistory);

        int2 p0 = base + int2(0, 0);
        int2 p1 = base + int2(1, 0);
        int2 p2 = base + int2(0, 1);
        int2 p3 = base + int2(1, 1);
        int2 vpExtent = int2(previous.viewportExtent);

        bool inBounds0 = all(p0 >= 0) && all(p0 < vpExtent);
        bool inBounds1 = all(p1 >= 0) && all(p1 < vpExtent);
        bool inBounds2 = all(p2 >= 0) && all(p2 < vpExtent);
        bool inBounds3 = all(p3 >= 0) && all(p3 < vpExtent);

        // Issue all 4 core history memory loads concurrently to hide latency
        uint2 s0 = inBounds0 ? prevHistory[p0] : uint2(0u);
        uint2 s1 = inBounds1 ? prevHistory[p1] : uint2(0u);
        uint2 s2 = inBounds2 ? prevHistory[p2] : uint2(0u);
        uint2 s3 = inBounds3 ? prevHistory[p3] : uint2(0u);

        HistorySample core[4];
        core[0] = reconstructHistorySample(p0, s0, inBounds0, previous);
        core[1] = reconstructHistorySample(p1, s1, inBounds1, previous);
        core[2] = reconstructHistorySample(p2, s2, inBounds2, previous);
        core[3] = reconstructHistorySample(p3, s3, inBounds3, previous);

        float2 f0 = 1.0 - fraction, f1 = fraction;
        float4 tapWeights = float4(f0.x * f0.y, f1.x * f0.y, f0.x * f1.y, f1.x * f1.y);
        float sum = 0.0, weightSum = 0.0, sampleCount = 0.0;
        [unroll]
        for (uint tap = 0u; tap < 4u; ++tap) {
            float weight = tapWeights[tap];
            if (weight <= 0.0) { continue; }
            HistorySample sample = core[tap];
            if (!sample.valid) { continue; }

            float3 deltaPos = sample.position - position;
            if (abs(dot(deltaPos, normal)) > tolerance) { continue; }

            int2 offset = int2(tap & 1u, tap >> 1u);
            int2 pixel = base + offset;
            HistorySample sLeftHist, sRightHist, sTopHist, sBottomHist;
            if (tap == 0u) {
                int2 pL = pixel - int2(1, 0);
                int2 pT = pixel - int2(0, 1);
                bool inL = all(pL >= 0) && all(pL < vpExtent);
                bool inT = all(pT >= 0) && all(pT < vpExtent);
                uint2 vL = inL ? prevHistory[pL] : uint2(0u);
                uint2 vT = inT ? prevHistory[pT] : uint2(0u);
                sLeftHist = reconstructHistorySample(pL, vL, inL, previous);
                sRightHist = core[1];
                sTopHist = reconstructHistorySample(pT, vT, inT, previous);
                sBottomHist = core[2];
            } else if (tap == 1u) {
                int2 pR = pixel + int2(1, 0);
                int2 pT = pixel - int2(0, 1);
                bool inR = all(pR >= 0) && all(pR < vpExtent);
                bool inT = all(pT >= 0) && all(pT < vpExtent);
                uint2 vR = inR ? prevHistory[pR] : uint2(0u);
                uint2 vT = inT ? prevHistory[pT] : uint2(0u);
                sLeftHist = core[0];
                sRightHist = reconstructHistorySample(pR, vR, inR, previous);
                sTopHist = reconstructHistorySample(pT, vT, inT, previous);
                sBottomHist = core[3];
            } else if (tap == 2u) {
                int2 pL = pixel - int2(1, 0);
                int2 pB = pixel + int2(0, 1);
                bool inL = all(pL >= 0) && all(pL < vpExtent);
                bool inB = all(pB >= 0) && all(pB < vpExtent);
                uint2 vL = inL ? prevHistory[pL] : uint2(0u);
                uint2 vB = inB ? prevHistory[pB] : uint2(0u);
                sLeftHist = reconstructHistorySample(pL, vL, inL, previous);
                sRightHist = core[3];
                sTopHist = core[0];
                sBottomHist = reconstructHistorySample(pB, vB, inB, previous);
            } else {
                int2 pR = pixel + int2(1, 0);
                int2 pB = pixel + int2(0, 1);
                bool inR = all(pR >= 0) && all(pR < vpExtent);
                bool inB = all(pB >= 0) && all(pB < vpExtent);
                uint2 vR = inR ? prevHistory[pR] : uint2(0u);
                uint2 vB = inB ? prevHistory[pB] : uint2(0u);
                sLeftHist = core[2];
                sRightHist = reconstructHistorySample(pR, vR, inR, previous);
                sTopHist = core[1];
                sBottomHist = reconstructHistorySample(pB, vB, inB, previous);
            }

            float3 oldNormal = temporalNormalFromNeighbors(sLeftHist, sRightHist, sTopHist, sBottomHist, sample.position, sample.linearDepth);
            if (dot(normal, oldNormal) < 0.9 || abs(dot(deltaPos, oldNormal)) > tolerance) { continue; }

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
    }
    currHistory[id.xy] = uint2((count << 16u) | (f32tof16(result) & 0xffffu), asuint(rawDepth));
    shadowTex[uint3(id.xy, layer)] = result;
}
#language slang 2026
module VirtualShadowMapTemporal;

import ShaderCore;
import VirtualShadowMapCommon;
using Metallic;
using Metallic.Shadow;

#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapTemporal.h"
#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapLODInterop.h"
[[vk::push_constant]] ConstantBuffer<VirtualShadowMapTemporalParams> params;
[SpecializationConstant]
const bool kTemporalIsOrtho = false;
[SpecializationConstant]
const bool kTemporalHasHistory = true;
static const uint kMaxReprojectedHistoryFrames = 8u;

struct TemporalViewBasis
{
    float3 eye; float nearTimesFar;
    float3 forward; float depthParamA;
    float3 right; float depthParamB;
    float3 up; float clearDepth;
    float3 originOrRayBase; float toleranceScale;
    float3 dirX; float orthoTolerance;
    float3 dirY; float nearPlane;
    float2 viewportExtent; float projScaleX; float projScaleY;
    float2 jitter; float farPlane; uint isOrtho;
};

TemporalViewBasis createTemporalViewBasis(BufferSpan<ViewConstants> views, uint index)
{
    ByteAddressBuffer buffer = resolveUniform(views.resource);
    uint baseOffset = views.byteOffset + index * sizeof(ViewConstants);

    float4 eye = asfloat(buffer.Load4(baseOffset + 0u));
    float4 center = asfloat(buffer.Load4(baseOffset + 16u));
    float4 upProjection = asfloat(buffer.Load4(baseOffset + 32u));
    float4 viewport = asfloat(buffer.Load4(baseOffset + 48u));
    float4 clipOrtho = asfloat(buffer.Load4(baseOffset + 64u));
    float4 jitter = asfloat(buffer.Load4(baseOffset + 160u));
    uint4 frame = buffer.Load4(baseOffset + 176u);

    TemporalViewBasis b;
    b.eye = eye.xyz;
    b.forward = normalize(center.xyz - eye.xyz);
    b.right = normalize(cross(b.forward, upProjection.xyz));
    b.up = cross(b.right, b.forward);
    float3 rightAspect = b.right * viewport.x;
    b.viewportExtent = viewport.yz;
    float2 invViewport = 1.0 / viewport.yz;
    float tanHalfFov = tan(viewport.w * 0.5);
    b.isOrtho = kTemporalIsOrtho ? 1u : 0u;
    float halfOrthoHeight = clipOrtho.z * 0.5;
    b.nearPlane = clipOrtho.x;
    b.farPlane = clipOrtho.y;
    float farMinusNear = clipOrtho.y - clipOrtho.x;
    b.nearTimesFar = b.nearPlane * b.farPlane;
    b.toleranceScale = kTemporalIsOrtho ? 1e-5 : ((2.0 * tanHalfFov * invViewport.y) * 0.5 + 1e-5);
    b.orthoTolerance = kTemporalIsOrtho ? (halfOrthoHeight * 2.0 * invViewport.y * 0.5) : 0.0;
    float2 ndcScale = float2(2.0, -2.0) * invViewport;
    bool reversed = clipOrtho.w > 0.5;
    b.clearDepth = reversed ? 0.0 : 1.0;
    b.jitter = frame.z != 0u ? jitter.xy : float2(0.0);
    float2 ndcBias = fma(0.5 + b.jitter, ndcScale, float2(-1.0, 1.0));

    if (kTemporalIsOrtho) {
        b.depthParamA = reversed ? -farMinusNear : farMinusNear;
        b.depthParamB = reversed ? b.farPlane : b.nearPlane;
        float invHalfOrthoHeight = 1.0 / halfOrthoHeight;
        b.projScaleX = invHalfOrthoHeight / viewport.x;
        b.projScaleY = invHalfOrthoHeight;
        b.originOrRayBase = b.eye + (halfOrthoHeight * ndcBias.x) * rightAspect + (halfOrthoHeight * ndcBias.y) * b.up;
        b.dirX = (halfOrthoHeight * ndcScale.x) * rightAspect;
        b.dirY = (halfOrthoHeight * ndcScale.y) * b.up;
    } else {
        b.depthParamA = reversed ? farMinusNear : -farMinusNear;
        b.depthParamB = reversed ? b.nearPlane : b.farPlane;
        b.projScaleX = 1.0 / (tanHalfFov * viewport.x);
        b.projScaleY = 1.0 / tanHalfFov;
        b.originOrRayBase = b.forward + (tanHalfFov * ndcBias.x) * rightAspect + (tanHalfFov * ndcBias.y) * b.up;
        b.dirX = (tanHalfFov * ndcScale.x) * rightAspect;
        b.dirY = (tanHalfFov * ndcScale.y) * b.up;
    }
    return b;
}

[ForceInline]
float vsmTemporalLinearDepth(float depth, TemporalViewBasis basis)
{
    float d = fma(depth, basis.depthParamA, basis.depthParamB);
    return kTemporalIsOrtho ? d : (basis.nearTimesFar / d);
}

[ForceInline]
float3 vsmTemporalPosition(int2 pixel, TemporalViewBasis basis, float linearDepth)
{
    float3 offset = basis.originOrRayBase + float(pixel.x) * basis.dirX + float(pixel.y) * basis.dirY;
    return kTemporalIsOrtho ? (offset + basis.forward * linearDepth) : (basis.eye + linearDepth * offset);
}

[ForceInline]
bool temporalReceiver(int2 pixel, TemporalViewBasis basis, float depth, out float3 position, out float linearDepth)
{
    position = float3(0.0); linearDepth = 0.0;
    if (depth == basis.clearDepth) { return false; }
    linearDepth = vsmTemporalLinearDepth(depth, basis);
    position = vsmTemporalPosition(pixel, basis, linearDepth);
    return true;
}

[ForceInline]
uint packTemporalOctNormal(float3 n)
{
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    if (!(l1 > 1e-6)) { return 0x3fffu; }
    float2 oct = n.xy * (1.0 / l1);
    if (n.z < 0.0) {
        oct = (1.0 - abs(oct.yx)) * float2(oct.x >= 0.0 ? 1.0 : -1.0, oct.y >= 0.0 ? 1.0 : -1.0);
    }
    uint2 q = uint2(round(saturate(oct * 0.5 + 0.5) * 126.0));
    return (q.x << 7u) | q.y;
}

[ForceInline]
float3 unpackTemporalOctNormal(uint packed)
{
    if (packed == 0x3fffu) { return float3(0.0); }
    float2 oct = float2(float((packed >> 7u) & 0x7fu), float(packed & 0x7fu)) * (2.0 / 126.0) - 1.0;
    float3 n = float3(oct.x, oct.y, 1.0 - abs(oct.x) - abs(oct.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

[ForceInline]
uint packTemporalHistoryWord(uint count, float visibility, uint octNormal)
{
    uint visQ = uint(round(saturate(visibility) * 2047.0));
    return (count << 25u) | (visQ << 14u) | octNormal;
}

[ForceInline]
void accumulateTemporalHistoryTap(int2 p, float weight, int2 vpExtent,
    RWTexture2D<uint2> prevHistory, TemporalViewBasis previous,
    float3 position, float3 normal, float tolerance,
    inout float sum, inout float sampleCount, inout float weightSum)
{
    if (weight <= 0.0 || any(p < 0) || any(p >= vpExtent)) { return; }
    uint2 stored = prevHistory[p];
    uint priorSamples = stored.x >> 25u;
    if (priorSamples == 0u) { return; }
    float linearDepth = vsmTemporalLinearDepth(asfloat(stored.y), previous);
    float3 oldPos = vsmTemporalPosition(p, previous, linearDepth);
    float3 delta = oldPos - position;
    if (abs(dot(delta, normal)) > tolerance) { return; }
    float3 oldNormal = unpackTemporalOctNormal(stored.x & 0x3fffu);
    if (dot(normal, oldNormal) < 0.9 || abs(dot(delta, oldNormal)) > tolerance) { return; }
    float oldVis = float((stored.x >> 14u) & 0x7ffu) * (1.0 / 2047.0);
    sum += weight * oldVis;
    sampleCount += weight * float(priorSamples);
    weightSum += weight;
}

[ForceInline]
float3 temporalNormalFromNeighbors(float4 left, float4 right, float4 top, float4 bottom, float4 center)
{
    bool hasLeft = left.w >= 0.0, hasRight = right.w >= 0.0;
    bool hasTop = top.w >= 0.0, hasBottom = bottom.w >= 0.0;
    bool useLeft = hasLeft && (!hasRight || abs(left.w - center.w) < abs(right.w - center.w));
    bool useTop = hasTop && (!hasBottom || abs(top.w - center.w) < abs(bottom.w - center.w));
    float3 dx = useLeft ? center.xyz - left.xyz : (hasRight ? right.xyz - center.xyz : float3(0.0));
    float3 dy = useTop ? center.xyz - top.xyz : (hasBottom ? bottom.xyz - center.xyz : float3(0.0));
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
    float2 projScale = float2(basis.projScaleX, basis.projScaleY);
    float2 ndcScaleFactors = kTemporalIsOrtho ? projScale : ((1.0 / z) * projScale);
    float2 ndc = float2(dot(delta, basis.right), dot(delta, basis.up)) * ndcScaleFactors;
    float2 uv = fma(ndc, float2(0.5, -0.5), float2(0.5, 0.5));
    pixel = fma(uv, basis.viewportExtent, -0.5 - basis.jitter);
    return all(pixel >= -0.5) && all(pixel < basis.viewportExtent - 0.5);
}

groupshared TemporalViewBasis gTemporalViews[2];
groupshared float4 gTemporalPositions[100];
groupshared float gTemporalRawDepth[100];
groupshared uint gTemporalLODChanged;

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmTemporalMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    if (lane == 0u) {
        gTemporalLODChanged = 0u;
        if (kTemporalHasHistory) {
            for (uint index = 0u; index < params.lodCaches.count; ++index) {
                StructuredBuffer<uint> cache = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(params.lodCaches.load(index)));
                gTemporalLODChanged |= cache[29u];
                StructuredBuffer<VirtualShadowMapLODContextGPU> contexts =
                    resolveUniform(ResourceHandle<StructuredBuffer<VirtualShadowMapLODContextGPU>>(cache[50u]));
                // A disabled/rebuilt page cache has no committed cut to compare.
                gTemporalLODChanged |= contexts[0].previousValid == 0u ? 1u : 0u;
            }
        }
    }
    if (lane < (kTemporalHasHistory ? 2u : 1u)) {
        gTemporalViews[lane] = createTemporalViewBasis(params.views, lane);
    }
    GroupMemoryBarrierWithGroupSync();
    TemporalViewBasis current = gTemporalViews[0];
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
        gTemporalPositions[i] = float4(pos, z);
        gTemporalRawDepth[i] = depth;
    }
    GroupMemoryBarrierWithGroupSync();

    if (id.x >= params.width || id.y >= params.height) { return; }

    uint localX = lane & 7u;
    uint localY = lane >> 3u;
    uint centerIndex = (localY + 1u) * 10u + (localX + 1u);
    float4 mySurface = gTemporalPositions[centerIndex];
    float rawDepth = gTemporalRawDepth[centerIndex];
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

    float3 normal = temporalNormalFromNeighbors(
        gTemporalPositions[centerIndex - 1u], gTemporalPositions[centerIndex + 1u],
        gTemporalPositions[centerIndex - 10u], gTemporalPositions[centerIndex + 10u], mySurface);

    if (kTemporalHasHistory && params.hasHistory != 0u && gTemporalLODChanged == 0u) {
        TemporalViewBasis previous = gTemporalViews[1];
        // History eligibility already matches the projection. With a fixed camera,
        // accumulate jittered samples per pixel to avoid repeatedly blurring shadow edges.
        bool sameCamera = all(previous.eye == current.eye) &&
            all(previous.forward == current.forward) &&
            all(previous.right == current.right);
        float tolerance = max(1e-4, linearDepth * current.toleranceScale + current.orthoTolerance);
        RWTexture2D<uint2> prevHistory = resolveUniform(params.previousHistory);
        if (sameCamera) {
            if (linearDepth > previous.nearPlane && linearDepth < previous.farPlane) {
                uint2 stored = prevHistory[int2(id.xy)];
                uint priorSamples = stored.x >> 25u;
                if (priorSamples != 0u) {
                    bool accepted = false;
                    float3 oldNormal = unpackTemporalOctNormal(stored.x & 0x3fffu);
                    if (dot(normal, oldNormal) >= 0.9) {
                        if (stored.y == asuint(rawDepth) && all(previous.jitter == current.jitter)) {
                            accepted = true;
                        } else {
                            float oldLinear = vsmTemporalLinearDepth(asfloat(stored.y), previous);
                            float3 delta = vsmTemporalPosition(int2(id.xy), previous, oldLinear) - position;
                            accepted = abs(dot(delta, normal)) <= tolerance && abs(dot(delta, oldNormal)) <= tolerance;
                        }
                    }
                    if (accepted) {
                        uint priorCount = min(priorSamples, params.maxFrames - 1u);
                        count = priorCount + 1u;
                        float oldVis = float((stored.x >> 14u) & 0x7ffu) * (1.0 / 2047.0);
                        result = (oldVis * float(priorCount) + raw) / float(count);
                    }
                }
            }
        } else {
            float2 previousPixel;
            if (temporalProject(position, previous, previousPixel)) {
                int2 base = int2(floor(previousPixel));
                float2 fraction = frac(previousPixel);
                float2 f0 = 1.0 - fraction, f1 = fraction;
                float4 tapWeights = float4(f0.x * f0.y, f1.x * f0.y, f0.x * f1.y, f1.x * f1.y);
                int2 vpExtent = int2(previous.viewportExtent);

                float sum = 0.0, weightSum = 0.0, sampleCount = 0.0;
                accumulateTemporalHistoryTap(base + int2(0, 0), tapWeights.x, vpExtent,
                    prevHistory, previous, position, normal, tolerance, sum, sampleCount, weightSum);
                accumulateTemporalHistoryTap(base + int2(1, 0), tapWeights.y, vpExtent,
                    prevHistory, previous, position, normal, tolerance, sum, sampleCount, weightSum);
                accumulateTemporalHistoryTap(base + int2(0, 1), tapWeights.z, vpExtent,
                    prevHistory, previous, position, normal, tolerance, sum, sampleCount, weightSum);
                accumulateTemporalHistoryTap(base + int2(1, 1), tapWeights.w, vpExtent,
                    prevHistory, previous, position, normal, tolerance, sum, sampleCount, weightSum);
                if (weightSum >= 0.5) {
                    // Repeated bilinear resampling widens the penumbra on moving views.
                    uint historyFrames = min(params.maxFrames, kMaxReprojectedHistoryFrames);
                    uint priorCount = min(uint(sampleCount / weightSum + 0.5), historyFrames - 1u);
                    count = priorCount + 1u;
                    result = (sum / weightSum * float(priorCount) + raw) / float(count);
                }
            }
        }
    }
    currHistory[id.xy] = uint2(packTemporalHistoryWord(count, result, packTemporalOctNormal(normal)), asuint(rawDepth));
    shadowTex[uint3(id.xy, layer)] = result;
}

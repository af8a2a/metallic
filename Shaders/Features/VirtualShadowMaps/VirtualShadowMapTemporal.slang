#language slang 2026
module VirtualShadowMapTemporal;

import ShaderCore;
import VirtualShadowMapCommon;
using Metallic;
using Metallic.Shadow;

#include "../../../Source/Runtime/Render/Shadow/VirtualShadowMapTemporal.h"
[[vk::push_constant]] ConstantBuffer<VirtualShadowMapTemporalParams> params;
[SpecializationConstant]
const bool kTemporalIsOrtho = false;
[SpecializationConstant]
const bool kTemporalHasHistory = true;

VSMTemporalViewBasis loadTemporalViewBasis(BufferSpan<VSMTemporalViewBasis> views, uint index)
{
    ByteAddressBuffer buffer = resolveUniform(views.resource);
    uint baseOffset = views.byteOffset + index * sizeof(VSMTemporalViewBasis);
    float4 r0 = asfloat(buffer.Load4(baseOffset + 0u));
    float4 r1 = asfloat(buffer.Load4(baseOffset + 16u));
    float4 r2 = asfloat(buffer.Load4(baseOffset + 32u));
    float4 r3 = asfloat(buffer.Load4(baseOffset + 48u));
    float4 r4 = asfloat(buffer.Load4(baseOffset + 64u));
    float4 r5 = asfloat(buffer.Load4(baseOffset + 80u));
    float4 r6 = asfloat(buffer.Load4(baseOffset + 96u));
    float4 r7 = asfloat(buffer.Load4(baseOffset + 112u));
    uint4 r8 = buffer.Load4(baseOffset + 128u);
    VSMTemporalViewBasis b;
    b.eye = r0.xyz; b.nearTimesFar = r0.w;
    b.forward = r1.xyz; b.depthParamA = r1.w;
    b.right = r2.xyz; b.depthParamB = r2.w;
    b.up = r3.xyz; b.clearDepth = r3.w;
    b.originOrRayBase = r4.xyz; b.toleranceScale = r4.w;
    b.dirX = r5.xyz; b.orthoTolerance = r5.w;
    b.dirY = r6.xyz; b.nearPlane = r6.w;
    b.viewportExtent = r7.xy; b.projScaleX = r7.z; b.projScaleY = r7.w;
    b.jitter = asfloat(r8.xy); b.farPlane = asfloat(r8.z); b.isOrtho = r8.w;
    return b;
}

[ForceInline]
float vsmTemporalLinearDepth(float depth, VSMTemporalViewBasis basis)
{
    float d = fma(depth, basis.depthParamA, basis.depthParamB);
    return kTemporalIsOrtho ? d : (basis.nearTimesFar / d);
}

[ForceInline]
float3 vsmTemporalPosition(int2 pixel, VSMTemporalViewBasis basis, float linearDepth)
{
    float3 offset = basis.originOrRayBase + float(pixel.x) * basis.dirX + float(pixel.y) * basis.dirY;
    return kTemporalIsOrtho ? (offset + basis.forward * linearDepth) : (basis.eye + linearDepth * offset);
}

[ForceInline]
bool temporalReceiver(int2 pixel, VSMTemporalViewBasis basis, float depth, out float3 position, out float linearDepth)
{
    position = float3(0.0); linearDepth = 0.0;
    if (depth == basis.clearDepth) { return false; }
    linearDepth = vsmTemporalLinearDepth(depth, basis);
    position = vsmTemporalPosition(pixel, basis, linearDepth);
    return true;
}

// Pack position.xyz and linearDepth (w < 0 indicates invalid) to keep live VGPRs compact.
[ForceInline]
float4 reconstructHistoryPosDepth(int2 pixel, uint2 stored, bool inBounds, VSMTemporalViewBasis basis)
{
    if (!inBounds || (stored.x >> 16u) == 0u) { return float4(0.0, 0.0, 0.0, -1.0); }
    float linearDepth = vsmTemporalLinearDepth(asfloat(stored.y), basis);
    return float4(vsmTemporalPosition(pixel, basis, linearDepth), linearDepth);
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

bool temporalProject(float3 position, VSMTemporalViewBasis basis, out float2 pixel)
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

groupshared float4 gTemporalPositions[100];
groupshared float gTemporalRawDepth[100];

[shader("compute")]
[numthreads(8, 8, 1)]
void vsmTemporalMain(uint3 id : SV_DispatchThreadID, uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    VSMTemporalViewBasis current = loadTemporalViewBasis(params.views, 0u);
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

    if (kTemporalHasHistory && params.hasHistory != 0u) {
        VSMTemporalViewBasis previous = loadTemporalViewBasis(params.views, 1u);
        float2 previousPixel;
        if (temporalProject(position, previous, previousPixel)) {
            float4 sLeft = gTemporalPositions[centerIndex - 1u];
            float4 sRight = gTemporalPositions[centerIndex + 1u];
            float4 sTop = gTemporalPositions[centerIndex - 10u];
            float4 sBottom = gTemporalPositions[centerIndex + 10u];

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

            uint2 s0 = inBounds0 ? prevHistory[p0] : uint2(0u);
            uint2 s1 = inBounds1 ? prevHistory[p1] : uint2(0u);
            uint2 s2 = inBounds2 ? prevHistory[p2] : uint2(0u);
            bool needCore3 = fraction.x != 0.0 || fraction.y != 0.0;
            uint2 s3 = (needCore3 && inBounds3) ? prevHistory[p3] : uint2(0u);

            float4 core0 = reconstructHistoryPosDepth(p0, s0, inBounds0, previous);
            float4 core1 = reconstructHistoryPosDepth(p1, s1, inBounds1, previous);
            float4 core2 = reconstructHistoryPosDepth(p2, s2, inBounds2, previous);
            float4 core3 = needCore3 ? reconstructHistoryPosDepth(p3, s3, inBounds3, previous) : float4(0.0, 0.0, 0.0, -1.0);

            float2 f0 = 1.0 - fraction, f1 = fraction;
            float4 tapWeights = float4(f0.x * f0.y, f1.x * f0.y, f0.x * f1.y, f1.x * f1.y);

            bool tapNeeded0 = tapWeights.x > 0.0 && core0.w >= 0.0 && (abs(dot(core0.xyz - position, normal)) <= tolerance);
            bool tapNeeded1 = tapWeights.y > 0.0 && core1.w >= 0.0 && (abs(dot(core1.xyz - position, normal)) <= tolerance);
            bool tapNeeded2 = tapWeights.z > 0.0 && core2.w >= 0.0 && (abs(dot(core2.xyz - position, normal)) <= tolerance);
            bool tapNeeded3 = tapWeights.w > 0.0 && core3.w >= 0.0 && (abs(dot(core3.xyz - position, normal)) <= tolerance);

            if (tapNeeded0 || tapNeeded1 || tapNeeded2 || tapNeeded3) {
                int2 pL0 = base + int2(-1, 0), pT0 = base + int2(0, -1);
                int2 pR0 = base + int2(2, 0),  pT1 = base + int2(1, -1);
                int2 pL1 = base + int2(-1, 1), pB0 = base + int2(0, 2);
                int2 pR1 = base + int2(2, 1),  pB1 = base + int2(1, 2);
                bool inL0 = all(pL0 >= 0) && all(pL0 < vpExtent);
                bool inT0 = all(pT0 >= 0) && all(pT0 < vpExtent);
                bool inR0 = all(pR0 >= 0) && all(pR0 < vpExtent);
                bool inT1 = all(pT1 >= 0) && all(pT1 < vpExtent);
                bool inL1 = all(pL1 >= 0) && all(pL1 < vpExtent);
                bool inB0 = all(pB0 >= 0) && all(pB0 < vpExtent);
                bool inR1 = all(pR1 >= 0) && all(pR1 < vpExtent);
                bool inB1 = all(pB1 >= 0) && all(pB1 < vpExtent);

                uint2 vL0 = (tapNeeded0 && inL0) ? prevHistory[pL0] : uint2(0u);
                uint2 vT0 = (tapNeeded0 && inT0) ? prevHistory[pT0] : uint2(0u);
                uint2 vR0 = (tapNeeded1 && inR0) ? prevHistory[pR0] : uint2(0u);
                uint2 vT1 = (tapNeeded1 && inT1) ? prevHistory[pT1] : uint2(0u);
                uint2 vL1 = (tapNeeded2 && inL1) ? prevHistory[pL1] : uint2(0u);
                uint2 vB0 = (tapNeeded2 && inB0) ? prevHistory[pB0] : uint2(0u);
                uint2 vR1 = (tapNeeded3 && inR1) ? prevHistory[pR1] : uint2(0u);
                uint2 vB1 = (tapNeeded3 && inB1) ? prevHistory[pB1] : uint2(0u);

                float sum = 0.0, weightSum = 0.0, sampleCount = 0.0;
                if (tapNeeded0) {
                    float3 oldNormal = temporalNormalFromNeighbors(
                        reconstructHistoryPosDepth(pL0, vL0, inL0, previous), core1,
                        reconstructHistoryPosDepth(pT0, vT0, inT0, previous), core2, core0);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(core0.xyz - position, oldNormal)) <= tolerance) {
                        sum += tapWeights.x * f16tof32(s0.x & 0xffffu);
                        sampleCount += tapWeights.x * float(s0.x >> 16u);
                        weightSum += tapWeights.x;
                    }
                }
                if (tapNeeded1) {
                    float3 oldNormal = temporalNormalFromNeighbors(
                        core0, reconstructHistoryPosDepth(pR0, vR0, inR0, previous),
                        reconstructHistoryPosDepth(pT1, vT1, inT1, previous), core3, core1);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(core1.xyz - position, oldNormal)) <= tolerance) {
                        sum += tapWeights.y * f16tof32(s1.x & 0xffffu);
                        sampleCount += tapWeights.y * float(s1.x >> 16u);
                        weightSum += tapWeights.y;
                    }
                }
                if (tapNeeded2) {
                    float3 oldNormal = temporalNormalFromNeighbors(
                        reconstructHistoryPosDepth(pL1, vL1, inL1, previous), core3,
                        core0, reconstructHistoryPosDepth(pB0, vB0, inB0, previous), core2);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(core2.xyz - position, oldNormal)) <= tolerance) {
                        sum += tapWeights.z * f16tof32(s2.x & 0xffffu);
                        sampleCount += tapWeights.z * float(s2.x >> 16u);
                        weightSum += tapWeights.z;
                    }
                }
                if (tapNeeded3) {
                    float3 oldNormal = temporalNormalFromNeighbors(
                        core2, reconstructHistoryPosDepth(pR1, vR1, inR1, previous),
                        core1, reconstructHistoryPosDepth(pB1, vB1, inB1, previous), core3);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(core3.xyz - position, oldNormal)) <= tolerance) {
                        sum += tapWeights.w * f16tof32(s3.x & 0xffffu);
                        sampleCount += tapWeights.w * float(s3.x >> 16u);
                        weightSum += tapWeights.w;
                    }
                }
                if (weightSum >= 0.5) {
                    uint priorCount = min(uint(sampleCount / weightSum + 0.5), params.maxFrames - 1u);
                    count = priorCount + 1u;
                    result = (sum / weightSum * float(priorCount) + raw) / float(count);
                }
            }
        }
    }
    currHistory[id.xy] = uint2((count << 16u) | (f32tof16(result) & 0xffffu), asuint(rawDepth));
    shadowTex[uint3(id.xy, layer)] = result;
}

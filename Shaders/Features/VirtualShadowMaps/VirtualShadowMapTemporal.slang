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
    float clearDepth;

    // Linear depth precomputed coefficients
    float depthParamA;
    float depthParamB;

    // Perspective ray direction precomputed basis
    float3 rayDirBase;
    float3 rayDirPixelX;
    float3 rayDirPixelY;
    float perspInvTanHalfFovAspect;
    float perspInvTanHalfFov;

    // Orthographic origin/directions precomputed basis
    float3 orthoOrigin;
    float3 orthoDirX;
    float3 orthoDirY;
    float orthoInvHalfHeightAspect;
    float orthoInvHalfHeight;
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
    // Skip 80 bytes of previous view camera constants (offset 80 to 159)
    float4 jitter = asfloat(buffer.Load4(baseOffset + 160u));
    uint4 frame = buffer.Load4(baseOffset + 176u);
    // Skip 16 bytes of outputSize (offset 192 to 207)

    TemporalViewBasis b;
    b.eye = eye.xyz;
    b.forward = normalize(center.xyz - eye.xyz);
    b.right = normalize(cross(b.forward, upProjection.xyz));
    b.up = cross(b.right, b.forward);
    b.viewportExtent = viewport.yz;
    b.invViewport = 1.0 / viewport.yz;
    b.viewportAspect = viewport.x;
    b.rightAspect = b.right * viewport.x;
    b.tanHalfFov = tan(viewport.w * 0.5);
    b.isOrtho = upProjection.w > 0.5 ? 1u : 0u;
    b.halfOrthoHeight = clipOrtho.z * 0.5;
    b.nearPlane = clipOrtho.x;
    b.farPlane = clipOrtho.y;
    b.farMinusNear = clipOrtho.y - clipOrtho.x;
    b.nearTimesFar = b.nearPlane * b.farPlane;
    b.toleranceScale = (b.isOrtho != 0u) ? 1e-5 : ((2.0 * b.tanHalfFov / b.viewportExtent.y) * 0.01 + 1e-5);
    b.orthoTolerance = (b.isOrtho != 0u) ? (b.halfOrthoHeight * 2.0 / b.viewportExtent.y * 0.01) : 0.0;
    b.ndcScale = float2(2.0, -2.0) * b.invViewport;
    b.reversed = clipOrtho.w > 0.5 ? 1u : 0u;
    b.clearDepth = b.reversed != 0u ? 0.0 : 1.0;
    b.jitter = frame.z != 0u ? jitter.xy : float2(0.0);
    b.ndcBias = fma(0.5 + b.jitter, b.ndcScale, float2(-1.0, 1.0));

    if (b.isOrtho != 0u) {
        b.depthParamA = b.reversed != 0u ? -b.farMinusNear : b.farMinusNear;
        b.depthParamB = b.reversed != 0u ? b.farPlane : b.nearPlane;
        float invHalfOrthoHeight = 1.0 / b.halfOrthoHeight;
        b.orthoInvHalfHeightAspect = invHalfOrthoHeight / viewport.x;
        b.orthoInvHalfHeight = invHalfOrthoHeight;
        b.orthoOrigin = b.eye + (b.halfOrthoHeight * b.ndcBias.x) * b.rightAspect + (b.halfOrthoHeight * b.ndcBias.y) * b.up;
        b.orthoDirX = (b.halfOrthoHeight * b.ndcScale.x) * b.rightAspect;
        b.orthoDirY = (b.halfOrthoHeight * b.ndcScale.y) * b.up;
        b.rayDirBase = float3(0.0);
        b.rayDirPixelX = float3(0.0);
        b.rayDirPixelY = float3(0.0);
        b.perspInvTanHalfFovAspect = 0.0;
        b.perspInvTanHalfFov = 0.0;
    } else {
        b.depthParamA = b.reversed != 0u ? b.farMinusNear : -b.farMinusNear;
        b.depthParamB = b.reversed != 0u ? b.nearPlane : b.farPlane;
        b.perspInvTanHalfFovAspect = 1.0 / (b.tanHalfFov * viewport.x);
        b.perspInvTanHalfFov = 1.0 / b.tanHalfFov;
        b.rayDirBase = b.forward + (b.tanHalfFov * b.ndcBias.x) * b.rightAspect + (b.tanHalfFov * b.ndcBias.y) * b.up;
        b.rayDirPixelX = (b.tanHalfFov * b.ndcScale.x) * b.rightAspect;
        b.rayDirPixelY = (b.tanHalfFov * b.ndcScale.y) * b.up;
        b.orthoOrigin = float3(0.0);
        b.orthoDirX = float3(0.0);
        b.orthoDirY = float3(0.0);
        b.orthoInvHalfHeightAspect = 0.0;
        b.orthoInvHalfHeight = 0.0;
    }
    return b;
}

[ForceInline]
float vsmTemporalLinearDepth(float depth, TemporalViewBasis basis)
{
    float d = fma(depth, basis.depthParamA, basis.depthParamB);
    return basis.isOrtho != 0u ? d : (basis.nearTimesFar / d);
}

[ForceInline]
float3 vsmTemporalPosition(int2 pixel, TemporalViewBasis basis, float linearDepth)
{
    if (basis.isOrtho != 0u) {
        return basis.orthoOrigin + basis.forward * linearDepth + float(pixel.x) * basis.orthoDirX + float(pixel.y) * basis.orthoDirY;
    } else {
        float3 rayDir = basis.rayDirBase + float(pixel.x) * basis.rayDirPixelX + float(pixel.y) * basis.rayDirPixelY;
        return basis.eye + linearDepth * rayDir;
    }
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

struct HistorySample
{
    bool valid;
    uint2 history;
    float3 position;
    float linearDepth;
};

[ForceInline]
HistorySample reconstructHistorySample(int2 pixel, uint2 stored, bool inBounds, TemporalViewBasis basis)
{
    HistorySample s;
    s.valid = false; s.history = uint2(0u); s.position = float3(0.0); s.linearDepth = 0.0;
    if (!inBounds || (stored.x >> 16u) == 0u) { return s; }
    float depth = asfloat(stored.y);
    s.linearDepth = vsmTemporalLinearDepth(depth, basis);
    s.position = vsmTemporalPosition(pixel, basis, s.linearDepth);
    s.valid = true;
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

bool temporalProject(float3 position, TemporalViewBasis basis, out float2 pixel)
{
    float3 delta = position - basis.eye;
    float z = dot(delta, basis.forward);
    pixel = float2(0.0);
    if (z <= basis.nearPlane || z >= basis.farPlane) { return false; }
    float2 ndcScaleFactors = basis.isOrtho != 0u ?
        float2(basis.orthoInvHalfHeightAspect, basis.orthoInvHalfHeight) :
        (1.0 / z) * float2(basis.perspInvTanHalfFovAspect, basis.perspInvTanHalfFov);
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
    TemporalViewBasis current = createTemporalViewBasis(params.views, 0u);
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

    if (params.hasHistory != 0u) {
        TemporalViewBasis previous = createTemporalViewBasis(params.views, 1u);
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
            float3 deltaPos0 = core[0].position - position;
            float3 deltaPos1 = core[1].position - position;
            float3 deltaPos2 = core[2].position - position;
            float3 deltaPos3 = core[3].position - position;

            bool tapNeeded0 = tapWeights.x > 0.0 && core[0].valid && (abs(dot(deltaPos0, normal)) <= tolerance);
            bool tapNeeded1 = tapWeights.y > 0.0 && core[1].valid && (abs(dot(deltaPos1, normal)) <= tolerance);
            bool tapNeeded2 = tapWeights.z > 0.0 && core[2].valid && (abs(dot(deltaPos2, normal)) <= tolerance);
            bool tapNeeded3 = tapWeights.w > 0.0 && core[3].valid && (abs(dot(deltaPos3, normal)) <= tolerance);

            if (tapNeeded0 || tapNeeded1 || tapNeeded2 || tapNeeded3) {
                int2 pL0 = base + int2(-1, 0);
                int2 pT0 = base + int2(0, -1);
                int2 pR0 = base + int2(2, 0);
                int2 pT1 = base + int2(1, -1);
                int2 pL1 = base + int2(-1, 1);
                int2 pB0 = base + int2(0, 2);
                int2 pR1 = base + int2(2, 1);
                int2 pB1 = base + int2(1, 2);

                bool inL0 = all(pL0 >= 0) && all(pL0 < vpExtent);
                bool inT0 = all(pT0 >= 0) && all(pT0 < vpExtent);
                bool inR0 = all(pR0 >= 0) && all(pR0 < vpExtent);
                bool inT1 = all(pT1 >= 0) && all(pT1 < vpExtent);
                bool inL1 = all(pL1 >= 0) && all(pL1 < vpExtent);
                bool inB0 = all(pB0 >= 0) && all(pB0 < vpExtent);
                bool inR1 = all(pR1 >= 0) && all(pR1 < vpExtent);
                bool inB1 = all(pB1 >= 0) && all(pB1 < vpExtent);

                // Batch issue all required exterior history loads concurrently
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
                    HistorySample sLeftHist = reconstructHistorySample(pL0, vL0, inL0, previous);
                    HistorySample sTopHist = reconstructHistorySample(pT0, vT0, inT0, previous);
                    float3 oldNormal = temporalNormalFromNeighbors(sLeftHist, core[1], sTopHist, core[2], core[0].position, core[0].linearDepth);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(deltaPos0, oldNormal)) <= tolerance) {
                        sum += tapWeights.x * f16tof32(core[0].history.x & 0xffffu);
                        sampleCount += tapWeights.x * float(core[0].history.x >> 16u);
                        weightSum += tapWeights.x;
                    }
                }
                if (tapNeeded1) {
                    HistorySample sRightHist = reconstructHistorySample(pR0, vR0, inR0, previous);
                    HistorySample sTopHist = reconstructHistorySample(pT1, vT1, inT1, previous);
                    float3 oldNormal = temporalNormalFromNeighbors(core[0], sRightHist, sTopHist, core[3], core[1].position, core[1].linearDepth);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(deltaPos1, oldNormal)) <= tolerance) {
                        sum += tapWeights.y * f16tof32(core[1].history.x & 0xffffu);
                        sampleCount += tapWeights.y * float(core[1].history.x >> 16u);
                        weightSum += tapWeights.y;
                    }
                }
                if (tapNeeded2) {
                    HistorySample sLeftHist = reconstructHistorySample(pL1, vL1, inL1, previous);
                    HistorySample sBottomHist = reconstructHistorySample(pB0, vB0, inB0, previous);
                    float3 oldNormal = temporalNormalFromNeighbors(sLeftHist, core[3], core[0], sBottomHist, core[2].position, core[2].linearDepth);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(deltaPos2, oldNormal)) <= tolerance) {
                        sum += tapWeights.z * f16tof32(core[2].history.x & 0xffffu);
                        sampleCount += tapWeights.z * float(core[2].history.x >> 16u);
                        weightSum += tapWeights.z;
                    }
                }
                if (tapNeeded3) {
                    HistorySample sRightHist = reconstructHistorySample(pR1, vR1, inR1, previous);
                    HistorySample sBottomHist = reconstructHistorySample(pB1, vB1, inB1, previous);
                    float3 oldNormal = temporalNormalFromNeighbors(core[2], sRightHist, core[1], sBottomHist, core[3].position, core[3].linearDepth);
                    if (dot(normal, oldNormal) >= 0.9 && abs(dot(deltaPos3, oldNormal)) <= tolerance) {
                        sum += tapWeights.w * f16tof32(core[3].history.x & 0xffffu);
                        sampleCount += tapWeights.w * float(core[3].history.x >> 16u);
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

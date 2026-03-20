// GPU meshlet culling compute shader.
// One thread per meshlet across all instances.
// Outputs a compact list of visible {instanceID, globalMeshletID} pairs.

#include "../../Source/Rendering/hzb_constants.h"

struct CullUniforms {
    float4x4 viewProj;
    float4x4 prevViewProj;
    float4x4 prevView;
    float4   cameraWorldPos;
    float4   prevCameraWorldPos;
    float2   prevProjScale;
    float2   hzbTextureSize;
    uint     totalDispatchCount;
    uint     instanceCount;
    uint     enableFrustumCull;
    uint     enableConeCull;
    uint     enableOcclusionCull;
    uint     hzbLevelCount;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
};

struct InstanceData {
    float4x4 mvp;
    float4x4 modelView;
    float4x4 worldMatrix;
    uint     meshletStart;
    uint     meshletCount;
    uint     instanceID;
    uint     pad;
};

struct GPUMeshletBounds {
    float4 center_radius;
    float4 cone_apex_pad;
    float4 cone_axis_cutoff;
};

struct MeshletDrawInfo {
    uint instanceID;
    uint globalMeshletID;
};

ConstantBuffer<CullUniforms>        cullUniforms;    // buffer(0)
StructuredBuffer<InstanceData>      instances;       // buffer(1)
StructuredBuffer<GPUMeshletBounds>  meshletBounds;   // buffer(2)
RWStructuredBuffer<MeshletDrawInfo> visibleMeshlets; // buffer(3)
RWByteAddressBuffer                 counters;        // buffer(4)
Texture2D<float>                    hzbPyramid[HZB_MAX_LEVELS]; // texture(5+)

float sampleHzb(uint level, uint2 coord) {
    switch (level) {
    case 0u: return hzbPyramid[0].Load(int3(coord, 0));
    case 1u: return hzbPyramid[1].Load(int3(coord, 0));
    case 2u: return hzbPyramid[2].Load(int3(coord, 0));
    case 3u: return hzbPyramid[3].Load(int3(coord, 0));
    case 4u: return hzbPyramid[4].Load(int3(coord, 0));
    case 5u: return hzbPyramid[5].Load(int3(coord, 0));
    case 6u: return hzbPyramid[6].Load(int3(coord, 0));
    case 7u: return hzbPyramid[7].Load(int3(coord, 0));
    case 8u: return hzbPyramid[8].Load(int3(coord, 0));
    default: return hzbPyramid[9].Load(int3(coord, 0));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMain(uint3 dtid : SV_DispatchThreadID) {
    uint flatID = dtid.x;
    if (flatID >= cullUniforms.totalDispatchCount)
        return;

    // Find which instance this thread belongs to via linear scan
    // (instance count is small, typically <100)
    uint instIdx = 0;
    uint remaining = flatID;
    for (uint i = 0; i < cullUniforms.instanceCount; i++) {
        if (remaining < instances[i].meshletCount) {
            instIdx = i;
            break;
        }
        remaining -= instances[i].meshletCount;
    }

    InstanceData inst = instances[instIdx];
    uint localMeshletIdx = remaining;
    uint globalMeshletID = inst.meshletStart + localMeshletIdx;

    GPUMeshletBounds b = meshletBounds[globalMeshletID];
    float3 centerOS = b.center_radius.xyz;
    float  radius   = b.center_radius.w;
    float3 coneAxis = b.cone_axis_cutoff.xyz;
    float  coneCutoff = b.cone_axis_cutoff.w;

    // Transform bounds center to world space
    float3 centerWS = mul(inst.worldMatrix, float4(centerOS, 1.0)).xyz;
    float3 sx = inst.worldMatrix[0].xyz;
    float3 sy = inst.worldMatrix[1].xyz;
    float3 sz = inst.worldMatrix[2].xyz;
    float maxScale = max(length(sx), max(length(sy), length(sz)));
    float worldRadius = radius * maxScale;

    bool culled = false;

    // Frustum culling in clip space
    if (cullUniforms.enableFrustumCull != 0) {
        // Extract frustum planes from viewProj.
        // viewProj is transposed for Slang, so vp[i] = row i of original MVP.
        // This matches MvpToPlanes: mt = transpose(m), planes from mt[3] ± mt[0/1/2].
        float4x4 vp = cullUniforms.viewProj;
        float4 planes[6];
        planes[0] = vp[3] + vp[0]; // Left
        planes[1] = vp[3] - vp[0]; // Right
        planes[2] = vp[3] + vp[1]; // Bottom
        planes[3] = vp[3] - vp[1]; // Top
        planes[4] = vp[2];         // Near
        planes[5] = vp[3] - vp[2]; // Far

        [unroll]
        for (int i = 0; i < 6; i++) {
            float len = length(planes[i].xyz);
            if (len > 1e-6) {
                float d = (dot(planes[i].xyz, centerWS) + planes[i].w) / len;
                if (d < -worldRadius) {
                    culled = true;
                    break;
                }
            }
        }
    }

    // Cone/backface culling in world space
    if (!culled && cullUniforms.enableConeCull != 0) {
        float3 cameraPosWS = cullUniforms.cameraWorldPos.xyz;
        // Rotate cone axis to world space (direction only)
        float3 coneAxisWS = normalize(mul((float3x3)inst.worldMatrix, coneAxis));
        float3 cameraToCenter = centerWS - cameraPosWS;
        float lenSq = dot(cameraToCenter, cameraToCenter);
        if (lenSq > 1e-12) {
            float lenV = sqrt(lenSq);
            float lhs = dot(cameraToCenter, coneAxisWS);
            float rhs = coneCutoff * lenV + worldRadius;
            if (lhs >= rhs)
                culled = true;
        }
    }

    if (!culled &&
        cullUniforms.enableOcclusionCull != 0 &&
        cullUniforms.hzbLevelCount > 0u) {
        float4 centerClip = mul(cullUniforms.prevViewProj, float4(centerWS, 1.0));
        if (centerClip.w > 1e-5) {
            float3 centerVS = mul(cullUniforms.prevView, float4(centerWS, 1.0)).xyz;
            float viewDepth = max(abs(centerVS.z), 1e-4);
            float2 ndcRadius = float2(cullUniforms.prevProjScale.x * worldRadius / viewDepth,
                                      cullUniforms.prevProjScale.y * worldRadius / viewDepth) *
                               cullUniforms.occlusionBoundsScale;
            float2 centerNdc = centerClip.xy / centerClip.w;

            float2 minUv = float2(centerNdc.x - ndcRadius.x, centerNdc.y + ndcRadius.y) *
                               float2(0.5, -0.5) +
                           float2(0.5, 0.5);
            float2 maxUv = float2(centerNdc.x + ndcRadius.x, centerNdc.y - ndcRadius.y) *
                               float2(0.5, -0.5) +
                           float2(0.5, 0.5);
            minUv = clamp(minUv, float2(0.0, 0.0), float2(1.0, 1.0));
            maxUv = clamp(maxUv, float2(0.0, 0.0), float2(1.0, 1.0));
            float2 uvMin = min(minUv, maxUv);
            float2 uvMax = max(minUv, maxUv);

            float2 rectSizePixels = max((uvMax - uvMin) * cullUniforms.hzbTextureSize, float2(1.0, 1.0));
            float maxExtent = max(rectSizePixels.x, rectSizePixels.y);
            uint level = 0u;
            if (maxExtent > 1.0) {
                level = min(cullUniforms.hzbLevelCount - 1u, (uint)ceil(log2(maxExtent)));
            }

            uint2 mipSize = uint2((uint)cullUniforms.hzbTextureSize.x, (uint)cullUniforms.hzbTextureSize.y);
            for (uint mip = 0u; mip < level; ++mip) {
                mipSize = uint2(max((mipSize.x + 1u) / 2u, 1u),
                                max((mipSize.y + 1u) / 2u, 1u));
            }

            float2 mipScale = float2((float)mipSize.x, (float)mipSize.y);
            uint2 minTexel = uint2(floor(uvMin * mipScale));
            uint2 maxTexel = uint2(floor(uvMax * mipScale));
            uint2 mipMaxCoord = uint2(max(mipSize.x, 1u) - 1u,
                                      max(mipSize.y, 1u) - 1u);
            minTexel = min(minTexel, mipMaxCoord);
            maxTexel = min(maxTexel, mipMaxCoord);

            // Reversed-Z: HZB stores min (farthest occluder). Take min of 4 corners.
            float sampledDepth = sampleHzb(level, minTexel);
            sampledDepth = min(sampledDepth, sampleHzb(level, uint2(maxTexel.x, minTexel.y)));
            sampledDepth = min(sampledDepth, sampleHzb(level, uint2(minTexel.x, maxTexel.y)));
            sampledDepth = min(sampledDepth, sampleHzb(level, maxTexel));

            float3 prevCameraToCenter = centerWS - cullUniforms.prevCameraWorldPos.xyz;
            float prevCameraDistanceSq = dot(prevCameraToCenter, prevCameraToCenter);
            if (prevCameraDistanceSq > 1e-10) {
                float3 nearestPointWS = centerWS - normalize(prevCameraToCenter) * worldRadius;
                float4 nearestClip = mul(cullUniforms.prevViewProj, float4(nearestPointWS, 1.0));
                if (nearestClip.w > 1e-5) {
                    float nearestDepth = saturate(nearestClip.z / nearestClip.w);
                    // Reversed-Z: closer = larger depth. Occluded if meshlet's
                    // nearest depth is less than the farthest occluder depth.
                    if (nearestDepth < sampledDepth - cullUniforms.occlusionDepthBias) {
                        culled = true;
                    }
                }
            }
        }
    }

    if (!culled) {
        uint slot;
        counters.InterlockedAdd(0, 1u, slot);
        MeshletDrawInfo info;
        info.instanceID = inst.instanceID;
        info.globalMeshletID = globalMeshletID;
        visibleMeshlets[slot] = info;
    }
}

// GPU meshlet culling compute shader.
// One threadgroup per visible instance from the classification front-end.
// Each thread processes a subset of that instance's meshlets and appends
// visible {sceneInstanceID, globalMeshletID} pairs.

#include "../../Source/Rendering/hzb_constants.h"
#include "../Shared/gpu_driven_helpers.slang"

struct CullUniforms {
    float4x4 viewProj;
    float4x4 prevViewProj;
    float4x4 prevView;
    float4   cameraWorldPos;
    float4   prevCameraWorldPos;
    float2   prevProjScale;
    float2   hzbTextureSize;
    uint     enableFrustumCull;
    uint     enableConeCull;
    uint     enableOcclusionCull;
    uint     hzbLevelCount;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
};

struct GPUMeshletBounds {
    float4 center_radius;
    float4 cone_apex_pad;
    float4 cone_axis_cutoff;
};

ConstantBuffer<CullUniforms>        cullUniforms;    // buffer(GPU_DRIVEN_CULL_UNIFORMS_BINDING)
StructuredBuffer<InstanceData>      instances;       // buffer(GPU_DRIVEN_CULL_INSTANCE_DATA_BINDING)
StructuredBuffer<GeometryData>      geometries;      // buffer(GPU_DRIVEN_CULL_GEOMETRY_DATA_BINDING)
StructuredBuffer<GPUMeshletBounds>  meshletBounds;   // buffer(GPU_DRIVEN_CULL_BOUNDS_BINDING)
StructuredBuffer<VisibleInstanceInfo> visibleInstances; // buffer(GPU_DRIVEN_CULL_VISIBLE_INSTANCES_BINDING)
RWStructuredBuffer<MeshletDrawInfo> visibleMeshlets; // buffer(GPU_DRIVEN_CULL_COMPACTION_OUTPUT_BINDING)
RWByteAddressBuffer                 counters;        // buffer(GPU_DRIVEN_CULL_COUNTER_BINDING)
Texture2D<float>                    hzbPyramid[HZB_MAX_LEVELS]; // texture(GPU_DRIVEN_CULL_HZB_TEXTURE_BINDING_BASE+)

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
[numthreads(64, 1, 1)]
void computeMain(uint3 groupID : SV_GroupID,
                 uint3 groupThreadID : SV_GroupThreadID) {
    VisibleInstanceInfo visibleInfo = visibleInstances[groupID.x];
    uint sceneInstanceID = visibleInfo.sceneInstanceID;
    InstanceData inst = instances[sceneInstanceID];
    GeometryData geometry = geometries[inst.geometryIndex];

    for (uint localMeshletIdx = groupThreadID.x;
         localMeshletIdx < geometry.meshletCount;
         localMeshletIdx += 64u) {
        uint globalMeshletID = geometry.meshletStart + localMeshletIdx;

        GPUMeshletBounds b = meshletBounds[globalMeshletID];
        float3 centerOS = b.center_radius.xyz;
        float  radius = b.center_radius.w;
        float3 coneAxis = b.cone_axis_cutoff.xyz;
        float  coneCutoff = b.cone_axis_cutoff.w;

        float3 centerWS = mul(inst.worldMatrix, float4(centerOS, 1.0)).xyz;
        float3 sx = inst.worldMatrix[0].xyz;
        float3 sy = inst.worldMatrix[1].xyz;
        float3 sz = inst.worldMatrix[2].xyz;
        float maxScale = max(length(sx), max(length(sy), length(sz)));
        float worldRadius = radius * maxScale;

        bool culled = false;

        if (cullUniforms.enableFrustumCull != 0u) {
            float4x4 vp = cullUniforms.viewProj;
            float4 planes[6];
            planes[0] = vp[3] + vp[0];
            planes[1] = vp[3] - vp[0];
            planes[2] = vp[3] + vp[1];
            planes[3] = vp[3] - vp[1];
            planes[4] = vp[2];
            planes[5] = vp[3] - vp[2];

            [unroll]
            for (int i = 0; i < 6; ++i) {
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

        if (!culled && cullUniforms.enableConeCull != 0u) {
            float3 cameraPosWS = cullUniforms.cameraWorldPos.xyz;
            float3 coneAxisWS = normalize(mul((float3x3)inst.worldMatrix, coneAxis));
            float3 cameraToCenter = centerWS - cameraPosWS;
            float lenSq = dot(cameraToCenter, cameraToCenter);
            if (lenSq > 1e-12) {
                float lenV = sqrt(lenSq);
                float lhs = dot(cameraToCenter, coneAxisWS);
                float rhs = coneCutoff * lenV + worldRadius;
                if (lhs >= rhs) {
                    culled = true;
                }
            }
        }

        if (!culled &&
            cullUniforms.enableOcclusionCull != 0u &&
            cullUniforms.hzbLevelCount > 0u) {
            float4 centerClip = mul(cullUniforms.prevViewProj, float4(centerWS, 1.0));
            if (centerClip.w > 1e-5) {
                float3 centerVS = mul(cullUniforms.prevView, float4(centerWS, 1.0)).xyz;
                float viewDepth = max(abs(centerVS.z), 1e-4);
                float2 ndcRadius = float2(cullUniforms.prevProjScale.x * worldRadius / viewDepth,
                                          cullUniforms.prevProjScale.y * worldRadius / viewDepth) *
                                   cullUniforms.occlusionBoundsScale;
                float2 centerNdc = centerClip.xy / centerClip.w;

                float2 ndcMin = centerNdc - ndcRadius;
                float2 ndcMax = centerNdc + ndcRadius;
                bool outsidePrevView = ndcMax.x < -1.0 || ndcMin.x > 1.0 ||
                                       ndcMax.y < -1.0 || ndcMin.y > 1.0;

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

                if (!outsidePrevView) {
                    float2 rectSizePixels =
                        max((uvMax - uvMin) * cullUniforms.hzbTextureSize, float2(1.0, 1.0));
                    float maxExtent = max(rectSizePixels.x, rectSizePixels.y);
                    uint level = 0u;
                    if (maxExtent > 1.0) {
                        level = min(cullUniforms.hzbLevelCount - 1u, (uint)ceil(log2(maxExtent)));
                    }

                    uint2 mipSize =
                        uint2((uint)cullUniforms.hzbTextureSize.x, (uint)cullUniforms.hzbTextureSize.y);
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

                    float sampledDepth = sampleHzb(level, minTexel);
                    sampledDepth = min(sampledDepth, sampleHzb(level, uint2(maxTexel.x, minTexel.y)));
                    sampledDepth = min(sampledDepth, sampleHzb(level, uint2(minTexel.x, maxTexel.y)));
                    sampledDepth = min(sampledDepth, sampleHzb(level, maxTexel));

                    float3 prevCameraToCenter = centerWS - cullUniforms.prevCameraWorldPos.xyz;
                    float prevCameraDistanceSq = dot(prevCameraToCenter, prevCameraToCenter);
                    if (prevCameraDistanceSq > 1e-10) {
                        float3 nearestPointWS =
                            centerWS - normalize(prevCameraToCenter) * worldRadius;
                        float4 nearestClip = mul(cullUniforms.prevViewProj, float4(nearestPointWS, 1.0));
                        if (nearestClip.w > 1e-5) {
                            float nearestDepth = saturate(nearestClip.z / nearestClip.w);
                            if (nearestDepth < sampledDepth - cullUniforms.occlusionDepthBias) {
                                culled = true;
                            }
                        }
                    }
                }
            }
        }

        if (!culled) {
            uint slot = gpuDrivenAppendWorkItemSlot(counters);
            MeshletDrawInfo info;
            info.instanceID = sceneInstanceID;
            info.globalMeshletID = globalMeshletID;
            visibleMeshlets[slot] = info;
        }
    }
}

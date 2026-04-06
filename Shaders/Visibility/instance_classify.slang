// GPU instance classification front-end.
// One thread per scene instance. Emits a compact visible-instance worklist
// that downstream meshlet stages consume.

#include "../../Source/Rendering/hzb_constants.h"
#include "../Shared/gpu_driven_helpers.slang"

struct InstanceClassifyUniforms {
    float4x4 viewProj;
    float4x4 prevViewProj;
    float4x4 prevView;
    float4   cameraWorldPos;
    float4   prevCameraWorldPos;
    float2   prevProjScale;
    float2   hzbTextureSize;
    uint     instanceCount;
    uint     enableFrustumCull;
    uint     enableOcclusionCull;
    uint     hzbLevelCount;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
};

ConstantBuffer<InstanceClassifyUniforms> classifyUniforms; // buffer(GPU_DRIVEN_INSTANCE_CLASSIFY_UNIFORMS_BINDING)
StructuredBuffer<InstanceData>           instances;        // buffer(GPU_DRIVEN_INSTANCE_CLASSIFY_INSTANCE_DATA_BINDING)
StructuredBuffer<GeometryData>           geometries;       // buffer(GPU_DRIVEN_INSTANCE_CLASSIFY_GEOMETRY_DATA_BINDING)
RWStructuredBuffer<VisibleInstanceInfo>  visibleInstances; // buffer(GPU_DRIVEN_INSTANCE_CLASSIFY_OUTPUT_BINDING)
RWByteAddressBuffer                      worklistState;    // buffer(GPU_DRIVEN_INSTANCE_CLASSIFY_STATE_BINDING)
Texture2D<float>                         hzbPyramid[HZB_MAX_LEVELS]; // texture(GPU_DRIVEN_INSTANCE_CLASSIFY_HZB_TEXTURE_BINDING_BASE+)

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
void computeMain(uint3 dtid : SV_DispatchThreadID) {
    uint sceneInstanceID = dtid.x;
    if (sceneInstanceID >= classifyUniforms.instanceCount) {
        return;
    }

    InstanceData inst = instances[sceneInstanceID];
    if ((inst.visibilityFlags & kGpuSceneInstanceVisible) == 0u) {
        return;
    }

    GeometryData geometry = geometries[inst.geometryIndex];
    if (geometry.meshletCount == 0u) {
        return;
    }

    float3 centerOS = geometry.boundsCenterRadius.xyz;
    float radius = geometry.boundsCenterRadius.w;

    float3 centerWS = mul(inst.worldMatrix, float4(centerOS, 1.0)).xyz;
    float3 sx = inst.worldMatrix[0].xyz;
    float3 sy = inst.worldMatrix[1].xyz;
    float3 sz = inst.worldMatrix[2].xyz;
    float maxScale = max(length(sx), max(length(sy), length(sz)));
    float worldRadius = radius * maxScale;

    bool culled = false;

    if (classifyUniforms.enableFrustumCull != 0u) {
        float4x4 vp = classifyUniforms.viewProj;
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

    float4 centerClip = mul(classifyUniforms.viewProj, float4(centerWS, 1.0));

    if (!culled &&
        classifyUniforms.enableOcclusionCull != 0u &&
        classifyUniforms.hzbLevelCount > 0u) {
        float4 prevCenterClip = mul(classifyUniforms.prevViewProj, float4(centerWS, 1.0));
        if (prevCenterClip.w > 1e-5) {
            float3 centerVS = mul(classifyUniforms.prevView, float4(centerWS, 1.0)).xyz;
            float viewDepth = max(abs(centerVS.z), 1e-4);
            float2 ndcRadius = float2(classifyUniforms.prevProjScale.x * worldRadius / viewDepth,
                                      classifyUniforms.prevProjScale.y * worldRadius / viewDepth) *
                               classifyUniforms.occlusionBoundsScale;
            float2 centerNdc = prevCenterClip.xy / prevCenterClip.w;

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
                    max((uvMax - uvMin) * classifyUniforms.hzbTextureSize, float2(1.0, 1.0));
                float maxExtent = max(rectSizePixels.x, rectSizePixels.y);
                uint level = 0u;
                if (maxExtent > 1.0) {
                    level = min(classifyUniforms.hzbLevelCount - 1u, (uint)ceil(log2(maxExtent)));
                }

                uint2 mipSize =
                    uint2((uint)classifyUniforms.hzbTextureSize.x, (uint)classifyUniforms.hzbTextureSize.y);
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

                float3 prevCameraToCenter = centerWS - classifyUniforms.prevCameraWorldPos.xyz;
                float prevCameraDistanceSq = dot(prevCameraToCenter, prevCameraToCenter);
                if (prevCameraDistanceSq > 1e-10) {
                    float3 nearestPointWS =
                        centerWS - normalize(prevCameraToCenter) * worldRadius;
                    float4 nearestClip = mul(classifyUniforms.prevViewProj, float4(nearestPointWS, 1.0));
                    if (nearestClip.w > 1e-5) {
                        float nearestDepth = saturate(nearestClip.z / nearestClip.w);
                        if (nearestDepth < sampledDepth - classifyUniforms.occlusionDepthBias) {
                            culled = true;
                        }
                    }
                }
            }
        }
    }

    if (culled) {
        return;
    }

    float cameraDistance = length(centerWS - classifyUniforms.cameraWorldPos.xyz);
    float depthMetric = centerClip.w > 1e-5 ? centerClip.w : max(cameraDistance, 1e-4);
    float projectedScale = maxScale / max(depthMetric, 1e-4);

    uint slot = gpuDrivenAppendWorkItemSlot(worklistState);

    VisibleInstanceInfo info;
    info.sceneInstanceID = sceneInstanceID;
    info.geometryIndex = inst.geometryIndex;
    info.lodRootNode = geometry.lodRootNode;
    info.classificationFlags = kVisibleInstanceClassificationVisible;
    if (((inst.visibilityFlags & kGpuSceneInstanceHasLod) != 0u) ||
        geometry.lodRootNode != 0xFFFFFFFFu) {
        info.classificationFlags |= kVisibleInstanceClassificationHasLod;
    }
    info.boundsCenterRadius = float4(centerWS, worldRadius);
    info.lodMetric = float4(depthMetric, cameraDistance, maxScale, projectedScale);
    visibleInstances[slot] = info;
}

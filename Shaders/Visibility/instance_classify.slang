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

#include "hzb_cull_helpers.slang"

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

    if (!culled && classifyUniforms.enableOcclusionCull != 0u) {
        culled = spherePreviousFrameOccluded(classifyUniforms.prevViewProj,
                                             classifyUniforms.prevView,
                                             classifyUniforms.prevCameraWorldPos,
                                             classifyUniforms.prevProjScale,
                                             classifyUniforms.hzbTextureSize,
                                             classifyUniforms.hzbLevelCount,
                                             classifyUniforms.occlusionBoundsScale,
                                             classifyUniforms.occlusionDepthBias,
                                             centerWS,
                                             worldRadius);
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

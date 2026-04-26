// GPU meshlet culling + ClusterLOD traversal compute shader.
// One threadgroup per visible instance from the classification front-end.
// Instances with `lodRootNode` traverse `nodes -> groups -> meshlets`;
// instances without runtime LOD fall back to their authored meshlet range.

#include "../../Source/Rendering/hzb_constants.h"
#include "../Shared/gpu_driven_helpers.slang"

struct CullUniforms {
    float4x4 viewProj;
    float4x4 view;
    float4x4 prevViewProj;
    float4x4 prevView;
    float4   cameraWorldPos;
    float4   prevCameraWorldPos;
    float2   projScale;
    float2   renderTargetSize;
    float2   prevProjScale;
    float2   hzbTextureSize;
    float2   currentHzbTextureSize;
    uint     enableFrustumCull;
    uint     enableConeCull;
    uint     enableOcclusionCull;
    uint     hzbLevelCount;
    float    lodReferencePixels;
    float    occlusionDepthBias;
    float    occlusionBoundsScale;
    uint     clusterLodEnabled;
    uint     enableResidencyStreaming;
    uint     residencyRequestFrameIndex;
    uint     cullPassIndex;
    uint     currentHzbLevelCount;
    uint     occlusionCullMode;
    uint     reserved1;
};

struct GPUMeshletBounds {
    float4 center_radius;
    float4 cone_apex_pad;
    float4 cone_axis_cutoff;
};

struct GPULodNode {
    float center[3];
    float radius;
    float maxError;
    uint  childOffset;
    uint  childCount;
    uint  isLeaf;
};

struct GPUClusterGroup {
    float center[3];
    float radius;
    float error;
    float parentError;
    uint  clusterStart;
    uint  clusterCount;
};

ConstantBuffer<CullUniforms>          cullUniforms;     // buffer(GPU_DRIVEN_CULL_UNIFORMS_BINDING)
StructuredBuffer<InstanceData>        instances;        // buffer(GPU_DRIVEN_CULL_INSTANCE_DATA_BINDING)
StructuredBuffer<GeometryData>        geometries;       // buffer(GPU_DRIVEN_CULL_GEOMETRY_DATA_BINDING)
StructuredBuffer<GPUMeshletBounds>    meshletBounds;    // buffer(GPU_DRIVEN_CULL_BOUNDS_BINDING)
StructuredBuffer<VisibleInstanceInfo> visibleInstances; // buffer(GPU_DRIVEN_CULL_VISIBLE_INSTANCES_BINDING)
RWStructuredBuffer<MeshletDrawInfo>   visibleMeshlets;  // buffer(GPU_DRIVEN_CULL_COMPACTION_OUTPUT_BINDING)
RWByteAddressBuffer                   counters;         // buffer(GPU_DRIVEN_CULL_COUNTER_BINDING)
StructuredBuffer<GPULodNode>          lodNodes;         // buffer(GPU_DRIVEN_CULL_LOD_NODE_BINDING)
StructuredBuffer<GPUClusterGroup>     lodGroups;        // buffer(GPU_DRIVEN_CULL_LOD_GROUP_BINDING)
StructuredBuffer<uint>                residentGroupMeshletIndices; // buffer(GPU_DRIVEN_CULL_LOD_GROUP_MESHLET_INDICES_BINDING)
StructuredBuffer<GPUMeshletBounds>    lodMeshletBounds; // buffer(GPU_DRIVEN_CULL_LOD_BOUNDS_BINDING)
RWStructuredBuffer<uint>              traversalStats;   // buffer(GPU_DRIVEN_CULL_TRAVERSAL_STATS_BINDING)
RWByteAddressBuffer                   groupResidency;   // buffer(GPU_DRIVEN_CULL_GROUP_RESIDENCY_BINDING)
RWStructuredBuffer<uint64_t>          lodGroupPageTable; // buffer(GPU_DRIVEN_CULL_LOD_GROUP_PAGE_TABLE_BINDING)
RWStructuredBuffer<ClusterResidencyRequest> residencyRequests; // buffer(GPU_DRIVEN_CULL_RESIDENCY_REQUEST_OUTPUT_BINDING)
RWByteAddressBuffer                   residencyRequestState; // buffer(GPU_DRIVEN_CULL_RESIDENCY_REQUEST_STATE_BINDING)
StructuredBuffer<uint>                sourceGroupMeshletIndices; // buffer(GPU_DRIVEN_CULL_LOD_GROUP_MESHLET_INDICES_SOURCE_BINDING)
RWStructuredBuffer<uint>              groupAgeBuffer;   // buffer(GPU_DRIVEN_CULL_GROUP_AGE_BINDING)
[[vk::binding(GPU_DRIVEN_CULL_HZB_TEXTURE_BINDING_BASE)]]
Texture2D<float>                      hzbPyramid[HZB_MAX_LEVELS];
[[vk::binding(GPU_DRIVEN_CULL_CURRENT_HZB_TEXTURE_BINDING_BASE)]]
Texture2D<float>                      currentHzbPyramid[HZB_MAX_LEVELS];

static const uint kTraversalStatLodInstances = 0u;
static const uint kTraversalStatFallbackInstances = 1u;
static const uint kTraversalStatTraversedNodes = 2u;
static const uint kTraversalStatOccludedNodes = 3u;
static const uint kTraversalStatCandidateGroups = 4u;
static const uint kTraversalStatSelectedGroups = 5u;
static const uint kTraversalStatOccludedGroups = 6u;
static const uint kTraversalStatCandidateClusterMeshlets = 7u;
static const uint kTraversalStatClusterMeshlets = 8u;
static const uint kTraversalStatCandidateFallbackMeshlets = 9u;
static const uint kTraversalStatFallbackMeshlets = 10u;
static const uint kTraversalStatMaxSelectedLod = 11u;
static const uint kTraversalStatHistogramBase = 12u;
static const uint kTraversalNodeStackCapacity = 64u;
static const uint kOcclusionCullModeTwoPhase = 0u;
static const uint kOcclusionCullModeCurrentDirect = 1u;

#define HZB_CULL_ENABLE_CURRENT_PYRAMID 1
#include "hzb_cull_helpers.slang"

float instanceMaxScale(InstanceData inst) {
    float3 sx = inst.worldMatrix[0].xyz;
    float3 sy = inst.worldMatrix[1].xyz;
    float3 sz = inst.worldMatrix[2].xyz;
    return max(length(sx), max(length(sy), length(sz)));
}

bool sphereFrustumCulled(float3 centerWS, float worldRadius) {
    if (cullUniforms.enableFrustumCull == 0u) {
        return false;
    }

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
                return true;
            }
        }
    }

    return false;
}

bool sphereTraversalOcclusionCulledForPass(float3 centerWS, float worldRadius, out bool hzbRejected) {
    hzbRejected = false;
    if (cullUniforms.enableOcclusionCull == 0u) {
        return false;
    }

    const bool previousHzbAvailable = cullUniforms.hzbLevelCount > 0u;
    const bool currentHzbAvailable = cullUniforms.currentHzbLevelCount > 0u;

    if (cullUniforms.cullPassIndex == 0u) {
        if (!previousHzbAvailable) {
            return false;
        }

        hzbRejected = spherePreviousFrameOccluded(cullUniforms.prevViewProj,
                                                  cullUniforms.prevView,
                                                  cullUniforms.prevCameraWorldPos,
                                                  cullUniforms.prevProjScale,
                                                  cullUniforms.hzbTextureSize,
                                                  cullUniforms.hzbLevelCount,
                                                  cullUniforms.occlusionBoundsScale,
                                                  cullUniforms.occlusionDepthBias,
                                                  centerWS,
                                                  worldRadius);
        return hzbRejected;
    }

    if (!currentHzbAvailable) {
        return false;
    }

    hzbRejected = sphereCurrentFrameOccluded(cullUniforms.viewProj,
                                             cullUniforms.view,
                                             cullUniforms.cameraWorldPos,
                                             cullUniforms.projScale,
                                             cullUniforms.currentHzbTextureSize,
                                             cullUniforms.currentHzbLevelCount,
                                             cullUniforms.occlusionBoundsScale,
                                             cullUniforms.occlusionDepthBias,
                                             centerWS,
                                             worldRadius);
    return hzbRejected;
}

bool sphereRenderableOcclusionCulledForPass(float3 centerWS, float worldRadius, out bool hzbRejected) {
    hzbRejected = false;
    if (cullUniforms.enableOcclusionCull == 0u) {
        return false;
    }

    const bool previousHzbAvailable = cullUniforms.hzbLevelCount > 0u;
    const bool currentHzbAvailable = cullUniforms.currentHzbLevelCount > 0u;

    if (cullUniforms.cullPassIndex == 0u) {
        if (!previousHzbAvailable) {
            return false;
        }

        hzbRejected = spherePreviousFrameOccluded(cullUniforms.prevViewProj,
                                                  cullUniforms.prevView,
                                                  cullUniforms.prevCameraWorldPos,
                                                  cullUniforms.prevProjScale,
                                                  cullUniforms.hzbTextureSize,
                                                  cullUniforms.hzbLevelCount,
                                                  cullUniforms.occlusionBoundsScale,
                                                  cullUniforms.occlusionDepthBias,
                                                  centerWS,
                                                  worldRadius);
        return hzbRejected;
    }

    if (cullUniforms.occlusionCullMode == kOcclusionCullModeCurrentDirect) {
        if (!currentHzbAvailable) {
            return false;
        }

        hzbRejected = sphereCurrentFrameOccluded(cullUniforms.viewProj,
                                                 cullUniforms.view,
                                                 cullUniforms.cameraWorldPos,
                                                 cullUniforms.projScale,
                                                 cullUniforms.currentHzbTextureSize,
                                                 cullUniforms.currentHzbLevelCount,
                                                 cullUniforms.occlusionBoundsScale,
                                                 cullUniforms.occlusionDepthBias,
                                                 centerWS,
                                                 worldRadius);
        return hzbRejected;
    }

    if (!previousHzbAvailable) {
        return true;
    }

    const bool wasOccludedByPreviousHzb =
        spherePreviousFrameOccluded(cullUniforms.prevViewProj,
                                    cullUniforms.prevView,
                                    cullUniforms.prevCameraWorldPos,
                                    cullUniforms.prevProjScale,
                                    cullUniforms.hzbTextureSize,
                                    cullUniforms.hzbLevelCount,
                                    cullUniforms.occlusionBoundsScale,
                                    cullUniforms.occlusionDepthBias,
                                    centerWS,
                                    worldRadius);
    if (!wasOccludedByPreviousHzb) {
        return true;
    }

    if (!currentHzbAvailable) {
        hzbRejected = true;
        return true;
    }

    hzbRejected = sphereCurrentFrameOccluded(cullUniforms.viewProj,
                                             cullUniforms.view,
                                             cullUniforms.cameraWorldPos,
                                             cullUniforms.projScale,
                                             cullUniforms.currentHzbTextureSize,
                                             cullUniforms.currentHzbLevelCount,
                                             cullUniforms.occlusionBoundsScale,
                                             cullUniforms.occlusionDepthBias,
                                             centerWS,
                                             worldRadius);
    return hzbRejected;
}

bool fineCullMeshlet(InstanceData inst, GPUMeshletBounds b, float maxScale) {
    float3 centerOS = b.center_radius.xyz;
    float radius = b.center_radius.w;
    float3 coneAxis = b.cone_axis_cutoff.xyz;
    float coneCutoff = b.cone_axis_cutoff.w;

    float3 centerWS = mul(inst.worldMatrix, float4(centerOS, 1.0)).xyz;
    float worldRadius = radius * maxScale;

    bool culled = sphereFrustumCulled(centerWS, worldRadius);

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

    if (!culled) {
        bool hzbRejected = false;
        culled = sphereRenderableOcclusionCulledForPass(centerWS, worldRadius, hzbRejected);
    }

    return culled;
}

bool coarseCullTraversalSphere(float3 centerWS, float worldRadius, out bool hzbRejected) {
    hzbRejected = false;
    if (sphereFrustumCulled(centerWS, worldRadius)) {
        return true;
    }

    return sphereTraversalOcclusionCulledForPass(centerWS, worldRadius, hzbRejected);
}

void emitVisibleMeshlet(uint sceneInstanceID,
                        uint globalMeshletID,
                        uint meshletSource,
                        uint lodLevel) {
    uint slot = gpuDrivenAppendWorkItemSlot(counters);
    MeshletDrawInfo info;
    info.instanceID = sceneInstanceID;
    info.globalMeshletID = globalMeshletID;
    info.meshletSource = meshletSource;
    info.lodLevel = lodLevel;
    visibleMeshlets[slot] = info;
}

uint selectLodLevel(VisibleInstanceInfo visibleInfo, GPULodNode lodRoot) {
    if (lodRoot.childCount <= 1u) {
        return 0u;
    }

    float cameraDistance = max(visibleInfo.lodMetric.y, 1e-4);
    float pixelRadius =
        visibleInfo.boundsCenterRadius.w * cullUniforms.projScale.y *
        (0.5 * cullUniforms.renderTargetSize.y) / cameraDistance;
    float lodRatio = cullUniforms.lodReferencePixels / max(pixelRadius, 1.0);
    uint lodLevel = (uint)floor(max(log2(lodRatio), 0.0));
    return min(lodLevel, lodRoot.childCount - 1u);
}

uint loadGroupResidencyState(uint groupIndex) {
    return groupResidency.Load(groupIndex * 4u);
}

uint64_t loadLodGroupResidentClusterStart(uint groupIndex) {
    return lodGroupPageTable[groupIndex];
}

void appendResidencyRequest(uint lodRootNode,
                            uint targetGroupIndex,
                            uint lodLevel) {
    uint slot = gpuDrivenAppendWorkItemSlot(residencyRequestState);
    ClusterResidencyRequest request;
    request.lodRootNode = lodRootNode;
    request.targetGroupIndex = targetGroupIndex;
    request.lodLevel = lodLevel;
    request.requestFrameIndex = cullUniforms.residencyRequestFrameIndex;
    residencyRequests[slot] = request;
}

void queueResidencyRequest(uint lodRootNode,
                           uint targetGroupIndex,
                           uint lodLevel) {
    const uint64_t residentAddress = loadLodGroupResidentClusterStart(targetGroupIndex);
    if (!clusterLodGroupPageAddressIsValid(residentAddress)) {
        const uint64_t currentFrameTag =
            clusterLodGroupPageInvalidAddress(cullUniforms.residencyRequestFrameIndex);
        uint64_t previousAddress = 0;
        InterlockedMax(lodGroupPageTable[targetGroupIndex], currentFrameTag, previousAddress);
        if (previousAddress == currentFrameTag) {
            return;
        }
    } else {
        uint previousState = 0u;
        groupResidency.InterlockedOr(targetGroupIndex * 4u,
                                     kClusterLodGroupResidencyRequested,
                                     previousState);
        if ((previousState & (kClusterLodGroupResidencyResident |
                              kClusterLodGroupResidencyRequested)) != 0u) {
            return;
        }
    }

    appendResidencyRequest(lodRootNode, targetGroupIndex, lodLevel);
}

void touchResidentGroup(uint groupIndex) {
    uint previousState = 0u;
    groupResidency.InterlockedOr(groupIndex * 4u,
                                 kClusterLodGroupResidencyTouched,
                                 previousState);
    groupAgeBuffer[groupIndex] = 0u;
    if ((previousState & kClusterLodGroupResidencyTouched) != 0u) {
        return;
    }

    appendResidencyRequest(0xFFFFFFFFu, groupIndex, 0u);
}

bool visibleGroupsResidentForLodRoot(InstanceData inst,
                                     float maxScale,
                                     uint lodRootNode,
                                     uint lodLevel,
                                     uint sceneInstanceID,
                                     uint rootNodeIndex,
                                     bool queueMissingRequests) {
    uint nodeStack[kTraversalNodeStackCapacity];
    uint stackSize = 0u;
    bool allVisibleGroupsResident = true;
    nodeStack[stackSize++] = rootNodeIndex;

    while (stackSize > 0u) {
        uint nodeIndex = nodeStack[--stackSize];
        GPULodNode node = lodNodes[nodeIndex];

        float3 nodeCenterWS =
            mul(inst.worldMatrix, float4(node.center[0], node.center[1], node.center[2], 1.0)).xyz;
        float nodeWorldRadius = node.radius * maxScale;
        bool nodeHzbRejected = false;
        if (coarseCullTraversalSphere(nodeCenterWS, nodeWorldRadius, nodeHzbRejected)) {
            continue;
        }

        if (node.isLeaf != 0u) {
            for (uint childIndex = 0u; childIndex < node.childCount; ++childIndex) {
                uint groupIndex = node.childOffset + childIndex;
                GPUClusterGroup group = lodGroups[groupIndex];
                float3 groupCenterWS =
                    mul(inst.worldMatrix, float4(group.center[0], group.center[1], group.center[2], 1.0)).xyz;
                float groupWorldRadius = group.radius * maxScale;
                bool groupHzbRejected = false;
                if (coarseCullTraversalSphere(groupCenterWS, groupWorldRadius, groupHzbRejected)) {
                    continue;
                }

                const uint groupState = loadGroupResidencyState(groupIndex);
                const uint64_t residentClusterStart = loadLodGroupResidentClusterStart(groupIndex);
                if ((groupState & kClusterLodGroupResidencyResident) == 0u ||
                    !clusterLodGroupPageAddressIsValid(residentClusterStart)) {
                    if (queueMissingRequests) {
                        queueResidencyRequest(lodRootNode,
                                              groupIndex,
                                              lodLevel);
                    }
                    allVisibleGroupsResident = false;
                }
            }
        } else {
            uint remainingCapacity = kTraversalNodeStackCapacity - stackSize;
            uint pushCount = min(node.childCount, remainingCapacity);
            for (uint childIndex = 0u; childIndex < pushCount; ++childIndex) {
                nodeStack[stackSize++] = node.childOffset + childIndex;
            }
        }
    }

    return allVisibleGroupsResident;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 groupID : SV_GroupID,
                 uint3 groupThreadID : SV_GroupThreadID) {
    VisibleInstanceInfo visibleInfo = visibleInstances[groupID.x];
    uint sceneInstanceID = visibleInfo.sceneInstanceID;
    InstanceData inst = instances[sceneInstanceID];
    GeometryData geometry = geometries[inst.geometryIndex];
    float maxScale = instanceMaxScale(inst);

    bool canTraverseClusterLod =
        cullUniforms.clusterLodEnabled != 0u &&
        (visibleInfo.classificationFlags & kVisibleInstanceClassificationHasLod) != 0u &&
        visibleInfo.lodRootNode != 0xFFFFFFFFu;

    if (canTraverseClusterLod) {
        if (groupThreadID.x != 0u) {
            return;
        }

        GPULodNode lodRoot = lodNodes[visibleInfo.lodRootNode];
        uint selectedLodLevel = selectLodLevel(visibleInfo, lodRoot);
        uint selectedRootNode = visibleInfo.lodRootNode;
        if (lodRoot.isLeaf == 0u && lodRoot.childCount > 0u) {
            const uint maxLodLevel = lodRoot.childCount - 1u;
            selectedLodLevel = min(selectedLodLevel, maxLodLevel);
            selectedRootNode = lodRoot.childOffset + selectedLodLevel;
        }

        uint histogramLevel = min(selectedLodLevel, kClusterTraversalStatsHistogramSize - 1u);
        InterlockedAdd(traversalStats[kTraversalStatLodInstances], 1u);
        InterlockedAdd(traversalStats[kTraversalStatHistogramBase + histogramLevel], 1u);
        InterlockedMax(traversalStats[kTraversalStatMaxSelectedLod], selectedLodLevel);

        uint nodeStack[kTraversalNodeStackCapacity];
        uint stackSize = 0u;
        nodeStack[stackSize++] = selectedRootNode;

        while (stackSize > 0u) {
            uint nodeIndex = nodeStack[--stackSize];
            GPULodNode node = lodNodes[nodeIndex];

            InterlockedAdd(traversalStats[kTraversalStatTraversedNodes], 1u);
            float3 nodeCenterWS = mul(inst.worldMatrix, float4(node.center[0], node.center[1], node.center[2], 1.0)).xyz;
            float nodeWorldRadius = node.radius * maxScale;
            bool nodeHzbRejected = false;
            if (coarseCullTraversalSphere(nodeCenterWS, nodeWorldRadius, nodeHzbRejected)) {
                if (nodeHzbRejected) {
                    InterlockedAdd(traversalStats[kTraversalStatOccludedNodes], 1u);
                }
                continue;
            }

            if (node.isLeaf != 0u) {
                for (uint childIndex = 0u; childIndex < node.childCount; ++childIndex) {
                    uint groupIndex = node.childOffset + childIndex;
                    GPUClusterGroup group = lodGroups[groupIndex];
                    InterlockedAdd(traversalStats[kTraversalStatCandidateGroups], 1u);
                    float3 groupCenterWS =
                        mul(inst.worldMatrix, float4(group.center[0], group.center[1], group.center[2], 1.0)).xyz;
                    float groupWorldRadius = group.radius * maxScale;
                    bool groupHzbRejected = false;
                    if (coarseCullTraversalSphere(groupCenterWS, groupWorldRadius, groupHzbRejected)) {
                        if (groupHzbRejected) {
                            InterlockedAdd(traversalStats[kTraversalStatOccludedGroups], 1u);
                        }
                        continue;
                    }

                    InterlockedAdd(traversalStats[kTraversalStatSelectedGroups], 1u);
                    InterlockedAdd(traversalStats[kTraversalStatCandidateClusterMeshlets], group.clusterCount);
                    bool useResidentHeap = false;
                    uint meshletIndexBase = group.clusterStart;
                    if (cullUniforms.enableResidencyStreaming != 0u) {
                        const uint groupState = loadGroupResidencyState(groupIndex);
                        const uint64_t residentClusterStart =
                            loadLodGroupResidentClusterStart(groupIndex);
                        if ((groupState & kClusterLodGroupResidencyResident) == 0u ||
                            !clusterLodGroupPageAddressIsValid(residentClusterStart)) {
                            queueResidencyRequest(visibleInfo.lodRootNode,
                                                  groupIndex,
                                                  selectedLodLevel);
                        } else {
                            meshletIndexBase = (uint)residentClusterStart;
                            useResidentHeap = true;
                            touchResidentGroup(groupIndex);
                        }
                    }

                    for (uint meshletIndex = 0u; meshletIndex < group.clusterCount; ++meshletIndex) {
                        const uint globalMeshletID = useResidentHeap
                                                         ? residentGroupMeshletIndices[meshletIndexBase + meshletIndex]
                                                         : sourceGroupMeshletIndices[group.clusterStart + meshletIndex];
                        if (fineCullMeshlet(inst, lodMeshletBounds[globalMeshletID], maxScale)) {
                            continue;
                        }

                        emitVisibleMeshlet(sceneInstanceID,
                                           globalMeshletID,
                                           kMeshletDrawSourceClusterLod,
                                           selectedLodLevel);
                        InterlockedAdd(traversalStats[kTraversalStatClusterMeshlets], 1u);
                    }
                }
            } else {
                uint remainingCapacity = kTraversalNodeStackCapacity - stackSize;
                uint pushCount = min(node.childCount, remainingCapacity);
                for (uint childIndex = 0u; childIndex < pushCount; ++childIndex) {
                    nodeStack[stackSize++] = node.childOffset + childIndex;
                }
            }
        }

        return;
    }

    if (groupThreadID.x == 0u) {
        InterlockedAdd(traversalStats[kTraversalStatFallbackInstances], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint localMeshletIdx = groupThreadID.x;
         localMeshletIdx < geometry.meshletCount;
         localMeshletIdx += 64u) {
        InterlockedAdd(traversalStats[kTraversalStatCandidateFallbackMeshlets], 1u);
        uint globalMeshletID = geometry.meshletStart + localMeshletIdx;
        if (fineCullMeshlet(inst, meshletBounds[globalMeshletID], maxScale)) {
            continue;
        }

        emitVisibleMeshlet(sceneInstanceID,
                           globalMeshletID,
                           kMeshletDrawSourceScene,
                           0u);
        InterlockedAdd(traversalStats[kTraversalStatFallbackMeshlets], 1u);
    }
}

#include "dag_cpu_mirror.h"

// ml.h must come after project headers to avoid MSVC intrinsic redefinition conflicts.
#include <ml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

static constexpr uint32_t kInvalidIndex = 0xFFFFFFFFu;

struct NodeTask {
    uint32_t instanceID;
    uint32_t nodeID;
    uint32_t depth;
};

struct EmittedCluster {
    uint32_t instanceID;
    uint32_t clusterID;
    uint32_t depth;
};

float4x4 loadMatrix(const float src[16]) {
    float4x4 m;
    std::memcpy(&m, src, 16 * sizeof(float));
    return m;
}

float cpuUniformScaleUpperBound(const float4x4& m) {
    float3 origin = float3(mul(m, float4(0.f, 0.f, 0.f, 1.f)).xyz);
    float3 axisX  = float3(mul(m, float4(1.f, 0.f, 0.f, 1.f)).xyz) - origin;
    float3 axisY  = float3(mul(m, float4(0.f, 1.f, 0.f, 1.f)).xyz) - origin;
    float3 axisZ  = float3(mul(m, float4(0.f, 0.f, 1.f, 1.f)).xyz) - origin;
    return std::max({length(axisX), length(axisY), length(axisZ)});
}

bool cpuTestForLod(const float4x4& worldMatrix,
                   float3 center, float radius, float error,
                   const DagCpuMirrorParams& p) {
    if (p.lodErrorThreshold <= 0.f || error <= 0.f || error >= 3.0e37f) {
        return false;
    }
    float scale = std::max(cpuUniformScaleUpperBound(worldMatrix), 1.0e-6f);
    float allowedErrorWS = error * scale;
    float3 worldCenter = float3(mul(worldMatrix, float4(center, 1.f)).xyz);
    float worldRadius = radius * scale;
    float3 camPos(p.cameraWorldPos[0], p.cameraWorldPos[1], p.cameraWorldPos[2]);
    float3 diff = camPos - worldCenter;
    float dist = length(diff);
    float cameraToSphereSurface = std::max(0.f, dist - worldRadius);
    float pixelThreshold = std::max(p.lodErrorThreshold, 1.0e-6f);
    float screenErrorConstant =
        std::tan(0.5f * p.cameraFovY) * 2.f * pixelThreshold /
        std::max(float(p.screenHeight), 1.f);
    return cameraToSphereSurface * screenErrorConstant >= allowedErrorWS;
}

// Emit one cluster, tracking depth distribution and overflow.
void emitCluster(uint32_t instanceID, uint32_t clusterID, uint32_t depth,
                 DagCpuMirrorStats& stats,
                 std::vector<EmittedCluster>& emitted,
                 uint32_t maxClusters) {
    if (stats.clustersEmitted >= maxClusters) {
        ++stats.clusterOverflow;
        return;
    }
    ++stats.clustersEmitted;
    EmittedCluster ec;
    ec.instanceID = instanceID;
    ec.clusterID  = clusterID;
    ec.depth      = depth;
    emitted.push_back(ec);
    if (depth > stats.maxDepthSeen) {
        stats.maxDepthSeen = depth;
    }
    const uint32_t bucket = depth < DagCpuMirrorStats::kMaxDepthBuckets
                                ? depth
                                : DagCpuMirrorStats::kMaxDepthBuckets - 1u;
    ++stats.clustersByDepth[bucket];
}

// Emit all clusters in a group, applying refine suppression.
void emitGroupClusters(uint32_t instanceID, uint32_t groupID, uint32_t depth,
                       const ClusterLODData& lodData,
                       const GPUSceneInstance& inst,
                       const DagCpuMirrorParams& params,
                       DagCpuMirrorStats& stats,
                       std::vector<EmittedCluster>& emitted) {
    if (groupID >= lodData.groups.size()) {
        return;
    }
    const GPUClusterGroup& group = lodData.groups[groupID];
    const float4x4 worldMatrix = loadMatrix(inst.worldMatrix);

    for (uint32_t i = 0; i < group.clusterCount; ++i) {
        const uint32_t idx = group.clusterStart + i;
        if (idx >= lodData.groupMeshletIndices.size()) {
            break;
        }
        const uint32_t clusterID = lodData.groupMeshletIndices[idx];

        if (clusterID >= lodData.clusterRefineInfos.size()) {
            emitCluster(instanceID, clusterID, depth, stats, emitted, params.maxClusters);
            continue;
        }

        const GPUClusterRefineInfo& ri = lodData.clusterRefineInfos[clusterID];
        if (ri.refineGroupCount == 0u || ri.refineGroupStart == kInvalidIndex) {
            emitCluster(instanceID, clusterID, depth, stats, emitted, params.maxClusters);
        } else if (ri.refineGroupStart >= lodData.groups.size()) {
            emitCluster(instanceID, clusterID, depth, stats, emitted, params.maxClusters);
        } else {
            const GPUClusterGroup& rg = lodData.groups[ri.refineGroupStart];
            float3 rgCenter(rg.center[0], rg.center[1], rg.center[2]);
            if (cpuTestForLod(worldMatrix, rgCenter, rg.radius, rg.parentError, params)) {
                ++stats.refineSuppressed;
            } else {
                ++stats.refineAccepted;
                emitCluster(instanceID, clusterID, depth, stats, emitted, params.maxClusters);
            }
        }
    }
}

// Emit all representative groups for a node (LOD cut path).
void emitNodeRepGroups(uint32_t instanceID, const GPULodNode& node, uint32_t depth,
                       const ClusterLODData& lodData,
                       const GPUSceneInstance& inst,
                       const DagCpuMirrorParams& params,
                       DagCpuMirrorStats& stats,
                       std::vector<EmittedCluster>& emitted) {
    for (uint32_t i = 0; i < node.representativeGroupCount; ++i) {
        const uint32_t idx = node.representativeGroupStart + i;
        if (idx >= lodData.nodeRepresentativeGroupIndices.size()) {
            break;
        }
        emitGroupClusters(instanceID,
                          lodData.nodeRepresentativeGroupIndices[idx],
                          depth,
                          lodData, inst, params, stats, emitted);
    }
}

} // namespace

DagCpuMirrorStats runDagCpuMirror(
    const GpuSceneTables&                       scene,
    const ClusterLODData&                       lodData,
    const DagCpuMirrorParams&                   params,
    const ClusterOcclusionState::DagCullStats*  gpuStats)
{
    DagCpuMirrorStats stats{};

    if (scene.instances.empty() ||
        lodData.nodes.empty() ||
        lodData.groups.empty() ||
        lodData.groupMeshletIndices.empty()) {
        return stats;
    }
    stats.available = true;

    const uint32_t queueCap = std::min(params.maxNodeTasks, 65536u);
    std::vector<NodeTask> queues[2];
    queues[0].reserve(queueCap);
    queues[1].reserve(queueCap);

    const uint32_t emitCap = std::min(params.maxClusters, 1u << 20u);
    std::vector<EmittedCluster> emitted;
    emitted.reserve(emitCap);

    // --- Seed instances into queue 0 ---
    const uint32_t instanceCount = static_cast<uint32_t>(scene.instances.size());
    for (uint32_t iid = 0; iid < instanceCount; ++iid) {
        const GPUSceneInstance& inst = scene.instances[iid];
        if ((inst.visibilityFlags & 1u) == 0u) {
            continue;
        }
        if (inst.geometryIndex >= scene.geometries.size()) {
            continue;
        }
        const GPUSceneGeometry& geom = scene.geometries[inst.geometryIndex];
        if (geom.lodRootNode == kInvalidIndex || geom.lodRootNode >= lodData.nodes.size()) {
            ++stats.invalidRoot;
            continue;
        }
        if (queues[0].size() >= params.maxNodeTasks) {
            ++stats.nodeOverflow;
            continue;
        }
        NodeTask t;
        t.instanceID = iid;
        t.nodeID     = geom.lodRootNode;
        t.depth      = 0u;
        queues[0].push_back(t);
        ++stats.seededInstances;
    }

    // --- BFS iterations ---
    for (uint32_t iteration = 0; iteration < params.maxIterations; ++iteration) {
        const uint32_t inQ  = iteration & 1u;
        const uint32_t outQ = (iteration + 1u) & 1u;
        queues[outQ].clear();

        const uint32_t inputCount =
            static_cast<uint32_t>(std::min(queues[inQ].size(), size_t(params.maxNodeTasks)));
        if (inputCount == 0u) {
            break;
        }

        for (uint32_t ti = 0; ti < inputCount; ++ti) {
            const NodeTask task = queues[inQ][ti];
            if (task.instanceID >= scene.instances.size()) {
                continue;
            }
            const GPUSceneInstance& inst = scene.instances[task.instanceID];
            if ((inst.visibilityFlags & 1u) == 0u) {
                continue;
            }
            if (task.nodeID >= lodData.nodes.size()) {
                continue;
            }

            const GPULodNode& node = lodData.nodes[task.nodeID];
            ++stats.nodeProcessed;

            const float4x4 worldMatrix = loadMatrix(inst.worldMatrix);
            float3 nodeCenter(node.center[0], node.center[1], node.center[2]);

            if (node.representativeGroupCount != 0u &&
                cpuTestForLod(worldMatrix, nodeCenter, node.radius, node.maxError, params)) {
                emitNodeRepGroups(task.instanceID, node, task.depth,
                                  lodData, inst, params, stats, emitted);
                ++stats.lodCulled;
                continue;
            }

            if (node.isLeaf != 0u) {
                for (uint32_t child = 0; child < node.childCount; ++child) {
                    emitGroupClusters(task.instanceID,
                                      node.childOffset + child,
                                      task.depth,
                                      lodData, inst, params, stats, emitted);
                }
                continue;
            }

            for (uint32_t child = 0; child < node.childCount; ++child) {
                if (queues[outQ].size() >= params.maxNodeTasks) {
                    ++stats.nodeOverflow;
                    continue;
                }
                NodeTask childTask;
                childTask.instanceID = task.instanceID;
                childTask.nodeID     = node.childOffset + child;
                childTask.depth      = task.depth + 1u;
                queues[outQ].push_back(childTask);
            }
        }
    }

    // --- Post-process: duplicate and unexpected cluster detection ---
    std::vector<uint64_t> keys;
    keys.reserve(emitted.size());
    for (size_t ei = 0; ei < emitted.size(); ++ei) {
        keys.push_back((uint64_t(emitted[ei].instanceID) << 32u) |
                        uint64_t(emitted[ei].clusterID));
    }
    std::sort(keys.begin(), keys.end());
    for (size_t ki = 1; ki < keys.size(); ++ki) {
        if (keys[ki] == keys[ki - 1]) {
            ++stats.duplicateCount;
        }
    }

    for (size_t ei = 0; ei < emitted.size(); ++ei) {
        const uint32_t iid = emitted[ei].instanceID;
        const uint32_t cid = emitted[ei].clusterID;
        if (iid >= scene.instances.size()) {
            ++stats.unexpectedCount;
            continue;
        }
        const GPUSceneInstance& inst = scene.instances[iid];
        if (inst.geometryIndex >= scene.geometries.size()) {
            ++stats.unexpectedCount;
            continue;
        }
        const GPUSceneGeometry& geom = scene.geometries[inst.geometryIndex];
        if (cid < geom.packedClusterStart ||
            cid >= geom.packedClusterStart + geom.packedClusterCount) {
            ++stats.unexpectedCount;
        }
    }

    // --- Compare with GPU counters ---
    if (gpuStats && gpuStats->readable) {
        stats.gpuCompareAvailable  = true;
        stats.diffNodeProcessed    = int32_t(stats.nodeProcessed)   - int32_t(gpuStats->nodeProcessed);
        stats.diffLodCulled        = int32_t(stats.lodCulled)        - int32_t(gpuStats->lodCulled);
        stats.diffRefineAccepted   = int32_t(stats.refineAccepted)   - int32_t(gpuStats->refineAccepted);
        stats.diffRefineSuppressed = int32_t(stats.refineSuppressed) - int32_t(gpuStats->refineSuppressed);

        const uint32_t gpuEmitted = gpuStats->phase0Visible +
                                    gpuStats->phase0Recheck +
                                    gpuStats->frustumRejected;
        stats.diffClustersEmitted = int32_t(stats.clustersEmitted) - int32_t(gpuEmitted);

        stats.lodCountersMatch =
            stats.diffNodeProcessed    == 0 &&
            stats.diffLodCulled        == 0 &&
            stats.diffRefineAccepted   == 0 &&
            stats.diffRefineSuppressed == 0 &&
            stats.diffClustersEmitted  == 0;
    }

    return stats;
}

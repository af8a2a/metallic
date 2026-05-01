#pragma once

#include "cluster_lod_builder.h"
#include "cluster_occlusion_state.h"
#include "gpu_scene.h"
#include <array>
#include <cstdint>

// Parameters that mirror the GPU push constants for a single DAG cull pass.
struct DagCpuMirrorParams {
    float    viewProj[16];
    float    cameraWorldPos[3];
    float    cameraFovY;
    float    lodErrorThreshold;
    uint32_t screenHeight;
    uint32_t maxIterations;
    uint32_t maxNodeTasks;
    // Maximum clusters retained for duplicate/unexpected diagnostics.
    // clustersEmitted still counts every processCluster call.
    uint32_t maxClusters;
    bool     useInstanceVisibility = false;
};

struct DagCpuMirrorStats {
    bool available = false;

    // Traversal counters — should match GPU exactly (no HZB dependency).
    uint32_t seededInstances   = 0;
    uint32_t nodeProcessed     = 0;
    uint32_t lodCulled         = 0;
    uint32_t refineAccepted    = 0;
    uint32_t refineSuppressed  = 0;
    uint32_t invalidRoot       = 0;
    uint32_t nodeOverflow      = 0;
    uint32_t clusterOverflow   = 0;

    // Total clusters reaching processCluster (before frustum/HZB), not capped by
    // the retained diagnostics list.
    // GPU invariant: clustersEmitted == phase0Visible + phase0Recheck +
    // frustumRejected + clusterOverflow
    uint32_t clustersEmitted = 0;

    // Integrity checks on the CPU-predicted output set.
    uint32_t duplicateCount   = 0;
    uint32_t unexpectedCount  = 0; // cluster outside geometry's packed range

    // LOD distribution: clusters emitted per DAG traversal depth.
    static constexpr uint32_t kMaxDepthBuckets = 16u;
    std::array<uint32_t, kMaxDepthBuckets> clustersByDepth = {};
    uint32_t maxDepthSeen = 0;

    // Comparison against GPU counters (populated when gpuStats != nullptr).
    bool     gpuCompareAvailable  = false;
    int32_t  diffSeededInstances   = 0;
    int32_t  diffNodeProcessed    = 0;
    int32_t  diffLodCulled        = 0;
    int32_t  diffRefineAccepted   = 0;
    int32_t  diffRefineSuppressed = 0;
    // CPU clustersEmitted vs GPU retained/rejected/overflow processCluster attempts.
    int32_t  diffClustersEmitted  = 0;
    bool     lodCountersMatch     = false;
};

// Run a CPU mirror of the GPU DAG traversal (phase 0, no HZB).
// gpuStats: optional GPU counters to diff against; pass nullptr to skip.
DagCpuMirrorStats runDagCpuMirror(
    const GpuSceneTables&                        scene,
    const ClusterLODData&                        lodData,
    const DagCpuMirrorParams&                    params,
    const ClusterOcclusionState::DagCullStats*   gpuStats = nullptr);

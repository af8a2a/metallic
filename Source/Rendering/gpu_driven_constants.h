#ifndef GPU_DRIVEN_CONSTANTS_H
#define GPU_DRIVEN_CONSTANTS_H

// Shared layout for append/consume stages that publish both stats and a typed
// 1D indirect command into the same byte-address buffer.
//
// word 0: append/write cursor used by producers
// word 1: published produced item count from the last build step
// word 2: consumed item count (available for consumers or diagnostics)
// words 3..5: 1D indirect command payload
//
// The current compute, task, and mesh indirect entry points all consume the
// same 3 uint payload on Metallic's backends.
#define GPU_DRIVEN_WORKLIST_WRITE_CURSOR_OFFSET_BYTES 0u
#define GPU_DRIVEN_WORKLIST_PRODUCED_COUNT_OFFSET_BYTES 4u
#define GPU_DRIVEN_WORKLIST_CONSUMED_COUNT_OFFSET_BYTES 8u
#define GPU_DRIVEN_WORKLIST_INDIRECT_ARGS_OFFSET_BYTES 12u
#define GPU_DRIVEN_WORKLIST_DISPATCH_X_OFFSET_BYTES 12u
#define GPU_DRIVEN_WORKLIST_DISPATCH_Y_OFFSET_BYTES 16u
#define GPU_DRIVEN_WORKLIST_DISPATCH_Z_OFFSET_BYTES 20u
#define GPU_DRIVEN_WORKLIST_STATE_WORD_COUNT 6u
#define GPU_DRIVEN_WORKLIST_STATE_BUFFER_SIZE 24u

// Compatibility aliases for the original counter/indirect helpers.
#define GPU_DRIVEN_COUNT_OFFSET_BYTES GPU_DRIVEN_WORKLIST_WRITE_CURSOR_OFFSET_BYTES
#define GPU_DRIVEN_DISPATCH_ARGS_OFFSET_BYTES GPU_DRIVEN_WORKLIST_INDIRECT_ARGS_OFFSET_BYTES
#define GPU_DRIVEN_DISPATCH_X_OFFSET_BYTES GPU_DRIVEN_WORKLIST_DISPATCH_X_OFFSET_BYTES
#define GPU_DRIVEN_DISPATCH_Y_OFFSET_BYTES GPU_DRIVEN_WORKLIST_DISPATCH_Y_OFFSET_BYTES
#define GPU_DRIVEN_DISPATCH_Z_OFFSET_BYTES GPU_DRIVEN_WORKLIST_DISPATCH_Z_OFFSET_BYTES
#define GPU_DRIVEN_DISPATCH_COUNTER_WORD_COUNT GPU_DRIVEN_WORKLIST_STATE_WORD_COUNT
#define GPU_DRIVEN_DISPATCH_COUNTER_BUFFER_SIZE GPU_DRIVEN_WORKLIST_STATE_BUFFER_SIZE

// Shared bindings for the instance classification front-end.
#define GPU_DRIVEN_INSTANCE_CLASSIFY_UNIFORMS_BINDING 0u
#define GPU_DRIVEN_INSTANCE_CLASSIFY_INSTANCE_DATA_BINDING 1u
#define GPU_DRIVEN_INSTANCE_CLASSIFY_GEOMETRY_DATA_BINDING 2u
#define GPU_DRIVEN_INSTANCE_CLASSIFY_OUTPUT_BINDING 3u
#define GPU_DRIVEN_INSTANCE_CLASSIFY_STATE_BINDING 4u
#define GPU_DRIVEN_INSTANCE_CLASSIFY_HZB_TEXTURE_BINDING_BASE 5u

// Shared bindings for the meshlet cull compaction pipeline.
#define GPU_DRIVEN_CULL_UNIFORMS_BINDING 0u
#define GPU_DRIVEN_CULL_INSTANCE_DATA_BINDING 1u
#define GPU_DRIVEN_CULL_GEOMETRY_DATA_BINDING 2u
#define GPU_DRIVEN_CULL_BOUNDS_BINDING 3u
#define GPU_DRIVEN_CULL_VISIBLE_INSTANCES_BINDING 4u
#define GPU_DRIVEN_CULL_COMPACTION_OUTPUT_BINDING 5u
#define GPU_DRIVEN_CULL_COUNTER_BINDING 6u
#define GPU_DRIVEN_CULL_LOD_NODE_BINDING 7u
#define GPU_DRIVEN_CULL_LOD_GROUP_BINDING 8u
#define GPU_DRIVEN_CULL_LOD_GROUP_MESHLET_INDICES_BINDING 9u
#define GPU_DRIVEN_CULL_LOD_BOUNDS_BINDING 10u
#define GPU_DRIVEN_CULL_TRAVERSAL_STATS_BINDING 11u
#define GPU_DRIVEN_CULL_GROUP_RESIDENCY_BINDING 12u
#define GPU_DRIVEN_CULL_LOD_NODE_RESIDENCY_BINDING GPU_DRIVEN_CULL_GROUP_RESIDENCY_BINDING
#define GPU_DRIVEN_CULL_LOD_GROUP_PAGE_TABLE_BINDING 13u
#define GPU_DRIVEN_CULL_RESIDENCY_REQUEST_OUTPUT_BINDING 14u
#define GPU_DRIVEN_CULL_RESIDENCY_REQUEST_STATE_BINDING 15u
#define GPU_DRIVEN_CULL_LOD_GROUP_MESHLET_INDICES_SOURCE_BINDING 16u
#define GPU_DRIVEN_CULL_GROUP_AGE_BINDING 17u
#define GPU_DRIVEN_CULL_HZB_TEXTURE_BINDING_BASE 18u
#define GPU_DRIVEN_HZB_TEXTURE_BINDING_COUNT 10u
#define GPU_DRIVEN_CULL_CURRENT_HZB_TEXTURE_BINDING_BASE \
    (GPU_DRIVEN_CULL_HZB_TEXTURE_BINDING_BASE + GPU_DRIVEN_HZB_TEXTURE_BINDING_COUNT)

// Shared bindings for the streaming age filter pipeline.
#define GPU_DRIVEN_STREAMING_AGE_UNIFORMS_BINDING 0u
#define GPU_DRIVEN_STREAMING_AGE_GROUP_RESIDENCY_BINDING 1u
#define GPU_DRIVEN_STREAMING_AGE_GROUP_AGE_BINDING 2u
#define GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_OUTPUT_BINDING 3u
#define GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_STATE_BINDING 4u
#define GPU_DRIVEN_STREAMING_AGE_STATS_BINDING 5u
#define GPU_DRIVEN_STREAMING_AGE_ACTIVE_RESIDENT_GROUPS_BINDING 6u

// Shared bindings for the streaming scene update pipeline.
#define GPU_DRIVEN_STREAMING_UPDATE_UNIFORMS_BINDING 0u
#define GPU_DRIVEN_STREAMING_UPDATE_SOURCE_GROUP_MESHLET_INDICES_BINDING 1u
#define GPU_DRIVEN_STREAMING_UPDATE_RESIDENT_GROUP_MESHLET_INDICES_BINDING 2u
#define GPU_DRIVEN_STREAMING_UPDATE_GROUP_PAGE_TABLE_BINDING 3u
#define GPU_DRIVEN_STREAMING_UPDATE_PATCHES_BINDING 4u
#define GPU_DRIVEN_STREAMING_UPDATE_STATS_BINDING 5u
#define GPU_DRIVEN_STREAMING_UPDATE_ACTIVE_RESIDENT_GROUPS_BINDING 6u
#define GPU_DRIVEN_STREAMING_UPDATE_ACTIVE_RESIDENT_PATCHES_BINDING 7u
#define GPU_DRIVEN_STREAMING_UPDATE_GROUP_RESIDENCY_BINDING 8u
#define GPU_DRIVEN_STREAMING_UPDATE_GROUP_AGE_BINDING 9u

// Shared bindings for meshlet visibility pipelines.
#define GPU_DRIVEN_VISIBILITY_GLOBAL_UNIFORMS_BINDING 0u
#define GPU_DRIVEN_VISIBILITY_POSITION_BINDING 1u
#define GPU_DRIVEN_VISIBILITY_NORMAL_BINDING 2u
#define GPU_DRIVEN_VISIBILITY_MESHLET_BINDING 3u
#define GPU_DRIVEN_VISIBILITY_MESHLET_VERTICES_BINDING 4u
#define GPU_DRIVEN_VISIBILITY_MESHLET_TRIANGLES_BINDING 5u
#define GPU_DRIVEN_VISIBILITY_BOUNDS_BINDING 6u
#define GPU_DRIVEN_VISIBILITY_UVS_BINDING 7u
#define GPU_DRIVEN_VISIBILITY_MATERIAL_IDS_BINDING 8u
#define GPU_DRIVEN_VISIBILITY_MATERIAL_BUFFER_BINDING 9u
#define GPU_DRIVEN_VISIBILITY_VISIBLE_MESHLETS_BINDING 10u
#define GPU_DRIVEN_VISIBILITY_INSTANCE_DATA_BINDING 11u
#define GPU_DRIVEN_VISIBILITY_LOD_MESHLET_BINDING 12u
#define GPU_DRIVEN_VISIBILITY_LOD_MESHLET_VERTICES_BINDING 13u
#define GPU_DRIVEN_VISIBILITY_LOD_MESHLET_TRIANGLES_BINDING 14u
#define GPU_DRIVEN_VISIBILITY_LOD_MATERIAL_IDS_BINDING 15u

// Shared bindings for deferred lighting when visibility encodes visible-worklist IDs.
#define GPU_DRIVEN_DEFERRED_VISIBLE_MESHLETS_BINDING 9u
#define GPU_DRIVEN_DEFERRED_VISIBLE_MESHLET_STATE_BINDING 10u
#define GPU_DRIVEN_DEFERRED_LOD_MESHLET_BINDING 11u
#define GPU_DRIVEN_DEFERRED_LOD_MESHLET_VERTICES_BINDING 12u
#define GPU_DRIVEN_DEFERRED_LOD_MESHLET_TRIANGLES_BINDING 13u
#define GPU_DRIVEN_DEFERRED_LOD_MATERIAL_IDS_BINDING 14u
#define GPU_DRIVEN_DEFERRED_INSTANCE_DATA_BINDING 15u

// Shared bindings for helper passes that convert counters into indirect args.
#define GPU_DRIVEN_BUILD_DISPATCH_COUNTER_BINDING 0u

// Shared bindings for the cluster visualization render pass.
#define GPU_DRIVEN_CLUSTER_VIS_UNIFORMS_BINDING 0u
#define GPU_DRIVEN_CLUSTER_VIS_CLUSTER_INFOS_BINDING 1u
#define GPU_DRIVEN_CLUSTER_VIS_PACKED_CLUSTERS_BINDING 2u
#define GPU_DRIVEN_CLUSTER_VIS_VERTEX_DATA_BINDING 3u
#define GPU_DRIVEN_CLUSTER_VIS_INDEX_DATA_BINDING 4u
#define GPU_DRIVEN_CLUSTER_VIS_INSTANCE_DATA_BINDING 5u

#ifdef __cplusplus

#include <cstdint>

namespace GpuDriven {

struct WorklistStateHeaderLayout {
    static constexpr uint32_t kWriteCursorOffset = GPU_DRIVEN_WORKLIST_WRITE_CURSOR_OFFSET_BYTES;
    static constexpr uint32_t kProducedCountOffset = GPU_DRIVEN_WORKLIST_PRODUCED_COUNT_OFFSET_BYTES;
    static constexpr uint32_t kConsumedCountOffset = GPU_DRIVEN_WORKLIST_CONSUMED_COUNT_OFFSET_BYTES;
    static constexpr uint32_t kIndirectArgsOffset = GPU_DRIVEN_WORKLIST_INDIRECT_ARGS_OFFSET_BYTES;
    static constexpr uint32_t kHeaderWordCount = kIndirectArgsOffset / sizeof(uint32_t);
    static constexpr uint32_t kWriteCursorWord = kWriteCursorOffset / sizeof(uint32_t);
    static constexpr uint32_t kProducedCountWord = kProducedCountOffset / sizeof(uint32_t);
    static constexpr uint32_t kConsumedCountWord = kConsumedCountOffset / sizeof(uint32_t);
};

struct ComputeDispatchCommandLayout : WorklistStateHeaderLayout {
    static constexpr uint32_t kDispatchXOffset = GPU_DRIVEN_WORKLIST_DISPATCH_X_OFFSET_BYTES;
    static constexpr uint32_t kDispatchYOffset = GPU_DRIVEN_WORKLIST_DISPATCH_Y_OFFSET_BYTES;
    static constexpr uint32_t kDispatchZOffset = GPU_DRIVEN_WORKLIST_DISPATCH_Z_OFFSET_BYTES;
    static constexpr uint32_t kDispatchXWord = kDispatchXOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchYWord = kDispatchYOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchZWord = kDispatchZOffset / sizeof(uint32_t);
    static constexpr uint32_t kCommandWordCount = 3u;
    static constexpr uint32_t kWordCount = GPU_DRIVEN_WORKLIST_STATE_WORD_COUNT;
    static constexpr uint32_t kBufferSize = GPU_DRIVEN_WORKLIST_STATE_BUFFER_SIZE;
};

struct TaskDispatchCommandLayout : WorklistStateHeaderLayout {
    static constexpr uint32_t kDispatchXOffset = GPU_DRIVEN_WORKLIST_DISPATCH_X_OFFSET_BYTES;
    static constexpr uint32_t kDispatchYOffset = GPU_DRIVEN_WORKLIST_DISPATCH_Y_OFFSET_BYTES;
    static constexpr uint32_t kDispatchZOffset = GPU_DRIVEN_WORKLIST_DISPATCH_Z_OFFSET_BYTES;
    static constexpr uint32_t kDispatchXWord = kDispatchXOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchYWord = kDispatchYOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchZWord = kDispatchZOffset / sizeof(uint32_t);
    static constexpr uint32_t kCommandWordCount = 3u;
    static constexpr uint32_t kWordCount = GPU_DRIVEN_WORKLIST_STATE_WORD_COUNT;
    static constexpr uint32_t kBufferSize = GPU_DRIVEN_WORKLIST_STATE_BUFFER_SIZE;
};

struct MeshDispatchCommandLayout : WorklistStateHeaderLayout {
    static constexpr uint32_t kDispatchXOffset = GPU_DRIVEN_WORKLIST_DISPATCH_X_OFFSET_BYTES;
    static constexpr uint32_t kDispatchYOffset = GPU_DRIVEN_WORKLIST_DISPATCH_Y_OFFSET_BYTES;
    static constexpr uint32_t kDispatchZOffset = GPU_DRIVEN_WORKLIST_DISPATCH_Z_OFFSET_BYTES;
    static constexpr uint32_t kDispatchXWord = kDispatchXOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchYWord = kDispatchYOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchZWord = kDispatchZOffset / sizeof(uint32_t);
    static constexpr uint32_t kCommandWordCount = 3u;
    static constexpr uint32_t kWordCount = GPU_DRIVEN_WORKLIST_STATE_WORD_COUNT;
    static constexpr uint32_t kBufferSize = GPU_DRIVEN_WORKLIST_STATE_BUFFER_SIZE;
};

using IndirectGridCommandLayout = ComputeDispatchCommandLayout;
using DispatchCounterLayout = ComputeDispatchCommandLayout;

struct InstanceClassifyBindings {
    static constexpr uint32_t kUniforms = GPU_DRIVEN_INSTANCE_CLASSIFY_UNIFORMS_BINDING;
    static constexpr uint32_t kInstances = GPU_DRIVEN_INSTANCE_CLASSIFY_INSTANCE_DATA_BINDING;
    static constexpr uint32_t kGeometries = GPU_DRIVEN_INSTANCE_CLASSIFY_GEOMETRY_DATA_BINDING;
    static constexpr uint32_t kOutput = GPU_DRIVEN_INSTANCE_CLASSIFY_OUTPUT_BINDING;
    static constexpr uint32_t kState = GPU_DRIVEN_INSTANCE_CLASSIFY_STATE_BINDING;
    static constexpr uint32_t kHzbTextureBase = GPU_DRIVEN_INSTANCE_CLASSIFY_HZB_TEXTURE_BINDING_BASE;
};

struct MeshletCullBindings {
    static constexpr uint32_t kUniforms = GPU_DRIVEN_CULL_UNIFORMS_BINDING;
    static constexpr uint32_t kInstances = GPU_DRIVEN_CULL_INSTANCE_DATA_BINDING;
    static constexpr uint32_t kGeometries = GPU_DRIVEN_CULL_GEOMETRY_DATA_BINDING;
    static constexpr uint32_t kBounds = GPU_DRIVEN_CULL_BOUNDS_BINDING;
    static constexpr uint32_t kVisibleInstances = GPU_DRIVEN_CULL_VISIBLE_INSTANCES_BINDING;
    static constexpr uint32_t kCompactionOutput = GPU_DRIVEN_CULL_COMPACTION_OUTPUT_BINDING;
    static constexpr uint32_t kCounter = GPU_DRIVEN_CULL_COUNTER_BINDING;
    static constexpr uint32_t kLodNodes = GPU_DRIVEN_CULL_LOD_NODE_BINDING;
    static constexpr uint32_t kLodGroups = GPU_DRIVEN_CULL_LOD_GROUP_BINDING;
    static constexpr uint32_t kLodGroupMeshletIndices = GPU_DRIVEN_CULL_LOD_GROUP_MESHLET_INDICES_BINDING;
    static constexpr uint32_t kLodBounds = GPU_DRIVEN_CULL_LOD_BOUNDS_BINDING;
    static constexpr uint32_t kTraversalStats = GPU_DRIVEN_CULL_TRAVERSAL_STATS_BINDING;
    static constexpr uint32_t kGroupResidency = GPU_DRIVEN_CULL_GROUP_RESIDENCY_BINDING;
    static constexpr uint32_t kLodNodeResidency = kGroupResidency;
    static constexpr uint32_t kLodGroupPageTable = GPU_DRIVEN_CULL_LOD_GROUP_PAGE_TABLE_BINDING;
    static constexpr uint32_t kResidencyRequests = GPU_DRIVEN_CULL_RESIDENCY_REQUEST_OUTPUT_BINDING;
    static constexpr uint32_t kResidencyRequestState = GPU_DRIVEN_CULL_RESIDENCY_REQUEST_STATE_BINDING;
    static constexpr uint32_t kLodGroupMeshletIndicesSource =
        GPU_DRIVEN_CULL_LOD_GROUP_MESHLET_INDICES_SOURCE_BINDING;
    static constexpr uint32_t kGroupAge = GPU_DRIVEN_CULL_GROUP_AGE_BINDING;
    static constexpr uint32_t kHzbTextureBase = GPU_DRIVEN_CULL_HZB_TEXTURE_BINDING_BASE;
    static constexpr uint32_t kCurrentHzbTextureBase =
        GPU_DRIVEN_CULL_CURRENT_HZB_TEXTURE_BINDING_BASE;
    static constexpr uint32_t kInstanceData = kInstances;
};

struct StreamingAgeFilterBindings {
    static constexpr uint32_t kUniforms = GPU_DRIVEN_STREAMING_AGE_UNIFORMS_BINDING;
    static constexpr uint32_t kGroupResidency = GPU_DRIVEN_STREAMING_AGE_GROUP_RESIDENCY_BINDING;
    static constexpr uint32_t kGroupAge = GPU_DRIVEN_STREAMING_AGE_GROUP_AGE_BINDING;
    static constexpr uint32_t kUnloadRequests =
        GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_OUTPUT_BINDING;
    static constexpr uint32_t kUnloadRequestState =
        GPU_DRIVEN_STREAMING_AGE_UNLOAD_REQUEST_STATE_BINDING;
    static constexpr uint32_t kStats = GPU_DRIVEN_STREAMING_AGE_STATS_BINDING;
    static constexpr uint32_t kActiveResidentGroups =
        GPU_DRIVEN_STREAMING_AGE_ACTIVE_RESIDENT_GROUPS_BINDING;
};

struct StreamingUpdateBindings {
    static constexpr uint32_t kUniforms = GPU_DRIVEN_STREAMING_UPDATE_UNIFORMS_BINDING;
    static constexpr uint32_t kSourceGroupMeshletIndices =
        GPU_DRIVEN_STREAMING_UPDATE_SOURCE_GROUP_MESHLET_INDICES_BINDING;
    static constexpr uint32_t kResidentGroupMeshletIndices =
        GPU_DRIVEN_STREAMING_UPDATE_RESIDENT_GROUP_MESHLET_INDICES_BINDING;
    static constexpr uint32_t kGroupPageTable =
        GPU_DRIVEN_STREAMING_UPDATE_GROUP_PAGE_TABLE_BINDING;
    static constexpr uint32_t kPatches = GPU_DRIVEN_STREAMING_UPDATE_PATCHES_BINDING;
    static constexpr uint32_t kStats = GPU_DRIVEN_STREAMING_UPDATE_STATS_BINDING;
    static constexpr uint32_t kActiveResidentGroups =
        GPU_DRIVEN_STREAMING_UPDATE_ACTIVE_RESIDENT_GROUPS_BINDING;
    static constexpr uint32_t kActiveResidentPatches =
        GPU_DRIVEN_STREAMING_UPDATE_ACTIVE_RESIDENT_PATCHES_BINDING;
    static constexpr uint32_t kGroupResidency =
        GPU_DRIVEN_STREAMING_UPDATE_GROUP_RESIDENCY_BINDING;
    static constexpr uint32_t kGroupAge = GPU_DRIVEN_STREAMING_UPDATE_GROUP_AGE_BINDING;
};

struct MeshletVisibilityBindings {
    static constexpr uint32_t kGlobalUniforms = GPU_DRIVEN_VISIBILITY_GLOBAL_UNIFORMS_BINDING;
    static constexpr uint32_t kPositions = GPU_DRIVEN_VISIBILITY_POSITION_BINDING;
    static constexpr uint32_t kNormals = GPU_DRIVEN_VISIBILITY_NORMAL_BINDING;
    static constexpr uint32_t kMeshlets = GPU_DRIVEN_VISIBILITY_MESHLET_BINDING;
    static constexpr uint32_t kMeshletVertices = GPU_DRIVEN_VISIBILITY_MESHLET_VERTICES_BINDING;
    static constexpr uint32_t kMeshletTriangles = GPU_DRIVEN_VISIBILITY_MESHLET_TRIANGLES_BINDING;
    static constexpr uint32_t kBounds = GPU_DRIVEN_VISIBILITY_BOUNDS_BINDING;
    static constexpr uint32_t kUvs = GPU_DRIVEN_VISIBILITY_UVS_BINDING;
    static constexpr uint32_t kMaterialIds = GPU_DRIVEN_VISIBILITY_MATERIAL_IDS_BINDING;
    static constexpr uint32_t kMaterials = GPU_DRIVEN_VISIBILITY_MATERIAL_BUFFER_BINDING;
    static constexpr uint32_t kVisibleMeshlets = GPU_DRIVEN_VISIBILITY_VISIBLE_MESHLETS_BINDING;
    static constexpr uint32_t kSceneInstances = GPU_DRIVEN_VISIBILITY_INSTANCE_DATA_BINDING;
    static constexpr uint32_t kLodMeshlets = GPU_DRIVEN_VISIBILITY_LOD_MESHLET_BINDING;
    static constexpr uint32_t kLodMeshletVertices = GPU_DRIVEN_VISIBILITY_LOD_MESHLET_VERTICES_BINDING;
    static constexpr uint32_t kLodMeshletTriangles = GPU_DRIVEN_VISIBILITY_LOD_MESHLET_TRIANGLES_BINDING;
    static constexpr uint32_t kLodMaterialIds = GPU_DRIVEN_VISIBILITY_LOD_MATERIAL_IDS_BINDING;
    static constexpr uint32_t kInstanceData = kSceneInstances;
};

struct DeferredLightingBindings {
    static constexpr uint32_t kPositions = 1u;
    static constexpr uint32_t kNormals = 2u;
    static constexpr uint32_t kMeshlets = 3u;
    static constexpr uint32_t kMeshletVertices = 4u;
    static constexpr uint32_t kMeshletTriangles = 5u;
    static constexpr uint32_t kUvs = 6u;
    static constexpr uint32_t kMaterialIds = 7u;
    static constexpr uint32_t kMaterials = 8u;
    static constexpr uint32_t kVisibleMeshlets = GPU_DRIVEN_DEFERRED_VISIBLE_MESHLETS_BINDING;
    static constexpr uint32_t kVisibleMeshletState = GPU_DRIVEN_DEFERRED_VISIBLE_MESHLET_STATE_BINDING;
    static constexpr uint32_t kLodMeshlets = GPU_DRIVEN_DEFERRED_LOD_MESHLET_BINDING;
    static constexpr uint32_t kLodMeshletVertices = GPU_DRIVEN_DEFERRED_LOD_MESHLET_VERTICES_BINDING;
    static constexpr uint32_t kLodMeshletTriangles = GPU_DRIVEN_DEFERRED_LOD_MESHLET_TRIANGLES_BINDING;
    static constexpr uint32_t kLodMaterialIds = GPU_DRIVEN_DEFERRED_LOD_MATERIAL_IDS_BINDING;
    static constexpr uint32_t kInstanceData = GPU_DRIVEN_DEFERRED_INSTANCE_DATA_BINDING;
};

struct BuildWorklistBindings {
    static constexpr uint32_t kState = GPU_DRIVEN_BUILD_DISPATCH_COUNTER_BINDING;
};

using BuildDispatchBindings = BuildWorklistBindings;

struct ClusterRenderBindings {
    static constexpr uint32_t kUniforms = GPU_DRIVEN_CLUSTER_VIS_UNIFORMS_BINDING;
    static constexpr uint32_t kClusterInfos = GPU_DRIVEN_CLUSTER_VIS_CLUSTER_INFOS_BINDING;
    static constexpr uint32_t kClusters = GPU_DRIVEN_CLUSTER_VIS_PACKED_CLUSTERS_BINDING;
    static constexpr uint32_t kVertexData = GPU_DRIVEN_CLUSTER_VIS_VERTEX_DATA_BINDING;
    static constexpr uint32_t kIndexData = GPU_DRIVEN_CLUSTER_VIS_INDEX_DATA_BINDING;
    static constexpr uint32_t kInstances = GPU_DRIVEN_CLUSTER_VIS_INSTANCE_DATA_BINDING;
};

static_assert(ComputeDispatchCommandLayout::kBufferSize ==
              ComputeDispatchCommandLayout::kWordCount * sizeof(uint32_t));
static_assert(TaskDispatchCommandLayout::kBufferSize ==
              TaskDispatchCommandLayout::kWordCount * sizeof(uint32_t));
static_assert(MeshDispatchCommandLayout::kBufferSize ==
              MeshDispatchCommandLayout::kWordCount * sizeof(uint32_t));

} // namespace GpuDriven

#endif

#endif

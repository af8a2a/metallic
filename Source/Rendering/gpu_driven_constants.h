#ifndef GPU_DRIVEN_CONSTANTS_H
#define GPU_DRIVEN_CONSTANTS_H

// Shared layout for append/compaction/prefix-sum stages that publish a uint32 count
// plus a 1D indirect grid command into the same byte-address buffer.
// The 3 uint grid payload is valid for both compute dispatch indirect and
// mesh-shader draw indirect on the current backends.
#define GPU_DRIVEN_COUNT_OFFSET_BYTES 0u
#define GPU_DRIVEN_DISPATCH_ARGS_OFFSET_BYTES 4u
#define GPU_DRIVEN_DISPATCH_X_OFFSET_BYTES 4u
#define GPU_DRIVEN_DISPATCH_Y_OFFSET_BYTES 8u
#define GPU_DRIVEN_DISPATCH_Z_OFFSET_BYTES 12u
#define GPU_DRIVEN_DISPATCH_COUNTER_WORD_COUNT 4u
#define GPU_DRIVEN_DISPATCH_COUNTER_BUFFER_SIZE 16u

// Shared bindings for the meshlet cull compaction pipeline.
#define GPU_DRIVEN_CULL_UNIFORMS_BINDING 0u
#define GPU_DRIVEN_CULL_INSTANCE_DATA_BINDING 1u
#define GPU_DRIVEN_CULL_GEOMETRY_DATA_BINDING 2u
#define GPU_DRIVEN_CULL_BOUNDS_BINDING 3u
#define GPU_DRIVEN_CULL_COMPACTION_OUTPUT_BINDING 4u
#define GPU_DRIVEN_CULL_COUNTER_BINDING 5u
#define GPU_DRIVEN_CULL_HZB_TEXTURE_BINDING_BASE 6u

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

// Shared bindings for helper passes that convert counters into indirect args.
#define GPU_DRIVEN_BUILD_DISPATCH_COUNTER_BINDING 0u

#ifdef __cplusplus

#include <cstdint>

namespace GpuDriven {

struct IndirectGridCommandLayout {
    static constexpr uint32_t kCountOffset = GPU_DRIVEN_COUNT_OFFSET_BYTES;
    static constexpr uint32_t kIndirectArgsOffset = GPU_DRIVEN_DISPATCH_ARGS_OFFSET_BYTES;
    static constexpr uint32_t kDispatchXOffset = GPU_DRIVEN_DISPATCH_X_OFFSET_BYTES;
    static constexpr uint32_t kDispatchYOffset = GPU_DRIVEN_DISPATCH_Y_OFFSET_BYTES;
    static constexpr uint32_t kDispatchZOffset = GPU_DRIVEN_DISPATCH_Z_OFFSET_BYTES;
    static constexpr uint32_t kWordCount = GPU_DRIVEN_DISPATCH_COUNTER_WORD_COUNT;
    static constexpr uint32_t kBufferSize = GPU_DRIVEN_DISPATCH_COUNTER_BUFFER_SIZE;
    static constexpr uint32_t kCountWord = kCountOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchXWord = kDispatchXOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchYWord = kDispatchYOffset / sizeof(uint32_t);
    static constexpr uint32_t kDispatchZWord = kDispatchZOffset / sizeof(uint32_t);
};

using DispatchCounterLayout = IndirectGridCommandLayout;

struct MeshletCullBindings {
    static constexpr uint32_t kUniforms = GPU_DRIVEN_CULL_UNIFORMS_BINDING;
    static constexpr uint32_t kInstances = GPU_DRIVEN_CULL_INSTANCE_DATA_BINDING;
    static constexpr uint32_t kGeometries = GPU_DRIVEN_CULL_GEOMETRY_DATA_BINDING;
    static constexpr uint32_t kBounds = GPU_DRIVEN_CULL_BOUNDS_BINDING;
    static constexpr uint32_t kCompactionOutput = GPU_DRIVEN_CULL_COMPACTION_OUTPUT_BINDING;
    static constexpr uint32_t kCounter = GPU_DRIVEN_CULL_COUNTER_BINDING;
    static constexpr uint32_t kHzbTextureBase = GPU_DRIVEN_CULL_HZB_TEXTURE_BINDING_BASE;
    static constexpr uint32_t kInstanceData = kInstances;
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
    static constexpr uint32_t kInstanceData = kSceneInstances;
};

struct BuildDispatchBindings {
    static constexpr uint32_t kCounter = GPU_DRIVEN_BUILD_DISPATCH_COUNTER_BINDING;
};

static_assert(IndirectGridCommandLayout::kBufferSize ==
              IndirectGridCommandLayout::kWordCount * sizeof(uint32_t));

} // namespace GpuDriven

#endif

#endif

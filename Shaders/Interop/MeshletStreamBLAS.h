#ifndef METALLIC_MESHLET_STREAM_BLAS_INTEROP
#define METALLIC_MESHLET_STREAM_BLAS_INTEROP

// Shared by the C++ header layout and GPU history consumers.
#define METALLIC_STREAM_BLAS_SEMANTIC_CHANGE_WORD 7u

// Asset cooking/loading validates this group size; published CLAS IDs use it
// as the page stride. Keep the CPU asset contract and ray decoder in sync.
#define METALLIC_STREAM_CLAS_CLUSTER_SHIFT 5u
#define METALLIC_STREAM_CLAS_CLUSTER_STRIDE (1u << METALLIC_STREAM_CLAS_CLUSTER_SHIFT)

#endif

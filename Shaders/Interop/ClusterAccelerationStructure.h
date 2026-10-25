#ifndef METALLIC_CLUSTER_ACCELERATION_STRUCTURE_INTEROP
#define METALLIC_CLUSTER_ACCELERATION_STRUCTURE_INTEROP

// Native triangle-build record consumed by indirect CLAS commands. Packed
// fields avoid C++ bitfield layout rules in GPU producers.
#define METALLIC_CLAS_INDEX_UINT8 1u
#define METALLIC_CLAS_INDEX_UINT16 2u
#define METALLIC_CLAS_INDEX_UINT32 4u
#define METALLIC_CLAS_GEOMETRY_OPAQUE (4u << 29u)
#define METALLIC_CLAS_PACK_COUNTS(triangles, vertices, truncateBits, indexType) \
    ((triangles) | ((vertices) << 9u) | ((truncateBits) << 18u) | ((indexType) << 24u))
#define METALLIC_CLAS_PACK_STRIDES(indexStride, vertexStride) \
    ((indexStride) | ((vertexStride) << 16u))

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
namespace metallic::render {
#define METALLIC_CLAS_PUBLIC
#else
#define METALLIC_CLAS_PUBLIC public
#endif

METALLIC_CLAS_PUBLIC struct ClusterAccelerationStructureTriangleIndirectBuildInfo
{
    METALLIC_CLAS_PUBLIC uint32_t clusterId;
    METALLIC_CLAS_PUBLIC uint32_t clusterFlags;
    METALLIC_CLAS_PUBLIC uint32_t countsAndIndexType;
    METALLIC_CLAS_PUBLIC uint32_t geometryIndexAndFlags;
    METALLIC_CLAS_PUBLIC uint32_t indexAndVertexStrides;
    // This API uses one geometry index per cluster and no opacity micromaps.
    // Producers set the remaining auxiliary fields and addresses to zero.
    METALLIC_CLAS_PUBLIC uint32_t auxiliaryStrides;
    METALLIC_CLAS_PUBLIC uint64_t indexBufferAddress;
    METALLIC_CLAS_PUBLIC uint64_t vertexBufferAddress;
    METALLIC_CLAS_PUBLIC uint64_t geometryIndexAndFlagsBufferAddress;
    METALLIC_CLAS_PUBLIC uint64_t opacityMicromapArrayAddress;
    METALLIC_CLAS_PUBLIC uint64_t opacityMicromapIndexBufferAddress;
};

#ifdef __cplusplus
static_assert(sizeof(ClusterAccelerationStructureTriangleIndirectBuildInfo) == 64);
static_assert(offsetof(ClusterAccelerationStructureTriangleIndirectBuildInfo, indexBufferAddress) == 24);
static_assert(offsetof(ClusterAccelerationStructureTriangleIndirectBuildInfo, vertexBufferAddress) == 32);
} // namespace metallic::render
#endif
#undef METALLIC_CLAS_PUBLIC

#endif

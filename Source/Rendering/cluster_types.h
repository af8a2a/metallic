#pragma once

#include <cfloat>
#include <cstdint>

// vk_lod_clusters-compatible data structures for GPU-driven cluster rendering.
// C++ mirrors of Shaders/Shared/cluster_scene.slang.

struct PackedCluster {
    uint8_t  triCountM1;
    uint8_t  vtxCountM1;
    uint8_t  lodLevel;
    uint8_t  groupChildIndex;
    uint8_t  attributeBits;
    uint8_t  localMaterialID;
    uint16_t reserved;
    uint32_t vertexByteOffset;
    uint32_t indexByteOffset;
};
static_assert(sizeof(PackedCluster) == 16, "PackedCluster must be 16 bytes");

struct TraversalMetric {
    float sphereX, sphereY, sphereZ, sphereRadius;
    float maxQuadricError;
    float parentError;
};
static_assert(sizeof(TraversalMetric) == 24, "TraversalMetric must be 24 bytes");

struct PackedGroup {
    TraversalMetric metric;
    uint32_t clusterStart;
    uint32_t clusterCount;
};
static_assert(sizeof(PackedGroup) == 32, "PackedGroup must be 32 bytes");

struct PackedNode {
    uint32_t packed;
    float _pad[3];
    TraversalMetric metric;
};
static_assert(sizeof(PackedNode) == 40, "PackedNode must be 40 bytes");

namespace PackedNodeBits {
    inline bool isGroup(uint32_t packed) { return (packed & 1u) != 0; }
    inline uint32_t childOffset(uint32_t packed) { return (packed >> 1) & 0x3FFFFFFu; }
    inline uint32_t childCount(uint32_t packed) { return ((packed >> 27) & 0x1Fu) + 1u; }
    inline uint32_t encode(bool isGroup, uint32_t offset, uint32_t count) {
        return (isGroup ? 1u : 0u)
             | ((offset & 0x3FFFFFFu) << 1)
             | (((count - 1u) & 0x1Fu) << 27);
    }
}

struct ClusterInfo {
    uint32_t instanceID;
    uint32_t clusterID;
};
static_assert(sizeof(ClusterInfo) == 8, "ClusterInfo must be 8 bytes");

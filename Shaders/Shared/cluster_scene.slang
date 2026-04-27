// vk_lod_clusters-compatible data structures for GPU-driven cluster rendering.
// Slang mirror of Source/Rendering/cluster_types.h.

struct PackedCluster {
    uint word0;             // [7:0]=triCountM1 [15:8]=vtxCountM1 [23:16]=lodLevel [31:24]=groupChildIndex
    uint word1;             // [7:0]=attributeBits [15:8]=localMaterialID [31:16]=reserved
    uint vertexByteOffset;
    uint indexByteOffset;
};

uint clusterTriCount(PackedCluster c)  { return (c.word0 & 0xFFu) + 1u; }
uint clusterVtxCount(PackedCluster c)  { return ((c.word0 >> 8u) & 0xFFu) + 1u; }
uint clusterLodLevel(PackedCluster c)  { return (c.word0 >> 16u) & 0xFFu; }
uint clusterGroupIdx(PackedCluster c)  { return (c.word0 >> 24u) & 0xFFu; }
uint clusterMaterialID(PackedCluster c){ return (c.word1 >> 8u) & 0xFFu; }

struct TraversalMetric {
    float sphereX, sphereY, sphereZ, sphereRadius;
    float maxQuadricError;
    float parentError;
};

struct PackedGroup {
    TraversalMetric metric;
    uint   clusterStart;
    uint   clusterCount;
};

struct PackedNode {
    uint packed;
    float3 _pad;
    TraversalMetric metric;
};

bool  nodeIsGroup(PackedNode n)     { return (n.packed & 1u) != 0u; }
uint  nodeChildOffset(PackedNode n) { return (n.packed >> 1u) & 0x3FFFFFFu; }
uint  nodeChildCount(PackedNode n)  { return ((n.packed >> 27u) & 0x1Fu) + 1u; }

struct ClusterInfo {
    uint instanceID;
    uint clusterID;
};

// Integer hash (Wang) for cluster visualization
float3 hashColor(uint id) {
    uint h = id;
    h = (h ^ 61u) ^ (h >> 16u);
    h = h + (h << 3u);
    h = h ^ (h >> 4u);
    h = h * 0x27d4eb2du;
    h = h ^ (h >> 15u);
    float r = float((h >>  0u) & 0xFFu) / 255.0;
    float g = float((h >>  8u) & 0xFFu) / 255.0;
    float b = float((h >> 16u) & 0xFFu) / 255.0;
    float mx = max(r, max(g, b));
    if (mx > 0.0) { r /= mx; g /= mx; b /= mx; }
    return float3(r, g, b);
}

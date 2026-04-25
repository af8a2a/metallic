// Cluster visualization mesh shader — vk_lod_clusters-compatible.
// Each threadgroup renders one cluster from the ClusterInfo worklist.

#include "../Shared/cluster_scene.slang"

struct ClusterVisUniforms {
    float4x4 viewProj;
    float3   lightDir;
    uint     colorMode;   // 0=Cluster, 1=Instance, 2=LOD, 3=Triangle
};

struct InstanceData {
    float4x4 worldMatrix;
    float4x4 prevWorldMatrix;
    uint geometryIndex;
    uint dispatchStart;
    uint sceneNodeIndex;
    uint visibilityFlags;
};

[[vk::push_constant]] ConstantBuffer<ClusterVisUniforms> uniforms;

[[vk::binding(1)]] StructuredBuffer<ClusterInfo>    clusterInfos;
[[vk::binding(2)]] StructuredBuffer<PackedCluster>  clusters;
[[vk::binding(3)]] ByteAddressBuffer                vertexData;
[[vk::binding(4)]] ByteAddressBuffer                indexData;
[[vk::binding(5)]] StructuredBuffer<InstanceData>   instanceData;

static const uint kMaxVertices  = 32;
static const uint kMaxTriangles = 32;

struct ClusterVertex {
    float4 clipPos : SV_Position;
    float3 wPos    : TEXCOORD0;
};

struct ClusterPrimitive {
    uint clusterID  : TEXCOORD1;
    uint instanceID : TEXCOORD2;
    uint lodLevel   : TEXCOORD3;
};

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void meshMain(
    in uint groupID       : SV_GroupID,
    in uint groupThreadID : SV_GroupThreadID,
    OutputVertices<ClusterVertex, kMaxVertices>       outVerts,
    OutputIndices<uint3, kMaxTriangles>               outTris,
    OutputPrimitives<ClusterPrimitive, kMaxTriangles> outPrims)
{
    ClusterInfo info = clusterInfos[groupID];
    PackedCluster cluster = clusters[info.clusterID];
    InstanceData inst = instanceData[info.instanceID];

    uint vtxCount = clusterVtxCount(cluster);
    uint triCount = clusterTriCount(cluster);

    if (groupThreadID == 0)
        SetMeshOutputCounts(vtxCount, triCount);

    GroupMemoryBarrierWithGroupSync();

    // Vertex processing: 1 thread per vertex (32 threads, max 32 verts)
    if (groupThreadID < vtxCount) {
        uint byteAddr = cluster.vertexByteOffset + groupThreadID * 12;
        float3 oPos = asfloat(vertexData.Load3(byteAddr));

        float4 wPos = mul(inst.worldMatrix, float4(oPos, 1.0));
        float4 cPos = mul(uniforms.viewProj, wPos);

        ClusterVertex v;
        v.clipPos = cPos;
        v.wPos = wPos.xyz;
        outVerts[groupThreadID] = v;
    }

    // Triangle processing: 1 thread per triangle (32 threads, max 32 tris)
    if (groupThreadID < triCount) {
        uint byteAddr = cluster.indexByteOffset + groupThreadID * 3;
        uint byteAddr1 = byteAddr + 1;
        uint byteAddr2 = byteAddr + 2;
        uint packed0 = indexData.Load(byteAddr  & ~3u);
        uint packed1 = indexData.Load(byteAddr1 & ~3u);
        uint packed2 = indexData.Load(byteAddr2 & ~3u);
        uint v0 = (packed0 >> ((byteAddr  & 3u) * 8u)) & 0xFFu;
        uint v1 = (packed1 >> ((byteAddr1 & 3u) * 8u)) & 0xFFu;
        uint v2 = (packed2 >> ((byteAddr2 & 3u) * 8u)) & 0xFFu;

        outTris[groupThreadID] = uint3(v0, v1, v2);

        ClusterPrimitive prim;
        prim.clusterID = info.clusterID;
        prim.instanceID = info.instanceID;
        prim.lodLevel = clusterLodLevel(cluster);
        outPrims[groupThreadID] = prim;
    }
}

[shader("fragment")]
float4 fragmentMain(ClusterVertex vertIn, ClusterPrimitive primIn,
                    uint primitiveID : SV_PrimitiveID) : SV_Target
{
    // Face normal from derivatives
    float3 dPdx = ddx(vertIn.wPos);
    float3 dPdy = ddy(vertIn.wPos);
    float3 faceNormal = normalize(cross(dPdx, dPdy));

    // Select visualization ID
    uint visID;
    if (uniforms.colorMode == 1u)
        visID = primIn.instanceID;
    else if (uniforms.colorMode == 2u)
        visID = primIn.lodLevel;
    else if (uniforms.colorMode == 3u)
        visID = primIn.clusterID * 256u + primitiveID;
    else
        visID = primIn.clusterID;

    float3 color = hashColor(visID);

    // Simple diffuse + ambient
    float NdotL = max(dot(faceNormal, normalize(uniforms.lightDir)), 0.0);
    float3 lit = color * (0.3 + 0.7 * NdotL);

    return float4(lit, 1.0);
}

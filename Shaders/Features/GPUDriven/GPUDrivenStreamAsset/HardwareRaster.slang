#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;

// Hardware mesh output and visibility fragment coverage.
public groupshared StreamRasterCluster g_StreamMeshCluster;
public groupshared uint g_StreamMeshValid;

// Prebinned clusters emit their complete indexed mesh in one workgroup.
// The legacy unbinned queue path retains its two 64-triangle draw slots.
public groupshared float4 g_StreamMeshClip[kMaxMeshletVertices];

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void gpuDrivenStreamAssetMeshMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices GPUDrivenStreamAssetVertexOutput vertices[kMaxMeshletVertices],
    out indices uint3 triangles[kMaxMeshletTriangles],
    out primitives GPUDrivenStreamAssetPrimitiveOutput primitives[kMaxMeshletTriangles],
    uniform GPUDrivenStreamAssetPush push)
{
    uint draw = hybridFlatGroup(groupID);
    bool prebinned = push.hybridClusterBuffer != 0xffffffffu;
    uint record = prebinned ? draw : draw / kMeshletTriangleChunkCount;
    uint chunk = prebinned ? 0u : draw % kMeshletTriangleChunkCount;
    bool overflowHardware = false;
    if (prebinned) {
        StructuredBuffer<uint> bins =
            resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.hybridClusterBuffer));
        uint item = record;
        overflowHardware = bins[14] != 0u;
        if (!hybridStreamHardwareRecord(bins, item, record)) { SetMeshOutputCounts(0, 0); return; }
    }
    if (lane == 0u) {
        bool retryLate;
        g_StreamMeshValid = loadStreamRasterCluster(record, !prebinned || overflowHardware,
            (!prebinned || overflowHardware) && chunk == 0u, push, g_StreamMeshCluster, retryLate) ? 1u : 0u;
        if (overflowHardware) { publishStreamHardwareRetry(record, retryLate, push); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamMeshValid == 0u) { SetMeshOutputCounts(0, 0); return; }
    StreamRasterCluster c = g_StreamMeshCluster;
    uint start = chunk * kMeshletTriangleChunkSize;
    if (start >= c.triangleCount) { SetMeshOutputCounts(0, 0); return; }
    uint count = prebinned ? c.triangleCount : min(c.triangleCount - start, kMeshletTriangleChunkSize);
    SetMeshOutputCounts(c.vertexCount, count);
    bool queueTriangles = !prebinned && push.hybridQueueBuffer != 0xffffffffu && (c.instanceFlags & 12u) == 0u;
    for (uint vertex = lane; vertex < c.vertexCount; vertex += 64u) {
        float4 clip = streamClusterClip(c, vertex, push);
        vertices[vertex].position = clip;
        vertices[vertex].doubleSided = (c.instanceFlags & 2u) != 0u;
        vertices[vertex].texcoord = (c.instanceFlags & 12u) != 0u ? streamClusterUv(c, vertex, push) : float2(0);
        vertices[vertex].instanceIndex = c.group.gpuSceneInstanceIndex;
        if (queueTriangles) { g_StreamMeshClip[vertex] = clip; }
    }
    if (queueTriangles) { GroupMemoryBarrierWithGroupSync(); }
    for (uint i = lane; i < count; i += 64u) {
        uint3 indices = streamClusterTriangle(c, start + i, push);
        uint id = packVisibilityId(c.recordIndex, start + i);
        bool queued = false;
        if (queueTriangles) {
            queued = hybridEnqueueTriangle(g_StreamMeshClip[indices.x], g_StreamMeshClip[indices.y],
                g_StreamMeshClip[indices.z], id, (c.instanceFlags & 2u) != 0u,
                push.hybridQueueBuffer);
        }
        triangles[i] = queued ? uint3(0u) : indices;
        primitives[i].visibilityId = id;
    }
}

[shader("fragment")]
uint gpuDrivenStreamAssetFragmentMain(GPUDrivenStreamAssetVertexOutput input, uint visibilityId : SV_PrimitiveID,
    bool frontFace : SV_IsFrontFace,
    uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    if (!frontFace && input.doubleSided == 0u) { discard; }
    if (!streamAlphaAccepts(input.instanceIndex, input.texcoord, push)) { discard; }
    return visibilityId;
}


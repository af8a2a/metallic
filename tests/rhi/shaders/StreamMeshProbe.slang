// Previous duplicated-vertex schedule retained only for raster equivalence tests.
#include "../../../Shaders/Features/GPUDriven/GPUDrivenStreamAsset.slang"
struct ReferenceStreamVertexOutput
{
    float4 position : SV_Position;
    nointerpolation uint visibilityId : TEXCOORD0;
    nointerpolation uint doubleSided : TEXCOORD1;
};

[shader("mesh")]
[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void referenceStreamMeshMain(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex,
    out vertices ReferenceStreamVertexOutput vertices[192],
    out indices uint3 triangles[kMeshletTriangleChunkSize], uniform GPUDrivenStreamAssetPush push)
{
    uint draw = hybridFlatGroup(groupID);
    uint record = draw / kMeshletTriangleChunkCount;
    uint chunk = draw % kMeshletTriangleChunkCount;
    bool prebinned = push.hybridClusterBuffer != 0xffffffffu;
    if (prebinned) {
        DescriptorHandle<StructuredBuffer<uint>> bins =
            DescriptorHandle<StructuredBuffer<uint>>(uint2(push.bufferShaderIndexBase + push.hybridClusterBuffer, 0));
        uint item = record;
        if (!hybridBinRecord(bins, 0u, item, record)) { SetMeshOutputCounts(0, 0); return; }
    }
    if (lane == 0u) {
        bool retryLate;
        g_StreamMeshValid = loadStreamRasterCluster(record, !prebinned, !prebinned && chunk == 0u,
            push, g_StreamMeshCluster, retryLate) ? 1u : 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_StreamMeshValid == 0u) { SetMeshOutputCounts(0, 0); return; }
    StreamRasterCluster c = g_StreamMeshCluster;
    uint start = chunk * kMeshletTriangleChunkSize;
    if (start >= c.triangleCount) { SetMeshOutputCounts(0, 0); return; }
    uint count = min(c.triangleCount - start, kMeshletTriangleChunkSize);
    SetMeshOutputCounts(count * 3u, count);
    for (uint i = lane; i < count; i += 64u) {
        uint3 indices = streamClusterTriangle(c, start + i, push);
        float4 a = streamClusterClip(c, indices.x, push);
        float4 b = streamClusterClip(c, indices.y, push);
        float4 d = streamClusterClip(c, indices.z, push);
        uint id = packVisibilityId(c.recordIndex, start + i);
        vertices[i * 3u].position = a;
        vertices[i * 3u + 1u].position = b;
        vertices[i * 3u + 2u].position = d;
        vertices[i * 3u].visibilityId = id;
        vertices[i * 3u + 1u].visibilityId = id;
        vertices[i * 3u + 2u].visibilityId = id;
        vertices[i * 3u].doubleSided = (c.instanceFlags & 2u) != 0u;
        vertices[i * 3u + 1u].doubleSided = (c.instanceFlags & 2u) != 0u;
        vertices[i * 3u + 2u].doubleSided = (c.instanceFlags & 2u) != 0u;
        bool queued = !prebinned && hybridEnqueueTriangle(a, b, d, id, (c.instanceFlags & 2u) != 0u, push.hybridQueueBuffer, push.bufferShaderIndexBase);
        triangles[i] = queued ? uint3(0u) : uint3(i * 3u, i * 3u + 1u, i * 3u + 2u);
    }
}

[shader("fragment")]
uint referenceStreamFragmentMain(ReferenceStreamVertexOutput input, bool frontFace : SV_IsFrontFace,
    uniform GPUDrivenStreamAssetPush push) : SV_Target0
{
    if (!frontFace && input.doubleSided == 0u) { discard; }
    return input.visibilityId;
}

import GPUDriven;
using Metallic.GPUDriven;
struct ProbePush {
    uint imageShaderIndexBase; uint bufferShaderIndexBase;
    uint vertices; uint reference; uint prepared; uint width; uint height;
    uint reversed; uint doubleSided; uint bits; uint incrementalDepth; uint triangleCount;
};
groupshared HybridScreenVertex screen[192];
[shader("compute")]
[numthreads(64,1,1)]
void compareMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ProbePush p)
{
    DescriptorHandle<StructuredBuffer<float4>> vertices = DescriptorHandle<StructuredBuffer<float4>>(uint2(p.bufferShaderIndexBase+p.vertices,0));
    uint triangle = group.x * 64u + lane;
    for (uint i=lane; i<192u; i+=64u) {
        uint vertex = group.x * 192u + i;
        if (vertex < p.triangleCount * 3u) { screen[i] = hybridPrepareVertex(vertices[vertex],float2(p.width,p.height),p.bits); }
    }
    GroupMemoryBarrierWithGroupSync();
    if (triangle >= p.triangleCount) { return; }
    float4 a=vertices[triangle*3u], b=vertices[triangle*3u+1u], c=vertices[triangle*3u+2u];
    if (!hybridTriangleFits(a,b,c,float2(p.width,p.height),32.0)) { return; }
    hybridRasterTriangle(a,b,c,triangle+1u,p.doubleSided!=0,p.width,p.height,p.reversed,p.bits,p.reference,p.bufferShaderIndexBase);
    hybridRasterPreparedTriangle(screen[lane*3u],screen[lane*3u+1u],screen[lane*3u+2u],triangle+1u,p.doubleSided!=0,
        p.width,p.height,p.reversed,p.bits,p.prepared,p.bufferShaderIndexBase,p.incrementalDepth!=0);
}

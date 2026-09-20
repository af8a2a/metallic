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

#include "Features/GPUDriven/StreamRasterWorkBins.slang"
groupshared uint originalBucket[128];
groupshared uint reorderedWork[128];
[shader("compute")]
[numthreads(128,1,1)]
void compareWorkBinsMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex, uniform ProbePush p)
{
    DescriptorHandle<StructuredBuffer<float4>> vertices = DescriptorHandle<StructuredBuffer<float4>>(uint2(p.bufferShaderIndexBase+p.vertices,0));
    uint triangle = group.x * 128u + lane;
    uint bucket = 3u;
    if (triangle < p.triangleCount) {
        float4 a=vertices[triangle*3u], b=vertices[triangle*3u+1u], c=vertices[triangle*3u+2u];
        if (hybridTriangleFits(a,b,c,float2(p.width,p.height),32.0)) {
            hybridRasterTriangle(a,b,c,triangle+1u,p.doubleSided!=0,p.width,p.height,p.reversed,p.bits,p.reference,p.bufferShaderIndexBase);
            bucket=streamRasterWorkBucket(a,b,c,p.doubleSided!=0,p.width,p.height,p.bits);
        }
    }
    originalBucket[lane]=bucket;
    uint work=streamRasterReorderWork(bucket,lane);
    reorderedWork[lane]=work;
    GroupMemoryBarrierWithGroupSync();
    // Every input ordinal must appear exactly once, in stable key order,
    // including empty and unused lanes. Pixel equality alone can hide a lost
    // or duplicated triangle behind another primitive.
    bool valid=(work>>7u)==originalBucket[work&127u];
    if (lane>0u) { valid = valid && work>reorderedWork[lane-1u]; }
    if (!valid) {
        DescriptorHandle<RWStructuredBuffer<uint64_t>> target=DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(p.bufferShaderIndexBase+p.prepared,0));
        InterlockedMax(target[0],~uint64_t(0));
    }
    if ((work>>7u)==3u) { return; }
    triangle=group.x*128u+(work&127u);
    if (p.incrementalDepth==3u) {
        hybridRasterPreparedTriangle(hybridPrepareVertex(vertices[triangle*3u],float2(p.width,p.height),p.bits),
            hybridPrepareVertex(vertices[triangle*3u+1u],float2(p.width,p.height),p.bits),
            hybridPrepareVertex(vertices[triangle*3u+2u],float2(p.width,p.height),p.bits),triangle+1u,
            p.doubleSided!=0,p.width,p.height,p.reversed,p.bits,p.prepared,p.bufferShaderIndexBase,false);
    } else {
        hybridRasterTriangle(vertices[triangle*3u],vertices[triangle*3u+1u],vertices[triangle*3u+2u],triangle+1u,
            p.doubleSided!=0,p.width,p.height,p.reversed,p.bits,p.prepared,p.bufferShaderIndexBase);
    }
}

#include "Features/GPUDriven/StreamRasterWorkload.slang"
// Independently count the reference rasterizer's visible pixels for one triangle
// at a time. There is no inter-triangle occlusion to hide attempted writes.
[shader("compute")]
[numthreads(1,1,1)]
void verifyWorkloadMain(uint3 group : SV_GroupID, uniform ProbePush p)
{
    if (group.x != 0) { return; }
    DescriptorHandle<StructuredBuffer<float4>> vertices = DescriptorHandle<StructuredBuffer<float4>>(uint2(p.bufferShaderIndexBase+p.vertices,0));
    DescriptorHandle<RWStructuredBuffer<uint64_t>> pixels = DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(p.bufferShaderIndexBase+p.reference,0));
    DescriptorHandle<RWStructuredBuffer<uint64_t>> result = DescriptorHandle<RWStructuredBuffer<uint64_t>>(uint2(p.bufferShaderIndexBase+p.prepared,0));
    uint64_t errors = 0, total = 0;
    for (uint t = 0; t < p.triangleCount; ++t) {
        float4 a = vertices[t*3], b = vertices[t*3+1], c = vertices[t*3+2];
        if (!hybridTriangleFits(a,b,c,float2(p.width,p.height),32.0)) { continue; }
        hybridRasterTriangle(a,b,c,t+1,p.doubleSided!=0,p.width,p.height,p.reversed,p.bits,p.reference,p.bufferShaderIndexBase);
        DeviceMemoryBarrier();
        uint64_t reference = 0;
        for (uint i = 0; i < p.width*p.height; ++i) {
            reference += uint(pixels[i]) != 0 ? 1u : 0u;
            pixels[i] = 0;
        }
        DeviceMemoryBarrier();
        uint64_t4 count = workloadTriangle(hybridPrepareVertex(a,float2(p.width,p.height),p.bits).position,
            hybridPrepareVertex(b,float2(p.width,p.height),p.bits).position,
            hybridPrepareVertex(c,float2(p.width,p.height),p.bits).position,p.doubleSided!=0,p.width,p.height,p.bits);
        errors += count.y != reference || count.x < reference ? 1u : 0u;
        total += reference;
    }
    result[0] = errors;
    result[1] = total;
}

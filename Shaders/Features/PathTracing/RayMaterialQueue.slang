import ShaderCore;
import ParameterRoot;
using Metallic;

struct RayMaterialQueueParameters
{
    BufferSpan<uint4> hits;
    BufferSpan<uint> materialPrograms;
    RWBufferSpan<uint4> bins;
    RWBufferSpan<uint> indices;
    RWBufferSpan<uint> cursors;
    RWBufferSpan<uint> status;
    uint hitCount, materialCount, programCount, capacity, generationLo, generationHi;
};
void add(RWBufferSpan<uint> buffer, uint index, uint value)
{
    uint ignored;
    resolveUniform(buffer.resource).InterlockedAdd(buffer.byteOffset+index*4u,value,ignored);
}
uint classify(uint4 hit, RayMaterialQueueParameters p)
{
    if (hit.x == 0xffffffffu) { return 0xffffffffu; } // miss, no material evaluation
    if (hit.z != p.generationLo || hit.w != p.generationHi) { return 0xfffffffdu; }
    if (hit.x >= p.materialCount) { return 0xfffffffeu; }
    uint program = p.materialPrograms.load(hit.x);
    return program < p.programCount ? program : 0xfffffffeu;
}
[shader("compute")]
[numthreads(64,1,1)]
void rayMaterialResetMain(uint3 tid : SV_DispatchThreadID)
{
    let p = getParameters<RayMaterialQueueParameters>();
    if (tid.x < p.programCount) { p.bins.store(tid.x,uint4(0)); p.cursors.store(tid.x,0); }
    if (tid.x < p.capacity) { p.indices.store(tid.x,0xffffffffu); }
    if (tid.x < 4) { p.status.store(tid.x,0); }
}
[shader("compute")]
[numthreads(64,1,1)]
void rayMaterialCountMain(uint3 tid : SV_DispatchThreadID)
{
    let p = getParameters<RayMaterialQueueParameters>();
    if (tid.x >= p.hitCount) { return; }
    uint program = classify(p.hits.load(tid.x),p);
    if (program >= p.programCount) { add(p.status,0xffffffffu-program,1); return; }
    uint ignored;
    resolveUniform(p.bins.resource).InterlockedAdd(p.bins.byteOffset+program*16+8,1,ignored);
}
[shader("compute")]
[numthreads(1,1,1)]
void rayMaterialPrefixMain(uint3 tid : SV_DispatchThreadID)
{
    let p = getParameters<RayMaterialQueueParameters>();
    uint offset = 0, overflow = 0;
    for (uint i=0;i<p.programCount;++i) {
        uint attempted = p.bins.load(i).z;
        uint accepted = min(attempted,p.capacity-offset);
        p.bins.store(i,uint4(offset,accepted,attempted,attempted-accepted));
        offset += accepted; overflow += attempted-accepted;
    }
    p.status.store(3,overflow);
}
[shader("compute")]
[numthreads(64,1,1)]
void rayMaterialScatterMain(uint3 tid : SV_DispatchThreadID)
{
    let p = getParameters<RayMaterialQueueParameters>();
    if (tid.x >= p.hitCount) { return; }
    uint program = classify(p.hits.load(tid.x),p);
    if (program >= p.programCount) { return; }
    uint slot;
    resolveUniform(p.cursors.resource).InterlockedAdd(p.cursors.byteOffset+program*4,1,slot);
    uint4 bin = p.bins.load(program);
    if (slot < bin.y) { p.indices.store(bin.x+slot,tid.x); }
}

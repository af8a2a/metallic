import ShaderCore;
import MaterialProgram;
import SlabClosure;
import SurfaceLighting;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;

#ifndef CLOSURE_PROGRAM
#define CLOSURE_PROGRAM 0
#endif
struct Params
{
    BufferSpan<uint> visibility;
    BufferSpan<float4> packets;
    RWBufferSpan<uint2> programTasks;
    RWBufferSpan<uint> control;
    RWBufferSpan<uint> familyIds;
    RWBufferSpan<float4> payload;
    RWBufferSpan<uint2> familyTasks;
    RWBufferSpan<float4> output;
    RWBufferSpan<uint> visits;
    uint width, height, programCount, slot, mode;
};
[[vk::push_constant]] ConstantBuffer<Params> g;
uint tileCount() { return (g.width * g.height + 31u) / 32u; }
void visit(uint pixel, uint stage)
{
    uint unused;
    resolveUniform(g.visits.resource).InterlockedAdd(g.visits.byteOffset + (pixel * 2u + stage) * 4u, 1u, unused);
}
[shader("compute")]
[numthreads(64, 1, 1)]
void resetMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x < 64u) { g.control.store(id.x, 0); }
    if (id.x < g.width * g.height) {
        g.familyIds.store(id.x, 0);
        g.output.store(id.x, float4(0));
        g.visits.store(id.x * 2u, 0); g.visits.store(id.x * 2u + 1u, 0);
    }
}
[shader("compute")]
[numthreads(32, 1, 1)]
void programClassifyMain(uint3 id : SV_DispatchThreadID)
{
    uint program = id.x < g.width * g.height ? g.visibility.load(id.x) : ~0u;
    for (uint bin = 0; bin < g.programCount; ++bin) {
        uint mask = WaveActiveBallot(program == bin).x;
        if (WaveIsFirstLane() && mask != 0) {
            uint offset;
            resolveUniform(g.control.resource).InterlockedAdd(g.control.byteOffset + bin * 4u, 1u, offset);
            g.programTasks.store(bin * tileCount() + offset, uint2(id.x / 32u, mask));
        }
    }
}
[shader("compute")]
[numthreads(1, 1, 1)]
void argumentsMain(uint3 id : SV_DispatchThreadID)
{
    for (uint bin = 0; bin < g.programCount; ++bin) {
        g.control.store(16u + bin * 3u, g.control.load(bin));
        g.control.store(17u + bin * 3u, 1); g.control.store(18u + bin * 3u, 1);
    }
    g.control.store(40u, g.control.load(8u)); g.control.store(41u, 1); g.control.store(42u, 1);
}
[shader("compute")]
[numthreads(32, 1, 1)]
void familyClassifyMain(uint3 id : SV_DispatchThreadID)
{
    uint family = id.x < g.width * g.height ? g.familyIds.load(id.x) : 0;
    uint mask = WaveActiveBallot(family == 1u).x;
    if (WaveIsFirstLane() && mask != 0) {
        uint offset;
        resolveUniform(g.control.resource).InterlockedAdd(g.control.byteOffset + 8u * 4u, 1u, offset);
        g.familyTasks.store(offset, uint2(id.x / 32u, mask));
    }
}
SurfaceMaterialContext contextAt(uint pixel)
{
    SurfaceMaterialContext context = (SurfaceMaterialContext)0;
    float2 uv = (float2(pixel % g.width, pixel / g.width) + 0.5) / float2(g.width, g.height);
    float2 xy = (frac(uv * float2(4, 2)) * 2 - 1) * 0.65;
    context.shadingNormalWS = context.geometryNormalWS = float3(xy, sqrt(1-dot(xy, xy)));
    context.texCoord0 = uv;
    context.frontFace = 1;
    return context;
}
struct Source : ISlabMaterialSource
{
    SlabMaterialInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        uint base = CLOSURE_PROGRAM * 5u;
        SlabMaterialInputs inputs;
        inputs.first.reflectance = g.packets.load(base);
        inputs.first.opticalDepth = g.packets.load(base + 1u);
        inputs.second.reflectance = g.packets.load(base + 2u);
        inputs.second.opticalDepth = g.packets.load(base + 3u);
        inputs.control = g.packets.load(base + 4u);
        // Distinct compile-time Material Programs, with spatially varying values.
        float pattern = 0.65 + 0.25 * sin(context.texCoord0.x * (7.0 + CLOSURE_PROGRAM) + context.texCoord0.y * 3.0);
        inputs.first.reflectance.xyz *= pattern;
        return inputs;
    }
};
float3 lighting(DualSlabClosure closure)
{
    let prepared = closure.prepare(float3(0, 0, 1), TransportMode.Radiance);
    float3 result = float3(0);
    for (uint light = 0; light < 16u; ++light) {
        float phi = float(light) * 0.392699081699;
        float3 wi = float3(0.6 * cos(phi), 0.6 * sin(phi), 0.8);
        result += evaluateSurfaceDirect(prepared, wi, float3(0.125), max(dot(closure.normal.xyz, wi), 0));
    }
    return result;
}
void storeClosure(uint pixel, DualSlabClosure closure)
{
    uint base = pixel * 6u;
    g.payload.store(base, closure.first.reflectance);
    g.payload.store(base + 1u, closure.first.opticalDepth);
    g.payload.store(base + 2u, closure.second.reflectance);
    g.payload.store(base + 3u, closure.second.opticalDepth);
    g.payload.store(base + 4u, closure.normal);
    g.payload.store(base + 5u, closure.control);
}
DualSlabClosure loadClosure(uint pixel)
{
    uint base = pixel * 6u;
    DualSlabClosure closure;
    closure.first.reflectance = g.payload.load(base);
    closure.first.opticalDepth = g.payload.load(base + 1u);
    closure.second.reflectance = g.payload.load(base + 2u);
    closure.second.opticalDepth = g.payload.load(base + 3u);
    closure.normal = g.payload.load(base + 4u);
    closure.control = g.payload.load(base + 5u);
    return closure;
}
bool programPixel(uint group, uint lane, out uint pixel)
{
    uint2 task = g.programTasks.load(CLOSURE_PROGRAM * tileCount() + group);
    pixel = task.x * 32u + lane;
    return (task.y & (1u << lane)) != 0u;
}
[shader("compute")]
[numthreads(32, 1, 1)]
void fusedMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint pixel;
    if (!programPixel(group.x, lane, pixel)) { return; }
    DualSlabMaterialProgram<Source> program;
    let material = program.evaluate(contextAt(pixel), (MaterialInstanceRef)0);
    visit(pixel, 0); visit(pixel, 1);
    g.output.store(pixel, float4(lighting(material.closure), float(CLOSURE_PROGRAM + 1u)));
}
[shader("compute")]
[numthreads(32, 1, 1)]
void evaluateMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint pixel;
    if (!programPixel(group.x, lane, pixel)) { return; }
    DualSlabMaterialProgram<Source> program;
    let material = program.evaluate(contextAt(pixel), (MaterialInstanceRef)0);
    visit(pixel, 0);
    storeClosure(pixel, material.closure);
    g.familyIds.store(pixel, 1u);
    g.output.store(pixel, float4(0, 0, 0, float(CLOSURE_PROGRAM + 1u)));
}
[shader("compute")]
[numthreads(32, 1, 1)]
void lightFamilyMain(uint3 group : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint2 task = g.familyTasks.load(group.x);
    if ((task.y & (1u << lane)) == 0u) { return; }
    uint pixel = task.x * 32u + lane;
    let closure = loadClosure(pixel);
    visit(pixel, 1);
    g.output.store(pixel, float4(lighting(closure), g.output.load(pixel).w));
}
// The same number of one-workgroup dispatches, with/without program changes.
// Each writes a distinct location, so no between-dispatch memory dependency is needed.
[shader("compute")]
[numthreads(1, 1, 1)]
void switchMain(uint3 id : SV_DispatchThreadID)
{
    g.output.store(g.slot, float4(float(CLOSURE_PROGRAM), float(g.slot), 0, 1));
}

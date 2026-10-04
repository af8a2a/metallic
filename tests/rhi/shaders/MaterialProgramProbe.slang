import ShaderCore;
import MaterialProgram;
using Metallic;
using Metallic.Material;

struct ProbeResult
{
    BSDFEval evaluation;
    BSDFSample sample;
    MaterialInstanceRef instance;
};
struct ProbeParams
{
    BufferSpan<float> parameters;
    BufferSpan<float> resources;
    BufferSpan<MaterialInstanceRef> instances;
    BufferSpan<SurfaceMaterialContext> contexts;
    RWBufferSpan<SurfaceMaterialContext> contextOutput;
    RWBufferSpan<ProbeResult> resultOutput;
};
[[vk::push_constant]] ConstantBuffer<ProbeParams> gParams;

// Test-only arithmetic program, not a physical BSDF or the Phase 3 interface.
interface IProbeProgram
{
    BSDFEval evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance);
}
struct ProbeProgram : IProbeProgram
{
    BSDFEval evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        BSDFEval result;
        result.f = float3(gParams.parameters.load(instance.parameterOffset / 4u),
            gParams.resources.load(instance.resourceOffset / 4u), context.positionWS.x + context.texCoord1.x);
        result.pdf = 0.25;
        result.flags = kBSDFReflection | kBSDFDiffuse;
        return result;
    }
}
BSDFEval invokeProgram<T : IProbeProgram>(T program, SurfaceMaterialContext context, MaterialInstanceRef instance)
{
    return program.evaluate(context, instance);
}

[shader("compute")]
[numthreads(8, 1, 1)]
void materialProgramProbeMain(uint lane : SV_GroupIndex)
{
    SurfaceMaterialContext context = gParams.contexts.load(lane & 1u);
    MaterialInstanceRef instance = gParams.instances.load(lane);
    ProbeProgram program;
    ProbeResult result;
    result.evaluation = invokeProgram(program, context, instance);
    result.sample.wiWS = context.shadingNormalWS;
    result.sample.f = result.evaluation.f;
    result.sample.pdf = result.evaluation.pdf;
    result.sample.eta = 1.0;
    result.sample.flags = result.evaluation.flags;
    result.instance = instance;
    gParams.contextOutput.store(lane, context);
    gParams.resultOutput.store(lane, result);
}

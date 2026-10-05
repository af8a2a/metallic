import ShaderCore;
import ParameterRoot;
import RayMaterialExecution;
import SlabClosure;
import RTXCRFiber;
import RTXCRHair;
using Metallic;
using Metallic.Material;

struct Params
{
    BufferSpan<uint4> hits;
    BufferSpan<uint> programs;
    BufferSpan<uint4> bins;
    BufferSpan<uint> indices;
    RWBufferSpan<float4> output;
    RWBufferSpan<uint> visits;
    uint hitCount, programCount, queued, generationLo, generationHi, bounce;
};
float random(inout uint state)
{
    state ^= state>>16; state *= 0x7feb352du; state ^= state>>15; state *= 0x846ca68bu; state ^= state>>16;
    return (float(state>>8)+0.5)/16777216.0;
}
struct SurfaceSource : ISlabMaterialSource
{
    uint program;
    SlabMaterialInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        SlabMaterialInputs result = (SlabMaterialInputs)0;
        // Two programs share DualSlab family but have distinct value expressions.
        float value = 0.25+0.1*sin(context.texCoord0.x*(program+1)+context.positionWS.x);
        result.first.reflectance = float4(value,0.2,0.1,0);
        result.first.opticalDepth = float4(0.15+abs(context.footprint.isotropicLod0)*0.02,0.3,0.5,0);
        result.second.reflectance = float4(0.6,0.3,0.2,0)*(1+float(instance.parameterOffset)*0.0001);
        result.control = float4(2,0,0,0);
        return result;
    }
};
struct FiberSource : IRTXCRHairParameterProvider
{
    [mutating] Metallic.RTXCR.HairMaterialData load(MaterialInstanceRef instance)
    {
        Metallic.RTXCR.HairMaterialData material = (Metallic.RTXCR.HairMaterialData)0;
        material.baseColor = float3(0.4,0.2,0.1); material.absorptionModel = Metallic.RTXCR.kHairAbsorptionModelNormalized;
        material.melanin = 0.25+instance.parameterOffset*0.0001; material.melaninRedness = 0.2;
        material.longitudinalRoughness = 0.35; material.azimuthalRoughness = 0.4;
        material.ior = 1.55; material.eta = 1/1.55; material.cuticleAngleInDegrees = 3; material.fresnelApproximation = 1;
        return material;
    }
};
[shader("compute")]
[numthreads(64,1,1)]
void resetProbeMain(uint3 id : SV_DispatchThreadID)
{
    let p = getParameters<Params>();
    if (id.x >= p.hitCount) { return; }
    p.output.store(id.x*3,0); p.output.store(id.x*3+1,0); p.output.store(id.x*3+2,0); p.visits.store(id.x,0);
}
[shader("compute")]
[numthreads(64,1,1)]
void executeProbeMain(uint3 tid : SV_DispatchThreadID)
{
    let p = getParameters<Params>();
    if (tid.x >= p.hitCount) { return; }
    uint index = tid.x;
    if (p.queued != 0) {
        uint total = 0;
        for (uint bin=0;bin<p.programCount;++bin) { total += p.bins.load(bin).y; }
        if (index >= total) { return; }
        index = p.indices.load(index);
    }
    uint4 hit = p.hits.load(index);
    if (hit.x >= p.programs.length() || hit.z != p.generationLo || hit.w != p.generationHi) { return; }
    uint programId = p.programs.load(hit.x);
    if (programId >= p.programCount) { return; }
    RayMaterialWork work = (RayMaterialWork)0;
    work.pathIndex = hit.y; work.seed = hit.y; work.bounce = p.bounce;
    work.instance.parameterOffset = hit.x*16; work.instance.resourceOffset = hit.x*32;
    float v = random(work.seed);
    work.surface.positionWS = float3(v,p.bounce,0);
    work.surface.texCoord0 = float2(v,1-v);
    work.surface.shadingNormalWS = work.surface.geometryNormalWS = float3(0,0,1);
    work.surface.tangentWS = float3(1,0,0); work.surface.bitangentWS = float3(0,1,0);
    work.surface.frontFace = 1; work.surface.texCoordMask = 1;
    work.surface.footprint.flags = kTextureFootprintIsotropic0;
    work.surface.footprint.isotropicLod0 = -7+float(p.bounce);
    work.woWS = normalize(float3(v-0.5,0.2+0.1*p.bounce,1));
    float4 u = float4(random(work.seed),random(work.seed),random(work.seed),random(work.seed));
    float3 wi = normalize(float3(-0.3,0.4,1));
    float3 value, direction, weight; float pdf;
    if (programId == 2) {
        RTXCRChiangMaterialProgram<FiberSource> program;
        let material = evaluateRayFiber(program,work);
        let prepared = prepareRayFiber(material.closure,work);
        let scatter = prepared.sampleWeighted(u);
        value = prepared.evalProjected(wi); direction = scatter.wiWS; weight = scatter.weight; pdf = scatter.pdf;
    } else {
        DualSlabMaterialProgram<SurfaceSource> program; program.source.program = programId;
        let material = evaluateRaySurface(program,work);
        let prepared = prepareRaySurface(material.closure,work,(SurfaceSamplingContext)0);
        let scatter = prepared.sampleWeighted(u.xyz);
        value = prepared.evalProjected(wi); direction = scatter.wiWS; weight = scatter.weight; pdf = scatter.pdf;
    }
    p.output.store(index*3,float4(value,pdf));
    p.output.store(index*3+1,float4(direction,asfloat(work.seed)));
    p.output.store(index*3+2,float4(weight,float(work.bounce)));
    uint ignored; resolveUniform(p.visits.resource).InterlockedAdd(p.visits.byteOffset+index*4,1,ignored);
}

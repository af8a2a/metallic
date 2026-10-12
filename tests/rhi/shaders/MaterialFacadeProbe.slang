import ShaderCore;
import MaterialProgram;
import FiberMaterial;
using Metallic;
using Metallic.Material;

struct ProbeParams
{
    RWBufferSpan<float4> output;
};
[[vk::push_constant]] ConstantBuffer<ProbeParams> gParams;

// Independent records deliberately have different shapes. The facade must
// preserve their concrete closure types and each program's mutable state.
struct ColorPrepared : IPreparedSurfaceClosure
{
    float3 color;
    float3 shadingNormal() { return float3(0, 0, 1); }
    BSDFEval eval(float3 wiWS)
    {
        BSDFEval result = BSDFEval();
        result.f = color;
        result.pdf = 0.25;
        return result;
    }
    BSDFSample sample(float3 random) { return BSDFSample(); }
}
struct ColorClosure : ISurfaceClosure
{
    typealias Prepared = ColorPrepared;
    float3 color;
    Prepared prepare(float3 woWS, TransportMode mode)
    {
        Prepared result;
        result.color = color;
        return result;
    }
}
struct ColorProgram : ISurfaceMaterialProgram
{
    typealias Closure = ColorClosure;
    uint calls;
    [mutating]
    SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        ++calls;
        SurfaceMaterialResult<Closure> result = SurfaceMaterialResult<Closure>();
        result.closure.color = float3(float(instance.parameterOffset) * 0.25,
            float(instance.resourceOffset) * 0.5, context.texCoord0.x);
        result.emission = float3(float(calls));
        return result;
    }
}

struct AmplitudePrepared : IPreparedSurfaceClosure
{
    float4 amplitudes;
    float3 shadingNormal() { return float3(0, 0, 1); }
    BSDFEval eval(float3 wiWS)
    {
        BSDFEval result = BSDFEval();
        result.f = amplitudes.yxz * 2.0;
        result.pdf = amplitudes.w;
        return result;
    }
    BSDFSample sample(float3 random) { return BSDFSample(); }
}
struct AmplitudeClosure : ISurfaceClosure
{
    typealias Prepared = AmplitudePrepared;
    float4 amplitudes;
    Prepared prepare(float3 woWS, TransportMode mode)
    {
        Prepared result;
        result.amplitudes = amplitudes;
        return result;
    }
}
struct AmplitudeProgram : ISurfaceMaterialProgram
{
    typealias Closure = AmplitudeClosure;
    uint calls;
    [mutating]
    SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        ++calls;
        SurfaceMaterialResult<Closure> result = SurfaceMaterialResult<Closure>();
        result.closure.amplitudes = float4(context.positionWS.x,
            float(instance.resourceOffset) * 0.25, float(instance.parameterOffset) * 0.5, 0.5);
        result.emission = float3(float(calls) * 2.0);
        return result;
    }
}

struct StrandPrepared : IPreparedFiberClosure
{
    float3 scattering;
    float3 evalProjected(float3 wiWS) { return scattering * (wiWS.x + 1.0); }
    FiberSample sampleWeighted(float4 random) { return FiberSample(); }
}
struct StrandClosure : IFiberClosure
{
    typealias Prepared = StrandPrepared;
    float3 scattering;
    Prepared prepare(float3 woWS, TransportMode mode)
    {
        Prepared result;
        result.scattering = scattering;
        return result;
    }
}
struct StrandProgram : IFiberMaterialProgram
{
    typealias Closure = StrandClosure;
    uint calls;
    [mutating]
    FiberMaterialResult<Closure> evaluate(FiberMaterialContext context, MaterialInstanceRef instance)
    {
        ++calls;
        FiberMaterialResult<Closure> result = FiberMaterialResult<Closure>();
        result.closure.scattering = float3(context.radius, context.crossSectionOffset,
            float(instance.parameterOffset + instance.resourceOffset));
        result.emission = float3(float(calls) * 3.0);
        return result;
    }
}

SurfaceMaterialResult<TProgram.Closure> invokeSurface<TProgram : ISurfaceMaterialProgram>(
    inout Material<TProgram> material, SurfaceMaterialContext context, MaterialInstanceRef instance)
{
    return material.evaluate(context, instance);
}
FiberMaterialResult<TProgram.Closure> invokeFiber<TProgram : IFiberMaterialProgram>(
    inout Material<TProgram> material, FiberMaterialContext context, MaterialInstanceRef instance)
{
    return material.evaluate(context, instance);
}

[shader("compute")]
[numthreads(8, 1, 1)]
void materialFacadeProbeMain(uint lane : SV_GroupIndex)
{
    SurfaceMaterialContext surface = SurfaceMaterialContext();
    surface.positionWS.x = float(lane) * 0.25;
    surface.texCoord0.x = float(lane) * 0.125;
    FiberMaterialContext fiber = FiberMaterialContext();
    fiber.radius = float(lane + 1u) * 0.125;
    fiber.crossSectionOffset = float(lane) * 0.25 - 1.0;
    MaterialInstanceRef instance;
    instance.parameterOffset = ((lane * 3u) & 7u) * 4u;
    instance.resourceOffset = (7u - lane) * 4u;

    Material<ColorProgram> color = Material<ColorProgram>();
    Material<AmplitudeProgram> amplitude = Material<AmplitudeProgram>();
    Material<StrandProgram> strand = Material<StrandProgram>();
    let firstColor = invokeSurface(color, surface, instance);
    let firstAmplitude = invokeSurface(amplitude, surface, instance);
    let firstStrand = invokeFiber(strand, fiber, instance);
    let secondColor = invokeSurface(color, surface, instance);
    let secondAmplitude = invokeSurface(amplitude, surface, instance);
    let secondStrand = invokeFiber(strand, fiber, instance);
    let colorPrepared = secondColor.closure.prepare(float3(0, 0, 1), TransportMode.Radiance);
    let amplitudePrepared = secondAmplitude.closure.prepare(float3(0, 0, 1), TransportMode.Radiance);
    let strandPrepared = secondStrand.closure.prepare(float3(0, 0, 1), TransportMode.Radiance);
    BSDFEval colorEvaluation = colorPrepared.eval(float3(1, 0, 0));
    BSDFEval amplitudeEvaluation = amplitudePrepared.eval(float3(1, 0, 0));
    uint base = lane * 6u;
    gParams.output.store(base, float4(colorEvaluation.f, colorEvaluation.pdf));
    gParams.output.store(base + 1u, float4(amplitudeEvaluation.f, amplitudeEvaluation.pdf));
    // Tangent direction has zero Surface cosine, but Fiber projected measure
    // remains nonzero. No surface normal/cosine conversion may enter this path.
    gParams.output.store(base + 2u, float4(strandPrepared.evalProjected(float3(1, 0, 0)), secondStrand.emission.x));
    gParams.output.store(base + 3u, float4(firstColor.emission.x, firstAmplitude.emission.x,
        firstStrand.emission.x, 0));
    gParams.output.store(base + 4u, float4(secondColor.emission.x, secondAmplitude.emission.x,
        secondStrand.emission.x, 0));
    gParams.output.store(base + 5u, float4(float(color.program.calls), float(amplitude.program.calls),
        float(strand.program.calls), 0));
#if MATERIAL_FACADE_INVALID_CONTEXT == 1
    color.evaluate(fiber, instance);
#elif MATERIAL_FACADE_INVALID_CONTEXT == 2
    strand.evaluate(surface, instance);
#endif
}

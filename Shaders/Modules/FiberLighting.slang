#language slang 2026
module FiberLighting;
import MaterialProgram;
import FiberMaterial;

namespace Metallic.Lighting
{
using Metallic.Material;

public struct FiberLightSample
{
    public float3 direction;
    public float3 illuminance;
    public float inverseSourcePdf;
};
public interface IFiberLightProvider
{
    public uint count();
    [mutating] public FiberLightSample sample(uint index);
    public float3 visibility(FiberLightSample light);
};

public float3 evaluateFiberDirect<TPrepared : IPreparedFiberClosure>(
    TPrepared prepared, float3 wiWS, float3 incidentRadiance)
{
    return max(prepared.evalProjected(wiWS), float3(0)) * incidentRadiance;
}

public float3 integrateFiberDirect<TPrepared : IPreparedFiberClosure, TLights : IFiberLightProvider>(
    TPrepared prepared, inout TLights lights)
{
    float3 result = 0;
    for (uint index = 0; index < lights.count(); ++index) {
        FiberLightSample light = lights.sample(index);
        if (light.inverseSourcePdf <= 0) { continue; }
        result += evaluateFiberDirect(prepared, light.direction, light.illuminance) *
            lights.visibility(light) * light.inverseSourcePdf;
    }
    return result;
}

public struct FiberShadingResult<TClosure : IFiberClosure>
{
    public FiberMaterialResult<TClosure> material;
    public TClosure.Prepared prepared;
    public float3 radiance;
};

public FiberShadingResult<TProgram.Closure> shadeFiber<TProgram : IFiberMaterialProgram, TLights : IFiberLightProvider>(
    inout TProgram program, FiberMaterialContext context, MaterialInstanceRef instance,
    float3 woWS, TransportMode mode, inout TLights lights)
{
    FiberShadingResult<TProgram.Closure> result;
    result.material = program.evaluate(context, instance);
    result.prepared = result.material.closure.prepare(woWS, mode);
    result.radiance = result.material.emission + integrateFiberDirect(result.prepared, lights);
    return result;
}
} // namespace Metallic.Lighting

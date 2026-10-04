#language slang 2026
module SurfaceLighting;

import MaterialProgram;

namespace Metallic.Lighting
{
using Metallic.Material;

// Li already includes light selection/visibility as appropriate to the caller.
// Geometry/cosine is explicit: lighting cannot access material inputs or rerun
// evaluate()/prepare(). These functions specialize on a concrete Prepared type.
public float3 evaluateSurfaceDirect<TPrepared : IPreparedSurfaceClosure>(
    TPrepared prepared, float3 wiWS, float3 incidentRadiance, float absoluteCosine)
{
    BSDFEval value = prepared.eval(wiWS);
    return value.f * incidentRadiance * max(absoluteCosine, 0.0);
}

// The integrator owns throughput, ray offsets, visibility, MIS and termination.
public BSDFSample sampleSurfaceContinuation<TPrepared : IPreparedSurfaceClosure>(TPrepared prepared, float3 random)
{
    return prepared.sample(random);
}

public float3 evaluateSurfaceProjected<TPrepared : IPreparedSurfaceClosure>(TPrepared prepared, float3 wiWS)
{
    return prepared.evalProjected(wiWS);
}

public BSDFWeightSample sampleSurfaceWeight<TPrepared : IPreparedSurfaceClosure>(TPrepared prepared, float3 random)
{
    return prepared.sampleWeighted(random);
}

public struct SurfaceLightSample
{
    public float3 direction;
    public float3 illuminance;
    public float distance;
    public float inverseSourcePdf;
    public uint source;
};

// Providers select/sample lights and trace visibility. They cannot access the
// MaterialProgram or Closure: material textures are already resolved.
public interface ISurfaceLightProvider
{
    public uint count();
    [mutating] public SurfaceLightSample sample(uint index);
    public float3 visibility(SurfaceLightSample light);
};

public float3 integrateDirectLighting<TPrepared : IPreparedSurfaceClosure, TLights : ISurfaceLightProvider>(
    TPrepared prepared, inout TLights lights, float3 initialRadiance, float3 throughput = float3(1))
{
    float3 radiance = initialRadiance;
    uint count = lights.count();
    for (uint index = 0u; index < count; ++index) {
        SurfaceLightSample light = lights.sample(index);
        if (light.inverseSourcePdf <= 0.0) { continue; }
        float3 weight = max(prepared.evalProjected(light.direction), float3(0));
        float3 contribution = weight * light.illuminance;
        if (max(contribution.x, max(contribution.y, contribution.z)) <= 0.0) { continue; }
        radiance += throughput * weight * light.illuminance * lights.visibility(light) * light.inverseSourcePdf;
    }
    return radiance;
}

public struct SurfaceShadingResult<TClosure : ISurfaceClosure>
{
    public SurfaceMaterialResult<TClosure> material;
    public TClosure.Prepared prepared;
    public float3 radiance;
};

public SurfaceShadingResult<TClosure> shadeEvaluatedSurface<TClosure : ISurfaceClosure, TLights : ISurfaceLightProvider>(
    SurfaceMaterialResult<TClosure> material, float3 woWS, TransportMode mode,
    SurfaceSamplingContext sampling, inout TLights lights)
{
    SurfaceShadingResult<TClosure> result;
    result.material = material;
    result.prepared = material.closure.prepare(woWS, mode, sampling);
    float3 emission = max(result.prepared.surfaceEmission(material.emission), float3(0));
    result.radiance = integrateDirectLighting(result.prepared, lights, emission);
    return result;
}

// One hit, one evaluate, one prepare; concrete Program/provider types are
// selected by the CPU compilation request, never by a material instance ID.
public SurfaceShadingResult<TMaterial.Closure> shadeSurface<
    TMaterial : ISurfaceMaterialProgram, TLights : ISurfaceLightProvider>(
    inout TMaterial material, SurfaceMaterialContext context, MaterialInstanceRef instance,
    float3 woWS, TransportMode mode, SurfaceSamplingContext sampling, inout TLights lights)
{
    return shadeEvaluatedSurface(material.evaluate(context, instance), woWS, mode, sampling, lights);
}
} // namespace Metallic.Lighting

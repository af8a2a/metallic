#language slang 2026
module DebugMirror;

import MaterialProgram;
import DebugLambert;

namespace Metallic.Material
{
// Ideal, single-sided tinted reflector. Directional eval is zero because its
// distribution is discrete; sample returns unit discrete probability.
public struct DebugMirrorPreparedClosure : IPreparedSurfaceClosure
{
    public float3 normalWS;
    public float3 reflectedDirection;
    public float3 reflectance;
    public bool valid;
    public float3 shadingNormal() { return normalWS; }
    public BSDFEval eval(float3 wiWS) { return (BSDFEval)0; }
    public BSDFSample sample(float3 random)
    {
        BSDFSample result = (BSDFSample)0;
        result.eta = 1.0;
        if (!valid) { return result; }
        result.wiWS = reflectedDirection;
        result.f = reflectance / abs(dot(normalWS, reflectedDirection));
        result.pdf = 1.0;
        result.flags = kBSDFReflection | kBSDFDelta;
        return result;
    }
};

public struct DebugMirrorClosure : ISurfaceClosure
{
    public typealias Prepared = DebugMirrorPreparedClosure;
    public float3 normalWS;
    public float3 reflectance;
    public Prepared prepare(float3 woWS, TransportMode mode)
    {
        Prepared result;
        result.normalWS = normalWS;
        result.reflectedDirection = reflect(-woWS, normalWS);
        result.reflectance = reflectance;
        result.valid = dot(woWS, normalWS) > 0.0;
        return result;
    }
};

public struct DebugMirrorMaterialProgram<TSource : IDebugLambertSource> : ISurfaceMaterialProgram
{
    public typealias Closure = DebugMirrorClosure;
    public TSource source;
    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        // Reuse only parameter/color/normal resolution; scattering is mirror.
        DebugLambertMaterialProgram<TSource> inputs;
        inputs.source = source;
        let resolved = inputs.evaluate(context, instance);
        SurfaceMaterialResult<Closure> result;
        result.closure.normalWS = resolved.closure.normalWS;
        result.closure.reflectance = resolved.closure.albedo;
        result.emission = resolved.emission;
        result.flags = resolved.flags;
        return result;
    }
};
} // namespace Metallic.Material

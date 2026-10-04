#language slang 2026
module SlabClosure;

import MaterialProgram;

namespace Metallic.Material
{
internal static const float kSlabPi = 3.14159265358979323846;

public struct SlabRecord
{
    public float4 reflectance;
    public float4 opticalDepth;
};
// Matches MaterialClosurePacket. Only MaterialProgram's source loads resources.
public struct SlabMaterialInputs
{
    public SlabRecord first;
    public SlabRecord second;
    public float4 control;
};
public interface ISlabMaterialSource
{
    public SlabMaterialInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance);
};

internal float4 resolveSlabNormal(SurfaceMaterialContext context)
{
    float3 normal = context.shadingNormalWS;
    if (dot(normal, normal) < 1e-20) { normal = context.geometryNormalWS; }
    if (dot(normal, normal) < 1e-20) { normal = float3(0, 0, 1); }
    return float4(normalize(normal), 0);
}

public struct SlabPreparedClosure : IPreparedSurfaceClosure
{
    public float3 normalWS;
    public float3 top;
    public float3 bottom;
    public float3 opticalDepth;
    public float cosineOut;
    public uint operation;
    public float weight;

    public float3 shadingNormal() { return normalWS; }
    public BSDFEval eval(float3 wiWS)
    {
        BSDFEval result = (BSDFEval)0;
        float cosineIn = dot(normalWS, wiWS);
        if (cosineOut <= 0.0 || cosineIn <= 0.0) { return result; }
        float3 reflectance = top;
        if (operation == 1u) {
            // Mix scattering functions, not authored material parameters.
            reflectance = (1.0 - weight) * top + weight * bottom;
        } else if (operation == 2u) {
            // Single traversal in/out through a diffuse top interface. The
            // missing multiple reflections/refraction are intentionally omitted.
            // This symmetric attenuation preserves reciprocity and bounds energy:
            // r + (1-r)^2*b*exp(-tau*(1/mu_i+1/mu_o)) <= 1.
            float3 transmission = 1.0 - top;
            float pathLength = rcp(max(cosineIn, 1e-6)) + rcp(max(cosineOut, 1e-6));
            reflectance = top + transmission * transmission * bottom * exp(-opticalDepth * pathLength);
        }
        result.f = reflectance / kSlabPi;
        // Both components share a basis and cosine proposal, so their complete
        // mixture PDF is exactly cos/pi (also a valid proposal for Layer).
        result.pdf = cosineIn / kSlabPi;
        result.flags = kBSDFReflection | kBSDFDiffuse;
        return result;
    }
    public BSDFSample sample(float3 random)
    {
        BSDFSample result = (BSDFSample)0;
        if (cosineOut <= 0.0) { return result; }
        float r2 = clamp(random.x, 0.0, 0.99999994);
        float phi = 2.0 * kSlabPi * random.y;
        float3 axis = abs(normalWS.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
        float3 tangent = normalize(cross(axis, normalWS));
        result.wiWS = tangent * (sqrt(r2) * cos(phi)) + cross(normalWS, tangent) * (sqrt(r2) * sin(phi)) +
            normalWS * sqrt(1.0 - r2);
        let value = eval(result.wiWS);
        result.f = value.f;
        result.pdf = value.pdf;
        result.flags = value.flags;
        result.eta = 1.0;
        return result;
    }
};

public struct SingleSlabClosure : ISurfaceClosure
{
    public typealias Prepared = SlabPreparedClosure;
    public SlabRecord slab;
    public float4 normal;
    public Prepared prepare(float3 woWS, TransportMode mode)
    {
        Prepared result = (Prepared)0;
        result.normalWS = normal.xyz;
        result.top = slab.reflectance.xyz;
        result.cosineOut = dot(normal.xyz, woWS);
        return result;
    }
};
public struct DualSlabClosure : ISurfaceClosure
{
    public typealias Prepared = SlabPreparedClosure;
    public SlabRecord first;
    public SlabRecord second;
    public float4 normal;
    public float4 control;
    public Prepared prepare(float3 woWS, TransportMode mode)
    {
        Prepared result = (Prepared)0;
        result.normalWS = normal.xyz;
        result.top = first.reflectance.xyz;
        result.bottom = second.reflectance.xyz;
        result.opticalDepth = first.opticalDepth.xyz;
        result.cosineOut = dot(normal.xyz, woWS);
        result.operation = uint(control.x);
        result.weight = control.y;
        return result;
    }
};

public struct SingleSlabMaterialProgram<TSource : ISlabMaterialSource> : ISurfaceMaterialProgram
{
    public typealias Closure = SingleSlabClosure;
    public TSource source;
    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        let inputs = source.evaluateInputs(context, instance);
        SurfaceMaterialResult<Closure> result = (SurfaceMaterialResult<Closure>)0;
        result.closure.slab = inputs.first;
        result.closure.normal = resolveSlabNormal(context);
        return result;
    }
};
public struct DualSlabMaterialProgram<TSource : ISlabMaterialSource> : ISurfaceMaterialProgram
{
    public typealias Closure = DualSlabClosure;
    public TSource source;
    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        let inputs = source.evaluateInputs(context, instance);
        SurfaceMaterialResult<Closure> result = (SurfaceMaterialResult<Closure>)0;
        result.closure.first = inputs.first;
        result.closure.second = inputs.second;
        result.closure.control = inputs.control;
        result.closure.normal = resolveSlabNormal(context);
        return result;
    }
};
} // namespace Metallic.Material

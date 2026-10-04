#language slang 2026
module DebugLambert;

import MaterialProgram;

namespace Metallic.Material
{
internal static const float kLambertPi = 3.14159265358979323846;

public struct DebugLambertInputs
{
    public float3 albedo;
    public float3 emission;
};

// The caller supplies concrete, read-only parameter/texture access. Only the
// Material stage owns this provider; neither closure stores it or any handles.
public interface IDebugLambertSource
{
    public DebugLambertInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance);
};

public struct DebugLambertPreparedClosure : IPreparedSurfaceClosure
{
    public float3 normalWS;
    public float3 reflectanceOverPi;
    public bool valid;
    public float3 shadingNormal() { return normalWS; }

    public BSDFEval eval(float3 wiWS)
    {
        BSDFEval result = (BSDFEval)0;
        float cosine = dot(normalWS, wiWS);
        if (valid && cosine > 0.0) {
            result.f = reflectanceOverPi;
            result.pdf = cosine / kLambertPi;
            result.flags = kBSDFReflection | kBSDFDiffuse;
        }
        return result;
    }

    public BSDFSample sample(float3 random)
    {
        BSDFSample result = (BSDFSample)0;
        if (!valid) { return result; }
        // Third variate is reserved for future lobe selection. Clamp the upper
        // endpoint to avoid a zero-density tangent sample after float rounding.
        float radiusSquared = clamp(random.x, 0.0, 0.99999994);
        float radius = sqrt(radiusSquared);
        float phi = 2.0 * kLambertPi * random.y;
        float3 axis = abs(normalWS.z) < 0.999 ? float3(0, 0, 1) : float3(0, 1, 0);
        float3 tangent = normalize(cross(axis, normalWS));
        float3 bitangent = cross(normalWS, tangent);
        result.wiWS = tangent * (radius * cos(phi)) + bitangent * (radius * sin(phi)) +
            normalWS * sqrt(1.0 - radiusSquared);
        BSDFEval evaluation = eval(result.wiWS);
        result.f = evaluation.f;
        result.pdf = evaluation.pdf;
        result.flags = evaluation.flags;
        result.eta = 1.0;
        return result;
    }
};

public struct DebugLambertClosure : ISurfaceClosure
{
    public typealias Prepared = DebugLambertPreparedClosure;
    public float3 albedo;
    public float3 normalWS;

    public Prepared prepare(float3 woWS, TransportMode transportMode)
    {
        Prepared prepared;
        prepared.normalWS = normalWS;
        prepared.reflectanceOverPi = albedo / kLambertPi;
        // This diagnostic Lambert is one-sided. Preserve the resolved normal;
        // callers needing two-sided material behavior must resolve it explicitly.
        prepared.valid = dot(normalWS, woWS) > 0.0;
        // Reflection at eta=1 has identical radiance/importance transport.
        return prepared;
    }
};

public struct DebugLambertMaterialProgram<TSource : IDebugLambertSource> : ISurfaceMaterialProgram
{
    public typealias Closure = DebugLambertClosure;
    public TSource source;

    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        DebugLambertInputs inputs = source.evaluateInputs(context, instance);
        SurfaceMaterialResult<Closure> result;
        result.closure.albedo = saturate(inputs.albedo);
        float squaredLength = dot(context.shadingNormalWS, context.shadingNormalWS);
        float3 normal = squaredLength > 1e-20 ? context.shadingNormalWS : context.geometryNormalWS;
        squaredLength = dot(normal, normal);
        result.closure.normalWS = squaredLength > 1e-20 ? normal * rsqrt(squaredLength) : float3(0, 0, 1);
        result.emission = max(inputs.emission, float3(0));
        result.flags = 0u;
        return result;
    }
};
} // namespace Metallic.Material

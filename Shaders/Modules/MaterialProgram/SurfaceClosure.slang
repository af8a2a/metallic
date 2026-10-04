#language slang 2026
implementing MaterialProgram;

namespace Metallic.Material
{
public enum TransportMode : uint
{
    Radiance = 0,
    Importance = 1,
};

public struct BSDFWeightSample
{
    public float3 wiWS;
    public float3 weight; // f * abs(cos(theta)) / pdf, including delta events
    public float pdf;
    public float eta;
    public uint flags;
};

// Integrator state affects sampling preparation, never material evaluation.
public struct SurfaceSamplingContext
{
    public float3 throughput;
    public float3 wavelengths;
    public float exteriorIor;
};

// Per-wo temporary state. Implementations must not load material parameters,
// sample material textures, or retain a MaterialProgram/resource provider here.
public interface IPreparedSurfaceClosure
{
    public BSDFEval eval(float3 wiWS);
    public BSDFSample sample(float3 random);
    public float3 shadingNormal();
    public float3 surfaceEmission(float3 resolvedEmission) { return resolvedEmission; }
    public float3 evalProjected(float3 wiWS)
    {
        return eval(wiWS).f * abs(dot(shadingNormal(), wiWS));
    }
    public float pdf(float3 wiWS) { return eval(wiWS).pdf; }
    public BSDFWeightSample sampleWeighted(float3 random)
    {
        BSDFSample value = sample(random);
        BSDFWeightSample result = (BSDFWeightSample)0;
        result.eta = 1.0;
        if (value.pdf <= 0.0) { return result; }
        result.wiWS = value.wiWS;
        result.weight = value.f * abs(dot(shadingNormal(), value.wiWS)) / value.pdf;
        result.pdf = value.pdf;
        result.eta = value.eta;
        result.flags = value.flags;
        return result;
    }
};

// View-independent resolved scattering values. prepare() runs once per wo,
// after material evaluation/normal mapping, before the light/sample loops.
public interface ISurfaceClosure
{
    public associatedtype Prepared : IPreparedSurfaceClosure;
    public Prepared prepare(float3 woWS, TransportMode transportMode);
    public Prepared prepare(float3 woWS, TransportMode transportMode, SurfaceSamplingContext sampling)
    {
        return prepare(woWS, transportMode);
    }
};

public struct SurfaceMaterialResult<TClosure : ISurfaceClosure>
{
    public TClosure closure;
    public float3 emission;
    public uint flags; // material-level flags, not BSDF event flags; zero in v1
};

// Instantiate concrete generic programs at compile time, never store an
// existential interface value or choose concrete code using an instance ID.
public interface ISurfaceMaterialProgram
{
    public associatedtype Closure : ISurfaceClosure;
    // Stochastic texture filtering may advance provider-local RNG state.
    // Resolved closures must not retain that provider or its resources.
    [mutating]
    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance);
};

// Optional lossless integration path for BSDF libraries whose native API
// returns projected values and importance weights. Avoid f/cos/pdf round trips.
public interface IWeightedPreparedSurfaceClosure : IPreparedSurfaceClosure
{
    // Compatibility marker for Phase 4 callers; overrides now dispatch through
    // IPreparedSurfaceClosure so every model shares one lighting loop.
};
} // namespace Metallic.Material

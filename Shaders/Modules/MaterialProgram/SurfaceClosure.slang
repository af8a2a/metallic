#language slang 2026
implementing MaterialProgram;

namespace Metallic.Material
{
public enum TransportMode : uint
{
    Radiance = 0,
    Importance = 1,
};

// Per-wo temporary state. Implementations must not load material parameters,
// sample material textures, or retain a MaterialProgram/resource provider here.
public interface IPreparedSurfaceClosure
{
    public BSDFEval eval(float3 wiWS);
    public BSDFSample sample(float3 random);
};

// View-independent resolved scattering values. prepare() runs once per wo,
// after material evaluation/normal mapping, before the light/sample loops.
public interface ISurfaceClosure
{
    public associatedtype Prepared : IPreparedSurfaceClosure;
    public Prepared prepare(float3 woWS, TransportMode transportMode);
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
    public SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance);
};
} // namespace Metallic.Material

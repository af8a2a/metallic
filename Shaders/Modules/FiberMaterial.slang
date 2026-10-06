#language slang 2026
module FiberMaterial;
import MaterialProgram;

namespace Metallic.Material
{
// Independent fiber domain. A DOTS triangle adapter leaves unavailable strand
// data zero and its validity bits clear; it must never invent curve metadata.
public static const uint kFiberHasRadius = 1u;
public static const uint kFiberHasStrandParameter = 2u;
public static const uint kFiberHasMaterialCoordinates = 4u;
public static const uint kFiberHasCrossSectionOffset = 8u;
public struct FiberMaterialContext
{
    public float3 positionWS;
    public float3 tangentWS;
    public float3 normalWS; // stable authored frame, never view-facing
    public float3 bitangentWS;
    public float radius;
    public float strandParameter;
    public float crossSectionOffset; // signed projected cylinder offset h in [-1,1]
    public float2 materialCoordinates;
    public TextureFootprint footprint;
    public uint flags;
};

public struct FiberSample
{
    public float3 wiWS;
    public float3 weight; // projected scattering / pdf; no Surface cosine
    public float pdf; // complete sampled density; zero means invalid
    public uint eventType; // model event, not Surface BSDF flags
};

public interface IPreparedFiberClosure
{
    // Projected fiber scattering. Lighting must not multiply by N dot L.
    public float3 evalProjected(float3 wiWS);
    public FiberSample sampleWeighted(float4 random);
};

public interface IFiberClosure
{
    public associatedtype Prepared : IPreparedFiberClosure;
    public Prepared prepare(float3 woWS, TransportMode mode);
};

public struct FiberMaterialResult<TClosure : IFiberClosure>
{
    public TClosure closure;
    public float3 emission;
};

public interface IFiberMaterialProgram
{
    public associatedtype Closure : IFiberClosure;
    [mutating]
    public FiberMaterialResult<Closure> evaluate(FiberMaterialContext context, MaterialInstanceRef instance);
};
} // namespace Metallic.Material

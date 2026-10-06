#language slang 2026
module RayMaterialExecution;
import MaterialProgram;
import FiberMaterial;

namespace Metallic.Material
{
// A hit owns its geometry, footprint, outgoing direction and instance reference.
// No screen coordinate, cached Closure/PreparedClosure or resource provider is
// transported between bounces. RNG belongs to the path, never a queue slot.
public struct RayMaterialWork
{
    public SurfaceMaterialContext surface;
    public float3 woWS;
    public uint pathIndex;
    public MaterialInstanceRef instance;
    public uint seed;
    public uint bounce;
};

public SurfaceMaterialResult<TProgram.Closure> evaluateRaySurface<TProgram : ISurfaceMaterialProgram>(
    inout TProgram program, RayMaterialWork work)
{
    return program.evaluate(work.surface, work.instance);
}

public FiberMaterialContext rayFiberContext(RayMaterialWork work)
{
    FiberMaterialContext context = (FiberMaterialContext)0;
    context.positionWS = work.surface.positionWS;
    context.normalWS = work.surface.shadingNormalWS;
    context.tangentWS = work.surface.tangentWS;
    context.bitangentWS = work.surface.bitangentWS;
    context.materialCoordinates = work.surface.texCoord0;
    context.footprint = work.surface.footprint;
    context.flags = kFiberHasMaterialCoordinates;
    return context; // Triangle/DOTS hits cannot invent radius or strand parameter.
}

public FiberMaterialResult<TProgram.Closure> evaluateRayFiber<TProgram : IFiberMaterialProgram>(
    inout TProgram program, RayMaterialWork work)
{
    return program.evaluate(rayFiberContext(work), work.instance);
}

public TClosure.Prepared prepareRaySurface<TClosure : ISurfaceClosure>(
    TClosure closure, RayMaterialWork work, SurfaceSamplingContext sampling)
{
    return closure.prepare(work.woWS, TransportMode.Radiance, sampling);
}

public TClosure.Prepared prepareRayFiber<TClosure : IFiberClosure>(TClosure closure, RayMaterialWork work)
{
    return closure.prepare(work.woWS, TransportMode.Radiance);
}
} // namespace Metallic.Material

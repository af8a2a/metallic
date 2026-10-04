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
} // namespace Metallic.Lighting

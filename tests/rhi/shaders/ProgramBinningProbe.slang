#include "MaterialBinningProbe.slang"
import DebugLambert;
import DebugMirror;
import SurfaceLighting;
using Metallic.Lighting;

struct ProgramProbeSource : IDebugLambertSource
{
    DebugLambertInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        DebugLambertInputs result;
        // A compile-time procedural definition, shared by many instances.
        result.albedo = float3(float(PROBE_STATIC_PROGRAM) / 64.0 * context.texCoord0.x);
        result.emission = float3(0);
        return result;
    }
};
struct ProgramProbeLights : ISurfaceLightProvider
{
    uint count() { return 1; }
    SurfaceLightSample sample(uint index)
    {
        return {float3(0, 0, 1), float3(1), 1.0, 1.0, 0u};
    }
    float3 visibility(SurfaceLightSample light) { return float3(1); }
};
float shadeProgramProbe(uint pixel)
{
#if PROBE_STATIC_PROGRAM == 0
    return 0;
#else
    SurfaceMaterialContext context = {};
    context.geometryNormalWS = context.shadingNormalWS = float3(0, 0, 1);
    context.texCoord0 = float2(float(1 + pixel % 11u) / 11.0, 0);
    MaterialInstanceRef instance = {};
#if PROBE_STATIC_PROGRAM % 2 == 0
    DebugLambertMaterialProgram<ProgramProbeSource> material = {};
#else
    DebugMirrorMaterialProgram<ProgramProbeSource> material = {};
#endif
    ProgramProbeLights lights;
    SurfaceSamplingContext sampling = {float3(1), float3(650, 550, 450), 1.0};
    let shaded = shadeSurface(material, context, instance, float3(0, 0, 1), TransportMode.Radiance, sampling, lights);
    return shaded.radiance.x + 0.25 * sampleSurfaceWeight(shaded.prepared, float3(0.25, 0.5, 0.75)).weight.x;
#endif
}

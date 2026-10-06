#define METALLIC_LIGHTING_NAMED_PARAMETERS 1
#include "OpenPBRClosureProbe.slang"
import DebugLambert;
import DebugMirror;

struct LightingProbeResources { ResourceHandle<RWByteAddressBuffer> counters; };
ProbeParams lightingProbeParameters()
{
    ProbeParams result = getConstants<ProbeParams>();
    result.counters = getResourceParameters<LightingProbeResources>().counters;
    return result;
}

// The CPU supplies the concrete type token; no instance/model switch exists.
#ifndef METALLIC_SURFACE_PROGRAM
#error Surface Material Compiler must select a concrete Program
#endif

struct LightingProbeSource : IOpenPBRMaterialSource, IDebugLambertSource
{
    PathTraceMaterial loadParameters(MaterialInstanceRef instance)
    {
        ProbeSource source;
        return source.loadParameters(instance);
    }
    float4 loadTexture(PathTraceTextureInfo info, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint)
    {
        ProbeSource source;
        return source.loadTexture(info, uv, fallback, push, cone, hit, rayDirection, deterministicMip0, seed, footprint);
    }
    DebugLambertInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        PathTraceMaterial material = loadParameters(instance);
        uint seed = 0;
        float4 color = loadTexture(material.baseColorTexture, context.texCoord0, float4(1),
            (ScenePathTracePush)0, (RayCone)0, (HitInfo)0, float3(0), true, seed, context.footprint);
        DebugLambertInputs result;
        result.albedo = material.baseColor.rgb * color.rgb;
        result.emission = float3(0);
        return result;
    }
};

bool traceDiagnosticScene(float3 origin, float3 direction, out SurfaceMaterialContext context, out uint materialIndex)
{
    context = (SurfaceMaterialContext)0;
    materialIndex = 0u;
    float t = 1e20;
    float3 local = origin - float3(0, 1, 0);
    float b = dot(local, direction);
    float determinant = b * b - dot(local, local) + 1.0;
    if (determinant >= 0.0) {
        float sphereT = -b - sqrt(determinant);
        if (sphereT > 0.0001) { t = sphereT; }
    }
    if (direction.y < -0.00001) {
        float planeT = -origin.y / direction.y;
        if (planeT > 0.0001 && planeT < t) { t = planeT; materialIndex = 1u; }
    }
    if (t == 1e20) { return false; }
    context.positionWS = origin + direction * t;
    float3 normal = materialIndex == 0u ? normalize(context.positionWS - float3(0, 1, 0)) : float3(0, 1, 0);
    context.geometryNormalWS = context.shadingNormalWS = normal;
    context.texCoord0 = materialIndex == 0u
        ? float2(atan2(normal.z, normal.x) / 6.28318530718 + 0.5, acos(clamp(normal.y, -1.0, 1.0)) / 3.14159265359)
        : context.positionWS.xz * 0.25;
    context.texCoordMask = 1u;
    context.frontFace = dot(normal, -direction) > 0.0 ? 1u : 0u;
    return true;
}

struct DiagnosticLights : ISurfaceLightProvider
{
    SurfaceMaterialContext context;
    uint count() { return gProbe.lightCount; }
    SurfaceLightSample sample(uint index)
    {
        countStage(4);
        float angle = float(index) * 6.28318530718 / max(gProbe.lightCount, 1u);
        SurfaceLightSample result;
        result.direction = normalize(float3(cos(angle), 1.5, sin(angle)));
        result.illuminance = float3(2.0 / max(gProbe.lightCount, 1u));
        result.inverseSourcePdf = 1.0;
        result.distance = 10000;
        result.source = index;
        return result;
    }
    float3 visibility(SurfaceLightSample light)
    {
        countStage(5);
        SurfaceMaterialContext shadowHit;
        uint materialIndex;
        return traceDiagnosticScene(context.positionWS + context.geometryNormalWS * 0.0005,
            light.direction, shadowHit, materialIndex) ? float3(0) : float3(1);
    }
};

[shader("compute")]
[numthreads(8, 8, 1)]
void surfaceLightingProbeMain(uint3 id : SV_DispatchThreadID)
{
    const uint width = 192, height = 128;
    if (id.x >= width || id.y >= height) { return; }
    float2 screen = (float2(id.xy) + 0.5) / float2(width, height) * 2 - 1;
    float3 origin = float3(0, 1.5, 4);
    float3 direction = normalize(float3(screen.x * 0.75 * float(width) / float(height), -screen.y * 0.75 - 0.12, -1));
    float3 radiance = float3(0), throughput = float3(1);
    float4 firstNormalPdf = float4(0), firstDirectionFlags = float4(0);
    for (uint bounce = 0; bounce < 3; ++bounce) {
        SurfaceMaterialContext context;
        uint materialIndex;
        if (!traceDiagnosticScene(origin, direction, context, materialIndex)) {
            radiance += throughput * float3(0.08, 0.12, 0.2);
            break;
        }
        context.tangentWS = normalize(cross(abs(context.shadingNormalWS.z) < 0.999 ? float3(0,0,1) : float3(0,1,0), context.shadingNormalWS));
        context.bitangentWS = cross(context.shadingNormalWS, context.tangentWS);
        MaterialInstanceRef instance;
        instance.parameterOffset = materialIndex * 16;
        instance.resourceOffset = 0;
        // All programs have a source; no model-specific preparation/lighting.
        METALLIC_SURFACE_PROGRAM<LightingProbeSource> program = {};
        DiagnosticLights lights;
        lights.context = context;
        SurfaceSamplingContext sampling;
        sampling.throughput = throughput;
        sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
        sampling.exteriorIor = OpenPBRVacuumIor;
        countStage(3);
        let shaded = shadeSurface(program, context, instance, -direction, TransportMode.Radiance, sampling, lights);
        radiance += throughput * shaded.radiance;
        float3 random = frac(float3(id.x * 0.75487766 + id.y * 0.56984029 + bounce * 0.438289,
            id.x * 0.438579 + id.y * 0.819172 + bounce * 0.246817, 0.5));
        countStage(6);
        BSDFWeightSample scatter = sampleSurfaceWeight(shaded.prepared, random);
        if (bounce == 0) {
            firstNormalPdf = float4(shaded.prepared.shadingNormal(), scatter.pdf);
            firstDirectionFlags = float4(scatter.wiWS, scatter.flags);
        }
        if (scatter.pdf <= 0) { break; }
        if (scatter.flags & kBSDFDelta) { countStage(7); }
        throughput *= scatter.weight;
        direction = scatter.wiWS;
        origin = context.positionWS + context.geometryNormalWS * (dot(direction, context.geometryNormalWS) >= 0 ? 0.0005 : -0.0005);
    }
    uint base = (id.y * width + id.x) * 3;
    gProbe.output.store(base, float4(radiance, 1));
    gProbe.output.store(base + 1, firstNormalPdf);
    gProbe.output.store(base + 2, firstDirectionFlags);
}

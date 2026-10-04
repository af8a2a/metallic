import ShaderCore;
import MaterialProgram;
import DebugLambert;
import SurfaceLighting;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;

struct ProbeParams
{
    BufferSpan<float4> parameters;
    RWBufferSpan<float4> output;
    ResourceHandle<Texture2D<float4>> texture;
    ResourceHandle<RWByteAddressBuffer> counters;
    uint width;
    uint height;
    uint mode;
    uint lightCount;
};
[[vk::push_constant]] ConstantBuffer<ProbeParams> gParams;

void countStage(uint stage)
{
    uint previous;
    resolveUniform(gParams.counters).InterlockedAdd(stage * 4u, 1u, previous);
}
struct CountedSource : IDebugLambertSource
{
    DebugLambertInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        countStage(0);
        float4 factors = gParams.parameters.load(instance.parameterOffset / 16u);
        uint2 texel = min(uint2(frac(context.texCoord0) * 2.0), uint2(1));
        countStage(4);
        float3 color = resolveUniform(gParams.texture).Load(int3(texel, 0)).rgb;
        DebugLambertInputs inputs;
        inputs.albedo = factors.rgb * color;
        inputs.emission = float3(factors.w);
        return inputs;
    }
}
struct CountedPrepared : IPreparedSurfaceClosure
{
    DebugLambertPreparedClosure value;
    BSDFEval eval(float3 wiWS) { countStage(2); return value.eval(wiWS); }
    BSDFSample sample(float3 random) { countStage(3); return value.sample(random); }
}
struct CountedClosure : ISurfaceClosure
{
    typealias Prepared = CountedPrepared;
    DebugLambertClosure value;
    Prepared prepare(float3 woWS, TransportMode mode)
    {
        countStage(1);
        Prepared prepared;
        prepared.value = value.prepare(woWS, mode);
        return prepared;
    }
}
struct CountedProgram : ISurfaceMaterialProgram
{
    typealias Closure = CountedClosure;
    SurfaceMaterialResult<Closure> evaluate(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        DebugLambertMaterialProgram<CountedSource> program;
        program.source = CountedSource();
        let resolved = program.evaluate(context, instance);
        SurfaceMaterialResult<Closure> result;
        result.closure.value = resolved.closure;
        result.emission = resolved.emission;
        result.flags = resolved.flags;
        return result;
    }
}
// Both callers constrain the program; lighting below only sees Prepared.
SurfaceMaterialResult<T.Closure> evaluateHit<T : ISurfaceMaterialProgram>(
    T program, SurfaceMaterialContext context, MaterialInstanceRef instance)
{
    return program.evaluate(context, instance);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void materialClosureContractMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    if (lane >= 1024u) { return; }
    SurfaceMaterialContext context = (SurfaceMaterialContext)0;
    context.geometryNormalWS = context.shadingNormalWS = (lane & 1u) ? float3(0, 0.6, 0.8) : float3(0, 0, 1);
    context.texCoord0 = (float2(lane & 1u, (lane >> 1u) & 1u) + 0.5) / 2.0;
    context.texCoordMask = 1u;
    context.frontFace = 1u;
    MaterialInstanceRef instance;
    instance.parameterOffset = (lane & 1u) * 16u;
    instance.resourceOffset = 0u;
    CountedProgram program;
    let material = evaluateHit(program, context, instance);
    float3 n = context.shadingNormalWS;
    let prepared = material.closure.prepare(n, TransportMode.Radiance);
    let second = material.closure.prepare(normalize(n + float3(0.3, 0, 0)), TransportMode.Radiance);
    let importance = material.closure.prepare(n, TransportMode.Importance);
    let back = material.closure.prepare(-n, TransportMode.Radiance);
    BSDFEval value = prepared.eval(n);
    BSDFSample sampled = sampleSurfaceContinuation(prepared, float3((lane + 0.5) / 1024.0, frac(lane * 0.6180339887), 0.75));
    BSDFEval matched = prepared.eval(sampled.wiWS);
    BSDFEval importanceValue = importance.eval(n);
    BSDFEval backValue = back.eval(n);
    BSDFEval secondValue = second.eval(n);
    BSDFSample backSample = sampleSurfaceContinuation(back, float3(0.5));
    float3 direct = float3(0);
    for (uint light = 0; light < 8u; ++light) {
        direct += evaluateSurfaceDirect(prepared, n, float3(1), 1.0);
    }
    uint base = lane * 10u;
    gParams.output.store(base, float4(value.f, value.pdf));
    gParams.output.store(base + 1u, float4(sampled.wiWS, sampled.pdf));
    gParams.output.store(base + 2u, float4(sampled.f * max(dot(n, sampled.wiWS), 0.0) / sampled.pdf, sampled.eta));
    gParams.output.store(base + 3u, float4(matched.f, matched.pdf));
    gParams.output.store(base + 4u, float4(importanceValue.f, importanceValue.pdf));
    gParams.output.store(base + 5u, float4(backValue.f, backValue.pdf));
    gParams.output.store(base + 6u, float4(direct, float(sampled.flags)));
    gParams.output.store(base + 7u, float4(secondValue.f, secondValue.pdf));
    gParams.output.store(base + 8u, float4(material.emission, float(material.flags)));
    gParams.output.store(base + 9u, float4(backSample.f, backSample.pdf));
}

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

[shader("compute")]
[numthreads(8, 8, 1)]
void materialClosureRenderMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= gParams.width || id.y >= gParams.height) { return; }
    float2 screen = (float2(id.xy) + 0.5) / float2(gParams.width, gParams.height) * 2.0 - 1.0;
    float3 origin = float3(0, 1.5, 4);
    float3 direction = normalize(float3(screen.x * 0.75 * float(gParams.width) / float(gParams.height),
        -screen.y * 0.75 - 0.12, -1));
    float3 radiance = float3(0), throughput = float3(1);
    CountedProgram program;
    uint depth = gParams.mode == 0u ? 1u : 3u;
    for (uint bounce = 0; bounce < depth; ++bounce) {
        SurfaceMaterialContext context;
        uint materialIndex;
        if (!traceDiagnosticScene(origin, direction, context, materialIndex)) {
            radiance += throughput * float3(0.08, 0.12, 0.2);
            break;
        }
        MaterialInstanceRef instance;
        instance.parameterOffset = materialIndex * 16u;
        instance.resourceOffset = 0u;
        let material = evaluateHit(program, context, instance);
        let prepared = material.closure.prepare(-direction, TransportMode.Radiance);
        radiance += throughput * material.emission;
        for (uint light = 0; light < gParams.lightCount; ++light) {
            float angle = float(light) * 6.28318530718 / max(gParams.lightCount, 1u);
            float3 wi = normalize(float3(cos(angle), 1.5, sin(angle)));
            SurfaceMaterialContext shadowHit;
            uint shadowMaterial;
            bool shadow = traceDiagnosticScene(context.positionWS + context.geometryNormalWS * 0.0005, wi, shadowHit, shadowMaterial);
            float3 incident = shadow ? float3(0) : float3(2.0 / max(gParams.lightCount, 1u));
            radiance += throughput * evaluateSurfaceDirect(prepared, wi, incident, abs(dot(context.shadingNormalWS, wi)));
        }
        if (gParams.mode == 0u) { break; }
        float3 random = frac(float3(id.x * 0.75487766 + id.y * 0.56984029 + bounce * 0.438289,
            id.x * 0.438579 + id.y * 0.819172 + bounce * 0.246817, 0.5));
        BSDFSample continuation = sampleSurfaceContinuation(prepared, random);
        if (continuation.pdf <= 0.0) { break; }
        throughput *= continuation.f * abs(dot(context.shadingNormalWS, continuation.wiWS)) / continuation.pdf;
        origin = context.positionWS + context.geometryNormalWS * 0.0005;
        direction = continuation.wiWS;
    }
    gParams.output.store(id.y * gParams.width + id.x, float4(radiance, 1));
}

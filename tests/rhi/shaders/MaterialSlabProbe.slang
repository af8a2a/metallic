import ShaderCore;
import SlabClosure;
import MaterialProgram;
import SurfaceLighting;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;

struct ProbeParams
{
    BufferSpan<float4> packets;
    RWBufferSpan<float4> output;
    uint caseIndex;
    float cosineOut;
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<ProbeParams> gParams;

struct PacketSource : ISlabMaterialSource
{
    SlabMaterialInputs evaluateInputs(SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        uint offset = instance.parameterOffset / 16u;
        SlabMaterialInputs result;
        result.first.reflectance = gParams.packets.load(offset);
        result.first.opticalDepth = gParams.packets.load(offset + 1u);
        result.second.reflectance = gParams.packets.load(offset + 2u);
        result.second.opticalDepth = gParams.packets.load(offset + 3u);
        result.control = gParams.packets.load(offset + 4u);
        return result;
    }
};
SurfaceMaterialContext makeContext(float3 n)
{
    SurfaceMaterialContext context = (SurfaceMaterialContext)0;
    context.shadingNormalWS = context.geometryNormalWS = n;
    context.frontFace = 1u;
    return context;
}
MaterialInstanceRef instanceRef()
{
    MaterialInstanceRef instance = (MaterialInstanceRef)0;
    instance.parameterOffset = gParams.caseIndex * 80u;
    return instance;
}
void contract<T : ISurfaceMaterialProgram>(inout T program, uint lane)
{
    if (lane >= 4096u) { return; }
    float3 n = (gParams.caseIndex & 1u) ? float3(0, 0.6, 0.8) : float3(0, 0, 1);
    float3 t = float3(1, 0, 0), b = cross(n, t);
    float3 wo = n * gParams.cosineOut + t * sqrt(1.0 - gParams.cosineOut * gParams.cosineOut);
    let material = program.evaluate(makeContext(n), instanceRef());
    let prepared = material.closure.prepare(wo, TransportMode.Radiance);
    float3 random = float3((lane + 0.5) / 4096.0, frac(lane * 0.6180339887), 0.5);
    let sample = sampleSurfaceContinuation(prepared, random);
    let matched = prepared.eval(sample.wiWS);
    let weighted = sampleSurfaceWeight(prepared, random);
    // Independent uniform hemisphere quadrature, not the cosine sampler.
    float mu = (lane + 0.5) / 4096.0;
    float phi = 6.28318530718 * frac(lane * 0.754877666);
    float3 wi = n * mu + sqrt(1 - mu * mu) * (t * cos(phi) + b * sin(phi));
    let uniform = prepared.eval(wi);
    let reciprocal = material.closure.prepare(wi, TransportMode.Radiance).eval(wo);
    let importance = material.closure.prepare(wo, TransportMode.Importance).eval(wi);
    let back = material.closure.prepare(-n, TransportMode.Radiance);
    let invalid = back.sample(random);
    let outside = prepared.eval(-n);
    uint base = lane * 9u;
    gParams.output.store(base, float4(sample.wiWS, sample.pdf));
    gParams.output.store(base + 1u, float4(sample.f, float(sample.flags)));
    gParams.output.store(base + 2u, float4(matched.f, prepared.pdf(sample.wiWS)));
    gParams.output.store(base + 3u, float4(uniform.f, uniform.pdf));
    gParams.output.store(base + 4u, float4(reciprocal.f, sample.eta));
    gParams.output.store(base + 5u, float4(importance.f, float(sizeof(T.Closure))));
    gParams.output.store(base + 6u, float4(invalid.f + outside.f, invalid.pdf + outside.pdf));
    gParams.output.store(base + 7u, float4(weighted.weight, weighted.pdf));
    // Endpoint variates must stay finite; x=1 is clamped inside sampler.
    let endpoint = prepared.sample(float3(1, 0, 0));
    gParams.output.store(base + 8u, float4(endpoint.f, endpoint.pdf));
}
[shader("compute")]
[numthreads(64, 1, 1)]
void singleContractMain(uint3 id : SV_DispatchThreadID)
{
    SingleSlabMaterialProgram<PacketSource> program;
    contract(program, id.x);
}
[shader("compute")]
[numthreads(64, 1, 1)]
void dualContractMain(uint3 id : SV_DispatchThreadID)
{
    DualSlabMaterialProgram<PacketSource> program;
    contract(program, id.x);
}

bool traceScene(float3 origin, float3 direction, out float3 position, out float3 normal)
{
    float3 local = origin - float3(0, 1, 0);
    float b = dot(local, direction);
    float determinant = b * b - dot(local, local) + 1;
    float distance = 1e20;
    bool plane = false;
    if (determinant >= 0) {
        float t = -b - sqrt(determinant);
        if (t > 1e-4) { distance = t; }
    }
    if (direction.y < -1e-5) {
        float t = -origin.y / direction.y;
        if (t > 1e-4 && t < distance) { distance = t; plane = true; }
    }
    position = origin + direction * distance;
    normal = plane ? float3(0, 1, 0) : normalize(position - float3(0, 1, 0));
    return distance < 1e20;
}
void renderScene<T : ISurfaceMaterialProgram>(inout T program, uint2 pixel)
{
    if (any(pixel >= uint2(gParams.width, gParams.height))) { return; }
    float2 screen = (float2(pixel) + 0.5) / float2(gParams.width, gParams.height) * 2 - 1;
    float3 radiance = float3(0);
    for (uint sampleIndex = 0; sampleIndex < 64u; ++sampleIndex) {
        float3 origin = float3(0, 1.5, 4);
        float3 direction = normalize(float3(screen.x * 0.75 * float(gParams.width) / float(gParams.height), -screen.y * 0.75 - 0.12, -1));
        float3 throughput = float3(1);
        for (uint bounce = 0; bounce < 3; ++bounce) {
            float3 position, normal;
            if (!traceScene(origin, direction, position, normal)) {
                radiance += throughput * float3(0.12, 0.16, 0.24);
                break;
            }
            let material = program.evaluate(makeContext(normal), instanceRef());
            let prepared = material.closure.prepare(-direction, TransportMode.Radiance);
            float3 wi = normalize(float3(-1, 2, 2));
            float3 shadowPosition, shadowNormal;
            float3 offset = position + normal * 0.0005;
            if (!traceScene(offset, wi, shadowPosition, shadowNormal)) {
                radiance += throughput * evaluateSurfaceDirect(prepared, wi, float3(2.5), max(dot(normal, wi), 0));
            }
            float3 random = frac(float3((sampleIndex + 0.5) / 64.0 + bounce * 0.438289,
                sampleIndex * 0.618034 + dot(float2(pixel), float2(0.438579, 0.819172)) + bounce * 0.246817, 0.5));
            let continuation = sampleSurfaceWeight(prepared, random);
            if (continuation.pdf <= 0) { break; }
            throughput *= continuation.weight;
            origin = offset;
            direction = continuation.wiWS;
        }
    }
    gParams.output.store(pixel.y * gParams.width + pixel.x, float4(radiance / 64.0, 1));
}
[shader("compute")]
[numthreads(8, 8, 1)]
void singleRenderMain(uint3 id : SV_DispatchThreadID)
{
    SingleSlabMaterialProgram<PacketSource> program;
    renderScene(program, id.xy);
}
[shader("compute")]
[numthreads(8, 8, 1)]
void dualRenderMain(uint3 id : SV_DispatchThreadID)
{
    DualSlabMaterialProgram<PacketSource> program;
    renderScene(program, id.xy);
}

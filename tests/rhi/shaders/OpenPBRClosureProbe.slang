import ShaderCore;
using Metallic;

struct ProbeParams
{
    BufferSpan<float4> parameters;
    RWBufferSpan<float4> output;
    ResourceHandle<Texture2D<float4>> texture;
    ResourceHandle<RWByteAddressBuffer> counters;
    uint lightCount;
};
#if METALLIC_LIGHTING_NAMED_PARAMETERS
#define gProbe lightingProbeParameters()
#else
[[vk::push_constant]] ConstantBuffer<ProbeParams> gProbe;
#endif

// Raw vendor uint16 tables normalized exactly as the production texture upload.
// Buffer lookup avoids embedding giant arrays in the driver's shader compiler.
static const uint kLutOffsets[7] = {2, 32770, 33794, 34818, 67586, 68610, 69634};
float probeLutTexel(int id, int3 p, int3 size)
{
    return gProbe.parameters.load(kLutOffsets[id] + uint((p.z * size.y + p.y) * size.x + p.x)).x;
}
float4 probeLut(int id, float3 uvw, int3 size)
{
    float3 p = clamp(uvw * float3(size) - 0.5, float3(0), float3(size - 1));
    int3 lo = int3(floor(p));
    int3 hi = min(lo + 1, size - 1);
    float3 f = p - float3(lo);
    float a = lerp(lerp(probeLutTexel(id, lo, size), probeLutTexel(id, int3(hi.x, lo.yz), size), f.x),
        lerp(probeLutTexel(id, int3(lo.x, hi.y, lo.z), size), probeLutTexel(id, int3(hi.xy, lo.z), size), f.x), f.y);
    float b = lerp(lerp(probeLutTexel(id, int3(lo.xy, hi.z), size), probeLutTexel(id, int3(hi.x, lo.y, hi.z), size), f.x),
        lerp(probeLutTexel(id, int3(lo.x, hi.yz), size), probeLutTexel(id, hi, size), f.x), f.y);
    return float4(lerp(a, b, f.z));
}
import OpenPBR;
using Metallic.OpenPBR;
struct ProbeOpenPBRContext : IOpenPBRContext
{
    static bool enableSheenAndCoat() { return false; }
    static bool enableDispersion() { return false; }
    static bool enableTranslucency() { return true; }
    static bool enableMetallic() { return true; }
    float4 sample2D(int id, float2 uv) { return probeLut(id, float3(uv, 0.5), int3(32, id == 6 ? 1 : 32, 1)); }
    float4 sample3D(int id, float3 uvw) { return probeLut(id, uvw, int3(32)); }
};
#define METALLIC_OPENPBR_CONTEXT_TYPE ProbeOpenPBRContext
#include "../../../Shaders/Features/PathTracing/OpenPBRSurface.slang"

// Exercise a second specialization with nonempty context state. Both prepare
// and subsequent evaluation must use the supplied LUT context, not a default.
struct StatefulOpenPBRContext : IOpenPBRContext
{
    float scale;
    static bool enableSheenAndCoat() { return true; }
    static bool enableDispersion() { return false; }
    static bool enableTranslucency() { return true; }
    static bool enableMetallic() { return true; }
    float4 sample2D(int id, float2 uv) { return ProbeOpenPBRContext().sample2D(id, uv) * scale; }
    float4 sample3D(int id, float3 uvw) { return ProbeOpenPBRContext().sample3D(id, uvw) * scale; }
};

void countStage(uint index)
{
    uint previous;
    resolveUniform(gProbe.counters).InterlockedAdd(index * 4, 1, previous);
}

struct ProbeSource : IOpenPBRMaterialSource
{
    PathTraceMaterial loadParameters(MaterialInstanceRef instance)
    {
        countStage(0);
        uint variant = instance.resourceOffset;
        float4 factors = gProbe.parameters.load(instance.parameterOffset / 16);
        PathTraceMaterial material = (PathTraceMaterial)0;
        material.baseColor = float4(factors.rgb, 1);
        material.specular = float4(1);
        material.params = float4(variant == 1 ? 1 : 0, variant == 3 ? 0.001 : 0.35, 0, 0);
        material.glassParams = float4(variant == 2 || variant == 3 || variant == 6 ? 1 : 0, 1.5, 0, 0);
        material.textureParams = float4(1, 1, 0, 0);
        material.emissive = float4(variant == 4 ? factors.rgb : float3(0), variant == 5 ? 1 : 0);
        material.baseColorTexture.textureIndex = 1;
        material.baseColorTexture.transform0 = float4(1, 0, 0, 1);
        material.baseColorTexture.transform1 = float4(0, 1, 0, 0);
        material.normalTexture.textureIndex = 2;
        if (variant == 7) { material.baseColor = float4(0); material.specular = float4(0); }
        return material;
    }
    float4 loadTexture(PathTraceTextureInfo info, float2 uv, float4 fallback,
        ScenePathTracePush push, RayCone cone, HitInfo hit, float3 rayDirection,
        bool deterministicMip0, inout uint seed, TextureFootprint footprint)
    {
        countStage(1);
        if (info.textureIndex == 0) { return fallback; }
        countStage(2);
        uint2 texel = info.textureIndex == 2 ? uint2(0, 1) : min(uint2(frac(uv) * 2), uint2(1));
        return resolveUniform(gProbe.texture).Load(int3(texel, 0));
    }
};

SurfaceMaterialResult<T.Closure> evaluateHit<T : ISurfaceMaterialProgram>(
    T program, SurfaceMaterialContext context, MaterialInstanceRef instance)
{
    return program.evaluate(context, instance);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRClosureProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    uint variant = lane / 128;
    OpenPBRMaterialProgram<ProbeSource> program = (OpenPBRMaterialProgram<ProbeSource>)0;
    program.fixedMetalness = -1;
    program.hit.tangent = float4(1, 0, 0, 1);
    SurfaceMaterialContext context = (SurfaceMaterialContext)0;
    context.shadingNormalWS = context.geometryNormalWS = float3(0, 0, 1);
    context.tangentWS = float3(1, 0, 0);
    context.bitangentWS = float3(0, 1, 0);
    context.texCoord0 = float2((lane & 1) * 0.5 + 0.25, ((lane >> 1) & 1) * 0.5 + 0.25);
    context.texCoordMask = 1;
    context.frontFace = variant == 6 ? 0 : 1;
    MaterialInstanceRef instance;
    instance.parameterOffset = (lane & 1) * 16;
    instance.resourceOffset = variant;
    let evaluated = evaluateHit(program, context, instance);
    let closure = evaluated.closure;
    float3 wo = normalize(float3(0.2, 0.1, variant == 6 ? -1 : 1));
    SurfaceSamplingContext sampling;
    sampling.throughput = float3(0.2, 0.7, 0.4);
    sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
    sampling.exteriorIor = 1.1;
    countStage(3);
    let prepared = closure.prepare(wo, TransportMode.Radiance, sampling);
    let native = openPBRPrepare(ProbeOpenPBRContext(), closure.inputs, sampling.throughput, sampling.wavelengths, sampling.exteriorIor, wo);
    OpenPBRClosure<StatefulOpenPBRContext> standalone;
    standalone.context.scale = 0.8;
    standalone.inputs = closure.inputs;
    standalone.occlusion = closure.occlusion;
    let standalonePrepared = standalone.prepare(wo, TransportMode.Radiance, sampling);
    let standaloneNative = openPBRPrepare(standalone.context, standalone.inputs,
        sampling.throughput, sampling.wavelengths, sampling.exteriorIor, wo);
    float3 error = float3(0);
    for (uint light = 0; light < gProbe.lightCount; ++light) {
        float3 wi = normalize(float3(float(light) * 0.13 - 0.3, 0.2, (light & 1) ? -1 : 1));
        float3 expected = openPBRGetSumOfDiffuseSpecular(openPBREval(ProbeOpenPBRContext(), native, wi));
        countStage(4);
        float3 projected = evaluateSurfaceProjected(prepared, wi);
        BSDFEval value = prepared.eval(wi);
        error = max(error, abs(projected - expected));
        error = max(error, abs(value.f * abs(dot(closure.inputs.geometry_basis.n, wi)) - expected));
        error = max(error, float3(abs(value.pdf - openPBRPdf(ProbeOpenPBRContext(), native, wi))));
        error = max(error, abs(standalonePrepared.evalProjected(wi) -
            openPBRGetSumOfDiffuseSpecular(openPBREval(standalone.context, standaloneNative, wi))));
        error = max(error, float3(abs(standalonePrepared.pdf(wi) - openPBRPdf(standalone.context, standaloneNative, wi))));
    }
    float3 random = float3((float(lane % 128) + 0.5) / 128, frac(float(lane) * 0.61803398875), frac(float(lane) * 0.41421356237));
    BSDFWeightSample weighted = sampleSurfaceWeight(prepared, random);
    BSDFSample sample = sampleSurfaceContinuation(prepared, random);
    float3 nativeDirection;
    OpenPBRDiffuseSpecular nativeWeight;
    float nativePdf;
    uint nativeType;
    openPBRSample(ProbeOpenPBRContext(), native, random, nativeDirection, nativeWeight, nativePdf, nativeType);
    float3 sampleError = float3(abs(weighted.pdf - nativePdf));
    if (nativePdf > 0) {
        sampleError = max(sampleError, abs(weighted.wiWS - nativeDirection));
        sampleError = max(sampleError, abs(weighted.weight - openPBRGetSumOfDiffuseSpecular(nativeWeight)));
        sampleError = max(sampleError, abs(sample.f * abs(dot(closure.inputs.geometry_basis.n, sample.wiWS)) / sample.pdf - weighted.weight));
        sampleError = max(sampleError, float3(abs(float(weighted.flags) - float(nativeType))));
    } else {
        sampleError = max(sampleError, abs(sample.f) + abs(weighted.weight));
    }
    float3 standaloneDirection;
    OpenPBRDiffuseSpecular standaloneWeight;
    float standalonePdf;
    uint standaloneType;
    openPBRSample(standalone.context, standaloneNative, random, standaloneDirection, standaloneWeight, standalonePdf, standaloneType);
    let standaloneSample = standalonePrepared.sampleWeighted(random);
    sampleError = max(sampleError, float3(abs(standaloneSample.pdf - standalonePdf)));
    if (standalonePdf > 0) {
        sampleError = max(sampleError, abs(standaloneSample.wiWS - standaloneDirection));
        sampleError = max(sampleError, abs(standaloneSample.weight - openPBRGetSumOfDiffuseSpecular(standaloneWeight)));
        sampleError = max(sampleError, float3(abs(float(standaloneSample.flags) - float(standaloneType))));
    }
    // Reusing the resolved closure for a second view must not load textures.
    let second = closure.prepare(normalize(float3(-0.4, 0.2, 1)), TransportMode.Radiance);
    let importance = closure.prepare(wo, TransportMode.Importance);
    float unsupported = importance.pdf(wo) + length(importance.eval(wo).f) + importance.sample(random).pdf;
    uint base = lane * 8;
    gProbe.output.store(base, float4(error, unsupported));
    gProbe.output.store(base + 1, float4(sampleError, weighted.pdf));
    gProbe.output.store(base + 2, float4(closure.inputs.base_color, closure.inputs.base_metalness));
    gProbe.output.store(base + 3, float4(closure.inputs.geometry_basis.n, weighted.eta));
    gProbe.output.store(base + 4, float4(abs(prepared.emission() - native.emission), float(weighted.flags)));
    gProbe.output.store(base + 5, float4(second.eval(wo).f, evaluated.flags));
    gProbe.output.store(base + 6, float4(evaluated.emission, closure.inputs.transmission_weight));
    gProbe.output.store(base + 7, float4(weighted.weight, sample.pdf));
}
